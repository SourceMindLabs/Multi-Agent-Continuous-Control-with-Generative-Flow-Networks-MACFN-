import torch
import torch.optim as optim
from flow_network import FlowNetwork
from inverse_transition_network import InverseTransitionNetwork
from flow_decomposition_network import FlowDecompositionNetwork

class MACFN:
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim, lr=1e-3):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Individual flow networks for each agent
        self.flow_networks = [FlowNetwork(state_dim + action_dim, hidden_dim, 1) for _ in range(num_agents)]
        self.inverse_transition_network = InverseTransitionNetwork(state_dim, action_dim)
        self.flow_decomposition_network = FlowDecompositionNetwork(state_dim, action_dim, num_agents)

        self.optimizers = [optim.Adam(fn.parameters(), lr=lr) for fn in self.flow_networks]
        self.inverse_optimizer = optim.Adam(self.inverse_transition_network.parameters(), lr=lr)
        self.decomposition_optimizer = optim.Adam(self.flow_decomposition_network.parameters(), lr=lr)

    def sample_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            state = states[i]
            action = torch.tanh(torch.randn(self.action_dim))  # Sample action from a normal distribution
            actions.append(action)
        return torch.stack(actions)

    def compute_flows(self, states, actions):
        flows = []
        for i in range(self.num_agents):
            state_action = torch.cat([states[i], actions[i]], dim=-1)
            flow = self.flow_networks[i](state_action)
            flows.append(flow)
        return torch.stack(flows)

    def decompose_flow(self, state, action):
        return self.flow_decomposition_network(state, action)

    def estimate_parent_state(self, state, action):
        return self.inverse_transition_network(state, action)

    def compute_inflows(self, states, actions, K=10):
        inflows = []
        for i in range(self.num_agents):
            sampled_actions = torch.randn(K, self.action_dim)
            parent_states = self.estimate_parent_state(states[i].repeat(K, 1), sampled_actions)
            inflow = torch.mean(self.flow_networks[i](torch.cat([parent_states, sampled_actions], dim=-1)))
            inflows.append(inflow)
        return torch.stack(inflows)

    def compute_outflows(self, states, actions, K=10):
        outflows = []
        for i in range(self.num_agents):
            sampled_actions = torch.randn(K, self.action_dim)
            outflow = torch.mean(self.flow_networks[i](torch.cat([states[i].repeat(K, 1), sampled_actions], dim=-1)))
            outflows.append(outflow)
        return torch.stack(outflows)

    def flow_matching_loss(self, states, actions, rewards, epsilon=1e-6):
        inflows = self.compute_inflows(states, actions)
        outflows = self.compute_outflows(states, actions)
        flows = self.compute_flows(states, actions)
        
        # Log-scale operation
        loss = torch.mean((torch.log(inflows + epsilon) - torch.log(outflows + rewards.unsqueeze(1) + epsilon))**2)
        
        # Flow decomposition loss
        decomposed_flows = self.decompose_flow(states.view(-1, self.state_dim), actions.view(-1, self.action_dim))
        decomposition_loss = torch.nn.functional.mse_loss(decomposed_flows, flows.view(-1, self.num_agents))
        
        return loss + decomposition_loss

    def update(self, states, actions, rewards):
        loss = self.flow_matching_loss(states, actions, rewards)
        
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        self.inverse_optimizer.zero_grad()
        self.decomposition_optimizer.zero_grad()
        
        loss.backward()
        
        for optimizer in self.optimizers:
            optimizer.step()
        self.inverse_optimizer.step()
        self.decomposition_optimizer.step()

        return loss.item()

    def train(self, env, episodes, max_steps):
        for episode in range(episodes):
            states = torch.tensor(env.reset(), dtype=torch.float32)
            total_reward = 0
            for step in range(max_steps):
                actions = self.sample_actions(states)
                next_states, rewards, done = env.step(actions.detach().numpy())
                next_states = torch.tensor(next_states, dtype=torch.float32)
                rewards = torch.tensor(rewards, dtype=torch.float32)

                loss = self.update(states, actions, rewards)

                states = next_states
                total_reward += rewards.sum().item()

                if done:
                    break

            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Loss: {loss:.4f}")
