import torch
from environment import MultiAgentContinuousEnv
from macfn import MACFN

# Hyperparameters
num_agents = 3
state_dim = 5
action_dim = 3
hidden_dim = 64
lr = 1e-3
episodes = 1000
max_steps = 50

# Environment and MACFN initialization
env = MultiAgentContinuousEnv(num_agents, state_dim, action_dim, max_steps)
macfn = MACFN(num_agents, state_dim, action_dim, hidden_dim, lr)

# Training
macfn.train(env, episodes, max_steps)
