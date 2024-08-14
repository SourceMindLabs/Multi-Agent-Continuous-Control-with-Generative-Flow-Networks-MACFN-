import torch.nn as nn
import torch.nn.functional as F
import torch

class FlowDecompositionNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(FlowDecompositionNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_agents)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x
