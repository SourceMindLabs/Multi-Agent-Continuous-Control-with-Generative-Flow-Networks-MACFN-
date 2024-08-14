import torch.nn as nn
import torch.nn.functional as F
import torch

class InverseTransitionNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(InverseTransitionNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
