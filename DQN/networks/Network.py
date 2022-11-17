import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

FC1_DIMS = 1024
FC2_DIMS = 512

DEVICE = torch.device('cpu')

class Network(nn.Module):
    
    def __init__(self, obs_space_shape, act_space_shape, learning_rate):
        super().__init__()
        
        self.fc1 = nn.Linear(*obs_space_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, act_space_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x