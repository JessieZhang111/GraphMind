import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

class LPLHPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LPLHPolicyNetwork, self).__init__()  
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

@dataclass
class LPLHDecision:
    action: int
    preference: float
    episodic_memory: float

class LPLHPolicy:
    def __init__(self, input_dim, output_dim):
        self.network = LPLHPolicyNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def decide(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.network(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        preference = action_probs[action].item()  
        return LPLHDecision(action, preference, 0.0)  # episodic_memory can be set as needed

    def compute_loss(self, action, target):
        action_one_hot = torch.zeros(action.shape[0], dtype=torch.float32)
        action_one_hot[action] = 1
        loss = nn.BCELoss()(action_one_hot, target)
        return loss

    def update(self, state, action, target):
        self.optimizer.zero_grad()
        loss = self.compute_loss(action, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
