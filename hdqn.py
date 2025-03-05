# hdqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque


class HDQNNetwork(nn.Module):
    def __init__(self, obs_dim, top_dim, sub_dim):
        """
        obs_dim: Dimension of the flattened observation (e.g. 240)
        top_dim: Number of top-level actions (e.g. 12)
        sub_dim: Number of sub-actions (e.g. 252)
        """
        super(HDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # Two heads: one for top-level Q-values and one for sub-action Q-values.
        self.top_head = nn.Linear(256, top_dim)
        self.sub_head = nn.Linear(256, sub_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        top_q = self.top_head(x)
        sub_q = self.sub_head(x)
        return top_q, sub_q


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def select_action(network, state, epsilon):
    """
    Select a hierarchical action.
    - For the top-level action, use epsilon-greedy.
    - For the sub-action, use greedy selection.
    """
    device = next(network.parameters()).device
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        top_q, sub_q = network(state_tensor)
    top_q = top_q.cpu().numpy()[0]
    sub_q = sub_q.cpu().numpy()[0]
    # Epsilon-greedy for top-level:
    if random.random() < epsilon:
        a_top = random.randrange(len(top_q))
    else:
        a_top = np.argmax(top_q)
    # Greedy for sub-action:
    a_sub = np.argmax(sub_q)
    return (a_top, a_sub)
