import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    The Actor-Critic network for the IPPO agent.

    This network implements the architecture described in the project plan:
    2x128 MLP -> 128-d GRU -> {policy logits, value}
    """
    def __init__(self, input_dim, action_dims, hidden_dim=128):
        """
        Initialize the Actor-Critic network.

        Args:
            input_dim (int): The dimension of the flattened input observation.
            action_dims (list[int]): A list containing the dimensions of the
                                     top-level and sub-action spaces.
            hidden_dim (int): The size of the hidden layers and GRU state.
        """
        super(ActorCritic, self).__init__()
        self.hidden_dim = hidden_dim

        # Shared MLP body
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Recurrent layer (GRU)
        self.gru = nn.GRU(hidden_dim, hidden_dim)

        # Actor heads for the multi-discrete action space
        self.policy_head_top = nn.Linear(hidden_dim, action_dims[0])
        self.policy_head_sub = nn.Linear(hidden_dim, action_dims[1])

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor (flattened observation).
            hidden_state (torch.Tensor): The hidden state for the GRU.

        Returns:
            tuple: A tuple containing:
                - top_level_logits (torch.Tensor): Logits for the top-level action.
                - sub_action_logits (torch.Tensor): Logits for the sub-action.
                - value (torch.Tensor): The state value from the critic.
                - hidden_state (torch.Tensor): The new hidden state from the GRU.
        """
        # Pass input through the shared MLP
        x = self.mlp(x)
        
        # The GRU expects input of shape (seq_len, batch, input_size).
        # We are processing one step at a time, so seq_len is 1.
        x, hidden_state = self.gru(x.unsqueeze(0), hidden_state)
        x = x.squeeze(0)

        # --- Actor ---
        # Get logits for each part of the action space
        top_level_logits = self.policy_head_top(x)
        sub_action_logits = self.policy_head_sub(x)
        
        # --- Critic ---
        # Get the value of the state
        value = self.value_head(x)

        return top_level_logits, sub_action_logits, value, hidden_state 