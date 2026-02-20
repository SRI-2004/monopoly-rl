import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    The Actor-Critic network for the IPPO agent with Hierarchical Action Embedding.

    This network implements the architecture described in the project plan:
    2x128 MLP -> 128-d GRU -> {policy logits, value}
    
    Enhanced with hierarchical action embedding to make sub-actions context-aware
    based on the selected top-level action.
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
        self.top_action_dim = action_dims[0]  # 12
        self.sub_action_dim = action_dims[1]  # 252

        # Shared MLP body
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Recurrent layer (GRU)
        self.gru = nn.GRU(hidden_dim, hidden_dim)

        # Top-level action head
        self.policy_head_top = nn.Linear(hidden_dim, self.top_action_dim)
        
        # Hierarchical embedding for top-level actions
        self.top_action_embedding = nn.Embedding(self.top_action_dim, 32)
        
        # Context-aware sub-action head
        self.sub_action_processor = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),  # GRU output + top action embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, self.sub_action_dim)
        )

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state, top_action=None, action_mask=None):
        """
        Forward pass through the network with hierarchical action embedding.

        Args:
            x (torch.Tensor): The input tensor (flattened observation).
            hidden_state (torch.Tensor): The hidden state for the GRU.
            top_action (torch.Tensor, optional): The top-level action for context.
                                               If None, uses argmax of top logits.
            action_mask (tuple, optional): Action mask [top_mask, sub_masks] for masking invalid actions.

        Returns:
            tuple: A tuple containing:
                - top_level_logits (torch.Tensor): Logits for the top-level action.
                - sub_action_logits (torch.Tensor): Context-aware logits for the sub-action.
                - value (torch.Tensor): The state value from the critic.
                - hidden_state (torch.Tensor): The new hidden state from the GRU.
        """
        # Pass input through the shared MLP
        x = self.mlp(x)
        
        # The GRU expects input of shape (seq_len, batch, input_size).
        # We are processing one step at a time, so seq_len is 1.
        x, hidden_state = self.gru(x.unsqueeze(0), hidden_state)
        gru_output = x.squeeze(0)

        # --- Top-level Action ---
        top_level_logits = self.policy_head_top(gru_output)
        
        # --- Hierarchical Sub-action with Context ---
        if top_action is not None:
            # Use provided top action for context
            top_embed = self.top_action_embedding(top_action)
        else:
            # Use argmax of top logits for context (deterministic for consistency)
            top_action_idx = torch.argmax(top_level_logits, dim=-1)
            top_embed = self.top_action_embedding(top_action_idx)
        
        # Combine GRU output with top-action context
        combined_features = torch.cat([gru_output, top_embed], dim=-1)
        sub_action_logits = self.sub_action_processor(combined_features)
        
        # --- Critic ---
        value = self.value_head(gru_output)

        return top_level_logits, sub_action_logits, value, hidden_state

    def get_action_and_value(self, x, hidden_state, deterministic=False, action_mask=None):
        """
        Optimized single-pass action and value computation with hierarchical sampling.
        
        This method avoids double forward passes by computing everything in one go
        and using the sampled top action for sub-action context.
        
        Args:
            x: Input observation tensor
            hidden_state: GRU hidden state
            deterministic: Whether to use deterministic action selection
            action_mask: Optional action mask [top_mask, sub_masks] where:
                        - top_mask: [12] boolean mask for top-level actions
                        - sub_masks: [12, 252] boolean masks for sub-actions per top-action
        """
        # Pass input through shared backbone once
        x = self.mlp(x)
        x, new_hidden = self.gru(x.unsqueeze(0), hidden_state)
        gru_output = x.squeeze(0)

        # Get top-level action logits
        top_logits = self.policy_head_top(gru_output)
        
        # Apply action mask to top-level actions if provided
        if action_mask is not None:
            top_mask, sub_masks = action_mask
            # Convert to tensor if needed
            if isinstance(top_mask, (list, tuple, np.ndarray)):
                top_mask = torch.tensor(top_mask, dtype=torch.bool, device=top_logits.device)
            
            # Ensure mask has same batch dimension as logits
            if top_mask.dim() == 1 and top_logits.dim() == 2:
                top_mask = top_mask.unsqueeze(0).expand_as(top_logits)
            
            # Mask invalid actions with large negative values
            masked_top_logits = top_logits.clone()
            masked_top_logits[~top_mask] = -1e8
            
            top_dist = Categorical(logits=masked_top_logits)
        else:
            top_dist = Categorical(logits=top_logits)
        
        if deterministic:
            top_action = torch.argmax(top_dist.logits, dim=-1)
        else:
            top_action = top_dist.sample()
        
        # Get context-aware sub-action logits using sampled top action
        top_embed = self.top_action_embedding(top_action)
        combined_features = torch.cat([gru_output, top_embed], dim=-1)
        sub_logits = self.sub_action_processor(combined_features)
        
        # Apply sub-action mask if provided
        if action_mask is not None:
            top_idx = top_action.item() if top_action.dim() == 0 else top_action[0].item()
            sub_mask = sub_masks[top_idx]
            
            # Convert to tensor if needed
            if isinstance(sub_mask, (list, tuple, np.ndarray)):
                sub_mask = torch.tensor(sub_mask, dtype=torch.bool, device=sub_logits.device)
            
            # Ensure mask has same batch dimension as logits
            if sub_mask.dim() == 1 and sub_logits.dim() == 2:
                sub_mask = sub_mask.unsqueeze(0).expand_as(sub_logits)
            
            # Mask invalid sub-actions
            masked_sub_logits = sub_logits.clone()
            masked_sub_logits[~sub_mask] = -1e8
            
            sub_dist = Categorical(logits=masked_sub_logits)
        else:
            sub_dist = Categorical(logits=sub_logits)
        
        if deterministic:
            sub_action = torch.argmax(sub_dist.logits, dim=-1)
        else:
            sub_action = sub_dist.sample()
        
        # Get value
        value = self.value_head(gru_output)
        
        return (top_action, sub_action, 
                top_dist.log_prob(top_action), sub_dist.log_prob(sub_action),
                value, new_hidden) 