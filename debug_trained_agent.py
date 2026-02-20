import torch
import numpy as np
from env_wrapper import MonopolyMAv2, preprocess_obs
from network import ActorCritic
from scripted_agent import ScriptedAgent
import json

def load_policy(ckpt_path, input_dim, action_dims, device):
    """Load a trained policy."""
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        hidden_dim = checkpoint["hidden_dim"]
        
        model = ActorCritic(input_dim, action_dims, hidden_dim).to(device)
        
        state_dict = checkpoint.get("policy_state_dict") or checkpoint.get("model_state_dict")
        
        # Handle torch.compile models
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def debug_trained_agent():
    """Debug the trained agent's action selection."""
    device = torch.device("cpu")
    
    # Load the trained agent
    ckpt_path = "/home/srinivasan/PycharmProjects/monopoly-rl/MARL+IPPO/rl_agent/checkpoints/enhanced_training/Monopoly_IPPO_1751640470/policy_player_3_update_1400.pt"
    board_json = "/home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json"
    
    # Create environment
    env = MonopolyMAv2(num_players=4, max_steps=100, board_json_path=board_json)
    
    # Get dimensions
    sample_obs_dict = env.observation_space("player_0").sample()
    total_input_dim = len(preprocess_obs(sample_obs_dict, 4, 0))
    action_dims = env.action_space("player_0").nvec.tolist()
    
    print(f"Input dim: {total_input_dim}")
    print(f"Action dims: {action_dims}")
    
    # Load trained agent
    trained_agent = load_policy(ckpt_path, total_input_dim, action_dims, device)
    if trained_agent is None:
        print("Failed to load trained agent")
        return
    
    # Load board metadata for scripted agents
    with open(board_json) as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}
    
    # Create scripted agents
    scripted_agents = {
        "player_1": ScriptedAgent(1, 4, board_meta),
        "player_2": ScriptedAgent(2, 4, board_meta),
        "player_3": ScriptedAgent(3, 4, board_meta)
    }
    
    # Initialize environment
    obs, info = env.reset(seed=42)
    
    # Initialize hidden state for trained agent
    hx = torch.zeros(1, 1, trained_agent.hidden_dim).to(device)
    
    print("\n=== DEBUGGING TRAINED AGENT ACTIONS ===")
    
    for step in range(20):  # Debug first 20 steps
        current_agent_idx = env.internal_env.game.current_player_index
        current_agent_id = f"player_{current_agent_idx}"
        current_player = env.internal_env.game.players[current_agent_idx]
        
        print(f"\nStep {step + 1}: Current player: {current_agent_id}")
        print(f"Player phase: {current_player.phase}")
        print(f"Player position: {current_player.position}")
        print(f"Player cash: {current_player.cash}")
        
        # Get current observation
        current_obs = obs[current_agent_id]
        
        # Check valid actions
        valid_actions = current_obs.get('action_mask', [])
        if len(valid_actions) >= 2:
            valid_top_actions = [i for i, mask in enumerate(valid_actions[0]) if mask]
            print(f"Valid top actions: {valid_top_actions}")
            print(f"Action mask shape: {[len(mask) for mask in valid_actions]}")
        
        if current_agent_id == "player_0":
            # This is our trained agent
            processed_obs_np = preprocess_obs(current_obs, 4, 0)
            agent_obs = torch.tensor(processed_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            print(f"Processed obs shape: {agent_obs.shape}")
            
            # Get action from trained agent
            with torch.no_grad():
                top_action, sub_action, _, _, _, hx = trained_agent.get_action_and_value(
                    agent_obs, hx, deterministic=True
                )
                
            top_action_val = top_action.item()
            sub_action_val = sub_action.item()
            
            print(f"Trained agent selected: top_action={top_action_val}, sub_action={sub_action_val}")
            
            # Check if this action is valid
            if len(valid_actions) >= 2 and len(valid_actions[0]) > top_action_val:
                is_valid_top = valid_actions[0][top_action_val]
                print(f"Is top action valid? {is_valid_top}")
                
                if is_valid_top and len(valid_actions[1]) > sub_action_val:
                    is_valid_sub = valid_actions[1][sub_action_val]
                    print(f"Is sub action valid? {is_valid_sub}")
                else:
                    print(f"Sub action out of range or top action invalid")
            else:
                print("Action mask format issue or top action out of range")
                
            action = (top_action_val, sub_action_val)
        else:
            # Scripted agent
            scripted_agent = scripted_agents[current_agent_id]
            action = scripted_agent.get_action(current_obs)
            print(f"Scripted agent selected: {action}")
        
        # Execute action
        action_dict = {current_agent_id: action}
        
        try:
            obs, rewards, terminated, truncated, info = env.step(action_dict)
            print(f"Action executed successfully")
            print(f"Rewards: {rewards}")
            
            if terminated[current_agent_id] or truncated[current_agent_id]:
                print("Game ended")
                break
                
        except Exception as e:
            print(f"Error executing action: {e}")
            break
    
    env.close()

if __name__ == "__main__":
    debug_trained_agent() 