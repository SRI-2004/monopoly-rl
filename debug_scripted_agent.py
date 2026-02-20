#!/usr/bin/env python3

import json
import numpy as np
import sys
sys.path.append('MARL+IPPO')
from env_wrapper import MonopolyMAv2
from scripted_agent import ScriptedAgent

def debug_scripted_agent_observations():
    """Debug the scripted agent's interpretation of observations."""
    
    # Create environment like in evaluation
    env = MonopolyMAv2(num_players=4, max_steps=100)
    agent_ids = env.possible_agents
    
    # Load board metadata
    with open('data.json') as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}
    
    # Create scripted agents
    agents = {}
    for agent_id in agent_ids:
        agents[agent_id] = ScriptedAgent(int(agent_id.split('_')[-1]), 4, board_meta)
    
    print("=== DEBUGGING SCRIPTED AGENT OBSERVATIONS ===")
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    for step in range(10):
        print(f"\n--- Step {step} ---")
        
        # Get the current player
        current_agent_idx = env.internal_env.game.current_player_index
        current_agent_id = f"player_{current_agent_idx}"
        current_player = env.internal_env.game.players[current_agent_idx]
        
        print(f"Current Player: {current_player.player_name} (ID: {current_agent_idx})")
        print(f"Actual Phase: {current_player.phase}")
        print(f"Actual Position: {current_player.current_position}")
        print(f"Actual Cash: {current_player.current_cash}")
        print(f"Actual In Jail: {current_player.currently_in_jail}")
        print(f"Actual Can Buy: {current_player.can_buy_property()}")
        
        # Get the observation for the current agent
        current_obs = obs[current_agent_id]
        print(f"Observation keys: {current_obs.keys()}")
        
        # Decode the player state from observation
        player_state = current_obs['player']
        print(f"Player state vector: {player_state}")
        
        # Decode the observation like the scripted agent does
        is_in_jail = player_state[10] > 0
        can_buy = player_state[12] > 0
        current_cash = player_state[7]
        current_pos = int(player_state[0])
        
        phase_vec = player_state[13:16]
        phase = np.argmax(phase_vec)  # 0: pre-roll, 1: post-roll, 2: out-of-turn
        
        print(f"Decoded from observation:")
        print(f"  Position: {current_pos}")
        print(f"  Cash: {current_cash}")
        print(f"  In Jail: {is_in_jail}")
        print(f"  Can Buy: {can_buy}")
        print(f"  Phase vector: {phase_vec}")
        print(f"  Phase: {phase} ({'pre-roll' if phase == 0 else 'post-roll' if phase == 1 else 'out-of-turn'})")
        
        # Compare with actual values
        print(f"Comparison:")
        print(f"  Position: {current_pos} vs {current_player.current_position} {'✓' if current_pos == current_player.current_position else '✗'}")
        print(f"  Cash: {current_cash} vs {current_player.current_cash} {'✓' if abs(current_cash - current_player.current_cash) < 0.01 else '✗'}")
        print(f"  In Jail: {is_in_jail} vs {current_player.currently_in_jail} {'✓' if is_in_jail == current_player.currently_in_jail else '✗'}")
        print(f"  Can Buy: {can_buy} vs {current_player.can_buy_property()} {'✓' if can_buy == current_player.can_buy_property() else '✗'}")
        
        # Get action from scripted agent
        action = agents[current_agent_id].get_action(current_obs)
        print(f"Scripted agent action: {action}")
        
        # Check if action is valid
        valid_actions = env.internal_env.game.get_valid_actions(current_player)
        top_action, sub_action = action
        if top_action < len(valid_actions) and valid_actions[top_action]:
            print(f"Action {top_action} is VALID")
        else:
            print(f"Action {top_action} is INVALID!")
            print(f"Valid actions: {[i for i, valid in enumerate(valid_actions) if valid]}")
            break
        
        # Step environment
        action_dict = {current_agent_id: action}
        obs, rewards, terminated, truncated, info = env.step(action_dict)
        
        # Check for errors
        if current_agent_id in info and 'error' in info[current_agent_id]:
            print(f"ERROR: {info[current_agent_id]['error']}")
            break
        
        if terminated[current_agent_id] or truncated[current_agent_id]:
            print("Game ended")
            break
    
    print("\n=== DEBUG COMPLETE ===")
    env.close()

if __name__ == "__main__":
    debug_scripted_agent_observations() 