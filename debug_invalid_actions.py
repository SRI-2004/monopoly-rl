#!/usr/bin/env python3

import json
import numpy as np
import sys
sys.path.append('MARL+IPPO')
from env_wrapper import MonopolyMAv2
from scripted_agent import ScriptedAgent

def debug_invalid_actions():
    """Debug to catch exactly when invalid actions occur."""
    
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
    
    print("=== DEBUGGING INVALID ACTIONS ===")
    
    # Run multiple games to catch the invalid action
    for game_num in range(20):
        print(f"\n=== GAME {game_num} ===")
        
        # Reset environment
        obs, _ = env.reset(seed=42 + game_num)
        
        step_count = 0
        while step_count < 100:
            step_count += 1
            
            # Get the current player and check their phase
            current_agent_idx = env.internal_env.game.current_player_index
            current_agent_id = f"player_{current_agent_idx}"
            current_player = env.internal_env.game.players[current_agent_idx]
            
            # --- MODIFICATION: Implement two-step action for pre-roll and post-roll ---
            
            # 1. Pre-roll step: Always attempt to "conclude" to roll the dice if in pre-roll phase
            if current_player.phase == 'pre-roll':
                pre_roll_action = (7, 0)  # Action 7 is "conclude"
                pre_roll_action_dict = {current_agent_id: pre_roll_action}
                
                # Check if pre-roll action is valid
                valid_actions = env.internal_env.game.get_valid_actions(current_player)
                if not valid_actions[7]:
                    print(f"INVALID PRE-ROLL ACTION!")
                    print(f"Player: {current_player.player_name}, Phase: {current_player.phase}")
                    print(f"Valid actions: {[i for i, valid in enumerate(valid_actions) if valid]}")
                    return
                
                # Execute the pre-roll action
                obs, rewards, terminated, truncated, info = env.step(pre_roll_action_dict)
                
                # Check for errors
                if current_agent_id in info and 'error' in info[current_agent_id]:
                    print(f"PRE-ROLL ERROR: {info[current_agent_id]['error']}")
                    return
                
                # Check if game ended during pre-roll
                if terminated[current_agent_id] or truncated[current_agent_id]:
                    break
                
                # Update current player info after pre-roll
                current_agent_idx = env.internal_env.game.current_player_index
                current_agent_id = f"player_{current_agent_idx}"
                current_player = env.internal_env.game.players[current_agent_idx]
            
            # 2. Post-roll step: Get the actual action from the agent
            # Get action from the appropriate agent
            agent = agents[current_agent_id]
            action = agent.get_action(obs[current_agent_id])
            
            # Check if post-roll action is valid
            valid_actions_post = env.internal_env.game.get_valid_actions(current_player)
            top_action, sub_action = action
            if not (top_action < len(valid_actions_post) and valid_actions_post[top_action]):
                print(f"INVALID POST-ROLL ACTION!")
                print(f"Player: {current_player.player_name}, Phase: {current_player.phase}")
                print(f"Position: {current_player.current_position}")
                print(f"Cash: {current_player.current_cash}")
                print(f"In Jail: {current_player.currently_in_jail}")
                print(f"Can Buy: {current_player.can_buy_property()}")
                print(f"Action attempted: {action}")
                print(f"Valid actions: {[i for i, valid in enumerate(valid_actions_post) if valid]}")
                
                # Show the observation the agent used
                current_obs = obs[current_agent_id]
                player_state = current_obs['player']
                print(f"Player state vector: {player_state}")
                
                # Decode the observation
                obs_is_in_jail = player_state[10] > 0
                obs_can_buy = player_state[12] > 0
                obs_current_cash = player_state[7]
                obs_current_pos = int(player_state[0])
                obs_phase_vec = player_state[13:16]
                obs_phase = np.argmax(obs_phase_vec)
                
                print(f"Observation decoded:")
                print(f"  Position: {obs_current_pos}")
                print(f"  Cash: {obs_current_cash}")
                print(f"  In Jail: {obs_is_in_jail}")
                print(f"  Can Buy: {obs_can_buy}")
                print(f"  Phase: {obs_phase} ({'pre-roll' if obs_phase == 0 else 'post-roll' if obs_phase == 1 else 'out-of-turn'})")
                
                return
            
            # Step environment with the post-roll action
            action_dict = {current_agent_id: action}
            obs, rewards, terminated, truncated, info = env.step(action_dict)
            
            # Check for errors
            if current_agent_id in info and 'error' in info[current_agent_id]:
                print(f"POST-ROLL ERROR: {info[current_agent_id]['error']}")
                return
            
            if terminated[current_agent_id] or truncated[current_agent_id]:
                break
    
    print("\n=== NO INVALID ACTIONS FOUND ===")
    env.close()

if __name__ == "__main__":
    debug_invalid_actions() 