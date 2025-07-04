#!/usr/bin/env python3

import numpy as np
import json
import sys
sys.path.append('.')
from env_wrapper import MonopolyMAv2

def test_get_out_of_jail_and_houses():
    """
    Simple test to check Get Out of Jail cards and house building.
    """
    print("=== TESTING GET OUT OF JAIL AND HOUSE BUILDING ===")
    
    # Create environment
    env = MonopolyMAv2(board_json_path="../data.json", num_players=4, max_steps=1000)
    agent_ids = env.possible_agents
    
    # Load board metadata
    with open('../data.json', 'r') as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}
    
    # Track statistics
    stats = {
        'get_out_of_jail_attempts': 0,
        'get_out_of_jail_successes': 0,
        'get_out_of_jail_failures': 0,
        'jail_fine_payments': 0,
        'house_building_attempts': 0,
        'house_building_successes': 0,
        'property_purchases': 0,
        'games_played': 0
    }
    
    # Run multiple games to test
    for game_num in range(20):
        print(f"\n--- Game {game_num + 1} ---")
        obs, _ = env.reset(seed=42 + game_num)
        stats['games_played'] += 1
        
        step_count = 0
        while step_count < 200:  # Limit steps per game
            step_count += 1
            
            # Get the current player
            current_agent_idx = env.internal_env.game.current_player_index
            current_agent_id = f"player_{current_agent_idx}"
            current_player = env.internal_env.game.players[current_agent_idx]
            
            # Get valid actions
            valid_actions = env.internal_env.game.get_valid_actions(current_player)
            valid_action_indices = [i for i, valid in enumerate(valid_actions) if valid]
            
            # Two-phase system like in evaluation
            if current_player.phase == 'pre-roll':
                pre_roll_action = (7, 0)  # Conclude to roll dice
                pre_roll_action_dict = {current_agent_id: pre_roll_action}
                obs, rewards, terminated, truncated, info = env.step(pre_roll_action_dict)
                
                if terminated[current_agent_id] or truncated[current_agent_id]:
                    break
                    
                current_agent_idx = env.internal_env.game.current_player_index
                current_agent_id = f"player_{current_agent_idx}"
                current_player = env.internal_env.game.players[current_agent_idx]
                valid_actions = env.internal_env.game.get_valid_actions(current_player)
                valid_action_indices = [i for i, valid in enumerate(valid_actions) if valid]
            
            # Choose action based on what's available
            action = None
            
            # Priority 1: Handle jail situations
            if current_player.currently_in_jail:
                if 8 in valid_action_indices:  # Use Get Out of Jail
                    action = (8, 0)
                    stats['get_out_of_jail_attempts'] += 1
                    print(f"  Step {step_count}: Trying Use Get Out of Jail (action 8)")
                elif 9 in valid_action_indices:  # Pay Jail Fine
                    action = (9, 0)
                    stats['jail_fine_payments'] += 1
                    print(f"  Step {step_count}: Paying Jail Fine (action 9)")
            
            # Priority 2: Buy property if available
            elif 10 in valid_action_indices and current_player.phase == 'post-roll':
                property_meta = board_meta.get(str(current_player.current_position))
                if property_meta and 'price' in property_meta and current_player.current_cash >= property_meta['price']:
                    action = (10, 1)  # Buy property
                    stats['property_purchases'] += 1
                    print(f"  Step {step_count}: Buying property {property_meta['name']} for ${property_meta['price']}")
                else:
                    action = (10, 0)  # Don't buy
            
            # Priority 3: Try to build houses if we have monopolies
            elif 2 in valid_action_indices and len(current_player.full_color_sets_possessed) > 0:
                action = (2, 0)  # Try to improve property (build house)
                stats['house_building_attempts'] += 1
                print(f"  Step {step_count}: Trying to build house (action 2)")
            
            # Priority 4: Default actions
            elif 7 in valid_action_indices:
                action = (7, 0)  # Conclude phase
            elif 6 in valid_action_indices:
                action = (6, 0)  # Skip turn
            else:
                # Pick any valid action
                if valid_action_indices:
                    action = (valid_action_indices[0], 0)
                else:
                    print(f"  ERROR: No valid actions available!")
                    break
            
            # Execute the action
            if action:
                action_dict = {current_agent_id: action}
                obs, rewards, terminated, truncated, info = env.step(action_dict)
                
                # Check for errors and successes
                if current_agent_id in info:
                    if 'error' in info[current_agent_id]:
                        error_msg = info[current_agent_id]['error']
                        print(f"    ERROR: {error_msg}")
                        
                        # Track specific errors
                        if "Get Out of Jail card" in error_msg:
                            stats['get_out_of_jail_failures'] += 1
                        elif action[0] == 8:  # Use Get Out of Jail
                            stats['get_out_of_jail_failures'] += 1
                    elif action[0] == 8:  # Use Get Out of Jail succeeded
                        stats['get_out_of_jail_successes'] += 1
                        print(f"    SUCCESS: Used Get Out of Jail card")
                    elif action[0] == 2:  # House building
                        if 'houses_built' in info[current_agent_id].get('behavioral_metrics', {}):
                            houses_built = info[current_agent_id]['behavioral_metrics']['houses_built']
                            if houses_built > 0:
                                stats['house_building_successes'] += 1
                                print(f"    SUCCESS: Built {houses_built} houses")
                
                if terminated[current_agent_id] or truncated[current_agent_id]:
                    break
        
        print(f"  Game {game_num + 1} completed in {step_count} steps")
    
    # Print final statistics
    print(f"\n=== FINAL STATISTICS ===")
    print(f"Games played: {stats['games_played']}")
    print(f"Get Out of Jail attempts: {stats['get_out_of_jail_attempts']}")
    print(f"Get Out of Jail successes: {stats['get_out_of_jail_successes']}")
    print(f"Get Out of Jail failures: {stats['get_out_of_jail_failures']}")
    print(f"Jail fine payments: {stats['jail_fine_payments']}")
    print(f"House building attempts: {stats['house_building_attempts']}")
    print(f"House building successes: {stats['house_building_successes']}")
    print(f"Property purchases: {stats['property_purchases']}")
    
    # Analysis
    print(f"\n=== ANALYSIS ===")
    if stats['get_out_of_jail_attempts'] > 0:
        success_rate = (stats['get_out_of_jail_successes'] / stats['get_out_of_jail_attempts']) * 100
        print(f"Get Out of Jail success rate: {success_rate:.1f}%")
        
        if stats['get_out_of_jail_failures'] > 0:
            print("❌ Get Out of Jail errors ARE occurring - this suggests players are trying to use cards they don't have")
            print("   This is likely due to RANDOMNESS - not all players draw Get Out of Jail cards")
        else:
            print("✅ No Get Out of Jail errors - all attempts were successful")
    else:
        print("No Get Out of Jail attempts made during testing")
    
    if stats['house_building_attempts'] > 0:
        house_success_rate = (stats['house_building_successes'] / stats['house_building_attempts']) * 100
        print(f"House building success rate: {house_success_rate:.1f}%")
        
        if stats['house_building_successes'] == 0:
            print("❌ House building is NOT working - no houses were built despite attempts")
            print("   This suggests an issue with monopoly detection or building logic")
        else:
            print("✅ House building IS working - houses were successfully built")
    else:
        print("No house building attempts made during testing")
    
    env.close()

if __name__ == "__main__":
    test_get_out_of_jail_and_houses() 