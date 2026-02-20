#!/usr/bin/env python3

import json
import numpy as np
import sys
import os

# Add the paths to import modules
sys.path.append('MARL+IPPO')
sys.path.append('.')

sys.path.append('MARL+IPPO')
from env_wrapper import MonopolyMAv2
from test_agent import TestAgent

def run_comprehensive_action_test():
    """
    Run a comprehensive test of all actions using the TestAgent.
    """
    print("=== COMPREHENSIVE ACTION TESTING ===")
    
    # Create environment
    board_json_path = "data.json"
    env = MonopolyMAv2(num_players=4, max_steps=500, board_json_path=board_json_path)
    agent_ids = env.possible_agents
    
    # Load board metadata
    with open(board_json_path) as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}
    
    # Create test agents
    agents = {}
    for agent_id in agent_ids:
        agents[agent_id] = TestAgent(int(agent_id.split('_')[-1]), 4, board_meta)
    
    # Track all actions tested across multiple games
    all_actions_tested = set()
    action_success_count = {i: 0 for i in range(12)}
    action_attempt_count = {i: 0 for i in range(12)}
    
    # Run multiple games to test different scenarios
    for game_num in range(5):
        print(f"\n{'='*50}")
        print(f"GAME {game_num + 1}")
        print(f"{'='*50}")
        
        # Reset agents for new game
        for agent in agents.values():
            agent.reset_tried_actions()
        
        # Reset environment
        obs, _ = env.reset(seed=42 + game_num)
        
        step_count = 0
        game_actions_tested = set()
        
        while step_count < 200:  # Allow longer games to test more actions
            step_count += 1
            
            # Get the current player
            current_agent_idx = env.internal_env.game.current_player_index
            current_agent_id = f"player_{current_agent_idx}"
            current_player = env.internal_env.game.players[current_agent_idx]
            
            print(f"\n--- Step {step_count} ---")
            print(f"Current Player: {current_player.player_name}, Phase: {current_player.phase}")
            print(f"Position: {current_player.current_position}, Cash: {current_player.current_cash}")
            print(f"Properties: {len(current_player.assets)}, In Jail: {current_player.currently_in_jail}")
            
            # Get valid actions
            valid_actions = env.internal_env.game.get_valid_actions(current_player)
            print(f"Valid actions: {[i for i, valid in enumerate(valid_actions) if valid]}")
            
            # Handle pre-roll phase
            if current_player.phase == 'pre-roll':
                pre_roll_action = (7, 0)
                pre_roll_action_dict = {current_agent_id: pre_roll_action}
                obs, rewards, terminated, truncated, info = env.step(pre_roll_action_dict)
                
                if terminated[current_agent_id] or truncated[current_agent_id]:
                    break
                    
                current_agent_idx = env.internal_env.game.current_player_index
                current_agent_id = f"player_{current_agent_idx}"
                current_player = env.internal_env.game.players[current_agent_idx]
                
                # Update valid actions after pre-roll
                valid_actions = env.internal_env.game.get_valid_actions(current_player)
                print(f"Valid actions after pre-roll: {[i for i, valid in enumerate(valid_actions) if valid]}")
            
            # Get action from test agent
            agent = agents[current_agent_id]
            action = agent.get_action(obs[current_agent_id])
            top_action, sub_action = action
            
            # Track action attempts
            action_attempt_count[top_action] += 1
            game_actions_tested.add(top_action)
            all_actions_tested.add(top_action)
            
            # Check if action is valid
            if top_action < len(valid_actions) and valid_actions[top_action]:
                print(f"✓ Action {action} is VALID")
                action_success_count[top_action] += 1
            else:
                print(f"✗ Action {action} is INVALID!")
                print(f"Valid actions: {[i for i, valid in enumerate(valid_actions) if valid]}")
                # Continue instead of breaking to test more actions
                print("Continuing with default action...")
                action = (7, 0)  # Default to conclude
            
            # Step environment
            action_dict = {current_agent_id: action}
            obs, rewards, terminated, truncated, info = env.step(action_dict)
            
            # Check for errors
            if current_agent_id in info and 'error' in info[current_agent_id]:
                print(f"ERROR: {info[current_agent_id]['error']}")
                # Continue instead of breaking
                continue
            
            if terminated[current_agent_id] or truncated[current_agent_id]:
                print(f"Game {game_num + 1} ended at step {step_count}")
                break
        
        print(f"\nGame {game_num + 1} actions tested: {sorted(game_actions_tested)}")
    
    # Final report
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST RESULTS")
    print(f"{'='*60}")
    
    action_names = {
        0: "Make Trade Offer (Sell)",
        1: "Make Trade Offer (Buy)", 
        2: "Improve Property",
        3: "Sell House/Hotel",
        4: "Sell Property",
        5: "Mortgage/Free Mortgage",
        6: "Skip Turn",
        7: "Conclude Phase",
        8: "Use Get Out of Jail",
        9: "Pay Jail Fine",
        10: "Buy Property",
        11: "Respond to Trade"
    }
    
    print(f"Total unique actions tested across all games: {len(all_actions_tested)}")
    print(f"Actions tested: {sorted(all_actions_tested)}")
    
    print("\nDetailed Action Results:")
    print("-" * 80)
    print(f"{'Action':<3} {'Name':<25} {'Attempts':<10} {'Successes':<10} {'Success Rate':<12}")
    print("-" * 80)
    
    for action in range(12):
        name = action_names.get(action, 'Unknown')
        attempts = action_attempt_count[action]
        successes = action_success_count[action]
        success_rate = f"{(successes/attempts*100):.1f}%" if attempts > 0 else "N/A"
        status = "✓" if action in all_actions_tested else "✗"
        
        print(f"{action:<3} {name:<25} {attempts:<10} {successes:<10} {success_rate:<12} {status}")
    
    print("\nActions NOT tested:")
    for action in range(12):
        if action not in all_actions_tested:
            print(f"  {action}: {action_names.get(action, 'Unknown')}")
    
    print("\nActions that had failures:")
    for action in range(12):
        attempts = action_attempt_count[action]
        successes = action_success_count[action]
        if attempts > 0 and successes < attempts:
            failure_rate = (attempts - successes) / attempts * 100
            print(f"  {action}: {action_names.get(action, 'Unknown')} - {failure_rate:.1f}% failure rate")
    
    env.close()
    
    return all_actions_tested, action_attempt_count, action_success_count


if __name__ == "__main__":
    # Run the comprehensive test
    try:
        tested_actions, attempts, successes = run_comprehensive_action_test()
        
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully tested {len(tested_actions)} out of 12 possible actions")
        
        if len(tested_actions) == 12:
            print("🎉 ALL ACTIONS TESTED SUCCESSFULLY!")
        else:
            print(f"⚠️  {12 - len(tested_actions)} actions were not tested")
            
        total_attempts = sum(attempts.values())
        total_successes = sum(successes.values())
        overall_success_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
        print(f"Overall success rate: {overall_success_rate:.1f}% ({total_successes}/{total_attempts})")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 