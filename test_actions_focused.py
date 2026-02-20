#!/usr/bin/env python3

import json
import numpy as np
import sys
import os

# Add the paths to import modules
sys.path.append('MARL+IPPO')
sys.path.append('.')

from env_wrapper import MonopolyMAv2
from test_agent import TestAgent

def test_specific_action_issues():
    """
    Test specific actions that have been causing issues.
    """
    print("=== FOCUSED ACTION TESTING ===")
    
    # Create environment
    board_json_path = "data.json"
    env = MonopolyMAv2(num_players=4, max_steps=100, board_json_path=board_json_path)
    agent_ids = env.possible_agents
    
    # Load board metadata
    with open(board_json_path) as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}
    
    # Create test agents
    agents = {}
    for agent_id in agent_ids:
        agents[agent_id] = TestAgent(int(agent_id.split('_')[-1]), 4, board_meta)
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Test specific problematic actions
    test_actions = [
        # Format: (action, sub_action, description)
        (0, 0, "Make Trade Offer (Sell) - sub 0"),
        (0, 1, "Make Trade Offer (Sell) - sub 1"),
        (0, 84, "Make Trade Offer (Sell) - sub 84"),
        (1, 0, "Make Trade Offer (Buy) - sub 0"),
        (1, 1, "Make Trade Offer (Buy) - sub 1"),
        (2, 0, "Improve Property - sub 0"),
        (2, 1, "Improve Property - sub 1"),
        (3, 0, "Sell House/Hotel - sub 0"),
        (3, 1, "Sell House/Hotel - sub 1"),
        (4, 0, "Sell Property - sub 0"),
        (4, 1, "Sell Property - sub 1"),
        (5, 0, "Mortgage/Free Mortgage - sub 0"),
        (5, 1, "Mortgage/Free Mortgage - sub 1"),
        (6, 0, "Skip Turn"),
        (7, 0, "Conclude Phase"),
        (8, 0, "Use Get Out of Jail"),
        (9, 0, "Pay Jail Fine"),
        (10, 0, "Buy Property - No"),
        (10, 1, "Buy Property - Yes"),
        (11, 0, "Respond to Trade - Reject"),
        (11, 1, "Respond to Trade - Accept"),
    ]
    
    results = {}
    
    for action_tuple in test_actions:
        top_action, sub_action, description = action_tuple
        
        print(f"\n--- Testing {description} ---")
        
        # Get the current player
        current_agent_idx = env.internal_env.game.current_player_index
        current_agent_id = f"player_{current_agent_idx}"
        current_player = env.internal_env.game.players[current_agent_idx]
        
        print(f"Current Player: {current_player.player_name}, Phase: {current_player.phase}")
        print(f"Position: {current_player.current_position}, Cash: {current_player.current_cash}")
        print(f"Properties: {len(current_player.assets)}, In Jail: {current_player.currently_in_jail}")
        
        # Get valid actions
        valid_actions = env.internal_env.game.get_valid_actions(current_player)
        print(f"Valid top-level actions: {[i for i, valid in enumerate(valid_actions) if valid]}")
        
        # Check if the top-level action is valid
        if top_action < len(valid_actions) and valid_actions[top_action]:
            print(f"✓ Top-level action {top_action} is valid")
            
            # Get valid sub-actions
            valid_sub_actions = env.internal_env.game.get_valid_subactions(current_player, top_action)
            print(f"Valid sub-actions: {len(valid_sub_actions)} total, {sum(valid_sub_actions)} valid")
            
            if sub_action < len(valid_sub_actions) and valid_sub_actions[sub_action]:
                print(f"✓ Sub-action {sub_action} is valid")
                
                # Try the action
                action_dict = {current_agent_id: (top_action, sub_action)}
                try:
                    obs, rewards, terminated, truncated, info = env.step(action_dict)
                    
                    if current_agent_id in info and 'error' in info[current_agent_id]:
                        print(f"✗ Action failed: {info[current_agent_id]['error']}")
                        results[description] = "FAILED"
                    else:
                        print(f"✓ Action succeeded")
                        results[description] = "SUCCESS"
                        
                except Exception as e:
                    print(f"✗ Action caused exception: {e}")
                    results[description] = "EXCEPTION"
            else:
                print(f"✗ Sub-action {sub_action} is invalid")
                if len(valid_sub_actions) > 0:
                    valid_indices = [i for i, valid in enumerate(valid_sub_actions) if valid]
                    print(f"Valid sub-action indices: {valid_indices[:10]}...")  # Show first 10
                results[description] = "INVALID_SUB"
        else:
            print(f"✗ Top-level action {top_action} is invalid")
            results[description] = "INVALID_TOP"
        
        # Reset environment for next test
        obs, _ = env.reset(seed=42)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FOCUSED TEST RESULTS")
    print(f"{'='*60}")
    
    success_count = sum(1 for result in results.values() if result == "SUCCESS")
    total_count = len(results)
    
    print(f"Successful actions: {success_count}/{total_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    
    print("\nDetailed Results:")
    for description, result in results.items():
        status_symbol = "✓" if result == "SUCCESS" else "✗"
        print(f"{status_symbol} {description:<35} {result}")
    
    print("\nIssue Summary:")
    issue_counts = {}
    for result in results.values():
        issue_counts[result] = issue_counts.get(result, 0) + 1
    
    for issue, count in issue_counts.items():
        print(f"  {issue}: {count} actions")
    
    env.close()
    return results


if __name__ == "__main__":
    try:
        results = test_specific_action_issues()
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        success_count = sum(1 for result in results.values() if result == "SUCCESS")
        total_count = len(results)
        
        if success_count == total_count:
            print("🎉 ALL TESTED ACTIONS WORKED!")
        else:
            print(f"⚠️  {total_count - success_count} out of {total_count} actions had issues")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 