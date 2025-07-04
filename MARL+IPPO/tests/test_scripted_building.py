#!/usr/bin/env python3

import numpy as np
import json
import sys
sys.path.append('.')
from env_wrapper import MonopolyMAv2
from scripted_agent import ScriptedAgent

def test_scripted_agent_building():
    """
    Test that the enhanced scripted agent can build houses on monopolies.
    """
    print("=== TESTING ENHANCED SCRIPTED AGENT HOUSE BUILDING ===")
    
    # Create environment
    env = MonopolyMAv2(board_json_path="../data.json", num_players=4, max_steps=1000)
    
    # Load board metadata
    with open('../data.json', 'r') as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}
    
    # Create scripted agent
    agent = ScriptedAgent(player_id=0, num_players=4, board_meta=board_meta)
    
    # Reset environment
    observations = env.reset()
    
    # Get the first player and game state
    player = env.internal_env.players[0]
    game = env.internal_env.game
    board = game.board
    
    print(f"Player: {player.player_name}")
    print(f"Initial cash: ${player.current_cash}")
    print(f"Initial assets: {player.assets}")
    print(f"Initial monopolies: {player.full_color_sets_possessed}")
    
    # Manually give player brown monopoly (cheapest to build on)
    brown_props = [1, 3]  # Mediterranean Ave and Baltic Ave
    
    print(f"\n=== GIVING PLAYER BROWN MONOPOLY ===")
    for prop_id in brown_props:
        prop_meta = board_meta[str(prop_id)]
        print(f"Giving player {prop_meta['name']} (ID: {prop_id})")
        
        # Add to player's assets
        player.add_asset(prop_id)
        
        # Update board state
        prop_idx = board.property_id_to_index[prop_id]
        owner_vector = np.zeros(board.num_owners, dtype=np.float32)
        owner_vector[player.player_id + 1] = 1.0
        board.state[prop_idx, 0:board.num_owners] = owner_vector
        
        # Update player counts and monopolies
        game._update_player_counts(player)
        
        # Update monopoly flags
        color_group = prop_meta.get("color_group")
        if color_group:
            board.update_monopoly_flag(color_group)
            game._update_player_monopolies(player)
    
    # Set player to pre-roll phase so they can build
    player.update_phase('pre-roll')
    
    print(f"\nAfter setup:")
    print(f"Player assets: {player.assets}")
    print(f"Player monopolies: {player.full_color_sets_possessed}")
    print(f"Player cash: ${player.current_cash}")
    print(f"Player phase: {player.phase}")
    
    # Check board monopoly flags
    for prop_id in brown_props:
        prop_idx = board.property_id_to_index[prop_id]
        monopoly_flag = board.state[prop_idx, 5]
        house_frac = board.state[prop_idx, 6]
        hotel_frac = board.state[prop_idx, 7]
        print(f"Property {prop_id}: monopoly={monopoly_flag}, houses={house_frac}, hotels={hotel_frac}")
    
    # Get current observation for the agent
    obs = env._get_all_observations()['player_0']
    
    print(f"\n=== TESTING AGENT DECISION ===")
    print(f"Agent color groups: {agent.color_groups}")
    
    # Test the agent's decision
    action = agent.get_action(obs)
    print(f"Agent chose action: {action}")
    
    # If it's a building action, try to execute it
    if action[0] == 2:  # Improve Property
        print(f"Agent wants to build! Action: {action}")
        
        # Execute the action
        try:
            _, reward, terminated, truncated, info = env.step({'player_0': action})
            print(f"Action result: {info.get('result', 'No result message')}")
            print(f"Reward: {reward}")
            
            # Check if houses were built
            total_houses_after = np.sum(board.state[:, 6])
            print(f"Total house fraction after action: {total_houses_after}")
            
            # Check specific properties
            for prop_id in brown_props:
                prop_idx = board.property_id_to_index[prop_id]
                house_frac = board.state[prop_idx, 6]
                hotel_frac = board.state[prop_idx, 7]
                print(f"Property {prop_id}: houses={house_frac}, hotels={hotel_frac}")
            
            # Check behavioral metrics
            behavioral_metrics = info.get('behavioral_metrics', {})
            print(f"Behavioral metrics: {behavioral_metrics}")
            
        except Exception as e:
            print(f"Error executing action: {e}")
    else:
        print(f"Agent did not choose to build (action {action[0]})")
    
    print(f"\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    test_scripted_agent_building()