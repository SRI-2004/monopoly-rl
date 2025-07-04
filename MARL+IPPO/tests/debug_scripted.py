#!/usr/bin/env python3

import numpy as np
import json
import sys
sys.path.append('.')
from env_wrapper import MonopolyMAv2
from scripted_agent import ScriptedAgent

def debug_scripted_agent():
    """
    Debug the scripted agent's house building logic.
    """
    print("=== DEBUGGING SCRIPTED AGENT HOUSE BUILDING ===")
    
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
    
    # Manually give player brown monopoly
    brown_props = [1, 3]
    
    for prop_id in brown_props:
        player.add_asset(prop_id)
        prop_idx = board.property_id_to_index[prop_id]
        owner_vector = np.zeros(board.num_owners, dtype=np.float32)
        owner_vector[player.player_id + 1] = 1.0
        board.state[prop_idx, 0:board.num_owners] = owner_vector
        game._update_player_counts(player)
        prop_meta = board_meta[str(prop_id)]
        color_group = prop_meta.get("color_group")
        if color_group:
            board.update_monopoly_flag(color_group)
            game._update_player_monopolies(player)
    
    player.update_phase('pre-roll')
    
    # Get observation
    obs = env._get_all_observations()['player_0']
    
    print(f"=== DEBUGGING EACH STEP ===")
    print(f"Player phase: {player.phase}")
    print(f"Player assets: {player.assets}")
    print(f"Player monopolies: {player.full_color_sets_possessed}")
    print(f"Player cash: ${player.current_cash}")
    
    # Debug the agent's internal logic
    print(f"\n=== STEP 1: GET OWNED PROPERTIES ===")
    owned_props = agent._get_owned_properties(obs)
    print(f"Owned properties detected: {owned_props}")
    
    print(f"\n=== BOARD STATE DEBUG ===")
    board_state = obs['board']
    board_2d = board_state.reshape(28, 8)
    print(f"Board state shape: {board_2d.shape}")
    print(f"Player ID: {agent.player_id}, Owner index: {agent.player_id + 1}")
    
    for i in range(min(10, 28)):  # Show first 10 properties
        owner_vec = board_2d[i, 0:5]  # First 5 are owners
        print(f"Property {i}: owners={owner_vec}")
    
    print(f"\n=== STEP 2: DETECT MONOPOLIES ===")
    monopolies = agent._detect_monopolies(owned_props)
    print(f"Monopolies detected: {monopolies}")
    
    print(f"\n=== STEP 3: GET BUILDABLE PROPERTIES ===")
    buildable_props = agent._get_buildable_properties(obs, monopolies)
    print(f"Buildable properties: {buildable_props}")
    
    print(f"\n=== STEP 4: CHOOSE BUILDING ACTION ===")
    building_choice = agent._choose_building_action(buildable_props, player.current_cash)
    print(f"Building choice: {building_choice}")
    
    if building_choice:
        print(f"\n=== STEP 5: CONVERT TO SUB-ACTION ===")
        sub_action = agent._convert_to_sub_action(building_choice)
        print(f"Sub-action: {sub_action}")
    
    # Check the actual board state for brown properties
    print(f"\n=== ACTUAL BOARD STATE FOR BROWN PROPERTIES ===")
    for prop_id in brown_props:
        prop_idx = board.property_id_to_index[prop_id]
        prop_state = board.state[prop_idx]
        print(f"Property {prop_id} (index {prop_idx}): {prop_state}")
    
    print(f"\n=== PROPERTY MAPPING DEBUG ===")
    print(f"Agent prop_id_to_meta keys: {list(agent.prop_id_to_meta.keys())[:10]}")
    for prop_id in brown_props:
        if prop_id in agent.prop_id_to_meta:
            print(f"Property {prop_id} meta: {agent.prop_id_to_meta[prop_id]}")
        else:
            print(f"Property {prop_id} not found in agent meta")

if __name__ == "__main__":
    debug_scripted_agent() 