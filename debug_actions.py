#!/usr/bin/env python3

import json
import numpy as np
import sys
sys.path.append('MARL+IPPO')
from monopoly_env.envs.monopoly_env import MonopolyEnv
from scripted_agent import ScriptedAgent

def debug_scripted_agent_actions():
    """Debug what actions the scripted agent is trying to take and why they might be invalid."""
    
    # Create environment
    env = MonopolyEnv(max_steps=50)
    
    # Load board metadata
    with open('monopoly_env/config.py') as f:
        config_content = f.read()
    
    # Extract board JSON path from config
    board_json_path = 'data.json'  # Default path
    
    with open(board_json_path) as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}
    
    # Create scripted agent
    agent = ScriptedAgent(0, 4, board_meta)
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    print("=== DEBUGGING SCRIPTED AGENT ACTIONS ===")
    
    for step in range(20):  # Debug first 20 steps
        current_player = env.players[env.game.current_player_index]
        
        print(f"\n--- Step {step} ---")
        print(f"Current Player: {current_player.player_name} (ID: {env.game.current_player_index})")
        print(f"Phase: {current_player.phase}")
        print(f"Position: {current_player.current_position}")
        print(f"Cash: {current_player.current_cash}")
        print(f"In Jail: {current_player.currently_in_jail}")
        print(f"Can Buy: {current_player.can_buy_property()}")
        
        # Get valid actions
        valid_actions = env.game.get_valid_actions(current_player)
        print(f"Valid Actions: {[i for i, valid in enumerate(valid_actions) if valid]}")
        
        # Get action from scripted agent
        action = agent.get_action(obs)
        print(f"Scripted Agent Action: {action}")
        
        # Check if action is valid
        top_action, sub_action = action
        if top_action < len(valid_actions) and valid_actions[top_action]:
            print(f"Action {top_action} is VALID")
        else:
            print(f"Action {top_action} is INVALID!")
            print(f"Valid actions are: {valid_actions}")
            
        # Check sub-action if needed
        actions_with_sub = {0, 1, 2, 3, 4, 5, 10, 11}
        if top_action in actions_with_sub:
            valid_sub_mask = env.game.get_valid_subactions(current_player, top_action)
            print(f"Valid sub-actions for action {top_action}: {valid_sub_mask}")
            if sub_action < len(valid_sub_mask) and valid_sub_mask[sub_action]:
                print(f"Sub-action {sub_action} is VALID")
            else:
                print(f"Sub-action {sub_action} is INVALID!")
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if 'error' in info:
            print(f"ERROR: {info['error']}")
            break
            
        if terminated or truncated:
            print("Game ended")
            break
    
    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_scripted_agent_actions() 