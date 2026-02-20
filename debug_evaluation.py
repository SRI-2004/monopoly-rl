#!/usr/bin/env python3

import json
import numpy as np
import sys
sys.path.append('MARL+IPPO')
from env_wrapper import MonopolyMAv2
from scripted_agent import ScriptedAgent

def debug_evaluation_two_phase():
    """Debug the two-phase system used in evaluation to understand invalid actions."""
    
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
    
    print("=== DEBUGGING EVALUATION TWO-PHASE SYSTEM ===")
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    step_count = 0
    while step_count < 50:  # Debug first 50 steps
        step_count += 1
        
        # Get the current player and check their phase
        current_agent_idx = env.internal_env.game.current_player_index
        current_agent_id = f"player_{current_agent_idx}"
        current_player = env.internal_env.game.players[current_agent_idx]
        
        print(f"\n--- Step {step_count} ---")
        print(f"Current Player: {current_player.player_name} (ID: {current_agent_idx})")
        print(f"Phase: {current_player.phase}")
        print(f"Position: {current_player.current_position}")
        print(f"Cash: {current_player.current_cash}")
        
        # Get valid actions for current state
        valid_actions = env.internal_env.game.get_valid_actions(current_player)
        print(f"Valid Actions: {[i for i, valid in enumerate(valid_actions) if valid]}")
        
        # --- MODIFICATION: Implement two-step action for pre-roll and post-roll ---
        
        # 1. Pre-roll step: Always attempt to "conclude" to roll the dice if in pre-roll phase
        if current_player.phase == 'pre-roll':
            pre_roll_action = (7, 0)  # Action 7 is "conclude"
            pre_roll_action_dict = {current_agent_id: pre_roll_action}
            
            print(f"PRE-ROLL: Executing action {pre_roll_action} for {current_agent_id}")
            
            # Check if pre-roll action is valid
            if not valid_actions[7]:
                print(f"ERROR: Pre-roll action 7 is INVALID! Valid actions: {valid_actions}")
            
            # Execute the pre-roll action
            obs, rewards, terminated, truncated, info = env.step(pre_roll_action_dict)
            
            # Check for errors
            if current_agent_id in info and 'error' in info[current_agent_id]:
                print(f"PRE-ROLL ERROR: {info[current_agent_id]['error']}")
                break
            
            # Check if game ended during pre-roll
            if terminated[current_agent_id] or truncated[current_agent_id]:
                print("Game ended during pre-roll")
                break
            
            # Update current player info after pre-roll
            current_agent_idx = env.internal_env.game.current_player_index
            current_agent_id = f"player_{current_agent_idx}"
            current_player = env.internal_env.game.players[current_agent_idx]
            
            print(f"After PRE-ROLL: Current Player: {current_player.player_name}, Phase: {current_player.phase}")
        
        # 2. Post-roll step: Get the actual action from the agent
        # Get action from the appropriate agent
        agent = agents[current_agent_id]
        action = agent.get_action(obs[current_agent_id])
        
        print(f"POST-ROLL: Agent {current_agent_id} wants to take action {action}")
        
        # Check if post-roll action is valid
        valid_actions_post = env.internal_env.game.get_valid_actions(current_player)
        print(f"Valid Actions for POST-ROLL: {[i for i, valid in enumerate(valid_actions_post) if valid]}")
        
        top_action, sub_action = action
        if top_action < len(valid_actions_post) and valid_actions_post[top_action]:
            print(f"POST-ROLL: Action {top_action} is VALID")
        else:
            print(f"POST-ROLL: Action {top_action} is INVALID!")
            print(f"Valid actions: {valid_actions_post}")
        
        # Step environment with the post-roll action
        action_dict = {current_agent_id: action}
        obs, rewards, terminated, truncated, info = env.step(action_dict)
        
        # Check for errors
        if current_agent_id in info and 'error' in info[current_agent_id]:
            print(f"POST-ROLL ERROR: {info[current_agent_id]['error']}")
            break
        
        if terminated[current_agent_id] or truncated[current_agent_id]:
            print("Game ended")
            break
    
    print("\n=== DEBUG COMPLETE ===")
    env.close()

if __name__ == "__main__":
    debug_evaluation_two_phase() 