#!/usr/bin/env python3

import numpy as np
import random

class TestAgent:
    """
    A comprehensive test agent for Monopoly that systematically tries all available actions.
    
    This agent follows a priority system to test different actions:
    1. Handle jail situations (actions 8, 9)
    2. Respond to trades (action 11)
    3. Try property management actions (2, 3, 4, 5) when possible
    4. Try trade offers (0, 1) when possible
    5. Buy properties when available (action 10)
    6. Skip or conclude when no other actions are available (6, 7)
    
    The agent keeps track of which actions it has tried to avoid repetition unless necessary.
    """
    
    def __init__(self, player_id, num_players, board_meta):
        self.player_id = player_id
        self.num_players = num_players
        self.board_meta = board_meta
        
        # Track which actions we've tried to avoid repetition
        self.tried_actions = set()
        self.action_priority = [
            8,   # Use Get Out of Jail
            9,   # Pay Jail Fine
            11,  # Respond to Trade
            2,   # Improve Property
            3,   # Sell House/Hotel
            4,   # Sell Property
            5,   # Mortgage/Free Mortgage
            0,   # Make Trade Offer (Sell)
            1,   # Make Trade Offer (Buy)
            10,  # Buy Property
            6,   # Skip Turn
            7,   # Conclude Phase
        ]
        
        # Track sub-actions we've tried for each top-level action
        self.tried_sub_actions = {i: set() for i in range(12)}
        
        print(f"TestAgent initialized for player {player_id}")
    
    def get_action(self, obs):
        """
        Get an action based on systematic testing of all available actions.
        
        Args:
            obs (dict): The observation for this agent.
            
        Returns:
            tuple: A tuple representing the (top_level_action, sub_action).
        """
        player_state = obs['player']
        
        # Decode player state
        is_in_jail = player_state[10] > 0
        can_buy = player_state[12] > 0
        current_cash = player_state[7]
        current_pos = int(player_state[0])
        
        phase_vec = player_state[13:16]
        phase = np.argmax(phase_vec)  # 0: pre-roll, 1: post-roll, 2: out-of-turn
        
        print(f"TestAgent decision - Phase: {['pre-roll', 'post-roll', 'out-of-turn'][phase]}, "
              f"Position: {current_pos}, Cash: {current_cash}, In Jail: {is_in_jail}, Can Buy: {can_buy}")
        
        # Priority 1: Handle jail situations (mandatory)
        if is_in_jail:
            if 9 not in self.tried_actions and current_cash > 50:
                self.tried_actions.add(9)
                print("TestAgent: Trying Pay Jail Fine (action 9)")
                return (9, 0)
            elif 8 not in self.tried_actions:
                self.tried_actions.add(8)
                print("TestAgent: Trying Use Get Out of Jail (action 8)")
                return (8, 0)
            else:
                # We've tried both jail actions, pick the better one
                if current_cash > 50:
                    print("TestAgent: Repeating Pay Jail Fine (action 9)")
                    return (9, 0)
                else:
                    print("TestAgent: Repeating Use Get Out of Jail (action 8)")
                    return (8, 0)
        
        # Priority 2: Respond to trades (if pending)
        if phase == 2:  # out-of-turn phase might have trades
            if 11 not in self.tried_actions:
                self.tried_actions.add(11)
                # Try both accept and reject
                if 0 not in self.tried_sub_actions[11]:
                    self.tried_sub_actions[11].add(0)
                    print("TestAgent: Trying Respond to Trade - Reject (action 11, sub 0)")
                    return (11, 0)
                elif 1 not in self.tried_sub_actions[11]:
                    self.tried_sub_actions[11].add(1)
                    print("TestAgent: Trying Respond to Trade - Accept (action 11, sub 1)")
                    return (11, 1)
        
        # Priority 3: Property management actions (when we have properties)
        # First check what valid actions are available
        valid_actions = obs.get('valid_actions', [True] * 12)  # Fallback if not available
        property_management_actions = [2, 3, 4, 5]  # Improve, Sell Houses, Sell Property, Mortgage
        for action in property_management_actions:
            if action not in self.tried_actions and action < len(valid_actions) and valid_actions[action]:
                # Try to find a valid sub-action for this action
                sub_action = self._find_untried_sub_action(action, player_state)
                if sub_action is not None:
                    self.tried_actions.add(action)
                    self.tried_sub_actions[action].add(sub_action)
                    action_names = {2: "Improve Property", 3: "Sell House/Hotel", 4: "Sell Property", 5: "Mortgage/Free Mortgage"}
                    print(f"TestAgent: Trying {action_names[action]} (action {action}, sub {sub_action})")
                    return (action, sub_action)
        
        # Priority 4: Trade offers (when in pre-roll or post-roll)
        if phase in [0, 1]:  # pre-roll or post-roll
            trade_actions = [0, 1]  # Make Trade Offer (Sell), Make Trade Offer (Buy)
            for action in trade_actions:
                if action not in self.tried_actions:
                    # Try a simple trade offer
                    sub_action = self._find_untried_sub_action(action, player_state)
                    if sub_action is not None:
                        self.tried_actions.add(action)
                        self.tried_sub_actions[action].add(sub_action)
                        action_names = {0: "Make Trade Offer (Sell)", 1: "Make Trade Offer (Buy)"}
                        print(f"TestAgent: Trying {action_names[action]} (action {action}, sub {sub_action})")
                        return (action, sub_action)
        
        # Priority 5: Buy property (when available)
        if can_buy and phase == 1:  # post-roll phase
            if 10 not in self.tried_actions:
                property_meta = self.board_meta.get(str(current_pos))
                if property_meta and 'price' in property_meta and current_cash >= property_meta['price']:
                    self.tried_actions.add(10)
                    self.tried_sub_actions[10].add(1)
                    print("TestAgent: Trying Buy Property - Yes (action 10, sub 1)")
                    return (10, 1)
                elif 0 not in self.tried_sub_actions[10]:
                    self.tried_actions.add(10)
                    self.tried_sub_actions[10].add(0)
                    print("TestAgent: Trying Buy Property - No (action 10, sub 0)")
                    return (10, 0)
        
        # Priority 6: Skip turn (test action)
        if 6 not in self.tried_actions:
            self.tried_actions.add(6)
            print("TestAgent: Trying Skip Turn (action 6)")
            return (6, 0)
        
        # Priority 7: Conclude phase (default action)
        print("TestAgent: Using default Conclude Phase (action 7)")
        return (7, 0)
    
    def _find_untried_sub_action(self, action, player_state):
        """
        Find an untried sub-action for the given top-level action.
        
        Returns:
            int or None: A valid sub-action index, or None if none available.
        """
        # Based on the codebase analysis, here are the correct sub-action ranges:
        # 0: Make Trade Offer (Sell) -> 252 = 3 players x 28 properties x 3 price tiers
        # 1: Make Trade Offer (Buy)  -> 252 = 3 players x 28 properties x 3 price tiers
        # 2: Improve Property        -> 44  = 22 properties x 2 building types (house/hotel)
        # 3: Sell House/Hotel        -> 44  = 22 properties x 2 building types
        # 4: Sell Property           -> 28  = one-hot over 28 properties
        # 5: Mortgage/Free Mortgage  -> 28  = one-hot over 28 properties
        # 10: Buy Property           -> 2   (0: decline, 1: buy)
        # 11: Respond to Trade       -> 2   (0: Reject, 1: Accept)
        
        if action in [0, 1]:  # Trade offers (252 sub-actions)
            # Try different combinations of player, property, and price
            for sub in [0, 1, 2, 28, 56, 84, 112, 140, 168, 196, 224, 251]:  # Sample across the range
                if sub not in self.tried_sub_actions[action]:
                    return sub
        
        elif action in [2, 3]:  # Improve/Sell houses (44 sub-actions)
            # Try different properties and building types
            for sub in [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 43]:  # Sample across the range
                if sub not in self.tried_sub_actions[action]:
                    return sub
        
        elif action in [4, 5]:  # Sell property, Mortgage (28 sub-actions)
            # Try different properties
            for sub in range(min(28, 15)):  # Try first 15 properties
                if sub not in self.tried_sub_actions[action]:
                    return sub
        
        elif action == 10:  # Buy Property (2 sub-actions)
            for sub in [0, 1]:
                if sub not in self.tried_sub_actions[action]:
                    return sub
        
        elif action == 11:  # Respond to Trade (2 sub-actions)
            for sub in [0, 1]:
                if sub not in self.tried_sub_actions[action]:
                    return sub
        
        return None
    
    def reset_tried_actions(self):
        """Reset the tried actions for a new game or testing session."""
        self.tried_actions.clear()
        self.tried_sub_actions = {i: set() for i in range(12)}
        print(f"TestAgent: Reset tried actions for player {self.player_id}")


def test_all_actions():
    """
    Test function to run the TestAgent and see what actions it tries.
    """
    import json
    import sys
    sys.path.append('.')
    from env_wrapper import MonopolyMAv2
    
    print("=== TESTING ALL ACTIONS WITH TEST AGENT ===")
    
    # Create environment
    env = MonopolyMAv2(board_json_path="../data.json", num_players=4, max_steps=200)
    agent_ids = env.possible_agents
    
    # Load board metadata
    with open('../data.json', 'r') as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}
    
    # Create test agents
    agents = {}
    for agent_id in agent_ids:
        agents[agent_id] = TestAgent(int(agent_id.split('_')[-1]), 4, board_meta)
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    step_count = 0
    actions_tested = set()
    
    while step_count < 100:  # Test for up to 100 steps
        step_count += 1
        
        # Get the current player
        current_agent_idx = env.internal_env.game.current_player_index
        current_agent_id = f"player_{current_agent_idx}"
        current_player = env.internal_env.game.players[current_agent_idx]
        
        print(f"\n--- Step {step_count} ---")
        print(f"Current Player: {current_player.player_name}, Phase: {current_player.phase}")
        
        # Get valid actions
        valid_actions = env.internal_env.game.get_valid_actions(current_player)
        print(f"Valid actions: {[i for i, valid in enumerate(valid_actions) if valid]}")
        
        # Two-phase system like in evaluation
        if current_player.phase == 'pre-roll':
            pre_roll_action = (7, 0)
            pre_roll_action_dict = {current_agent_id: pre_roll_action}
            obs, rewards, terminated, truncated, info = env.step(pre_roll_action_dict)
            
            if terminated[current_agent_id] or truncated[current_agent_id]:
                break
                
            current_agent_idx = env.internal_env.game.current_player_index
            current_agent_id = f"player_{current_agent_idx}"
            current_player = env.internal_env.game.players[current_agent_idx]
        
        # Get action from test agent
        agent = agents[current_agent_id]
        action = agent.get_action(obs[current_agent_id])
        actions_tested.add(action[0])
        
        # Check if action is valid
        valid_actions_post = env.internal_env.game.get_valid_actions(current_player)
        top_action, sub_action = action
        
        if top_action < len(valid_actions_post) and valid_actions_post[top_action]:
            print(f"✓ Action {action} is VALID")
        else:
            print(f"✗ Action {action} is INVALID!")
            print(f"Valid actions: {[i for i, valid in enumerate(valid_actions_post) if valid]}")
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
    
    print(f"\n=== TEST COMPLETE ===")
    print(f"Actions tested: {sorted(actions_tested)}")
    print(f"Total unique actions tested: {len(actions_tested)}")
    
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
    
    print("\nActions tested:")
    for action in sorted(actions_tested):
        print(f"  {action}: {action_names.get(action, 'Unknown')}")
    
    print("\nActions NOT tested:")
    for action in range(12):
        if action not in actions_tested:
            print(f"  {action}: {action_names.get(action, 'Unknown')}")
    
    env.close()


if __name__ == "__main__":
    test_all_actions() 