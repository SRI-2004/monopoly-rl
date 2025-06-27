import numpy as np

class ScriptedAgent:
    """
    A simple scripted agent for Monopoly.

    This agent follows a basic set of rules:
    - If it lands on an unowned property, it buys it if it can afford it.
    - If in jail, it pays the fine to get out immediately.
    - Otherwise, it concludes its turn without taking optional actions like
      building houses or trading.
    """
    def __init__(self, player_id, num_players, board_meta):
        self.player_id = player_id
        self.num_players = num_players
        self.board_meta = board_meta # Dict of property metadata

    def get_action(self, obs):
        """
        Get an action based on the current observation.

        Args:
            obs (dict): The observation for this agent.

        Returns:
            tuple: A tuple representing the (top_level_action, sub_action).
        """
        player_state = obs['player'] # This is the raw vector
        
        # We need to decode the player state vector to make decisions.
        # Based on monopoly_env/core/player.py:
        # 0: pos, 1-4: status, 5-6: jail cards, 7: cash, 8: RRs, 9: utils, 10: in_jail, 12: can_buy, 13-15: phase
        
        is_in_jail = player_state[10] > 0
        can_buy = player_state[12] > 0
        current_cash = player_state[7]
        current_pos = int(player_state[0])
        
        phase_vec = player_state[13:16]
        phase = np.argmax(phase_vec) # 0: pre-roll, 1: post-roll, 2: out-of-turn
        
        # Rule 1: Get out of jail if possible
        if is_in_jail and phase == 0: # Pre-roll phase
            # Action 9 is "Pay Jail Fine"
            if current_cash > 50: # Assuming fine is 50
                return (9, 0)

        # Rule 2: Buy property if the option is available
        if can_buy and phase == 1: # Post-roll phase
            property_meta = self.board_meta.get(str(current_pos))
            if property_meta and current_cash >= property_meta['price']:
                # Action 10 is "Buy Property", Sub-action 1 is "Yes"
                return (10, 1)
            else:
                # Sub-action 0 is "No"
                return (10, 0)
        
        # Default action: Conclude the current phase/turn
        # Action 6 is "Conclude Phase"
        return (6, 0) 