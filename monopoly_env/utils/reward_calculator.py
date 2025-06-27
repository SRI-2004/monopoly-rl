# monopoly_env/utils/reward_calculator.py

import numpy as np


class RewardCalculator:
    """
    Computes net worth and rewards for players based on a re-engineered reward structure.
    This structure is designed to provide a much richer signal to the RL agent by
    valuing assets based on their potential future income (Present Value of rent)
    and providing shaping bonuses for desirable actions like buying and building.
    """

    def __init__(self, board, players, config, epsilon=1e-6):
        """
        Parameters:
            board: An instance of the Board class.
            players: A list of Player instances.
            config (dict): A dictionary loaded from reward_config.json.
            epsilon (float): Small value to avoid division by zero.
        """
        self.board = board
        self.players = players
        self.config = config
        self.epsilon = epsilon

        # Calculate lambda (rent PV factor) from gamma in the config
        self.lambda_rent_pv = 1 / (40 * (1 - self.config['gamma']))
        
        # This is the terminal reward for winning or losing.
        self.win_bonus = self.config['win_bonus']

        # Tweak: Normalize time penalty by number of players for consistent per-round penalty
        self.time_penalty_per_step = self.config['time_penalty'] / len(self.players)

        # Initialize previous net worth for each player.
        self.prev_networth = {p.player_name: self.compute_networth(p) for p in players}
        
        # --- Behavioral Metrics Tracking ---
        self.behavioral_metrics = {
            "properties_purchased": 0,
            "houses_built": 0,
            "jail_fines_paid": 0,
        }

    def reset_baseline(self, players):
        self.prev_networth = {p.player_name: self.compute_networth(p) for p in players}
        self.behavioral_metrics = {
            "properties_purchased": 0,
            "houses_built": 0,
            "jail_fines_paid": 0,
        }

    def get_and_reset_behavioral_metrics(self):
        """Returns the current behavioral metrics and resets them."""
        metrics = self.behavioral_metrics.copy()
        self.behavioral_metrics = {
            "properties_purchased": 0,
            "houses_built": 0,
            "jail_fines_paid": 0,
        }
        return metrics

    def compute_networth(self, player):
        """
        Compute a player's net worth based on the re-engineered valuation rules.
        """
        networth = player.current_cash
        networth -= self.config['cash_penalty_beta'] * max(0, self.config['cash_threshold'] - player.current_cash)

        # Tweak: Pre-calculate non-street asset values outside the loop to prevent errors.
        owned_property_ids = [pid for pid in player.assets if pid not in player.mortgaged_assets]
        
        # --- Railroad Valuation ---
        railroad_ids = [pid for pid in owned_property_ids if self.board.properties_meta_by_id[pid]['type'] == 'railroad']
        num_railroads_owned = len(railroad_ids)
        if num_railroads_owned > 0:
            base_rr_price = self.board.properties_meta_by_id[railroad_ids[0]].get('price', 200)
            # Rent is 25 * 2^(n-1). This is the PV of all future rents from the set.
            rent_pv = self.lambda_rent_pv * (25 * (2**(num_railroads_owned - 1)))
            networth += base_rr_price * num_railroads_owned + rent_pv

        # --- Utility Valuation ---
        utility_ids = [pid for pid in owned_property_ids if self.board.properties_meta_by_id[pid]['type'] == 'utility']
        num_utilities_owned = len(utility_ids)
        if num_utilities_owned > 0:
            base_util_price = self.board.properties_meta_by_id[utility_ids[0]].get('price', 150)
            # Tweak: Use correct 4x/10x multiplier.
            rent_multiplier = 10 if num_utilities_owned == 2 else 4
            rent_pv = self.lambda_rent_pv * (self.config['avg_dice_roll'] * rent_multiplier)
            networth += base_util_price * num_utilities_owned + rent_pv

        for prop_id in owned_property_ids:
            prop_meta = self.board.properties_meta_by_id[prop_id]
            if prop_meta['type'] != 'street':
                continue # Skip non-street properties as they are handled above

            prop_state_idx = self.board.property_id_to_state_idx[prop_id]
            base_price = prop_meta.get('price', 0)
            house_frac = self.board.state[prop_state_idx, 6]
            hotel_frac = self.board.state[prop_state_idx, 7]

            if hotel_frac > 0:
                rent = prop_meta.get('rent_with_hotel', 0)
                # The hotel contribution factor is a small extra bump to reflect liquidity locked in assets.
                hotel_cost = prop_meta.get('house_cost', 0) * 5
                networth += base_price + self.lambda_rent_pv * rent + self.config['hotel_contribution_factor'] * hotel_cost
            elif house_frac > 0:
                num_houses = int(round(house_frac / 0.25))
                # Fix: Look up rent from the rent schedule dict using the number of houses
                rent = prop_meta.get('rent', {}).get(str(num_houses), 0)
                house_cost = prop_meta.get('house_cost', 0)
                networth += base_price + self.lambda_rent_pv * rent + self.config['house_contribution_factor'] * num_houses * house_cost
            else: # Undeveloped
                # Fix: Look up the base rent from the rent schedule dict
                rent = prop_meta.get('rent', {}).get('0', 0)
                value = base_price + self.lambda_rent_pv * rent
                
                # Tweak: Fix monopoly detection to use correct integer player IDs from board state.
                color_group = prop_meta['color_group']
                color_group_prop_ids = [p['id'] for p in self.board.properties_meta if p.get('color_group') == color_group]
                
                owner_ids = set()
                is_monopoly_candidate = True
                for c_prop_id in color_group_prop_ids:
                    c_prop_idx = self.board.property_id_to_state_idx.get(c_prop_id)
                    if c_prop_idx is None: continue
                    
                    owner_vec = self.board.state[c_prop_idx, 0:self.board.num_owners]
                    owner_id = np.argmax(owner_vec)
                    
                    if owner_id == 0: # Bank owned
                        is_monopoly_candidate = False
                        break
                    owner_ids.add(owner_id)
                
                # Check if the set has one unique owner and that owner is the current player
                if is_monopoly_candidate and len(owner_ids) == 1 and player.player_id in owner_ids:
                    value *= 2
                networth += value

        return networth

    def compute_dense_reward(self, current_player, action_info=None):
        """
        Compute the dense reward for the current step based on the new structure.
        """
        if action_info is None:
            action_info = {}

        current_networth = {p.player_name: self.compute_networth(p) for p in self.players}

        # 1. Net-worth ratio term
        cp_net = current_networth[current_player.player_name]
        opponents_net = sum(current_networth[p.player_name] for p in self.players if p != current_player)
        ratio_current = cp_net / max(self.epsilon, opponents_net)

        prev_cp_net = self.prev_networth.get(current_player.player_name, cp_net)
        opponents_prev_net = sum(self.prev_networth.get(p.player_name, current_networth[p.player_name]) for p in self.players if p != current_player)
        ratio_prev = prev_cp_net / max(self.epsilon, opponents_prev_net)
        networth_ratio_term = ratio_current - ratio_prev
        
        # Tweak: Clip the ratio term to prevent instability from bankruptcy spikes.
        networth_ratio_term = np.clip(networth_ratio_term, -self.config['ratio_clip'], self.config['ratio_clip'])

        # 2. Log growth term
        log_growth_term = np.log(cp_net / (prev_cp_net + self.epsilon)) if prev_cp_net > 0 and cp_net > 0 else 0.0
        log_growth_term = np.clip(log_growth_term, -self.config['log_growth_clip'], self.config['log_growth_clip'])

        # 3. Buy bonus
        buy_bonus = self.config['buy_bonus'] if action_info.get('purchased_property', False) else 0.0
        if buy_bonus > 0:
            self.behavioral_metrics['properties_purchased'] += 1
            
        # 4. Build bonus
        houses_built = action_info.get('houses_built', 0)
        build_bonus = self.config['build_bonus_per_house'] * houses_built
        if houses_built > 0:
            self.behavioral_metrics['houses_built'] += houses_built

        # Track jail fines paid for analytics
        if action_info.get('paid_jail_fine', False):
            self.behavioral_metrics['jail_fines_paid'] += 1

        # 5. Time penalty (now using normalized value)
        time_penalty = self.time_penalty_per_step

        # Update previous net worth for the next step.
        self.prev_networth = current_networth

        return networth_ratio_term + log_growth_term + buy_bonus + build_bonus - time_penalty

    def compute_sparse_reward(self, current_player):
        """
        Compute a terminal (sparse) reward based on the player's status.
        Uses the 'win_bonus' from the config file.
        """
        if current_player.status == "won":
            return self.win_bonus
        elif current_player.status == "lost":
            return -self.win_bonus
        else:
            return 0
