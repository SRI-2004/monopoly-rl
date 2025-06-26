# monopoly_env/utils/reward_calculator.py

import numpy as np


class RewardCalculator:
    """
    Computes net worth and rewards for players based on property holdings, cash, and improvements.

    Net Worth Calculation:
      - Hotels:
            For each property with a hotel:
                contribution = (basePrice - mortgageValue) * 3 + hotel_cost
                where hotel_cost is assumed to be 4× the house_cost.
      - Developed (House) Properties:
            For each property with houses (and no hotel):
                number_of_houses = house_fraction / 0.25
                contribution = (basePrice - mortgageValue) * 2 + (house_cost * number_of_houses)
      - Undeveloped Properties:
            If the property's color group is fully owned by the player, multiplier = 1.5;
            otherwise, multiplier = 1.25.
                contribution = (basePrice - mortgageValue) * multiplier
      - Cash Contribution:
            Simply the current cash balance.
      - Cash Penalty:
            If cash < threshold (T), apply penalty = β × (T - cash).

    Reward Components:
      - Dense Reward is computed per step using:
            (r1): Change in net-worth ratio versus opponents.
            (r2): Logarithm of personal net-worth growth.
            r_dense = r1 + r2.
      - Sparse Reward (Terminal):
            +win_bonus if the player won,
            -win_bonus if lost,
            0 otherwise.

    The RewardCalculator stores the previous net worth per player to compute the step‐wise differences.
    """

    def __init__(self, board, players, cash_threshold=200, penalty_beta=1.0, epsilon=1e-6, win_bonus=1000):
        """
        Parameters:
            board: An instance of the Board class.
            players: A list of Player instances.
            cash_threshold (int): The threshold below which a cash penalty is applied.
            penalty_beta (float): The scaling factor for the cash penalty.
            epsilon (float): Small value to avoid division by zero.
            win_bonus (float): The bonus reward for winning.
        """
        self.board = board
        self.players = players
        self.cash_threshold = cash_threshold
        self.penalty_beta = penalty_beta
        self.epsilon = epsilon
        self.win_bonus = win_bonus

        # Initialize previous net worth for each player.
        self.prev_networth = {p.player_name: self.compute_networth(p) for p in players}

    def reset_baseline(self, players):
        self.prev_networth = {p.player_name: self.compute_networth(p) for p in players}

    def compute_networth(self, player):
        """
        Compute a player's net worth based on:
            - Property contributions (hotels, houses, or undeveloped properties)
            - Current cash minus a penalty if cash is low.

        Returns:
            networth (float)
        """
        # Start with the player's cash.
        networth = player.current_cash

        # Extract property metadata into vectorized arrays.
        props = self.board.properties_meta
        num_props = len(props)
        property_ids = np.array([prop["id"] for prop in props])
        base_prices = np.array([prop.get("price", 0) for prop in props], dtype=float)
        mortgage_values = np.array([prop.get("mortgage_value", 0) for prop in props], dtype=float)
        house_costs = np.array([prop.get("house_cost", 0) for prop in props], dtype=float)
        # Color groups as strings (we'll need these for the undeveloped case).
        color_groups = np.array([prop.get("color_group", "") for prop in props])

        # Retrieve improvement info from the board state.
        # Assumes board.state is a NumPy array of shape (num_props, 8),
        # with index 6 for house fraction and index 7 for hotel fraction.
        house_fracs = self.board.state[:, 6]
        hotel_fracs = self.board.state[:, 7]

        # Determine which properties are owned by the player.
        # This produces a boolean array of shape (num_props,).
        owned_mask = np.array([pid in player.assets for pid in property_ids])

        # Initialize contributions vector.
        contributions = np.zeros(num_props)

        if owned_mask.any():
            owned_indices = np.where(owned_mask)[0]

            # --- Hotel Contributions ---
            # For owned properties where a hotel is present (hotel_frac > 0).
            hotel_mask = hotel_fracs[owned_indices] > 0
            if hotel_mask.any():
                idx = owned_indices[hotel_mask]
                # Hotel cost is assumed to be 4× house_cost.
                contributions[idx] = (base_prices[idx] - mortgage_values[idx]) * 3 + (4 * house_costs[idx])

            # --- House Contributions ---
            # For owned properties with houses (house_frac > 0) and no hotel.
            house_mask = (house_fracs[owned_indices] > 0) & (hotel_fracs[owned_indices] == 0)
            if house_mask.any():
                idx = owned_indices[house_mask]
                # Compute the number of houses (house_frac is in increments of 0.25).
                num_houses = np.round(house_fracs[idx] / 0.25)
                contributions[idx] = (base_prices[idx] - mortgage_values[idx]) * 2 + (house_costs[idx] * num_houses)

            # --- Undeveloped Contributions ---
            # For properties with no houses and no hotel.
            undeveloped_mask = (house_fracs[owned_indices] == 0) & (hotel_fracs[owned_indices] == 0)
            if undeveloped_mask.any():
                idx = owned_indices[undeveloped_mask]
                for i in idx:
                    color = color_groups[i]
                    # Get all property ids for this color.
                    all_color_ids = np.array([prop["id"] for prop in props if prop.get("color_group", "") == color])
                    # If the player owns the full color set, use a multiplier of 1.5; otherwise, 1.25.
                    multiplier = 1.5 if (set(all_color_ids).issubset(player.assets)) else 1.25
                    contributions[i] = (base_prices[i] - mortgage_values[i]) * multiplier

        # Add all property contributions to the net worth.
        networth += contributions.sum()

        # Apply cash penalty if cash is below the threshold.
        penalty = self.penalty_beta * (
                    self.cash_threshold - player.current_cash) if player.current_cash < self.cash_threshold else 0
        networth -= penalty

        return networth

    def compute_dense_reward(self, current_player):
        """
        Compute the dense reward for the current step for a given player.

        It calculates:
            (r1): Change in net-worth ratio versus opponents.
            (r2): Logarithm of net worth growth.

        Returns:
            dense_reward (float)
        """
        # Compute current net worth for all players.
        current_networth = {p.player_name: self.compute_networth(p) for p in self.players}

        # For the current player.
        cp_net = current_networth[current_player.player_name]
        # Sum net worth of opponents.
        opponents_net = sum(current_networth[p.player_name] for p in self.players if p != current_player)
        ratio_current = cp_net / max(self.epsilon, opponents_net)

        # Previous net worth values.
        prev_cp_net = self.prev_networth[current_player.player_name]
        opponents_prev_net = sum(self.prev_networth[p.player_name] for p in self.players if p != current_player)
        ratio_prev = prev_cp_net / max(self.epsilon, opponents_prev_net)

        r1 = ratio_current - ratio_prev
        r2 = np.log(cp_net / (prev_cp_net + self.epsilon)) if prev_cp_net > 0 and cp_net > 0 else 0

        # Update previous net worth for the next step.
        self.prev_networth = current_networth

        return r1 + r2

    def compute_sparse_reward(self, current_player):
        """
        Compute a terminal (sparse) reward based on the player's status.

        Returns:
            +win_bonus if the player won,
            -win_bonus if the player lost,
            0 otherwise.
        """
        if current_player.status == "won":
            return self.win_bonus
        elif current_player.status == "lost":
            return -self.win_bonus
        else:
            return 0
