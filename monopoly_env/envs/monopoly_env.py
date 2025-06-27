import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os

from monopoly_env.core import initialize_game
from monopoly_env.utils.reward_calculator import RewardCalculator
from monopoly_env import config


class MonopolyEnv(gym.Env):
    """
    Custom Gym Environment for a simplified Monopoly game.

    Observations:
        Dict({
            "player": Box(shape=(16,), dtype=np.float32)  - The current player's state.
            "board": Box(shape=(224,), dtype=np.float32)   - The flattened board state.
            "current_player_id": Discrete(num_players)
            "pending_trade_valid": Discrete(2)
            "trade_details": Box(shape=(4,), dtype=np.float32) # [from, to, prop, price_norm]
        })

    Actions:
        Tuple consisting of:
          - Top-level action: Discrete(12) (values 0–11, which are adjusted to 1–12 internally)
          - Sub-action parameter: Discrete(252)

        If a particular top-level action does not require a sub-action parameter,
        the agent may simply pass 0.

    Reward:
        For now, rewards are set to 0 by default. You can extend this logic later.

    Episode Termination:
        The episode ends when a player's status becomes "won" or "lost" or when a maximum
        number of steps is reached.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, board_json_path=config.DEFAULT_BOARD_JSON, num_players=config.NUM_PLAYERS, max_steps=config.MAX_STEPS, player_names=config.PLAYER_NAMES, reward_config_path=config.REWARD_CONFIG_PATH):
        super(MonopolyEnv, self).__init__()

        # Set default player names if none are provided.
        if num_players > len(config.PLAYER_NAMES):
            raise ValueError(f"Not enough player names in config for {num_players} players.")
        
        player_names = config.PLAYER_NAMES[:num_players]

        self.max_steps = max_steps
        self.current_step = 0

        # Define observation space: a dict with player and board states.
        self.observation_space = spaces.Dict({
            "player": spaces.Box(low=-np.inf, high=np.inf, shape=(config.PLAYER_DIM,), dtype=np.float32),
            "board": spaces.Box(low=-np.inf, high=np.inf, shape=(config.BOARD_DIM,), dtype=np.float32),
            "current_player_id": spaces.Discrete(num_players),
            "pending_trade_valid": spaces.Discrete(2),
            "trade_details": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32) # [from, to, prop, price_norm]
        })

        # Define action space:
        # Top-level actions: 12 discrete choices.
        # Sub-action: Discrete(252) covers the maximum parameter space needed.
        self.action_space = spaces.MultiDiscrete([config.TOP_LEVEL_ACTIONS, config.SUB_ACTION_DIM])


        # Initialize game logic using our core initializer.
        # This creates the board (from the JSON file) and a list of players.
        self.game = initialize_game(board_json_path, player_names)
        self.players = self.game.players

        # Ensure game logic's current player index is set to 0.
        self.game.current_player_index = 0
        
        # Load the new reward configuration
        with open(reward_config_path) as f:
            reward_config = json.load(f)

        self.reward_calculator = RewardCalculator(
            self.game.board, 
            self.game.players,
            config=reward_config
        )

        # Reset the environment to start a new game.
        self.reset()

    def seed(self, seed=None):
        from gymnasium.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Parameters:
            seed (int, optional): The random seed for this environment.
            options (dict, optional): Additional options (unused here).

        Returns:
            observation (dict): The initial observation.
            info (dict): Additional info (empty dict in this case).
        """
        if seed is not None:
            self.seed(seed)

        self.current_step = 0

        # Reset the board.
        self.game.board.reset()

        # Reset each player's attributes.
        for player in self.players:
            player.current_position = 0
            player.update_status("waiting_for_move")
            player.update_phase("pre-roll")
            player.current_cash = config.STARTING_CASH
            player.assets = set()
            player.full_color_sets_possessed = set()
            player.mortgaged_assets = set()
            player.clear_outstanding_offer()
            player.set_option_to_buy(False)
            player.currently_in_jail = False

        # Clear any pending trades and set the current player to the first.
        self.game.pending_trade = None
        self.game.current_player_index = 0

        self.reward_calculator.reset_baseline(self.game.players)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """
        Execute one step within the environment.

        Parameters:
            action (tuple): A tuple (top_level_action, sub_action).

        Returns:
            observation (dict): The new observation after the action.
            reward (float): The reward obtained from the action.
            terminated (bool): Whether the episode has terminated (win/loss).
            truncated (bool): Whether the episode was truncated (e.g., max_steps reached).
            info (dict): Additional info (e.g., a message from game_logic).
        """
        self.current_step += 1

        # Unpack the action tuple.
        top_level_action, sub_action = action

        # Get the current player.
        current_player = self.players[self.game.current_player_index]

        # --- Action Validation ---
        # 1. Check if the action is within the defined action space bounds.
        if not self.action_space.contains(action):
            return self._get_obs(), -1.0, False, False, {"error": "Action out of bounds."}

        # Retrieve the valid top-level actions mask for the current player.
        valid_actions = self.game.get_valid_actions(current_player)

        # 2. Check if the chosen top-level action is logically valid in the current game state.
        if not valid_actions[top_level_action]:
            info = {"error": "Invalid top-level action selected according to valid actions mask."}
            return self._get_obs(), -1.0, False, False, info

        # 3. For top-level actions that require sub-actions, check the sub-action validity.
        actions_with_sub = {0, 1, 2, 3, 4, 5, 10, 11}
        if top_level_action in actions_with_sub:
            valid_sub_mask = self.game.get_valid_subactions(current_player, top_level_action)
            if sub_action < 0 or sub_action >= len(valid_sub_mask) or not valid_sub_mask[sub_action]:
                info = {"error": "Invalid sub-action selected according to valid sub-actions mask."}
                return self._get_obs(), -1.0, False, False, info

        # Adjust the top-level action to match game_logic expectations (1-indexed).
        adjusted_action = top_level_action + 1

        # Get state before action to calculate houses built
        old_house_frac_sum = np.sum(self.game.board.state[:, 6])

        try:
            # Process the action using the game logic.
            result_msg = self.game.process_action(adjusted_action, sub_action)
            # Immediately handle bankruptcy after an action, as cash levels might have changed.
            self._handle_bankruptcy()
        except Exception as e:
            result_msg = f"Error: {str(e)}"
            reward = -1.0
            terminated = False
            truncated = False
            info = {"error": str(e)}
            return self._get_obs(), reward, terminated, truncated, info

        # --- New Reward Calculation Logic ---
        # Calculate houses built by comparing board state
        new_house_frac_sum = np.sum(self.game.board.state[:, 6])
        houses_built = int(round((new_house_frac_sum - old_house_frac_sum) / 0.25))

        # Create action_info dict for the reward calculator
        action_info = {
            'purchased_property': adjusted_action == 2, # Assuming action 2 is "purchase"
            'houses_built': houses_built,
            'paid_jail_fine': adjusted_action == 6, # Assuming action 6 is "pay jail fine"
        }

        # Compute dense reward for the current player using the new logic
        dense_reward = self.reward_calculator.compute_dense_reward(current_player, action_info)

        # Determine if the episode is finished naturally
        terminated = any(p.status == "won" for p in self.players)
        truncated = False

        # Handle truncation tie-breaker
        if not terminated and self.current_step >= self.max_steps:
            truncated = True
            terminated = True # In Gym, truncation also signals the end of an episode
            
            # Find winner by net worth
            net_worths = {p.player_name: self.reward_calculator.compute_networth(p) for p in self.players}
            if net_worths:
                winner_name = max(net_worths, key=net_worths.get)
                for p in self.players:
                    if p.player_name == winner_name:
                        p.status = "won" # Set winner status for sparse reward calculation
        
        # If the episode is done (by win or truncation), compute the sparse (terminal) reward.
        sparse_reward = self.reward_calculator.compute_sparse_reward(current_player) if terminated else 0.0

        # The final reward for this step
        reward = dense_reward + sparse_reward

        # The info dict should contain all relevant data for analytics
        net_worths_for_info = {p.player_name: self.reward_calculator.compute_networth(p) for p in self.players}
        info = {
            "result": result_msg, 
            "step": self.current_step,
            "behavioral_metrics": self.reward_calculator.get_and_reset_behavioral_metrics(),
            "net_worths": net_worths_for_info,
            "status": current_player.status,
            'semantic_features': {p.player_name: {'net_worth': w} for p, w in zip(self.players, net_worths_for_info.values())}
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """
        Construct the observation from the current game state.

        Returns:
            A dict containing the current player's state vector and the board state.
        """
        # Current player's state (16 dims).
        current_player = self.players[self.game.current_player_index]
        player_state = current_player.get_state_vector()

        # Board state (flattened to 224 dims).
        board_state = self.game.board.get_state(flatten=True)
        
        # Trade details
        trade = self.game.pending_trade
        pending_trade_valid = 1 if trade else 0
        trade_details = np.zeros(4, dtype=np.float32)
        if trade:
            # Normalize trade details to be [0, 1]
            from_player_idx = self.game.players.index(trade['from_player'])
            to_player_idx = self.game.players.index(trade['to_player'])
            prop_idx = trade['property_index']
            price = trade['offer_price']
            
            trade_details[0] = from_player_idx / (len(self.players) -1) if len(self.players) > 1 else 0
            trade_details[1] = to_player_idx / (len(self.players) -1) if len(self.players) > 1 else 0
            trade_details[2] = prop_idx / (self.game.board.num_properties - 1) if self.game.board.num_properties > 1 else 0
            # Normalize price by max possible cash, assume it's starting cash * 2 for simplicity
            max_price = config.STARTING_CASH * 2 
            trade_details[3] = price / max_price

        return {
            "player": player_state, 
            "board": board_state,
            "current_player_id": self.game.current_player_index,
            "pending_trade_valid": pending_trade_valid,
            "trade_details": trade_details
        }

    def _handle_bankruptcy(self):
        """
        Check for and handle player bankruptcy.
        A player is bankrupt if their cash is negative and they have no assets to sell/mortgage.
        This method is now handled by the core game logic.
        """
        pass

    def render(self, mode="human"):
        """
        Render the current state of the game.

        For this simple implementation, print out key information about the current player
        and a brief summary of the board state.
        """
        current_player = self.players[self.game.current_player_index]
        print("----- Current Game State -----")
        print(f"Player: {current_player.player_name}")
        print(f"Position: {current_player.current_position}")
        print(f"Cash: {current_player.current_cash}")
        print(f"Phase: {current_player.phase}")
        board_state = self.game.board.get_state(flatten=True)
        print("Board State (first 16 values):", board_state[:16], "...")
        print("-------------------------------")

    def close(self):
        """
        Clean up the environment (if needed).
        """
        pass
