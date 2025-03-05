import gymnasium as gym
from gymnasium import spaces
import numpy as np

from monopoly_env.core import initialize_game
from monopoly_env.utils.reward_calculator import RewardCalculator


class MonopolyEnv(gym.Env):
    """
    Custom Gym Environment for a simplified Monopoly game.

    Observations:
        Dict({
            "player": Box(shape=(16,), dtype=np.float32)  - The current player's state.
            "board": Box(shape=(224,), dtype=np.float32)   - The flattened board state.
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
        The episode ends when a player’s status becomes "won" or "lost" or when a maximum
        number of steps is reached.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, board_json_path="/home/srinivasan/PycharmProjects/monopoly-rl/data.json", player_names=None, max_steps=1000):
        super(MonopolyEnv, self).__init__()

        # Set default player names if none are provided.
        if player_names is None:
            player_names = ["Bob", "Charlie", "Diana"]

        self.max_steps = max_steps
        self.current_step = 0

        # Define observation space: a dict with player and board states.
        self.observation_space = spaces.Dict({
            "player": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32),
            "board": spaces.Box(low=-np.inf, high=np.inf, shape=(224,), dtype=np.float32)
        })

        # Define action space:
        # Top-level actions: 12 discrete choices.
        # Sub-action: Discrete(252) covers the maximum parameter space needed.
        self.action_space = spaces.MultiDiscrete([12, 252])


        # Initialize game logic using our core initializer.
        # This creates the board (from the JSON file) and a list of players.
        self.game = initialize_game(board_json_path, player_names)
        self.players = self.game.players

        # Ensure game logic's current player index is set to 0.
        self.game.current_player_index = 0
        self.reward_calculator = RewardCalculator(self.game.board, self.game.players)

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
            player.current_cash = 1500
            player.assets = set()
            player.full_color_sets_possessed = set()
            player.mortgaged_assets = set()
            player.clear_outstanding_offer()
            player.set_option_to_buy(False)
            player.currently_in_jail = False

        # Clear any pending trades and set the current player to the first.
        self.game.pending_trade = None
        self.game.current_player_index = 0

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

        # Retrieve the valid top-level actions mask for the current player.
        valid_actions = self.game.get_valid_actions(current_player)

        # Check if the chosen top-level action is valid.
        if not valid_actions[top_level_action]:
            result_msg = (f"Action {top_level_action + 1} is not valid in the current phase "
                          f"for {current_player.player_name}.")
            reward = -1.0  # Negative reward for choosing an invalid top-level action.
            terminated = False
            truncated = False
            info = {"error": "Invalid top-level action selected according to valid actions mask."}
            return self._get_obs(), reward, terminated, truncated, info

        # For top-level actions that require sub-actions, check the sub-action validity.
        actions_with_sub = {0, 1, 2, 3, 4, 5, 10, 11}
        if top_level_action in actions_with_sub:
            valid_sub_mask = self.game.get_valid_subactions(current_player, top_level_action)
            # Check that the provided sub_action index is within range.
            if sub_action < 0 or sub_action >= len(valid_sub_mask):
                result_msg = (f"Sub-action index {sub_action} out of range for top-level action "
                              f"{top_level_action + 1}.")
                reward = -1.0  # Negative reward for out-of-range sub-action.
                terminated = False
                truncated = False
                info = {"error": "Sub-action index out of range."}
                return self._get_obs(), reward, terminated, truncated, info
            # Check if the selected sub-action is valid.
            if not valid_sub_mask[sub_action]:
                result_msg = (f"Sub-action {sub_action} is not valid for top-level action "
                              f"{top_level_action + 1} for {current_player.player_name}.")
                reward = -1.0  # Negative reward for an invalid sub-action.
                terminated = False
                truncated = False
                info = {"error": "Invalid sub-action selected according to valid sub-actions mask."}
                return self._get_obs(), reward, terminated, truncated, info

        # Adjust the top-level action to match game_logic expectations (1-indexed).
        adjusted_action = top_level_action + 1

        try:
            # Process the action using the game logic.
            result_msg = self.game.process_action(adjusted_action, sub_action)
        except Exception as e:
            result_msg = f"Error: {str(e)}"
            reward = -1.0
            terminated = False
            truncated = False
            info = {"error": str(e)}
            return self._get_obs(), reward, terminated, truncated, info

        # Compute dense reward for the current player.
        dense_reward = self.reward_calculator.compute_dense_reward(current_player)

        # Determine if the episode is finished.
        terminated = False
        for player in self.players:
            if player.status in ["won", "lost"]:
                terminated = True
                break

        # Check if max steps reached (we mark as truncated if so).
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            terminated = True

        # If the episode is done, compute the sparse (terminal) reward.
        sparse_reward = self.reward_calculator.compute_sparse_reward(current_player) if terminated else 0.0

        # Combined reward is the sum of the dense and sparse rewards plus a valid action bonus.
        reward = dense_reward + sparse_reward + 1.0  # +1 bonus for taking a valid action.

        info = {"result": result_msg, "step": self.current_step}
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

        return {"player": player_state, "board": board_state}

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
