"""
Configuration file for the Monopoly RL Environment.

This file centralizes all constants and hyperparameters used across the project,
such as observation dimensions, action dimensions, reward parameters, and game settings.
"""

# -------------------------
# Observation & Action Dimensions
# -------------------------
PLAYER_DIM = 16                   # Dimensionality of the player's state
BOARD_DIM = 224                   # Dimensionality of the board's state
OBSERVATION_DIM = PLAYER_DIM + BOARD_DIM  # Total observation size (240)

TOP_LEVEL_ACTIONS = 12            # Number of top-level actions (1-12)
SUB_ACTION_DIM = 252              # Maximum sub-action parameter space size

# -------------------------
# Board and Property Settings
# -------------------------
NUM_PROPERTIES = 28               # Total number of properties on the board
PROPERTY_DIM = 8                  # Dimensions per property (owner, mortgaged flag, monopoly flag, house/hotel counts)

DEFAULT_BOARD_JSON = "data.json"  # Default path to the board data JSON file

# -------------------------
# Player Settings
# -------------------------
NUM_PLAYERS = 4                 # Default number of players
STARTING_CASH = 1500            # Starting cash for each player
PLAYER_NAMES = ["Alice", "Bob", "Charlie", "Diana"] # Default player names

# -------------------------
# Reward Parameters
# -------------------------
# For computing cash penalty if cash falls below threshold
CASH_THRESHOLD = 200            # Threshold for low cash penalty
PENALTY_BETA = 1.0              # Scaling factor for the penalty

EPSILON = 1e-6                  # Small constant to avoid division by zero in reward calculations
WIN_BONUS = 1000                # Bonus reward for winning (terminal reward)

# -------------------------
# Game Phase Labels (for clarity)
# -------------------------
PHASES = {
    "pre_roll": "pre-roll",
    "post_roll": "post-roll",
    "out_of_turn": "out-of-turn"
}

# -------------------------
# Additional Settings
# -------------------------
MAX_STEPS = 1000                # Maximum number of steps per episode (can be overridden in the envs)
