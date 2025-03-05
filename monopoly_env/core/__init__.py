"""
Core Package for Monopoly Environment

This package contains the main components for the game:
    - Board: Manages the board state (224 dimensions).
    - Player: Manages individual player state (16 dimensions).
    - GameLogic: Implements the core game mechanics (dice, property transactions, phases, etc.).

The helper function 'initialize_game' creates a default game instance with a board and players.
"""

from .board import Board
from .player import Player
from .game_logic import GameLogic

__all__ = ["Board", "Player", "GameLogic", "initialize_game"]


def initialize_game(board_json_path, player_names, starting_cash=1500):
    """
    Initialize a new game instance.

    This function creates Player objects from the provided player names,
    initializes the Board from the JSON file at board_json_path, and returns
    a GameLogic instance that ties everything together.

    Parameters:
        board_json_path (str): Path to the JSON file containing board data.
        player_names (list of str): List of names for each player.
        starting_cash (int): The starting cash balance for each player (default: 1500).

    Returns:
        GameLogic: An initialized game logic instance.
    """
    players = []
    for name in player_names:
        # Create a new Player with the provided name and starting cash.
        players.append(Player(player_name=name, current_cash=starting_cash))

    # Create and return the GameLogic instance using the board JSON and list of players.
    return GameLogic(board_json_path, players)
