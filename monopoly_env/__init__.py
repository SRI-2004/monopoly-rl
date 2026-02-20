"""
Monopoly Environment Package

This package provides a comprehensive Monopoly game environment for reinforcement learning research.
It includes the core game logic, environment wrappers, and utilities for training RL agents.

Main Components:
- core: Core game logic including Board, Player, and GameLogic
- envs: Environment wrappers for different RL frameworks
- utils: Utility functions and reward calculators
- config: Configuration constants and settings
"""

from . import config
from . import core
from . import envs
from . import utils

__version__ = "1.0.0"
__author__ = "Monopoly RL Team"
__all__ = ["config", "core", "envs", "utils"] 