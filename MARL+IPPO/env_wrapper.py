import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
from pettingzoo import ParallelEnv
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monopoly_env.envs.monopoly_env import MonopolyEnv
from monopoly_env import config

def preprocess_obs(obs, num_players, agent_id_int):
    """
    Flattens and preprocesses the observation dictionary from the environment.
    Crucially, it adds a one-hot vector for the agent's own ID, so a shared
    policy can distinguish "self" from "other".
    
    Returns a NumPy array.
    """
    player_obs = obs['player']
    board_obs = obs['board']
    trade_details = obs['trade_details']
    pending_trade = np.array([obs['pending_trade_valid']], dtype=np.float32)

    # One-hot encoding for the agent's own ID
    agent_id_one_hot = np.zeros(num_players, dtype=np.float32)
    agent_id_one_hot[agent_id_int] = 1.0
    
    flat_obs = np.concatenate([
        player_obs,
        board_obs,
        trade_details,
        pending_trade,
        agent_id_one_hot
    ])
    
    return flat_obs

class MonopolyMAv2(ParallelEnv):
    """
    A PettingZoo-style wrapper for the MonopolyEnv.

    This wrapper converts the single-agent MonopolyEnv into a multi-agent environment
    that follows the PettingZoo Parallel API. This allows it to be used with
    multi-agent reinforcement learning algorithms.
    """
    metadata = {"render.modes": ["human"], "name": "monopoly_ma_v2"}

    def __init__(self, board_json_path=config.DEFAULT_BOARD_JSON, num_players=config.NUM_PLAYERS, max_steps=config.MAX_STEPS):
        """
        Initialize the multi-agent Monopoly environment.

        Args:
            board_json_path (str): Path to the board layout JSON file.
            num_players (int): The number of players in the game.
            max_steps (int): The maximum number of steps per episode.
        """
        self.internal_env = MonopolyEnv(
            board_json_path=board_json_path, 
            num_players=num_players, 
            max_steps=max_steps
        )
        
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {i: agent for i, agent in enumerate(self.possible_agents)}

        # The action space is the same for all agents
        self._action_space = self.internal_env.action_space
        
        # The observation space needs to be modified to include the player ID one-hot encoding
        self._observation_space = self._get_modified_observation_space(
            self.internal_env.observation_space, num_players
        )

    def _get_modified_observation_space(self, original_obs_space, num_players):
        """
        Modifies the observation space to include a one-hot player ID.

        This fulfills requirement 2.b: "Player-ID One-Hot in Obs".
        """
        # Get shapes and bounds from the original observation space
        player_obs_space = original_obs_space['player']
        
        # New player vector includes the original vector plus the one-hot encoding
        new_player_vec_len = player_obs_space.shape[0] + num_players
        
        # Create a new Box space for the player observation
        new_player_obs_space = Box(
            low=np.concatenate([player_obs_space.low, np.zeros(num_players, dtype=np.float32)]),
            high=np.concatenate([player_obs_space.high, np.ones(num_players, dtype=np.float32)]),
            shape=(new_player_vec_len,),
            dtype=np.float32
        )

        # Return the new composite observation space
        return Dict({
            "player": new_player_obs_space,
            "board": original_obs_space['board'],
            "current_player_id": original_obs_space['current_player_id'],
            "pending_trade_valid": original_obs_space['pending_trade_valid'],
            "trade_details": original_obs_space['trade_details']
        })

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns initial observations and info.
        """
        if seed is not None:
            self.internal_env.seed(seed)
        
        obs, info = self.internal_env.reset()
        self.agents = self.possible_agents[:]
        self.step_count = 0  # Reset step counter
        
        observations = self._get_all_observations()
        infos = {agent: info for agent in self.agents}

        return observations, infos

    def _get_all_observations(self):
        """
        Gets the observation for every agent in the environment.

        This is done by temporarily setting the `current_player_index` in the
        internal environment to each player's index and generating their observation.
        """
        observations = {}
        original_player_idx = self.internal_env.game.current_player_index
        num_players = len(self.possible_agents)

        for i, agent_id in enumerate(self.possible_agents):
            # Temporarily set the player index to get their specific observation
            self.internal_env.game.current_player_index = i
            obs = self.internal_env._get_obs()
            
            # Append the one-hot player ID to the player state vector
            player_id_onehot = np.zeros(num_players, dtype=np.float32)
            player_id_onehot[i] = 1.0
            obs["player"] = np.concatenate([obs["player"], player_id_onehot])

            observations[agent_id] = obs

        # Restore the original current player index
        self.internal_env.game.current_player_index = original_player_idx
        return observations

    def step(self, actions):
        """
        Takes a dictionary of actions and steps the environment for the current agent.
        """
        current_agent_idx = self.internal_env.game.current_player_index
        current_agent_id = self.agent_name_mapping[current_agent_idx]
        
        # If the current agent is not in the actions dict, it might have terminated.
        if current_agent_id not in actions:
            # Handle case where dead agent is expected to act. Step with a dummy action (e.g., skip turn).
            action = (6, 0) # Skip turn action
        else:
            action = actions[current_agent_id]

        # Step the internal environment
        _, reward, terminated, truncated, info = self.internal_env.step(action)

        self.step_count += 1
        is_max_steps_reached = self.step_count >= self.internal_env.max_steps

        # Distribute rewards: current player gets the reward, others get 0
        rewards = {agent: 0.0 for agent in self.possible_agents}
        rewards[current_agent_id] = reward
        
        # Game is over for everyone if one player is terminated OR max steps are reached
        game_over = terminated or truncated or is_max_steps_reached
        if game_over:
            self.agents = []
        
        terminations = {agent: terminated for agent in self.possible_agents}
        # Truncate for all agents if the base env truncates OR we hit the max step limit
        truncations = {agent: truncated or is_max_steps_reached for agent in self.possible_agents}

        observations = self._get_all_observations()
        infos = {agent: info for agent in self.possible_agents}

        # Add semantic features to the info dict for each agent
        # This fulfills requirement 2.c.
        semantic_features = {}
        for i, agent_id in enumerate(self.possible_agents):
            player = self.internal_env.game.players[i]
            semantic_features[agent_id] = {
                "cash": player.current_cash,
                "net_worth": player.get_net_worth(self.internal_env.game.board.get_board_meta()),
                "num_properties": len(player.assets),
                "num_monopolies": len(player.full_color_sets_possessed),
                "status": player.status
            }

        for agent_id in self.possible_agents:
            if agent_id in infos:
                infos[agent_id]['semantic_features'] = semantic_features
            else:
                infos[agent_id] = {'semantic_features': semantic_features}

        return observations, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        """Renders the environment."""
        return self.internal_env.render(mode)

    def close(self):
        """Closes the environment."""
        self.internal_env.close() 