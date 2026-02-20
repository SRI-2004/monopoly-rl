import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np
from pettingzoo import ParallelEnv
import sys
import os
from datetime import datetime
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monopoly_env.envs.monopoly_env import MonopolyEnv
from monopoly_env import config

def preprocess_obs(obs, num_players, agent_id_int):
    """
    Preprocesses the observation dictionary into a flat numpy array.
    
    Args:
        obs: Dictionary observation from the environment
        num_players: Number of players in the game
        agent_id_int: Integer ID of the current agent
    
    Returns:
        Flattened numpy array ready for neural network input
    """
    # Start with the player vector (already includes one-hot player ID)
    player_vec = obs["player"]
    
    # Add the board state
    board_vec = obs["board"].flatten()
    
    # Add pending trade information
    pending_trade_vec = np.array([obs["pending_trade_valid"]], dtype=np.float32)
    
    # Add trade details (flattened)
    trade_details_vec = obs["trade_details"].flatten()
    
    # Concatenate all vectors (removed redundant current_player_id)
    full_obs = np.concatenate([
        player_vec,
        board_vec,
        pending_trade_vec,
        trade_details_vec
    ])
    
    return full_obs.astype(np.float32)

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
        
        # Initialize comprehensive metrics tracking
        self.episode_metrics = {
            'start_time': None,
            'step_count': 0,
            'player_actions': {agent: [] for agent in self.possible_agents},
            'player_rewards': {agent: [] for agent in self.possible_agents},
            'behavioral_counters': {agent: self._init_behavioral_counters() for agent in self.possible_agents},
            'economic_indicators': {agent: [] for agent in self.possible_agents},
            'game_events': []
        }

    def _init_behavioral_counters(self):
        """Initialize behavioral counters for comprehensive tracking."""
        return {
            'properties_purchased': 0,
            'houses_built': 0,
            'hotels_built': 0,
            'trades_proposed': 0,
            'trades_accepted': 0,
            'trades_rejected': 0,
            'jail_fines_paid': 0,
            'jail_cards_used': 0,
            'mortgage_actions': 0,
            'unmortgage_actions': 0,
            'property_sales': 0,
            'bankruptcy_risk_episodes': 0,
            'turns_in_jail': 0,
            'dice_rolls': 0,
            'total_rent_paid': 0,
            'total_rent_received': 0,
            'tax_payments': 0,
            'chance_card_draws': 0,
            'community_chest_draws': 0
        }

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
        
        # Reset comprehensive metrics tracking
        self.episode_metrics = {
            'start_time': datetime.now(),
            'step_count': 0,
            'player_actions': {agent: [] for agent in self.possible_agents},
            'player_rewards': {agent: [] for agent in self.possible_agents},
            'behavioral_counters': {agent: self._init_behavioral_counters() for agent in self.possible_agents},
            'economic_indicators': {agent: [] for agent in self.possible_agents},
            'game_events': []
        }
        
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
            
            # Add action masks for this player
            current_player = self.internal_env.game.players[i]
            valid_top_actions = self.internal_env.game.get_valid_actions(current_player)
            
            # Get valid sub-actions for each top-level action
            valid_sub_actions = []
            for top_action_idx in range(12):  # 12 top-level actions
                if valid_top_actions[top_action_idx]:
                    sub_mask = self.internal_env.game.get_valid_subactions(current_player, top_action_idx)
                    # Pad or truncate to 252 dimensions (max sub-action space)
                    if len(sub_mask) < 252:
                        padded_mask = np.zeros(252, dtype=bool)
                        padded_mask[:len(sub_mask)] = sub_mask
                        valid_sub_actions.append(padded_mask)
                    else:
                        valid_sub_actions.append(sub_mask[:252])
                else:
                    # If top action is invalid, all sub-actions are invalid
                    valid_sub_actions.append(np.zeros(252, dtype=bool))
            
            obs["action_mask"] = [valid_top_actions, valid_sub_actions]

            observations[agent_id] = obs

        # Restore the original current player index
        self.internal_env.game.current_player_index = original_player_idx
        return observations

    def _update_behavioral_metrics(self, agent_id, action, reward, info):
        """Update comprehensive behavioral metrics for the agent."""
        counters = self.episode_metrics['behavioral_counters'][agent_id]
        
        # Track actions
        action_top = action[0] if isinstance(action, tuple) else action
        self.episode_metrics['player_actions'][agent_id].append({
            'step': self.step_count,
            'action_top': action_top,
            'action_sub': action[1] if isinstance(action, tuple) and len(action) > 1 else 0,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        })
        
        # Track rewards
        self.episode_metrics['player_rewards'][agent_id].append(reward)
        
        # Update behavioral counters based on action and info
        if 'result' in info:
            result = info['result']
            
            # Property purchases - look for multiple patterns
            if any(word in result.lower() for word in ['purchased', 'bought', 'acquired']):
                counters['properties_purchased'] += 1
                
            # House/hotel building
            if 'built' in result.lower():
                if 'house' in result.lower():
                    counters['houses_built'] += 1
                elif 'hotel' in result.lower():
                    counters['hotels_built'] += 1
                    
            # Trading
            if 'trade' in result.lower():
                if 'proposed' in result.lower():
                    counters['trades_proposed'] += 1
                elif 'accepted' in result.lower():
                    counters['trades_accepted'] += 1
                elif 'rejected' in result.lower():
                    counters['trades_rejected'] += 1
                    
            # Jail actions
            if 'jail' in result.lower():
                if 'fine' in result.lower():
                    counters['jail_fines_paid'] += 1
                elif 'card' in result.lower():
                    counters['jail_cards_used'] += 1
                    
            # Mortgage actions
            if 'mortgage' in result.lower():
                if 'mortgaged' in result.lower():
                    counters['mortgage_actions'] += 1
                elif 'unmortgaged' in result.lower():
                    counters['unmortgage_actions'] += 1
                    
            # Property sales
            if 'sold' in result.lower():
                counters['property_sales'] += 1
                
            # Tax payments
            if 'tax' in result.lower():
                counters['tax_payments'] += 1
                
            # Card draws
            if 'chance' in result.lower():
                counters['chance_card_draws'] += 1
            elif 'community chest' in result.lower():
                counters['community_chest_draws'] += 1
                
        # Track economic indicators
        player_idx = int(agent_id.split('_')[1])
        player = self.internal_env.game.players[player_idx]
        
        economic_data = {
            'step': self.step_count,
            'cash': player.current_cash,
            'net_worth': player.get_net_worth(self.internal_env.game.board.get_board_meta()),
            'num_properties': len(player.assets),
            'num_monopolies': len(player.full_color_sets_possessed),
            'debt_ratio': max(0, -player.current_cash) / max(1, player.get_net_worth(self.internal_env.game.board.get_board_meta())),
            'property_value': sum(prop.get('price', 0) for prop_id in player.assets 
                                for prop in [self.internal_env.game.board.get_board_meta().get(str(prop_id), {})] if prop),
            'liquidity_ratio': player.current_cash / max(1, player.get_net_worth(self.internal_env.game.board.get_board_meta())),
            'status': player.status,
            'phase': player.phase,
            'position': player.current_position,
            'turns_in_jail': player.currently_in_jail,
            'timestamp': datetime.now().isoformat()
        }
        
        self.episode_metrics['economic_indicators'][agent_id].append(economic_data)
        
        # Check for bankruptcy risk
        if player.current_cash < 0:
            counters['bankruptcy_risk_episodes'] += 1
            
        # Track jail time
        if player.currently_in_jail:
            counters['turns_in_jail'] += 1
            
        # Track dice rolls (approximate based on phase changes)
        if player.phase == 'post-roll':
            counters['dice_rolls'] += 1

    def _generate_comprehensive_info(self, agent_id, action, reward, info):
        """Generate comprehensive info dictionary for the agent."""
        player_idx = int(agent_id.split('_')[1])
        player = self.internal_env.game.players[player_idx]
        
        # Update behavioral metrics
        self._update_behavioral_metrics(agent_id, action, reward, info)
        
        # Get current behavioral counters
        behavioral_metrics = self.episode_metrics['behavioral_counters'][agent_id].copy()
        
        # Add derived metrics
        behavioral_metrics.update({
            'trade_success_rate': (behavioral_metrics['trades_accepted'] / 
                                 max(1, behavioral_metrics['trades_proposed'])),
            'jail_escape_efficiency': (behavioral_metrics['jail_cards_used'] / 
                                     max(1, behavioral_metrics['jail_fines_paid'] + behavioral_metrics['jail_cards_used'])),
            'property_development_rate': (behavioral_metrics['houses_built'] + behavioral_metrics['hotels_built']) / 
                                       max(1, behavioral_metrics['properties_purchased']),
            'financial_stress_frequency': behavioral_metrics['bankruptcy_risk_episodes'] / max(1, self.step_count),
            'activity_level': len(self.episode_metrics['player_actions'][agent_id]) / max(1, self.step_count)
        })
        
        # Add current economic state
        economic_state = {
            'cash': player.current_cash,
            'net_worth': player.get_net_worth(self.internal_env.game.board.get_board_meta()),
            'num_properties': len(player.assets),
            'num_monopolies': len(player.full_color_sets_possessed),
            'liquidity_ratio': player.current_cash / max(1, player.get_net_worth(self.internal_env.game.board.get_board_meta())),
            'debt_ratio': max(0, -player.current_cash) / max(1, player.get_net_worth(self.internal_env.game.board.get_board_meta())),
            'status': player.status,
            'phase': player.phase,
            'position': player.current_position,
            'in_jail': player.currently_in_jail
        }
        
        # Combine all information
        comprehensive_info = info.copy()
        comprehensive_info.update({
            'behavioral_metrics': behavioral_metrics,
            'economic_state': economic_state,
            'step_count': self.step_count,
            'game_duration': (datetime.now() - self.episode_metrics['start_time']).total_seconds() if self.episode_metrics['start_time'] else 0
        })
        
        return comprehensive_info

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
        self.episode_metrics['step_count'] = self.step_count
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
        
        # Generate comprehensive info for each agent
        infos = {}
        for agent_id in self.possible_agents:
            if agent_id == current_agent_id:
                # Current agent gets comprehensive info
                infos[agent_id] = self._generate_comprehensive_info(agent_id, action, reward, info)
            else:
                # Other agents get basic info
                infos[agent_id] = {
                    'step_count': self.step_count,
                    'current_agent': current_agent_id,
                    'game_duration': (datetime.now() - self.episode_metrics['start_time']).total_seconds() if self.episode_metrics['start_time'] else 0
                }

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
        
        # Add episode-level metrics when game ends
        if game_over:
            episode_summary = self._generate_episode_summary()
            for agent_id in self.possible_agents:
                infos[agent_id]['episode_summary'] = episode_summary
                infos[agent_id]['all_behavioral_metrics'] = {
                    agent: counters for agent, counters in self.episode_metrics['behavioral_counters'].items()
                }
        
        # Always include current behavioral metrics for the active agent
        if current_agent_id in infos and 'behavioral_metrics' in infos[current_agent_id]:
            # Update the global step count in behavioral metrics
            infos[current_agent_id]['behavioral_metrics']['global_step'] = self.step_count

        return observations, rewards, terminations, truncations, infos

    def _generate_episode_summary(self):
        """Generate comprehensive episode summary for analysis."""
        if not self.episode_metrics['start_time']:
            return {}
            
        episode_duration = (datetime.now() - self.episode_metrics['start_time']).total_seconds()
        
        # Calculate winner
        winner = None
        max_net_worth = -float('inf')
        for i, agent_id in enumerate(self.possible_agents):
            player = self.internal_env.game.players[i]
            net_worth = player.get_net_worth(self.internal_env.game.board.get_board_meta())
            if player.status == 'won' or net_worth > max_net_worth:
                max_net_worth = net_worth
                winner = agent_id
        
        # Aggregate behavioral metrics
        total_properties = sum(counters['properties_purchased'] for counters in self.episode_metrics['behavioral_counters'].values())
        total_houses = sum(counters['houses_built'] for counters in self.episode_metrics['behavioral_counters'].values())
        total_trades = sum(counters['trades_proposed'] for counters in self.episode_metrics['behavioral_counters'].values())
        
        # Calculate economic metrics
        final_net_worths = {}
        wealth_inequality = 0
        total_wealth = 0
        
        for i, agent_id in enumerate(self.possible_agents):
            player = self.internal_env.game.players[i]
            net_worth = player.get_net_worth(self.internal_env.game.board.get_board_meta())
            final_net_worths[agent_id] = net_worth
            total_wealth += net_worth
        
        # Calculate Gini coefficient for wealth inequality
        if total_wealth > 0:
            net_worths = list(final_net_worths.values())
            net_worths.sort()
            n = len(net_worths)
            cumulative_wealth = np.cumsum(net_worths)
            wealth_inequality = (n + 1 - 2 * np.sum(cumulative_wealth) / cumulative_wealth[-1]) / n
        
        return {
            'episode_duration_seconds': episode_duration,
            'total_steps': self.step_count,
            'winner': winner,
            'final_net_worths': final_net_worths,
            'wealth_inequality_gini': wealth_inequality,
            'total_properties_purchased': total_properties,
            'total_houses_built': total_houses,
            'total_trades_proposed': total_trades,
            'game_completion_type': 'natural' if self.step_count < self.internal_env.max_steps else 'truncated',
            'player_behavioral_summary': {
                agent_id: {
                    'total_actions': len(self.episode_metrics['player_actions'][agent_id]),
                    'total_reward': sum(self.episode_metrics['player_rewards'][agent_id]),
                    'avg_reward_per_action': np.mean(self.episode_metrics['player_rewards'][agent_id]) if self.episode_metrics['player_rewards'][agent_id] else 0,
                    'final_economic_state': self.episode_metrics['economic_indicators'][agent_id][-1] if self.episode_metrics['economic_indicators'][agent_id] else {}
                }
                for agent_id in self.possible_agents
            }
        }

    def render(self, mode="human"):
        """Renders the environment."""
        return self.internal_env.render(mode)

    def close(self):
        """Closes the environment."""
        self.internal_env.close() 