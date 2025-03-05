# dqn.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from monopoly_env.envs.monopoly_env import MonopolyEnv


class FlattenActionWrapper(gym.ActionWrapper):
    """
    A wrapper that converts a MultiDiscrete action space of the form
    MultiDiscrete([top_dim, sub_dim]) into a single Discrete action space
    of size top_dim * sub_dim.

    The mapping is:
        flat_action = a_top * sub_dim + a_sub
    """

    def __init__(self, env):
        super(FlattenActionWrapper, self).__init__(env)
        # Assume original action space is Tuple(Discrete(top_dim), Discrete(sub_dim))
        self.top_dim = env.action_space.spaces[0].n
        self.sub_dim = env.action_space.spaces[1].n
        self.action_space = spaces.Discrete(self.top_dim * self.sub_dim)

    def action(self, action):
        """
        Converts the flat discrete action into a tuple (a_top, a_sub).
        """
        a_top = action // self.sub_dim
        a_sub = action % self.sub_dim
        return (a_top, a_sub)


def make_env(seed: int):
    def _init():
        env = MonopolyEnv()
        env = FlattenActionWrapper(env)
        env.seed(seed)
        return env

    return _init




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    train_dqn(total_timesteps=args.timesteps, num_envs=args.envs)
