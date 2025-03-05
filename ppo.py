# ppo.py
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
import argparse
from monopoly_env.envs.monopoly_env import MonopolyEnv


def make_env(seed: int):
    """
    Returns a function that creates an instance of MonopolyEnv with a given seed.
    """

    def _init():
        env = MonopolyEnv()  # You can pass additional parameters if needed.
        env.seed(seed)
        return env

    return _init


def train_ppo(total_timesteps=1_000_000, num_envs=4):
    # (Optionally) check a single env.
    single_env = MonopolyEnv()
    check_env(single_env, warn=True)

    # Create a vectorized environment.
    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Create and train the PPO model.
    # We use "MultiInputPolicy" to support dictionary observations.
    model = PPO("MultiInputPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_monopoly")
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps for training")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments")
    args = parser.parse_args()
    train_ppo(total_timesteps=args.timesteps, num_envs=args.envs)
