# train_agent.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from gymnasium.utils.env_checker import check_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter
from ppo import make_env
from dqn import FlattenActionWrapper
from monopoly_env.envs.monopoly_env import MonopolyEnv
from hdqn import HDQNNetwork, ReplayBuffer, select_action

# Helper function to flatten the observation (assumes obs is a dict with "player" and "board")
def flatten_obs(obs):
    return np.concatenate([obs["player"], obs["board"]], axis=0)

def train_hdqn(num_episodes, max_steps, batch_size, gamma, learning_rate, buffer_capacity,
               target_update_freq, epsilon_start, epsilon_end, epsilon_decay):
    writer = SummaryWriter(log_dir="runs/hdqn_monopoly")
    episode_rewards = []
    episode_losses = []
    epsilon_values = []

    # Create environment
    env = MonopolyEnv()
    obs_sample, _ = env.reset(seed=0)
    flat_obs = flatten_obs(obs_sample)
    obs_dim = flat_obs.shape[0]   # e.g., 240
    top_dim = 12
    sub_dim = 252

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = HDQNNetwork(obs_dim, top_dim, sub_dim).to(device)
    target_net = HDQNNetwork(obs_dim, top_dim, sub_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)
    total_timesteps = 0

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        state = flatten_obs(obs)
        episode_reward = 0
        losses_this_episode = []
        for step in range(max_steps):
            total_timesteps += 1
            # Epsilon decay schedule
            epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (total_timesteps / epsilon_decay))
            action = select_action(policy_net, state, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = flatten_obs(next_obs)
            done = terminated or truncated
            episode_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(np.array(states)).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)

                top_q, sub_q = policy_net(states)
                a_tops = torch.LongTensor([a[0] for a in actions]).unsqueeze(1).to(device)
                a_subs = torch.LongTensor([a[1] for a in actions]).unsqueeze(1).to(device)
                q_top_taken = top_q.gather(1, a_tops).squeeze(1)
                q_sub_taken = sub_q.gather(1, a_subs).squeeze(1)
                q_values = q_top_taken + q_sub_taken

                with torch.no_grad():
                    next_top_q, next_sub_q = target_net(next_states)
                    next_top_max = next_top_q.max(1)[0]
                    next_sub_max = next_sub_q.max(1)[0]
                    next_q = next_top_max + next_sub_max
                    target = rewards + gamma * next_q * (1 - dones)

                loss = F.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_this_episode.append(loss.item())

            if done:
                break

            if total_timesteps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(episode_reward)
        avg_loss = np.mean(losses_this_episode) if losses_this_episode else 0
        episode_losses.append(avg_loss)
        epsilon_values.append(epsilon)

        print(f"Episode {episode} Reward: {episode_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}")

        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Loss/Episode", avg_loss, episode)
        writer.add_scalar("Epsilon/Episode", epsilon, episode)

    env.close()
    torch.save(policy_net.state_dict(), "hdqn_monopoly.pth")
    writer.close()

def train_dqn(total_timesteps=100_000, num_envs=4):
    # Use a single environment instance for checking.
    env_single = FlattenActionWrapper(MonopolyEnv())
    check_env(env_single, warn=True)

    # For DQN, we typically use DummyVecEnv.
    env = DummyVecEnv([make_env(i) for i in range(num_envs)])

    # Create the DQN model with TensorBoard logging.
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="runs/dqn_monopoly")
    model.learn(total_timesteps=total_timesteps)
    model.save("dqn_monopoly")
    env.close()

def train_ppo(total_timesteps=100_000, num_envs=4):
    # (Optionally) check a single env.

    # Optionally, you can run an environment check.
    # Note: Running check_env() on a vectorized env might not work,
    # so we check a single instance.
    single_env = MonopolyEnv()
    check_env(single_env, warn=True)

    # Create a vectorized environment using SubprocVecEnv.
    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    # Create and train the PPO model with TensorBoard logging.
    # "MultiInputPolicy" is used to support dictionary observations.
    model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log="runs/ppo_monopoly")
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_monopoly")
    vec_env.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="hdqn", help="Algorithm to use (currently only 'hdqn' is supported)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--buffer_capacity", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Target network update frequency")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.1, help="Ending epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=100000, help="Timesteps over which epsilon decays")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps for training")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments")
    args = parser.parse_args()

    if args.algo == "hdqn":
        train_hdqn(num_episodes=args.episodes,
                   max_steps=args.max_steps,
                   batch_size=args.batch_size,
                   gamma=args.gamma,
                   learning_rate=args.lr,
                   buffer_capacity=args.buffer_capacity,
                   target_update_freq=args.target_update_freq,
                   epsilon_start=args.epsilon_start,
                   epsilon_end=args.epsilon_end,
                   epsilon_decay=args.epsilon_decay)
    elif args.algo == "dqn":
        train_dqn(total_timesteps=args.timesteps, num_envs=args.envs)
    elif args.algo == "ppo":
        train_ppo(total_timesteps=args.timesteps, num_envs=args.envs)
    else:
        raise ValueError("Unknown algorithm. Currently only 'hdqn' is supported.")

if __name__ == "__main__":
    main()
