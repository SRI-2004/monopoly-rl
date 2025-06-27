import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import json
import os
from tqdm import tqdm

from env_wrapper import MonopolyMAv2
from network import ActorCritic
from vectorized_env import SyncVecEnv
from scripted_agent import ScriptedAgent

def preprocess_obs(obs, device, num_players):
    """
    Flattens and preprocesses the observation dictionary from the environment.
    """
    player_obs = obs['player']
    board_obs = obs['board']
    trade_details = obs['trade_details']
    pending_trade = np.array([obs['pending_trade_valid']], dtype=np.float32)

    current_player_one_hot = np.zeros(num_players, dtype=np.float32)
    current_player_one_hot[obs['current_player_id']] = 1.0
    
    flat_obs = np.concatenate([
        player_obs,
        board_obs,
        trade_details,
        pending_trade,
        current_player_one_hot
    ])
    
    return torch.tensor(flat_obs, dtype=torch.float32).to(device)


def main(args):
    """
    Main training loop for Independent PPO (IPPO) on Monopoly.
    """
    # --- Hyperparameter Loading ---
    if args.hyperparameters:
        with open(args.hyperparameters) as f:
            hyperparams_from_json = json.load(f)
        
        # Override args with global values from the JSON file
        global_hyperparams = hyperparams_from_json.get("global", {})
        for key, value in global_hyperparams.items():
            setattr(args, key, value)
        
        per_agent_hyperparams = hyperparams_from_json.get("per_agent", {})
    else:
        per_agent_hyperparams = {}

    run_name = f"Monopoly_IPPO_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # --- Checkpointing Setup ---
    if args.save_path:
        save_path = os.path.join(args.save_path, run_name)
        os.makedirs(save_path, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 1. Vectorized Environment Setup
    def make_env(seed):
        def _f():
            env = MonopolyMAv2(num_players=args.num_players, max_steps=args.max_steps, board_json_path=args.board_json)
            env.reset(seed=seed)
            return env
        return _f

    envs = SyncVecEnv([make_env(args.seed + i) for i in range(args.num_envs)])
    agent_ids = envs.possible_agents
    num_agents = len(agent_ids)
    
    # --- Dynamically get observation and action space dimensions ---
    # To get the flattened observation size, we can process a sample observation
    sample_obs, _ = envs.reset()
    sample_obs_for_one_env = {k: v[0] for k, v in sample_obs[agent_ids[0]].items()}
    processed_sample = preprocess_obs(sample_obs_for_one_env, 'cpu', num_agents)
    total_input_dim = processed_sample.shape[0]
    print(f"Dynamically determined input dim: {total_input_dim}")
    
    action_dims = [envs.action_space('player_0').nvec[0], envs.action_space('player_0').nvec[1]]

    # Create a resolved set of hyperparameters for each agent
    # Priority: Per-agent JSON > Global JSON / Command-line
    resolved_hyperparams = {}
    for agent_id in agent_ids:
        # Start with the global/cmd-line defaults
        agent_h = vars(args).copy()
        # Update with this agent's specific settings
        agent_h.update(per_agent_hyperparams.get(agent_id, {}))
        resolved_hyperparams[agent_id] = agent_h

    # 2. IPPO Setup: One policy and optimizer per player
    policies = {
        agent_id: ActorCritic(total_input_dim, action_dims, hidden_dim=resolved_hyperparams[agent_id].get("hidden_dim")).to(device) 
        for agent_id in agent_ids
    }
    
    optimizer_map = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW
    }

    # Setup optimizers with potentially different types and learning rates
    optimizers = {}
    for agent_id in agent_ids:
        agent_h = resolved_hyperparams[agent_id]
        optimizer_name = agent_h.get("optimizer", "Adam")
        optimizer_class = optimizer_map.get(optimizer_name)
        if not optimizer_class:
            raise ValueError(f"Unknown optimizer '{optimizer_name}' for agent {agent_id}")
        
        agent_lr = agent_h.get("lr")
        optimizers[agent_id] = optimizer_class(policies[agent_id].parameters(), lr=agent_lr, eps=1e-5)

    # --- Curriculum Learning Setup ---
    scripted_agents = {}
    if args.curriculum_steps > 0:
        # Load board metadata for scripted agent logic
        with open(args.board_json) as f:
            board_data = json.load(f)
        board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}

        learning_agent_id = "player_0" # For simplicity, agent 0 is the only learner
        for agent_id in agent_ids:
            if agent_id != learning_agent_id:
                scripted_agents[agent_id] = ScriptedAgent(
                    player_id=agent_id, 
                    num_players=args.num_players,
                    board_meta=board_meta
                )
        print(f"Curriculum enabled: Training {learning_agent_id} against {len(scripted_agents)} scripted agents.")


    # 3. PPO Storage Setup
    # Each agent needs its own storage buffer, initialized on the correct device
    storage = {
        agent_id: {
            "obs": torch.zeros((args.num_steps, args.num_envs, total_input_dim), device=device),
            "log_probs_top": torch.zeros((args.num_steps, args.num_envs), device=device),
            "log_probs_sub": torch.zeros((args.num_steps, args.num_envs), device=device),
            "actions_top": torch.zeros((args.num_steps, args.num_envs), device=device),
            "actions_sub": torch.zeros((args.num_steps, args.num_envs), device=device),
            "rewards": torch.zeros((args.num_steps, args.num_envs), device=device),
            "values": torch.zeros((args.num_steps, args.num_envs), device=device),
            "dones": torch.zeros((args.num_steps, args.num_envs), device=device),
        }
        for agent_id in agent_ids
    }

    # 4. Training Loop
    global_step = 0
    start_time = time.time()
    
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = {agent_id: torch.zeros(args.num_envs).to(device) for agent_id in agent_ids}
    # GRU hidden states, one for each agent in each parallel environment
    next_hiddens = {
        agent_id: torch.zeros(1, args.num_envs, resolved_hyperparams[agent_id].get("hidden_dim")).to(device)
        for agent_id in agent_ids
    }
    
    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)

    for update in tqdm(range(1, num_updates + 1), desc="Training Progress"):
        # --- Rollout Phase ---
        for step in range(args.num_steps):
            global_step += 1 * args.num_envs
            
            # --- Acting ---
            env_actions = [{} for _ in range(args.num_envs)]
            
            # Determine which agents are learning vs. scripted for this step
            is_curriculum_phase = global_step < args.curriculum_steps
            
            with torch.no_grad():
                for agent_id in agent_ids:
                    # If agent is scripted, get action from its logic
                    if is_curriculum_phase and agent_id in scripted_agents:
                        for i in range(args.num_envs):
                            single_env_obs = {k: v[i] for k, v in next_obs[agent_id].items()}
                            action = scripted_agents[agent_id].get_action(single_env_obs)
                            env_actions[i][agent_id] = action
                        continue # Move to the next agent

                    # Otherwise, it's a learning agent, use the policy
                    agent_obs_dict = next_obs[agent_id]
                    
                    processed_obs_list = []
                    for i in range(args.num_envs):
                        single_env_obs = {k: v[i] for k, v in agent_obs_dict.items()}
                        processed_obs_list.append(preprocess_obs(single_env_obs, device, args.num_players))
                    
                    processed_obs = torch.stack(processed_obs_list)
                    
                    policy = policies[agent_id]
                    hiddens = next_hiddens[agent_id]
                    
                    top_logits, sub_logits, value, new_hiddens = policy(processed_obs, hiddens)

                    top_dist = torch.distributions.Categorical(logits=top_logits)
                    sub_dist = torch.distributions.Categorical(logits=sub_logits)
                    
                    top_action = top_dist.sample()
                    sub_action = sub_dist.sample()

                    # Store results only for learning agents
                    storage[agent_id]["obs"][step] = processed_obs
                    storage[agent_id]["values"][step] = value.squeeze()
                    storage[agent_id]["log_probs_top"][step] = top_dist.log_prob(top_action)
                    storage[agent_id]["log_probs_sub"][step] = sub_dist.log_prob(sub_action)
                    storage[agent_id]["actions_top"][step] = top_action
                    storage[agent_id]["actions_sub"][step] = sub_action
                    storage[agent_id]["dones"][step] = next_done[agent_id]
                    next_hiddens[agent_id] = new_hiddens
                    
                    # Format actions for the environment
                    for i in range(args.num_envs):
                        env_actions[i][agent_id] = (top_action[i].item(), sub_action[i].item())

            # --- Stepping the Environment ---
            next_obs, rewards, terminated, truncated, infos = envs.step(env_actions)
            
            for agent_id in agent_ids:
                # Only store rewards for learning agents
                if not (is_curriculum_phase and agent_id in scripted_agents):
                    storage[agent_id]["rewards"][step] = torch.tensor(rewards[agent_id]).to(device)
                next_done[agent_id] = torch.tensor(terminated[agent_id], dtype=torch.float32).to(device)
        
        # --- PPO Update Phase ---
        with torch.no_grad():
            for agent_id in agent_ids:
                # Skip update for scripted agents
                if is_curriculum_phase and agent_id in scripted_agents:
                    continue

                # To calculate advantages, we need the value of the next state
                agent_obs_dict = next_obs[agent_id]
                processed_obs_list = []
                for i in range(args.num_envs):
                    single_env_obs = {k: v[i] for k, v in agent_obs_dict.items()}
                    processed_obs_list.append(preprocess_obs(single_env_obs, device, args.num_players))
                processed_obs = torch.stack(processed_obs_list)

                _, _, next_value, _ = policies[agent_id](processed_obs, next_hiddens[agent_id])
                next_value = next_value.reshape(1, -1)
                
                # Calculate advantages using GAE
                advantages = torch.zeros_like(storage[agent_id]["rewards"]).to(device)
                last_gae_lam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done[agent_id]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - storage[agent_id]["dones"][t + 1]
                        nextvalues = storage[agent_id]["values"][t + 1]
                    
                    delta = storage[agent_id]["rewards"][t] + args.gamma * nextvalues * nextnonterminal - storage[agent_id]["values"][t]
                    advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * nextnonterminal * last_gae_lam
                
                storage[agent_id]["advantages"] = advantages
                storage[agent_id]["returns"] = advantages + storage[agent_id]["values"]


        # --- Update Policy and Value Networks ---
        for agent_id in agent_ids:
            # Skip update for scripted agents
            if is_curriculum_phase and agent_id in scripted_agents:
                continue

            # Get agent-specific hyperparams for the update
            agent_h = resolved_hyperparams[agent_id]
            clip_coef = agent_h.get("clip_coef")
            entropy_coef = agent_h.get("entropy_coef")

            # Flatten the batch
            b_obs = storage[agent_id]["obs"].reshape((-1, total_input_dim))
            b_log_probs_top = storage[agent_id]["log_probs_top"].reshape(-1)
            b_log_probs_sub = storage[agent_id]["log_probs_sub"].reshape(-1)
            b_actions_top = storage[agent_id]["actions_top"].reshape(-1)
            b_actions_sub = storage[agent_id]["actions_sub"].reshape(-1)
            b_advantages = storage[agent_id]["advantages"].reshape(-1)
            b_returns = storage[agent_id]["returns"].reshape(-1)
            b_values = storage[agent_id]["values"].reshape(-1)

            # Optimizing the policy and value network
            inds = np.arange(args.num_envs * args.num_steps)
            for epoch in range(args.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, args.num_envs * args.num_steps, args.minibatch_size):
                    end = start + args.minibatch_size
                    minibatch_inds = inds[start:end]

                    # This is a simplification. For GRU, we should handle sequences.
                    # A proper implementation would sample sequences and manage hidden states.
                    # For now, we treat them as independent samples which is not ideal for GRU.
                    mb_obs = b_obs[minibatch_inds]
                    
                    # Dummy hidden state for minibatch
                    mb_hidden = torch.zeros(1, len(minibatch_inds), resolved_hyperparams[agent_id].get("hidden_dim")).to(device)

                    top_logits, sub_logits, new_values, _ = policies[agent_id](mb_obs, mb_hidden)
                    new_values = new_values.view(-1)
                    
                    top_dist = torch.distributions.Categorical(logits=top_logits)
                    sub_dist = torch.distributions.Categorical(logits=sub_logits)

                    # Policy loss
                    log_ratio_top = top_dist.log_prob(b_actions_top[minibatch_inds]) - b_log_probs_top[minibatch_inds]
                    log_ratio_sub = sub_dist.log_prob(b_actions_sub[minibatch_inds]) - b_log_probs_sub[minibatch_inds]
                    
                    # We can sum log_ratios as they are independent actions
                    log_ratio = log_ratio_top + log_ratio_sub
                    ratio = log_ratio.exp()

                    mb_advs = b_advantages[minibatch_inds]
                    
                    pg_loss1 = -mb_advs * ratio
                    pg_loss2 = -mb_advs * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    v_loss = 0.5 * ((new_values - b_returns[minibatch_inds]) ** 2).mean()

                    # Entropy loss
                    entropy = (top_dist.entropy() + sub_dist.entropy()).mean()
                    
                    loss = pg_loss - entropy_coef * entropy + v_loss * 0.5

                    optimizers[agent_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policies[agent_id].parameters(), 0.5)
                    optimizers[agent_id].step()

        # Logging to TensorBoard
        writer.add_scalar("charts/learning_rate", optimizers["player_0"].param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.item(), global_step)
        writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)

        # --- Save Checkpoint ---
        if args.save_path and update % args.checkpoint_freq == 0:
            is_curriculum_phase = global_step < args.curriculum_steps
            for agent_id in agent_ids:
                # We only save learning agents
                if not (is_curriculum_phase and agent_id in scripted_agents):
                    ckpt_path = os.path.join(save_path, f"policy_{agent_id}_update_{update}.pt")
                    torch.save({
                        "policy_state_dict": policies[agent_id].state_dict(),
                        "optimizer_state_dict": optimizers[agent_id].state_dict(),
                        "global_step": global_step,
                        "update": update,
                        "hidden_dim": resolved_hyperparams[agent_id].get("hidden_dim")
                    }, ckpt_path)
            tqdm.write(f"Checkpoints saved at update {update}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hyperparameters", type=str, default="rl_agent/hyperparameters.json", help="Path to a JSON file with hyperparameters")
    parser.add_argument("--board-json", type=str, default="/home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json", help="Path to board json")
    parser.add_argument("--num-players", type=int, default=4, help="Number of players")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total timesteps for training")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    # PPO hyper-set from plan
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    # Add new args
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128, help="Number of steps to run in each environment per policy update")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--curriculum-steps", type=int, default=0, help="Number of timesteps to train with scripted opponents. If 0, disabled.")
    parser.add_argument("--save-path", type=str, default="rl_agent/checkpoints", help="Path to save checkpoints.")
    parser.add_argument("--checkpoint-freq", type=int, default=100, help="Frequency (in updates) to save checkpoints.")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Default optimizer (Adam, AdamW)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="The hidden dimension of the neural network")
    
    args = parser.parse_args()
    main(args) 