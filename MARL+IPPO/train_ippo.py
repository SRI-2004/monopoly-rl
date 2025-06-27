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
import math

from env_wrapper import MonopolyMAv2, preprocess_obs
from network import ActorCritic
from scripted_agent import ScriptedAgent
from parallel_env import MultiProcessingVecEnv

def schedule_factory(schedule_str, total_timesteps):
    """
    Parses a schedule string and returns a function that takes the current step
    and returns the hyperparameter value.
    Format: "type:start_val->end_val@transition_step"
    """
    if isinstance(schedule_str, (int, float)):
        return lambda step: schedule_str

    parts = schedule_str.split(':')
    schedule_type = parts[0]
    values_str = parts[1]

    transition_step = total_timesteps
    if '@' in values_str:
        values_str, transition_str = values_str.split('@')
        transition_step = int(float(transition_str))

    value_parts = values_str.split('->')
    start_val = float(value_parts[0])
    end_val = float(value_parts[1]) if len(value_parts) > 1 else start_val

    if schedule_type == 'linear':
        def schedule(step):
            progress = min(1.0, step / transition_step)
            return start_val + progress * (end_val - start_val)
        return schedule
    elif schedule_type == 'cosine':
        def schedule(step):
            progress = min(1.0, step / transition_step)
            return end_val + 0.5 * (start_val - end_val) * (1 + math.cos(math.pi * progress))
        return schedule
    elif schedule_type == 'piecewise':
         def schedule(step):
            return start_val if step < transition_step else end_val
         return schedule
    elif schedule_type == 'fixed':
        return lambda step: start_val
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def train(args):
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
    if args.track:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    else:
        writer = None

    # --- Checkpointing Setup ---
    if args.save_path:
        save_path = os.path.join(args.save_path, run_name)
        os.makedirs(save_path, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 1. Vectorized Environment Setup
    def make_env(seed):
        def _f():
            # Use the base environment, not the PettingZoo wrapper, for the curriculum phase
            env = MonopolyMAv2(num_players=args.num_players, max_steps=args.max_steps, board_json_path=args.board_json)
            env.reset(seed=seed)
            return env
        return _f

    # The new parallel env handles preprocessing internally
    envs = MultiProcessingVecEnv([make_env(args.seed + i) for i in range(args.num_envs)], seed=args.seed)
    agent_ids = envs.possible_agents
    num_agents = envs.num_agents
    
    # --- Dynamically get observation and action space dimensions ---
    total_input_dim = envs.single_observation_space_shape[0]
    print(f"Dynamically determined input dim: {total_input_dim}")
    
    action_dims = [envs.action_space('player_0').nvec[0], envs.action_space('player_0').nvec[1]]

    # Create a resolved set of hyperparameters for each agent
    # Priority: Per-agent JSON > Global JSON / Command-line
    resolved_hyperparams = {}
    for agent_id in agent_ids:
        agent_h = vars(args).copy()
        agent_h.update(per_agent_hyperparams.get(agent_id, {}))
        resolved_hyperparams[agent_id] = agent_h

    # --- Scheduler Setup ---
    # Create schedule functions for each agent for dynamic hyperparams
    schedule_funcs = {agent_id: {} for agent_id in agent_ids}
    for agent_id, agent_h in resolved_hyperparams.items():
        schedule_funcs[agent_id]['lr'] = schedule_factory(agent_h.get("lr"), args.total_timesteps)
        schedule_funcs[agent_id]['entropy'] = schedule_factory(agent_h.get("entropy_schedule", args.entropy_coef), args.total_timesteps)
        schedule_funcs[agent_id]['clip'] = schedule_factory(agent_h.get("clip_schedule", args.clip_coef), args.total_timesteps)


    # 2. IPPO Setup: One policy and optimizer per player
    policies = {}
    for agent_id in agent_ids:
        policy = ActorCritic(total_input_dim, action_dims, hidden_dim=resolved_hyperparams[agent_id].get("hidden_dim")).to(device)
        if args.compile:
            # Check if torch.compile is available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                print(f"Compiling policy for {agent_id}...")
                policy = torch.compile(policy)
            else:
                print("torch.compile not available, skipping.")
        policies[agent_id] = policy

    
    optimizer_map = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "RMSprop": optim.RMSprop,
    }

    # Setup optimizers with potentially different types and learning rates
    optimizers = {}
    for agent_id in agent_ids:
        agent_h = resolved_hyperparams[agent_id]
        optimizer_name = agent_h.get("optimizer", "Adam")
        optimizer_class = optimizer_map.get(optimizer_name)
        if not optimizer_class:
            raise ValueError(f"Unknown optimizer '{optimizer_name}' for agent {agent_id}")
        
        # We set the initial LR here, but it will be updated by the scheduler each step
        initial_lr = schedule_funcs[agent_id]['lr'](0)
        
        optimizer_kwargs = {'lr': initial_lr, 'eps': 1e-5}
        if optimizer_name == "RMSprop":
            optimizer_kwargs['alpha'] = agent_h.get("rmsprop_alpha", 0.99)
        
        optimizers[agent_id] = optimizer_class(policies[agent_id].parameters(), **optimizer_kwargs)

    # Automatic Mixed Precision (AMP) scalers, one for each optimizer
    scalers = {
        agent_id: torch.cuda.amp.GradScaler(enabled=args.cuda)
        for agent_id in agent_ids
    }

    # --- Curriculum Learning Setup ---
    scripted_agents = {}
    curriculum_env = None
    if args.curriculum_steps > 0:
        # For the curriculum phase, we use a single, non-parallel environment
        # because the logic is simpler and it's a limited-time phase.
        curriculum_env = make_env(args.seed)() # Create one instance
        
        with open(args.board_json) as f:
            board_data = json.load(f)
        board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}

        learning_agent_id = "player_0" 
        for agent_id in agent_ids:
            if agent_id != learning_agent_id:
                # Note: The player_id here is the string 'player_X', not an integer
                scripted_agents[agent_id] = ScriptedAgent(
                    player_id=agent_id.split('_')[-1], # The scripted agent expects an integer ID
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
    
    # Initial reset for the correct environment (curriculum or parallel)
    is_curriculum_phase = global_step < args.curriculum_steps
    if is_curriculum_phase:
        next_obs_dict, _ = curriculum_env.reset(seed=args.seed)
        # In curriculum mode, num_envs is 1
        next_done = {agent_id: torch.zeros(1).to(device) for agent_id in agent_ids}
        next_hiddens = {
            agent_id: torch.zeros(1, 1, resolved_hyperparams[agent_id].get("hidden_dim")).to(device)
            for agent_id in agent_ids if agent_id not in scripted_agents
        }
    else:
        next_obs_stacked, _ = envs.reset()
        next_done = {agent_id: torch.zeros(args.num_envs).to(device) for agent_id in agent_ids}
        next_hiddens = {
            agent_id: torch.zeros(1, args.num_envs, resolved_hyperparams[agent_id].get("hidden_dim")).to(device)
            for agent_id in agent_ids
        }

    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)

    for update in tqdm(range(1, num_updates + 1), desc="Training Progress"):
        # --- Anneal Hyperparameters ---
        current_step = update * args.num_steps * args.num_envs
        for agent_id, optimizer in optimizers.items():
            new_lr = schedule_funcs[agent_id]['lr'](current_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # --- Rollout Phase ---
        for step in range(args.num_steps):
            global_step += 1 * args.num_envs
            is_curriculum_phase = global_step < args.curriculum_steps
            
            # --- Acting ---
            action_dict = {}
            
            with torch.no_grad():
                if is_curriculum_phase:
                    # --- Curriculum Acting (Single Environment) ---
                    current_player_id = curriculum_env.agent_selection
                    if current_player_id in scripted_agents:
                        action = scripted_agents[current_player_id].get_action(next_obs_dict[current_player_id])
                    else: # Learning agent's turn
                        processed_obs_np = preprocess_obs(next_obs_dict[current_player_id], num_agents, int(current_player_id.split('_')[-1]))
                        processed_obs = torch.tensor(processed_obs_np, dtype=torch.float32, device=device).unsqueeze(0) # Add batch dim
                        
                        policy = policies[current_player_id]
                        hiddens = next_hiddens[current_player_id]
                        top_logits, sub_logits, value, new_hiddens = policy(processed_obs, hiddens)

                        top_dist = torch.distributions.Categorical(logits=top_logits)
                        sub_dist = torch.distributions.Categorical(logits=sub_logits)
                        top_action = top_dist.sample()
                        sub_action = sub_dist.sample()
                        action = (top_action.item(), sub_action.item())
                        
                        # Store results for the single learning agent
                        storage[current_player_id]["obs"][step] = processed_obs.squeeze(0)
                        storage[current_player_id]["values"][step] = value.squeeze()
                        storage[current_player_id]["log_probs_top"][step] = top_dist.log_prob(top_action)
                        storage[current_player_id]["log_probs_sub"][step] = sub_dist.log_prob(sub_action)
                        storage[current_player_id]["actions_top"][step] = top_action
                        storage[current_player_id]["actions_sub"][step] = sub_action
                        storage[current_player_id]["dones"][step] = next_done[current_player_id]
                        next_hiddens[current_player_id] = new_hiddens
                    
                    curriculum_env.step(action)

                else:
                    # --- Self-Play Acting (Parallel Environments) ---
                    for i, agent_id in enumerate(agent_ids):
                        agent_obs = torch.tensor(next_obs_stacked[:, i, :], dtype=torch.float32, device=device)
                        policy = policies[agent_id]
                        hiddens = next_hiddens[agent_id]
                        
                        top_logits, sub_logits, value, new_hiddens = policy(agent_obs, hiddens)

                        top_dist = torch.distributions.Categorical(logits=top_logits)
                        sub_dist = torch.distributions.Categorical(logits=sub_logits)
                        top_action = top_dist.sample()
                        sub_action = sub_dist.sample()

                        storage[agent_id]["obs"][step] = agent_obs
                        storage[agent_id]["values"][step] = value.squeeze()
                        storage[agent_id]["log_probs_top"][step] = top_dist.log_prob(top_action)
                        storage[agent_id]["log_probs_sub"][step] = sub_dist.log_prob(sub_action)
                        storage[agent_id]["actions_top"][step] = top_action
                        storage[agent_id]["actions_sub"][step] = sub_action
                        storage[agent_id]["dones"][step] = next_done[agent_id]
                        next_hiddens[agent_id] = new_hiddens
                        
                        action_dict[agent_id] = torch.stack([top_action, sub_action], dim=1).cpu().numpy()

            # --- Stepping the Environment ---
            if is_curriculum_phase:
                # For curriculum, rewards/dones are extracted after the single step
                # This part is complex because PettingZoo is not designed for this kind of external loop.
                # A proper implementation would integrate the scripted agent *inside* the environment logic.
                # For now, we make a simplifying assumption that reward is 0 and done is false.
                # This is a limitation of this curriculum approach.
                rewards = {agent_id: [0.0] for agent_id in agent_ids if agent_id not in scripted_agents}
                terminated = {agent_id: [False] for agent_id in agent_ids}
            else:
                next_obs_stacked, rewards, terminated, truncated, infos = envs.step(action_dict)
            
            # Store rewards and dones for all learning agents
            for agent_id in policies.keys(): # Iterate over learning agents
                if not (is_curriculum_phase and agent_id in scripted_agents):
                    storage[agent_id]["rewards"][step] = torch.tensor(rewards[agent_id], device=device)
                    # Done is when an env is terminated OR truncated
                    done_val = np.logical_or(terminated[agent_id], truncated.get(agent_id, False))
                    next_done[agent_id] = torch.tensor(done_val, dtype=torch.float32).to(device)

        # --- PPO Update Phase ---
        with torch.no_grad():
            if is_curriculum_phase:
                # To calculate advantages, we need the value of the next state
                learning_agent_id = "player_0"
                processed_obs_np = preprocess_obs(next_obs_dict[learning_agent_id], num_agents, 0)
                processed_obs = torch.tensor(processed_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                _, _, next_value, _ = policies[learning_agent_id](processed_obs, next_hiddens[learning_agent_id])
                next_value = next_value.reshape(1, -1)
                
                # Compute advantages and update for the single learning agent
                agent_h = resolved_hyperparams[learning_agent_id]
                advantages = torch.zeros_like(storage[learning_agent_id]["rewards"]).to(device)
                last_gae_lambda = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_non_terminal = 1.0 - next_done[learning_agent_id]
                        next_return = next_value
                    else:
                        next_non_terminal = 1.0 - storage[learning_agent_id]["dones"][t + 1]
                        next_return = storage[learning_agent_id]["values"][t + 1]
                    
                    delta = storage[learning_agent_id]["rewards"][t] + agent_h['gamma'] * next_return * next_non_terminal - storage[learning_agent_id]["values"][t]
                    advantages[t] = last_gae_lambda = delta + agent_h['gamma'] * agent_h['gae_lambda'] * next_non_terminal * last_gae_lambda
                returns = advantages + storage[learning_agent_id]["values"]
            else:
                # --- Standard GAE and Advantage Calculation (Parallel) ---
                for i, agent_id in enumerate(agent_ids):
                    agent_obs = torch.tensor(next_obs_stacked[:, i, :], dtype=torch.float32, device=device)
                    _, _, next_value, _ = policies[agent_id](agent_obs, next_hiddens[agent_id])
                    next_value = next_value.reshape(1, -1)
                    
                    agent_h = resolved_hyperparams[agent_id]
                    advantages = torch.zeros_like(storage[agent_id]["rewards"]).to(device)
                    last_gae_lambda = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            next_non_terminal = 1.0 - next_done[agent_id]
                            next_return = next_value
                        else:
                            next_non_terminal = 1.0 - storage[agent_id]["dones"][t + 1]
                            next_return = storage[agent_id]["values"][t + 1]
                        
                        delta = storage[agent_id]["rewards"][t] + agent_h['gamma'] * next_return * next_non_terminal - storage[agent_id]["values"][t]
                        advantages[t] = last_gae_lambda = delta + agent_h['gamma'] * agent_h['gae_lambda'] * next_non_terminal * last_gae_lambda
                    
                    storage[agent_id]["returns"] = advantages + storage[agent_id]["values"]

        # --- Update Policies ---
        for agent_id in policies.keys():
            if is_curriculum_phase and agent_id != "player_0":
                continue # Skip update for scripted and non-learning agents in curriculum
            
            agent_h = resolved_hyperparams[agent_id]
            
            if is_curriculum_phase:
                b_obs = storage[agent_id]["obs"].reshape((-1, total_input_dim))
                b_log_probs_top = storage[agent_id]["log_probs_top"].reshape(-1)
                b_log_probs_sub = storage[agent_id]["log_probs_sub"].reshape(-1)
                b_actions_top = storage[agent_id]["actions_top"].reshape(-1)
                b_actions_sub = storage[agent_id]["actions_sub"].reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = storage[agent_id]["values"].reshape(-1)
            else:
                b_obs = storage[agent_id]["obs"].reshape((-1, total_input_dim))
                b_log_probs_top = storage[agent_id]["log_probs_top"].reshape(-1)
                b_log_probs_sub = storage[agent_id]["log_probs_sub"].reshape(-1)
                b_actions_top = storage[agent_id]["actions_top"].reshape(-1)
                b_actions_sub = storage[agent_id]["actions_sub"].reshape(-1)
                b_advantages = storage[agent_id]["returns"].reshape(-1) # Note: returns are calculated and stored per-agent
                b_returns = storage[agent_id]["returns"].reshape(-1)
                b_values = storage[agent_id]["values"].reshape(-1)

            # Optimizing the policy and value network
            clip_fracs = []
            
            # Get current scheduled hyperparams for this agent
            current_clip_coef = schedule_funcs[agent_id]['clip'](current_step)
            current_entropy_coef = schedule_funcs[agent_id]['entropy'](current_step)

            for epoch in range(agent_h['update_epochs']):
                batch_size = b_obs.shape[0]
                minibatch_size = b_obs.shape[0] // agent_h.get('num_minibatches', args.num_minibatches)
                
                if is_curriculum_phase:
                    idxs = np.arange(batch_size)
                else:
                    idxs = np.random.permutation(batch_size)

                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    minibatch_inds = idxs[start:end]

                    mb_obs = b_obs[minibatch_inds]
                    
                    # Get new values, log probs, and entropy from the policy
                    with torch.cuda.amp.autocast(enabled=args.cuda):
                        top_logits, sub_logits, new_value, _ = policies[agent_id](mb_obs, None) # Hidden state is not used in update
                        
                        top_dist = torch.distributions.Categorical(logits=top_logits)
                        sub_dist = torch.distributions.Categorical(logits=sub_logits)

                        new_log_probs_top = top_dist.log_prob(b_actions_top[minibatch_inds])
                        new_log_probs_sub = sub_dist.log_prob(b_actions_sub[minibatch_inds])
                        
                        new_log_probs = new_log_probs_top + new_log_probs_sub
                        b_log_probs = b_log_probs_top[minibatch_inds] + b_log_probs_sub[minibatch_inds]

                        entropy = (top_dist.entropy() + sub_dist.entropy()).mean()
                        new_value = new_value.view(-1)

                        # Value loss
                        v_loss_unclipped = ((new_value - b_returns[minibatch_inds]) ** 2)
                        if agent_h.get("clip_vloss", False):
                            v_clipped = b_values[minibatch_inds] + torch.clamp(
                                new_value - b_values[minibatch_inds],
                                -current_clip_coef,
                                current_clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[minibatch_inds]) ** 2
                            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        else:
                            v_loss = 0.5 * v_loss_unclipped.mean()

                        # Policy loss
                        mb_advs = b_advantages[minibatch_inds]
                        if agent_h.get("norm_adv", False):
                            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                        
                        log_ratio = new_log_probs - b_log_probs
                        ratio = torch.exp(log_ratio)

                        with torch.no_grad():
                            clip_fracs.append(((ratio - 1.0).abs() > current_clip_coef).float().mean().item())

                        pg_loss1 = -mb_advs * ratio
                        pg_loss2 = -mb_advs * torch.clamp(ratio, 1 - current_clip_coef, 1 + current_clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Total loss
                        loss = pg_loss - current_entropy_coef * entropy + v_loss * agent_h['vf_coef']

                    # Backward pass and optimization step
                    optimizers[agent_id].zero_grad()
                    scalers[agent_id].scale(loss).backward()
                    # Gradient clipping
                    scalers[agent_id].unscale_(optimizers[agent_id])
                    nn.utils.clip_grad_norm_(policies[agent_id].parameters(), agent_h.get("max_grad_norm", 0.5))
                    scalers[agent_id].step(optimizers[agent_id])
                    scalers[agent_id].update()

            # --- Logging ---
            if args.track:
                global_step = update * args.num_steps * args.num_envs
                writer.add_scalar(f"losses/policy_loss_agent_{agent_id}", pg_loss.item(), global_step)
                writer.add_scalar(f"losses/value_loss_agent_{agent_id}", v_loss.item(), global_step)
                writer.add_scalar(f"losses/entropy_agent_{agent_id}", entropy.item(), global_step)
                writer.add_scalar(f"charts/avg_value_agent_{agent_id}", new_value.mean().item(), global_step)
                writer.add_scalar(f"charts/lr_agent_{agent_id}", new_lr, global_step)
                writer.add_scalar(f"charts/clip_coef_agent_{agent_id}", current_clip_coef, global_step)
                writer.add_scalar(f"charts/entropy_coef_agent_{agent_id}", current_entropy_coef, global_step)
        # Log other important metrics once per update, from agent 0
        if args.track:
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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

    envs.close()
    if args.track:
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hyperparameters", type=str, default="rl_agent/hyperparameters.json", help="Path to a JSON file with hyperparameters")
    parser.add_argument("--board-json", type=str, default="/home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json", help="Path to board json")
    parser.add_argument("--num-players", type=int, default=4, help="Number of players")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total number of timesteps for training")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    # PPO hyper-set from plan
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4, help="Number of epochs to update the policy network for")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-schedule", type=str, default="fixed:0.01", help="Entropy coefficient schedule")
    parser.add_argument("--clip-schedule", type=str, default="fixed:0.2", help="Clipping coefficient schedule")
    # Add new args
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=256, help="Number of steps to run in each environment per policy update")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-minibatches", type=int, default=8, help="Number of minibatches to split a batch into")
    parser.add_argument("--curriculum-steps", type=int, default=0, help="Number of timesteps to train with scripted opponents. If 0, disabled.")
    parser.add_argument("--save-path", type=str, default="rl_agent/checkpoints", help="Path to save checkpoints.")
    parser.add_argument("--checkpoint-freq", type=int, default=100, help="Frequency (in updates) to save checkpoints.")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Default optimizer (Adam, AdamW)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="The hidden dimension of the neural network")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate of the optimizer")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the policy network")
    parser.add_argument("--track", action="store_true", help="Enable tensorboard tracking")
    
    args = parser.parse_args()
    train(args) 