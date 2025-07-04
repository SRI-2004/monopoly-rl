import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import pandas as pd
from datetime import datetime

from env_wrapper import MonopolyMAv2, preprocess_obs
from network import ActorCritic
from scripted_agent import ScriptedAgent
from parallel_env import MultiProcessingVecEnv

def schedule_factory(schedule_str, total_timesteps):
    """
    Creates a schedule function from a string specification.
    
    Supported formats:
    - "fixed:VALUE" - constant value
    - "linear:START->END" - linear interpolation from START to END
    - "exponential:START->END" - exponential decay from START to END
    """
    if schedule_str.startswith("fixed:"):
        value = float(schedule_str.split(":")[1])
        def schedule(step):
            return value
        return schedule
    elif schedule_str.startswith("linear:"):
        start_end = schedule_str.split(":")[1]
        start, end = map(float, start_end.split("->"))
        def schedule(step):
            progress = min(step / total_timesteps, 1.0)
            return start + (end - start) * progress
        return schedule
    elif schedule_str.startswith("exponential:"):
        start_end = schedule_str.split(":")[1]
        start, end = map(float, start_end.split("->"))
        def schedule(step):
            progress = min(step / total_timesteps, 1.0)
            return start * (end / start) ** progress
        return schedule
    else:
        # Default to fixed value
        value = float(schedule_str)
        def schedule(step):
            return value
        return schedule

# Global metrics storage for comprehensive logging
training_metrics = {
    'episode_data': [],
    'step_data': [],
    'behavioral_metrics': [],
    'performance_metrics': [],
    'economic_metrics': []
}

def log_step_metrics(step, agent_id, obs, action, reward, info, global_step):
    """Log detailed step-level metrics for comprehensive analysis."""
    step_data = {
        'global_step': global_step,
        'step': step,
        'agent_id': agent_id,
        'timestamp': datetime.now().isoformat(),
        'action_top': action[0] if isinstance(action, tuple) else action,
        'action_sub': action[1] if isinstance(action, tuple) and len(action) > 1 else 0,
        'reward': reward,
        'phase': info.get('phase', 'unknown'),
        'current_player': info.get('current_player', 'unknown'),
        'error': info.get('error', None),
        'result': info.get('result', None)
    }
    
    # Add semantic features if available
    if 'semantic_features' in info and agent_id in info['semantic_features']:
        semantic = info['semantic_features'][agent_id]
        step_data.update({
            'cash': semantic.get('cash', 0),
            'net_worth': semantic.get('net_worth', 0),
            'num_properties': semantic.get('num_properties', 0),
            'num_monopolies': semantic.get('num_monopolies', 0),
            'status': semantic.get('status', 'active')
        })
    
    # Add behavioral metrics if available
    if 'behavioral_metrics' in info:
        behavioral = info['behavioral_metrics']
        step_data.update({
            'properties_purchased': behavioral.get('properties_purchased', 0),
            'houses_built': behavioral.get('houses_built', 0),
            'hotels_built': behavioral.get('hotels_built', 0),
            'trades_proposed': behavioral.get('trades_proposed', 0),
            'trades_accepted': behavioral.get('trades_accepted', 0),
            'jail_fines_paid': behavioral.get('jail_fines_paid', 0),
            'mortgage_actions': behavioral.get('mortgage_actions', 0),
            'bankruptcy_risk': behavioral.get('bankruptcy_risk', 0.0)
        })
    
    training_metrics['step_data'].append(step_data)

def log_episode_metrics(episode, agent_performances, game_length, winner, global_step):
    """Log episode-level metrics for comprehensive analysis."""
    episode_data = {
        'episode': episode,
        'global_step': global_step,
        'timestamp': datetime.now().isoformat(),
        'game_length': game_length,
        'winner': winner,
        'agent_performances': agent_performances
    }
    
    # Calculate episode-level statistics
    total_rewards = sum(perf.get('total_reward', 0) for perf in agent_performances.values())
    avg_reward = total_rewards / len(agent_performances) if agent_performances else 0
    
    episode_data.update({
        'total_rewards': total_rewards,
        'avg_reward': avg_reward,
        'reward_variance': np.var([perf.get('total_reward', 0) for perf in agent_performances.values()]),
        'houses_built_total': sum(perf.get('houses_built', 0) for perf in agent_performances.values()),
        'trades_total': sum(perf.get('trades_proposed', 0) for perf in agent_performances.values()),
        'bankruptcies': sum(1 for perf in agent_performances.values() if perf.get('status') == 'bankrupt')
    })
    
    training_metrics['episode_data'].append(episode_data)

def save_training_metrics(save_path, update):
    """Save training metrics to lightweight format for later analysis."""
    if not training_metrics['step_data'] and not training_metrics['episode_data']:
        return
    
    metrics_dir = os.path.join(save_path, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save step data to parquet for efficient storage
    if training_metrics['step_data']:
        step_df = pd.DataFrame(training_metrics['step_data'])
        step_file = os.path.join(metrics_dir, f'step_metrics_update_{update}.parquet')
        step_df.to_parquet(step_file, index=False)
    
    # Save episode data to JSONL for human readability
    if training_metrics['episode_data']:
        episode_file = os.path.join(metrics_dir, f'episode_metrics_update_{update}.jsonl')
        with open(episode_file, 'w') as f:
            for episode in training_metrics['episode_data']:
                f.write(json.dumps(episode) + '\n')
    
    # Clear metrics after saving to prevent memory issues
    training_metrics['step_data'].clear()
    training_metrics['episode_data'].clear()

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
    # FIX: Use save_path directly instead of creating timestamped subdirectory
    if args.save_path:
        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
        print(f"Checkpoints will be saved to: {save_path}")
    else:
        save_path = None

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
        # For curriculum learning, we'll use a single environment but still use the parallel API
        curriculum_env = MonopolyMAv2(num_players=args.num_players, max_steps=args.max_steps, board_json_path=args.board_json)
        
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
    # Use max of 1 and num_envs to handle both curriculum and parallel modes
    storage_batch_size = max(1, args.num_envs)
    storage = {}
    for agent_id in agent_ids:
        storage[agent_id] = {
            "obs": torch.zeros((args.num_steps, storage_batch_size, total_input_dim)).to(device),
            "actions_top": torch.zeros((args.num_steps, storage_batch_size)).to(device),
            "actions_sub": torch.zeros((args.num_steps, storage_batch_size)).to(device),
            "log_probs_top": torch.zeros((args.num_steps, storage_batch_size)).to(device),
            "log_probs_sub": torch.zeros((args.num_steps, storage_batch_size)).to(device),
            "rewards": torch.zeros((args.num_steps, storage_batch_size)).to(device),
            "dones": torch.zeros((args.num_steps, storage_batch_size)).to(device),
            "values": torch.zeros((args.num_steps, storage_batch_size)).to(device),
            "returns": torch.zeros((args.num_steps, storage_batch_size)).to(device),
        }

    # 4. Training Loop
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.num_steps // args.num_envs
    
    # Initialize environment and hidden states
    if args.curriculum_steps > 0:
        next_obs_dict, _ = curriculum_env.reset()
        next_done = {agent_id: torch.zeros(1).to(device) for agent_id in agent_ids}
        # Hidden states for curriculum (batch size = 1)
        next_hiddens = {agent_id: torch.zeros(1, 1, resolved_hyperparams[agent_id].get("hidden_dim", 128)).to(device) for agent_id in agent_ids}
    else:
        next_obs_stacked, _ = envs.reset()
        next_done = {agent_id: torch.zeros(args.num_envs).to(device) for agent_id in agent_ids}
        # Hidden states for parallel envs (batch size = num_envs)
        next_hiddens = {agent_id: torch.zeros(1, args.num_envs, resolved_hyperparams[agent_id].get("hidden_dim", 128)).to(device) for agent_id in agent_ids}

    for update in tqdm(range(1, num_updates + 1), desc="Training Progress"):
        current_step = update * args.num_steps * args.num_envs
        is_curriculum_phase = current_step < args.curriculum_steps
        
        # Check if we're transitioning from curriculum to parallel mode
        was_curriculum_phase = (current_step - args.num_steps * args.num_envs) < args.curriculum_steps
        transitioning_to_parallel = was_curriculum_phase and not is_curriculum_phase
        
        if transitioning_to_parallel:
            # Initialize parallel environment and reset hidden states
            next_obs_stacked, _ = envs.reset()
            next_done = {agent_id: torch.zeros(args.num_envs).to(device) for agent_id in agent_ids}
            # Resize hidden states for parallel mode
            next_hiddens = {agent_id: torch.zeros(1, args.num_envs, resolved_hyperparams[agent_id].get("hidden_dim", 128)).to(device) for agent_id in agent_ids}

        # Update learning rates
        for agent_id in agent_ids:
            new_lr = schedule_funcs[agent_id]['lr'](current_step)
            optimizers[agent_id].param_groups[0]['lr'] = new_lr

        # Collect rollouts
        for step in range(args.num_steps):
            global_step += 1 * args.num_envs
            
            # --- Acting ---
            action_dict = {}
            
            with torch.no_grad():
                if is_curriculum_phase:
                    # --- Curriculum Acting (Single Environment with Parallel API) ---
                    # Get current player from environment
                    current_player_idx = curriculum_env.internal_env.game.current_player_index
                    current_player_id = f"player_{current_player_idx}"
                    
                    # Create action dict for all agents
                    for agent_id in agent_ids:
                        if agent_id in scripted_agents:
                            # Use scripted agent
                            action = scripted_agents[agent_id].get_action(next_obs_dict[agent_id])
                            action_dict[agent_id] = action
                        else:
                            # Use learning agent policy
                            processed_obs_np = preprocess_obs(next_obs_dict[agent_id], num_agents, int(agent_id.split('_')[-1]))
                            obs_tensor = torch.tensor(processed_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                            
                            top_logits, sub_logits, value, next_hiddens[agent_id] = policies[agent_id](obs_tensor, next_hiddens[agent_id])
                            
                            top_dist = torch.distributions.Categorical(logits=top_logits)
                            sub_dist = torch.distributions.Categorical(logits=sub_logits)
                            
                            top_action = top_dist.sample()
                            sub_action = sub_dist.sample()
                            
                            action_dict[agent_id] = (top_action.item(), sub_action.item())
                            
                            # Store for learning (only for the current learning agent)
                            if agent_id == current_player_id:
                                storage[agent_id]["obs"][step, 0] = obs_tensor.squeeze(0)
                                storage[agent_id]["actions_top"][step, 0] = top_action
                                storage[agent_id]["actions_sub"][step, 0] = sub_action
                                storage[agent_id]["log_probs_top"][step, 0] = top_dist.log_prob(top_action)
                                storage[agent_id]["log_probs_sub"][step, 0] = sub_dist.log_prob(sub_action)
                                storage[agent_id]["values"][step, 0] = value.flatten()
                                storage[agent_id]["dones"][step, 0] = next_done[agent_id]
                    
                    # Step the environment
                    next_obs_dict, rewards, terminated, truncated, info = curriculum_env.step(action_dict)
                    
                    # Log comprehensive metrics for curriculum phase
                    if current_player_id in info:
                        log_step_metrics(step, current_player_id, next_obs_dict[current_player_id], 
                                       action_dict[current_player_id], rewards[current_player_id], 
                                       info[current_player_id], global_step)
                    
                    # Store rewards and update done flags
                    for agent_id in agent_ids:
                        if agent_id == current_player_id and agent_id not in scripted_agents:
                            storage[agent_id]["rewards"][step, 0] = rewards[agent_id]
                        next_done[agent_id] = torch.tensor(float(terminated[agent_id] or truncated[agent_id])).to(device)
                else:
                    # --- Parallel Acting ---
                    for i, agent_id in enumerate(agent_ids):
                        agent_obs = torch.tensor(next_obs_stacked[:, i, :], dtype=torch.float32, device=device)
                        
                        top_logits, sub_logits, value, next_hiddens[agent_id] = policies[agent_id](agent_obs, next_hiddens[agent_id])
                        
                        top_dist = torch.distributions.Categorical(logits=top_logits)
                        sub_dist = torch.distributions.Categorical(logits=sub_logits)
                        
                        top_action = top_dist.sample()
                        sub_action = sub_dist.sample()
                        
                        action_dict[agent_id] = (top_action.cpu().numpy(), sub_action.cpu().numpy())
                        
                        storage[agent_id]["obs"][step] = agent_obs
                        storage[agent_id]["actions_top"][step] = top_action
                        storage[agent_id]["actions_sub"][step] = sub_action
                        storage[agent_id]["log_probs_top"][step] = top_dist.log_prob(top_action)
                        storage[agent_id]["log_probs_sub"][step] = sub_dist.log_prob(sub_action)
                        storage[agent_id]["values"][step] = value.flatten()
                        storage[agent_id]["dones"][step] = next_done[agent_id]
                    
                    # Step the environment
                    next_obs_stacked, rewards, terminated, truncated, info = envs.step(action_dict)
                    
                    # Log comprehensive metrics for parallel phase
                    for agent_id in agent_ids:
                        if agent_id in info:
                            log_step_metrics(step, agent_id, next_obs_stacked, action_dict[agent_id], 
                                           rewards[agent_id], info[agent_id], global_step)
                    
                    # Store rewards and update done flags
                    for agent_id in agent_ids:
                        storage[agent_id]["rewards"][step] = torch.tensor(rewards[agent_id]).to(device)
                        next_done[agent_id] = torch.tensor(terminated[agent_id] | truncated[agent_id], dtype=torch.float32).to(device)

        # --- Advantage Calculation ---
        with torch.no_grad():
            if is_curriculum_phase:
                # --- Curriculum GAE Calculation ---
                for agent_id in agent_ids:
                    if agent_id in scripted_agents:
                        continue
                    
                    # Get the next value for GAE calculation
                    processed_obs_np = preprocess_obs(next_obs_dict[agent_id], num_agents, int(agent_id.split('_')[-1]))
                    obs_tensor = torch.tensor(processed_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                    _, _, next_value, _ = policies[agent_id](obs_tensor, next_hiddens[agent_id])
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
            if is_curriculum_phase and agent_id in scripted_agents:
                continue # Skip update for scripted agents in curriculum
            
            agent_h = resolved_hyperparams[agent_id]
            
            # Determine effective batch size for this agent (curriculum vs parallel)
            effective_batch_size = 1 if is_curriculum_phase else args.num_envs
            
            b_obs = storage[agent_id]["obs"][:, :effective_batch_size].reshape((-1, total_input_dim))
            b_log_probs_top = storage[agent_id]["log_probs_top"][:, :effective_batch_size].reshape(-1)
            b_log_probs_sub = storage[agent_id]["log_probs_sub"][:, :effective_batch_size].reshape(-1)
            b_actions_top = storage[agent_id]["actions_top"][:, :effective_batch_size].reshape(-1)
            b_actions_sub = storage[agent_id]["actions_sub"][:, :effective_batch_size].reshape(-1)
            b_advantages = storage[agent_id]["returns"][:, :effective_batch_size].reshape(-1)
            b_returns = storage[agent_id]["returns"][:, :effective_batch_size].reshape(-1)
            b_values = storage[agent_id]["values"][:, :effective_batch_size].reshape(-1)

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

            # --- Enhanced Logging ---
            if args.track:
                global_step_for_logging = update * args.num_steps * args.num_envs
                writer.add_scalar(f"losses/policy_loss_agent_{agent_id}", pg_loss.item(), global_step_for_logging)
                writer.add_scalar(f"losses/value_loss_agent_{agent_id}", v_loss.item(), global_step_for_logging)
                writer.add_scalar(f"losses/entropy_agent_{agent_id}", entropy.item(), global_step_for_logging)
                writer.add_scalar(f"charts/avg_value_agent_{agent_id}", new_value.mean().item(), global_step_for_logging)
                writer.add_scalar(f"charts/lr_agent_{agent_id}", new_lr, global_step_for_logging)
                writer.add_scalar(f"charts/clip_coef_agent_{agent_id}", current_clip_coef, global_step_for_logging)
                writer.add_scalar(f"charts/entropy_coef_agent_{agent_id}", current_entropy_coef, global_step_for_logging)
                writer.add_scalar(f"charts/clip_frac_agent_{agent_id}", np.mean(clip_fracs), global_step_for_logging)
                
                # Add advantage and return statistics
                writer.add_scalar(f"charts/advantages_mean_agent_{agent_id}", b_advantages.mean().item(), global_step_for_logging)
                writer.add_scalar(f"charts/advantages_std_agent_{agent_id}", b_advantages.std().item(), global_step_for_logging)
                writer.add_scalar(f"charts/returns_mean_agent_{agent_id}", b_returns.mean().item(), global_step_for_logging)
                writer.add_scalar(f"charts/returns_std_agent_{agent_id}", b_returns.std().item(), global_step_for_logging)
                
        # Log other important metrics once per update
        if args.track:
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/update", update, global_step)
            writer.add_scalar("charts/curriculum_phase", int(is_curriculum_phase), global_step)

        # --- Save Checkpoint ---
        if save_path and update % args.checkpoint_freq == 0:
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
                        "hidden_dim": resolved_hyperparams[agent_id].get("hidden_dim"),
                        "training_phase": "curriculum" if is_curriculum_phase else "parallel"
                    }, ckpt_path)
            tqdm.write(f"Checkpoints saved at update {update} to {save_path}")
        
        # --- Save Training Metrics ---
        if save_path and update % (args.checkpoint_freq * 2) == 0:  # Save metrics less frequently
            save_training_metrics(save_path, update)

    # Final checkpoint save
    if save_path:
        final_ckpt_dir = os.path.join(save_path, "final")
        os.makedirs(final_ckpt_dir, exist_ok=True)
        for agent_id in agent_ids:
            if not (is_curriculum_phase and agent_id in scripted_agents):
                ckpt_path = os.path.join(final_ckpt_dir, f"policy_{agent_id}_final.pt")
                torch.save({
                    "policy_state_dict": policies[agent_id].state_dict(),
                    "optimizer_state_dict": optimizers[agent_id].state_dict(),
                    "global_step": global_step,
                    "update": num_updates,
                    "hidden_dim": resolved_hyperparams[agent_id].get("hidden_dim"),
                    "training_phase": "final"
                }, ckpt_path)
        
        # Save final metrics
        save_training_metrics(save_path, num_updates)
        
        # Save training summary
        training_summary = {
            "total_timesteps": args.total_timesteps,
            "num_updates": num_updates,
            "final_global_step": global_step,
            "curriculum_steps": args.curriculum_steps,
            "training_time_seconds": time.time() - start_time,
            "hyperparameters": vars(args),
            "agent_hyperparameters": resolved_hyperparams
        }
        
        with open(os.path.join(save_path, "training_summary.json"), 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"Training completed! Final checkpoints saved to: {final_ckpt_dir}")

    envs.close()
    if args.track:
        writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hyperparameters", type=str, default="rl_agent/hyperparameters.json", help="Path to a JSON file with hyperparameters")
    parser.add_argument("--board-json", type=str, default="/home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json", help="Path to board json")
    parser.add_argument("--num-players", type=int, default=4, help="Number of players")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps per episode")
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