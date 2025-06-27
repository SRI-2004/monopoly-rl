import torch
import numpy as np
import time
from argparse import ArgumentParser
import json
import os
from tqdm import tqdm

from env_wrapper import MonopolyMAv2
from network import ActorCritic
from scripted_agent import ScriptedAgent
from vectorized_env import SyncVecEnv

def preprocess_obs(obs, device, num_players):
    """
    Flattens and preprocesses the observation dictionary from the environment.
    (Copied from train_ippo.py for consistency)
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


def load_policy(ckpt_path, input_dim, action_dims, device):
    """Loads a policy from a checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Get hidden_dim from the checkpoint, with a fallback for older models
    hidden_dim = checkpoint.get("hidden_dim", 256) # Temporarily change to 256
    
    policy = ActorCritic(input_dim, action_dims, hidden_dim=hidden_dim).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval() # Set to evaluation mode
    return policy

def main(args):
    """
    Main evaluation loop for trained Monopoly agents.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 1. Environment Setup
    # For evaluation, we can run environments in parallel to speed up stat collection.
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
    sample_obs, _ = envs.reset()
    sample_obs_for_one_env = {k: v[0] for k, v in sample_obs[agent_ids[0]].items()}
    processed_sample = preprocess_obs(sample_obs_for_one_env, 'cpu', num_agents)
    total_input_dim = processed_sample.shape[0]
    
    action_dims = [envs.action_space('player_0').nvec[0], envs.action_space('player_0').nvec[1]]

    # 2. Load Policies or Instantiate Scripted Agents
    agents = {}
    ckpt_paths = [args.p0_ckpt, args.p1_ckpt, args.p2_ckpt, args.p3_ckpt]

    # Load board metadata for scripted agents
    with open(args.board_json) as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}

    for i, agent_id in enumerate(agent_ids):
        if i < len(ckpt_paths) and ckpt_paths[i]:
            if ckpt_paths[i].lower() == 'scripted':
                print(f"Loading ScriptedAgent for {agent_id}")
                agents[agent_id] = ScriptedAgent(agent_id, num_agents, board_meta)
            else:
                print(f"Loading policy for {agent_id} from {ckpt_paths[i]}")
                agents[agent_id] = load_policy(ckpt_paths[i], total_input_dim, action_dims, device)
        else:
            print(f"Defaulting to ScriptedAgent for {agent_id}")
            agents[agent_id] = ScriptedAgent(agent_id, num_agents, board_meta)

    # 3. Evaluation Loop
    win_counts = {agent_id: 0 for agent_id in agent_ids}
    games_played = 0

    # Behavioral metrics storage
    behavioral_stats = {
        agent_id: {
            "cash_variances": [],
            "houses_built": 0,
            "trades_initiated": 0,
            "bankruptcies": {"to_bank": 0, "to_player": 0}
        } for agent_id in agent_ids
    }
    
    # Per-game, per-env storage
    cash_histories = [[{agent_id: [] for agent_id in agent_ids}] for _ in range(args.num_envs)]
    
    obs, _ = envs.reset(seed=args.seed)
    
    # Create hidden states with the correct dimensions based on the loaded policies
    hiddens = {}
    for agent_id, agent in agents.items():
        if isinstance(agent, ActorCritic):
            hiddens[agent_id] = torch.zeros(1, args.num_envs, agent.hidden_dim).to(device)
        else:
            # Scripted agents don't need hidden states, but we can create a placeholder
            hiddens[agent_id] = None

    print(f"Starting evaluation for {args.num_games} games...")
    
    pbar = tqdm(total=args.num_games, desc="Evaluating Games")
    while games_played < args.num_games:
        
        env_actions = [{} for _ in range(args.num_envs)]
        with torch.no_grad():
            for agent_id, agent in agents.items():
                agent_obs_dict = obs[agent_id]
                
                if isinstance(agent, ScriptedAgent):
                    for i in range(args.num_envs):
                        single_env_obs = {k: v[i] for k, v in agent_obs_dict.items()}
                        action = agent.get_action(single_env_obs)
                        env_actions[i][agent_id] = action
                else: # It's a learning agent (ActorCritic policy)
                    processed_obs_list = [preprocess_obs({k: v[i] for k, v in agent_obs_dict.items()}, device, num_agents) for i in range(args.num_envs)]
                    processed_obs = torch.stack(processed_obs_list)
                    
                    top_logits, sub_logits, _, hiddens[agent_id] = agent(processed_obs, hiddens[agent_id])
                    
                    # Deterministic actions for evaluation
                    top_action = torch.argmax(top_logits, dim=1)
                    sub_action = torch.argmax(sub_logits, dim=1)

                    for i in range(args.num_envs):
                        env_actions[i][agent_id] = (top_action[i].item(), sub_action[i].item())

        obs, rewards, terminated, truncated, infos = envs.step(env_actions)

        # Check for finished games
        for i in range(args.num_envs):
            if terminated['player_0'][i] or truncated['player_0'][i]: # Done is same for all agents in an env
                games_played += 1
                pbar.update(1)
                
                if games_played >= args.num_games:
                    break
                
                # Semantic features are the same for all agents in an env's info dict.
                # We can grab them from any agent, e.g., player_0.
                if 'player_0' in infos[i]:
                    semantic_info = infos[i]['player_0'].get('semantic_features', {})
                    info_result = infos[i]['player_0'].get('result', '')

                    # Find the winner
                    for agent_id, features in semantic_info.items():
                        if features.get('status') == 'won':
                            win_counts[agent_id] += 1
                            break
                        if features.get('status') == 'lost':
                            # Check bankruptcy cause from the info message
                            if "bankrupt to the bank" in info_result:
                                behavioral_stats[agent_id]['bankruptcies']['to_bank'] += 1
                            elif "bankrupt to" in info_result:
                                behavioral_stats[agent_id]['bankruptcies']['to_player'] += 1
                
                # Process cash variance for the finished game
                for agent_id, cash_history in cash_histories[i][0].items():
                    if cash_history:
                        behavioral_stats[agent_id]['cash_variances'].append(np.var(cash_history))
                
                # Reset storage for this env
                cash_histories[i] = [{agent_id: [] for agent_id in agent_ids}]
                
                # Reset hidden states for this environment
                for agent_id in agent_ids:
                    hiddens[agent_id][:, i, :] = 0

        if games_played >= args.num_games:
            break

        # Track behavioral metrics from infos
        for i in range(args.num_envs):
            # The info dict is structured as {agent_id: agent_info_dict}
            agent_infos = infos[i]

            # Track cash
            for agent_id in agent_ids:
                # Features are duplicated for each agent's info dict, so we can get it from its own key
                agent_semantic_features = agent_infos.get(agent_id, {}).get('semantic_features', {})
                if agent_id in agent_semantic_features:
                    cash_histories[i][0][agent_id].append(agent_semantic_features[agent_id]['cash'])
            
            # Check for specific actions in the info message for the current player
            # The result message is the same for all agents
            current_player_info = agent_infos.get('player_0', {}).get('result', '')
            current_player_idx = envs.envs[i].internal_env.game.current_player_index
            current_player_id = f"player_{current_player_idx}"

            if "improved property" in current_player_info:
                behavioral_stats[current_player_id]['houses_built'] += 1
            if "offered a trade" in current_player_info:
                 behavioral_stats[current_player_id]['trades_initiated'] += 1

    pbar.close()
    # 4. Report Results
    print("\n--- Evaluation Results ---")
    print(f"Total Games Played: {games_played}")
    for agent_id, wins in win_counts.items():
        win_rate = (wins / games_played) * 100 if games_played > 0 else 0
        print(f"  - {agent_id}: {wins} wins ({win_rate:.2f}%)")

    print("\n--- Behavioral Metrics ---")
    for agent_id, stats in behavioral_stats.items():
        avg_cash_variance = np.mean(stats['cash_variances']) if stats['cash_variances'] else 0
        houses_per_100_turns = (stats['houses_built'] / (args.num_games * args.max_steps)) * 10000 # Rough estimate
        
        print(f"\n  Metrics for {agent_id}:")
        print(f"    - Avg. Cash Variance per Game: {avg_cash_variance:.2f}")
        print(f"    - Houses Built (total): {stats['houses_built']}")
        print(f"    - Trades Initiated (total): {stats['trades_initiated']}")
        print(f"    - Bankruptcies (to Bank): {stats['bankruptcies']['to_bank']}")
        print(f"    - Bankruptcies (to Player): {stats['bankruptcies']['to_player']}")

    print("------------------------\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to play for evaluation")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments for evaluation")
    parser.add_argument("--p0-ckpt", type=str, default="scripted", help="Path to checkpoint for player 0, or 'scripted'")
    parser.add_argument("--p1-ckpt", type=str, default="scripted", help="Path to checkpoint for player 1, or 'scripted'")
    parser.add_argument("--p2-ckpt", type=str, default="scripted", help="Path to checkpoint for player 2, or 'scripted'")
    parser.add_argument("--p3-ckpt", type=str, default="scripted", help="Path to checkpoint for player 3, or 'scripted'")
    parser.add_argument("--board-json", type=str, default="/home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json", help="Path to board json")
    parser.add_argument("--num-players", type=int, default=4, help="Number of players")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=101, help="Random seed for evaluation")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    
    args = parser.parse_args()
    main(args) 