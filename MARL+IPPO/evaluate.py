import torch
import numpy as np
import time
from argparse import ArgumentParser
import json
import os
from tqdm import tqdm

from env_wrapper import MonopolyMAv2, preprocess_obs
from network import ActorCritic
from scripted_agent import ScriptedAgent
from parallel_env import MultiProcessingVecEnv

def load_policy(ckpt_path, input_dim, action_dims, device):
    """Loads a policy from a checkpoint, now with robust key checking."""
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Bug Fix #2: Abort if hidden_dim is missing. No safe default exists.
        if "hidden_dim" not in checkpoint:
            raise KeyError("Checkpoint is missing the required 'hidden_dim' key. Cannot determine model architecture.")
        hidden_dim = checkpoint["hidden_dim"]
        
        # Initialize the model with the correct architecture
        model = ActorCritic(input_dim, action_dims, hidden_dim).to(device)
        
        # Bug Fix #1: Support old and new state_dict keys for backward compatibility.
        state_dict = checkpoint.get("policy_state_dict") or checkpoint.get("model_state_dict")
        if not state_dict:
            raise KeyError("Could not find 'policy_state_dict' or 'model_state_dict' in checkpoint.")

        # Handle models saved with torch.compile()
        is_compiled = any(k.startswith('_orig_mod.') for k in state_dict.keys())
        if is_compiled:
            print(f"  [Info] Checkpoint for {ckpt_path.split('/')[-1]} appears to be from a torch.compile'd model. Adjusting keys.")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"FATAL: Error loading checkpoint from {ckpt_path}: {e}")
        # Return None to ensure the calling code handles the failure.
        return None

def run_evaluation_parallel(args, agents, device):
    """Fast-path evaluation for when all agents are neural networks."""
    print("\n[Path Check] All agents are learned policies. Running evaluation in fast, parallel mode.")
    
    def make_env(seed):
        def _f():
            # Bug Fix #9: Properly seed the worker's RNGs.
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = MonopolyMAv2(num_players=args.num_players, max_steps=args.max_steps, board_json_path=args.board_json)
            env.reset(seed=seed)
            return env
        return _f

    envs = MultiProcessingVecEnv([make_env(args.seed + i) for i in range(args.num_envs)], seed=args.seed)
    agent_ids = envs.possible_agents
    num_agents = envs.num_agents

    # --- Data Structures for Detailed Stats ---
    # Per-game stats, aggregated after each game finishes
    final_stats = {
        "game_lengths": [],
        "bankruptcy_margins": [],
    }
    for agent_id in agent_ids:
        final_stats[agent_id] = {
            "total_rewards": [],
            "final_net_worths": [],
            "wins": 0,
            # Behavioral counters
            "properties_purchased": 0,
            "houses_built": 0,
            "jail_fines_paid": 0,
        }

    # Per-step, in-progress data for each parallel environment
    in_progress_stats = [
        {
            "steps": 0,
            "rewards": {agent_id: 0 for agent_id in agent_ids}
        } for _ in range(args.num_envs)
    ]

    hx = {
        agent_id: torch.zeros(1, args.num_envs, agent.hidden_dim).to(device)
        for agent_id, agent in agents.items() if isinstance(agent, ActorCritic)
    }

    games_played = 0
    loop_counter = 0
    
    obs, infos = envs.reset()

    pbar = tqdm(total=args.num_games, desc="Evaluating Games (Parallel)")
    while games_played < args.num_games:
        loop_counter += 1
        # Update in-progress step counters
        for i in range(args.num_envs):
            in_progress_stats[i]["steps"] += 1
        
        if loop_counter % 500 == 0:
            print(f"  [Progress Update] Simulation has run for {loop_counter} parallel steps. Games finished so far: {games_played}/{args.num_games}")

        # --- MODIFICATION: Implement two-step action for pre-roll and post-roll ---
        
        # 1. Pre-roll step: Always attempt to "conclude" to roll the dice.
        # The environment's internal logic will only accept this if the agent is in the 'pre-roll' phase.
        pre_roll_action = np.array([[7, 0]] * args.num_envs) # Action 7 is "conclude"
        pre_roll_action_dict = {agent_id: pre_roll_action for agent_id in agent_ids}
        
        # We only need the next observation from this step. Rewards/dones are handled in the main step.
        obs, _, _, _, _ = envs.step(pre_roll_action_dict)

        # 2. Post-roll step: Get the actual action from the policy using the new observation.
        action_dict = {}
        with torch.no_grad():
            for i, agent_id in enumerate(agent_ids):
                agent = agents[agent_id]
                # The observation from the parallel env is now pre-processed.
                agent_obs = torch.tensor(obs[:, i, :], dtype=torch.float32, device=device)
                
                top_logits, sub_logits, _, hx[agent_id] = agent(agent_obs, hx[agent_id])
                top_action = torch.argmax(top_logits, dim=1)
                sub_action = torch.argmax(sub_logits, dim=1)
                action_dict[agent_id] = torch.stack([top_action, sub_action], dim=1).cpu().numpy()

        obs, rewards, terminated, truncated, infos = envs.step(action_dict)
        
        # Bug Fix #5: Correct reward accumulation.
        for agent_id, r_vec in rewards.items():
            for env_i in range(args.num_envs):
                in_progress_stats[env_i]["rewards"][agent_id] += r_vec[env_i]

        dones = np.logical_or(
            np.any(list(terminated.values()), axis=0),
            np.any(list(truncated.values()), axis=0)
        )

        for i, done in enumerate(dones):
            if done and games_played < args.num_games:
                games_played += 1
                pbar.update(1)
                
                # --- Game has finished: process and store final stats ---
                final_info = infos[i].get('final_info')

                if not final_info:
                    print(f"Warning: A game finished in env {i} but 'final_info' was not found. Skipping analytics for this game.")
                    continue
                
                # Bug Fix #3 & #4: Correctly parse per-agent data from final_info.
                # The info from one agent's perspective contains global data we need.
                first_agent_name = agent_ids[0]
                agent_perspective_info = final_info.get(first_agent_name, {})
                
                game_length = agent_perspective_info.get('step', in_progress_stats[i]["steps"])
                final_stats["game_lengths"].append(game_length)
                
                winner_id = None
                final_net_worths_dict = {}
                semantic_features = agent_perspective_info.get('semantic_features', {})

                for agent_id in agent_ids:
                    # Extract this agent's specific data
                    agent_final_info = final_info.get(agent_id, {})
                    behavior_metrics = agent_final_info.get('behavioral_metrics', {})
                    
                    # Net Worth and core stats
                    final_net_worth = semantic_features.get(agent_id, {}).get('net_worth', 0)
                    final_net_worths_dict[agent_id] = final_net_worth
                    final_stats[agent_id]["total_rewards"].append(in_progress_stats[i]["rewards"][agent_id])
                    final_stats[agent_id]["final_net_worths"].append(final_net_worth)
                    
                    # Behavioral stats (now correctly sourced per-agent)
                    final_stats[agent_id]["properties_purchased"] += behavior_metrics.get("properties_purchased", 0)
                    final_stats[agent_id]["houses_built"] += behavior_metrics.get("houses_built", 0)
                    final_stats[agent_id]["jail_fines_paid"] += behavior_metrics.get("jail_fines_paid", 0)

                    # Check for a natural winner
                    if agent_final_info.get('status') == 'won':
                        winner_id = agent_id
                
                # If no natural winner, it was a truncation; award win to player with highest net worth.
                if winner_id is None and final_net_worths_dict:
                    # Ensure there are actually values to compare
                    if any(final_net_worths_dict.values()):
                         winner_id = max(final_net_worths_dict, key=final_net_worths_dict.get)

                # Now that we have a definitive winner, log stats
                if winner_id:
                    final_stats[winner_id]["wins"] += 1
                    winner_net_worth = final_net_worths_dict.get(winner_id, 0)
                    loser_net_worths = [nw for aid, nw in final_net_worths_dict.items() if aid != winner_id]
                    if loser_net_worths:
                        margin = winner_net_worth - max(loser_net_worths)
                        final_stats["bankruptcy_margins"].append(margin)

                # Reset the in-progress stats for this environment
                in_progress_stats[i] = {
                    "steps": 0,
                    "rewards": {agent_id: 0 for agent_id in agent_ids}
                }
                
                # Reset hidden state for this environment
                for agent_id in hx:
                    hx[agent_id][:, i, :] = 0
        
        if games_played >= args.num_games:
            break
    
    pbar.close()
    envs.close()
    return final_stats, games_played

def run_evaluation_sequential(args, agents, device):
    """Compatibility-path evaluation for when at least one agent is scripted."""
    print("\n[Path Check] Scripted agent detected. Running evaluation in sequential, non-parallel mode (slower).")
    
    env = MonopolyMAv2(num_players=args.num_players, max_steps=args.max_steps, board_json_path=args.board_json)
    agent_ids = env.possible_agents
    num_agents = len(agent_ids)

    win_counts = {agent_id: 0 for agent_id in agent_ids}
    
    pbar = tqdm(total=args.num_games, desc="Evaluating Games (Sequential)")
    for game in range(args.num_games):
        obs, _ = env.reset(seed=args.seed + game)
        hx = {
            agent_id: torch.zeros(1, 1, agent.hidden_dim).to(device) for agent_id, agent in agents.items() if isinstance(agent, ActorCritic)
        }
        
        while True:
            current_agent_id = env.agent_selection
            agent = agents[current_agent_id]
            agent_obs = obs[current_agent_id]
            
            # --- MODIFICATION: Handle pre-roll and post-roll phases ---
            is_neural_agent = not isinstance(agent, ScriptedAgent)
            player_phase = env.game.players[int(current_agent_id.split('_')[-1])].phase

            # If it's a neural agent in the pre-roll phase, it MUST roll the dice.
            # The "conclude" action (index 7) is what triggers the dice roll.
            if is_neural_agent and player_phase == 'pre-roll':
                action = (7, 0) # Action 7 is "conclude" -> roll dice
                obs, rewards, terminated, truncated, info = env.step(action)

                # After rolling, the agent might have won/lost immediately (e.g. from a card)
                if terminated[current_agent_id] or truncated[current_agent_id]:
                    # Game is over, find the winner from the environment's perspective
                    for ag_id in agent_ids:
                        # PettingZoo's env.rewards is a cumulative dict. The winner is the one with a positive final reward.
                        # This is a simplification; a more robust method would be to check agent status.
                        if env.game.players[int(ag_id.split('_')[-1])].status == 'won':
                             win_counts[ag_id] += 1
                             break
                    break # End this game loop

                # The active agent hasn't changed, but we need the new observation for the post-roll action
                agent_obs = obs[current_agent_id]

            # Now, in the post-roll phase (or if it's a scripted agent), decide the actual move.
            with torch.no_grad():
                if isinstance(agent, ScriptedAgent):
                    action = agent.get_action(agent_obs)
                else: # Neural agent in post-roll
                    processed_obs_np = preprocess_obs(agent_obs, num_agents, int(current_agent_id.split('_')[-1]))
                    processed_obs = torch.tensor(processed_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                    top_logits, sub_logits, _, hx[current_agent_id] = agent(processed_obs, hx[current_agent_id])
                    top_action = torch.argmax(top_logits, dim=1)
                    sub_action = torch.argmax(sub_logits, dim=1)
                    action = (top_action.item(), sub_action.item())
            
            obs, rewards, terminated, truncated, info = env.step(action)
            
            if terminated[current_agent_id] or truncated[current_agent_id]:
                # Game is over, find the winner
                for agent_id in agent_ids:
                    if env.game.players[int(agent_id.split('_')[-1])].status == 'won':
                        win_counts[agent_id] += 1
                        break
                break
        pbar.update(1)

    pbar.close()
    env.close()
    return win_counts, args.num_games


def main(args):
    """
    Main evaluation loop for trained Monopoly agents.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # We need a temporary env to get space dimensions for policy loading
    temp_env = MonopolyMAv2(num_players=args.num_players, board_json_path=args.board_json)
    num_agents = len(temp_env.possible_agents)
    sample_obs_dict = temp_env.observation_space("player_0").sample()
    total_input_dim = len(preprocess_obs(sample_obs_dict, num_agents, 0))
    action_dims = temp_env.action_space("player_0").nvec.tolist()
    temp_env.close()
    print(f"Dynamically determined input dim: {total_input_dim}")

    # 2. Load Policies or Instantiate Scripted Agents
    agents = {}
    ckpt_paths = [args.p0_ckpt, args.p1_ckpt, args.p2_ckpt, args.p3_ckpt]
    is_scripted_present = False

    # Load board metadata for scripted agents
    with open(args.board_json) as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}

    agent_ids_list = [f"player_{i}" for i in range(args.num_players)]
    for i, agent_id in enumerate(agent_ids_list):
        if i < len(ckpt_paths) and ckpt_paths[i]:
            if ckpt_paths[i].lower() == 'scripted':
                print(f"Loading ScriptedAgent for {agent_id}")
                agents[agent_id] = ScriptedAgent(agent_id.split('_')[-1], num_agents, board_meta)
                is_scripted_present = True
            else:
                print(f"Loading policy for {agent_id} from {ckpt_paths[i]}")
                policy = load_policy(ckpt_paths[i], total_input_dim, action_dims, device)
                if policy is None:
                    print(f"Failed to load policy for {agent_id}, exiting.")
                    return
                agents[agent_id] = policy
        else:
            print(f"Defaulting to ScriptedAgent for {agent_id}")
            agents[agent_id] = ScriptedAgent(agent_id.split('_')[-1], num_agents, board_meta)
            is_scripted_present = True

    # 3. Run Evaluation
    if is_scripted_present:
        # Note: Detailed stats are not implemented for sequential mode yet.
        # This is a limitation for evaluating against scripted agents.
        win_counts, games_played = run_evaluation_sequential(args, agents, device)
        # Create a mock final_stats for basic win rate reporting
        final_stats = {agent_id: {"wins": wins} for agent_id, wins in win_counts.items()}
    else:
        final_stats, games_played = run_evaluation_parallel(args, agents, device)

    # 4. Aggregate and Print Results
    print("\n--- Evaluation Complete ---")
    print(f"Total Games Played: {games_played}")

    print("\n--- Core Performance ---")
    avg_game_length = np.mean(final_stats.get("game_lengths", [0]))
    print(f"  - Average Game Length: {avg_game_length:.2f} steps")
    for agent_id in agents.keys():
        stats = final_stats.get(agent_id, {})
        wins = stats.get("wins", 0)
        win_rate = (wins / games_played) * 100 if games_played > 0 else 0
        mean_reward = np.mean(stats.get("total_rewards", [0]))
        print(f"  - {agent_id}: {wins} wins ({win_rate:.2f}%) | Avg. Reward: {mean_reward:.2f}")

    print("\n--- Economic Profile ---")
    for agent_id in agents.keys():
        stats = final_stats.get(agent_id, {})
        mean_net_worth = np.mean(stats.get("final_net_worths", [0]))
        print(f"  - {agent_id}: Avg. Final Net Worth: ${mean_net_worth:,.2f}")
    
    avg_bankruptcy_margin = np.mean(final_stats.get("bankruptcy_margins", [0]))
    print(f"  - Average Bankruptcy Margin (for winning games): ${avg_bankruptcy_margin:,.2f}")

    print("\n--- Behavioral Analytics ---")
    for agent_id in agents.keys():
        stats = final_stats.get(agent_id, {})
        print(f"  - {agent_id}:")
        print(f"    - Properties Purchased: {stats.get('properties_purchased', 0)}")
        print(f"    - Houses Built: {stats.get('houses_built', 0)}")
        print(f"    - Jail Fines Paid: {stats.get('jail_fines_paid', 0)}")

    print("------------------------\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to play for evaluation")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments for evaluation")
    parser.add_argument("--p0-ckpt", type=str, default="scripted", help="Path to checkpoint for player 0, or 'scripted'")
    parser.add_argument("--p1-ckpt", type=str, default="scripted", help="Path to checkpoint for player 1, or 'scripted'")
    parser.add_argument("--p2-ckpt", type=str, default="scripted", help="Path to checkpoint for player 2, or 'scripted'")
    parser.add_argument("--p3-ckpt", type=str, default="scripted", help="Path to checkpoint for player 3, or 'scripted'")
    parser.add_argument("--board-json", type=str, default="/home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json", help="Path to board json")
    parser.add_argument("--num-players", type=int, default=4, help="Number of players")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=101, help="Random seed for evaluation")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    
    args = parser.parse_args()
    main(args) 