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
    """Loads a policy from a checkpoint."""
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Get hidden_dim from the checkpoint, with a fallback for older models
        hidden_dim = checkpoint.get("hidden_dim", 256) # Temporarily change to 256
        
        # Now we can initialize the model with the correct dimensions
        model = ActorCritic(input_dim, action_dims, hidden_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading checkpoint from {ckpt_path}: {e}")
        return None

def run_evaluation_parallel(args, agents, device):
    """Fast-path evaluation for when all agents are neural networks."""
    print("\nRunning evaluation in fast, parallel mode.")
    
    def make_env(seed):
        def _f():
            env = MonopolyMAv2(num_players=args.num_players, max_steps=args.max_steps, board_json_path=args.board_json)
            env.reset(seed=seed)
            return env
        return _f

    envs = MultiProcessingVecEnv([make_env(args.seed + i) for i in range(args.num_envs)], seed=args.seed)
    agent_ids = envs.possible_agents
    num_agents = envs.num_agents

    hx = {
        agent_id: torch.zeros(1, args.num_envs, agent.hidden_dim).to(device) for agent_id, agent in agents.items() if isinstance(agent, ActorCritic)
    }

    win_counts = {agent_id: 0 for agent_id in agent_ids}
    games_played = 0
    
    obs, _ = envs.reset()

    pbar = tqdm(total=args.num_games, desc="Evaluating Games (Parallel)")
    while games_played < args.num_games:
        action_dict = {}
        with torch.no_grad():
            for i, agent_id in enumerate(agent_ids):
                agent = agents[agent_id]
                agent_obs = torch.tensor(obs[:, i, :], dtype=torch.float32, device=device)
                
                top_logits, sub_logits, _, hx[agent_id] = agent(agent_obs, hx[agent_id])
                top_action = torch.argmax(top_logits, dim=1)
                sub_action = torch.argmax(sub_logits, dim=1)
                action_dict[agent_id] = torch.stack([top_action, sub_action], dim=1).cpu().numpy()

        obs, rewards, terminated, truncated, infos = envs.step(action_dict)
        
        dones = np.logical_or(
            np.any(list(terminated.values()), axis=0),
            np.any(list(truncated.values()), axis=0)
        )

        for i, done in enumerate(dones):
            if done:
                games_played += 1
                pbar.update(1)
                final_info = infos[i].get('final_info', {})
                for agent_id, agent_info in final_info.items():
                    if agent_info.get('status') == 'won':
                        win_counts[agent_id] += 1
                
                for agent_id in hx:
                    hx[agent_id][:, i, :] = 0
        
        if games_played >= args.num_games:
            break
    
    pbar.close()
    envs.close()
    return win_counts, games_played

def run_evaluation_sequential(args, agents, device):
    """Compatibility-path evaluation for when at least one agent is scripted."""
    print("\nScripted agent detected. Running evaluation in sequential mode (slower).")
    
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

            with torch.no_grad():
                if isinstance(agent, ScriptedAgent):
                    action = agent.get_action(agent_obs)
                else:
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
        win_counts, games_played = run_evaluation_sequential(args, agents, device)
    else:
        win_counts, games_played = run_evaluation_parallel(args, agents, device)

    # 4. Aggregate and Print Results
    print("\n--- Evaluation Complete ---")
    print(f"Total Games Played: {games_played}")
    for agent_id, wins in win_counts.items():
        win_rate = (wins / games_played) * 100 if games_played > 0 else 0
        print(f"  - {agent_id}: {wins} wins ({win_rate:.2f}%)")
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