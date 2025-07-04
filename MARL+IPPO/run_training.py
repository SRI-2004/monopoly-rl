#!/usr/bin/env python3
"""
Enhanced Monopoly RL Training Command
Provides easy access to curriculum learning, self-play, and population-based training
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Enhanced Monopoly RL Training")
    
    # Training mode selection
    parser.add_argument("--mode", choices=["full", "curriculum", "self-play", "population", "quick"], 
                       default="full", help="Training mode")
    
    # Configuration options
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Custom experiment name")
    parser.add_argument("--total-timesteps", type=int, default=25000000,
                       help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=32,
                       help="Number of parallel environments")
    parser.add_argument("--max-steps", type=int, default=2000,
                       help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--gpu", action="store_true", default=True,
                       help="Use GPU if available")
    parser.add_argument("--tensorboard", action="store_true", default=True,
                       help="Enable TensorBoard logging")
    parser.add_argument("--compile", action="store_true", default=True,
                       help="Enable torch.compile optimization")
    
    # Phase-specific options
    parser.add_argument("--curriculum-steps", type=int, default=5000000,
                       help="Steps for curriculum learning phase")
    parser.add_argument("--self-play-steps", type=int, default=15000000,
                       help="Steps for self-play phase")
    parser.add_argument("--population-steps", type=int, default=5000000,
                       help="Steps for population-based training phase")
    
    args = parser.parse_args()
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"monopoly_{args.mode}_{timestamp}"
    
    # Base paths
    base_dir = f"rl_agent/checkpoints/{args.experiment_name}"
    hyperparams_file = "enhanced_hyperparameters.json"
    board_json = "/home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json"
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"🚀 Starting Enhanced Monopoly RL Training")
    print(f"📊 Experiment: {args.experiment_name}")
    print(f"🎯 Mode: {args.mode}")
    print(f"📁 Output: {base_dir}")
    print(f"⏱️  Total timesteps: {args.total_timesteps:,}")
    
    # Common arguments
    common_args = [
        f"--hyperparameters {hyperparams_file}",
        f"--board-json {board_json}",
        f"--num-players 4",
        f"--num-steps 512",
        f"--seed {args.seed}",
        f"--save-path {base_dir}",
    ]
    
    if args.gpu:
        common_args.append("--cuda")
    if args.tensorboard:
        common_args.append("--track")
    if args.compile:
        common_args.append("--compile")
    
    common_str = " ".join(common_args)
    
    success = True
    
    if args.mode in ["full", "curriculum"]:
        print(f"\n🎓 Phase 1: Curriculum Learning ({args.curriculum_steps:,} steps)")
        cmd = f"""python3 train_ippo.py {common_str} \
            --total-timesteps {args.curriculum_steps} \
            --curriculum-steps {args.curriculum_steps} \
            --max-steps 1000 \
            --num-envs 16 \
            --checkpoint-freq 100"""
        
        if not run_command(cmd, "Curriculum Learning"):
            success = False
            if args.mode == "full":
                print("❌ Curriculum phase failed, stopping full training")
                sys.exit(1)
    
    if args.mode in ["full", "self-play"] and success:
        print(f"\n🎮 Phase 2: Self-Play Training ({args.self_play_steps:,} steps)")
        
        # Find curriculum checkpoint if available
        curriculum_checkpoint = ""
        if args.mode == "full":
            curriculum_dir = f"{base_dir}/curriculum"
            if os.path.exists(curriculum_dir):
                checkpoints = [f for f in os.listdir(curriculum_dir) if f.startswith("policy_player_0_update_")]
                if checkpoints:
                    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
                    curriculum_checkpoint = f"--load-checkpoint {curriculum_dir}/{latest_checkpoint}"
        
        cmd = f"""python3 train_ippo.py {common_str} \
            --total-timesteps {args.self_play_steps} \
            --curriculum-steps 0 \
            --max-steps 2000 \
            --num-envs 32 \
            --checkpoint-freq 200 \
            --seed {args.seed + 1} \
            {curriculum_checkpoint}"""
        
        if not run_command(cmd, "Self-Play Training"):
            success = False
            if args.mode == "full":
                print("❌ Self-play phase failed, stopping full training")
                sys.exit(1)
    
    if args.mode in ["full", "population"] and success:
        print(f"\n🧬 Phase 3: Population-Based Training ({args.population_steps:,} steps)")
        
        # Find self-play checkpoint if available
        selfplay_checkpoint = ""
        if args.mode == "full":
            selfplay_dir = f"{base_dir}/self_play"
            if os.path.exists(selfplay_dir):
                checkpoints = [f for f in os.listdir(selfplay_dir) if f.startswith("policy_player_0_update_")]
                if checkpoints:
                    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
                    selfplay_checkpoint = f"--load-checkpoint {selfplay_dir}/{latest_checkpoint}"
        
        cmd = f"""python3 train_ippo.py {common_str} \
            --total-timesteps {args.population_steps} \
            --curriculum-steps 0 \
            --max-steps 3000 \
            --num-envs 48 \
            --checkpoint-freq 150 \
            --seed {args.seed + 2} \
            {selfplay_checkpoint}"""
        
        if not run_command(cmd, "Population-Based Training"):
            success = False
    
    if args.mode == "quick":
        print(f"\n⚡ Quick Training Mode (reduced timesteps)")
        cmd = f"""python3 train_ippo.py {common_str} \
            --total-timesteps 1000000 \
            --curriculum-steps 300000 \
            --max-steps 2000 \
            --num-envs 24 \
            --checkpoint-freq 50"""
        
        if not run_command(cmd, "Quick Training"):
            success = False
    
    if success:
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {base_dir}")
        print(f"📊 View training progress: tensorboard --logdir=runs")
        
        # Run final evaluation
        print(f"\n📈 Running final evaluation...")
        eval_cmd = f"""python3 evaluate.py \
            --checkpoint-dir {base_dir} \
            --num-games 50 \
            --max-steps 3000 \
            --board-json {board_json} \
            --output-file {base_dir}/final_evaluation.json"""
        
        run_command(eval_cmd, "Final Evaluation")
        
    else:
        print(f"\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 