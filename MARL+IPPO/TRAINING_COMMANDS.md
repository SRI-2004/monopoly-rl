# Enhanced Monopoly RL Training Commands

This document provides comprehensive training commands for the Enhanced Monopoly RL system, incorporating **Curriculum Learning**, **Self-Play**, and **Population-Based Training**.

## 🚀 Quick Start

### Option 1: Full Training Pipeline (Recommended)
```bash
# Run all three phases: Curriculum → Self-Play → Population-Based
python3 run_training.py --mode full --experiment-name my_monopoly_experiment
```

### Option 2: Using the Bash Script
```bash
# Run the complete training pipeline
./train_enhanced.sh
```

### Option 3: Quick Test Training
```bash
# Reduced timesteps for testing
python3 run_training.py --mode quick --experiment-name test_run
```

## 📊 Training Phases

### Phase 1: Curriculum Learning (5M steps)
Trains agents against scripted opponents with simplified game mechanics.

```bash
python3 run_training.py --mode curriculum --curriculum-steps 5000000
```

**Direct command:**
```bash
python3 train_ippo.py \
    --hyperparameters enhanced_hyperparameters.json \
    --board-json /home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json \
    --num-players 4 \
    --total-timesteps 5000000 \
    --curriculum-steps 5000000 \
    --max-steps 1000 \
    --num-envs 16 \
    --num-steps 512 \
    --cuda \
    --track \
    --compile \
    --checkpoint-freq 100 \
    --seed 42
```

### Phase 2: Self-Play Training (15M steps)
Self-play with opponent sampling from checkpoint pool.

```bash
python3 run_training.py --mode self-play --self-play-steps 15000000
```

**Direct command:**
```bash
python3 train_ippo.py \
    --hyperparameters enhanced_hyperparameters.json \
    --board-json /home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json \
    --num-players 4 \
    --total-timesteps 15000000 \
    --curriculum-steps 0 \
    --max-steps 2000 \
    --num-envs 32 \
    --num-steps 512 \
    --cuda \
    --track \
    --compile \
    --checkpoint-freq 200 \
    --seed 43
```

### Phase 3: Population-Based Training (5M steps)
Population-based training with diverse strategies and evolutionary selection.

```bash
python3 run_training.py --mode population --population-steps 5000000
```

**Direct command:**
```bash
python3 train_ippo.py \
    --hyperparameters enhanced_hyperparameters.json \
    --board-json /home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json \
    --num-players 4 \
    --total-timesteps 5000000 \
    --curriculum-steps 0 \
    --max-steps 3000 \
    --num-envs 48 \
    --num-steps 512 \
    --cuda \
    --track \
    --compile \
    --checkpoint-freq 150 \
    --seed 44
```

## 🎯 Training Configurations

### High-Performance Training
```bash
python3 run_training.py \
    --mode full \
    --num-envs 64 \
    --total-timesteps 50000000 \
    --experiment-name high_performance_run
```

### Resource-Constrained Training
```bash
python3 run_training.py \
    --mode full \
    --num-envs 8 \
    --total-timesteps 10000000 \
    --max-steps 1500 \
    --experiment-name resource_constrained
```

### Custom Phase Durations
```bash
python3 run_training.py \
    --mode full \
    --curriculum-steps 3000000 \
    --self-play-steps 20000000 \
    --population-steps 7000000 \
    --experiment-name custom_phases
```

## 🔧 Advanced Options

### Custom Hyperparameters
```bash
python3 train_ippo.py \
    --hyperparameters custom_hyperparameters.json \
    --total-timesteps 25000000 \
    --curriculum-steps 5000000 \
    --cuda \
    --track \
    --compile
```

### Specific GPU Configuration
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 run_training.py \
    --mode full \
    --num-envs 48 \
    --experiment-name multi_gpu_training
```

### Resume from Checkpoint
```bash
python3 train_ippo.py \
    --hyperparameters enhanced_hyperparameters.json \
    --load-checkpoint rl_agent/checkpoints/experiment_name/policy_player_0_update_1000.pt \
    --total-timesteps 10000000 \
    --cuda \
    --track
```

## 📈 Monitoring and Evaluation

### Start TensorBoard
```bash
tensorboard --logdir=runs --port=6006 --bind_all
```

### Run Evaluation
```bash
python3 evaluate.py \
    --checkpoint-dir rl_agent/checkpoints/experiment_name \
    --num-games 100 \
    --max-steps 5000 \
    --board-json /home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json \
    --output-file evaluation_results.json
```

## 🧬 Population-Based Training Features

The enhanced training system includes:

- **Diverse Agent Strategies**: 4 different player archetypes with unique biases
- **Evolutionary Selection**: Mutation and crossover of successful strategies
- **Checkpoint Pool Management**: Maintains diverse opponent pool
- **Strategy Diversity Metrics**: Ensures population diversity

### Agent Archetypes:
1. **Aggressive Property Buyer** (`player_0`): High property purchase bias
2. **Conservative Cash Manager** (`player_1`): Risk-averse, cash-focused
3. **Balanced Strategic Player** (`player_2`): Adaptive, well-rounded
4. **High-Risk High-Reward** (`player_3`): Aggressive building and trading

## 📋 Training Schedule

| Phase | Duration | Focus | Key Features |
|-------|----------|-------|--------------|
| **Curriculum** | 5M steps | Basic learning | Scripted opponents, simplified mechanics |
| **Self-Play** | 15M steps | Strategic depth | Opponent sampling, checkpoint pool |
| **Population** | 5M steps | Diversity | Evolutionary strategies, mutation |

**Total Training Time**: 25M steps (~8-12 hours on modern GPU)

## 🎮 Example Training Sessions

### Development/Testing
```bash
python3 run_training.py --mode quick --experiment-name dev_test
```

### Production Training
```bash
python3 run_training.py \
    --mode full \
    --experiment-name production_v1 \
    --num-envs 64 \
    --total-timesteps 50000000
```

### Ablation Study
```bash
# Curriculum only
python3 run_training.py --mode curriculum --experiment-name ablation_curriculum

# Self-play only
python3 run_training.py --mode self-play --experiment-name ablation_selfplay

# Population only
python3 run_training.py --mode population --experiment-name ablation_population
```

## 📊 Expected Outputs

After training completion, you'll find:

```
rl_agent/checkpoints/experiment_name/
├── curriculum/
│   ├── policy_player_0_update_*.pt
│   ├── policy_player_1_update_*.pt
│   └── ...
├── self_play/
│   ├── policy_player_0_update_*.pt
│   └── ...
├── population/
│   ├── policy_player_0_update_*.pt
│   └── ...
├── final_evaluation.json
├── training_summary.json
└── training.log
```

## 🔍 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce `--num-envs` or `--max-steps`
2. **Slow Training**: Enable `--compile` and use multiple GPUs
3. **Checkpoint Loading**: Ensure checkpoint paths are correct
4. **TensorBoard**: Check port 6006 is available

### Performance Tips:

- Use `--compile` for 15-20% speedup
- Increase `--num-envs` for better GPU utilization
- Monitor GPU memory usage with `nvidia-smi`
- Use SSD storage for faster checkpoint I/O

## 🎯 Next Steps

After training completion:

1. **Evaluate Performance**: Run comprehensive evaluation
2. **Analyze Strategies**: Study learned behaviors
3. **Tournament Play**: Pit different checkpoints against each other
4. **Strategy Analysis**: Examine decision patterns and game outcomes

---

**Happy Training! 🎲🤖** 