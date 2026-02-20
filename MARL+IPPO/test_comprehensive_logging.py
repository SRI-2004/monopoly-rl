#!/usr/bin/env python3
"""
Test script to demonstrate comprehensive logging capabilities.

This script runs a short training session to showcase the enhanced metrics logging
and analysis capabilities that have been added to the Monopoly RL training system.
"""

import torch
import numpy as np
import json
import os
import tempfile
from datetime import datetime

from env_wrapper import MonopolyMAv2
from network import ActorCritic
from scripted_agent import ScriptedAgent

def test_comprehensive_logging():
    """Test the comprehensive logging system with a short training run."""
    print("🧪 Testing Comprehensive Logging System")
    print("="*50)
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Test directory: {temp_dir}")
        
        # Initialize environment
        env = MonopolyMAv2(
            board_json_path="/home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json",
            num_players=4,
            max_steps=1000
        )
        
        # Load board metadata for scripted agents
        with open("/home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json") as f:
            board_data = json.load(f)
        board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}
        
        # Create mixed agent setup: 1 neural network + 3 scripted agents
        device = torch.device("cpu")  # Use CPU for testing
        
        # Get environment dimensions
        sample_obs = env.reset()[0]['player_0']
        from env_wrapper import preprocess_obs
        obs_dim = len(preprocess_obs(sample_obs, 4, 0))
        action_dims = [12, 252]  # Top-level and sub-actions
        
        # Create one neural network agent
        neural_agent = ActorCritic(obs_dim, action_dims, hidden_dim=64).to(device)
        
        # Create scripted agents
        scripted_agents = {
            'player_1': ScriptedAgent(1, 4, board_meta),
            'player_2': ScriptedAgent(2, 4, board_meta),
            'player_3': ScriptedAgent(3, 4, board_meta)
        }
        
        print(f"🤖 Agent setup:")
        print(f"  player_0: Neural Network (ActorCritic)")
        print(f"  player_1-3: Enhanced Scripted Agents")
        
        # Run test episodes
        num_test_episodes = 3
        max_steps_per_episode = 200  # Increase to allow for episode completion
        
        all_metrics = []
        
        for episode in range(num_test_episodes):
            print(f"\n🎮 Episode {episode + 1}/{num_test_episodes}")
            
            obs, info = env.reset()
            episode_metrics = {
                'episode': episode,
                'start_time': datetime.now().isoformat(),
                'steps': [],
                'final_summary': None
            }
            
            # Initialize hidden state for neural agent
            hidden_state = torch.zeros(1, 1, 64).to(device)
            
            for step in range(max_steps_per_episode):
                # Get current agent
                current_agent_idx = env.internal_env.game.current_player_index
                current_agent_id = f"player_{current_agent_idx}"
                
                # Choose action based on agent type
                if current_agent_id == 'player_0':
                    # Neural network agent
                    obs_tensor = torch.tensor(preprocess_obs(obs[current_agent_id], 4, 0), 
                                            dtype=torch.float32, device=device).unsqueeze(0)
                    
                    with torch.no_grad():
                        top_logits, sub_logits, value, hidden_state = neural_agent(obs_tensor, hidden_state)
                        top_dist = torch.distributions.Categorical(logits=top_logits)
                        sub_dist = torch.distributions.Categorical(logits=sub_logits)
                        
                        top_action = top_dist.sample()
                        sub_action = sub_dist.sample()
                        
                        action = (top_action.item(), sub_action.item())
                else:
                    # Scripted agent
                    action = scripted_agents[current_agent_id].get_action(obs[current_agent_id])
                
                # Step environment
                action_dict = {current_agent_id: action}
                obs, rewards, terminated, truncated, info = env.step(action_dict)
                
                # Log step metrics
                step_info = info[current_agent_id]
                step_metrics = {
                    'step': step,
                    'agent': current_agent_id,
                    'action': action,
                    'reward': rewards[current_agent_id],
                    'info_keys': list(step_info.keys()),
                    'has_behavioral_metrics': 'behavioral_metrics' in step_info,
                    'has_economic_state': 'economic_state' in step_info,
                    'has_semantic_features': 'semantic_features' in step_info
                }
                
                # Extract key metrics if available
                if 'behavioral_metrics' in step_info:
                    behavioral = step_info['behavioral_metrics']
                    step_metrics['houses_built'] = behavioral.get('houses_built', 0)
                    step_metrics['properties_purchased'] = behavioral.get('properties_purchased', 0)
                    step_metrics['trades_proposed'] = behavioral.get('trades_proposed', 0)
                    
                if 'economic_state' in step_info:
                    economic = step_info['economic_state']
                    step_metrics['cash'] = economic.get('cash', 0)
                    step_metrics['net_worth'] = economic.get('net_worth', 0)
                    step_metrics['num_properties'] = economic.get('num_properties', 0)
                    step_metrics['num_monopolies'] = economic.get('num_monopolies', 0)
                
                episode_metrics['steps'].append(step_metrics)
                
                # Print interesting events
                if 'result' in step_info and step_info['result']:
                    print(f"  Step {step}: {current_agent_id} -> {step_info['result']}")
                
                # Check for game end
                if terminated[current_agent_id] or truncated[current_agent_id]:
                    print(f"  🏁 Game ended at step {step}")
                    
                    # Extract episode summary if available (check all agents' info)
                    episode_summary = None
                    for agent_id, agent_info in info.items():
                        if 'episode_summary' in agent_info:
                            episode_summary = agent_info['episode_summary']
                            break
                    
                    if episode_summary:
                        episode_metrics['final_summary'] = episode_summary
                        
                        print(f"  📊 Episode Summary:")
                        print(f"    Winner: {episode_summary.get('winner', 'Unknown')}")
                        print(f"    Total steps: {episode_summary.get('total_steps', 0)}")
                        print(f"    Houses built: {episode_summary.get('total_houses_built', 0)}")
                        print(f"    Trades proposed: {episode_summary.get('total_trades_proposed', 0)}")
                        
                        if 'final_net_worths' in episode_summary:
                            print(f"    Final net worths:")
                            for agent, net_worth in episode_summary['final_net_worths'].items():
                                print(f"      {agent}: ${net_worth:,.2f}")
                    else:
                        print(f"  ⚠️ No episode summary found in info")
                    
                    break
                    
            episode_metrics['end_time'] = datetime.now().isoformat()
            episode_metrics['total_steps'] = len(episode_metrics['steps'])
            all_metrics.append(episode_metrics)
        
        # Analyze collected metrics
        print(f"\n📈 METRICS ANALYSIS")
        print("="*50)
        
        total_steps = sum(len(ep['steps']) for ep in all_metrics)
        total_neural_actions = sum(1 for ep in all_metrics for step in ep['steps'] if step['agent'] == 'player_0')
        total_scripted_actions = total_steps - total_neural_actions
        
        print(f"📊 Data Collection Summary:")
        print(f"  Total episodes: {num_test_episodes}")
        print(f"  Total steps logged: {total_steps}")
        print(f"  Neural agent actions: {total_neural_actions}")
        print(f"  Scripted agent actions: {total_scripted_actions}")
        
        # Analyze metric availability
        steps_with_behavioral = sum(1 for ep in all_metrics for step in ep['steps'] if step['has_behavioral_metrics'])
        steps_with_economic = sum(1 for ep in all_metrics for step in ep['steps'] if step['has_economic_state'])
        steps_with_semantic = sum(1 for ep in all_metrics for step in ep['steps'] if step['has_semantic_features'])
        
        print(f"\n📋 Metric Coverage:")
        print(f"  Steps with behavioral metrics: {steps_with_behavioral}/{total_steps} ({100*steps_with_behavioral/total_steps:.1f}%)")
        print(f"  Steps with economic state: {steps_with_economic}/{total_steps} ({100*steps_with_economic/total_steps:.1f}%)")
        print(f"  Steps with semantic features: {steps_with_semantic}/{total_steps} ({100*steps_with_semantic/total_steps:.1f}%)")
        
        # Analyze behavioral patterns
        total_houses = sum(step.get('houses_built', 0) for ep in all_metrics for step in ep['steps'])
        total_properties = sum(step.get('properties_purchased', 0) for ep in all_metrics for step in ep['steps'])
        total_trades = sum(step.get('trades_proposed', 0) for ep in all_metrics for step in ep['steps'])
        
        print(f"\n🏠 Behavioral Insights:")
        print(f"  Total houses built: {total_houses}")
        print(f"  Total properties purchased: {total_properties}")
        print(f"  Total trades proposed: {total_trades}")
        
        # Economic analysis
        final_cash_values = []
        final_net_worths = []
        
        for ep in all_metrics:
            if ep['final_summary'] and 'final_net_worths' in ep['final_summary']:
                net_worths = list(ep['final_summary']['final_net_worths'].values())
                final_net_worths.extend(net_worths)
                
        if final_net_worths:
            print(f"\n💰 Economic Analysis:")
            print(f"  Average final net worth: ${np.mean(final_net_worths):,.2f}")
            print(f"  Net worth std dev: ${np.std(final_net_worths):,.2f}")
            print(f"  Min net worth: ${min(final_net_worths):,.2f}")
            print(f"  Max net worth: ${max(final_net_worths):,.2f}")
        
        # Save test results
        test_results = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'num_episodes': num_test_episodes,
                'max_steps_per_episode': max_steps_per_episode,
                'total_steps_logged': total_steps
            },
            'coverage_stats': {
                'behavioral_metrics_coverage': steps_with_behavioral / total_steps,
                'economic_state_coverage': steps_with_economic / total_steps,
                'semantic_features_coverage': steps_with_semantic / total_steps
            },
            'behavioral_summary': {
                'total_houses_built': total_houses,
                'total_properties_purchased': total_properties,
                'total_trades_proposed': total_trades
            },
            'economic_summary': {
                'avg_final_net_worth': np.mean(final_net_worths) if final_net_worths else 0,
                'net_worth_std': np.std(final_net_worths) if final_net_worths else 0
            },
            'episodes': all_metrics
        }
        
        # Save to temporary file for inspection
        test_file = os.path.join(temp_dir, 'comprehensive_logging_test.json')
        with open(test_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to: {test_file}")
        
        # Demonstrate the logging capabilities
        print(f"\n✅ COMPREHENSIVE LOGGING TEST RESULTS")
        print("="*50)
        
        success_criteria = [
            ("Behavioral metrics logged", steps_with_behavioral > 0),
            ("Economic state tracked", steps_with_economic > 0),
            ("Semantic features available", steps_with_semantic > 0),
            ("Episode summaries generated", any(ep['final_summary'] for ep in all_metrics)),
            ("Multi-agent data collected", total_neural_actions > 0 and total_scripted_actions > 0),
            ("Game events captured", total_properties > 0 or total_trades > 0)
        ]
        
        for criterion, passed in success_criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {criterion}: {status}")
        
        all_passed = all(passed for _, passed in success_criteria)
        
        if all_passed:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"The comprehensive logging system is working correctly.")
            print(f"Training runs will now generate detailed metrics for:")
            print(f"  • Step-level behavioral analysis")
            print(f"  • Economic indicator tracking")
            print(f"  • Multi-agent interaction patterns")
            print(f"  • Episode-level summaries")
            print(f"  • Statistical analysis data")
        else:
            print(f"\n⚠️ Some tests failed. Check the implementation.")
        
        env.close()
        return all_passed

if __name__ == "__main__":
    success = test_comprehensive_logging()
    exit(0 if success else 1) 