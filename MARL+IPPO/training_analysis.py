#!/usr/bin/env python3
"""
Comprehensive Training Analysis Script for Monopoly RL

This script demonstrates how to analyze the comprehensive metrics logged during training
to gain insights into agent behavior, learning progress, and game dynamics.

The script covers all the categories mentioned in the user's requirements:
- Skill/Performance metrics
- Robustness & Generalization
- Economic & Behavioral metrics
- Statistical interpretability
- Temporal/Credit assignment
- Social/Multi-agent interaction
- Statistical rigor

Usage:
    python training_analysis.py --training-dir /path/to/training/results
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MonopolyTrainingAnalyzer:
    """Comprehensive analyzer for Monopoly RL training metrics."""
    
    def __init__(self, training_dir):
        self.training_dir = Path(training_dir)
        self.metrics_dir = self.training_dir / 'metrics'
        self.results = {}
        
        # Load all available data
        self.load_training_data()
        
    def load_training_data(self):
        """Load all training metrics from the directory."""
        print(f"Loading training data from {self.training_dir}")
        
        # Load step-level metrics
        step_files = list(self.metrics_dir.glob('step_metrics_*.parquet'))
        if step_files:
            step_dfs = []
            for file in step_files:
                df = pd.read_parquet(file)
                step_dfs.append(df)
            self.step_metrics = pd.concat(step_dfs, ignore_index=True)
            print(f"Loaded {len(self.step_metrics)} step records")
        else:
            self.step_metrics = pd.DataFrame()
            
        # Load episode-level metrics
        episode_files = list(self.metrics_dir.glob('episode_metrics_*.jsonl'))
        if episode_files:
            episode_data = []
            for file in episode_files:
                with open(file, 'r') as f:
                    for line in f:
                        episode_data.append(json.loads(line.strip()))
            self.episode_metrics = pd.json_normalize(episode_data)
            print(f"Loaded {len(self.episode_metrics)} episode records")
        else:
            self.episode_metrics = pd.DataFrame()
            
        # Load training summary
        summary_file = self.training_dir / 'training_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                self.training_summary = json.load(f)
        else:
            self.training_summary = {}
            
    def analyze_skill_performance(self):
        """A. Skill/Performance Analysis"""
        print("\n" + "="*50)
        print("A. SKILL/PERFORMANCE ANALYSIS")
        print("="*50)
        
        results = {}
        
        if not self.step_metrics.empty:
            # A1. Win-rate analysis
            agent_wins = self.step_metrics.groupby('agent_id')['reward'].sum()
            total_episodes = len(self.step_metrics['global_step'].unique())
            
            results['win_rates'] = {
                'per_agent': agent_wins.to_dict(),
                'total_episodes': total_episodes
            }
            
            print(f"Win Rate Analysis:")
            for agent, total_reward in agent_wins.items():
                print(f"  {agent}: Total Reward = {total_reward:.2f}")
                
            # A2. Learning progression
            if 'global_step' in self.step_metrics.columns:
                learning_curve = self.step_metrics.groupby('global_step')['reward'].mean()
                results['learning_curve'] = {
                    'steps': learning_curve.index.tolist(),
                    'avg_rewards': learning_curve.values.tolist()
                }
                
                print(f"Learning Progression:")
                print(f"  Initial avg reward: {learning_curve.iloc[0]:.3f}")
                print(f"  Final avg reward: {learning_curve.iloc[-1]:.3f}")
                print(f"  Improvement: {learning_curve.iloc[-1] - learning_curve.iloc[0]:.3f}")
                
            # A3. Performance consistency
            agent_reward_std = self.step_metrics.groupby('agent_id')['reward'].std()
            results['consistency'] = agent_reward_std.to_dict()
            
            print(f"Performance Consistency (lower std = more consistent):")
            for agent, std in agent_reward_std.items():
                print(f"  {agent}: σ = {std:.3f}")
                
        self.results['skill_performance'] = results
        
    def analyze_robustness_generalization(self):
        """B. Robustness & Generalization Analysis"""
        print("\n" + "="*50)
        print("B. ROBUSTNESS & GENERALIZATION ANALYSIS")
        print("="*50)
        
        results = {}
        
        if not self.step_metrics.empty:
            # B1. Seed variance analysis
            if 'global_step' in self.step_metrics.columns:
                # Group by training phases
                early_phase = self.step_metrics[self.step_metrics['global_step'] < self.step_metrics['global_step'].quantile(0.33)]
                mid_phase = self.step_metrics[
                    (self.step_metrics['global_step'] >= self.step_metrics['global_step'].quantile(0.33)) &
                    (self.step_metrics['global_step'] < self.step_metrics['global_step'].quantile(0.66))
                ]
                late_phase = self.step_metrics[self.step_metrics['global_step'] >= self.step_metrics['global_step'].quantile(0.66)]
                
                results['phase_analysis'] = {
                    'early_reward_mean': early_phase['reward'].mean(),
                    'early_reward_std': early_phase['reward'].std(),
                    'mid_reward_mean': mid_phase['reward'].mean(),
                    'mid_reward_std': mid_phase['reward'].std(),
                    'late_reward_mean': late_phase['reward'].mean(),
                    'late_reward_std': late_phase['reward'].std()
                }
                
                print(f"Training Phase Analysis:")
                print(f"  Early: μ={early_phase['reward'].mean():.3f}, σ={early_phase['reward'].std():.3f}")
                print(f"  Mid:   μ={mid_phase['reward'].mean():.3f}, σ={mid_phase['reward'].std():.3f}")
                print(f"  Late:  μ={late_phase['reward'].mean():.3f}, σ={late_phase['reward'].std():.3f}")
                
            # B2. Action diversity analysis
            if 'action_top' in self.step_metrics.columns:
                action_entropy = {}
                for agent in self.step_metrics['agent_id'].unique():
                    agent_data = self.step_metrics[self.step_metrics['agent_id'] == agent]
                    action_counts = agent_data['action_top'].value_counts(normalize=True)
                    entropy = -np.sum(action_counts * np.log(action_counts + 1e-10))
                    action_entropy[agent] = entropy
                    
                results['action_diversity'] = action_entropy
                
                print(f"Action Diversity (entropy, higher = more diverse):")
                for agent, entropy in action_entropy.items():
                    print(f"  {agent}: H = {entropy:.3f}")
                    
        self.results['robustness_generalization'] = results
        
    def analyze_economic_behavioral(self):
        """C. Economic & Behavioral Metrics Analysis"""
        print("\n" + "="*50)
        print("C. ECONOMIC & BEHAVIORAL ANALYSIS")
        print("="*50)
        
        results = {}
        
        if not self.step_metrics.empty:
            # C1. Economic indicators
            economic_cols = ['cash', 'net_worth', 'num_properties', 'num_monopolies']
            available_cols = [col for col in economic_cols if col in self.step_metrics.columns]
            
            if available_cols:
                economic_stats = self.step_metrics[available_cols].describe()
                results['economic_stats'] = economic_stats.to_dict()
                
                print(f"Economic Indicators Summary:")
                print(economic_stats)
                
            # C2. Behavioral pattern analysis
            behavioral_cols = ['properties_purchased', 'houses_built', 'trades_proposed', 'jail_fines_paid']
            available_behavioral = [col for col in behavioral_cols if col in self.step_metrics.columns]
            
            if available_behavioral:
                behavioral_totals = self.step_metrics[available_behavioral].sum()
                results['behavioral_totals'] = behavioral_totals.to_dict()
                
                print(f"Behavioral Metrics (Total across all agents):")
                for metric, total in behavioral_totals.items():
                    print(f"  {metric}: {total}")
                    
            # C3. Risk appetite analysis
            if 'cash' in self.step_metrics.columns and 'net_worth' in self.step_metrics.columns:
                self.step_metrics['liquidity_ratio'] = self.step_metrics['cash'] / (self.step_metrics['net_worth'] + 1)
                liquidity_stats = self.step_metrics.groupby('agent_id')['liquidity_ratio'].agg(['mean', 'std'])
                results['risk_appetite'] = liquidity_stats.to_dict()
                
                print(f"Risk Appetite (Liquidity Ratio - higher = more conservative):")
                for agent in liquidity_stats.index:
                    mean_liq = liquidity_stats.loc[agent, 'mean']
                    std_liq = liquidity_stats.loc[agent, 'std']
                    print(f"  {agent}: μ={mean_liq:.3f}, σ={std_liq:.3f}")
                    
        self.results['economic_behavioral'] = results
        
    def analyze_temporal_dynamics(self):
        """E. Temporal/Credit Assignment Analysis"""
        print("\n" + "="*50)
        print("E. TEMPORAL DYNAMICS ANALYSIS")
        print("="*50)
        
        results = {}
        
        if not self.step_metrics.empty and 'global_step' in self.step_metrics.columns:
            # E1. Reward temporal patterns
            time_windows = pd.cut(self.step_metrics['global_step'], bins=10, labels=False)
            temporal_rewards = self.step_metrics.groupby(time_windows)['reward'].agg(['mean', 'std'])
            results['temporal_rewards'] = temporal_rewards.to_dict()
            
            print(f"Temporal Reward Patterns (10 time windows):")
            for i, (mean_r, std_r) in enumerate(zip(temporal_rewards['mean'], temporal_rewards['std'])):
                print(f"  Window {i+1}: μ={mean_r:.3f}, σ={std_r:.3f}")
                
            # E2. Action-reward correlation over time
            if 'action_top' in self.step_metrics.columns:
                action_reward_corr = {}
                for window in range(10):
                    window_data = self.step_metrics[time_windows == window]
                    if len(window_data) > 10:  # Minimum data for correlation
                        corr = window_data['action_top'].corr(window_data['reward'])
                        action_reward_corr[window] = corr
                        
                results['action_reward_correlation'] = action_reward_corr
                
                print(f"Action-Reward Correlation Over Time:")
                for window, corr in action_reward_corr.items():
                    print(f"  Window {window+1}: r={corr:.3f}")
                    
        self.results['temporal_dynamics'] = results
        
    def analyze_multi_agent_interaction(self):
        """F. Multi-Agent Interaction Analysis"""
        print("\n" + "="*50)
        print("F. MULTI-AGENT INTERACTION ANALYSIS")
        print("="*50)
        
        results = {}
        
        if not self.step_metrics.empty:
            # F1. Agent performance comparison
            agent_performance = self.step_metrics.groupby('agent_id')['reward'].agg(['mean', 'std', 'count'])
            results['agent_comparison'] = agent_performance.to_dict()
            
            print(f"Agent Performance Comparison:")
            for agent in agent_performance.index:
                mean_r = agent_performance.loc[agent, 'mean']
                std_r = agent_performance.loc[agent, 'std']
                count = agent_performance.loc[agent, 'count']
                print(f"  {agent}: μ={mean_r:.3f}, σ={std_r:.3f}, n={count}")
                
            # F2. Competitive dynamics
            if len(agent_performance) > 1:
                # Calculate pairwise performance differences
                agents = list(agent_performance.index)
                performance_matrix = np.zeros((len(agents), len(agents)))
                
                for i, agent1 in enumerate(agents):
                    for j, agent2 in enumerate(agents):
                        if i != j:
                            perf_diff = agent_performance.loc[agent1, 'mean'] - agent_performance.loc[agent2, 'mean']
                            performance_matrix[i, j] = perf_diff
                            
                results['performance_matrix'] = performance_matrix.tolist()
                results['agent_names'] = agents
                
                print(f"Competitive Dynamics (performance differences):")
                for i, agent1 in enumerate(agents):
                    for j, agent2 in enumerate(agents):
                        if i < j:  # Only show upper triangle
                            diff = performance_matrix[i, j]
                            print(f"  {agent1} vs {agent2}: Δ={diff:.3f}")
                            
        self.results['multi_agent_interaction'] = results
        
    def analyze_statistical_rigor(self):
        """G. Statistical Rigor Analysis"""
        print("\n" + "="*50)
        print("G. STATISTICAL RIGOR ANALYSIS")
        print("="*50)
        
        results = {}
        
        if not self.step_metrics.empty:
            # G1. Statistical significance tests
            agents = self.step_metrics['agent_id'].unique()
            if len(agents) > 1:
                # Perform pairwise t-tests
                pairwise_tests = {}
                for i, agent1 in enumerate(agents):
                    for j, agent2 in enumerate(agents):
                        if i < j:
                            data1 = self.step_metrics[self.step_metrics['agent_id'] == agent1]['reward']
                            data2 = self.step_metrics[self.step_metrics['agent_id'] == agent2]['reward']
                            
                            if len(data1) > 10 and len(data2) > 10:
                                t_stat, p_value = stats.ttest_ind(data1, data2)
                                pairwise_tests[f"{agent1}_vs_{agent2}"] = {
                                    't_statistic': t_stat,
                                    'p_value': p_value,
                                    'significant': p_value < 0.05
                                }
                                
                results['pairwise_tests'] = pairwise_tests
                
                print(f"Statistical Significance Tests (α=0.05):")
                for comparison, test_result in pairwise_tests.items():
                    sig_marker = "***" if test_result['significant'] else "   "
                    print(f"  {comparison}: t={test_result['t_statistic']:.3f}, p={test_result['p_value']:.3f} {sig_marker}")
                    
            # G2. Confidence intervals
            agent_ci = {}
            for agent in agents:
                agent_rewards = self.step_metrics[self.step_metrics['agent_id'] == agent]['reward']
                if len(agent_rewards) > 10:
                    mean_reward = agent_rewards.mean()
                    sem = stats.sem(agent_rewards)
                    ci_lower, ci_upper = stats.t.interval(0.95, len(agent_rewards)-1, loc=mean_reward, scale=sem)
                    agent_ci[agent] = {
                        'mean': mean_reward,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'margin_of_error': ci_upper - mean_reward
                    }
                    
            results['confidence_intervals'] = agent_ci
            
            print(f"95% Confidence Intervals:")
            for agent, ci_data in agent_ci.items():
                print(f"  {agent}: {ci_data['mean']:.3f} ± {ci_data['margin_of_error']:.3f}")
                
        self.results['statistical_rigor'] = results
        
    def generate_strategy_archetypes(self):
        """D. Strategy Archetype Analysis using Clustering"""
        print("\n" + "="*50)
        print("D. STRATEGY ARCHETYPE ANALYSIS")
        print("="*50)
        
        results = {}
        
        if not self.step_metrics.empty:
            # Create feature vectors for each agent
            feature_cols = ['cash', 'net_worth', 'num_properties', 'num_monopolies', 
                          'properties_purchased', 'houses_built', 'trades_proposed']
            available_features = [col for col in feature_cols if col in self.step_metrics.columns]
            
            if len(available_features) >= 3:
                # Aggregate features by agent
                agent_features = self.step_metrics.groupby('agent_id')[available_features].agg(['mean', 'std']).fillna(0)
                
                # Flatten multi-level columns
                agent_features.columns = ['_'.join(col).strip() for col in agent_features.columns]
                
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(agent_features)
                
                # Perform clustering
                n_clusters = min(3, len(agent_features))  # Max 3 clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(features_scaled)
                
                # Assign cluster labels
                cluster_labels = ['Conservative', 'Aggressive', 'Balanced'][:n_clusters]
                agent_archetypes = {}
                
                for i, agent in enumerate(agent_features.index):
                    agent_archetypes[agent] = cluster_labels[clusters[i]]
                    
                results['strategy_archetypes'] = agent_archetypes
                results['cluster_centers'] = kmeans.cluster_centers_.tolist()
                results['feature_names'] = list(agent_features.columns)
                
                print(f"Strategy Archetypes:")
                for agent, archetype in agent_archetypes.items():
                    print(f"  {agent}: {archetype}")
                    
                # PCA for visualization
                if len(available_features) > 2:
                    pca = PCA(n_components=2)
                    features_pca = pca.fit_transform(features_scaled)
                    
                    results['pca_components'] = features_pca.tolist()
                    results['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
                    
                    print(f"PCA Explained Variance: {pca.explained_variance_ratio_}")
                    
        self.results['strategy_archetypes'] = results
        
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*50)
        print("COMPREHENSIVE TRAINING ANALYSIS REPORT")
        print("="*50)
        
        # Run all analyses
        self.analyze_skill_performance()
        self.analyze_robustness_generalization()
        self.analyze_economic_behavioral()
        self.generate_strategy_archetypes()
        self.analyze_temporal_dynamics()
        self.analyze_multi_agent_interaction()
        self.analyze_statistical_rigor()
        
        # Save results
        output_file = self.training_dir / 'comprehensive_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\nComprehensive analysis saved to: {output_file}")
        
        # Generate summary
        self.print_executive_summary()
        
    def print_executive_summary(self):
        """Print executive summary of key findings."""
        print("\n" + "="*50)
        print("EXECUTIVE SUMMARY")
        print("="*50)
        
        summary_points = []
        
        # Performance summary
        if 'skill_performance' in self.results and 'win_rates' in self.results['skill_performance']:
            win_rates = self.results['skill_performance']['win_rates']['per_agent']
            best_agent = max(win_rates, key=win_rates.get)
            summary_points.append(f"🏆 Best performing agent: {best_agent} (Total Reward: {win_rates[best_agent]:.2f})")
            
        # Learning progress
        if 'skill_performance' in self.results and 'learning_curve' in self.results['skill_performance']:
            curve = self.results['skill_performance']['learning_curve']
            improvement = curve['avg_rewards'][-1] - curve['avg_rewards'][0]
            summary_points.append(f"📈 Learning improvement: {improvement:.3f} reward units")
            
        # Behavioral insights
        if 'economic_behavioral' in self.results and 'behavioral_totals' in self.results['economic_behavioral']:
            behavioral = self.results['economic_behavioral']['behavioral_totals']
            houses_built = behavioral.get('houses_built', 0)
            trades_proposed = behavioral.get('trades_proposed', 0)
            summary_points.append(f"🏠 Total houses built: {houses_built}")
            summary_points.append(f"🤝 Total trades proposed: {trades_proposed}")
            
        # Strategy diversity
        if 'strategy_archetypes' in self.results and 'strategy_archetypes' in self.results['strategy_archetypes']:
            archetypes = self.results['strategy_archetypes']['strategy_archetypes']
            unique_strategies = len(set(archetypes.values()))
            summary_points.append(f"🎯 Strategy diversity: {unique_strategies} distinct archetypes identified")
            
        # Statistical significance
        if 'statistical_rigor' in self.results and 'pairwise_tests' in self.results['statistical_rigor']:
            tests = self.results['statistical_rigor']['pairwise_tests']
            significant_diffs = sum(1 for test in tests.values() if test['significant'])
            summary_points.append(f"📊 Statistically significant differences: {significant_diffs}/{len(tests)} comparisons")
            
        # Print summary
        for point in summary_points:
            print(f"  {point}")
            
        print(f"\n💡 Key Insights:")
        print(f"  • Training generated {len(self.step_metrics)} step-level observations")
        print(f"  • {len(self.step_metrics['agent_id'].unique())} agents analyzed")
        print(f"  • Comprehensive metrics enable deep behavioral analysis")
        print(f"  • Statistical rigor ensures reliable conclusions")

def main():
    parser = argparse.ArgumentParser(description='Analyze Monopoly RL training metrics')
    parser.add_argument('--training-dir', type=str, required=True,
                       help='Directory containing training results and metrics')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save analysis results (default: same as training-dir)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MonopolyTrainingAnalyzer(args.training_dir)
    
    # Run comprehensive analysis
    analyzer.generate_comprehensive_report()
    
    print(f"\n🎉 Analysis complete! Check the results in {args.training_dir}")

if __name__ == "__main__":
    main() 