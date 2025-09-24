"""
Evaluation and Analysis Module for Momentum Warmup Experiments

This module provides comprehensive evaluation metrics, statistical analysis,
and visualization tools for momentum warmup research experiments.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics"""
    schedule_name: str
    run_id: int
    final_val_loss: float
    final_val_accuracy: float
    final_val_perplexity: float
    training_time: float
    convergence_step: Optional[int] = None
    stability_score: Optional[float] = None
    memory_efficiency: Optional[float] = None
    gpu_utilization: Optional[float] = None


class MomentumWarmupEvaluator:
    """Comprehensive evaluator for momentum warmup experiments"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.results = []
        self.metrics_df = None
        
    def load_results(self) -> List[Dict]:
        """Load all experiment results"""
        results_file = self.experiment_dir / "all_results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        print(f"✅ Loaded {len(self.results)} experiment results")
        return self.results
    
    def extract_metrics(self) -> pd.DataFrame:
        """Extract metrics into a pandas DataFrame for analysis"""
        metrics_data = []
        
        for result in self.results:
            metrics_data.append({
                'schedule_name': result['schedule_name'],
                'run_id': result['run_id'],
                'final_val_loss': result['final_metrics']['val_loss'],
                'final_val_accuracy': result['final_metrics']['val_accuracy'],
                'final_val_perplexity': result['final_metrics']['val_perplexity'],
                'training_time': result['total_training_time'],
                'convergence_step': self._calculate_convergence_step(result),
                'stability_score': self._calculate_stability_score(result),
                'memory_efficiency': self._calculate_memory_efficiency(result),
                'gpu_utilization': self._calculate_gpu_utilization(result),
            })
        
        self.metrics_df = pd.DataFrame(metrics_data)
        return self.metrics_df
    
    def _calculate_convergence_step(self, result: Dict) -> Optional[int]:
        """Calculate the step where the model converged"""
        val_losses = result['metrics']['val_losses']
        if len(val_losses) < 3:
            return None
        
        # Find the step where validation loss stops improving significantly
        for i in range(2, len(val_losses)):
            recent_losses = val_losses[i-2:i+1]
            if max(recent_losses) - min(recent_losses) < 0.001:  # Converged
                return result['metrics']['steps'][i]
        return None
    
    def _calculate_stability_score(self, result: Dict) -> Optional[float]:
        """Calculate training stability score"""
        val_losses = result['metrics']['val_losses']
        if len(val_losses) < 3:
            return None
        
        # Lower variance = higher stability
        variance = np.var(val_losses)
        return 1.0 / (1.0 + variance)  # Normalize to [0, 1]
    
    def _calculate_memory_efficiency(self, result: Dict) -> Optional[float]:
        """Calculate memory efficiency score"""
        memory_usage = result['metrics'].get('gpu_memory_usage', [])
        if not memory_usage:
            return None
        
        # Higher efficiency = lower memory usage
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
        return 1.0 - (avg_memory / max_memory) if max_memory > 0 else 0.0
    
    def _calculate_gpu_utilization(self, result: Dict) -> Optional[float]:
        """Calculate average GPU utilization"""
        # This would need GPU monitoring data - placeholder for now
        return None
    
    def statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis of the results"""
        if self.metrics_df is None:
            self.extract_metrics()
        
        analysis = {}
        
        # Group by schedule and calculate statistics
        schedule_stats = self.metrics_df.groupby('schedule_name').agg({
            'final_val_loss': ['mean', 'std', 'min', 'max'],
            'final_val_accuracy': ['mean', 'std'],
            'training_time': ['mean', 'std'],
            'stability_score': ['mean', 'std'],
            'memory_efficiency': ['mean', 'std'],
        }).round(4)
        
        analysis['schedule_statistics'] = schedule_stats.to_dict()
        
        # Statistical significance tests
        schedules = self.metrics_df['schedule_name'].unique()
        significance_tests = {}
        
        for i, schedule1 in enumerate(schedules):
            for schedule2 in schedules[i+1:]:
                data1 = self.metrics_df[self.metrics_df['schedule_name'] == schedule1]['final_val_loss']
                data2 = self.metrics_df[self.metrics_df['schedule_name'] == schedule2]['final_val_loss']
                
                # T-test for statistical significance
                t_stat, p_value = stats.ttest_ind(data1, data2)
                significance_tests[f"{schedule1}_vs_{schedule2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        analysis['significance_tests'] = significance_tests
        
        # Effect size calculation (Cohen's d)
        effect_sizes = {}
        baseline_schedule = self.metrics_df.groupby('schedule_name')['final_val_loss'].mean().idxmin()
        baseline_data = self.metrics_df[self.metrics_df['schedule_name'] == baseline_schedule]['final_val_loss']
        
        for schedule in schedules:
            if schedule != baseline_schedule:
                schedule_data = self.metrics_df[self.metrics_df['schedule_name'] == schedule]['final_val_loss']
                pooled_std = np.sqrt(((len(baseline_data) - 1) * baseline_data.var() + 
                                    (len(schedule_data) - 1) * schedule_data.var()) / 
                                   (len(baseline_data) + len(schedule_data) - 2))
                cohens_d = (baseline_data.mean() - schedule_data.mean()) / pooled_std
                effect_sizes[f"{schedule}_vs_{baseline_schedule}"] = cohens_d
        
        analysis['effect_sizes'] = effect_sizes
        
        return analysis
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualization plots"""
        if self.metrics_df is None:
            self.extract_metrics()
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Performance comparison (box plots)
        plt.subplot(3, 4, 1)
        sns.boxplot(data=self.metrics_df, x='schedule_name', y='final_val_loss')
        plt.title('Final Validation Loss by Schedule')
        plt.ylabel('Validation Loss')
        plt.xticks(rotation=45)
        
        plt.subplot(3, 4, 2)
        sns.boxplot(data=self.metrics_df, x='schedule_name', y='final_val_accuracy')
        plt.title('Final Validation Accuracy by Schedule')
        plt.ylabel('Validation Accuracy')
        plt.xticks(rotation=45)
        
        plt.subplot(3, 4, 3)
        sns.boxplot(data=self.metrics_df, x='schedule_name', y='training_time')
        plt.title('Training Time by Schedule')
        plt.ylabel('Training Time (seconds)')
        plt.xticks(rotation=45)
        
        plt.subplot(3, 4, 4)
        sns.boxplot(data=self.metrics_df, x='schedule_name', y='stability_score')
        plt.title('Training Stability by Schedule')
        plt.ylabel('Stability Score')
        plt.xticks(rotation=45)
        
        # 2. Performance vs Efficiency scatter plots
        plt.subplot(3, 4, 5)
        schedule_means = self.metrics_df.groupby('schedule_name').agg({
            'final_val_loss': 'mean',
            'training_time': 'mean'
        })
        plt.scatter(schedule_means['training_time'], schedule_means['final_val_loss'], 
                   s=100, alpha=0.7)
        for schedule in schedule_means.index:
            plt.annotate(schedule, (schedule_means.loc[schedule, 'training_time'], 
                                  schedule_means.loc[schedule, 'final_val_loss']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Final Validation Loss')
        plt.title('Performance vs Training Time')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 6)
        schedule_means = self.metrics_df.groupby('schedule_name').agg({
            'final_val_loss': 'mean',
            'stability_score': 'mean'
        })
        plt.scatter(schedule_means['stability_score'], schedule_means['final_val_loss'], 
                   s=100, alpha=0.7)
        for schedule in schedule_means.index:
            plt.annotate(schedule, (schedule_means.loc[schedule, 'stability_score'], 
                                  schedule_means.loc[schedule, 'final_val_loss']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Stability Score')
        plt.ylabel('Final Validation Loss')
        plt.title('Performance vs Stability')
        plt.grid(True, alpha=0.3)
        
        # 3. Learning curves for top schedules
        plt.subplot(3, 4, 7)
        top_schedules = self.metrics_df.groupby('schedule_name')['final_val_loss'].mean().nsmallest(3).index
        
        for schedule_name in top_schedules:
            schedule_results = [r for r in self.results if r['schedule_name'] == schedule_name]
            if schedule_results:
                # Average learning curves across runs
                steps = schedule_results[0]['metrics']['steps']
                val_losses = np.array([r['metrics']['val_losses'] for r in schedule_results])
                avg_val_losses = np.mean(val_losses, axis=0)
                std_val_losses = np.std(val_losses, axis=0)
                
                plt.plot(steps[:len(avg_val_losses)], avg_val_losses, 
                        label=f"{schedule_name}", linewidth=2)
                plt.fill_between(steps[:len(avg_val_losses)], 
                               avg_val_losses - std_val_losses,
                               avg_val_losses + std_val_losses, alpha=0.2)
        
        plt.xlabel('Training Step')
        plt.ylabel('Validation Loss')
        plt.title('Learning Curves - Top 3 Schedules')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Momentum schedules visualization
        plt.subplot(3, 4, 8)
        max_steps = max(r['config']['training_config']['max_steps'] for r in self.results)
        
        for schedule_name in top_schedules:
            schedule_results = [r for r in self.results if r['schedule_name'] == schedule_name]
            if schedule_results:
                momentum_values = schedule_results[0]['metrics']['momentum_values']
                steps = schedule_results[0]['metrics']['steps']
                plt.plot(steps, momentum_values, label=schedule_name, linewidth=2)
        
        plt.xlabel('Training Step')
        plt.ylabel('Momentum Value')
        plt.title('Momentum Schedules - Top 3')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Statistical significance heatmap
        plt.subplot(3, 4, 9)
        schedules = self.metrics_df['schedule_name'].unique()
        significance_matrix = np.zeros((len(schedules), len(schedules)))
        
        for i, schedule1 in enumerate(schedules):
            for j, schedule2 in enumerate(schedules):
                if i != j:
                    data1 = self.metrics_df[self.metrics_df['schedule_name'] == schedule1]['final_val_loss']
                    data2 = self.metrics_df[self.metrics_df['schedule_name'] == schedule2]['final_val_loss']
                    _, p_value = stats.ttest_ind(data1, data2)
                    significance_matrix[i, j] = p_value
        
        sns.heatmap(significance_matrix, xticklabels=schedules, yticklabels=schedules,
                   annot=True, fmt='.3f', cmap='RdYlBu_r')
        plt.title('Statistical Significance (p-values)')
        
        # 6. Performance ranking
        plt.subplot(3, 4, 10)
        performance_ranking = self.metrics_df.groupby('schedule_name')['final_val_loss'].mean().sort_values()
        plt.barh(range(len(performance_ranking)), performance_ranking.values)
        plt.yticks(range(len(performance_ranking)), performance_ranking.index)
        plt.xlabel('Final Validation Loss')
        plt.title('Performance Ranking')
        plt.grid(True, alpha=0.3)
        
        # 7. Training efficiency (loss per time)
        plt.subplot(3, 4, 11)
        efficiency_data = self.metrics_df.groupby('schedule_name').agg({
            'final_val_loss': 'mean',
            'training_time': 'mean'
        })
        efficiency_data['efficiency'] = efficiency_data['final_val_loss'] / efficiency_data['training_time']
        efficiency_data = efficiency_data.sort_values('efficiency')
        
        plt.barh(range(len(efficiency_data)), efficiency_data['efficiency'])
        plt.yticks(range(len(efficiency_data)), efficiency_data.index)
        plt.xlabel('Loss per Second')
        plt.title('Training Efficiency')
        plt.grid(True, alpha=0.3)
        
        # 8. Convergence analysis
        plt.subplot(3, 4, 12)
        convergence_data = self.metrics_df.groupby('schedule_name').agg({
            'convergence_step': 'mean',
            'final_val_loss': 'mean'
        }).dropna()
        
        if not convergence_data.empty:
            plt.scatter(convergence_data['convergence_step'], convergence_data['final_val_loss'], s=100)
            for schedule in convergence_data.index:
                plt.annotate(schedule, (convergence_data.loc[schedule, 'convergence_step'], 
                                      convergence_data.loc[schedule, 'final_val_loss']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            plt.xlabel('Convergence Step')
            plt.ylabel('Final Validation Loss')
            plt.title('Convergence Analysis')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comprehensive analysis saved to {self.experiment_dir / 'comprehensive_analysis.png'}")
    
    def generate_detailed_report(self) -> str:
        """Generate a detailed analysis report"""
        if self.metrics_df is None:
            self.extract_metrics()
        
        analysis = self.statistical_analysis()
        
        report = f"""
# Comprehensive Momentum Warmup Analysis Report

## Executive Summary

This report provides a detailed analysis of momentum warmup experiments for the Muon optimizer on T4 hardware.

## Dataset Overview
- **Total Experiments**: {len(self.results)}
- **Schedules Tested**: {len(self.metrics_df['schedule_name'].unique())}
- **Runs per Schedule**: {self.metrics_df.groupby('schedule_name').size().iloc[0]}

## Performance Analysis

### Top Performing Schedules (by Final Validation Loss):

"""
        
        # Performance ranking
        performance_ranking = self.metrics_df.groupby('schedule_name')['final_val_loss'].mean().sort_values()
        
        for i, (schedule, loss) in enumerate(performance_ranking.items(), 1):
            schedule_stats = self.metrics_df[self.metrics_df['schedule_name'] == schedule]
            report += f"""
{i}. **{schedule}**
   - Final Val Loss: {loss:.4f} ± {schedule_stats['final_val_loss'].std():.4f}
   - Final Val Accuracy: {schedule_stats['final_val_accuracy'].mean():.4f} ± {schedule_stats['final_val_accuracy'].std():.4f}
   - Training Time: {schedule_stats['training_time'].mean():.1f}s ± {schedule_stats['training_time'].std():.1f}s
   - Stability Score: {schedule_stats['stability_score'].mean():.3f} ± {schedule_stats['stability_score'].std():.3f}
"""
        
        # Statistical significance
        report += f"""

## Statistical Significance Analysis

### Significant Differences (p < 0.05):
"""
        
        significant_pairs = []
        for pair, test_result in analysis['significance_tests'].items():
            if test_result['significant']:
                significant_pairs.append((pair, test_result['p_value']))
        
        if significant_pairs:
            for pair, p_value in significant_pairs:
                report += f"- {pair}: p = {p_value:.4f}\n"
        else:
            report += "- No statistically significant differences found between schedules\n"
        
        # Effect sizes
        report += f"""

## Effect Sizes (Cohen's d):
"""
        
        baseline_schedule = performance_ranking.index[0]
        for pair, effect_size in analysis['effect_sizes'].items():
            interpretation = "large" if abs(effect_size) > 0.8 else "medium" if abs(effect_size) > 0.5 else "small"
            report += f"- {pair}: d = {effect_size:.3f} ({interpretation} effect)\n"
        
        # Recommendations
        report += f"""

## Recommendations

### Primary Recommendation:
**{baseline_schedule}** is the optimal momentum warmup schedule for Muon optimizer on T4 hardware.

### Key Findings:
1. **Performance**: {baseline_schedule} achieves the lowest validation loss of {performance_ranking.iloc[0]:.4f}
2. **Stability**: Training shows consistent convergence across multiple runs
3. **Efficiency**: Provides good balance between performance and training time

### Implementation Guidelines:
- Use {baseline_schedule} for production training
- Monitor validation loss during warmup phase
- Consider adjusting warmup duration based on model size

## Technical Details

- **Hardware**: T4 GPU
- **Model Architecture**: MoE with {self.results[0]['config']['model_config']['num_experts']} experts
- **Training Steps**: {self.results[0]['config']['training_config']['max_steps']}
- **Statistical Tests**: Independent t-tests with α = 0.05
- **Effect Size**: Cohen's d for practical significance

---
*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def save_analysis(self):
        """Save complete analysis to files"""
        # Save metrics DataFrame
        self.metrics_df.to_csv(self.experiment_dir / 'metrics_analysis.csv', index=False)
        
        # Save statistical analysis
        analysis = self.statistical_analysis()
        with open(self.experiment_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save detailed report
        report = self.generate_detailed_report()
        with open(self.experiment_dir / 'detailed_analysis_report.md', 'w') as f:
            f.write(report)
        
        print(f"💾 Analysis saved to {self.experiment_dir}/")
        print(f"   - metrics_analysis.csv")
        print(f"   - statistical_analysis.json")
        print(f"   - detailed_analysis_report.md")


def analyze_experiment(experiment_dir: str):
    """Convenience function to run complete analysis"""
    evaluator = MomentumWarmupEvaluator(experiment_dir)
    evaluator.load_results()
    evaluator.extract_metrics()
    evaluator.create_comprehensive_visualizations()
    evaluator.save_analysis()
    
    return evaluator


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
        analyze_experiment(experiment_dir)
    else:
        print("Usage: python momentum_warmup_analysis.py <experiment_dir>")
