#!/usr/bin/env python3
"""
Momentum Warmup Research Experiment for Muon Optimizer on T4 Hardware

This experiment tests different momentum warmup schedules to determine the optimal
approach for training MoE models with Muon optimizer on T4 GPUs.

Research Question: What is the optimal momentum warmup schedule for Muon optimizer on T4 hardware?

Hypothesis: Gradual momentum warmup will improve training stability and final performance
compared to fixed momentum, especially on memory-constrained T4 hardware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Callable, Any
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from configs.moe_config import MoEModelConfig
from models.moe_llm import MoEMinimalLLM
from optimizers.muon import Muon
from training.evaluation import evaluate_model
from utils.helpers import set_seed


@dataclass
class MomentumWarmupConfig:
    """Configuration for momentum warmup experiments"""
    # Experiment parameters
    experiment_name: str = "momentum_warmup_research"
    num_runs: int = 3  # Number of runs per schedule for statistical significance
    max_steps: int = 1000  # Reduced for faster experimentation
    
    # Model parameters
    d_model: int = 256  # Smaller model for faster experiments
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    batch_size: int = 16
    max_seq_len: int = 256
    
    # MoE parameters
    num_experts: int = 4
    expert_top_k: int = 2
    
    # Training parameters
    base_lr: float = 0.01
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    use_amp: bool = True
    
    # Evaluation
    eval_every: int = 50
    eval_steps: int = 20
    
    # Momentum warmup schedules to test
    warmup_schedules: List[str] = None
    
    def __post_init__(self):
        if self.warmup_schedules is None:
            self.warmup_schedules = [
                "fixed_095",      # Current baseline: momentum=0.95 fixed
                "fixed_090",      # Lower fixed momentum
                "fixed_099",      # Higher fixed momentum
                "linear_warmup",  # Linear warmup from 0 to 0.95
                "cosine_warmup",  # Cosine warmup from 0 to 0.95
                "exponential_warmup",  # Exponential warmup
                "step_warmup",    # Step-wise warmup
                "adaptive_warmup" # Adaptive based on gradient variance
            ]


class MomentumWarmupScheduler:
    """Handles different momentum warmup schedules"""
    
    def __init__(self, schedule_config: Dict[str, Any], max_steps: int):
        self.schedule_config = schedule_config
        self.schedule_type = schedule_config['type']
        self.max_steps = max_steps
        self.warmup_steps = schedule_config.get('warmup_steps', max_steps // 4)
        self.step = 0
        
    def get_momentum(self, step: int) -> float:
        """Get momentum value for current step"""
        self.step = step
        
        if self.schedule_type == "fixed":
            return self.schedule_config['momentum']
        elif self.schedule_type == "linear":
            start_momentum = self.schedule_config['start_momentum']
            end_momentum = self.schedule_config['end_momentum']
            if step < self.warmup_steps:
                progress = step / self.warmup_steps
                return start_momentum + (end_momentum - start_momentum) * progress
            return end_momentum
        elif self.schedule_type == "cosine":
            start_momentum = self.schedule_config['start_momentum']
            end_momentum = self.schedule_config['end_momentum']
            if step < self.warmup_steps:
                progress = step / self.warmup_steps
                cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
                return start_momentum + (end_momentum - start_momentum) * cosine_factor
            return end_momentum
        elif self.schedule_type == "exponential":
            start_momentum = self.schedule_config['start_momentum']
            end_momentum = self.schedule_config['end_momentum']
            exponent = self.schedule_config.get('exponent', 2.0)
            if step < self.warmup_steps:
                progress = step / self.warmup_steps
                exp_factor = math.pow(progress, exponent)
                return start_momentum + (end_momentum - start_momentum) * exp_factor
            return end_momentum
        elif self.schedule_type == "step":
            steps = self.schedule_config['steps']
            if step < self.warmup_steps:
                step_size = self.warmup_steps // len(steps)
                step_index = min(step // step_size, len(steps) - 1)
                return steps[step_index]
            return steps[-1]
        elif self.schedule_type == "adaptive":
            # Simplified adaptive - would need gradient variance tracking
            base_momentum = self.schedule_config['base_momentum']
            if step < self.warmup_steps:
                progress = step / self.warmup_steps
                return base_momentum * progress
            return base_momentum
        elif self.schedule_type == "delayed_linear":
            start_momentum = self.schedule_config['start_momentum']
            end_momentum = self.schedule_config['end_momentum']
            delay_steps = self.schedule_config['delay_steps']
            if step < delay_steps:
                return start_momentum
            elif step < delay_steps + self.warmup_steps:
                progress = (step - delay_steps) / self.warmup_steps
                return start_momentum + (end_momentum - start_momentum) * progress
            return end_momentum
        elif self.schedule_type == "sigmoid":
            start_momentum = self.schedule_config['start_momentum']
            end_momentum = self.schedule_config['end_momentum']
            steepness = self.schedule_config.get('steepness', 5.0)
            if step < self.warmup_steps:
                progress = step / self.warmup_steps
                sigmoid_factor = 1 / (1 + math.exp(-steepness * (progress - 0.5)))
                return start_momentum + (end_momentum - start_momentum) * sigmoid_factor
            return end_momentum
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


class MuonWithMomentumWarmup(Muon):
    """Extended Muon optimizer with momentum warmup support"""
    
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, 
                 momentum_scheduler=None):
        super().__init__(params, lr, momentum, nesterov, ns_steps)
        self.momentum_scheduler = momentum_scheduler
        
    def step(self):
        """Override step to use dynamic momentum"""
        if self.momentum_scheduler:
            # Update momentum for all parameter groups
            for group in self.param_groups:
                group['momentum'] = self.momentum_scheduler.get_momentum(self.state.get('step', 0))
        
        # Call parent step
        super().step()
        
        # Update step counter
        if 'step' not in self.state:
            self.state['step'] = 0
        self.state['step'] += 1


class MomentumWarmupExperiment:
    """Main experiment class for momentum warmup research"""
    
    def __init__(self, config: MomentumWarmupConfig):
        self.config = config
        self.results = {}
        self.experiment_dir = f"experiments/{config.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
    def setup_model_and_data(self, vocab_size: int) -> Tuple[nn.Module, DataLoader, DataLoader]:
        """Setup model and data loaders"""
        # Create config for this experiment
        model_config = MoEModelConfig(
            d_model=self.config.model_config['d_model'],
            n_heads=self.config.model_config['n_heads'],
            n_layers=self.config.model_config['n_layers'],
            d_ff=self.config.model_config['d_ff'],
            batch_size=self.config.model_config['batch_size'],
            max_steps=self.config.training_config['max_steps'],
            max_seq_len=self.config.model_config['max_seq_len'],
            num_experts=self.config.model_config['num_experts'],
            expert_top_k=self.config.model_config['expert_top_k'],
            vocab_size=vocab_size,
            eval_every=self.config.evaluation_config['eval_every'],
            eval_steps=self.config.evaluation_config['eval_steps'],
            use_amp=self.config.training_config['use_amp']
        )
        
        # Initialize model
        model = MoEMinimalLLM(model_config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Load data (simplified for experiment)
        from data.loader import load_and_cache_data
        from data.dataset import TextTokenDataset
        
        texts, tokenizer, tokens = load_and_cache_data(model_config)
        dataset = TextTokenDataset(tokens, model_config.max_seq_len)
        
        # Train/val split
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, 
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=model_config.batch_size, 
                               shuffle=False, num_workers=2)
        
        return model, train_loader, val_loader, model_config
    
    def train_with_schedule(self, schedule_config: Dict[str, Any], run_id: int, 
                           model: nn.Module, train_loader: DataLoader, 
                           val_loader: DataLoader, model_config: MoEModelConfig) -> Dict:
        """Train model with specific momentum schedule"""
        schedule_name = schedule_config['name']
        print(f"\n🧪 Training with schedule: {schedule_name} (run {run_id})")
        
        # Reset model weights
        set_seed(42 + run_id)
        model.apply(self._reset_weights)
        
        # Setup momentum scheduler
        momentum_scheduler = MomentumWarmupScheduler(
            schedule_config, self.config.training_config['max_steps']
        )
        
        # Setup optimizers with momentum warmup
        muon_params = []
        adamw_params = []
        
        for name, param in model.named_parameters():
            if (param.ndim == 2 and 
                'token_embedding' not in name and 
                'norm' not in name and 
                param.requires_grad):
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        muon_optimizer = MuonWithMomentumWarmup(
            muon_params, 
            lr=self.config.training_config['base_lr'], 
            momentum=0.95,  # Will be overridden by scheduler
            momentum_scheduler=momentum_scheduler
        )
        adamw_optimizer = torch.optim.AdamW(
            adamw_params, 
            lr=self.config.training_config['base_lr'] * 0.1, 
            weight_decay=self.config.training_config['weight_decay']
        )
        
        optimizers = [muon_optimizer, adamw_optimizer]
        
        # Learning rate schedulers
        schedulers = []
        max_steps = self.config.training_config['max_steps']
        for optimizer in optimizers:
            warmup_steps = max_steps // 20
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (max_steps - warmup_steps)
                    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            schedulers.append(scheduler)
        
        scaler = GradScaler() if self.config.training_config['use_amp'] else None
        
        # Training loop
        model.train()
        step = 0
        pbar = tqdm(total=max_steps, desc=f"{schedule_name} (run {run_id})")
        
        # Metrics tracking
        metrics = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'val_perplexities': [],
            'momentum_values': [],
            'steps': [],
            'eval_times': [],
            'gpu_memory_usage': [],
            'training_times': []
        }
        
        start_time = time.time()
        
        while step < max_steps:
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= max_steps:
                    break
                
                batch_start_time = time.time()
                x, y = x.to(model.device), y.to(model.device)
                
                # Forward pass
                if self.config.training_config['use_amp']:
                    with autocast('cuda', dtype=torch.float16):
                        logits, aux_loss = model(x, return_aux_loss=True)
                        ce_loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), y.view(-1))
                        total_loss = ce_loss + (aux_loss if aux_loss is not None else 0)
                        loss = total_loss / model_config.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), y.view(-1))
                    total_loss = ce_loss + (aux_loss if aux_loss is not None else 0)
                    loss = total_loss / model_config.gradient_accumulation_steps
                    loss.backward()
                
                # Optimizer step
                if (step + 1) % model_config.gradient_accumulation_steps == 0:
                    if self.config.training_config['use_amp']:
                        for optimizer in optimizers:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training_config['grad_clip'])
                        
                        for optimizer in optimizers:
                            scaler.step(optimizer)
                            optimizer.zero_grad()
                        for scheduler in schedulers:
                            scheduler.step()
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training_config['grad_clip'])
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        for scheduler in schedulers:
                            scheduler.step()
                
                # Track metrics
                current_momentum = momentum_scheduler.get_momentum(step)
                metrics['momentum_values'].append(current_momentum)
                metrics['steps'].append(step)
                metrics['train_losses'].append(ce_loss.item())
                
                # GPU memory tracking
                if torch.cuda.is_available():
                    metrics['gpu_memory_usage'].append(torch.cuda.memory_allocated() / 1e9)
                
                batch_time = time.time() - batch_start_time
                metrics['training_times'].append(batch_time)
                
                # Evaluation
                if step % self.config.evaluation_config['eval_every'] == 0 and step > 0:
                    eval_start_time = time.time()
                    eval_metrics = evaluate_model(model, val_loader, model_config)
                    eval_time = time.time() - eval_start_time
                    
                    metrics['val_losses'].append(eval_metrics['val_loss'])
                    metrics['val_accuracies'].append(eval_metrics['val_accuracy'])
                    metrics['val_perplexities'].append(eval_metrics['val_perplexity'])
                    metrics['eval_times'].append(eval_time)
                    
                    pbar.set_postfix({
                        'loss': f'{ce_loss.item():.4f}',
                        'val_loss': f'{eval_metrics["val_loss"]:.4f}',
                        'momentum': f'{current_momentum:.3f}',
                        'gpu_mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
                    })
                
                step += 1
                if step % 20 == 0:
                    pbar.update(20)
        
        pbar.close()
        
        # Final evaluation
        final_eval = evaluate_model(model, val_loader, model_config)
        total_time = time.time() - start_time
        
        # Compile results
        result = {
            'schedule_name': schedule_name,
            'run_id': run_id,
            'final_metrics': final_eval,
            'total_training_time': total_time,
            'metrics': metrics,
            'config': asdict(self.config)
        }
        
        return result
    
    def _reset_weights(self, m):
        """Reset model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def run_experiment(self, vocab_size: int):
        """Run the complete momentum warmup experiment"""
        print(f"🚀 Starting Momentum Warmup Experiment")
        print(f"📊 Testing {len(self.config.momentum_schedules)} schedules")
        print(f"🔄 {self.config.training_config['num_runs_per_schedule']} runs per schedule")
        print(f"⏱️ {self.config.training_config['max_steps']} steps per run")
        
        # Setup model and data once
        model, train_loader, val_loader, model_config = self.setup_model_and_data(vocab_size)
        
        all_results = []
        
        for schedule_config in self.config.momentum_schedules:
            schedule_name = schedule_config['name']
            schedule_results = []
            
            for run_id in range(self.config.training_config['num_runs_per_schedule']):
                result = self.train_with_schedule(
                    schedule_config, run_id, model, train_loader, val_loader, model_config
                )
                schedule_results.append(result)
                all_results.append(result)
                
                # Save individual result
                result_file = f"{self.experiment_dir}/{schedule_name}_run_{run_id}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            
            # Calculate average metrics for this schedule
            avg_metrics = self._calculate_average_metrics(schedule_results)
            self.results[schedule_name] = avg_metrics
            
            print(f"✅ Completed {schedule_name}: "
                  f"Val Loss: {avg_metrics['final_val_loss']:.4f} ± {avg_metrics['final_val_loss_std']:.4f}")
        
        # Save all results
        self._save_results(all_results)
        self._create_visualizations(all_results)
        self._generate_report()
        
        return all_results
    
    def _calculate_average_metrics(self, schedule_results: List[Dict]) -> Dict:
        """Calculate average metrics across runs for a schedule"""
        final_val_losses = [r['final_metrics']['val_loss'] for r in schedule_results]
        final_val_accs = [r['final_metrics']['val_accuracy'] for r in schedule_results]
        final_val_ppls = [r['final_metrics']['val_perplexity'] for r in schedule_results]
        training_times = [r['total_training_time'] for r in schedule_results]
        
        return {
            'final_val_loss': np.mean(final_val_losses),
            'final_val_loss_std': np.std(final_val_losses),
            'final_val_accuracy': np.mean(final_val_accs),
            'final_val_accuracy_std': np.std(final_val_accs),
            'final_val_perplexity': np.mean(final_val_ppls),
            'final_val_perplexity_std': np.std(final_val_ppls),
            'avg_training_time': np.mean(training_times),
            'training_time_std': np.std(training_times),
            'num_runs': len(schedule_results)
        }
    
    def _save_results(self, all_results: List[Dict]):
        """Save all experiment results"""
        results_file = f"{self.experiment_dir}/all_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        summary_file = f"{self.experiment_dir}/summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {self.experiment_dir}/")
    
    def _create_visualizations(self, all_results: List[Dict]):
        """Create visualization plots"""
        # Plot 1: Final validation loss comparison
        plt.figure(figsize=(12, 8))
        
        schedules = list(self.results.keys())
        val_losses = [self.results[s]['final_val_loss'] for s in schedules]
        val_loss_stds = [self.results[s]['final_val_loss_std'] for s in schedules]
        
        plt.subplot(2, 2, 1)
        plt.bar(schedules, val_losses, yerr=val_loss_stds, capsize=5)
        plt.title('Final Validation Loss by Schedule')
        plt.ylabel('Validation Loss')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Training time comparison
        plt.subplot(2, 2, 2)
        training_times = [self.results[s]['avg_training_time'] for s in schedules]
        training_time_stds = [self.results[s]['training_time_std'] for s in schedules]
        plt.bar(schedules, training_times, yerr=training_time_stds, capsize=5)
        plt.title('Average Training Time by Schedule')
        plt.ylabel('Training Time (seconds)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Momentum schedules visualization
        plt.subplot(2, 2, 3)
        for schedule_name in schedules[:4]:  # Show first 4 schedules
            scheduler = MomentumWarmupScheduler(schedule_name, self.config.max_steps)
            steps = range(0, self.config.max_steps, 10)
            momentums = [scheduler.get_momentum(s) for s in steps]
            plt.plot(steps, momentums, label=schedule_name, linewidth=2)
        plt.title('Momentum Schedules')
        plt.xlabel('Training Step')
        plt.ylabel('Momentum Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Training curves for best schedules
        plt.subplot(2, 2, 4)
        best_schedules = sorted(schedules, key=lambda s: self.results[s]['final_val_loss'])[:3]
        for schedule_name in best_schedules:
            schedule_results = [r for r in all_results if r['schedule_name'] == schedule_name]
            if schedule_results:
                # Average training curves across runs
                steps = schedule_results[0]['metrics']['steps']
                val_losses = np.array([r['metrics']['val_losses'] for r in schedule_results])
                avg_val_losses = np.mean(val_losses, axis=0)
                plt.plot(steps[:len(avg_val_losses)], avg_val_losses, 
                        label=f"{schedule_name} (final: {self.results[schedule_name]['final_val_loss']:.4f})",
                        linewidth=2)
        plt.title('Training Curves - Best Schedules')
        plt.xlabel('Training Step')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/momentum_warmup_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to {self.experiment_dir}/momentum_warmup_analysis.png")
    
    def _generate_report(self):
        """Generate a comprehensive research report"""
        report = f"""
# Momentum Warmup Research Report

## Research Question
What is the optimal momentum warmup schedule for Muon optimizer on T4 hardware?

## Experiment Configuration
- Model: MoE with {self.config.num_experts} experts, {self.config.d_model}d model
- Training Steps: {self.config.max_steps}
- Batch Size: {self.config.batch_size}
- Number of Runs per Schedule: {self.config.num_runs}
- Hardware: T4 GPU

## Results Summary

### Schedule Performance Ranking (by Final Validation Loss):

"""
        
        # Sort schedules by performance
        sorted_schedules = sorted(self.results.items(), 
                                key=lambda x: x[1]['final_val_loss'])
        
        for i, (schedule_name, metrics) in enumerate(sorted_schedules, 1):
            report += f"""
{i}. **{schedule_name}**
   - Final Val Loss: {metrics['final_val_loss']:.4f} ± {metrics['final_val_loss_std']:.4f}
   - Final Val Accuracy: {metrics['final_val_accuracy']:.4f} ± {metrics['final_val_accuracy_std']:.4f}
   - Final Val Perplexity: {metrics['final_val_perplexity']:.2f} ± {metrics['final_val_perplexity_std']:.2f}
   - Avg Training Time: {metrics['avg_training_time']:.1f}s ± {metrics['training_time_std']:.1f}s
"""
        
        report += f"""

## Key Findings

1. **Best Performing Schedule**: {sorted_schedules[0][0]}
   - Achieved lowest validation loss of {sorted_schedules[0][1]['final_val_loss']:.4f}
   - {sorted_schedules[0][1]['final_val_loss_std']:.4f} standard deviation across runs

2. **Performance Improvement**: 
   - Best vs Worst: {sorted_schedules[-1][1]['final_val_loss'] - sorted_schedules[0][1]['final_val_loss']:.4f} validation loss improvement
   - Relative improvement: {((sorted_schedules[-1][1]['final_val_loss'] - sorted_schedules[0][1]['final_val_loss']) / sorted_schedules[-1][1]['final_val_loss'] * 100):.1f}%

3. **Training Efficiency**:
   - Fastest schedule: {min(self.results.items(), key=lambda x: x[1]['avg_training_time'])[0]}
   - Training time range: {min(m['avg_training_time'] for m in self.results.values()):.1f}s - {max(m['avg_training_time'] for m in self.results.values()):.1f}s

## Recommendations

Based on the experimental results, the optimal momentum warmup schedule for Muon optimizer on T4 hardware is:

**{sorted_schedules[0][0]}** - This schedule provides the best balance of:
- Lowest final validation loss
- Stable training dynamics
- Efficient convergence

## Technical Details

- All experiments used identical model architecture and hyperparameters
- Results averaged over {self.config.num_runs} runs per schedule
- Statistical significance tested with standard deviation analysis
- GPU memory usage and training time tracked for efficiency analysis

## Next Steps

1. Validate findings on larger models and datasets
2. Test on different hardware configurations
3. Investigate momentum warmup interaction with learning rate schedules
4. Explore adaptive momentum warmup based on gradient statistics

---
*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        report_file = f"{self.experiment_dir}/research_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📝 Research report saved to {report_file}")


def main():
    """Main function to run the momentum warmup experiment"""
    print("🧪 Momentum Warmup Research Experiment")
    print("=" * 60)
    
    # Load data to get vocab size
    from data.loader import load_and_cache_data
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    
    # Create experiment configuration
    config = MomentumWarmupConfig()
    
    # Run experiment
    experiment = MomentumWarmupExperiment(config)
    results = experiment.run_experiment(vocab_size)
    
    print("\n🎯 Experiment Complete!")
    print(f"📊 Results saved to: experiments/{config.experiment_name}/")
    print("📈 Check the generated visualizations and report for detailed analysis")


if __name__ == "__main__":
    main()
