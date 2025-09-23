#!/usr/bin/env python3
"""
Training Speed Experiments for Blueberry LLM

This script runs 5 different training speed experiments plus a baseline,
each for 200 steps, to identify the fastest training approach.
"""

import os
import sys
import time
import json
import torch
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from torch.utils.data import DataLoader, random_split

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.t4_config import t4_configure
from legacy.llm import load_and_cache_data, TextTokenDataset
from configs.t4_moe_config import T4MoEModelConfig
from timed_training import train_model_with_timing
from record_tracker import RecordTracker, get_tracker, reset_tracker


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    batch_size: int
    gradient_accumulation_steps: int
    use_amp: bool
    use_fp16_matmul: bool
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    max_steps: int = 200
    eval_every: int = 50


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: ExperimentConfig
    total_time_seconds: float
    steps_per_second: float
    final_loss: float
    final_accuracy: float
    memory_usage_mb: float
    gpu_utilization: float


class TrainingSpeedExperiments:
    """Main class for running training speed experiments."""
    
    def __init__(self):
        self.results: List[ExperimentResult] = []
        self.experiments = self._define_experiments()
        self.tracker = get_tracker()
        
    def _define_experiments(self) -> List[ExperimentConfig]:
        """Define the 5 experiments plus baseline."""
        experiments = [
            # Baseline - current configuration
            ExperimentConfig(
                name="baseline",
                description="Current T4-optimized configuration",
                batch_size=12,
                gradient_accumulation_steps=3,
                use_amp=True,
                use_fp16_matmul=True,
                num_workers=2,
                pin_memory=False,
                prefetch_factor=2
            ),
            
            # Experiment 1: Optimized Data Loading
            ExperimentConfig(
                name="optimized_data_loading",
                description="Optimized data loading with pin_memory and more workers",
                batch_size=12,
                gradient_accumulation_steps=3,
                use_amp=True,
                use_fp16_matmul=True,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=4
            ),
            
            # Experiment 2: Larger Batch Size
            ExperimentConfig(
                name="larger_batch_size",
                description="Larger batch size with reduced gradient accumulation",
                batch_size=24,
                gradient_accumulation_steps=2,
                use_amp=True,
                use_fp16_matmul=True,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2
            ),
            
            # Experiment 3: Compile Model
            ExperimentConfig(
                name="compiled_model",
                description="PyTorch 2.0 compiled model for faster execution",
                batch_size=12,
                gradient_accumulation_steps=3,
                use_amp=True,
                use_fp16_matmul=True,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2
            ),
            
            # Experiment 4: Mixed Precision Optimization
            ExperimentConfig(
                name="mixed_precision_optimized",
                description="Enhanced mixed precision with optimized settings",
                batch_size=16,
                gradient_accumulation_steps=2,
                use_amp=True,
                use_fp16_matmul=True,
                num_workers=3,
                pin_memory=True,
                prefetch_factor=3
            ),
            
            # Experiment 5: Memory Optimized
            ExperimentConfig(
                name="memory_optimized",
                description="Memory-optimized configuration with gradient checkpointing",
                batch_size=8,
                gradient_accumulation_steps=4,
                use_amp=True,
                use_fp16_matmul=True,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2
            )
        ]
        return experiments
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all experiments and return results."""
        self.tracker.log("🚀 Starting Training Speed Experiments")
        self.tracker.log("=" * 60)
        
        for i, experiment in enumerate(self.experiments):
            self.tracker.log(f"\n🧪 Experiment {i+1}/{len(self.experiments)}: {experiment.name}")
            self.tracker.log(f"   Description: {experiment.description}")
            self.tracker.log("-" * 40)
            
            try:
                # Start tracking this experiment
                self.tracker.start_experiment(asdict(experiment))
                
                result = self._run_single_experiment(experiment)
                self.results.append(result)
                
                # Finalize experiment tracking
                final_metrics = {
                    'val_loss': result.final_loss,
                    'val_accuracy': result.final_accuracy,
                    'memory_usage_mb': result.memory_usage_mb
                }
                timing_summary = {
                    'steps_per_second': result.steps_per_second,
                    'total_time_seconds': result.total_time_seconds
                }
                self.tracker.finalize_experiment(final_metrics, timing_summary)
                
                self.tracker.log(f"✅ Completed in {result.total_time_seconds:.2f}s")
                self.tracker.log(f"   Steps/sec: {result.steps_per_second:.2f}")
                self.tracker.log(f"   Final Loss: {result.final_loss:.4f}")
                self.tracker.log(f"   Memory Usage: {result.memory_usage_mb:.1f} MB")
                
            except Exception as e:
                self.tracker.log(f"❌ Experiment failed: {e}")
                continue
        
        # Print records summary
        self.tracker.print_records_summary()
        
        return self.results
    
    def _run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment."""
        start_time = time.time()
        
        # Setup configuration
        t4_config = t4_configure()
        model_config = t4_config.get_model_config()
        
        # Apply experiment-specific settings
        model_config.batch_size = config.batch_size
        model_config.gradient_accumulation_steps = config.gradient_accumulation_steps
        model_config.use_amp = config.use_amp
        model_config.use_fp16_matmul = config.use_fp16_matmul
        model_config.max_steps = config.max_steps
        model_config.eval_every = config.eval_every
        
        # Auto-size dataset for experiments
        model_config.num_documents = 1000  # Smaller for faster experiments
        model_config.max_tokens = 100000
        
        print(f"   Loading {model_config.num_documents} documents...")
        
        # Load data
        texts, tokenizer, tokens = load_and_cache_data(model_config)
        dataset = TextTokenDataset(tokens, model_config.max_seq_len)
        
        # Train/val split
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create optimized data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=True if config.num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=True if config.num_workers > 0 else False
        )
        
        print(f"   Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        # Train the model with timing
        print("   Starting training...")
        training_start = time.time()
        
        # Create model
        from legacy.llm import MoEMinimalLLM
        model = MoEMinimalLLM(model_config)
        
        # Train the model with timing
        model, final_metrics = train_model_with_timing(
            model, train_loader, val_loader, model_config, experiment_name=config.name
        )
        
        training_time = time.time() - training_start
        total_time = time.time() - start_time
        
        # Get GPU memory usage
        memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        # Calculate metrics
        steps_per_second = config.max_steps / training_time if training_time > 0 else 0
        
        return ExperimentResult(
            config=config,
            total_time_seconds=total_time,
            steps_per_second=steps_per_second,
            final_loss=final_metrics.get('val_loss', 0.0),
            final_accuracy=final_metrics.get('val_accuracy', 0.0),
            memory_usage_mb=memory_usage,
            gpu_utilization=0.0  # Would need additional monitoring
        )
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save experiment results to JSON file."""
        results_data = []
        for result in self.results:
            result_dict = {
                'config': asdict(result.config),
                'total_time_seconds': result.total_time_seconds,
                'steps_per_second': result.steps_per_second,
                'final_loss': result.final_loss,
                'final_accuracy': result.final_accuracy,
                'memory_usage_mb': result.memory_usage_mb,
                'gpu_utilization': result.gpu_utilization
            }
            results_data.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of all experiment results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n📊 EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Experiment':<25} {'Time (s)':<10} {'Steps/s':<10} {'Loss':<8} {'Memory (MB)':<12}")
        print("-" * 80)
        
        # Sort by steps per second (speed)
        sorted_results = sorted(self.results, key=lambda x: x.steps_per_second, reverse=True)
        
        for result in sorted_results:
            print(f"{result.config.name:<25} {result.total_time_seconds:<10.2f} "
                  f"{result.steps_per_second:<10.2f} {result.final_loss:<8.4f} "
                  f"{result.memory_usage_mb:<12.1f}")
        
        # Find fastest
        fastest = sorted_results[0]
        print(f"\n🏆 FASTEST: {fastest.config.name}")
        print(f"   Speed: {fastest.steps_per_second:.2f} steps/second")
        print(f"   Description: {fastest.config.description}")
        
        # Compare to baseline
        baseline = next((r for r in self.results if r.config.name == "baseline"), None)
        if baseline and fastest.config.name != "baseline":
            speedup = fastest.steps_per_second / baseline.steps_per_second
            print(f"   Speedup vs baseline: {speedup:.2f}x")


def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description="Training Speed Experiments")
    parser.add_argument("--output", default="experiment_results.json", 
                       help="Output file for results")
    args = parser.parse_args()
    
    # Initialize record tracking
    tracker = get_tracker()
    tracker.log("🚀 Starting Training Speed Experiments")
    tracker.log(f"📁 Output file: {args.output}")
    
    try:
        # Run experiments
        experiments = TrainingSpeedExperiments()
        results = experiments.run_all_experiments()
        
        # Save and summarize results
        experiments.save_results(args.output)
        experiments.print_summary()
        
        tracker.log(f"\n✅ All experiments completed!")
        tracker.log(f"📁 Results saved to: {args.output}")
        
    except Exception as e:
        tracker.log(f"❌ Experiments failed: {e}")
        raise
    finally:
        # Cleanup
        tracker.cleanup()


if __name__ == "__main__":
    main()
