#!/usr/bin/env python3
"""
Simple Training Speed Challenge: Baseline vs Memory Optimized

This script runs a focused speedrun challenge between baseline and memory-optimized
configurations to establish performance measurement standards.
"""

import os
import sys
import time
import json
import torch
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
from torch.utils.data import DataLoader, random_split

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.t4_config import t4_configure
from legacy.llm import load_and_cache_data, TextTokenDataset, MoEMinimalLLM
from configs.t4_moe_config import T4MoEModelConfig
from timed_training import train_moe_model_timed
from record_tracker import get_tracker


@dataclass
class SpeedrunConfig:
    """Configuration for a speedrun experiment."""
    name: str
    description: str
    batch_size: int
    gradient_accumulation_steps: int
    use_amp: bool
    use_fp16_matmul: bool
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    max_steps: int = 50
    eval_every: int = 25
    eval_steps: int = 15  # Limit validation batches for speed


@dataclass
class SpeedrunResult:
    """Results from a speedrun experiment."""
    config: SpeedrunConfig
    total_time_seconds: float
    steps_per_second: float
    final_loss: float
    final_accuracy: float
    memory_usage_mb: float
    avg_step_time_ms: float
    peak_memory_mb: float


class TrainingSpeedrunChallenge:
    """Minimal training speedrun challenge."""
    
    def __init__(self):
        self.results: List[SpeedrunResult] = []
        self.configs = self._define_speedrun_configs()
        self.tracker = get_tracker()
        
    def _define_speedrun_configs(self) -> List[SpeedrunConfig]:
        """Define the two speedrun configurations."""
        configs = [
            # Baseline Configuration
            SpeedrunConfig(
                name="baseline",
                description="Baseline T4-optimized configuration",
                batch_size=12,
                gradient_accumulation_steps=3,
                use_amp=True,
                use_fp16_matmul=True,
                num_workers=2,
                pin_memory=False,
                prefetch_factor=2,
                max_steps=50,
                eval_every=10,
                eval_steps=10
            ),
            
            # Memory Optimized Configuration
            SpeedrunConfig(
                name="memory_optimized",
                description="Memory-optimized configuration",
                batch_size=8,
                gradient_accumulation_steps=4,
                use_amp=True,
                use_fp16_matmul=True,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                max_steps=50,
                eval_every=10,
                eval_steps=10
            )
        ]
        return configs
    
    def run_speedrun_challenge(self) -> List[SpeedrunResult]:
        """Run the speedrun challenge."""
        self.tracker.log("🏁 Starting Training Speedrun Challenge")
        self.tracker.log("=" * 60)
        self.tracker.log(f"📊 Challenge: Baseline vs Memory Optimized")
        self.tracker.log(f"⏱️  Steps per experiment: {self.configs[0].max_steps}")
        self.tracker.log(f"🎯 Goal: Establish performance measurement standards")
        
        for i, config in enumerate(self.configs):
            self.tracker.log(f"\n🏃 Speedrun {i+1}/{len(self.configs)}: {config.name}")
            self.tracker.log(f"   Description: {config.description}")
            self.tracker.log(f"   Starting at {time.strftime('%H:%M:%S')}...")
            
            # Also print to console for remote monitoring
            print(f"\n🏃 Speedrun {i+1}/{len(self.configs)}: {config.name}")
            print(f"   Description: {config.description}")
            print(f"   Starting at {time.strftime('%H:%M:%S')}...")
            
            try:
                # Start tracking this experiment
                self.tracker.start_experiment(asdict(config))
                
                result = self._run_single_speedrun(config)
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
                
                # Print results to console
                print(f"✅ Speedrun {i+1} completed: {config.name}")
                print(f"   Steps/sec: {result.steps_per_second:.2f}")
                print(f"   Time: {result.total_time_seconds:.2f}s")
                print(f"   Avg step time: {result.avg_step_time_ms:.2f}ms")
                print(f"   Final Loss: {result.final_loss:.4f}")
                print(f"   Peak Memory: {result.peak_memory_mb:.1f} MB")
                
            except Exception as e:
                self.tracker.log(f"❌ Speedrun failed: {e}")
                print(f"❌ Speedrun {i+1} failed: {config.name}")
                print(f"   Error: {e}")
                continue
        
        # Print final challenge results
        self._print_challenge_results()
        
        return self.results
    
    def _run_single_speedrun(self, config: SpeedrunConfig) -> SpeedrunResult:
        """Run a single speedrun experiment."""
        start_time = time.time()
        
        # Setup configuration
        t4_config = t4_configure()
        model_config = t4_config.get_model_config()
        
        # Apply speedrun-specific settings
        model_config.batch_size = config.batch_size
        model_config.gradient_accumulation_steps = config.gradient_accumulation_steps
        model_config.use_amp = config.use_amp
        model_config.use_fp16_matmul = config.use_fp16_matmul
        model_config.max_steps = config.max_steps
        model_config.eval_every = config.eval_every
        model_config.eval_steps = config.eval_steps
        
        # Auto-size dataset for speedrun
        model_config.num_documents = 1000  # Reduced for faster testing
        model_config.max_tokens = 100000   # Reduced for faster testing
        
        print(f"   Loading {model_config.num_documents} documents...")
        
        # Load data
        print("   📚 Loading data...")
        texts, tokenizer, tokens = load_and_cache_data(model_config)
        
        print("   🔧 Creating dataset...")
        dataset = TextTokenDataset(tokens, model_config.max_seq_len)
        
        # Train/val split
        print("   🔄 Splitting dataset...")
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor
        )
        
        # Create model
        print("   🧠 Creating model...")
        model = MoEMinimalLLM(model_config)
        
        # Train the model
        print("   🚀 Starting training...")
        training_start = time.time()
        
        model, final_metrics = train_moe_model_timed(
            model_config, train_loader, val_loader, time_limit_minutes=1
        )
        
        training_time = time.time() - training_start
        total_time = time.time() - start_time
        
        # Get memory usage
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        avg_step_time_ms = (training_time * 1000) / config.max_steps
        
        return SpeedrunResult(
            config=config,
            total_time_seconds=total_time,
            steps_per_second=config.max_steps / training_time if training_time > 0 else 0,
            final_loss=final_metrics.get('val_loss', 0.0),
            final_accuracy=final_metrics.get('val_accuracy', 0.0),
            memory_usage_mb=peak_memory,
            avg_step_time_ms=avg_step_time_ms,
            peak_memory_mb=peak_memory
        )
    
    def _print_challenge_results(self):
        """Print the final challenge results."""
        if not self.results:
            print("❌ No results to compare.")
            return
        
        print(f"\n🏁 SPEEDRUN CHALLENGE RESULTS")
        print("=" * 60)
        
        # Sort by steps per second (speed)
        sorted_results = sorted(self.results, key=lambda x: x.steps_per_second, reverse=True)
        
        for result in sorted_results:
            print(f"{result.config.name}: {result.steps_per_second:.2f} steps/sec, "
                  f"{result.total_time_seconds:.1f}s, {result.final_loss:.4f} loss")
        
        # Find winner
        winner = sorted_results[0]
        print(f"\n🏆 Winner: {winner.config.name} ({winner.steps_per_second:.2f} steps/sec)")
        
        if len(self.results) > 1:
            baseline = next((r for r in self.results if r.config.name == "baseline"), None)
            if baseline and winner.config.name != "baseline":
                speedup = winner.steps_per_second / baseline.steps_per_second
                print(f"📈 {speedup:.2f}x speedup over baseline")
    
    def save_results(self, filename: str = "speedrun_challenge_results.json"):
        """Save speedrun results to JSON file."""
        results_data = []
        for result in self.results:
            result_dict = {
                'config': asdict(result.config),
                'total_time_seconds': result.total_time_seconds,
                'steps_per_second': result.steps_per_second,
                'final_loss': result.final_loss,
                'final_accuracy': result.final_accuracy,
                'memory_usage_mb': result.memory_usage_mb,
                'avg_step_time_ms': result.avg_step_time_ms,
                'peak_memory_mb': result.peak_memory_mb
            }
            results_data.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.tracker.log(f"\n💾 Results saved to {filename}")


def main():
    """Main function to run the speedrun challenge."""
    parser = argparse.ArgumentParser(description="Training Speedrun Challenge")
    parser.add_argument("--output", default="speedrun_challenge_results.json", 
                       help="Output file for results")
    args = parser.parse_args()
    
    try:
        # Run speedrun challenge
        challenge = TrainingSpeedrunChallenge()
        results = challenge.run_speedrun_challenge()
        
        # Save results
        challenge.save_results(args.output)
        
        print(f"\n✅ Speedrun challenge completed!")
        print(f"📁 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Speedrun challenge failed: {e}")
        raise


if __name__ == "__main__":
    main()
