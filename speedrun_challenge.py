#!/usr/bin/env python3
"""
Minimal Training Speed Challenge: Baseline vs Memory Optimized

This script runs a focused speedrun challenge between baseline and memory-optimized
configurations for 1000 steps each to establish performance measurement standards.
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
from legacy.llm import load_and_cache_data, TextTokenDataset, MoEMinimalLLM
from configs.t4_moe_config import T4MoEModelConfig
from timed_training import train_model_with_timing
from record_tracker import RecordTracker, get_tracker, reset_tracker


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
    max_steps: int = 10
    eval_every: int = 5
    eval_steps: int = 10  # Limit validation batches for speed


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
                max_steps=10,
                eval_every=5,
                eval_steps=10
            ),
            
            # Memory Optimized Configuration
            SpeedrunConfig(
                name="memory_optimized",
                description="Memory-optimized configuration (winner from previous tests)",
                batch_size=8,
                gradient_accumulation_steps=4,
                use_amp=True,
                use_fp16_matmul=True,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                max_steps=10,
                eval_every=5,
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
        model_config.num_documents = 2000
        model_config.max_tokens = 200000
        
        print(f"   Loading {model_config.num_documents} documents...")
        
        # Load data
        print("   📚 Loading and tokenizing data...")
        texts, tokenizer, tokens = load_and_cache_data(model_config)
        print("   ✅ Data loaded successfully")
        
        print("   🔧 Creating dataset...")
        dataset = TextTokenDataset(tokens, model_config.max_seq_len)
        print(f"   ✅ Dataset created with {len(dataset)} samples")
        
        # Train/val split
        print("   🔄 Splitting dataset into train/val...")
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        print(f"   ✅ Split complete: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Create optimized data loaders
        print("   🔧 Creating data loaders...")
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
        print(f"   ✅ Data loaders created (batch_size={model_config.batch_size}, workers={config.num_workers})")
        
        # Create model
        print("   🧠 Creating model...")
        model = MoEMinimalLLM(model_config)
        print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train the model with timing
        print("   🚀 Starting training...")
        print(f"   📊 Training for {config.max_steps} steps")
        print(f"   🔧 Batch size: {model_config.batch_size}")
        print(f"   🔧 Gradient accumulation: {model_config.gradient_accumulation_steps}")
        print(f"   🔧 Mixed precision: {model_config.use_amp}")
        training_start = time.time()
        
        model, final_metrics = train_model_with_timing(
            model, train_loader, val_loader, model_config, experiment_name=config.name
        )
        
        training_time = time.time() - training_start
        total_time = time.time() - start_time
        
        # Get detailed memory usage
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
        print("=" * 80)
        print(f"{'Configuration':<20} {'Steps/s':<10} {'Time(s)':<10} {'Step(ms)':<10} {'Loss':<8} {'Memory(MB)':<12}")
        print("-" * 80)
        
        # Sort by steps per second (speed)
        sorted_results = sorted(self.results, key=lambda x: x.steps_per_second, reverse=True)
        
        for result in sorted_results:
            print(f"{result.config.name:<20} {result.steps_per_second:<10.2f} "
                  f"{result.total_time_seconds:<10.2f} {result.avg_step_time_ms:<10.2f} "
                  f"{result.final_loss:<8.4f} {result.peak_memory_mb:<12.1f}")
        
        # Find winner and calculate improvement
        winner = sorted_results[0]
        baseline = next((r for r in self.results if r.config.name == "baseline"), None)
        
        print(f"\n🏆 WINNER: {winner.config.name}")
        print(f"   Speed: {winner.steps_per_second:.2f} steps/second")
        print(f"   Time: {winner.total_time_seconds:.2f} seconds")
        print(f"   Avg step time: {winner.avg_step_time_ms:.2f}ms")
        
        if baseline and winner.config.name != "baseline":
            speedup = winner.steps_per_second / baseline.steps_per_second
            time_saved = baseline.total_time_seconds - winner.total_time_seconds
            print(f"\n📈 IMPROVEMENT OVER BASELINE:")
            print(f"   Speed improvement: {speedup:.2f}x")
            print(f"   Time saved: {time_saved:.2f} seconds")
            print(f"   Memory efficiency: {baseline.peak_memory_mb/winner.peak_memory_mb:.2f}x")
        
        # Performance measurement standards
        print(f"\n📊 PERFORMANCE MEASUREMENT STANDARDS:")
        print(f"   Primary metric: Steps per second")
        print(f"   Secondary metrics: Total time, Average step time")
        print(f"   Memory metric: Peak GPU memory usage")
        print(f"   Quality metric: Final validation loss")
    
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
    
    # Initialize record tracking
    tracker = get_tracker()
    tracker.log("🏁 Starting Training Speedrun Challenge")
    tracker.log(f"📁 Output file: {args.output}")
    
    try:
        # Run speedrun challenge
        challenge = TrainingSpeedrunChallenge()
        results = challenge.run_speedrun_challenge()
        
        # Save results
        challenge.save_results(args.output)
        
        tracker.log(f"\n✅ Speedrun challenge completed!")
        tracker.log(f"📁 Results saved to: {args.output}")
        
    except Exception as e:
        tracker.log(f"❌ Speedrun challenge failed: {e}")
        raise
    finally:
        # Cleanup
        tracker.cleanup()


if __name__ == "__main__":
    main()
