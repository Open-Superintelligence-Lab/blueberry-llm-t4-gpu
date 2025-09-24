#!/usr/bin/env python3
"""
Momentum Warmup Research Runner

This script provides an easy interface to run momentum warmup experiments
with different configurations and schedules.
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from research_agents.momentum_warmup_experiment import MomentumWarmupExperiment
from research_agents.momentum_warmup_config import (
    MomentumWarmupExperimentConfig,
    QUICK_EXPERIMENT,
    FULL_EXPERIMENT,
    T4_OPTIMIZED_EXPERIMENT
)


def run_experiment(config_name: str, custom_config: MomentumWarmupExperimentConfig = None):
    """Run a momentum warmup experiment with the specified configuration"""
    
    print(f"🚀 Starting Momentum Warmup Experiment: {config_name}")
    print("=" * 80)
    
    # Select configuration
    if custom_config:
        config = custom_config
    elif config_name == "quick":
        config = QUICK_EXPERIMENT
    elif config_name == "full":
        config = FULL_EXPERIMENT
    elif config_name == "t4":
        config = T4_OPTIMIZED_EXPERIMENT
    else:
        print(f"❌ Unknown configuration: {config_name}")
        print("Available configurations: quick, full, t4")
        return
    
    print(f"📋 Configuration: {config.experiment_name}")
    print(f"📝 Description: {config.description}")
    print(f"🎯 Target Hardware: {config.target_hardware}")
    print(f"🔄 Schedules to test: {len(config.momentum_schedules)}")
    print(f"📊 Runs per schedule: {config.training_config['num_runs_per_schedule']}")
    print(f"⏱️ Max steps per run: {config.training_config['max_steps']}")
    
    # Load data to get vocab size
    print("\n📚 Loading data...")
    from data.loader import load_and_cache_data
    from configs.moe_config import MoEModelConfig
    
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    print(f"✅ Loaded data with vocab size: {vocab_size}")
    
    # Create and run experiment
    print(f"\n🧪 Creating experiment...")
    experiment = MomentumWarmupExperiment(config)
    
    start_time = time.time()
    results = experiment.run_experiment(vocab_size)
    total_time = time.time() - start_time
    
    print(f"\n🎯 Experiment Complete!")
    print(f"⏱️ Total experiment time: {total_time/60:.1f} minutes")
    print(f"📊 Results saved to: experiments/{config.experiment_name}/")
    
    # Print summary
    print(f"\n📈 Quick Summary:")
    best_schedule = min(experiment.results.items(), key=lambda x: x[1]['final_val_loss'])
    print(f"🏆 Best Schedule: {best_schedule[0]}")
    print(f"📉 Best Val Loss: {best_schedule[1]['final_val_loss']:.4f}")
    print(f"📊 Std Dev: {best_schedule[1]['final_val_loss_std']:.4f}")
    
    return results


def list_schedules():
    """List all available momentum schedules"""
    config = MomentumWarmupExperimentConfig()
    print("📋 Available Momentum Schedules:")
    print("=" * 50)
    
    for i, schedule in enumerate(config.momentum_schedules, 1):
        print(f"{i:2d}. {schedule['name']}")
        print(f"    {schedule['description']}")
        print(f"    Type: {schedule['type']}")
        print()


def create_custom_experiment():
    """Interactive creation of custom experiment configuration"""
    print("🔧 Creating Custom Experiment Configuration")
    print("=" * 50)
    
    # Get basic parameters
    experiment_name = input("Experiment name: ").strip() or "custom_momentum_experiment"
    max_steps = int(input("Max training steps (default 1000): ") or "1000")
    num_runs = int(input("Number of runs per schedule (default 3): ") or "3")
    
    # Get model size
    print("\nModel Configuration:")
    d_model = int(input("Model dimension (default 256): ") or "256")
    n_layers = int(input("Number of layers (default 4): ") or "4")
    batch_size = int(input("Batch size (default 16): ") or "16")
    
    # Create custom config
    custom_config = MomentumWarmupExperimentConfig(
        experiment_name=experiment_name,
        model_config={
            'd_model': d_model,
            'n_layers': n_layers,
            'batch_size': batch_size,
        },
        training_config={
            'max_steps': max_steps,
            'num_runs_per_schedule': num_runs,
        }
    )
    
    print(f"\n✅ Created custom configuration: {experiment_name}")
    return custom_config


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Run momentum warmup experiments for Muon optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_momentum_experiment.py quick          # Run quick experiment
  python run_momentum_experiment.py full           # Run full experiment  
  python run_momentum_experiment.py t4             # Run T4-optimized experiment
  python run_momentum_experiment.py --list         # List available schedules
  python run_momentum_experiment.py --custom       # Create custom experiment
        """
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        choices=['quick', 'full', 't4'],
        help='Experiment configuration to run'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available momentum schedules'
    )
    
    parser.add_argument(
        '--custom',
        action='store_true',
        help='Create and run custom experiment'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without running experiment'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_schedules()
        return
    
    if args.custom:
        custom_config = create_custom_experiment()
        if not args.dry_run:
            run_experiment("custom", custom_config)
        return
    
    if not args.config:
        parser.print_help()
        return
    
    if args.dry_run:
        print(f"🔍 Dry run mode - would run {args.config} experiment")
        return
    
    # Run the experiment
    run_experiment(args.config)


if __name__ == "__main__":
    main()
