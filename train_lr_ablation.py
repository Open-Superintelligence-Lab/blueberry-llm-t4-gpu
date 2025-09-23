import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed

# Set environment variables to fix warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch.compile to avoid inductor warnings


def train_with_lr(config: MoEModelConfig, train_loader: DataLoader, val_loader: DataLoader, lr: float, run_name: str):
    """Train model with specific learning rate and return evaluation metrics"""
    print(f"\n{'='*60}")
    print(f"🧪 TRAINING: {run_name} (LR={lr})")
    print(f"{'='*60}")
    
    # Create a copy of config with modified learning rate
    lr_config = MoEModelConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        batch_size=config.batch_size,
        max_steps=config.max_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        muon_lr=lr,  # Set the specific learning rate
        max_seq_len=config.max_seq_len,
        num_documents=config.num_documents,
        max_tokens=config.max_tokens,
        eval_every=config.eval_every,
        eval_steps=config.eval_steps,
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        grad_clip=config.grad_clip,
        use_amp=config.use_amp,
        vocab_size=config.vocab_size,
        log_milestones=config.log_milestones,
        num_experts=config.num_experts,
        expert_top_k=config.expert_top_k,
        load_balancing_weight=config.load_balancing_weight
    )
    
    # Train model and get evaluation metrics with history
    start_time = time.time()
    model, final_metrics, eval_history = train_moe_model(lr_config, train_loader, val_loader, return_eval_history=True)
    total_time = time.time() - start_time
    
    print(f"\n🎯 {run_name} Results:")
    print(f"⏱️ Training time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    
    return final_metrics, total_time, eval_history


def plot_lr_comparison(results: dict):
    """Plot validation loss comparison for different learning rates"""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (lr, data) in enumerate(results.items()):
        final_metrics, training_time, eval_history = data
        
        # Extract time and loss data
        if eval_history:
            # Convert timestamps to elapsed time in minutes
            start_time = eval_history[0]['time']
            elapsed_times = [(eval_point['time'] - start_time) / 60 for eval_point in eval_history]
            losses = [eval_point['val_loss'] for eval_point in eval_history]
            
            plt.plot(elapsed_times, losses, color=colors[i % len(colors)], 
                    linewidth=2, label=f'LR={lr}', marker='o', markersize=6)
        else:
            # Fallback if no history available
            plt.plot([0, training_time/60], [final_metrics['val_loss'], final_metrics['val_loss']], 
                    color=colors[i % len(colors)], linewidth=2, 
                    label=f'LR={lr}', marker='o', markersize=8)
    
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Learning Rate Ablation Study\nValidation Loss vs Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lr_ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n📈 Learning rate ablation plot saved as 'lr_ablation_comparison.png'")


def main():
    """Main learning rate ablation script"""
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed for reproducibility
    set_seed(42)

    # Load data first to get vocab_size
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size

    # Create base config for comprehensive learning rate sweep
    base_config = MoEModelConfig(
        vocab_size=vocab_size,
        max_steps=200,  # Comprehensive training: 200 steps total
        eval_every=40,  # Evaluate at steps 40, 80, 120, 160, 200 (5 evaluations)
        batch_size=24,
        muon_lr=0.01  # Base learning rate
    )

    dataset = TextTokenDataset(tokens, base_config.max_seq_len)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=base_config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=0)

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    print(f"📊 Comprehensive learning rate sweep: {base_config.max_steps} steps total")
    print(f"📊 Evaluation at steps: 40, 80, 120, 160, 200 (5 evaluations)")

    # Learning rates to test (comprehensive sweep)
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    # Store results for comparison
    results = {}
    
    print(f"\n🧪 LEARNING RATE ABLATION STUDY")
    print(f"Testing learning rates: {learning_rates}")
    print(f"Each run will train for {base_config.max_steps} steps")
    
    # Train with each learning rate
    for lr in learning_rates:
        run_name = f"LR_{lr}"
        final_metrics, training_time, eval_history = train_with_lr(
            base_config, train_loader, val_loader, lr, run_name
        )
        results[lr] = (final_metrics, training_time, eval_history)
    
    # Plot comparison
    plot_lr_comparison(results)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 LEARNING RATE ABLATION SUMMARY")
    print(f"{'='*60}")
    for lr, (final_metrics, training_time, eval_history) in results.items():
        print(f"LR={lr:6.3f}: Val Loss={final_metrics['val_loss']:.4f}, "
              f"Val Acc={final_metrics['val_accuracy']:.4f}, "
              f"Val PPL={final_metrics['val_perplexity']:.2f}")
    
    # Find best learning rate
    best_lr = min(results.keys(), key=lambda lr: results[lr][0]['val_loss'])
    print(f"\n🏆 Best learning rate: {best_lr} (lowest validation loss: {results[best_lr][0]['val_loss']:.4f})")


if __name__ == "__main__":
    main()
