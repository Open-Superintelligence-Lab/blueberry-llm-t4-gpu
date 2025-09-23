#!/usr/bin/env python3
"""
Time-limited training wrapper for Blueberry LLM.
This script adds time-based training control with interrupt handling.
"""

import os
import sys
import signal
import time
import argparse
from contextlib import contextmanager

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, random_split
from core.t4_config import t4_configure
from legacy.llm import train_moe_model, load_and_cache_data, TextTokenDataset, MoEModelConfig


class TimeLimitedTraining:
    """Wrapper for time-limited training with interrupt handling."""
    
    def __init__(self, max_time_minutes: int = 30):
        self.max_time_minutes = max_time_minutes
        self.max_time_seconds = max_time_minutes * 60
        self.start_time = None
        self.interrupted = False
        self.original_sigint_handler = None
        
    def setup_interrupt_handling(self):
        """Setup graceful interrupt handling."""
        self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        
    def cleanup_interrupt_handling(self):
        """Restore original signal handler."""
        if self.original_sigint_handler:
            signal.signal(signal.SIGINT, self.original_sigint_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\n🛑 Training interrupted by user (Ctrl+C)")
        self.interrupted = True
        
    @contextmanager
    def time_limit_context(self):
        """Context manager for time-limited execution."""
        self.start_time = time.time()
        self.setup_interrupt_handling()
        
        try:
            yield self
        finally:
            self.cleanup_interrupt_handling()
    
    def is_time_up(self) -> bool:
        """Check if time limit has been reached."""
        if self.start_time is None:
            return False
        
        elapsed_time = time.time() - self.start_time
        return elapsed_time >= self.max_time_seconds
    
    def get_remaining_time(self) -> float:
        """Get remaining time in seconds."""
        if self.start_time is None:
            return self.max_time_seconds
        
        elapsed_time = time.time() - self.start_time
        return max(0, self.max_time_seconds - elapsed_time)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0
        
        return time.time() - self.start_time


def train_moe_model_timed(config: MoEModelConfig, train_loader: DataLoader, val_loader: DataLoader, 
                         time_limit_minutes: int = 30):
    """Train the MoE model with time limit and interrupt handling."""
    print(f"\n🚀 Training MoE model with {config.num_experts} experts (top-{config.expert_top_k})")
    print(f"⏰ Time limit: {time_limit_minutes} minutes")
    
    # Initialize time-limited training wrapper
    time_limited = TimeLimitedTraining(time_limit_minutes)
    
    with time_limited.time_limit_context():
        # Import required modules
        from legacy.llm import set_seed, MoEMinimalLLM, setup_muon_optimizer, evaluate_model
        import torch.nn.functional as F
        from torch.cuda.amp import autocast, GradScaler
        import math
        from tqdm import tqdm
        
        # Initialize model
        set_seed(42)
        model = MoEMinimalLLM(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        active_params = sum(p.numel() for n, p in model.named_parameters()
                           if 'expert' not in n)
        expert_params = total_params - active_params

        print(f"  📊 Total parameters: {total_params:,}")
        print(f"  📊 Active parameters: {active_params:,}")
        print(f"  📊 Expert parameters: {expert_params:,}")
        print(f"  📊 Parameter efficiency: {active_params/total_params:.1%} active per forward pass")

        # Setup optimizers
        optimizers = setup_muon_optimizer(model, config)

        # Learning rate schedule
        schedulers = []
        for optimizer in optimizers:
            warmup_steps = config.max_steps // 20
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            schedulers.append(scheduler)

        scaler = GradScaler() if config.use_amp else None

        # Training loop with time limit
        model.train()
        step = 0
        pbar = tqdm(total=config.max_steps, desc="Training MoE")
        
        # Track training metrics
        training_start_time = time.time()
        best_val_loss = float('inf')
        max_time_seconds = time_limit_minutes * 60

        while step < config.max_steps and not time_limited.interrupted:
            # Check time limit
            elapsed_time = time.time() - training_start_time
            if elapsed_time >= max_time_seconds:
                print(f"\n⏰ Time limit reached ({time_limit_minutes} minutes)")
                break
                
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= config.max_steps or time_limited.interrupted:
                    break
                
                # Check time limit during batch processing
                elapsed_time = time.time() - training_start_time
                if elapsed_time >= max_time_seconds:
                    break

                x, y = x.to(device), y.to(device)

                # Forward pass
                if config.use_amp:
                    with autocast():
                        logits, aux_loss = model(x, return_aux_loss=True)
                        ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                        # Combine main loss and auxiliary loss
                        total_loss = ce_loss
                        if aux_loss is not None:
                            total_loss = total_loss + aux_loss

                        loss = total_loss / config.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss

                    loss = total_loss / config.gradient_accumulation_steps
                    loss.backward()

                # Optimizer step
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    if config.use_amp:
                        for optimizer in optimizers:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                        for optimizer in optimizers:
                            scaler.step(optimizer)
                            optimizer.zero_grad()
                        for scheduler in schedulers:
                            scheduler.step()
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        for scheduler in schedulers:
                            scheduler.step()

                # Logging with time info
                if step % 100 == 0:
                    with torch.no_grad():
                        predictions = logits.argmax(dim=-1)
                        accuracy = (predictions == y).float().mean().item()
                        current_loss = ce_loss.item()
                        perplexity = math.exp(min(current_loss, 20))
                    
                    # Calculate remaining time
                    remaining_minutes = (max_time_seconds - elapsed_time) / 60
                    elapsed_minutes = elapsed_time / 60

                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'aux': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                        'acc': f'{accuracy:.3f}',
                        'ppl': f'{perplexity:.1f}',
                        'time': f'{elapsed_minutes:.1f}m/{time_limit_minutes}m',
                        'left': f'{remaining_minutes:.1f}m'
                    })

                # Evaluation
                if step % config.eval_every == 0 and step > 0:
                    eval_metrics = evaluate_model(model, val_loader, config)
                    print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                          f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                          f"Val PPL: {eval_metrics['val_perplexity']:.2f}")
                    
                    # Track best model
                    if eval_metrics['val_loss'] < best_val_loss:
                        best_val_loss = eval_metrics['val_loss']
                        print(f"🎉 New best validation loss: {best_val_loss:.4f}")

                # Milestone evaluations
                if step in getattr(config, 'log_milestones', ()):    
                    eval_metrics = evaluate_model(model, val_loader, config)
                    print(f"\n🧪 Milestone {step}: Val Loss: {eval_metrics['val_loss']:.4f}")

                step += 1
                if step % 20 == 0:
                    pbar.update(20)

        pbar.close()

        # Final evaluation
        final_eval = evaluate_model(model, val_loader, config)
        
        # Calculate final training metrics
        total_training_time = time.time() - training_start_time
        
        print(f"\n📊 Final Results:")
        print(f"   Val Loss: {final_eval['val_loss']:.4f}")
        print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
        print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")
        print(f"   Training Time: {total_training_time/60:.1f} minutes")
        print(f"   Steps Completed: {step}")
        print(f"   Steps per Second: {step/total_training_time:.2f}")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        
        if time_limited.interrupted:
            print(f"   ⚠️  Training was interrupted by user")
        elif elapsed_time >= max_time_seconds:
            print(f"   ⏰ Training stopped due to time limit")
        else:
            print(f"   ✅ Training completed normally")

        # Add timing info to final metrics
        final_eval.update({
            'training_time_minutes': total_training_time / 60,
            'steps_completed': step,
            'steps_per_second': step / total_training_time if total_training_time > 0 else 0,
            'best_val_loss': best_val_loss,
            'interrupted': time_limited.interrupted,
            'time_limit_reached': elapsed_time >= max_time_seconds
        })

        return model, final_eval
