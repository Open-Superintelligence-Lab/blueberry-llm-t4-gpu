#!/usr/bin/env python3
"""
Timed Training Module for Blueberry LLM

This module provides training functions with detailed timing measurements
for performance analysis and optimization.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import math
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from timing import TrainingTimer, get_timer, reset_timer
from configs.t4_moe_config import T4MoEModelConfig


def train_model_with_timing(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: T4MoEModelConfig,
    device: Optional[torch.device] = None,
    experiment_name: str = "experiment"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train model with detailed timing measurements.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Model configuration
        device: Device to train on
        experiment_name: Name of the experiment for logging
        
    Returns:
        Tuple of (trained_model, final_metrics)
    """
    # Reset timer for this experiment
    reset_timer()
    timer = get_timer()
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\n🚀 Starting timed training: {experiment_name}")
    print("=" * 60)
    
    # Setup optimizers and schedulers
    from optimizers import setup_optimizers, get_lr_scheduler
    optimizers = setup_optimizers(model, config, use_warmup=True)
    schedulers = [get_lr_scheduler(opt, config, "cosine_warmup") for opt in optimizers]
    
    # Setup gradient scaler for mixed precision
    scaler = GradScaler('cuda') if config.use_amp else None
    
    # Training state
    step = 0
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    # Training metrics
    total_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0
    
    model.train()
    pbar = tqdm(total=config.max_steps, desc=f"Training {experiment_name}")
    
    print(f"📊 Training for {config.max_steps} steps")
    print(f"🔧 Batch size: {config.batch_size}")
    print(f"🔧 Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"🔧 Mixed precision: {config.use_amp}")
    
    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break
            
            # Time data loading
            with timer.time_data_loading():
                x, y = x.to(device), y.to(device)
            
            # Time forward pass
            with timer.time_forward():
                if config.use_amp:
                    with autocast('cuda'):
                        if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
                            logits, aux_loss = model(x, return_aux_loss=True)
                        else:
                            logits = model(x)
                            aux_loss = None
                        
                        ce_loss = F.cross_entropy(
                            logits.view(-1, config.vocab_size), 
                            y.view(-1)
                        )
                        
                        total_loss = ce_loss
                        if aux_loss is not None:
                            total_loss = total_loss + aux_loss
                        
                        loss = total_loss / config.gradient_accumulation_steps
                else:
                    if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
                        logits, aux_loss = model(x, return_aux_loss=True)
                    else:
                        logits = model(x)
                        aux_loss = None
                    
                    ce_loss = F.cross_entropy(
                        logits.view(-1, config.vocab_size), 
                        y.view(-1)
                    )
                    
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss
                    
                    loss = total_loss / config.gradient_accumulation_steps
            
            # Time backward pass
            with timer.time_backward():
                if config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Accumulate metrics
            total_loss += ce_loss.item()
            total_aux_loss += aux_loss.item() if aux_loss is not None else 0.0
            num_batches += 1
            
            step += 1
            
            # Optimizer step (only when accumulation is complete)
            if step % config.gradient_accumulation_steps == 0:
                with timer.time_optimizer():
                    if config.use_amp:
                        # Unscale gradients for clipping
                        for optimizer in optimizers:
                            scaler.unscale_(optimizer)
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        
                        # Optimizer step
                        for optimizer in optimizers:
                            scaler.step(optimizer)
                            optimizer.zero_grad(set_to_none=True)
                        
                        # Update scaler
                        scaler.update()
                        
                        # Update learning rate
                        for scheduler in schedulers:
                            scheduler.step()
                    else:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        
                        # Optimizer step
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                        
                        # Update learning rate
                        for scheduler in schedulers:
                            scheduler.step()
            
            # Update progress bar
            if step % 20 == 0:
                pbar.update(20)
                pbar.set_postfix({
                    'loss': f"{ce_loss.item():.4f}",
                    'aux': f"{aux_loss.item() if aux_loss is not None else 0.0:.4f}",
                    'step': step
                })
            
            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model_with_timing(model, val_loader, config)
                
                # Check for best model
                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']
                    print(f"🎉 New best validation loss: {best_val_loss:.4f}")
    
    pbar.close()
    
    # Stop memory monitoring
    timer.stop_memory_monitoring()
    
    # Final evaluation
    final_metrics = evaluate_model_with_timing(model, val_loader, config, final=True)
    
    # Add timing information
    timing_summary = timer.get_summary()
    final_metrics.update({
        'experiment_name': experiment_name,
        'timing_summary': timing_summary,
        'training_time_minutes': timing_summary['total_time_seconds'] / 60,
        'steps_per_second': timing_summary['steps_per_second'],
        'final_step': step,
        'best_val_loss': best_val_loss
    })
    
    print(f"\n🎯 Training completed for {experiment_name}!")
    timer.print_summary()
    
    return model, final_metrics


def evaluate_model_with_timing(
    model: nn.Module,
    val_loader: DataLoader,
    config: T4MoEModelConfig,
    final: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model with timing measurements.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        config: Model configuration
        final: Whether this is the final evaluation
        
    Returns:
        Evaluation metrics
    """
    timer = get_timer()
    model.eval()
    
    total_loss = 0.0
    total_aux_loss = 0.0
    correct = 0
    total = 0
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for x, y in val_loader:
            with timer.time_data_loading():
                x, y = x.to(device), y.to(device)
            
            with timer.time_forward():
                if config.use_amp:
                    with autocast('cuda'):
                        if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
                            logits, aux_loss = model(x, return_aux_loss=True)
                        else:
                            logits = model(x)
                            aux_loss = None
                        
                        loss = F.cross_entropy(
                            logits.view(-1, config.vocab_size), 
                            y.view(-1)
                        )
                else:
                    if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
                        logits, aux_loss = model(x, return_aux_loss=True)
                    else:
                        logits = model(x)
                        aux_loss = None
                    
                    loss = F.cross_entropy(
                        logits.view(-1, config.vocab_size), 
                        y.view(-1)
                    )
            
            total_loss += loss.item()
            if aux_loss is not None:
                total_aux_loss += aux_loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == y).sum().item()
            total += y.numel()
    
    model.train()
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    avg_aux_loss = total_aux_loss / len(val_loader)
    accuracy = correct / total
    perplexity = math.exp(min(avg_loss, 20))
    
    metrics = {
        'val_loss': avg_loss,
        'val_aux_loss': avg_aux_loss,
        'val_accuracy': accuracy,
        'val_perplexity': perplexity
    }
    
    prefix = "📊 Final" if final else f"Step {timer.metrics.total_steps}"
    print(f"\n{prefix} Evaluation:")
    print(f"   Val Loss: {avg_loss:.4f}")
    print(f"   Val Accuracy: {accuracy:.4f}")
    print(f"   Val Perplexity: {perplexity:.2f}")
    
    return metrics


def benchmark_training_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    config: T4MoEModelConfig,
    device: torch.device,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark a single training step multiple times for accurate timing.
    
    Args:
        model: Model to benchmark
        x: Input tensor
        y: Target tensor
        config: Model configuration
        device: Device to run on
        num_iterations: Number of iterations to run
        
    Returns:
        Timing statistics
    """
    model.train()
    
    # Warmup
    for _ in range(10):
        _ = model(x)
        if x.grad is not None:
            x.grad.zero_()
    
    # Benchmark forward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        if config.use_amp:
            with autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = time.perf_counter() - start_time
    
    # Benchmark backward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        if config.use_amp:
            with autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            scaler = GradScaler('cuda')
            scaler.scale(loss).backward()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            loss.backward()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    backward_time = time.perf_counter() - start_time
    
    return {
        'forward_time_per_step': forward_time / num_iterations,
        'backward_time_per_step': backward_time / num_iterations,
        'total_time_per_step': (forward_time + backward_time) / num_iterations,
        'steps_per_second': num_iterations / (forward_time + backward_time)
    }
