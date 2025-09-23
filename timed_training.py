#!/usr/bin/env python3
"""
Simple Training Module for Blueberry LLM

This module provides basic training functions with simple timing measurements.
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
from typing import Dict, Any, Optional, Tuple

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from configs.t4_moe_config import T4MoEModelConfig
from record_tracker import get_tracker


def train_model_with_timing(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: T4MoEModelConfig,
    device: Optional[torch.device] = None,
    experiment_name: str = "experiment"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train model with simple timing measurements.
    
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
    tracker = get_tracker()
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    tracker.log(f"\n🚀 Starting training: {experiment_name}")
    
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
    
    model.train()
    pbar = tqdm(total=config.max_steps, desc=f"Training {experiment_name}")
    
    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass
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
            
            # Backward pass
            if config.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            step += 1
            
            # Optimizer step (only when accumulation is complete)
            if step % config.gradient_accumulation_steps == 0:
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
            
            # Update progress bar and record tracking
            if step % 20 == 0:
                pbar.update(20)
                pbar.set_postfix({
                    'loss': f"{ce_loss.item():.4f}",
                    'aux': f"{aux_loss.item() if aux_loss is not None else 0.0:.4f}",
                    'step': step
                })
                
                # Update record tracker
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                step_time_ms = (time.time() - training_start_time) * 1000 / max(step, 1)
                tracker.update_training_progress(step, config.max_steps, ce_loss.item(), step_time_ms, memory_mb)
            
            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                max_eval_batches = getattr(config, 'eval_steps', 50)
                actual_batches = min(max_eval_batches, len(val_loader))
                print(f"\n🔍 Running validation at step {step}...")
                val_start = time.time()
                eval_metrics = evaluate_model_with_timing(model, val_loader, config)
                val_time = time.time() - val_start
                print(f"   ✅ Validation completed in {val_time:.2f}s")
                
                # Update record tracker with validation results
                tracker.update_validation(step, eval_metrics['val_loss'], 
                                        eval_metrics['val_accuracy'], val_time * 1000)
                
                # Check for best model
                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']
                    tracker.log(f"🎉 New best validation loss: {best_val_loss:.4f}")
    
    pbar.close()
    
    # Final evaluation
    final_metrics = evaluate_model_with_timing(model, val_loader, config, final=True)
    
    # Add timing information
    training_time = time.time() - training_start_time
    final_metrics.update({
        'experiment_name': experiment_name,
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'steps_per_second': step / training_time if training_time > 0 else 0,
        'final_step': step,
        'best_val_loss': best_val_loss
    })
    
    tracker.log(f"\n🎯 Training completed for {experiment_name}!")
    
    return model, final_metrics


def evaluate_model_with_timing(
    model: nn.Module,
    val_loader: DataLoader,
    config: T4MoEModelConfig,
    final: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model with simple timing.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        config: Model configuration
        final: Whether this is the final evaluation
        
    Returns:
        Evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_aux_loss = 0.0
    correct = 0
    total = 0
    
    device = next(model.parameters()).device
    
    # Limit evaluation batches for speed (use config.eval_steps or default to 50)
    max_eval_batches = getattr(config, 'eval_steps', 50)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_eval_batches:
                break
            
            x, y = x.to(device), y.to(device)
            
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
    
    # Calculate metrics based on actual number of batches processed
    num_batches_processed = min(max_eval_batches, len(val_loader))
    avg_loss = total_loss / num_batches_processed
    avg_aux_loss = total_aux_loss / num_batches_processed
    accuracy = correct / total if total > 0 else 0.0
    perplexity = math.exp(min(avg_loss, 20))
    
    metrics = {
        'val_loss': avg_loss,
        'val_aux_loss': avg_aux_loss,
        'val_accuracy': accuracy,
        'val_perplexity': perplexity
    }
    
    prefix = "📊 Final" if final else "Validation"
    print(f"\n{prefix} Evaluation:")
    print(f"   Val Loss: {avg_loss:.4f}")
    print(f"   Val Accuracy: {accuracy:.4f}")
    print(f"   Val Perplexity: {perplexity:.2f}")
    
    return metrics


