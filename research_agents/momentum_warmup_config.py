#!/usr/bin/env python3
"""
Configuration for Momentum Warmup Research Experiments

This file contains all the experimental configurations and schedules
for testing different momentum warmup strategies with the Muon optimizer.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import math


@dataclass
class MomentumWarmupExperimentConfig:
    """Main configuration for momentum warmup experiments"""
    
    # Experiment metadata
    experiment_name: str = "momentum_warmup_t4_research"
    description: str = "Research optimal momentum warmup schedule for Muon optimizer on T4 hardware"
    
    # Hardware and performance tracking
    target_hardware: str = "T4_GPU"
    monitor_gpu_utilization: bool = True
    track_memory_usage: bool = True
    track_training_time: bool = True
    
    # Model configuration (smaller for faster experiments)
    model_config: Dict[str, Any] = None
    
    # Training configuration
    training_config: Dict[str, Any] = None
    
    # Evaluation configuration
    evaluation_config: Dict[str, Any] = None
    
    # Momentum warmup schedules to test
    momentum_schedules: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = {
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 4,
                'd_ff': 1024,
                'batch_size': 16,
                'max_seq_len': 256,
                'num_experts': 4,
                'expert_top_k': 2,
                'vocab_size': None,  # Will be set from data
            }
        
        if self.training_config is None:
            self.training_config = {
                'max_steps': 1000,
                'base_lr': 0.01,
                'weight_decay': 0.1,
                'grad_clip': 1.0,
                'use_amp': True,
                'gradient_accumulation_steps': 4,
                'num_runs_per_schedule': 3,
            }
        
        if self.evaluation_config is None:
            self.evaluation_config = {
                'eval_every': 50,
                'eval_steps': 20,
                'metrics_to_track': [
                    'val_loss', 'val_accuracy', 'val_perplexity',
                    'train_loss', 'momentum_value', 'learning_rate',
                    'gpu_memory_usage', 'training_time_per_step'
                ]
            }
        
        if self.momentum_schedules is None:
            self.momentum_schedules = self._get_default_momentum_schedules()
    
    def _get_default_momentum_schedules(self) -> List[Dict[str, Any]]:
        """Get the default set of momentum schedules to test"""
        return [
            {
                'name': 'fixed_095',
                'description': 'Fixed momentum = 0.95 (current baseline)',
                'type': 'fixed',
                'momentum': 0.95,
                'warmup_steps': 0,
            },
            {
                'name': 'fixed_090',
                'description': 'Fixed momentum = 0.90 (lower baseline)',
                'type': 'fixed',
                'momentum': 0.90,
                'warmup_steps': 0,
            },
            {
                'name': 'fixed_099',
                'description': 'Fixed momentum = 0.99 (higher baseline)',
                'type': 'fixed',
                'momentum': 0.99,
                'warmup_steps': 0,
            },
            {
                'name': 'linear_warmup',
                'description': 'Linear warmup from 0 to 0.95',
                'type': 'linear',
                'start_momentum': 0.0,
                'end_momentum': 0.95,
                'warmup_steps': 250,  # 25% of training
            },
            {
                'name': 'cosine_warmup',
                'description': 'Cosine warmup from 0 to 0.95',
                'type': 'cosine',
                'start_momentum': 0.0,
                'end_momentum': 0.95,
                'warmup_steps': 250,
            },
            {
                'name': 'exponential_warmup',
                'description': 'Exponential warmup from 0.1 to 0.95',
                'type': 'exponential',
                'start_momentum': 0.1,
                'end_momentum': 0.95,
                'warmup_steps': 250,
                'exponent': 2.0,
            },
            {
                'name': 'step_warmup',
                'description': 'Step-wise warmup: 0.5 → 0.7 → 0.85 → 0.95',
                'type': 'step',
                'steps': [0.5, 0.7, 0.85, 0.95],
                'warmup_steps': 250,
            },
            {
                'name': 'adaptive_warmup',
                'description': 'Adaptive warmup based on gradient variance',
                'type': 'adaptive',
                'base_momentum': 0.95,
                'warmup_steps': 250,
                'adaptation_factor': 0.1,
            },
            {
                'name': 'delayed_warmup',
                'description': 'Delayed warmup starting at step 100',
                'type': 'delayed_linear',
                'start_momentum': 0.0,
                'end_momentum': 0.95,
                'warmup_steps': 200,
                'delay_steps': 100,
            },
            {
                'name': 'sigmoid_warmup',
                'description': 'Sigmoid warmup for smooth transition',
                'type': 'sigmoid',
                'start_momentum': 0.0,
                'end_momentum': 0.95,
                'warmup_steps': 250,
                'steepness': 5.0,
            }
        ]


class MomentumScheduleCalculator:
    """Calculator for different momentum warmup schedules"""
    
    @staticmethod
    def calculate_momentum(schedule_config: Dict[str, Any], step: int, max_steps: int) -> float:
        """Calculate momentum value for given step and schedule"""
        schedule_type = schedule_config['type']
        warmup_steps = schedule_config.get('warmup_steps', max_steps // 4)
        
        if schedule_type == 'fixed':
            return schedule_config['momentum']
        
        elif schedule_type == 'linear':
            start_momentum = schedule_config['start_momentum']
            end_momentum = schedule_config['end_momentum']
            if step < warmup_steps:
                progress = step / warmup_steps
                return start_momentum + (end_momentum - start_momentum) * progress
            return end_momentum
        
        elif schedule_type == 'cosine':
            start_momentum = schedule_config['start_momentum']
            end_momentum = schedule_config['end_momentum']
            if step < warmup_steps:
                progress = step / warmup_steps
                cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
                return start_momentum + (end_momentum - start_momentum) * cosine_factor
            return end_momentum
        
        elif schedule_type == 'exponential':
            start_momentum = schedule_config['start_momentum']
            end_momentum = schedule_config['end_momentum']
            exponent = schedule_config.get('exponent', 2.0)
            if step < warmup_steps:
                progress = step / warmup_steps
                exp_factor = math.pow(progress, exponent)
                return start_momentum + (end_momentum - start_momentum) * exp_factor
            return end_momentum
        
        elif schedule_type == 'step':
            steps = schedule_config['steps']
            if step < warmup_steps:
                step_size = warmup_steps // len(steps)
                step_index = min(step // step_size, len(steps) - 1)
                return steps[step_index]
            return steps[-1]
        
        elif schedule_type == 'adaptive':
            # Simplified adaptive - would need gradient variance tracking
            base_momentum = schedule_config['base_momentum']
            if step < warmup_steps:
                progress = step / warmup_steps
                return base_momentum * progress
            return base_momentum
        
        elif schedule_type == 'delayed_linear':
            start_momentum = schedule_config['start_momentum']
            end_momentum = schedule_config['end_momentum']
            delay_steps = schedule_config['delay_steps']
            if step < delay_steps:
                return start_momentum
            elif step < delay_steps + warmup_steps:
                progress = (step - delay_steps) / warmup_steps
                return start_momentum + (end_momentum - start_momentum) * progress
            return end_momentum
        
        elif schedule_type == 'sigmoid':
            start_momentum = schedule_config['start_momentum']
            end_momentum = schedule_config['end_momentum']
            steepness = schedule_config.get('steepness', 5.0)
            if step < warmup_steps:
                progress = step / warmup_steps
                sigmoid_factor = 1 / (1 + math.exp(-steepness * (progress - 0.5)))
                return start_momentum + (end_momentum - start_momentum) * sigmoid_factor
            return end_momentum
        
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")


# Predefined experiment configurations for different scenarios
QUICK_EXPERIMENT = MomentumWarmupExperimentConfig(
    experiment_name="quick_momentum_test",
    training_config={
        'max_steps': 200,
        'num_runs_per_schedule': 2,
        'base_lr': 0.01,
        'weight_decay': 0.1,
        'grad_clip': 1.0,
        'use_amp': True,
        'gradient_accumulation_steps': 4,
    },
    momentum_schedules=[
        {'name': 'fixed_095', 'type': 'fixed', 'momentum': 0.95, 'warmup_steps': 0},
        {'name': 'linear_warmup', 'type': 'linear', 'start_momentum': 0.0, 'end_momentum': 0.95, 'warmup_steps': 50},
        {'name': 'cosine_warmup', 'type': 'cosine', 'start_momentum': 0.0, 'end_momentum': 0.95, 'warmup_steps': 50},
    ]
)

FULL_EXPERIMENT = MomentumWarmupExperimentConfig(
    experiment_name="full_momentum_warmup_research",
    training_config={
        'max_steps': 2000,
        'num_runs_per_schedule': 5,
    }
)

T4_OPTIMIZED_EXPERIMENT = MomentumWarmupExperimentConfig(
    experiment_name="t4_optimized_momentum_research",
    description="Optimized for T4 hardware constraints",
    model_config={
        'd_model': 384,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1536,
        'batch_size': 12,  # Smaller batch for T4 memory
        'max_seq_len': 512,
        'num_experts': 8,
        'expert_top_k': 2,
    },
    training_config={
        'max_steps': 1500,
        'num_runs_per_schedule': 3,
        'use_amp': True,
        'gradient_accumulation_steps': 8,  # Compensate for smaller batch
    }
)
