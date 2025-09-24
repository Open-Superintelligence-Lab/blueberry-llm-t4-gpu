"""Experiment Runner - executes research experiments on T4 GPU."""

import os
import json
import time
import threading
import subprocess
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch

from research_agents.base_agent import ResearchExperiment
from research_queue.experiment_queue import ExperimentQueue, ExperimentStatus
from research_data.database import ResearchDatabase
from configs.moe_config import MoEModelConfig
from models.moe_llm import MoEMinimalLLM
from training.trainer import train_moe_model
from data.loader import create_data_loaders


class ExperimentRunner:
    """Runs research experiments on T4 GPU."""
    
    def __init__(self, 
                 queue: ExperimentQueue,
                 database: ResearchDatabase,
                 gpu_monitor: bool = True):
        self.queue = queue
        self.database = database
        self.gpu_monitor = gpu_monitor
        self.logger = logging.getLogger("experiment_runner")
        
        # GPU information
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        
        # Running state
        self.is_running = False
        self.current_experiment = None
        self.run_thread = None
        
        self.logger.info(f"Experiment runner initialized on {self.gpu_name} ({self.gpu_memory:.1f}GB)")
    
    def start(self):
        """Start the experiment runner."""
        if self.is_running:
            self.logger.warning("Experiment runner is already running")
            return
        
        self.is_running = True
        self.run_thread = threading.Thread(target=self._run_experiments, daemon=True)
        self.run_thread.start()
        self.logger.info("Experiment runner started")
    
    def stop(self):
        """Stop the experiment runner."""
        self.is_running = False
        if self.run_thread:
            self.run_thread.join(timeout=5)
        self.logger.info("Experiment runner stopped")
    
    def _run_experiments(self):
        """Main experiment running loop."""
        while self.is_running:
            try:
                # Get next experiment
                experiment = self.queue.get_next_experiment()
                
                if experiment is None:
                    time.sleep(5)  # Wait 5 seconds before checking again
                    continue
                
                # Start experiment
                if self.queue.start_experiment(experiment.id):
                    self.current_experiment = experiment
                    self.logger.info(f"Starting experiment: {experiment.title}")
                    
                    # Run the experiment
                    success, results = self._execute_experiment(experiment)
                    
                    # Complete or fail experiment
                    if success:
                        self.queue.complete_experiment(experiment.id, results)
                        self.database.save_experiment(experiment)
                        self.logger.info(f"Completed experiment: {experiment.title}")
                    else:
                        error_msg = results.get('error_message', 'Unknown error')
                        self.queue.fail_experiment(experiment.id, error_msg)
                        self.logger.error(f"Failed experiment: {experiment.title} - {error_msg}")
                    
                    self.current_experiment = None
                
            except Exception as e:
                self.logger.error(f"Error in experiment runner: {e}")
                if self.current_experiment:
                    self.queue.fail_experiment(self.current_experiment.id, str(e))
                    self.current_experiment = None
                time.sleep(10)  # Wait before retrying
    
    def _execute_experiment(self, experiment: ResearchExperiment) -> tuple[bool, Dict[str, Any]]:
        """Execute a single experiment."""
        try:
            # Create experiment-specific configuration
            config = self._create_experiment_config(experiment)
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(config)
            
            # Run training
            start_time = time.time()
            model, final_eval = train_moe_model(config, train_loader, val_loader)
            execution_time = time.time() - start_time
            
            # Collect results
            results = {
                'final_evaluation': final_eval,
                'execution_time': execution_time,
                'config_used': self._config_to_dict(config),
                'gpu_info': {
                    'gpu_name': self.gpu_name,
                    'gpu_memory_gb': self.gpu_memory,
                    'device': str(self.device)
                },
                'experiment_parameters': experiment.parameters,
                'completed_at': datetime.now().isoformat()
            }
            
            # Save detailed results to database
            for metric_name, metric_value in final_eval.items():
                self.database.save_result(experiment.id, metric_name, metric_value)
            
            # Save model if specified in parameters
            if experiment.parameters.get('save_model', False):
                model_path = f"models/experiment_{experiment.id}.pt"
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), model_path)
                results['model_saved_path'] = model_path
            
            return True, results
            
        except Exception as e:
            self.logger.error(f"Error executing experiment {experiment.id}: {e}")
            return False, {'error_message': str(e)}
    
    def _create_experiment_config(self, experiment: ResearchExperiment) -> MoEModelConfig:
        """Create experiment-specific configuration."""
        # Start with default config
        config = MoEModelConfig()
        
        # Apply experiment parameters
        for param_name, param_value in experiment.parameters.items():
            if hasattr(config, param_name):
                setattr(config, param_name, param_value)
            else:
                self.logger.warning(f"Unknown parameter: {param_name}")
        
        # Ensure T4 GPU compatibility
        config = self._optimize_for_t4(config)
        
        return config
    
    def _optimize_for_t4(self, config: MoEModelConfig) -> MoEModelConfig:
        """Optimize configuration for T4 GPU constraints."""
        # T4 GPU has 16GB VRAM, so we need to be conservative
        
        # Adjust batch size based on model size
        if config.d_model > 512:
            config.batch_size = min(config.batch_size, 16)
        else:
            config.batch_size = min(config.batch_size, 32)
        
        # Adjust sequence length for memory
        if config.max_seq_len > 1024:
            config.max_seq_len = 1024
        
        # Limit number of experts for memory efficiency
        config.num_experts = min(config.num_experts, 16)
        
        # Ensure gradient accumulation is reasonable
        if config.gradient_accumulation_steps > 8:
            config.gradient_accumulation_steps = 8
        
        return config
    
    def _config_to_dict(self, config: MoEModelConfig) -> Dict[str, Any]:
        """Convert config to dictionary for storage."""
        return {
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'd_ff': config.d_ff,
            'batch_size': config.batch_size,
            'max_steps': config.max_steps,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'muon_lr': config.muon_lr,
            'max_seq_len': config.max_seq_len,
            'num_documents': config.num_documents,
            'max_tokens': config.max_tokens,
            'eval_every': config.eval_every,
            'eval_steps': config.eval_steps,
            'weight_decay': config.weight_decay,
            'dropout': config.dropout,
            'grad_clip': config.grad_clip,
            'use_amp': config.use_amp,
            'num_experts': config.num_experts,
            'expert_top_k': config.expert_top_k,
            'load_balancing_weight': config.load_balancing_weight
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current runner status."""
        return {
            'is_running': self.is_running,
            'current_experiment': self.current_experiment.id if self.current_experiment else None,
            'current_experiment_title': self.current_experiment.title if self.current_experiment else None,
            'gpu_name': self.gpu_name,
            'gpu_memory_gb': self.gpu_memory,
            'device': str(self.device),
            'queue_status': self.queue.get_queue_status()
        }
    
    def run_single_experiment(self, experiment_id: str) -> tuple[bool, Dict[str, Any]]:
        """Run a single experiment by ID."""
        experiment = self.queue.get_experiment(experiment_id)
        if not experiment:
            return False, {'error_message': f'Experiment {experiment_id} not found'}
        
        if experiment.status != ExperimentStatus.PENDING.value:
            return False, {'error_message': f'Experiment {experiment_id} is not pending'}
        
        # Start experiment
        if not self.queue.start_experiment(experiment_id):
            return False, {'error_message': f'Failed to start experiment {experiment_id}'}
        
        try:
            # Execute experiment
            success, results = self._execute_experiment(experiment)
            
            # Complete or fail
            if success:
                self.queue.complete_experiment(experiment_id, results)
                self.database.save_experiment(experiment)
            else:
                error_msg = results.get('error_message', 'Unknown error')
                self.queue.fail_experiment(experiment_id, error_msg)
            
            return success, results
            
        except Exception as e:
            self.queue.fail_experiment(experiment_id, str(e))
            return False, {'error_message': str(e)}
    
    def cancel_current_experiment(self) -> bool:
        """Cancel the currently running experiment."""
        if not self.current_experiment:
            return False
        
        experiment_id = self.current_experiment.id
        self.queue.fail_experiment(experiment_id, "Cancelled by user")
        self.current_experiment = None
        
        self.logger.info(f"Cancelled experiment: {experiment_id}")
        return True
    
    def get_gpu_utilization(self) -> Dict[str, Any]:
        """Get current GPU utilization."""
        if not torch.cuda.is_available():
            return {'available': False}
        
        try:
            # Get GPU memory info
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            
            # Try to get utilization via nvidia-smi if available
            utilization = None
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    utilization = int(result.stdout.strip())
            except:
                pass
            
            return {
                'available': True,
                'memory_allocated_gb': round(allocated, 2),
                'memory_reserved_gb': round(reserved, 2),
                'memory_total_gb': round(self.gpu_memory, 2),
                'memory_usage_percent': round((allocated / self.gpu_memory) * 100, 1),
                'utilization_percent': utilization
            }
            
        except Exception as e:
            self.logger.error(f"Error getting GPU utilization: {e}")
            return {'available': False, 'error': str(e)}
