#!/usr/bin/env python3
"""
Enhanced Training Timer for precise timing measurements.

This module provides detailed timing measurements for training steps,
including GPU synchronization and memory usage tracking.
"""

import time
import torch
import psutil
import threading
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingMetrics:
    """Container for timing metrics."""
    step_times: List[float] = field(default_factory=list)
    forward_times: List[float] = field(default_factory=list)
    backward_times: List[float] = field(default_factory=list)
    optimizer_times: List[float] = field(default_factory=list)
    data_loading_times: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    gpu_memory_mb: List[float] = field(default_factory=list)
    
    def add_step_time(self, time_seconds: float):
        """Add a step timing measurement."""
        self.step_times.append(time_seconds)
    
    def add_forward_time(self, time_seconds: float):
        """Add a forward pass timing measurement."""
        self.forward_times.append(time_seconds)
    
    def add_backward_time(self, time_seconds: float):
        """Add a backward pass timing measurement."""
        self.backward_times.append(time_seconds)
    
    def add_optimizer_time(self, time_seconds: float):
        """Add an optimizer step timing measurement."""
        self.optimizer_times.append(time_seconds)
    
    def add_data_loading_time(self, time_seconds: float):
        """Add a data loading timing measurement."""
        self.data_loading_times.append(time_seconds)
    
    def add_memory_usage(self, memory_mb: float):
        """Add a memory usage measurement."""
        self.memory_usage_mb.append(memory_mb)
    
    def add_gpu_memory_usage(self, memory_mb: float):
        """Add a GPU memory usage measurement."""
        self.gpu_memory_mb.append(memory_mb)
    
    def get_average_step_time(self) -> float:
        """Get average step time."""
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0.0
    
    def get_steps_per_second(self) -> float:
        """Get steps per second."""
        avg_time = self.get_average_step_time()
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_total_time(self) -> float:
        """Get total training time."""
        return sum(self.step_times)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if not self.memory_usage_mb:
            return {"avg_memory_mb": 0.0, "max_memory_mb": 0.0}
        
        return {
            "avg_memory_mb": sum(self.memory_usage_mb) / len(self.memory_usage_mb),
            "max_memory_mb": max(self.memory_usage_mb)
        }
    
    def get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics."""
        if not self.gpu_memory_mb:
            return {"avg_gpu_memory_mb": 0.0, "max_gpu_memory_mb": 0.0}
        
        return {
            "avg_gpu_memory_mb": sum(self.gpu_memory_mb) / len(self.gpu_memory_mb),
            "max_gpu_memory_mb": max(self.gpu_memory_mb)
        }


class TrainingTimer:
    """Enhanced training timer with detailed measurements."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = TimingMetrics()
        self.current_step_start = None
        self.monitoring = False
        self.monitor_thread = None
        
    def start_step(self):
        """Start timing a training step."""
        self.current_step_start = time.perf_counter()
        
        # Start memory monitoring if not already running
        if not self.monitoring:
            self.start_memory_monitoring()
    
    def end_step(self):
        """End timing a training step."""
        if self.current_step_start is None:
            return
        
        step_time = time.perf_counter() - self.current_step_start
        self.metrics.add_step_time(step_time)
        
        # Record memory usage
        self._record_memory_usage()
        
        self.current_step_start = None
    
    @contextmanager
    def time_forward(self):
        """Context manager for timing forward pass."""
        start_time = time.perf_counter()
        yield
        forward_time = time.perf_counter() - start_time
        self.metrics.add_forward_time(forward_time)
    
    @contextmanager
    def time_backward(self):
        """Context manager for timing backward pass."""
        start_time = time.perf_counter()
        yield
        backward_time = time.perf_counter() - start_time
        self.metrics.add_backward_time(backward_time)
    
    @contextmanager
    def time_optimizer(self):
        """Context manager for timing optimizer step."""
        start_time = time.perf_counter()
        yield
        optimizer_time = time.perf_counter() - start_time
        self.metrics.add_optimizer_time(optimizer_time)
    
    @contextmanager
    def time_data_loading(self):
        """Context manager for timing data loading."""
        start_time = time.perf_counter()
        yield
        data_time = time.perf_counter() - start_time
        self.metrics.add_data_loading_time(data_time)
    
    def start_memory_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_memory_monitoring(self):
        """Stop background memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _memory_monitor_loop(self):
        """Background memory monitoring loop."""
        while self.monitoring:
            self._record_memory_usage()
            time.sleep(0.1)  # Monitor every 100ms
    
    def _record_memory_usage(self):
        """Record current memory usage."""
        # CPU memory
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics.add_memory_usage(cpu_memory_mb)
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            self.metrics.add_gpu_memory_usage(gpu_memory_mb)
    
    def get_summary(self) -> Dict[str, any]:
        """Get a summary of timing metrics."""
        memory_stats = self.metrics.get_memory_stats()
        gpu_memory_stats = self.metrics.get_gpu_memory_stats()
        
        return {
            "total_steps": len(self.metrics.step_times),
            "total_time_seconds": self.metrics.get_total_time(),
            "average_step_time": self.metrics.get_average_step_time(),
            "steps_per_second": self.metrics.get_steps_per_second(),
            "average_forward_time": sum(self.metrics.forward_times) / len(self.metrics.forward_times) if self.metrics.forward_times else 0.0,
            "average_backward_time": sum(self.metrics.backward_times) / len(self.metrics.backward_times) if self.metrics.backward_times else 0.0,
            "average_optimizer_time": sum(self.metrics.optimizer_times) / len(self.metrics.optimizer_times) if self.metrics.optimizer_times else 0.0,
            "average_data_loading_time": sum(self.metrics.data_loading_times) / len(self.metrics.data_loading_times) if self.metrics.data_loading_times else 0.0,
            "memory_stats": memory_stats,
            "gpu_memory_stats": gpu_memory_stats
        }
    
    def print_summary(self):
        """Print a formatted summary of timing metrics."""
        summary = self.get_summary()
        
        print("\n⏱️  TRAINING TIMING SUMMARY")
        print("=" * 50)
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Total Time: {summary['total_time_seconds']:.2f} seconds")
        print(f"Average Step Time: {summary['average_step_time']:.4f} seconds")
        print(f"Steps per Second: {summary['steps_per_second']:.2f}")
        print(f"Average Forward Time: {summary['average_forward_time']:.4f} seconds")
        print(f"Average Backward Time: {summary['average_backward_time']:.4f} seconds")
        print(f"Average Optimizer Time: {summary['average_optimizer_time']:.4f} seconds")
        print(f"Average Data Loading Time: {summary['average_data_loading_time']:.4f} seconds")
        print(f"Average CPU Memory: {summary['memory_stats']['avg_memory_mb']:.1f} MB")
        print(f"Max CPU Memory: {summary['memory_stats']['max_memory_mb']:.1f} MB")
        print(f"Average GPU Memory: {summary['gpu_memory_stats']['avg_gpu_memory_mb']:.1f} MB")
        print(f"Max GPU Memory: {summary['gpu_memory_stats']['max_gpu_memory_mb']:.1f} MB")


# Global timer instance
_timer: Optional[TrainingTimer] = None


def get_timer() -> TrainingTimer:
    """Get the global timer instance."""
    global _timer
    if _timer is None:
        _timer = TrainingTimer()
    return _timer


def reset_timer():
    """Reset the global timer."""
    global _timer
    if _timer:
        _timer.stop_memory_monitoring()
    _timer = None


def time_training_step(func: Callable) -> Callable:
    """Decorator to time training steps."""
    def wrapper(*args, **kwargs):
        timer = get_timer()
        timer.start_step()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            timer.end_step()
    return wrapper


def time_function(func: Callable) -> Callable:
    """Decorator to time any function."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            print(f"⏱️  {func.__name__} took {end_time - start_time:.4f} seconds")
    return wrapper
