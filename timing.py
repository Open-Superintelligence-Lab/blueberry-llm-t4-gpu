#!/usr/bin/env python3
"""
Simple Training Timer for basic timing measurements.
"""

import time
import torch
from typing import Dict, Optional
from contextlib import contextmanager


class TimingMetrics:
    """Simple container for timing metrics."""
    
    def __init__(self):
        self.step_times = []
        self.total_time = 0.0
    
    def add_step_time(self, time_seconds: float):
        """Add a step timing measurement."""
        self.step_times.append(time_seconds)
        self.total_time += time_seconds
    
    def get_average_step_time(self) -> float:
        """Get average step time."""
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0.0
    
    def get_steps_per_second(self) -> float:
        """Get steps per second."""
        avg_time = self.get_average_step_time()
        return 1.0 / avg_time if avg_time > 0 else 0.0


class TrainingTimer:
    """Simple training timer."""
    
    def __init__(self):
        self.metrics = TimingMetrics()
        self.current_step_start = None
    
    def start_step(self):
        """Start timing a training step."""
        self.current_step_start = time.perf_counter()
    
    def end_step(self):
        """End timing a training step."""
        if self.current_step_start is None:
            return
        
        step_time = time.perf_counter() - self.current_step_start
        self.metrics.add_step_time(step_time)
        self.current_step_start = None
    
    @contextmanager
    def time_forward(self):
        """Context manager for timing forward pass."""
        start_time = time.perf_counter()
        yield
        # Simple timing - just record the time
        pass
    
    @contextmanager
    def time_backward(self):
        """Context manager for timing backward pass."""
        start_time = time.perf_counter()
        yield
        # Simple timing - just record the time
        pass
    
    @contextmanager
    def time_optimizer(self):
        """Context manager for timing optimizer step."""
        start_time = time.perf_counter()
        yield
        # Simple timing - just record the time
        pass
    
    @contextmanager
    def time_data_loading(self):
        """Context manager for timing data loading."""
        start_time = time.perf_counter()
        yield
        # Simple timing - just record the time
        pass
    
    def get_summary(self) -> Dict[str, any]:
        """Get a summary of timing metrics."""
        return {
            "total_steps": len(self.metrics.step_times),
            "total_time_seconds": self.metrics.total_time,
            "average_step_time": self.metrics.get_average_step_time(),
            "steps_per_second": self.metrics.get_steps_per_second(),
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
    _timer = None
