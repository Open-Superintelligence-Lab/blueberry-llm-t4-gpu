#!/usr/bin/env python3
"""
Record Tracking System for Training Speed Experiments

This module implements the same record tracking technique as reference_measurement.py
for tracking training speed records and performance benchmarks.
"""

import os
import sys
import uuid
import time
import json
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch

# Read the code of this file ASAP for logging (same as reference)
with open(sys.argv[0]) as f:
    code = f.read()


@dataclass
class SpeedRecord:
    """A speed record entry."""
    experiment_name: str
    steps_per_second: float
    total_time_seconds: float
    final_loss: float
    final_accuracy: float
    memory_usage_mb: float
    config: Dict[str, Any]
    timestamp: str
    run_id: str
    hardware_info: Dict[str, Any]
    code_hash: str


@dataclass
class PerformanceBenchmark:
    """Performance benchmark data."""
    experiment_name: str
    step_times: List[float]
    forward_times: List[float]
    backward_times: List[float]
    optimizer_times: List[float]
    data_loading_times: List[float]
    memory_usage_history: List[float]
    gpu_memory_history: List[float]
    config: Dict[str, Any]
    timestamp: str
    run_id: str


class RecordTracker:
    """Main record tracking system."""
    
    def __init__(self, experiment_name: str = "training_speed_experiment"):
        self.experiment_name = experiment_name
        self.run_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        self.logfile = f"logs/{self.run_id}.txt"
        
        # Initialize logging
        self._log_setup()
        
        # Record storage
        self.current_record: Optional[SpeedRecord] = None
        self.benchmark_data: Optional[PerformanceBenchmark] = None
        
    def _log_setup(self):
        """Setup logging system (same as reference)."""
        print(f"📊 Record tracking started: {self.logfile}")
        
        # Log the code (same as reference)
        self.log(f"# Training Speed Experiment Code")
        self.log("=" * 100)
        self.log(code)
        self.log("=" * 100)
        
        # Log environment info
        self.log(f"Running Python {sys.version}")
        self.log(f"Running PyTorch {torch.version.__version__}")
        if torch.cuda.is_available():
            self.log(f"CUDA available: {torch.version.cuda}")
            self.log(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.log(f"GPU {i}: {props.name} ({props.total_memory // 1024 // 1024} MB)")
        else:
            self.log("CUDA not available - running on CPU")
        
        # Log nvidia-smi if available
        try:
            import subprocess
            nvidia_smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
            self.log("nvidia-smi output:")
            self.log(nvidia_smi)
        except:
            self.log("nvidia-smi not available")
        
        self.log("=" * 100)
    
    def log(self, message: str, console: bool = True):
        """Log a message (same as reference print0)."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        if console:
            print(log_message)
        
        with open(self.logfile, "a") as f:
            f.write(log_message + "\n")
    
    def start_experiment(self, config: Dict[str, Any]):
        """Start tracking an experiment."""
        self.log(f"🚀 Starting experiment: {config.get('name', 'unknown')}")
        self.log(f"   Description: {config.get('description', 'No description')}")
        self.log(f"   Config: {json.dumps(config, indent=2)}")
        
        # Get hardware info
        hardware_info = self._get_hardware_info()
        
        self.current_record = SpeedRecord(
            experiment_name=config.get('name', 'unknown'),
            steps_per_second=0.0,
            total_time_seconds=0.0,
            final_loss=0.0,
            final_accuracy=0.0,
            memory_usage_mb=0.0,
            config=config,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            run_id=self.run_id,
            hardware_info=hardware_info,
            code_hash=hash(code)
        )
    
    def update_training_progress(self, step: int, total_steps: int, loss: float, 
                                step_time_ms: float, memory_mb: float):
        """Update training progress (same timing format as reference)."""
        elapsed_time_ms = (time.time() - self.start_time) * 1000
        step_avg_ms = elapsed_time_ms / max(step, 1)
        
        # Log in same format as reference
        self.log(f"step:{step}/{total_steps} loss:{loss:.4f} "
                f"train_time:{elapsed_time_ms:.0f}ms step_avg:{step_avg_ms:.2f}ms "
                f"memory:{memory_mb:.1f}MB", console=True)
    
    def update_validation(self, step: int, val_loss: float, val_accuracy: float, 
                         val_time_ms: float):
        """Update validation results."""
        self.log(f"step:{step} val_loss:{val_loss:.4f} val_acc:{val_accuracy:.4f} "
                f"val_time:{val_time_ms:.0f}ms", console=True)
    
    def finalize_experiment(self, final_metrics: Dict[str, Any], 
                           timing_summary: Dict[str, Any]):
        """Finalize the experiment record."""
        if not self.current_record:
            return
        
        # Update record with final results
        self.current_record.steps_per_second = timing_summary.get('steps_per_second', 0.0)
        self.current_record.total_time_seconds = timing_summary.get('total_time_seconds', 0.0)
        self.current_record.final_loss = final_metrics.get('val_loss', 0.0)
        self.current_record.final_accuracy = final_metrics.get('val_accuracy', 0.0)
        self.current_record.memory_usage_mb = final_metrics.get('memory_usage_mb', 0.0)
        
        # Log final results
        self.log(f"✅ Experiment completed: {self.current_record.experiment_name}")
        self.log(f"   Steps/sec: {self.current_record.steps_per_second:.2f}")
        self.log(f"   Total time: {self.current_record.total_time_seconds:.2f}s")
        self.log(f"   Final loss: {self.current_record.final_loss:.4f}")
        self.log(f"   Final accuracy: {self.current_record.final_accuracy:.4f}")
        self.log(f"   Memory usage: {self.current_record.memory_usage_mb:.1f} MB")
        
        # Save record
        self._save_record()
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        info = {
            "python_version": sys.version,
            "pytorch_version": torch.version.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpus": []
            })
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "name": props.name,
                    "memory_mb": props.total_memory // 1024 // 1024,
                    "compute_capability": f"{props.major}.{props.minor}"
                })
        
        return info
    
    def _save_record(self):
        """Save the current record."""
        if not self.current_record:
            return
        
        # Save individual record
        record_file = f"logs/{self.run_id}_record.json"
        with open(record_file, "w") as f:
            json.dump(asdict(self.current_record), f, indent=2)
        
        # Update master records file
        self._update_master_records()
        
        self.log(f"💾 Record saved to {record_file}")
    
    def _update_master_records(self):
        """Update the master records file."""
        master_file = "logs/speed_records.json"
        
        # Load existing records
        records = []
        if os.path.exists(master_file):
            try:
                with open(master_file, "r") as f:
                    records = json.load(f)
            except:
                records = []
        
        # Add current record
        if self.current_record:
            records.append(asdict(self.current_record))
        
        # Save updated records
        with open(master_file, "w") as f:
            json.dump(records, f, indent=2)
        
        self.log(f"📊 Master records updated: {master_file}")
    
    def get_speed_records(self) -> List[SpeedRecord]:
        """Get all speed records."""
        master_file = "logs/speed_records.json"
        
        if not os.path.exists(master_file):
            return []
        
        try:
            with open(master_file, "r") as f:
                records_data = json.load(f)
            return [SpeedRecord(**record) for record in records_data]
        except:
            return []
    
    def print_records_summary(self):
        """Print a summary of all records."""
        records = self.get_speed_records()
        
        if not records:
            self.log("No speed records found.")
            return
        
        # Sort by steps per second
        records.sort(key=lambda x: x.steps_per_second, reverse=True)
        
        self.log("\n🏆 SPEED RECORDS SUMMARY")
        self.log("=" * 100)
        self.log(f"{'Rank':<4} {'Experiment':<25} {'Steps/s':<10} {'Time(s)':<10} {'Loss':<8} {'Memory(MB)':<12} {'Date':<20}")
        self.log("-" * 100)
        
        for i, record in enumerate(records[:10]):  # Top 10
            self.log(f"{i+1:<4} {record.experiment_name:<25} {record.steps_per_second:<10.2f} "
                    f"{record.total_time_seconds:<10.2f} {record.final_loss:<8.4f} "
                    f"{record.memory_usage_mb:<12.1f} {record.timestamp:<20}")
        
        # Show best record
        if records:
            best = records[0]
            self.log(f"\n🥇 BEST RECORD: {best.experiment_name}")
            self.log(f"   Speed: {best.steps_per_second:.2f} steps/second")
            self.log(f"   Time: {best.total_time_seconds:.2f} seconds")
            self.log(f"   Loss: {best.final_loss:.4f}")
            self.log(f"   Date: {best.timestamp}")
            self.log(f"   Run ID: {best.run_id}")
    
    def save_benchmark_data(self, benchmark: PerformanceBenchmark):
        """Save detailed benchmark data."""
        benchmark_file = f"logs/{self.run_id}_benchmark.json"
        with open(benchmark_file, "w") as f:
            json.dump(asdict(benchmark), f, indent=2)
        
        self.log(f"📈 Benchmark data saved to {benchmark_file}")
    
    def cleanup(self):
        """Cleanup and finalize logging."""
        total_time = time.time() - self.start_time
        self.log(f"\n🏁 Record tracking completed in {total_time:.2f} seconds")
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() // 1024 // 1024
            reserved_memory = torch.cuda.max_memory_reserved() // 1024 // 1024
            self.log(f"Peak GPU memory: {peak_memory} MB allocated, {reserved_memory} MB reserved")
        
        self.log(f"📁 All logs saved to: {self.logfile}")
        self.log(f"🆔 Run ID: {self.run_id}")


# Global tracker instance
_tracker: Optional[RecordTracker] = None


def get_tracker() -> RecordTracker:
    """Get the global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = RecordTracker()
    return _tracker


def reset_tracker():
    """Reset the global tracker."""
    global _tracker
    if _tracker:
        _tracker.cleanup()
    _tracker = None


def log_record(message: str, console: bool = True):
    """Log a message using the global tracker."""
    tracker = get_tracker()
    tracker.log(message, console)
