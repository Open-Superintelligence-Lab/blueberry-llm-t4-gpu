#!/usr/bin/env python3
"""
Record Tracking System for Training Speed Experiments

Simple system for tracking training speed records and performance benchmarks.
"""

import os
import sys
import uuid
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import torch


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
        
    def _log_setup(self):
        """Setup logging system."""
        print(f"📊 Record tracking started: {self.logfile}")
        
        # Log basic environment info
        self.log(f"Python: {sys.version.split()[0]}")
        self.log(f"PyTorch: {torch.version.__version__}")
        if torch.cuda.is_available():
            self.log(f"CUDA: {torch.version.cuda}")
            self.log(f"GPU: {torch.cuda.get_device_name()}")
        else:
            self.log("CUDA not available - running on CPU")
    
    def log(self, message: str, console: bool = True):
        """Log a message."""
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
            code_hash=hash(str(config))
        )
    
    def update_training_progress(self, step: int, total_steps: int, loss: float, 
                                step_time_ms: float, memory_mb: float):
        """Update training progress."""
        elapsed_time_ms = (time.time() - self.start_time) * 1000
        step_avg_ms = elapsed_time_ms / max(step, 1)
        
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
        """Get basic hardware information."""
        info = {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.version.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_mb": torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
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
        """Print a simple summary of all records."""
        records = self.get_speed_records()
        
        if not records:
            self.log("No speed records found.")
            return
        
        # Sort by steps per second
        records.sort(key=lambda x: x.steps_per_second, reverse=True)
        
        self.log(f"\n🏆 Speed Records: {len(records)} experiments")
        
        for i, record in enumerate(records[:5]):  # Top 5
            self.log(f"{i+1}. {record.experiment_name}: {record.steps_per_second:.2f} steps/sec")
        
        if records:
            best = records[0]
            self.log(f"\n🥇 Best: {best.experiment_name} ({best.steps_per_second:.2f} steps/sec)")
    
    def cleanup(self):
        """Cleanup and finalize logging."""
        total_time = time.time() - self.start_time
        self.log(f"\n🏁 Record tracking completed in {total_time:.2f} seconds")
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() // 1024 // 1024
            self.log(f"Peak GPU memory: {peak_memory} MB")
        
        self.log(f"📁 Logs saved to: {self.logfile}")


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
