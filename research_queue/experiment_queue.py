"""Experiment Queue Management System."""

import json
import uuid
import threading
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
from enum import Enum
import os

from research_agents.base_agent import ResearchExperiment


class ExperimentStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentPriority(Enum):
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


class ExperimentQueue:
    """Manages the queue of research experiments."""
    
    def __init__(self, queue_file: str = "research_data/experiment_queue.json"):
        self.queue_file = queue_file
        self.experiments: Dict[str, ResearchExperiment] = {}
        self.queue_order: List[str] = []
        self.currently_running: Optional[str] = None
        self.lock = threading.Lock()
        
        # Load existing queue
        self._load_queue()
    
    def _load_queue(self):
        """Load experiment queue from file."""
        os.makedirs("research_data", exist_ok=True)
        
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, 'r') as f:
                    data = json.load(f)
                
                # Load experiments
                for exp_id, exp_data in data.get("experiments", {}).items():
                    exp_data["created_at"] = datetime.fromisoformat(exp_data["created_at"])
                    if exp_data.get("started_at"):
                        exp_data["started_at"] = datetime.fromisoformat(exp_data["started_at"])
                    if exp_data.get("completed_at"):
                        exp_data["completed_at"] = datetime.fromisoformat(exp_data["completed_at"])
                    
                    experiment = ResearchExperiment(**exp_data)
                    self.experiments[exp_id] = experiment
                
                # Load queue order
                self.queue_order = data.get("queue_order", [])
                self.currently_running = data.get("currently_running")
                
            except Exception as e:
                print(f"Error loading queue: {e}")
                self.experiments = {}
                self.queue_order = []
    
    def _save_queue(self):
        """Save experiment queue to file."""
        with self.lock:
            try:
                # Convert experiments to serializable format
                experiments_data = {}
                for exp_id, experiment in self.experiments.items():
                    exp_dict = asdict(experiment)
                    # Convert datetime objects to ISO strings
                    for key, value in exp_dict.items():
                        if isinstance(value, datetime):
                            exp_dict[key] = value.isoformat()
                    experiments_data[exp_id] = exp_dict
                
                queue_data = {
                    "experiments": experiments_data,
                    "queue_order": self.queue_order,
                    "currently_running": self.currently_running,
                    "last_updated": datetime.now().isoformat()
                }
                
                with open(self.queue_file, 'w') as f:
                    json.dump(queue_data, f, indent=2)
                    
            except Exception as e:
                print(f"Error saving queue: {e}")
    
    def add_experiment(self, experiment: ResearchExperiment) -> str:
        """Add an experiment to the queue."""
        with self.lock:
            # Ensure experiment has an ID
            if not experiment.id:
                experiment.id = str(uuid.uuid4())
            
            # Add to experiments dict
            self.experiments[experiment.id] = experiment
            
            # Add to queue order based on priority
            self._insert_by_priority(experiment.id, experiment.priority)
            
            # Save queue
            self._save_queue()
            
            return experiment.id
    
    def _insert_by_priority(self, experiment_id: str, priority: int):
        """Insert experiment into queue maintaining priority order."""
        # Remove if already in queue
        if experiment_id in self.queue_order:
            self.queue_order.remove(experiment_id)
        
        # Find insertion point
        insert_index = 0
        for i, queued_id in enumerate(self.queue_order):
            if queued_id in self.experiments:
                queued_priority = self.experiments[queued_id].priority
                if priority > queued_priority:
                    insert_index = i
                    break
                insert_index = i + 1
        
        # Insert at the correct position
        self.queue_order.insert(insert_index, experiment_id)
    
    def get_next_experiment(self) -> Optional[ResearchExperiment]:
        """Get the next experiment to run."""
        with self.lock:
            if self.currently_running:
                return None  # Another experiment is running
            
            for exp_id in self.queue_order:
                if exp_id in self.experiments:
                    experiment = self.experiments[exp_id]
                    if experiment.status == ExperimentStatus.PENDING.value:
                        return experiment
            
            return None
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Mark an experiment as running."""
        with self.lock:
            if experiment_id in self.experiments:
                experiment = self.experiments[experiment_id]
                if experiment.status == ExperimentStatus.PENDING.value:
                    experiment.status = ExperimentStatus.RUNNING.value
                    experiment.started_at = datetime.now()
                    self.currently_running = experiment_id
                    self._save_queue()
                    return True
            return False
    
    def complete_experiment(self, experiment_id: str, results: Dict[str, Any] = None) -> bool:
        """Mark an experiment as completed."""
        with self.lock:
            if experiment_id in self.experiments:
                experiment = self.experiments[experiment_id]
                experiment.status = ExperimentStatus.COMPLETED.value
                experiment.completed_at = datetime.now()
                if results:
                    experiment.results = results
                
                if self.currently_running == experiment_id:
                    self.currently_running = None
                
                self._save_queue()
                return True
            return False
    
    def fail_experiment(self, experiment_id: str, error_message: str = None) -> bool:
        """Mark an experiment as failed."""
        with self.lock:
            if experiment_id in self.experiments:
                experiment = self.experiments[experiment_id]
                experiment.status = ExperimentStatus.FAILED.value
                experiment.completed_at = datetime.now()
                if error_message:
                    experiment.error_message = error_message
                
                if self.currently_running == experiment_id:
                    self.currently_running = None
                
                self._save_queue()
                return True
            return False
    
    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel an experiment."""
        with self.lock:
            if experiment_id in self.experiments:
                experiment = self.experiments[experiment_id]
                if experiment.status in [ExperimentStatus.PENDING.value, ExperimentStatus.QUEUED.value]:
                    experiment.status = ExperimentStatus.CANCELLED.value
                    if experiment_id in self.queue_order:
                        self.queue_order.remove(experiment_id)
                    self._save_queue()
                    return True
            return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self.lock:
            status_counts = {}
            for status in ExperimentStatus:
                status_counts[status.value] = 0
            
            for experiment in self.experiments.values():
                status_counts[experiment.status] += 1
            
            return {
                "total_experiments": len(self.experiments),
                "queue_length": len([exp_id for exp_id in self.queue_order 
                                   if exp_id in self.experiments and 
                                   self.experiments[exp_id].status == ExperimentStatus.PENDING.value]),
                "currently_running": self.currently_running,
                "status_counts": status_counts,
                "queue_order": self.queue_order[:10]  # First 10 in queue
            }
    
    def get_experiment(self, experiment_id: str) -> Optional[ResearchExperiment]:
        """Get a specific experiment by ID."""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self, 
                        status: Optional[ExperimentStatus] = None,
                        limit: int = 50) -> List[ResearchExperiment]:
        """List experiments with optional filtering."""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [exp for exp in experiments if exp.status == status.value]
        
        # Sort by priority and creation time
        experiments.sort(key=lambda x: (-x.priority, x.created_at))
        
        return experiments[:limit]
    
    def reorder_experiment(self, experiment_id: str, new_priority: int) -> bool:
        """Reorder an experiment in the queue by changing its priority."""
        with self.lock:
            if experiment_id in self.experiments:
                experiment = self.experiments[experiment_id]
                experiment.priority = new_priority
                self._insert_by_priority(experiment_id, new_priority)
                self._save_queue()
                return True
            return False
    
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get statistics about experiments in the queue."""
        with self.lock:
            if not self.experiments:
                return {"total": 0, "average_duration": 0, "priority_distribution": {}}
            
            total_duration = 0
            completed_count = 0
            priority_dist = {}
            
            for experiment in self.experiments.values():
                if experiment.status == ExperimentStatus.COMPLETED.value:
                    total_duration += experiment.estimated_duration
                    completed_count += 1
                
                priority_dist[experiment.priority] = priority_dist.get(experiment.priority, 0) + 1
            
            return {
                "total": len(self.experiments),
                "completed": completed_count,
                "average_duration": total_duration / max(completed_count, 1),
                "priority_distribution": priority_dist,
                "estimated_remaining_time": sum(
                    exp.estimated_duration for exp in self.experiments.values()
                    if exp.status == ExperimentStatus.PENDING.value
                )
            }
    
    def clear_completed_experiments(self) -> int:
        """Remove completed experiments from the queue."""
        with self.lock:
            completed_ids = [
                exp_id for exp_id, exp in self.experiments.items()
                if exp.status == ExperimentStatus.COMPLETED.value
            ]
            
            for exp_id in completed_ids:
                del self.experiments[exp_id]
                if exp_id in self.queue_order:
                    self.queue_order.remove(exp_id)
            
            self._save_queue()
            return len(completed_ids)
