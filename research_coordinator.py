"""Research Coordinator - orchestrates the multi-agent research system."""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from research_agents.base_agent import ResearchExperiment
from research_agents.suggestion_agent import SuggestionAgent
from research_agents.review_agent import ReviewAgent
from research_agents.improvement_agent import ImprovementAgent
from research_queue.experiment_queue import ExperimentQueue, ExperimentStatus
from research_data.database import ResearchDatabase
from research_execution.experiment_runner import ExperimentRunner
from dataclasses import asdict


class ResearchCoordinator:
    """Coordinates the multi-agent research system."""
    
    def __init__(self, openrouter_api_key: str):
        self.openrouter_api_key = openrouter_api_key
        self.logger = logging.getLogger("research_coordinator")
        
        # Initialize components
        self.database = ResearchDatabase()
        self.queue = ExperimentQueue()
        self.runner = ExperimentRunner(self.queue, self.database)
        
        # Initialize agents
        self.suggestion_agent = SuggestionAgent(openrouter_api_key)
        self.review_agent = ReviewAgent(openrouter_api_key)
        self.improvement_agent = ImprovementAgent(openrouter_api_key)
        
        # Configuration
        self.config = self._load_config()
        
        self.logger.info("Research coordinator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load coordinator configuration."""
        config_path = "configs/research_coordinator_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        default_config = {
            "auto_review_threshold": 7.0,
            "auto_improve_threshold": 5.0,
            "max_experiments_per_suggestion": 5,
            "auto_start_runner": False,
            "review_timeout_minutes": 30,
            "improvement_iterations": 3
        }
        
        os.makedirs("configs", exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def generate_research_suggestions(self, 
                                    domain: str = None,
                                    focus_area: str = None,
                                    count: int = 5) -> List[ResearchExperiment]:
        """Generate research suggestions using the suggestion agent."""
        self.logger.info(f"Generating {count} research suggestions")
        
        suggestions = self.suggestion_agent.generate_research_suggestions(
            domain=domain,
            focus_area=focus_area,
            count=count
        )
        
        # Save agent interaction
        for suggestion in suggestions:
            self.database.save_agent_interaction(
                suggestion.id,
                "suggestion_agent",
                "generate_suggestion",
                {"domain": domain, "focus_area": focus_area, "count": count},
                asdict(suggestion)
            )
        
        self.logger.info(f"Generated {len(suggestions)} research suggestions")
        return suggestions
    
    def review_experiments(self, experiments: List[ResearchExperiment]) -> List[Dict[str, Any]]:
        """Review experiments using the review agent."""
        self.logger.info(f"Reviewing {len(experiments)} experiments")
        
        reviews = []
        for experiment in experiments:
            review = self.review_agent.review_experiment(experiment)
            reviews.append(review)
            
            # Save review to database
            self.database.save_review(experiment.id, review)
            
            # Save agent interaction
            self.database.save_agent_interaction(
                experiment.id,
                "review_agent",
                "review_experiment",
                asdict(experiment),
                review
            )
        
        self.logger.info(f"Completed reviews for {len(reviews)} experiments")
        return reviews
    
    def improve_experiments(self, 
                          experiments: List[ResearchExperiment],
                          reviews: List[Dict[str, Any]]) -> List[ResearchExperiment]:
        """Improve experiments based on reviews."""
        self.logger.info(f"Improving {len(experiments)} experiments")
        
        improved_experiments = []
        
        for experiment, review in zip(experiments, reviews):
            # Only improve if score is below threshold
            if review.get('weighted_score', 0) < self.config['auto_improve_threshold']:
                improved = self.improvement_agent.improve_experiment(experiment, review)
                improved_experiments.append(improved)
                
                # Save agent interaction
                self.database.save_agent_interaction(
                    improved.id,
                    "improvement_agent",
                    "improve_experiment",
                    {"original_experiment": asdict(experiment), "review": review},
                    asdict(improved)
                )
            else:
                # Keep original if score is good enough
                improved_experiments.append(experiment)
        
        self.logger.info(f"Improved {len(improved_experiments)} experiments")
        return improved_experiments
    
    def add_experiments_to_queue(self, experiments: List[ResearchExperiment]) -> List[str]:
        """Add experiments to the queue."""
        experiment_ids = []
        
        for experiment in experiments:
            # Save to database
            self.database.save_experiment(experiment)
            
            # Add to queue
            exp_id = self.queue.add_experiment(experiment)
            experiment_ids.append(exp_id)
        
        self.logger.info(f"Added {len(experiment_ids)} experiments to queue")
        return experiment_ids
    
    def start_auto_research_cycle(self, 
                                domain: str = None,
                                focus_area: str = None,
                                max_cycles: int = 10) -> Dict[str, Any]:
        """Start an automated research cycle."""
        self.logger.info("Starting automated research cycle")
        
        cycle_results = {
            'cycles_completed': 0,
            'total_suggestions': 0,
            'total_reviews': 0,
            'total_improvements': 0,
            'experiments_added': 0,
            'start_time': datetime.now().isoformat()
        }
        
        try:
            for cycle in range(max_cycles):
                self.logger.info(f"Starting research cycle {cycle + 1}/{max_cycles}")
                
                # Generate suggestions
                suggestions = self.generate_research_suggestions(
                    domain=domain,
                    focus_area=focus_area,
                    count=self.config['max_experiments_per_suggestion']
                )
                cycle_results['total_suggestions'] += len(suggestions)
                
                # Review suggestions
                reviews = self.review_experiments(suggestions)
                cycle_results['total_reviews'] += len(reviews)
                
                # Improve experiments based on reviews
                improved_experiments = self.improve_experiments(suggestions, reviews)
                cycle_results['total_improvements'] += len([exp for exp in improved_experiments 
                                                          if exp.id != exp.metadata.get('original_experiment_id')])
                
                # Add approved experiments to queue
                approved_experiments = [
                    exp for exp, review in zip(improved_experiments, reviews)
                    if review.get('recommendation') in ['approve', 'revise'] and
                       review.get('weighted_score', 0) >= self.config['auto_review_threshold']
                ]
                
                if approved_experiments:
                    exp_ids = self.add_experiments_to_queue(approved_experiments)
                    cycle_results['experiments_added'] += len(exp_ids)
                
                cycle_results['cycles_completed'] += 1
                
                # Stop if no new experiments were added
                if not approved_experiments:
                    self.logger.info("No new experiments approved, stopping cycle")
                    break
        
        except Exception as e:
            self.logger.error(f"Error in auto research cycle: {e}")
            cycle_results['error'] = str(e)
        
        cycle_results['end_time'] = datetime.now().isoformat()
        self.logger.info(f"Completed research cycle: {cycle_results}")
        
        return cycle_results
    
    def start_experiment_runner(self):
        """Start the experiment runner."""
        self.runner.start()
        self.logger.info("Experiment runner started")
    
    def stop_experiment_runner(self):
        """Stop the experiment runner."""
        self.runner.stop()
        self.logger.info("Experiment runner stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            'coordinator_status': 'active',
            'runner_status': self.runner.get_status(),
            'queue_status': self.queue.get_queue_status(),
            'database_statistics': self.database.get_research_statistics(),
            'gpu_utilization': self.runner.get_gpu_utilization(),
            'config': self.config
        }
    
    def run_single_experiment(self, experiment_id: str) -> tuple[bool, Dict[str, Any]]:
        """Run a single experiment."""
        return self.runner.run_single_experiment(experiment_id)
    
    def get_experiment_details(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an experiment."""
        experiment = self.queue.get_experiment(experiment_id)
        if not experiment:
            return None
        
        results = self.database.get_experiment_results(experiment_id)
        reviews = self.database.get_experiment_reviews(experiment_id)
        
        return {
            'experiment': asdict(experiment),
            'results': results,
            'reviews': reviews
        }
    
    def export_research_data(self, output_file: str = None) -> str:
        """Export all research data to JSON."""
        if not output_file:
            output_file = f"research_data/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'experiments': [asdict(exp) for exp in self.database.list_experiments(limit=1000)],
            'statistics': self.database.get_research_statistics(),
            'system_status': self.get_system_status()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported research data to {output_file}")
        return output_file
