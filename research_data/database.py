"""Database layer for storing research data and results."""

import json
import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict
import threading

from research_agents.base_agent import ResearchExperiment


class ResearchDatabase:
    """SQLite database for storing research data and results."""
    
    def __init__(self, db_path: str = "research_data/research.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    hypothesis TEXT,
                    parameters TEXT,
                    priority INTEGER,
                    estimated_duration INTEGER,
                    status TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    results TEXT,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            # Results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    metric_type TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Reviews table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    reviewer TEXT,
                    overall_score REAL,
                    weighted_score REAL,
                    recommendation TEXT,
                    priority_assessment INTEGER,
                    success_probability REAL,
                    review_data TEXT,
                    created_at TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Agent interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    agent_name TEXT,
                    interaction_type TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Research insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    insight_type TEXT,
                    insight_text TEXT,
                    confidence_score REAL,
                    created_at TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            conn.commit()
            conn.close()
    
    def save_experiment(self, experiment: ResearchExperiment) -> bool:
        """Save an experiment to the database."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Convert experiment to dict
                exp_dict = asdict(experiment)
                
                # Convert datetime objects to strings
                for key in ['created_at', 'started_at', 'completed_at']:
                    if exp_dict.get(key) and isinstance(exp_dict[key], datetime):
                        exp_dict[key] = exp_dict[key].isoformat()
                
                # Insert or update experiment
                cursor.execute("""
                    INSERT OR REPLACE INTO experiments 
                    (id, title, description, hypothesis, parameters, priority, 
                     estimated_duration, status, created_at, started_at, completed_at, 
                     results, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    exp_dict['id'],
                    exp_dict['title'],
                    exp_dict['description'],
                    exp_dict['hypothesis'],
                    json.dumps(exp_dict['parameters']),
                    exp_dict['priority'],
                    exp_dict['estimated_duration'],
                    exp_dict['status'],
                    exp_dict['created_at'],
                    exp_dict.get('started_at'),
                    exp_dict.get('completed_at'),
                    json.dumps(exp_dict.get('results', {})),
                    exp_dict.get('error_message'),
                    json.dumps(exp_dict.get('metadata', {}))
                ))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                print(f"Error saving experiment: {e}")
                return False
    
    def get_experiment(self, experiment_id: str) -> Optional[ResearchExperiment]:
        """Get an experiment by ID."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
                row = cursor.fetchone()
                
                conn.close()
                
                if row:
                    return self._row_to_experiment(row)
                return None
                
            except Exception as e:
                print(f"Error getting experiment: {e}")
                return None
    
    def _row_to_experiment(self, row: Tuple) -> ResearchExperiment:
        """Convert database row to ResearchExperiment object."""
        return ResearchExperiment(
            id=row[0],
            title=row[1],
            description=row[2],
            hypothesis=row[3],
            parameters=json.loads(row[4]) if row[4] else {},
            priority=row[5],
            estimated_duration=row[6],
            status=row[7],
            created_at=datetime.fromisoformat(row[8]) if row[8] else None,
            started_at=datetime.fromisoformat(row[9]) if row[9] else None,
            completed_at=datetime.fromisoformat(row[10]) if row[10] else None,
            results=json.loads(row[11]) if row[11] else None,
            error_message=row[12],
            metadata=json.loads(row[13]) if row[13] else {}
        )
    
    def list_experiments(self, 
                        status: Optional[str] = None,
                        limit: int = 100) -> List[ResearchExperiment]:
        """List experiments with optional filtering."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                query = "SELECT * FROM experiments"
                params = []
                
                if status:
                    query += " WHERE status = ?"
                    params.append(status)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                conn.close()
                
                return [self._row_to_experiment(row) for row in rows]
                
            except Exception as e:
                print(f"Error listing experiments: {e}")
                return []
    
    def save_review(self, experiment_id: str, review_data: Dict[str, Any]) -> bool:
        """Save experiment review."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO experiment_reviews 
                    (experiment_id, reviewer, overall_score, weighted_score, 
                     recommendation, priority_assessment, success_probability, 
                     review_data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    review_data.get('reviewer', 'unknown'),
                    review_data.get('overall_score', 0),
                    review_data.get('weighted_score', 0),
                    review_data.get('recommendation', 'unknown'),
                    review_data.get('priority_assessment', 0),
                    review_data.get('success_probability', 0),
                    json.dumps(review_data),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                print(f"Error saving review: {e}")
                return False
    
    def save_result(self, experiment_id: str, metric_name: str, 
                   metric_value: float, metric_type: str = "performance") -> bool:
        """Save experiment result metric."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO experiment_results 
                    (experiment_id, metric_name, metric_value, metric_type, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (experiment_id, metric_name, metric_value, metric_type, datetime.now().isoformat()))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                print(f"Error saving result: {e}")
                return False
    
    def save_agent_interaction(self, experiment_id: str, agent_name: str,
                              interaction_type: str, input_data: Any, output_data: Any) -> bool:
        """Save agent interaction."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO agent_interactions 
                    (experiment_id, agent_name, interaction_type, input_data, output_data, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, agent_name, interaction_type,
                    json.dumps(input_data) if input_data else None,
                    json.dumps(output_data) if output_data else None,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                print(f"Error saving agent interaction: {e}")
                return False
    
    def save_insight(self, experiment_id: str, insight_type: str, 
                    insight_text: str, confidence_score: float = 0.5) -> bool:
        """Save research insight."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO research_insights 
                    (experiment_id, insight_type, insight_text, confidence_score, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (experiment_id, insight_type, insight_text, confidence_score, datetime.now().isoformat()))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                print(f"Error saving insight: {e}")
                return False
    
    def get_experiment_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all results for an experiment."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT metric_name, metric_value, metric_type, timestamp 
                    FROM experiment_results 
                    WHERE experiment_id = ?
                    ORDER BY timestamp
                """, (experiment_id,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'metric_name': row[0],
                        'metric_value': row[1],
                        'metric_type': row[2],
                        'timestamp': row[3]
                    })
                
                conn.close()
                return results
                
            except Exception as e:
                print(f"Error getting results: {e}")
                return []
    
    def get_experiment_reviews(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all reviews for an experiment."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT reviewer, overall_score, weighted_score, recommendation, 
                           priority_assessment, success_probability, review_data, created_at
                    FROM experiment_reviews 
                    WHERE experiment_id = ?
                    ORDER BY created_at DESC
                """, (experiment_id,))
                
                reviews = []
                for row in cursor.fetchall():
                    reviews.append({
                        'reviewer': row[0],
                        'overall_score': row[1],
                        'weighted_score': row[2],
                        'recommendation': row[3],
                        'priority_assessment': row[4],
                        'success_probability': row[5],
                        'review_data': json.loads(row[6]) if row[6] else {},
                        'created_at': row[7]
                    })
                
                conn.close()
                return reviews
                
            except Exception as e:
                print(f"Error getting reviews: {e}")
                return []
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get research statistics."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Count experiments by status
                cursor.execute("SELECT status, COUNT(*) FROM experiments GROUP BY status")
                status_counts = dict(cursor.fetchall())
                
                # Average scores
                cursor.execute("SELECT AVG(weighted_score) FROM experiment_reviews")
                avg_score = cursor.fetchone()[0] or 0
                
                # Success rate
                cursor.execute("""
                    SELECT COUNT(*) FROM experiments 
                    WHERE status = 'completed' AND error_message IS NULL
                """)
                successful = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'completed'")
                total_completed = cursor.fetchone()[0]
                
                success_rate = successful / max(total_completed, 1)
                
                # Most productive agent
                cursor.execute("""
                    SELECT agent_name, COUNT(*) as interaction_count 
                    FROM agent_interactions 
                    GROUP BY agent_name 
                    ORDER BY interaction_count DESC 
                    LIMIT 1
                """)
                top_agent = cursor.fetchone()
                
                conn.close()
                
                return {
                    'status_counts': status_counts,
                    'average_review_score': round(avg_score, 2),
                    'success_rate': round(success_rate, 2),
                    'total_experiments': sum(status_counts.values()),
                    'most_active_agent': top_agent[0] if top_agent else None,
                    'agent_interactions': top_agent[1] if top_agent else 0
                }
                
            except Exception as e:
                print(f"Error getting statistics: {e}")
                return {}
