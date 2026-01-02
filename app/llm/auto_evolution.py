"""
Auto-Evolution System for Kempian LLM
Automatically collects feedback and triggers retraining
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoEvolutionSystem:
    """System for automatic model evolution through feedback collection"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use instance folder in backend
            db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'instance', 'llm_feedback.db')
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize feedback database"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                user_id TEXT,
                task_type TEXT,
                input_text TEXT,
                output_text TEXT,
                user_feedback TEXT,
                outcome TEXT,
                quality_score REAL,
                model_version TEXT,
                context TEXT
            )
        """)
        
        # Training runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                model_version TEXT,
                examples_used INTEGER,
                training_loss REAL,
                validation_loss REAL,
                deployed BOOLEAN
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("LLM feedback database initialized")
    
    def log_interaction(
        self,
        task_type: str,
        input_text: str,
        output_text: str,
        user_id: Optional[str] = None,
        feedback: Optional[str] = None,
        outcome: Optional[str] = None,
        context: Optional[Dict] = None,
        model_version: str = "v1.0"
    ):
        """Log user interaction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        quality_score = self._calculate_quality_score(feedback, outcome)
        context_json = json.dumps(context) if context else None
        
        cursor.execute("""
            INSERT INTO interactions 
            (timestamp, user_id, task_type, input_text, output_text, 
             user_feedback, outcome, quality_score, model_version, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            user_id,
            task_type,
            input_text,
            output_text,
            feedback,
            outcome,
            quality_score,
            model_version,
            context_json
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Logged LLM interaction: {task_type}, quality_score: {quality_score}")
    
    def _calculate_quality_score(self, feedback: Optional[str], outcome: Optional[str]) -> float:
        """Calculate quality score for interaction"""
        score = 0.5  # Base score
        
        if feedback == "positive":
            score += 0.3
        elif feedback == "negative":
            score -= 0.3
        
        if outcome in ["hired", "job_posted", "successful"]:
            score += 0.2
        elif outcome in ["rejected", "failed", "unsuccessful"]:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def get_training_data(
        self,
        min_quality: float = 0.7,
        limit: int = 1000,
        task_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Get high-quality interactions for training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if task_types:
            placeholders = ",".join(["?"] * len(task_types))
            query = f"""
                SELECT task_type, input_text, output_text, quality_score, context
                FROM interactions
                WHERE quality_score >= ? AND task_type IN ({placeholders})
                ORDER BY quality_score DESC, timestamp DESC
                LIMIT ?
            """
            params = [min_quality] + task_types + [limit]
        else:
            query = """
                SELECT task_type, input_text, output_text, quality_score, context
                FROM interactions
                WHERE quality_score >= ?
                ORDER BY quality_score DESC, timestamp DESC
                LIMIT ?
            """
            params = [min_quality, limit]
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        training_data = []
        for row in results:
            context = json.loads(row[4]) if row[4] else {}
            training_data.append({
                "task_type": row[0],
                "input": row[1],
                "output": row[2],
                "quality": row[3],
                "context": context
            })
        
        logger.info(f"Retrieved {len(training_data)} training examples")
        return training_data
    
    def should_retrain(
        self, 
        days_since_last: int = 7, 
        min_new_interactions: int = 1000
    ) -> bool:
        """Check if model should be retrained"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count interactions since last training
        cutoff_date = datetime.now() - timedelta(days=days_since_last)
        cursor.execute("""
            SELECT COUNT(*) FROM interactions
            WHERE timestamp > ?
        """, (cutoff_date,))
        
        count = cursor.fetchone()[0]
        
        # Check last training run
        cursor.execute("""
            SELECT timestamp FROM training_runs
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        last_training = cursor.fetchone()
        conn.close()
        
        # Retrain if:
        # 1. More than min_new_interactions in last period
        # 2. Or no training has been done yet
        # 3. Or last training was more than 7 days ago
        if last_training is None:
            return count >= 100  # First training needs less data
        
        last_training_date = datetime.fromisoformat(last_training[0])
        days_since_training = (datetime.now() - last_training_date).days
        
        should_retrain = (
            count >= min_new_interactions or
            days_since_training >= 7
        )
        
        logger.info(f"Should retrain: {should_retrain} (count: {count}, days_since: {days_since_training})")
        return should_retrain
    
    def log_training_run(
        self,
        model_version: str,
        examples_used: int,
        training_loss: float,
        validation_loss: Optional[float] = None,
        deployed: bool = False
    ):
        """Log training run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO training_runs
            (timestamp, model_version, examples_used, training_loss, validation_loss, deployed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            model_version,
            examples_used,
            training_loss,
            validation_loss,
            deployed
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Logged training run: {model_version}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about collected data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total interactions
        cursor.execute("SELECT COUNT(*) FROM interactions")
        total_interactions = cursor.fetchone()[0]
        
        # High quality interactions
        cursor.execute("SELECT COUNT(*) FROM interactions WHERE quality_score >= 0.7")
        high_quality = cursor.fetchone()[0]
        
        # By task type
        cursor.execute("""
            SELECT task_type, COUNT(*) as count, AVG(quality_score) as avg_quality
            FROM interactions
            GROUP BY task_type
        """)
        by_task_type = {
            row[0]: {"count": row[1], "avg_quality": row[2]} 
            for row in cursor.fetchall()
        }
        
        # Recent interactions (last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        cursor.execute("SELECT COUNT(*) FROM interactions WHERE timestamp > ?", (cutoff_date,))
        recent_interactions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_interactions": total_interactions,
            "high_quality_interactions": high_quality,
            "recent_interactions_7d": recent_interactions,
            "by_task_type": by_task_type
        }

