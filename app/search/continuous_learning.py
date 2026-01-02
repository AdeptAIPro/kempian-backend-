"""
Continuous Learning Pipeline
Automated retraining and model versioning with drift detection.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from app.search.xgboost_ranking_model import get_ranking_model, train_ranking_model
from app.search.feedback_collector import get_feedback_collector
from app.search.evaluation_metrics import get_metrics_evaluator
from app.search.ranking_feature_extractor import get_feature_extractor

logger = logging.getLogger(__name__)


class ContinuousLearningPipeline:
    """Automated continuous learning pipeline"""
    
    def __init__(self, model_storage_path: str = 'models/'):
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.feedback_collector = get_feedback_collector()
        self.feature_extractor = get_feature_extractor()
        self.metrics_evaluator = get_metrics_evaluator()
        
        self.current_model_version = None
        self.model_versions: List[Dict] = []
    
    def incremental_update(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Incremental model update with recent feedback
        
        Args:
            days_back: Number of days to look back for feedback
        
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting incremental update with {days_back} days of feedback")
        
        # Get recent feedback
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        feedback_batch = self.feedback_collector.get_feedback_batch(
            limit=5000,
            start_date=start_date,
            end_date=end_date
        )
        
        if len(feedback_batch) < 100:
            logger.warning(f"Insufficient feedback for training: {len(feedback_batch)} records")
            return {'status': 'insufficient_data', 'feedback_count': len(feedback_batch)}
        
        # Convert feedback to training data
        training_data = self._prepare_training_data(feedback_batch)
        
        # Load current model
        current_model = self._load_current_model()
        
        # Fine-tune model (incremental update)
        # For XGBoost, we retrain with all data including new feedback
        updated_model = self._retrain_model(training_data, current_model)
        
        # Evaluate new model
        metrics = self._evaluate_model(updated_model, feedback_batch)
        
        # Check for model drift
        drift_detected = self._detect_drift(current_model, updated_model, metrics)
        
        # Version and save model
        version_info = self._version_model(updated_model, metrics, drift_detected)
        
        return {
            'status': 'success',
            'version': version_info['version_id'],
            'metrics': metrics,
            'drift_detected': drift_detected,
            'feedback_count': len(feedback_batch)
        }
    
    def full_retrain(self, days_back: int = 90) -> Dict[str, Any]:
        """
        Full model retrain with all historical data
        
        Args:
            days_back: Number of days to look back
        
        Returns:
            Training results
        """
        logger.info(f"Starting full retrain with {days_back} days of data")
        
        # Get all feedback
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        feedback_batch = self.feedback_collector.get_feedback_batch(
            limit=50000,
            start_date=start_date,
            end_date=end_date
        )
        
        if len(feedback_batch) < 1000:
            logger.warning(f"Insufficient feedback for full retrain: {len(feedback_batch)} records")
            return {'status': 'insufficient_data'}
        
        # Prepare training data
        training_data = self._prepare_training_data(feedback_batch)
        
        # Split for validation
        split_idx = int(len(training_data) * 0.8)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Train new model
        model_path = self.model_storage_path / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        training_results = train_ranking_model(
            train_data,
            val_data,
            str(model_path)
        )
        
        # Evaluate
        metrics = self._evaluate_model_from_path(str(model_path), val_data)
        
        # Version model
        version_info = self._version_model_from_path(str(model_path), metrics)
        
        return {
            'status': 'success',
            'version': version_info['version_id'],
            'metrics': metrics,
            'training_results': training_results
        }
    
    def _prepare_training_data(self, feedback_batch: List) -> List[Dict[str, Any]]:
        """Prepare training data from feedback"""
        training_data = []
        
        for feedback in feedback_batch:
            # Load job and candidate data
            job = self._load_job(feedback.job_id)
            candidate = self._load_candidate(feedback.candidate_id)
            
            if not job or not candidate:
                continue
            
            # Extract features
            features = self.feature_extractor.extract_features(
                job_description=job.get('description', ''),
                candidate=candidate,
                job_location=job.get('location'),
                job_required_skills=job.get('required_skill_ids', [])
            )
            
            # Add label
            features['label'] = feedback.label
            features['job_id'] = feedback.job_id
            features['candidate_id'] = feedback.candidate_id
            
            training_data.append(features)
        
        return training_data
    
    def _load_current_model(self):
        """Load current production model"""
        # Find latest model version
        model_files = sorted(self.model_storage_path.glob('model_*.json'), reverse=True)
        
        if model_files:
            model = get_ranking_model(str(model_files[0]))
            return model
        
        return None
    
    def _retrain_model(self, training_data: List[Dict], current_model) -> Any:
        """Retrain model with new data"""
        # For incremental update, combine with existing training data
        # In production, you'd load previous training data and combine
        
        # Split for validation
        split_idx = int(len(training_data) * 0.8)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Train new model
        model_path = self.model_storage_path / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        train_ranking_model(train_data, val_data, str(model_path))
        
        return get_ranking_model(str(model_path))
    
    def _evaluate_model(self, model, feedback_batch: List) -> Dict[str, float]:
        """Evaluate model performance"""
        # Extract true labels and predictions
        y_true = []
        y_pred = []
        
        for feedback in feedback_batch[:100]:  # Sample for evaluation
            job = self._load_job(feedback.job_id)
            candidate = self._load_candidate(feedback.candidate_id)
            
            if not job or not candidate:
                continue
            
            # Extract features
            features = self.feature_extractor.extract_features(
                job_description=job.get('description', ''),
                candidate=candidate
            )
            
            # Predict
            pred_score = model.predict_score(features)
            
            y_true.append(feedback.label)
            y_pred.append(pred_score)
        
        # Calculate metrics
        metrics = self.metrics_evaluator.evaluate(y_true, y_pred)
        
        return {
            'precision_at_5': metrics.precision_at_5,
            'precision_at_10': metrics.precision_at_10,
            'ndcg_at_10': metrics.ndcg_at_10,
            'mrr': metrics.mrr
        }
    
    def _evaluate_model_from_path(self, model_path: str, val_data: List[Dict]) -> Dict[str, float]:
        """Evaluate model from file path"""
        model = get_ranking_model(model_path)
        
        y_true = [d['label'] for d in val_data]
        y_pred = [model.predict_score({k: v for k, v in d.items() if k != 'label'}) for d in val_data]
        
        metrics = self.metrics_evaluator.evaluate(y_true, y_pred)
        
        return {
            'precision_at_5': metrics.precision_at_5,
            'precision_at_10': metrics.precision_at_10,
            'ndcg_at_10': metrics.ndcg_at_10,
            'mrr': metrics.mrr
        }
    
    def _detect_drift(
        self,
        current_model: Any,
        new_model: Any,
        new_metrics: Dict[str, float]
    ) -> bool:
        """Detect model drift"""
        if not current_model:
            return False
        
        # Compare metrics
        # In production, you'd compare against baseline metrics
        baseline_precision = 0.85  # Target precision@5
        
        if new_metrics['precision_at_5'] < baseline_precision * 0.95:  # 5% drop
            logger.warning("Model drift detected: precision dropped significantly")
            return True
        
        return False
    
    def _version_model(
        self,
        model: Any,
        metrics: Dict[str, float],
        drift_detected: bool
    ) -> Dict[str, Any]:
        """Version and save model"""
        version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        version_info = {
            'version_id': version_id,
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics,
            'drift_detected': drift_detected,
            'rollback_available': True
        }
        
        # Save version info
        version_file = self.model_storage_path / f"{version_id}_info.json"
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        self.model_versions.append(version_info)
        self.current_model_version = version_id
        
        return version_info
    
    def _version_model_from_path(
        self,
        model_path: str,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Version model from file path"""
        version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        version_info = {
            'version_id': version_id,
            'model_path': model_path,
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics,
            'rollback_available': True
        }
        
        # Save version info
        version_file = self.model_storage_path / f"{version_id}_info.json"
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        return version_info
    
    def rollback_model(self, version_id: str) -> bool:
        """Rollback to previous model version"""
        # Find version
        version_file = self.model_storage_path / f"{version_id}_info.json"
        
        if not version_file.exists():
            logger.error(f"Version {version_id} not found")
            return False
        
        with open(version_file, 'r') as f:
            version_info = json.load(f)
        
        model_path = version_info.get('model_path')
        if model_path and Path(model_path).exists():
            # Load and set as current
            self.current_model_version = version_id
            logger.info(f"Rolled back to version {version_id}")
            return True
        
        return False
    
    def _load_job(self, job_id: str) -> Optional[Dict]:
        """Load job data (implement based on your storage)"""
        # This would load from your database
        return None
    
    def _load_candidate(self, candidate_id: str) -> Optional[Dict]:
        """Load candidate data (implement based on your storage)"""
        # This would load from your database
        return None


# Global instance
_continuous_learning = None

def get_continuous_learning(model_storage_path: str = 'models/') -> ContinuousLearningPipeline:
    """Get or create global continuous learning pipeline"""
    global _continuous_learning
    if _continuous_learning is None:
        _continuous_learning = ContinuousLearningPipeline(model_storage_path)
    return _continuous_learning

