"""
XGBoost Ranking Model
Complete training and serving pipeline for learning-to-rank.
"""

import logging
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

from app.search.ranking_feature_extractor import get_feature_extractor

logger = logging.getLogger(__name__)

# Feature order (must match feature extractor output)
FEATURE_ORDER = [
    'dense_similarity',
    'cross_encoder_score',
    'tfidf_score',
    'exact_skill_count',
    'weighted_skill_match',
    'skill_match_ratio',
    'preferred_skill_match',
    'candidate_experience_years',
    'job_experience_required',
    'experience_match',
    'experience_gap',
    'seniority_match_distance',
    'seniority_match',
    'location_distance_km',
    'location_distance_score',
    'timezone_compatibility',
    'remote_eligible_alignment',
    'certification_match',
    'certification_match_count',
    'education_match',
    'domain_match',
    'days_since_resume_update',
    'resume_recency_score',
    'data_completeness',
    'has_resume_text',
    'candidate_response_rate',
    'recruiter_interaction_score',
    'source_reliability',
    'skill_diversity',
    'achievement_impact_score'
]


class XGBoostRankingModel:
    """XGBoost ranking model for candidate-job matching"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.feature_extractor = get_feature_extractor()
        self.model_metadata = {}
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train XGBoost ranking model
        
        Args:
            training_data: List of training examples with features and labels
            validation_data: Optional validation data
            output_path: Path to save trained model
        
        Returns:
            Training metrics and model info
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        logger.info(f"Training XGBoost model on {len(training_data)} examples")
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Separate features and labels
        feature_cols = [col for col in FEATURE_ORDER if col in df.columns]
        X = df[feature_cols].values
        y = df['label'].values
        
        # Group by job_id for pairwise ranking
        if 'job_id' in df.columns:
            groups = df.groupby('job_id').size().values
        else:
            # Create dummy groups if no job_id
            groups = np.ones(len(df), dtype=int)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(groups)
        
        # Validation data
        dval = None
        if validation_data:
            val_df = pd.DataFrame(validation_data)
            X_val = val_df[feature_cols].values
            y_val = val_df['label'].values
            
            if 'job_id' in val_df.columns:
                val_groups = val_df.groupby('job_id').size().values
            else:
                val_groups = np.ones(len(val_df), dtype=int)
            
            dval = xgb.DMatrix(X_val, label=y_val)
            dval.set_group(val_groups)
        
        # Training parameters
        params = {
            'objective': 'rank:pairwise',
            'eta': 0.1,
            'max_depth': 8,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'ndcg@10',
            'tree_method': 'hist',
            'verbosity': 1
        }
        
        # Train model
        evals = [(dtrain, 'train')]
        if dval:
            evals.append((dval, 'validation'))
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=10
        )
        
        # Get feature importance
        feature_importance = self.model.get_score(importance_type='gain')
        
        # Calculate metrics
        train_pred = self.model.predict(dtrain)
        train_metrics = self._calculate_metrics(y, train_pred, groups)
        
        val_metrics = None
        if dval:
            val_pred = self.model.predict(dval)
            val_metrics = self._calculate_metrics(y_val, val_pred, val_groups)
        
        # Save model
        if output_path:
            self.save_model(output_path, {
                'trained_at': datetime.now().isoformat(),
                'training_size': len(training_data),
                'feature_importance': feature_importance,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': feature_importance,
            'model_path': output_path
        }
    
    def predict_score(self, features: Dict[str, float]) -> float:
        """Predict ranking score from features"""
        if not self.model:
            raise ValueError("Model not loaded or trained")
        
        # Convert features to array in correct order
        feature_array = np.array([[
            features.get(feature, 0.0) for feature in FEATURE_ORDER
        ]])
        
        dmatrix = xgb.DMatrix(feature_array)
        score = self.model.predict(dmatrix)[0]
        
        return float(score)
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """Predict scores for batch of candidates"""
        if not self.model:
            raise ValueError("Model not loaded or trained")
        
        # Convert to array
        feature_array = np.array([
            [features.get(feature, 0.0) for feature in FEATURE_ORDER]
            for features in features_list
        ])
        
        dmatrix = xgb.DMatrix(feature_array)
        scores = self.model.predict(dmatrix)
        
        return [float(score) for score in scores]
    
    def save_model(self, model_path: str, metadata: Optional[Dict] = None):
        """Save model to disk"""
        if not self.model:
            raise ValueError("No model to save")
        
        # Save XGBoost model
        self.model.save_model(model_path)
        
        # Save metadata
        if metadata:
            metadata_path = model_path.replace('.json', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self.model_path = model_path
        self.model_metadata = metadata or {}
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load model from disk"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.json', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
        
        self.model_path = model_path
        logger.info(f"Model loaded from {model_path}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
        """Calculate ranking metrics"""
        from sklearn.metrics import ndcg_score
        
        try:
            # Calculate nDCG@10
            # Reshape for ndcg_score (requires 2D arrays)
            y_true_2d = y_true.reshape(-1, 1)
            y_pred_2d = y_pred.reshape(-1, 1)
            
            ndcg_10 = ndcg_score(y_true_2d, y_pred_2d, k=10)
            
            # Calculate precision@5 and precision@10
            # Sort by predicted score
            sorted_indices = np.argsort(y_pred)[::-1]
            sorted_labels = y_true[sorted_indices]
            
            precision_5 = np.mean(sorted_labels[:5]) if len(sorted_labels) >= 5 else np.mean(sorted_labels)
            precision_10 = np.mean(sorted_labels[:10]) if len(sorted_labels) >= 10 else np.mean(sorted_labels)
            
            # Calculate MRR
            mrr = self._calculate_mrr(y_true, y_pred)
            
            return {
                'ndcg@10': float(ndcg_10),
                'precision@5': float(precision_5),
                'precision@10': float(precision_10),
                'mrr': float(mrr)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'ndcg@10': 0.0,
                'precision@5': 0.0,
                'precision@10': 0.0,
                'mrr': 0.0
            }
    
    def _calculate_mrr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Reciprocal Rank"""
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_labels = y_true[sorted_indices]
        
        # Find first relevant item (label == 1)
        for i, label in enumerate(sorted_labels):
            if label == 1:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.model:
            return {}
        
        importance = self.model.get_score(importance_type='gain')
        return importance


# Global instance
_ranking_model = None

def get_ranking_model(model_path: Optional[str] = None) -> XGBoostRankingModel:
    """Get or create global ranking model instance"""
    global _ranking_model
    if _ranking_model is None:
        _ranking_model = XGBoostRankingModel(model_path)
    return _ranking_model

def train_ranking_model(
    training_data: List[Dict[str, Any]],
    validation_data: Optional[List[Dict[str, Any]]] = None,
    output_path: str = 'ranking_model.json'
) -> Dict[str, Any]:
    """Convenience function to train ranking model"""
    model = XGBoostRankingModel()
    return model.train(training_data, validation_data, output_path)

