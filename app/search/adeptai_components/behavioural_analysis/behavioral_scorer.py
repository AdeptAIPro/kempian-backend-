from __future__ import annotations
from typing import Dict, Optional, Any
import numpy as np
from datetime import datetime
import logging
import json

try:
    import lightgbm as lgb
except Exception:
    lgb = None  # Allow runtime without LightGBM for environments that can’t compile it

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleConfig:
    """Configuration for ensemble integration."""
    def __init__(self,
                 enable_cross_validation: bool = False,
                 ensemble_size: int = 5,
                 method: str = "weighted_average"):
        self.enable_cross_validation = enable_cross_validation
        self.ensemble_size = ensemble_size
        self.method = method


class PredictionResult:
    """Container for prediction results."""
    def __init__(self,
                 predictions: Dict[str, float],
                 confidence: float,
                 feature_importance: Dict[str, float],
                 ensemble_contributions: Dict[str, float],
                 metadata: Dict[str, Any]):
        self.predictions = predictions
        self.confidence = confidence
        self.feature_importance = feature_importance
        self.ensemble_contributions = ensemble_contributions
        self.metadata = metadata


class BehavioralScorer:
    """
    Enhanced behavioral scorer with ensemble integration, confidence scoring,
    and feature importance tracking for multi-modal system integration.
    """

    def __init__(self, 
                 lgbm_model_path: Optional[str] = None,
                 ensemble_config: Optional[EnsembleConfig] = None):
        
        self.model = None
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.feature_history = []
        self.prediction_history = []
        self.ensemble_models = {}
        
        # Initialize LightGBM model if available
        if lgb and lgbm_model_path:
            try:
                self.model = lgb.Booster(model_file=lgbm_model_path)
                logger.info(f"LightGBM model loaded from {lgbm_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load LightGBM model: {e}")
                self.model = None
        
        # Initialize ensemble models if enabled
        if self.ensemble_config.enable_cross_validation:
            self._initialize_ensemble_models()
        
        # Feature importance cache
        self.feature_importance_cache = {}
        
        logger.info(f"BehavioralScorer initialized with ensemble method: {self.ensemble_config.method.value if hasattr(self.ensemble_config.method, 'value') else str(self.ensemble_config.method)}")

    def _initialize_ensemble_models(self):
        """Initialize ensemble models for cross-validation"""
        try:
            # Create multiple models with different random seeds for ensemble diversity
            for i in range(self.ensemble_config.ensemble_size):
                if self.model is not None:
                    # Clone the base model with different parameters
                    ensemble_model = self._create_ensemble_model(i)
                    self.ensemble_models[f"ensemble_{i}"] = ensemble_model
                    
            logger.info(f"Initialized {len(self.ensemble_models)} ensemble models")
        except Exception as e:
            logger.warning(f"Failed to initialize ensemble models: {e}")

    def _create_ensemble_model(self, seed: int):
        """Create an ensemble model variant"""
        if self.model is not None and lgb:
            # Create a new model with slightly different parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': seed
            }
            
            # For now, return the base model (in practice, you'd train new models)
            return self.model
        return None

    def _to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features to vector with stable feature order"""
        # Ensure stable feature order
        ORDER = [
            "semantic_match",
            "progression",
            "linguistic_complexity",
            "leadership_alignment",
            "collaboration_alignment",
            "innovation_alignment",
            "adaptability_alignment",
            "confidence",
            "positivity",
            "empathy",
            "stress_inverse"
        ]
        return np.array([[float(features.get(k, 0.0)) for k in ORDER]], dtype=np.float32)

    def predict(self, features: Dict[str, float], 
                ensemble_inputs: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """
        Enhanced prediction with ensemble integration and confidence scoring
        
        Args:
            features: Input features for prediction
            ensemble_inputs: Additional inputs from other models in the ensemble
        
        Returns:
            PredictionResult with predictions, confidence, and feature importance
        """
        
        # Store feature history for analysis
        self.feature_history.append({
            'features': features.copy(),
            'timestamp': datetime.now(),
            'ensemble_inputs': ensemble_inputs
        })
        
        # Get base predictions
        base_predictions = self._get_base_predictions(features)
        
        # Apply ensemble integration if enabled
        if self.ensemble_config.enable_cross_validation and self.ensemble_models:
            ensemble_predictions = self._apply_ensemble_integration(features, ensemble_inputs)
            final_predictions = self._combine_predictions(base_predictions, ensemble_predictions)
        else:
            final_predictions = base_predictions
            ensemble_predictions = {}
        
        # Calculate confidence score
        confidence = self._calculate_confidence(features, final_predictions, ensemble_predictions)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(features, final_predictions)
        
        # Calculate ensemble contributions
        ensemble_contributions = self._calculate_ensemble_contributions(
            base_predictions, ensemble_predictions
        )
        
        # Create result
        result = PredictionResult(
            predictions=final_predictions,
            confidence=confidence,
            feature_importance=feature_importance,
            ensemble_contributions=ensemble_contributions,
            metadata={
                'ensemble_method': self.ensemble_config.method.value if hasattr(self.ensemble_config.method, 'value') else str(self.ensemble_config.method),
                'ensemble_size': len(self.ensemble_models),
                'feature_count': len(features),
                'model_type': 'lightgbm' if self.model else 'fallback'
            }
        )
        
        # Store prediction history
        self.prediction_history.append(result)
        
        return result

    def _get_base_predictions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get base predictions from the primary model"""
        X = self._to_vector(features)

        if self.model is not None:
            pred = self.model.predict(X)[0]
            if isinstance(pred, (list, np.ndarray)) and len(pred) >= 3:
                leadership, innovation, stability = float(pred[0]), float(pred[1]), float(pred[2])
            else:
                leadership = innovation = stability = float(pred) if isinstance(pred, (int, float)) else 0.5
        else:
            # Fallback: weighted semantic + behavioral signals (beyond keywords)
            leadership = (
                0.35 * features.get("leadership_alignment", 0.0) +
                0.20 * features.get("collaboration_alignment", 0.0) +
                0.20 * features.get("confidence", 0.0) +
                0.15 * features.get("semantic_match", 0.0) +
                0.10 * features.get("progression", 0.0)
            )
            innovation = (
                0.40 * features.get("innovation_alignment", 0.0) +
                0.25 * features.get("adaptability_alignment", 0.0) +
                0.20 * features.get("semantic_match", 0.0) +
                0.15 * features.get("progression", 0.0)
            )
            stability = (
                0.40 * features.get("semantic_match", 0.0) +
                0.25 * (1.0 - min(1.0, features.get("stress_inverse", 0.0))) * 0.0 +  # keep neutral in fallback
                0.20 * (1.0 - features.get("adaptability_alignment", 0.0)) +
                0.15 * (1.0 - features.get("progression", 0.0))
            )

        # Clamp to [0,1] for readability
        def clamp(x): return float(max(0.0, min(1.0, x)))
        return {
            "leadership": clamp(leadership),
            "innovation": clamp(innovation),
            "stability": clamp(stability),
        }

    def _apply_ensemble_integration(self, features: Dict[str, float], 
                                   ensemble_inputs: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Apply ensemble integration methods"""
        
        if self.ensemble_config.method == "weighted_average":
            return self._weighted_average_ensemble(features, ensemble_inputs)
        elif self.ensemble_config.method == "stacking":
            return self._stacking_ensemble(features, ensemble_inputs)
        elif self.ensemble_config.method == "voting":
            return self._voting_ensemble(features, ensemble_inputs)
        elif self.ensemble_config.method == "bagging":
            return self._bagging_ensemble(features, ensemble_inputs)
        else:
            return self._weighted_average_ensemble(features, ensemble_inputs)

    def _weighted_average_ensemble(self, features: Dict[str, float], 
                                  ensemble_inputs: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Weighted average ensemble method"""
        
        # Get predictions from all ensemble models
        ensemble_predictions = {}
        for name, model in self.ensemble_models.items():
            if model is not None:
                try:
                    X = self._to_vector(features)
                    pred = model.predict(X)[0]
                    if isinstance(pred, (list, np.ndarray)) and len(pred) >= 3:
                        ensemble_predictions[name] = {
                            "leadership": float(pred[0]),
                            "innovation": float(pred[1]),
                            "stability": float(pred[2])
                        }
                    else:
                        ensemble_predictions[name] = {
                            "leadership": float(pred),
                            "innovation": float(pred),
                            "stability": float(pred)
                        }
                except Exception as e:
                    logger.warning(f"Failed to get prediction from {name}: {e}")
                    continue
        
        # Apply weights if specified
        weights = getattr(self.ensemble_config, 'weights', {})
        if not weights:
            # Equal weights if none specified
            weights = {name: 1.0 / len(ensemble_predictions) for name in ensemble_predictions.keys()}
        
        # Calculate weighted average
        weighted_predictions = {"leadership": 0.0, "innovation": 0.0, "stability": 0.0}
        total_weight = 0.0
        
        for name, preds in ensemble_predictions.items():
            weight = weights.get(name, 1.0)
            total_weight += weight
            
            for dimension in weighted_predictions:
                weighted_predictions[dimension] += preds[dimension] * weight
        
        # Normalize by total weight
        if total_weight > 0:
            for dimension in weighted_predictions:
                weighted_predictions[dimension] /= total_weight
        
        return weighted_predictions

    def _stacking_ensemble(self, features: Dict[str, float], 
                          ensemble_inputs: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Stacking ensemble method using meta-learner"""
        # For now, implement as weighted average
        # In practice, you'd train a meta-learner on the ensemble predictions
        return self._weighted_average_ensemble(features, ensemble_inputs)

    def _voting_ensemble(self, features: Dict[str, float], 
                        ensemble_inputs: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Voting ensemble method"""
        # Get predictions from all models
        all_predictions = []
        for name, model in self.ensemble_models.items():
            if model is not None:
                try:
                    X = self._to_vector(features)
                    pred = model.predict(X)[0]
                    if isinstance(pred, (list, np.ndarray)) and len(pred) >= 3:
                        all_predictions.append({
                            "leadership": float(pred[0]),
                            "innovation": float(pred[1]),
                            "stability": float(pred[2])
                        })
                    else:
                        all_predictions.append({
                            "leadership": float(pred),
                            "innovation": float(pred),
                            "stability": float(pred)
                        })
                except Exception as e:
                    logger.warning(f"Failed to get prediction from {name}: {e}")
                    continue
        
        if not all_predictions:
            return {"leadership": 0.5, "innovation": 0.5, "stability": 0.5}
        
        # Calculate median for each dimension (robust to outliers)
        dimensions = ["leadership", "innovation", "stability"]
        voting_predictions = {}
        
        for dimension in dimensions:
            values = [pred[dimension] for pred in all_predictions]
            voting_predictions[dimension] = float(np.median(values))
        
        return voting_predictions

    def _bagging_ensemble(self, features: Dict[str, float], 
                         ensemble_inputs: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Bagging ensemble method"""
        # For now, implement as simple average
        # In practice, you'd use bootstrap sampling
        return self._weighted_average_ensemble(features, ensemble_inputs)

    def _combine_predictions(self, base_predictions: Dict[str, float], 
                            ensemble_predictions: Dict[str, float]) -> Dict[str, float]:
        """Combine base and ensemble predictions"""
        
        if not ensemble_predictions:
            return base_predictions
        
        # Simple combination: 70% ensemble, 30% base
        combined = {}
        for dimension in base_predictions:
            ensemble_val = ensemble_predictions.get(dimension, base_predictions[dimension])
            combined[dimension] = 0.7 * ensemble_val + 0.3 * base_predictions[dimension]
        
        return combined

    def _calculate_confidence(self, features: Dict[str, float], 
                             predictions: Dict[str, float], 
                             ensemble_predictions: Dict[str, Any]) -> float:
        """Calculate confidence score for predictions"""
        
        confidence_factors = []
        
        # 1. Feature quality confidence
        feature_confidence = self._calculate_feature_confidence(features)
        confidence_factors.append(feature_confidence)
        
        # 2. Ensemble agreement confidence
        if ensemble_predictions:
            ensemble_confidence = self._calculate_ensemble_agreement(ensemble_predictions)
            confidence_factors.append(ensemble_confidence)
        
        # 3. Model confidence (if available)
        if self.model is not None:
            model_confidence = 0.8  # Placeholder - in practice, get from model
            confidence_factors.append(model_confidence)
        
        # 4. Prediction consistency confidence
        consistency_confidence = self._calculate_prediction_consistency(predictions)
        confidence_factors.append(consistency_confidence)
        
        # 5. Feature coverage confidence
        coverage_confidence = self._calculate_feature_coverage(features)
        confidence_factors.append(coverage_confidence)
        
        # Calculate overall confidence as weighted average
        if confidence_factors:
            # Give more weight to ensemble agreement and feature quality
            weights = [0.3, 0.25, 0.2, 0.15, 0.1][:len(confidence_factors)]
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            confidence = sum(cf * w for cf, w in zip(confidence_factors, weights))
            return min(1.0, max(0.0, confidence))
        
        return 0.5  # Default confidence

    def _calculate_feature_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence based on feature quality"""
        
        # Check for missing or extreme values
        missing_count = sum(1 for v in features.values() if v is None or np.isnan(v))
        extreme_count = sum(1 for v in features.values() if v < -1.0 or v > 2.0)
        
        # Check feature variance (some variance is good, too much might indicate noise)
        values = [v for v in features.values() if v is not None and not np.isnan(v)]
        if len(values) > 1:
            variance = np.var(values)
            variance_score = min(1.0, 1.0 - abs(variance - 0.25))  # Optimal variance around 0.25
        else:
            variance_score = 0.5
        
        # Calculate overall feature confidence
        missing_penalty = missing_count / len(features) if features else 0.0
        extreme_penalty = extreme_count / len(features) if features else 0.0
        
        feature_confidence = (1.0 - missing_penalty - extreme_penalty) * variance_score
        return max(0.0, min(1.0, feature_confidence))

    def _calculate_ensemble_agreement(self, ensemble_predictions: Dict[str, Any]) -> float:
        """Calculate confidence based on ensemble agreement"""
        
        if not ensemble_predictions:
            return 0.5
        
        # Extract predictions for each dimension
        dimensions = ["leadership", "innovation", "stability"]
        agreement_scores = []
        
        for dimension in dimensions:
            values = []
            for model_preds in ensemble_predictions.values():
                if isinstance(model_preds, dict) and dimension in model_preds:
                    values.append(model_preds[dimension])
            
            if len(values) > 1:
                # Calculate standard deviation (lower = higher agreement)
                std_dev = np.std(values)
                # Convert to agreement score (0 = no agreement, 1 = perfect agreement)
                agreement = max(0.0, 1.0 - std_dev)
                agreement_scores.append(agreement)
        
        if agreement_scores:
            return np.mean(agreement_scores)
        return 0.5

    def _calculate_prediction_consistency(self, predictions: Dict[str, float]) -> float:
        """Calculate confidence based on prediction consistency"""
        
        values = list(predictions.values())
        if len(values) < 2:
            return 0.5
        
        # Check if predictions are in reasonable ranges
        reasonable_count = sum(1 for v in values if 0.0 <= v <= 1.0)
        range_score = reasonable_count / len(values)
        
        # Check for logical consistency (e.g., leadership and innovation shouldn't be too low together)
        if 'leadership' in predictions and 'innovation' in predictions:
            leadership = predictions['leadership']
            innovation = predictions['innovation']
            
            # If both are very low, it might indicate an issue
            if leadership < 0.2 and innovation < 0.2:
                consistency_score = 0.3
            elif leadership > 0.8 and innovation > 0.8:
                consistency_score = 0.9
            else:
                consistency_score = 0.7
        else:
            consistency_score = 0.7
        
        return (range_score + consistency_score) / 2.0

    def _calculate_feature_coverage(self, features: Dict[str, float]) -> float:
        """Calculate confidence based on feature coverage"""
        
        if not features:
            return 0.0
        
        # Count non-zero and non-null features
        valid_features = sum(1 for v in features.values() 
                           if v is not None and v != 0.0 and not np.isnan(v))
        
        coverage = valid_features / len(features)
        
        # Bonus for having a good mix of different feature types
        feature_types = {
            'semantic': ['semantic_match', 'progression', 'linguistic_complexity'],
            'behavioral': ['leadership_alignment', 'collaboration_alignment', 'innovation_alignment'],
            'emotional': ['confidence', 'positivity', 'empathy', 'stress_inverse']
        }
        
        type_coverage = 0
        for feature_type, type_features in feature_types.items():
            if any(f in features for f in type_features):
                type_coverage += 1
        
        type_bonus = type_coverage / len(feature_types)
        
        return min(1.0, coverage + type_bonus * 0.2)

    def _calculate_feature_importance(self, features: Dict[str, float], 
                                     predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate feature importance for interpretability"""
        
        # Check cache first
        cache_key = hash(tuple(sorted(features.items())))
        if cache_key in self.feature_importance_cache:
            return self.feature_importance_cache[cache_key]
        
        if self.model is not None and lgb:
            # Use LightGBM's built-in feature importance if available
            try:
                importance = self.model.feature_importance(importance_type='gain')
                feature_names = [
                    "semantic_match", "progression", "linguistic_complexity",
                    "leadership_alignment", "collaboration_alignment", "innovation_alignment",
                    "adaptability_alignment", "confidence", "positivity", "empathy", "stress_inverse"
                ]
                
                # Normalize importance scores
                total_importance = sum(importance)
                if total_importance > 0:
                    normalized_importance = {name: imp / total_importance 
                                          for name, imp in zip(feature_names, importance)}
                else:
                    normalized_importance = {name: 1.0 / len(feature_names) 
                                          for name in feature_names}
                
                self.feature_importance_cache[cache_key] = normalized_importance
                return normalized_importance
                
            except Exception as e:
                logger.warning(f"Failed to get LightGBM feature importance: {e}")
        
        # Fallback: calculate importance based on correlation with predictions
        importance_scores = {}
        for feature_name, feature_value in features.items():
            if feature_value is not None and not np.isnan(feature_value):
                # Simple heuristic: importance based on feature value and prediction correlation
                # In practice, you'd use more sophisticated methods like permutation importance
                importance_scores[feature_name] = min(1.0, abs(feature_value))
            else:
                importance_scores[feature_name] = 0.0
        
        # Normalize importance scores
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            normalized_importance = {name: imp / total_importance 
                                  for name, imp in importance_scores.items()}
        else:
            normalized_importance = {name: 1.0 / len(importance_scores) 
                                  for name in importance_scores.keys()}
        
        self.feature_importance_cache[cache_key] = normalized_importance
        return normalized_importance

    def _calculate_ensemble_contributions(self, base_predictions: Dict[str, float], 
                                        ensemble_predictions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate how much each ensemble component contributed"""
        
        contributions = {
            'base_model': 0.3,  # Base model always contributes 30%
            'ensemble_models': 0.7 if ensemble_predictions else 0.0
        }
        
        # Add individual ensemble model contributions if available
        if ensemble_predictions:
            for model_name in ensemble_predictions.keys():
                contributions[f'ensemble_{model_name}'] = 0.7 / len(ensemble_predictions)
        
        return contributions
    
    def score(self, text: str, job_description: str = None) -> float:
        """
        Backward compatibility method for simple scoring
        
        Args:
            text: Resume or profile text
            job_description: Optional job description for matching
        
        Returns:
            Overall behavioral score (0-1)
        """
        # Extract basic features from text
        # This is a simplified version - in practice, you'd extract full features
        features = {
            "semantic_match": 0.5,  # Placeholder - would be calculated from text
            "progression": 0.5,
            "linguistic_complexity": 0.5,
            "leadership_alignment": 0.5,
            "collaboration_alignment": 0.5,
            "innovation_alignment": 0.5,
            "adaptability_alignment": 0.5,
            "confidence": 0.5,
            "positivity": 0.5,
            "empathy": 0.5,
            "stress_inverse": 0.5
        }
        
        # Get prediction
        result = self.predict(features)
        
        # Return average of all behavioral dimensions as overall score
        if result.predictions:
            return float(sum(result.predictions.values()) / len(result.predictions))
        return 0.5


# Convenience function for backward compatibility
def predict_behavioral_scores(features: Dict[str, float]) -> Dict[str, float]:
    """
    Convenience function for backward compatibility
    
    Args:
        features: Input features for prediction
    
    Returns:
        Dictionary with behavioral scores
    """
    scorer = BehavioralScorer()
    result = scorer.predict(features)
    return result.predictions


# Example usage
if __name__ == "__main__":
    # Example features
    example_features = {
        "semantic_match": 0.85,
        "progression": 0.72,
        "linguistic_complexity": 0.68,
        "leadership_alignment": 0.78,
        "collaboration_alignment": 0.82,
        "innovation_alignment": 0.75,
        "adaptability_alignment": 0.70,
        "confidence": 0.80,
        "positivity": 0.75,
        "empathy": 0.68,
        "stress_inverse": 0.72
    }
    
    # Create enhanced behavioral scorer
    scorer = BehavioralScorer(ensemble_config=EnsembleConfig(
        method="weighted_average",
        enable_cross_validation=True,
        ensemble_size=3
    ))
    
    # Get prediction with confidence and feature importance
    result = scorer.predict(example_features)
    
    print("Enhanced Behavioral Scoring Results:")
    print("=" * 50)
    print(f"Predictions: {result.predictions}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Feature Importance: {result.feature_importance}")
    print(f"Ensemble Contributions: {result.ensemble_contributions}")
    print(f"Metadata: {result.metadata}")
    
    # Get performance summary
    performance = scorer.get_performance_summary()
    print(f"\nPerformance Summary: {performance}")
    
    # Test backward compatibility
    simple_scores = scorer.predict_simple(example_features)
    print(f"\nSimple Prediction (Backward Compatible): {simple_scores}")
    
    # Test convenience function
    legacy_scores = predict_behavioral_scores(example_features)
    print(f"Legacy Function Scores: {legacy_scores}")
