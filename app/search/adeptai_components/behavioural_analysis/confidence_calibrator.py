"""
Confidence Calibrator for Multi-Modal AI Systems
================================================
Provides calibrated confidence scores for predictions through:
- Uncertainty quantification and reliability scoring
- Prediction calibration and confidence mapping
- Ensemble agreement analysis and outlier detection
- Adaptive confidence thresholds and quality assessment

Expected Impact: Improved prediction reliability and decision-making confidence
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from datetime import datetime
import json
from collections import defaultdict
import warnings
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

# Import existing components
from .multi_modal_engine import MultiModalEngine, ModelType, ModelPrediction, EnsemblePrediction
from .behavioral_scorer import BehavioralScorer, PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CalibrationMethod(Enum):
    """Methods for confidence calibration"""
    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"
    TEMPERATURE_SCALING = "temperature_scaling"
    ENSEMBLE_CALIBRATION = "ensemble_calibration"
    ADAPTIVE_CALIBRATION = "adaptive_calibration"


class UncertaintyType(Enum):
    """Types of uncertainty quantification"""
    ALEATORIC = "aleatoric"  # Data uncertainty
    EPISTEMIC = "epistemic"   # Model uncertainty
    TOTAL = "total"           # Combined uncertainty


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration"""
    method: CalibrationMethod = CalibrationMethod.ENSEMBLE_CALIBRATION
    uncertainty_type: UncertaintyType = UncertaintyType.TOTAL
    calibration_folds: int = 5
    temperature: float = 1.0
    reliability_threshold: float = 0.8
    outlier_detection: bool = True
    ensemble_agreement_weight: float = 0.3
    feature_quality_weight: float = 0.2
    model_confidence_weight: float = 0.2
    historical_performance_weight: float = 0.15
    prediction_consistency_weight: float = 0.15
    save_calibration_model: bool = True
    auto_recalibrate: bool = True
    recalibration_threshold: float = 0.1


@dataclass
class CalibratedPrediction:
    """Calibrated prediction with confidence and uncertainty"""
    original_prediction: Any
    calibrated_confidence: float
    uncertainty_score: float
    reliability_score: float
    calibration_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'original_prediction': self.original_prediction,
            'calibrated_confidence': self.calibrated_confidence,
            'uncertainty_score': self.uncertainty_score,
            'reliability_score': self.reliability_score,
            'calibration_metadata': self.calibration_metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class CalibrationResult:
    """Results from confidence calibration"""
    calibration_model: Any
    calibration_metrics: Dict[str, float]
    reliability_thresholds: Dict[str, float]
    uncertainty_quantiles: Dict[str, List[float]]
    calibration_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'calibration_metrics': self.calibration_metrics,
            'reliability_thresholds': self.reliability_thresholds,
            'uncertainty_quantiles': self.uncertainty_quantiles,
            'calibration_history': self.calibration_history,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class ConfidenceCalibrator:
    """
    Advanced confidence calibrator that provides calibrated confidence scores
    for predictions with comprehensive uncertainty quantification.
    """
    
    def __init__(self, 
                 multi_modal_engine: Optional[MultiModalEngine] = None,
                 config: Optional[CalibrationConfig] = None):
        
        self.engine = multi_modal_engine or MultiModalEngine()
        self.config = config or CalibrationConfig()
        self.calibration_model = None
        self.calibration_history = []
        self.reliability_thresholds = {}
        self.uncertainty_quantiles = {}
        
        # Initialize calibration model
        self._initialize_calibration_model()
        
        logger.info(f"ConfidenceCalibrator initialized with {self.config.method.value} method")
    
    def _initialize_calibration_model(self):
        """Initialize the calibration model based on configuration"""
        try:
            if self.config.method == CalibrationMethod.PLATT_SCALING:
                self.calibration_model = self._create_platt_scaling_model()
            elif self.config.method == CalibrationMethod.ISOTONIC_REGRESSION:
                self.calibration_model = self._create_isotonic_regression_model()
            elif self.config.method == CalibrationMethod.TEMPERATURE_SCALING:
                self.calibration_model = self._create_temperature_scaling_model()
            elif self.config.method == CalibrationMethod.ENSEMBLE_CALIBRATION:
                self.calibration_model = self._create_ensemble_calibration_model()
            elif self.config.method == CalibrationMethod.ADAPTIVE_CALIBRATION:
                self.calibration_model = self._create_adaptive_calibration_model()
            
            logger.info(f"Calibration model initialized: {self.config.method.value}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize calibration model: {e}")
            self.calibration_model = None
    
    def _create_platt_scaling_model(self):
        """Create Platt scaling calibration model"""
        try:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=42)
        except ImportError:
            logger.warning("Scikit-learn not available for Platt scaling")
            return None
    
    def _create_isotonic_regression_model(self):
        """Create isotonic regression calibration model"""
        try:
            from sklearn.isotonic import IsotonicRegression
            return IsotonicRegression(out_of_bounds='clip')
        except ImportError:
            logger.warning("Scikit-learn not available for isotonic regression")
            return None
    
    def _create_temperature_scaling_model(self):
        """Create temperature scaling calibration model"""
        # Temperature scaling is a simple parameter
        return {'temperature': self.config.temperature}
    
    def _create_ensemble_calibration_model(self):
        """Create ensemble-based calibration model"""
        return {
            'ensemble_weights': self.config.ensemble_agreement_weight,
            'feature_weights': self.config.feature_quality_weight,
            'model_weights': self.config.model_confidence_weight,
            'historical_weights': self.config.historical_performance_weight,
            'consistency_weights': self.config.prediction_consistency_weight
        }
    
    def _create_adaptive_calibration_model(self):
        """Create adaptive calibration model"""
        return {
            'base_model': self._create_ensemble_calibration_model(),
            'adaptation_rate': 0.1,
            'min_samples': 100
        }
    
    def calibrate_confidence(self, 
                            prediction_result: PredictionResult,
                            features: Optional[Dict[str, Any]] = None,
                            historical_data: Optional[List[Dict[str, Any]]] = None) -> CalibratedPrediction:
        """
        Calibrate confidence scores for a prediction result
        
        Args:
            prediction_result: Original prediction result
            features: Input features used for prediction
            historical_data: Historical prediction data for calibration
        
        Returns:
            CalibratedPrediction with calibrated confidence and uncertainty
        """
        
        logger.debug(f"Calibrating confidence for prediction: {prediction_result.predictions}")
        
        # Calculate various confidence factors
        ensemble_confidence = self._calculate_ensemble_agreement_confidence(prediction_result)
        feature_confidence = self._calculate_feature_quality_confidence(features) if features else 0.5
        model_confidence = self._calculate_model_confidence(prediction_result)
        historical_confidence = self._calculate_historical_performance_confidence(historical_data)
        consistency_confidence = self._calculate_prediction_consistency_confidence(prediction_result)
        
        # Combine confidence factors using weighted average
        confidence_factors = {
            'ensemble': ensemble_confidence,
            'feature': feature_confidence,
            'model': model_confidence,
            'historical': historical_confidence,
            'consistency': consistency_confidence
        }
        
        weights = {
            'ensemble': self.config.ensemble_agreement_weight,
            'feature': self.config.feature_quality_weight,
            'model': self.config.model_confidence_weight,
            'historical': self.config.historical_performance_weight,
            'consistency': self.config.prediction_consistency_weight
        }
        
        # Calculate weighted confidence
        calibrated_confidence = self._calculate_weighted_confidence(confidence_factors, weights)
        
        # Calculate uncertainty score
        uncertainty_score = self._calculate_uncertainty_score(confidence_factors, prediction_result)
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(calibrated_confidence, uncertainty_score)
        
        # Create calibration metadata
        calibration_metadata = {
            'confidence_factors': confidence_factors,
            'weights': weights,
            'calibration_method': self.config.method.value,
            'uncertainty_type': self.config.uncertainty_type.value,
            'outlier_detected': self._detect_outliers(prediction_result, features),
            'calibration_model_version': self._get_calibration_model_version()
        }
        
        # Create calibrated prediction
        calibrated_prediction = CalibratedPrediction(
            original_prediction=prediction_result,
            calibrated_confidence=calibrated_confidence,
            uncertainty_score=uncertainty_score,
            reliability_score=reliability_score,
            calibration_metadata=calibration_metadata
        )
        
        # Store calibration history
        self._store_calibration_history(calibrated_prediction)
        
        # Auto-recalibrate if needed
        if self.config.auto_recalibrate:
            self._check_recalibration_needed()
        
        logger.debug(f"Confidence calibrated: {calibrated_confidence:.3f}, Uncertainty: {uncertainty_score:.3f}")
        
        return calibrated_prediction
    
    def _calculate_ensemble_agreement_confidence(self, prediction_result: PredictionResult) -> float:
        """Calculate confidence based on ensemble agreement"""
        
        if not hasattr(prediction_result, 'ensemble_contributions'):
            return 0.5
        
        ensemble_contributions = prediction_result.ensemble_contributions
        
        if not ensemble_contributions:
            return 0.5
        
        # Calculate agreement among ensemble models
        contribution_values = list(ensemble_contributions.values())
        
        if len(contribution_values) < 2:
            return 0.5
        
        # Higher agreement (lower variance) means higher confidence
        variance = np.var(contribution_values)
        agreement_score = max(0.0, 1.0 - variance)
        
        return min(1.0, agreement_score)
    
    def _calculate_feature_quality_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence based on feature quality"""
        
        if not features:
            return 0.5
        
        quality_scores = []
        
        # Check feature completeness
        total_features = len(features)
        non_null_features = sum(1 for v in features.values() if v is not None)
        completeness_score = non_null_features / total_features if total_features > 0 else 0.0
        quality_scores.append(completeness_score)
        
        # Check feature diversity
        if total_features > 1:
            feature_values = [v for v in features.values() if v is not None]
            if feature_values:
                # Some diversity is good, too much might indicate noise
                variance = np.var(feature_values)
                diversity_score = min(1.0, 1.0 - abs(variance - 0.25))
                quality_scores.append(diversity_score)
        
        # Check feature ranges
        range_scores = []
        for value in features.values():
            if isinstance(value, (int, float)) and value is not None:
                if 0.0 <= value <= 1.0:
                    range_scores.append(1.0)
                elif -1.0 <= value <= 2.0:
                    range_scores.append(0.8)
                else:
                    range_scores.append(0.3)
        
        if range_scores:
            quality_scores.append(np.mean(range_scores))
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _calculate_model_confidence(self, prediction_result: PredictionResult) -> float:
        """Calculate confidence based on model performance"""
        
        if not hasattr(prediction_result, 'metadata'):
            return 0.5
        
        metadata = prediction_result.metadata
        
        # Check model type
        model_type = metadata.get('model_type', 'unknown')
        if model_type == 'lightgbm':
            base_confidence = 0.8
        elif model_type == 'fallback':
            base_confidence = 0.6
        else:
            base_confidence = 0.7
        
        # Check ensemble size
        ensemble_size = metadata.get('ensemble_size', 0)
        if ensemble_size > 0:
            ensemble_boost = min(0.2, ensemble_size * 0.05)
            base_confidence += ensemble_boost
        
        return min(1.0, base_confidence)
    
    def _calculate_historical_performance_confidence(self, historical_data: Optional[List[Dict[str, Any]]]) -> float:
        """Calculate confidence based on historical performance"""
        
        if not historical_data or len(historical_data) < 10:
            return 0.5
        
        # Calculate recent performance metrics
        recent_predictions = historical_data[-50:]  # Last 50 predictions
        
        if not recent_predictions:
            return 0.5
        
        # Calculate average confidence and reliability
        confidences = []
        reliabilities = []
        
        for pred in recent_predictions:
            if 'calibrated_confidence' in pred:
                confidences.append(pred['calibrated_confidence'])
            if 'reliability_score' in pred:
                reliabilities.append(pred['reliability_score'])
        
        if confidences and reliabilities:
            avg_confidence = np.mean(confidences)
            avg_reliability = np.mean(reliabilities)
            
            # Higher historical performance means higher confidence
            historical_score = (avg_confidence + avg_reliability) / 2.0
            
            # Boost confidence if historical performance is good
            if historical_score > 0.8:
                return min(1.0, historical_score + 0.1)
            else:
                return historical_score
        
        return 0.5
    
    def _calculate_prediction_consistency_confidence(self, prediction_result: PredictionResult) -> float:
        """Calculate confidence based on prediction consistency"""
        
        predictions = prediction_result.predictions
        
        if not predictions:
            return 0.5
        
        values = list(predictions.values())
        
        if len(values) < 2:
            return 0.5
        
        # Check if predictions are in reasonable ranges
        reasonable_count = sum(1 for v in values if 0.0 <= v <= 1.0)
        range_score = reasonable_count / len(values)
        
        # Check for logical consistency
        consistency_score = 0.7  # Default
        
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
        
        return (range_score + consistency_score) / 2.0
    
    def _calculate_weighted_confidence(self, confidence_factors: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted confidence from multiple factors"""
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            return np.mean(list(confidence_factors.values()))
        
        weighted_sum = 0.0
        for factor, weight in weights.items():
            if factor in confidence_factors:
                weighted_sum += confidence_factors[factor] * weight
        
        return weighted_sum / total_weight
    
    def _calculate_uncertainty_score(self, confidence_factors: Dict[str, float], prediction_result: PredictionResult) -> float:
        """Calculate uncertainty score based on confidence factors and prediction"""
        
        # Base uncertainty is inverse of average confidence
        base_uncertainty = 1.0 - np.mean(list(confidence_factors.values()))
        
        # Additional uncertainty from prediction variance
        predictions = prediction_result.predictions
        if predictions and len(predictions) > 1:
            pred_values = list(predictions.values())
            pred_variance = np.var(pred_values)
            variance_uncertainty = min(0.3, pred_variance)
        else:
            variance_uncertainty = 0.0
        
        # Additional uncertainty from ensemble disagreement
        ensemble_uncertainty = 0.0
        if hasattr(prediction_result, 'ensemble_contributions'):
            ensemble_contributions = prediction_result.ensemble_contributions
            if ensemble_contributions and len(ensemble_contributions) > 1:
                contribution_values = list(ensemble_contributions.values())
                ensemble_variance = np.var(contribution_values)
                ensemble_uncertainty = min(0.2, ensemble_variance)
        
        # Combine uncertainty sources
        total_uncertainty = base_uncertainty + variance_uncertainty + ensemble_uncertainty
        
        return min(1.0, total_uncertainty)
    
    def _calculate_reliability_score(self, calibrated_confidence: float, uncertainty_score: float) -> float:
        """Calculate reliability score based on confidence and uncertainty"""
        
        # Reliability is high when confidence is high and uncertainty is low
        confidence_component = calibrated_confidence
        uncertainty_component = 1.0 - uncertainty_score
        
        # Weight the components
        reliability = 0.7 * confidence_component + 0.3 * uncertainty_component
        
        return min(1.0, max(0.0, reliability))
    
    def _detect_outliers(self, prediction_result: PredictionResult, features: Optional[Dict[str, Any]]) -> bool:
        """Detect if the prediction is an outlier"""
        
        if not self.config.outlier_detection:
            return False
        
        # Simple outlier detection based on prediction values
        predictions = prediction_result.predictions
        if not predictions:
            return False
        
        pred_values = list(predictions.values())
        
        # Check for extreme values
        extreme_count = sum(1 for v in pred_values if v < 0.0 or v > 1.0)
        if extreme_count > 0:
            return True
        
        # Check for very low or very high predictions
        very_low = sum(1 for v in pred_values if v < 0.1)
        very_high = sum(1 for v in pred_values if v > 0.9)
        
        if very_low > len(pred_values) * 0.8 or very_high > len(pred_values) * 0.8:
            return True
        
        return False
    
    def _get_calibration_model_version(self) -> str:
        """Get version information for the calibration model"""
        return f"{self.config.method.value}_{self.config.uncertainty_type.value}_v1.0"
    
    def _store_calibration_history(self, calibrated_prediction: CalibratedPrediction):
        """Store calibration history for analysis"""
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'calibrated_confidence': calibrated_prediction.calibrated_confidence,
            'uncertainty_score': calibrated_prediction.uncertainty_score,
            'reliability_score': calibrated_prediction.reliability_score,
            'calibration_method': calibrated_prediction.calibration_metadata['calibration_method'],
            'outlier_detected': calibrated_prediction.calibration_metadata['outlier_detected']
        }
        
        self.calibration_history.append(history_entry)
        
        # Keep only recent history to prevent memory issues
        if len(self.calibration_history) > 1000:
            self.calibration_history = self.calibration_history[-500:]
    
    def _check_recalibration_needed(self):
        """Check if recalibration is needed based on recent performance"""
        
        if len(self.calibration_history) < 100:
            return
        
        recent_history = self.calibration_history[-100:]
        
        # Calculate recent performance metrics
        recent_confidences = [h['calibrated_confidence'] for h in recent_history]
        recent_reliabilities = [h['reliability_score'] for h in recent_history]
        
        if not recent_confidences or not recent_reliabilities:
            return
        
        avg_confidence = np.mean(recent_confidences)
        avg_reliability = np.mean(recent_reliabilities)
        
        # Check if performance has degraded significantly
        if avg_confidence < 0.6 or avg_reliability < 0.6:
            logger.info("Performance degradation detected, recalibration recommended")
            self._trigger_recalibration()
    
    def _trigger_recalibration(self):
        """Trigger recalibration of the confidence model"""
        
        logger.info("Triggering recalibration of confidence model")
        
        # For now, just reinitialize the model
        # In practice, you'd retrain on recent data
        self._initialize_calibration_model()
        
        # Update calibration history
        recalibration_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'recalibration_triggered',
            'reason': 'performance_degradation',
            'previous_avg_confidence': np.mean([h['calibrated_confidence'] for h in self.calibration_history[-100:]])
        }
        
        self.calibration_history.append(recalibration_entry)
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration performance"""
        
        if not self.calibration_history:
            return {}
        
        recent_history = self.calibration_history[-100:] if len(self.calibration_history) >= 100 else self.calibration_history
        
        confidences = [h['calibrated_confidence'] for h in recent_history]
        uncertainties = [h['uncertainty_score'] for h in recent_history]
        reliabilities = [h['reliability_score'] for h in recent_history]
        
        summary = {
            'total_calibrations': len(self.calibration_history),
            'recent_calibrations': len(recent_history),
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'avg_uncertainty': float(np.mean(uncertainties)) if uncertainties else 0.0,
            'avg_reliability': float(np.mean(reliabilities)) if reliabilities else 0.0,
            'confidence_std': float(np.std(confidences)) if confidences else 0.0,
            'uncertainty_std': float(np.std(uncertainties)) if uncertainties else 0.0,
            'reliability_std': float(np.std(reliabilities)) if reliabilities else 0.0,
            'calibration_method': self.config.method.value,
            'uncertainty_type': self.config.uncertainty_type.value,
            'outlier_detection_enabled': self.config.outlier_detection,
            'auto_recalibration_enabled': self.config.auto_recalibrate
        }
        
        return summary
    
    def save_calibration_model(self, filepath: str):
        """Save calibration model to file"""
        
        if not self.config.save_calibration_model:
            return
        
        try:
            calibration_data = {
                'config': {
                    'method': self.config.method.value,
                    'uncertainty_type': self.config.uncertainty_type.value,
                    'weights': {
                        'ensemble': self.config.ensemble_agreement_weight,
                        'feature': self.config.feature_quality_weight,
                        'model': self.config.model_confidence_weight,
                        'historical': self.config.historical_performance_weight,
                        'consistency': self.config.prediction_consistency_weight
                    }
                },
                'calibration_history': self.calibration_history[-100:],  # Save recent history
                'reliability_thresholds': self.reliability_thresholds,
                'uncertainty_quantiles': self.uncertainty_quantiles,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2, default=str)
            
            logger.info(f"Calibration model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration model: {e}")
    
    def load_calibration_model(self, filepath: str):
        """Load calibration model from file"""
        
        try:
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)
            
            # Update configuration
            if 'config' in calibration_data:
                config = calibration_data['config']
                if 'method' in config:
                    self.config.method = CalibrationMethod(config['method'])
                if 'uncertainty_type' in config:
                    self.config.uncertainty_type = UncertaintyType(config['uncertainty_type'])
                if 'weights' in config:
                    weights = config['weights']
                    self.config.ensemble_agreement_weight = weights.get('ensemble', 0.3)
                    self.config.feature_quality_weight = weights.get('feature', 0.2)
                    self.config.model_confidence_weight = weights.get('model', 0.2)
                    self.config.historical_performance_weight = weights.get('historical', 0.15)
                    self.config.prediction_consistency_weight = weights.get('consistency', 0.15)
            
            # Load history and thresholds
            if 'calibration_history' in calibration_data:
                self.calibration_history = calibration_data['calibration_history']
            if 'reliability_thresholds' in calibration_data:
                self.reliability_thresholds = calibration_data['reliability_thresholds']
            if 'uncertainty_quantiles' in calibration_data:
                self.uncertainty_quantiles = calibration_data['uncertainty_quantiles']
            
            logger.info(f"Calibration model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load calibration model: {e}")


# Convenience function for quick confidence calibration
def calibrate_prediction_confidence(prediction_result: PredictionResult,
                                   features: Optional[Dict[str, Any]] = None,
                                   historical_data: Optional[List[Dict[str, Any]]] = None,
                                   config: Optional[CalibrationConfig] = None) -> CalibratedPrediction:
    """
    Quick function for calibrating prediction confidence
    
    Args:
        prediction_result: Original prediction result
        features: Input features used for prediction
        historical_data: Historical prediction data for calibration
        config: Configuration for calibration
    
    Returns:
        CalibratedPrediction with calibrated confidence and uncertainty
    """
    calibrator = ConfidenceCalibrator(config=config)
    return calibrator.calibrate_confidence(prediction_result, features, historical_data)


# Example usage
if __name__ == "__main__":
    # Example prediction result
    example_prediction = PredictionResult(
        predictions={'leadership': 0.8, 'innovation': 0.7, 'stability': 0.6},
        confidence=0.75,
        feature_importance={'feature1': 0.3, 'feature2': 0.7},
        ensemble_contributions={'model1': 0.4, 'model2': 0.6},
        metadata={'model_type': 'lightgbm', 'ensemble_size': 2}
    )
    
    # Example features
    example_features = {
        'resume_length': 0.8,
        'experience_years': 0.7,
        'skill_count': 0.6,
        'education_level': 0.9
    }
    
    # Create calibration configuration
    config = CalibrationConfig(
        method=CalibrationMethod.ENSEMBLE_CALIBRATION,
        uncertainty_type=UncertaintyType.TOTAL,
        outlier_detection=True,
        auto_recalibrate=True
    )
    
    # Calibrate confidence
    try:
        calibrated_result = calibrate_prediction_confidence(
            example_prediction, 
            example_features, 
            config=config
        )
        
        print("Confidence Calibration Results:")
        print("=" * 50)
        print(f"Original Confidence: {example_prediction.confidence:.3f}")
        print(f"Calibrated Confidence: {calibrated_result.calibrated_confidence:.3f}")
        print(f"Uncertainty Score: {calibrated_result.uncertainty_score:.3f}")
        print(f"Reliability Score: {calibrated_result.reliability_score:.3f}")
        print(f"Outlier Detected: {calibrated_result.calibration_metadata['outlier_detected']}")
        
        # Get calibration summary
        calibrator = ConfidenceCalibrator(config=config)
        summary = calibrator.get_calibration_summary()
        print(f"\nCalibration Summary: {summary}")
        
    except Exception as e:
        print(f"Error during confidence calibration: {e}")
        print("This might be due to missing dependencies or insufficient data.")
