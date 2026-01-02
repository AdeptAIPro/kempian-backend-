"""
Multi-Modal AI Engine for Enhanced Candidate Matching
====================================================
Combines multiple AI models (semantic, emotional, domain, career, behavioral) 
into a unified matching engine with ensemble learning and model fusion.

Key Features:
- Ensemble learning with multiple AI models
- Dynamic model fusion and weighted scoring
- Adaptive confidence scoring
- Cross-model validation and consistency checking
- Performance monitoring and model selection optimization

Expected Impact: +8-10% accuracy improvement through model ensemble
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from datetime import datetime
import json

# Import existing behavioral analysis components
from .semantic_analyzer import SemanticAnalyzer
from .emotion_analyzer import EmotionAnalyzer
from .domain_bert import DomainSpecificBERT, DomainType
from .career_gnn import PretrainedCareerGNN, CareerGraphBuilder
from .behavioral_scorer import BehavioralScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of AI models in the ensemble"""
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"
    DOMAIN = "domain"
    CAREER = "career"
    BEHAVIORAL = "behavioral"


@dataclass
class ModelPrediction:
    """Individual model prediction with confidence and metadata"""
    model_type: ModelType
    predictions: Dict[str, float]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_score(self, dimension: str) -> float:
        """Get prediction score for a specific dimension"""
        return self.predictions.get(dimension, 0.0)


@dataclass
class EnsemblePrediction:
    """Final ensemble prediction combining all models"""
    overall_score: float
    dimension_scores: Dict[str, float]
    model_contributions: Dict[ModelType, float]
    ensemble_confidence: float
    consistency_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'overall_score': self.overall_score,
            'dimension_scores': self.dimension_scores,
            'model_contributions': {k.value: v for k, v in self.model_contributions.items()},
            'ensemble_confidence': self.ensemble_confidence,
            'consistency_score': self.consistency_score,
            'timestamp': self.timestamp.isoformat()
        }


class ModelEnsemble:
    """Manages individual AI models and their predictions"""
    
    def __init__(self, 
                 semantic_model: Optional[SemanticAnalyzer] = None,
                 emotion_model: Optional[EmotionAnalyzer] = None,
                 domain_model: Optional[DomainSpecificBERT] = None,
                 career_model: Optional[PretrainedCareerGNN] = None,
                 behavioral_model: Optional[BehavioralScorer] = None):
        
        self.models = {
            ModelType.SEMANTIC: semantic_model or SemanticAnalyzer(),
            ModelType.EMOTIONAL: emotion_model or EmotionAnalyzer(),
            ModelType.DOMAIN: domain_model or DomainSpecificBERT(),
            ModelType.CAREER: career_model or PretrainedCareerGNN(),
            ModelType.BEHAVIORAL: behavioral_model or BehavioralScorer()
        }
        
        # Model weights (can be dynamically adjusted)
        self.model_weights = {
            ModelType.SEMANTIC: 0.25,
            ModelType.EMOTIONAL: 0.15,
            ModelType.DOMAIN: 0.20,
            ModelType.CAREER: 0.20,
            ModelType.BEHAVIORAL: 0.20
        }
        
        # Model performance tracking
        self.performance_history = {model_type: [] for model_type in ModelType}
        
    def get_model(self, model_type: ModelType):
        """Get a specific model from the ensemble"""
        return self.models.get(model_type)
    
    def is_model_available(self, model_type: ModelType) -> bool:
        """Check if a model is available and functional"""
        model = self.models.get(model_type)
        if model is None:
            return False
        
        # Check for disabled models (e.g., due to missing dependencies)
        if hasattr(model, 'disabled') and model.disabled:
            return False
            
        return True
    
    def get_available_models(self) -> List[ModelType]:
        """Get list of available and functional models"""
        return [model_type for model_type in ModelType if self.is_model_available(model_type)]


class WeightedEnsembleLearner:
    """Advanced ensemble learning with dynamic weight adjustment"""
    
    def __init__(self, ensemble: ModelEnsemble):
        self.ensemble = ensemble
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_history = []
        
    def calculate_ensemble_prediction(self, 
                                   predictions: List[ModelPrediction],
                                   job_context: Dict[str, Any] = None) -> EnsemblePrediction:
        """Calculate final ensemble prediction using weighted averaging"""
        
        if not predictions:
            raise ValueError("No predictions provided for ensemble")
        
        # Filter available models
        available_predictions = [p for p in predictions if self.ensemble.is_model_available(p.model_type)]
        
        if not available_predictions:
            raise ValueError("No available models for ensemble prediction")
        
        # Dynamic weight adjustment based on job context
        adjusted_weights = self._adjust_weights_for_context(available_predictions, job_context)
        
        # Calculate weighted ensemble scores
        dimension_scores = self._calculate_weighted_scores(available_predictions, adjusted_weights)
        
        # Calculate overall score
        overall_score = np.mean(list(dimension_scores.values()))
        
        # Calculate model contributions
        model_contributions = self._calculate_model_contributions(available_predictions, adjusted_weights)
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(available_predictions, adjusted_weights)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(available_predictions)
        
        return EnsemblePrediction(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            model_contributions=model_contributions,
            ensemble_confidence=ensemble_confidence,
            consistency_score=consistency_score
        )
    
    def _adjust_weights_for_context(self, 
                                  predictions: List[ModelPrediction], 
                                  job_context: Dict[str, Any] = None) -> Dict[ModelType, float]:
        """Dynamically adjust model weights based on job context"""
        
        base_weights = self.ensemble.model_weights.copy()
        
        if not job_context:
            return base_weights
        
        # Adjust weights based on job domain
        domain = job_context.get('domain', 'general')
        if domain in ['tech', 'software', 'engineering']:
            base_weights[ModelType.SEMANTIC] *= 1.2
            base_weights[ModelType.CAREER] *= 1.1
        elif domain in ['healthcare', 'medical']:
            base_weights[ModelType.DOMAIN] *= 1.3
            base_weights[ModelType.BEHAVIORAL] *= 1.1
        elif domain in ['finance', 'banking']:
            base_weights[ModelType.DOMAIN] *= 1.2
            base_weights[ModelType.EMOTIONAL] *= 0.9  # Less emphasis on emotions in finance
        
        # Adjust weights based on seniority level
        seniority = job_context.get('seniority', 'mid')
        if seniority in ['senior', 'lead', 'executive']:
            base_weights[ModelType.CAREER] *= 1.2
            base_weights[ModelType.BEHAVIORAL] *= 1.1
        elif seniority in ['junior', 'entry']:
            base_weights[ModelType.SEMANTIC] *= 1.1
            base_weights[ModelType.EMOTIONAL] *= 1.1
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        return {k: v / total_weight for k, v in base_weights.items()}
    
    def _calculate_weighted_scores(self, 
                                 predictions: List[ModelPrediction], 
                                 weights: Dict[ModelType, float]) -> Dict[str, float]:
        """Calculate weighted scores for each dimension"""
        
        dimension_scores = {}
        dimension_weights = {}
        
        for pred in predictions:
            weight = weights.get(pred.model_type, 0.0)
            for dimension, score in pred.predictions.items():
                if dimension not in dimension_scores:
                    dimension_scores[dimension] = 0.0
                    dimension_weights[dimension] = 0.0
                
                dimension_scores[dimension] += score * weight
                dimension_weights[dimension] += weight
        
        # Normalize by total weights
        for dimension in dimension_scores:
            if dimension_weights[dimension] > 0:
                dimension_scores[dimension] /= dimension_weights[dimension]
        
        return dimension_scores
    
    def _calculate_model_contributions(self, 
                                     predictions: List[ModelPrediction], 
                                     weights: Dict[ModelType, float]) -> Dict[ModelType, float]:
        """Calculate how much each model contributed to the final prediction"""
        return {pred.model_type: weights.get(pred.model_type, 0.0) for pred in predictions}
    
    def _calculate_ensemble_confidence(self, 
                                     predictions: List[ModelPrediction], 
                                     weights: Dict[ModelType, float]) -> float:
        """Calculate overall confidence of the ensemble"""
        
        # Weighted average of individual model confidences
        weighted_confidence = sum(
            pred.confidence * weights.get(pred.model_type, 0.0) 
            for pred in predictions
        )
        
        # Boost confidence if models agree (consistency)
        consistency_boost = self._calculate_consistency_score(predictions) * 0.2
        
        return min(1.0, weighted_confidence + consistency_boost)
    
    def _calculate_consistency_score(self, predictions: List[ModelPrediction]) -> float:
        """Calculate how consistent the model predictions are"""
        
        if len(predictions) < 2:
            return 1.0
        
        # Calculate standard deviation across models for each dimension
        all_dimensions = set()
        for pred in predictions:
            all_dimensions.update(pred.predictions.keys())
        
        consistency_scores = []
        for dimension in all_dimensions:
            scores = [pred.get_score(dimension) for pred in predictions if dimension in pred.predictions]
            if len(scores) > 1:
                std_dev = np.std(scores)
                # Lower std dev = higher consistency
                consistency = max(0.0, 1.0 - std_dev)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0


class MultiModalEngine:
    """
    Main multi-modal AI engine that orchestrates all models and provides unified matching
    """
    
    def __init__(self, 
                 model_ensemble: Optional[ModelEnsemble] = None,
                 ensemble_learner: Optional[WeightedEnsembleLearner] = None,
                 cache_predictions: bool = True):
        
        self.ensemble = model_ensemble or ModelEnsemble()
        self.ensemble_learner = ensemble_learner or WeightedEnsembleLearner(self.ensemble)
        self.cache_predictions = cache_predictions
        self.prediction_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'cache_hits': 0,
            'average_processing_time': 0.0,
            'model_accuracy_tracking': {}
        }
        
        logger.info(f"MultiModalEngine initialized with {len(self.ensemble.get_available_models())} available models")
    
    def analyze_candidate(self, 
                         candidate_data: Dict[str, Any],
                         job_description: str,
                         job_context: Optional[Dict[str, Any]] = None) -> EnsemblePrediction:
        """
        Analyze a candidate using all available models and return ensemble prediction
        
        Args:
            candidate_data: Dictionary containing candidate information
            job_description: Job description text
            job_context: Additional job context (domain, seniority, etc.)
        
        Returns:
            EnsemblePrediction with combined results from all models
        """
        
        start_time = datetime.now()
        
        # Check cache first
        cache_key = self._generate_cache_key(candidate_data, job_description)
        if self.cache_predictions and cache_key in self.prediction_cache:
            self.performance_metrics['cache_hits'] += 1
            logger.info(f"Cache hit for candidate analysis")
            return self.prediction_cache[cache_key]
        
        # Generate predictions from all available models
        predictions = []
        
        # Semantic analysis
        if self.ensemble.is_model_available(ModelType.SEMANTIC):
            semantic_pred = self._run_semantic_analysis(candidate_data, job_description)
            if semantic_pred:
                predictions.append(semantic_pred)
        
        # Emotional analysis
        if self.ensemble.is_model_available(ModelType.EMOTIONAL):
            emotion_pred = self._run_emotion_analysis(candidate_data)
            if emotion_pred:
                predictions.append(emotion_pred)
        
        # Domain-specific analysis
        if self.ensemble.is_model_available(ModelType.DOMAIN):
            domain_pred = self._run_domain_analysis(candidate_data, job_description)
            if domain_pred:
                predictions.append(domain_pred)
        
        # Career trajectory analysis
        if self.ensemble.is_model_available(ModelType.CAREER):
            career_pred = self._run_career_analysis(candidate_data)
            if career_pred:
                predictions.append(career_pred)
        
        # Behavioral scoring
        if self.ensemble.is_model_available(ModelType.BEHAVIORAL):
            behavioral_pred = self._run_behavioral_analysis(candidate_data, job_description)
            if behavioral_pred:
                predictions.append(behavioral_pred)
        
        if not predictions:
            raise RuntimeError("No models available for prediction")
        
        # Generate ensemble prediction
        ensemble_prediction = self.ensemble_learner.calculate_ensemble_prediction(
            predictions, job_context
        )
        
        # Cache the result
        if self.cache_predictions:
            self.prediction_cache[cache_key] = ensemble_prediction
        
        # Update performance metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_performance_metrics(processing_time)
        
        logger.info(f"Generated ensemble prediction with {len(predictions)} models in {processing_time:.2f}s")
        
        return ensemble_prediction
    
    def _run_semantic_analysis(self, candidate_data: Dict[str, Any], job_description: str) -> Optional[ModelPrediction]:
        """Run semantic analysis using the semantic analyzer"""
        try:
            semantic_model = self.ensemble.get_model(ModelType.SEMANTIC)
            
            resume_text = candidate_data.get('resume_text', '')
            if not resume_text:
                return None
            
            # Extract resume segments
            resume_segments = self._extract_resume_segments(resume_text)
            
            # Calculate semantic scores
            resume_jd_similarity = semantic_model.resume_jd_similarity(resume_text, job_description)
            
            # Analyze progression if career history available
            career_history = candidate_data.get('career_history', [])
            progression_score = semantic_model.analyze_progression([role.get('title', '') for role in career_history])
            
            # Get exemplar alignment
            exemplar_scores = semantic_model.exemplar_alignment(resume_text)
            
            # Segment alignment
            segment_scores = semantic_model.segment_alignment(resume_segments, job_description)
            avg_segment_score = np.mean([score for _, score in segment_scores]) if segment_scores else 0.5
            
            predictions = {
                'semantic_match': resume_jd_similarity,
                'progression': progression_score,
                'leadership_alignment': exemplar_scores.get('leadership', 0.0),
                'collaboration_alignment': exemplar_scores.get('collaboration', 0.0),
                'innovation_alignment': exemplar_scores.get('innovation', 0.0),
                'adaptability_alignment': exemplar_scores.get('adaptability', 0.0),
                'segment_alignment': avg_segment_score
            }
            
            # Calculate confidence based on text quality and length
            confidence = min(1.0, len(resume_text) / 1000.0)  # Normalize by expected length
            
            return ModelPrediction(
                model_type=ModelType.SEMANTIC,
                predictions=predictions,
                confidence=confidence,
                metadata={'resume_length': len(resume_text), 'segments_analyzed': len(resume_segments)}
            )
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return None
    
    def _run_emotion_analysis(self, candidate_data: Dict[str, Any]) -> Optional[ModelPrediction]:
        """Run emotional analysis using the emotion analyzer"""
        try:
            emotion_model = self.ensemble.get_model(ModelType.EMOTIONAL)
            
            # Analyze text from multiple sources
            text_sources = []
            
            resume_text = candidate_data.get('resume_text', '')
            if resume_text:
                text_sources.append(resume_text)
            
            linkedin_summary = candidate_data.get('linkedin_data', {}).get('summary', '')
            if linkedin_summary:
                text_sources.append(linkedin_summary)
            
            if not text_sources:
                return None
            
            # Combine all text for analysis
            combined_text = ' '.join(text_sources)
            
            # Run emotion analysis
            emotion_results = emotion_model.analyze(combined_text)
            
            predictions = {
                'confidence': emotion_results['factors'].get('confidence', 0.0),
                'positivity': emotion_results['factors'].get('positivity', 0.0),
                'empathy': emotion_results['factors'].get('empathy', 0.0),
                'stress_inverse': 1.0 - emotion_results['factors'].get('stress', 0.0)
            }
            
            # Calculate confidence based on text length and emotion distribution
            emotion_distribution = emotion_results['emotions']
            confidence = min(1.0, len(combined_text) / 500.0)  # Normalize by expected length
            
            return ModelPrediction(
                model_type=ModelType.EMOTIONAL,
                predictions=predictions,
                confidence=confidence,
                metadata={'text_length': len(combined_text), 'emotion_distribution': emotion_distribution}
            )
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return None
    
    def _run_domain_analysis(self, candidate_data: Dict[str, Any], job_description: str) -> Optional[ModelPrediction]:
        """Run domain-specific analysis using the domain BERT"""
        try:
            domain_model = self.ensemble.get_model(ModelType.DOMAIN)
            
            if hasattr(domain_model, 'disabled') and domain_model.disabled:
                return None
            
            resume_text = candidate_data.get('resume_text', '')
            if not resume_text:
                return None
            
            # Detect domain from resume
            detected_domain = domain_model.detect_domain(resume_text)
            
            # Get domain-specific embeddings and similarity
            resume_embedding = domain_model.get_embedding(resume_text)
            job_embedding = domain_model.get_embedding(job_description)
            
            # Calculate domain-specific similarity
            domain_similarity = domain_model.cosine_similarity(resume_embedding, job_embedding)
            
            # Domain alignment score
            domain_alignment = 1.0 if detected_domain == domain_model.detect_domain(job_description) else 0.5
            
            predictions = {
                'domain_similarity': domain_similarity,
                'domain_alignment': domain_alignment,
                'domain_expertise': domain_similarity * domain_alignment
            }
            
            # Confidence based on domain detection certainty
            confidence = 0.8 if detected_domain != DomainType.GENERAL else 0.6
            
            return ModelPrediction(
                model_type=ModelType.DOMAIN,
                predictions=predictions,
                confidence=confidence,
                metadata={'detected_domain': detected_domain.value if hasattr(detected_domain, 'value') else str(detected_domain) if detected_domain else 'unknown'}
            )
            
        except Exception as e:
            logger.error(f"Error in domain analysis: {e}")
            return None
    
    def _run_career_analysis(self, candidate_data: Dict[str, Any]) -> Optional[ModelPrediction]:
        """Run career trajectory analysis using the career GNN"""
        try:
            career_model = self.ensemble.get_model(ModelType.CAREER)
            
            if hasattr(career_model, 'disabled') and career_model.disabled:
                return None
            
            career_history = candidate_data.get('career_history', [])
            if not career_history:
                return None
            
            # Build career graph
            career_graph = CareerGraphBuilder.build_career_graph(career_history)
            
            # Get career predictions
            career_predictions = career_model.predict_career_patterns(career_graph)
            
            predictions = {
                'leadership_potential': career_predictions.get('leadership', 0.0),
                'innovation_capacity': career_predictions.get('innovation', 0.0),
                'growth_trajectory': career_predictions.get('growth', 0.0),
                'career_stability': career_predictions.get('stability', 0.0)
            }
            
            # Confidence based on career history length and quality
            confidence = min(1.0, len(career_history) / 5.0)  # Normalize by expected career length
            
            return ModelPrediction(
                model_type=ModelType.CAREER,
                predictions=predictions,
                confidence=confidence,
                metadata={'career_positions': len(career_history)}
            )
            
        except Exception as e:
            logger.error(f"Error in career analysis: {e}")
            return None
    
    def _run_behavioral_analysis(self, candidate_data: Dict[str, Any], job_description: str) -> Optional[ModelPrediction]:
        """Run behavioral analysis using the behavioral scorer"""
        try:
            behavioral_model = self.ensemble.get_model(ModelType.BEHAVIORAL)
            
            resume_text = candidate_data.get('resume_text', '')
            if not resume_text:
                return None
            
            # Extract features for behavioral scoring
            features = self._extract_behavioral_features(candidate_data, job_description)
            
            # Get behavioral predictions
            behavioral_scores = behavioral_model.predict(features)
            
            predictions = {
                'leadership': behavioral_scores.get('leadership', 0.0),
                'innovation': behavioral_scores.get('innovation', 0.0),
                'stability': behavioral_scores.get('stability', 0.0)
            }
            
            # Confidence based on feature availability
            feature_coverage = len([f for f in features.values() if f > 0]) / len(features)
            confidence = feature_coverage
            
            return ModelPrediction(
                model_type=ModelType.BEHAVIORAL,
                predictions=predictions,
                confidence=confidence,
                metadata={'feature_coverage': feature_coverage}
            )
            
        except Exception as e:
            logger.error(f"Error in behavioral analysis: {e}")
            return None
    
    def _extract_resume_segments(self, resume_text: str) -> List[str]:
        """Extract meaningful segments from resume text"""
        # Simple segmentation by bullet points and line breaks
        segments = []
        
        # Split by bullet points
        bullet_segments = [seg.strip() for seg in resume_text.split('•') if seg.strip()]
        segments.extend(bullet_segments)
        
        # Split by line breaks
        line_segments = [seg.strip() for seg in resume_text.split('\n') if seg.strip() and len(seg.strip()) > 10]
        segments.extend(line_segments)
        
        # Remove duplicates and short segments
        unique_segments = list(set(segments))
        return [seg for seg in unique_segments if len(seg) > 20]
    
    def _extract_behavioral_features(self, candidate_data: Dict[str, Any], job_description: str) -> Dict[str, float]:
        """Extract behavioral features for the behavioral scorer"""
        features = {
            'semantic_match': 0.0,
            'progression': 0.0,
            'linguistic_complexity': 0.0,
            'leadership_alignment': 0.0,
            'collaboration_alignment': 0.0,
            'innovation_alignment': 0.0,
            'adaptability_alignment': 0.0,
            'confidence': 0.0,
            'positivity': 0.0,
            'empathy': 0.0,
            'stress_inverse': 0.0
        }
        
        # Get semantic features if available
        semantic_model = self.ensemble.get_model(ModelType.SEMANTIC)
        if semantic_model:
            resume_text = candidate_data.get('resume_text', '')
            if resume_text:
                features['semantic_match'] = semantic_model.resume_jd_similarity(resume_text, job_description)
                
                # Career progression
                career_history = candidate_data.get('career_history', [])
                if career_history:
                    features['progression'] = semantic_model.analyze_progression(
                        [role.get('title', '') for role in career_history]
                    )
                
                # Exemplar alignment
                exemplar_scores = semantic_model.exemplar_alignment(resume_text)
                features['leadership_alignment'] = exemplar_scores.get('leadership', 0.0)
                features['collaboration_alignment'] = exemplar_scores.get('collaboration', 0.0)
                features['innovation_alignment'] = exemplar_scores.get('innovation', 0.0)
                features['adaptability_alignment'] = exemplar_scores.get('adaptability', 0.0)
        
        # Get emotional features if available
        emotion_model = self.ensemble.get_model(ModelType.EMOTIONAL)
        if emotion_model:
            text_sources = []
            resume_text = candidate_data.get('resume_text', '')
            if resume_text:
                text_sources.append(resume_text)
            
            linkedin_summary = candidate_data.get('linkedin_data', {}).get('summary', '')
            if linkedin_summary:
                text_sources.append(linkedin_summary)
            
            if text_sources:
                combined_text = ' '.join(text_sources)
                emotion_results = emotion_model.analyze(combined_text)
                features['confidence'] = emotion_results['factors'].get('confidence', 0.0)
                features['positivity'] = emotion_results['factors'].get('positivity', 0.0)
                features['empathy'] = emotion_results['factors'].get('empathy', 0.0)
                features['stress_inverse'] = 1.0 - emotion_results['factors'].get('stress', 0.0)
        
        # Linguistic complexity (simple heuristic)
        resume_text = candidate_data.get('resume_text', '')
        if resume_text:
            words = resume_text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            features['linguistic_complexity'] = min(1.0, avg_word_length / 8.0)  # Normalize
        
        return features
    
    def _generate_cache_key(self, candidate_data: Dict[str, Any], job_description: str) -> str:
        """Generate cache key for predictions"""
        import hashlib
        
        # Create a stable representation of the data
        candidate_hash = hashlib.md5(
            json.dumps(candidate_data, sort_keys=True).encode()
        ).hexdigest()
        
        job_hash = hashlib.md5(job_description.encode()).hexdigest()
        
        return f"{candidate_hash}_{job_hash}"
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking metrics"""
        self.performance_metrics['total_predictions'] += 1
        self.performance_metrics['average_processing_time'] = (
            (self.performance_metrics['average_processing_time'] * 
             (self.performance_metrics['total_predictions'] - 1) + processing_time) / 
            self.performance_metrics['total_predictions']
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the multi-modal engine"""
        return {
            'total_predictions': self.performance_metrics['total_predictions'],
            'cache_hits': self.performance_metrics['cache_hits'],
            'cache_hit_rate': (self.performance_metrics['cache_hits'] / 
                              max(1, self.performance_metrics['total_predictions'])),
            'average_processing_time': self.performance_metrics['average_processing_time'],
            'available_models': len(self.ensemble.get_available_models()),
            'model_weights': {k.value: v for k, v in self.ensemble.model_weights.items()}
        }
    
    def save_model_weights(self, filepath: str):
        """Save current model weights to file"""
        weights_data = {
            'model_weights': {k.value: v for k, v in self.ensemble.model_weights.items()},
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        logger.info(f"Model weights saved to {filepath}")
    
    def load_model_weights(self, filepath: str):
        """Load model weights from file"""
        try:
            with open(filepath, 'r') as f:
                weights_data = json.load(f)
            
            # Update model weights
            for model_name, weight in weights_data['model_weights'].items():
                model_type = ModelType(model_name)
                if model_type in self.ensemble.model_weights:
                    self.ensemble.model_weights[model_type] = weight
            
            logger.info(f"Model weights loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
    
    def optimize_weights(self, validation_data: List[Tuple[Dict, str, float]]):
        """
        Optimize model weights using validation data
        
        Args:
            validation_data: List of (candidate_data, job_description, true_score) tuples
        """
        logger.info("Starting weight optimization...")
        
        # Simple grid search for weight optimization
        best_weights = self.ensemble.model_weights.copy()
        best_score = 0.0
        
        # Weight adjustment ranges
        weight_adjustments = [-0.1, -0.05, 0.0, 0.05, 0.1]
        
        for _ in range(10):  # 10 optimization iterations
            current_weights = self.ensemble.model_weights.copy()
            
            # Randomly adjust weights
            for model_type in current_weights:
                adjustment = np.random.choice(weight_adjustments)
                current_weights[model_type] = max(0.0, min(1.0, current_weights[model_type] + adjustment))
            
            # Normalize weights
            total_weight = sum(current_weights.values())
            current_weights = {k: v / total_weight for k, v in current_weights.items()}
            
            # Test weights on validation data
            self.ensemble.model_weights = current_weights
            validation_score = self._evaluate_weights(validation_data)
            
            if validation_score > best_score:
                best_score = validation_score
                best_weights = current_weights.copy()
                logger.info(f"New best validation score: {best_score:.4f}")
        
        # Apply best weights
        self.ensemble.model_weights = best_weights
        logger.info(f"Weight optimization completed. Best validation score: {best_score:.4f}")
    
    def _evaluate_weights(self, validation_data: List[Tuple[Dict, str, float]]) -> float:
        """Evaluate current weights on validation data"""
        predictions = []
        true_scores = []
        
        for candidate_data, job_description, true_score in validation_data:
            try:
                ensemble_pred = self.analyze_candidate(candidate_data, job_description)
                predictions.append(ensemble_pred.overall_score)
                true_scores.append(true_score)
            except Exception as e:
                logger.warning(f"Error evaluating candidate: {e}")
                continue
        
        if not predictions:
            return 0.0
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(predictions, true_scores)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0


# Convenience function for quick analysis
def analyze_candidate_quick(candidate_data: Dict[str, Any], 
                           job_description: str,
                           job_context: Optional[Dict[str, Any]] = None) -> EnsemblePrediction:
    """
    Quick analysis function for single candidate evaluation
    
    Args:
        candidate_data: Dictionary containing candidate information
        job_description: Job description text
        job_context: Additional job context (domain, seniority, etc.)
    
    Returns:
        EnsemblePrediction with combined results from all models
    """
    engine = MultiModalEngine()
    return engine.analyze_candidate(candidate_data, job_description, job_context)


# Example usage and testing
if __name__ == "__main__":
    # Example candidate data
    example_candidate = {
        'resume_text': """
        Senior Software Engineer at TechCorp (2020-2023)
        • Led cross-functional team of 8 developers to deliver cloud-native application
        • Implemented microservices architecture improving system performance by 40%
        • Mentored junior developers and conducted code reviews
        
        Software Engineer at StartupInc (2018-2020)
        • Developed RESTful APIs and frontend components
        • Collaborated with product and design teams
        • Participated in agile development processes
        """,
        'career_history': [
            {'title': 'Software Engineer', 'company': 'StartupInc', 'duration': 24},
            {'title': 'Senior Software Engineer', 'company': 'TechCorp', 'duration': 36}
        ],
        'linkedin_data': {
            'summary': 'Passionate software engineer with expertise in cloud technologies and team leadership.'
        }
    }
    
    example_job = """
    We are looking for a Senior Software Engineer to join our growing team.
    The ideal candidate should have:
    - 5+ years of software development experience
    - Experience with cloud technologies and microservices
    - Leadership skills and ability to mentor junior developers
    - Strong collaboration and communication skills
    """
    
    # Run analysis
    try:
        result = analyze_candidate_quick(example_candidate, example_job)
        print("Analysis Results:")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Ensemble Confidence: {result.ensemble_confidence:.3f}")
        print(f"Consistency Score: {result.consistency_score:.3f}")
        print("\nDimension Scores:")
        for dimension, score in result.dimension_scores.items():
            print(f"  {dimension}: {score:.3f}")
        print("\nModel Contributions:")
        for model, contribution in result.model_contributions.items():
            print(f"  {model.value}: {contribution:.3f}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("This might be due to missing dependencies or model files.")

