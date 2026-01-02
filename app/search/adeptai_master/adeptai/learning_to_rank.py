"""
Learning-to-Rank Model for Candidate Ranking

Implements a LambdaRank-based learning-to-rank model using LightGBM
to learn optimal feature weights from user feedback data.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
# Optional: probability calibration and SHAP explainability
try:
    from sklearn.isotonic import IsotonicRegression  # used as a lightweight calibrator
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    IsotonicRegression = None  # type: ignore

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

# Try to import XGBoost as fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class LTRFeatures:
    """Feature vector for learning-to-rank"""
    keyword_score: float
    semantic_score: float
    cross_encoder_score: float
    skill_overlap: float
    experience_match: float
    domain_match: float
    exact_match: float
    skill_count: int
    experience_years: float
    query_length: int
    candidate_text_length: int
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([
            self.keyword_score,
            self.semantic_score,
            self.cross_encoder_score,
            self.skill_overlap,
            self.experience_match,
            self.domain_match,
            self.exact_match,
            self.skill_count / 10.0,  # Normalize
            self.experience_years / 20.0,  # Normalize
            self.query_length / 50.0,  # Normalize
            self.candidate_text_length / 1000.0  # Normalize
        ], dtype=np.float32)


class LearningToRankModel:
    """
    Learning-to-Rank model for candidate ranking using LambdaRank objective.
    
    Features:
    - keyword_score: Keyword-based matching score
    - semantic_score: Semantic similarity score
    - cross_encoder_score: Cross-encoder reranking score
    - skill_overlap: Skill matching score
    - experience_match: Experience level matching
    - domain_match: Domain classification match
    - exact_match: Exact word matching
    - skill_count: Number of skills
    - experience_years: Years of experience
    - query_length: Query text length
    - candidate_text_length: Candidate text length
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Learning-to-Rank model.
        
        Args:
            model_path: Path to saved model file (optional)
        """
        self.model = None
        self.model_path = model_path or os.path.join("model", "ltr_model.pkl")
        self.is_trained = False
        self.model_version = "v1"
        self.calibrator: Optional[IsotonicRegression] = None
        
        # Feature names for explainability
        self.feature_names = [
            'keyword_score',
            'semantic_score',
            'cross_encoder_score',
            'skill_overlap',
            'experience_match',
            'domain_match',
            'exact_match',
            'skill_count',
            'experience_years',
            'query_length',
            'candidate_text_length'
        ]
        
        # Initialize model if available
        if LIGHTGBM_AVAILABLE:
            self._init_model()
        elif XGBOOST_AVAILABLE:
            self._init_xgboost_model()
        else:
            logger.warning("Neither LightGBM nor XGBoost available. LTR model will not work.")
        
        # Load saved model if exists
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _init_model(self):
        """Initialize LightGBM ranker model"""
        if not LIGHTGBM_AVAILABLE:
            return
        
        self.model = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            n_estimators=100
        )
        logger.info("Initialized LightGBM LTR model")
    
    def _init_xgboost_model(self):
        """Initialize XGBoost ranker model as fallback"""
        if not XGBOOST_AVAILABLE:
            return
        
        self.model = xgb.XGBRanker(
            objective='rank:ndcg',
            tree_method='hist',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            verbosity=0
        )
        logger.info("Initialized XGBoost LTR model (fallback)")
    
    def extract_features(
        self,
        query: str,
        candidate: Dict[str, Any],
        keyword_score: float = 0.0,
        semantic_score: float = 0.0,
        cross_encoder_score: float = 0.0
    ) -> LTRFeatures:
        """
        Extract features from query-candidate pair.
        
        Args:
            query: Search query
            candidate: Candidate data dictionary
            keyword_score: Pre-computed keyword score
            semantic_score: Pre-computed semantic score
            cross_encoder_score: Pre-computed cross-encoder score
            
        Returns:
            LTRFeatures object with extracted features
        """
        skills = candidate.get('skills', [])
        resume_text = candidate.get('resume_text', '')
        experience_years = candidate.get('total_experience_years', 0)
        full_name = candidate.get('full_name', '')
        
        # Combine candidate text
        candidate_text = f"{full_name} {' '.join(skills)} {resume_text}".lower()
        query_lower = query.lower()
        
        # Calculate skill overlap
        skill_overlap = self._compute_skill_overlap(query_lower, skills)
        
        # Calculate experience match
        experience_match = self._compute_experience_match(query_lower, experience_years)
        
        # Calculate domain match (simplified)
        domain_match = self._compute_domain_match(query_lower, candidate_text)
        
        # Calculate exact match
        exact_match = self._compute_exact_match(query_lower, candidate_text)
        
        return LTRFeatures(
            keyword_score=keyword_score,
            semantic_score=semantic_score,
            cross_encoder_score=cross_encoder_score,
            skill_overlap=skill_overlap,
            experience_match=experience_match,
            domain_match=domain_match,
            exact_match=exact_match,
            skill_count=len(skills),
            experience_years=float(experience_years),
            query_length=len(query.split()),
            candidate_text_length=len(candidate_text.split())
        )
    
    def _compute_skill_overlap(self, query: str, skills: List[str]) -> float:
        """Compute skill overlap score"""
        if not skills:
            return 0.0
        
        query_words = set(query.split())
        skill_matches = 0
        
        for skill in skills:
            skill_words = set(skill.lower().split())
            if query_words.intersection(skill_words):
                skill_matches += 1
        
        return min(skill_matches / len(skills), 1.0) if skills else 0.0
    
    def _compute_experience_match(self, query: str, experience_years: float) -> float:
        """Compute experience level match score"""
        query_lower = query.lower()
        
        # Experience level keywords
        senior_keywords = {'senior', 'lead', 'principal', 'architect', 'manager', 'director', 'experienced'}
        junior_keywords = {'junior', 'entry', 'graduate', 'intern', 'associate', 'fresh'}
        
        has_senior = any(kw in query_lower for kw in senior_keywords)
        has_junior = any(kw in query_lower for kw in junior_keywords)
        
        if has_senior:
            if experience_years >= 5:
                return 1.0
            elif experience_years >= 3:
                return 0.7
            else:
                return 0.3
        elif has_junior:
            if experience_years <= 3:
                return 1.0
            elif experience_years <= 5:
                return 0.7
            else:
                return 0.3
        else:
            # No specific level mentioned, prefer 3+ years
            if experience_years >= 3:
                return 0.7
            elif experience_years >= 1:
                return 0.5
            else:
                return 0.2
    
    def _compute_domain_match(self, query: str, candidate_text: str) -> float:
        """Compute domain match score"""
        # Domain keywords
        domains = {
            'technology': {'python', 'java', 'javascript', 'react', 'node', 'aws', 'docker', 'kubernetes', 'developer', 'engineer', 'software', 'programming'},
            'healthcare': {'nurse', 'doctor', 'physician', 'medical', 'healthcare', 'hospital', 'clinic', 'patient', 'rn', 'md'},
            'finance': {'finance', 'banking', 'investment', 'accounting', 'financial', 'analyst', 'cpa', 'cfa'},
            'education': {'teacher', 'professor', 'instructor', 'educator', 'academic', 'education'},
            'marketing': {'marketing', 'advertising', 'brand', 'digital', 'social media', 'content', 'seo', 'marketer'}
        }
        
        query_lower = query.lower()
        candidate_lower = candidate_text.lower()
        
        # Find query domain
        query_domain = None
        for domain, keywords in domains.items():
            if any(kw in query_lower for kw in keywords):
                query_domain = domain
                break
        
        if not query_domain:
            return 0.5  # Neutral score if domain unclear
        
        # Check candidate domain match
        candidate_domain = None
        for domain, keywords in domains.items():
            if any(kw in candidate_lower for kw in keywords):
                candidate_domain = domain
                break
        
        if candidate_domain == query_domain:
            return 1.0
        elif candidate_domain:
            return 0.3  # Different domain
        else:
            return 0.5  # Candidate domain unclear
    
    def _compute_exact_match(self, query: str, candidate_text: str) -> float:
        """Compute exact word match score"""
        query_words = set(query.split())
        candidate_words = set(candidate_text.split())
        
        if not query_words:
            return 0.0
        
        matches = len(query_words.intersection(candidate_words))
        return matches / len(query_words)
    
    def train(
        self,
        query_candidate_pairs: List[Tuple[str, Dict[str, Any]]],
        labels: List[float],
        query_groups: Optional[List[int]] = None,
        feature_scores: Optional[List[Dict[str, float]]] = None
    ):
        """
        Train the learning-to-rank model.
        
        Args:
            query_candidate_pairs: List of (query, candidate) tuples
            labels: Relevance labels (0-4 scale, higher is better)
            query_groups: Group IDs for each query (for grouping candidates by query)
            feature_scores: Optional pre-computed feature scores for each pair
        """
        if not self.model:
            logger.error("Model not initialized. Cannot train.")
            return
        
        if len(query_candidate_pairs) != len(labels):
            raise ValueError("query_candidate_pairs and labels must have same length")
        
        # Extract features
        X = []
        y = np.array(labels, dtype=np.float32)
        
        for i, (query, candidate) in enumerate(query_candidate_pairs):
            if feature_scores and i < len(feature_scores):
                scores = feature_scores[i]
                features = self.extract_features(
                    query,
                    candidate,
                    keyword_score=scores.get('keyword_score', 0.0),
                    semantic_score=scores.get('semantic_score', 0.0),
                    cross_encoder_score=scores.get('cross_encoder_score', 0.0)
                )
            else:
                features = self.extract_features(query, candidate)
            
            X.append(features.to_array())
        
        X = np.array(X)
        
        # Generate query groups if not provided
        if query_groups is None:
            # Assume each query has multiple candidates
            # For simplicity, create groups based on query hash
            query_groups = []
            current_group = 0
            prev_query = None
            for query, _ in query_candidate_pairs:
                if query != prev_query:
                    current_group += 1
                    prev_query = query
                query_groups.append(current_group)
        
        query_groups = np.array(query_groups)
        
        # Train model
        logger.info(f"Training LTR model on {len(X)} samples...")
        
        if LIGHTGBM_AVAILABLE and isinstance(self.model, lgb.LGBMRanker):
            self.model.fit(
                X,
                y,
                group=query_groups,
                verbose=False
            )
        elif XGBOOST_AVAILABLE and isinstance(self.model, xgb.XGBRanker):
            # XGBoost requires different format
            self.model.fit(
                X,
                y,
                qid=query_groups
            )
        
        self.is_trained = True
        logger.info("LTR model training completed")
    
    def predict(self, query: str, candidates: List[Dict[str, Any]], 
                feature_scores: Optional[List[Dict[str, float]]] = None) -> List[float]:
        """
        Predict relevance scores for query-candidate pairs.
        
        Args:
            query: Search query
            candidates: List of candidate dictionaries
            feature_scores: Optional pre-computed feature scores for each candidate
            
        Returns:
            List of predicted relevance scores
        """
        if not self.model or not self.is_trained:
            # Fallback to simple weighted average
            logger.warning("LTR model not trained. Using fallback scoring.")
            return self._fallback_predict(query, candidates, feature_scores)
        
        # Extract features for all candidates
        X = []
        for i, candidate in enumerate(candidates):
            if feature_scores and i < len(feature_scores):
                scores = feature_scores[i]
                features = self.extract_features(
                    query,
                    candidate,
                    keyword_score=scores.get('keyword_score', 0.0),
                    semantic_score=scores.get('semantic_score', 0.0),
                    cross_encoder_score=scores.get('cross_encoder_score', 0.0)
                )
            else:
                features = self.extract_features(query, candidate)
            
            X.append(features.to_array())
        
        X = np.array(X)
        
        # Predict scores
        scores = self.model.predict(X)
        
        # Optional calibration (monotonic scaling)
        if self.calibrator is not None:
            try:
                scores = self.calibrator.predict(scores)
            except Exception:
                pass
        
        return scores.tolist()
    
    def _fallback_predict(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        feature_scores: Optional[List[Dict[str, float]]] = None
    ) -> List[float]:
        """Fallback prediction using weighted average when model not trained"""
        scores = []
        for i, candidate in enumerate(candidates):
            if feature_scores and i < len(feature_scores):
                fs = feature_scores[i]
                # Weighted combination similar to original system
                keyword = fs.get('keyword_score', 0.0)
                semantic = fs.get('semantic_score', 0.0)
                cross_encoder = fs.get('cross_encoder_score', 0.0)
                
                if cross_encoder > 0:
                    score = 0.7 * (0.6 * keyword + 0.4 * semantic) + 0.3 * cross_encoder
                elif semantic > 0:
                    score = 0.6 * keyword + 0.4 * semantic
                else:
                    score = keyword
            else:
                # Extract features and compute simple score
                features = self.extract_features(query, candidate)
                score = (
                    0.3 * features.keyword_score +
                    0.3 * features.semantic_score +
                    0.2 * features.skill_overlap +
                    0.1 * features.experience_match +
                    0.1 * features.domain_match
                )
            
            scores.append(score)
        
        return scores
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model to file"""
        if not self.model or not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names,
                    'is_trained': self.is_trained,
                    'model_version': self.model_version,
                    'calibrator': self.calibrator
                }, f)
            logger.info(f"Saved LTR model to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, path: Optional[str] = None):
        """Load trained model from file"""
        path = path or self.model_path
        
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_names = data.get('feature_names', self.feature_names)
                self.is_trained = data.get('is_trained', False)
                self.model_version = data.get('model_version', self.model_version)
                self.calibrator = data.get('calibrator', None)
            
            logger.info(f"Loaded LTR model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.model or not self.is_trained:
            return {}
        
        try:
            if LIGHTGBM_AVAILABLE and isinstance(self.model, lgb.LGBMRanker):
                importance = self.model.feature_importances_
            elif XGBOOST_AVAILABLE and isinstance(self.model, xgb.XGBRanker):
                importance = self.model.feature_importances_
            else:
                return {}
            
            # Normalize importance
            total = sum(importance)
            if total > 0:
                importance = importance / total
            
            return dict(zip(self.feature_names, importance))
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}

    def fit_calibrator(self, validation_scores: List[float], validation_labels: List[float]) -> None:
        """Optionally fit an isotonic calibrator to map raw scores to calibrated scores.
        This is useful for interpretability and threshold selection.
        """
        if not SKLEARN_AVAILABLE:
            logger.info("sklearn not available; skipping calibrator fit")
            return
        try:
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(np.array(validation_scores), np.array(validation_labels))
            self.calibrator = calibrator
            logger.info("Calibrator fitted for LTR scores")
        except Exception as e:
            logger.warning(f"Failed to fit calibrator: {e}")


# Global instance
_ltr_model = None


def get_ltr_model(model_path: Optional[str] = None) -> LearningToRankModel:
    """Get or create global LTR model instance"""
    global _ltr_model
    if _ltr_model is None:
        _ltr_model = LearningToRankModel(model_path=model_path)
    return _ltr_model

