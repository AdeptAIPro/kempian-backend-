"""
Candidate-Job Fit Prediction Model

Predicts the probability that a candidate will be hired for a job
based on historical hiring data and candidate-job features.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

# Try to import sklearn as fallback
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available.")


class JobFitPredictor:
    """
    Candidate-Job Fit Prediction Model
    
    Predicts hire probability based on:
    - Skill match percentage
    - Experience match
    - Education match
    - Domain alignment
    - Behavioral scores
    - Market data (salary, demand)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Job Fit Predictor
        
        Args:
            model_path: Path to saved model (optional)
        """
        self.model_path = model_path or os.path.join("model", "job_fit_predictor.pkl")
        self.is_trained = False
        self.calibrated: bool = False
        
        # Initialize model
        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=100,
                verbose=-1
            )
        elif SKLEARN_AVAILABLE:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        else:
            self.model = None
            logger.warning("No ML library available. Fit prediction will not work.")
        
        # Load saved model if exists
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_features(self, candidate: Dict[str, Any], job: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from candidate-job pair
        
        Args:
            candidate: Candidate data dictionary
            job: Job posting data dictionary
            
        Returns:
            Feature vector
        """
        features = []
        
        # Candidate features
        candidate_skills = candidate.get('skills', [])
        candidate_experience = candidate.get('total_experience_years', 0)
        candidate_education = candidate.get('education', '')
        candidate_domain = candidate.get('domain', 'unknown')
        
        # Job features
        job_skills = job.get('required_skills', [])
        job_experience_required = job.get('experience_required', 0)
        job_education_required = job.get('education_required', '')
        job_domain = job.get('domain', 'unknown')
        job_salary_min = job.get('salary_min', 0)
        job_salary_max = job.get('salary_max', 0)
        
        # 1. Skill match percentage
        if job_skills:
            matching_skills = len(set(candidate_skills) & set(job_skills))
            skill_match = matching_skills / len(job_skills)
        else:
            skill_match = 0.0
        features.append(skill_match)
        
        # 2. Experience match
        if job_experience_required > 0:
            experience_match = min(candidate_experience / job_experience_required, 1.0)
        else:
            experience_match = 1.0 if candidate_experience >= 3 else 0.5
        features.append(experience_match)
        
        # 3. Education match (simplified)
        education_match = 1.0 if candidate_education and job_education_required and job_education_required.lower() in candidate_education.lower() else 0.5
        features.append(education_match)
        
        # 4. Domain alignment
        domain_match = 1.0 if candidate_domain == job_domain else 0.0
        features.append(domain_match)
        
        # 5. Skill count match
        candidate_skill_count = len(candidate_skills)
        job_skill_count = len(job_skills)
        skill_count_ratio = candidate_skill_count / max(job_skill_count, 1)
        features.append(min(skill_count_ratio, 2.0))  # Cap at 2x
        
        # 6. Experience ratio
        experience_ratio = candidate_experience / max(job_experience_required, 1)
        features.append(min(experience_ratio, 2.0))  # Cap at 2x
        
        # 7. Salary match (if available)
        candidate_salary_expectation = candidate.get('salary_expectation', 0)
        if job_salary_min > 0 and candidate_salary_expectation > 0:
            if job_salary_min <= candidate_salary_expectation <= job_salary_max:
                salary_match = 1.0
            elif candidate_salary_expectation < job_salary_min:
                salary_match = 0.8  # Candidate willing to accept lower
            else:
                salary_match = 0.3  # Candidate expects more
        else:
            salary_match = 0.5  # Unknown
        features.append(salary_match)
        
        # 8. Behavioral scores (if available)
        behavioral_score = candidate.get('behavioral_score', 0.5)
        features.append(behavioral_score)
        
        # 9. Market demand (if available)
        market_demand = job.get('market_demand', 0.5)
        features.append(market_demand)
        
        # 10. Candidate quality score
        quality_score = candidate.get('quality_score', 0.5)
        features.append(quality_score)
        
        return np.array(features, dtype=np.float32)
    
    def train(
        self,
        candidates: List[Dict[str, Any]],
        jobs: List[Dict[str, Any]],
        outcomes: List[int],  # 1 for hired, 0 for not hired
        test_size: float = 0.2
    ):
        """
        Train the job fit predictor
        
        Args:
            candidates: List of candidate dictionaries
            jobs: List of job dictionaries
            outcomes: List of hire outcomes (1 = hired, 0 = not hired)
            test_size: Fraction of data to use for testing
        """
        if not self.model:
            logger.error("Model not initialized. Cannot train.")
            return
        
        if len(candidates) != len(jobs) or len(candidates) != len(outcomes):
            raise ValueError("candidates, jobs, and outcomes must have same length")
        
        logger.info(f"Training job fit predictor on {len(candidates)} examples...")
        
        # Extract features
        X = []
        y = np.array(outcomes, dtype=np.int32)
        
        for candidate, job in zip(candidates, jobs):
            features = self.extract_features(candidate, job)
            X.append(features)
        
        X = np.array(X)
        
        # Split data
        if test_size > 0:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        
        # Optional probability calibration when supported
        try:
            if hasattr(self.model, 'predict_proba') and SKLEARN_AVAILABLE:
                calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv='prefit')
                calibrator.fit(X_train, y_train)
                self.model = calibrator
                self.calibrated = True
                logger.info("Calibrated probabilities with isotonic regression")
        except Exception as e:
            logger.debug(f"Calibration skipped: {e}")
        
        # Evaluate
        if X_test is not None:
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            logger.info(f"Training accuracy: {train_score:.4f}")
            logger.info(f"Test accuracy: {test_score:.4f}")
        
        self.is_trained = True
        logger.info("Job fit predictor training completed!")
    
    def predict_fit(self, candidate: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict candidate-job fit
        
        Args:
            candidate: Candidate data dictionary
            job: Job posting data dictionary
            
        Returns:
            Dictionary with fit probability, grade, and key factors
        """
        if not self.model or not self.is_trained:
            # Fallback to simple scoring
            return self._simple_fit_score(candidate, job)
        
        # Extract features
        features = self.extract_features(candidate, job)
        features = features.reshape(1, -1)
        
        # Predict
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                fit_probability = float(probabilities[1])  # Probability of hire
            else:
                prediction = self.model.predict(features)[0]
                fit_probability = float(prediction)
            
            # Get feature importance if available
            key_factors = self._get_key_factors(features, candidate, job)
            
            return {
                'fit_probability': fit_probability,
                'fit_grade': self._grade_probability(fit_probability),
                'key_factors': key_factors,
                'recommendation': self._get_recommendation(fit_probability)
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._simple_fit_score(candidate, job)
    
    def _simple_fit_score(self, candidate: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback scoring"""
        candidate_skills = set(candidate.get('skills', []))
        job_skills = set(job.get('required_skills', []))
        
        if job_skills:
            skill_match = len(candidate_skills & job_skills) / len(job_skills)
        else:
            skill_match = 0.5
        
        return {
            'fit_probability': skill_match,
            'fit_grade': self._grade_probability(skill_match),
            'key_factors': ['skill_match'],
            'recommendation': 'consider' if skill_match > 0.5 else 'review'
        }
    
    def _grade_probability(self, probability: float) -> str:
        """Convert probability to grade"""
        if probability >= 0.8:
            return 'A'
        elif probability >= 0.6:
            return 'B'
        elif probability >= 0.4:
            return 'C'
        else:
            return 'D'
    
    def _get_recommendation(self, probability: float) -> str:
        """Get recommendation based on probability"""
        if probability >= 0.8:
            return 'strong_match'
        elif probability >= 0.6:
            return 'good_match'
        elif probability >= 0.4:
            return 'fair_match'
        else:
            return 'weak_match'
    
    def _get_key_factors(self, features: np.ndarray, candidate: Dict, job: Dict) -> List[str]:
        """Get key factors contributing to fit prediction"""
        factors = []
        
        if features[0] > 0.7:  # Skill match
            factors.append('strong_skill_match')
        if features[1] > 0.8:  # Experience match
            factors.append('experience_match')
        if features[3] > 0.5:  # Domain match
            factors.append('domain_alignment')
        if features[6] > 0.8:  # Salary match
            factors.append('salary_alignment')
        
        return factors if factors else ['general_fit']
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.model or not self.is_trained:
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_names = [
                    'skill_match', 'experience_match', 'education_match', 'domain_match',
                    'skill_count_ratio', 'experience_ratio', 'salary_match',
                    'behavioral_score', 'market_demand', 'quality_score'
                ]
                return dict(zip(feature_names, importance))
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model"""
        if not self.model or not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'is_trained': self.is_trained,
                    'calibrated': self.calibrated
                }, f)
            logger.info(f"Saved job fit predictor to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, path: Optional[str] = None):
        """Load trained model"""
        path = path or self.model_path
        
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.is_trained = data.get('is_trained', False)
                self.calibrated = data.get('calibrated', False)
            
            logger.info(f"Loaded job fit predictor from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


# Global instance
_job_fit_predictor = None


def get_job_fit_predictor(model_path: Optional[str] = None) -> JobFitPredictor:
    """Get or create global job fit predictor instance"""
    global _job_fit_predictor
    if _job_fit_predictor is None:
        _job_fit_predictor = JobFitPredictor(model_path=model_path)
    return _job_fit_predictor

