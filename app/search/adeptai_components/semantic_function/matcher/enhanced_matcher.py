import os
import joblib
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from linkedin_api import Linkedin
import xgboost as xgb
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
import numpy as np
import pandas as pd

from matcher.base_matcher import TalentMatcher
from matcher.models import CandidateProfile
from matcher.llm_service import LLMService
from matcher.utils import dummy_parser

class EnhancedTalentMatcher(TalentMatcher):
    def __init__(self, openai_api_key, linkedin_username, linkedin_password):
        super().__init__(openai_api_key)
        self.llm = LLMService(openai_api_key)
        self.linkedin_api = Linkedin(linkedin_username, linkedin_password)
        self.setup_selenium()
        self.learning_data = []
        self.model_path = "model/xgboost_matchmaking_model.pkl"
        self.ensemble_model_path = "model/ensemble_matchmaking_model.pkl"
        self.use_ensemble = True  # Flag to switch between XGBoost and ensemble
        self.stability_threshold = 0.05  # CV standard deviation threshold
        self.load_or_create_model()

    def setup_selenium(self):
        options = Options()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)

    def load_or_create_model(self):
        """Load existing model or create a new one"""
        if self.use_ensemble and os.path.exists(self.ensemble_model_path):
            try:
                self.learning_model = joblib.load(self.ensemble_model_path)
                print("Loaded existing ensemble model")
                return
            except Exception as e:
                print(f"Error loading ensemble model: {e}")
        
        if os.path.exists(self.model_path):
            try:
                self.learning_model = joblib.load(self.model_path)
                print("Loaded existing XGBoost model")
            except Exception as e:
                print(f"Error loading model: {e}, creating new one")
                self.create_default_model()
        else:
            self.create_default_model()

    def create_default_model(self):
        """Create a default model with improved stability"""
        if self.use_ensemble:
            # Create ensemble with Random Forest for stability
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,          # Reduced from 6 for stability
                learning_rate=0.05,   # Reduced from 0.1 for stability
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,        # L1 regularization
                reg_lambda=0.1,       # L2 regularization
                min_child_weight=3,   # Prevent overfitting
                random_state=42,
                objective='reg:squarederror',
                eval_metric='rmse'
            )
            
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            
            # Ensemble with higher weight on Random Forest for stability
            self.learning_model = VotingRegressor([
                ('xgb', xgb_model),
                ('rf', rf_model)
            ], weights=[0.4, 0.6])  # Favor RF for stability
            
            model_path = self.ensemble_model_path
            print("Created ensemble model (XGBoost + Random Forest)")
        else:
            # Improved XGBoost with regularization
            self.learning_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,          # Reduced for stability
                learning_rate=0.05,   # Reduced for stability
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,        # L1 regularization
                reg_lambda=0.1,       # L2 regularization
                min_child_weight=3,   # Prevent overfitting
                random_state=42,
                objective='reg:squarederror',
                eval_metric='rmse'
            )
            model_path = self.model_path
            print("Created stabilized XGBoost model")
        
        # Create dummy training data to initialize the model
        dummy_X = np.random.rand(10, 13)  # Match feature count
        dummy_y = np.random.rand(10)
        self.learning_model.fit(dummy_X, dummy_y)
        
        # Save the initialized model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.learning_model, model_path)

    def extract_job_requirements(self, job_description):
        raw = self.llm(job_description)
        return dummy_parser(raw)

    def validate_linkedin_profile(self, url): 
        return {}
    
    def validate_nursing_license(self, license_number, state): 
        return {}
    
    def _calculate_location_match(self, required, actual): 
        return 1.0 if required in actual else 0.0
    
    def _validate_mandatory_skills(self, required, resume): 
        return sum(skill in resume for skill in required) / len(required)

    def _create_feature_vector(self, job, candidate):
        """Create feature vector with improved feature engineering"""
        r = self.extract_job_requirements(job)
        
        # Enhanced feature engineering for better stability
        features = [
            min(len(job), 10000),                                   # job_description_length (capped)
            min(len(candidate.resume), 10000),                      # resume_length (capped)
            len(r["mandatory_skills"]),                            # num_mandatory_skills
            len(r.get("preferred_skills", [])),                    # num_preferred_skills
            1 if candidate.linkedin_url else 0,                   # has_linkedin
            1 if candidate.license_number else 0,                 # has_license
            self._calculate_location_match(r["location"], candidate.current_location),  # location_match
            min(r.get("years_experience", 0), 50),                 # required_experience (capped)
            len(r.get("education_requirements", [])),              # num_education_req
            min(len(candidate.resume.split()), 5000),              # resume_word_count (capped)
        ]
        
        # Additional stabilized features
        skills_in_resume = sum(1 for skill in r["mandatory_skills"] if skill.lower() in candidate.resume.lower())
        features.extend([
            skills_in_resume,                                      # mandatory_skills_found
            skills_in_resume / max(len(r["mandatory_skills"]), 1), # skill_match_ratio
            min(len(set(candidate.resume.lower().split()) & 
                   set(' '.join(r["mandatory_skills"]).lower().split())), 100),  # keyword_overlap (capped)
        ])
        
        return features

    def _evaluate_model_stability(self, X, y):
        """Evaluate model stability using cross-validation"""
        if len(X) < 30:  # Need sufficient data for stability evaluation
            return True, 0.0
            
        try:
            # Use RepeatedKFold for better stability assessment
            cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
            
            if self.use_ensemble:
                model = self.learning_model
            else:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    min_child_weight=3,
                    random_state=42
                )
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            cv_std = scores.std()
            
            print(f"Cross-validation R² scores: {scores.mean():.4f} ± {cv_std:.4f}")
            
            # Check if model is stable enough
            is_stable = cv_std < self.stability_threshold
            return is_stable, cv_std
            
        except Exception as e:
            print(f"Error in stability evaluation: {e}")
            return True, 0.0

    def _retrain_model(self):
        """Retrain model with stability improvements"""
        if len(self.learning_data) < 20:  # Need more data for stable training
            return
            
        print(f"Retraining model with {len(self.learning_data)} samples")
        
        try:
            X, y = zip(*self.learning_data)
            X = np.array(X)
            y = np.array(y)
            
            # Evaluate current model stability
            is_stable, cv_std = self._evaluate_model_stability(X, y)
            
            if not is_stable:
                print(f"Model instability detected (CV std: {cv_std:.4f}), switching to ensemble")
                self.use_ensemble = True
            
            # Split data for training and validation
            if len(X) > 40:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train = X, y
                X_test, y_test = X, y
            
            # Create appropriate model based on stability
            if self.use_ensemble:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    min_child_weight=3,
                    random_state=42,
                    objective='reg:squarederror',
                    eval_metric='rmse'
                )
                
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                )
                
                self.learning_model = VotingRegressor([
                    ('xgb', xgb_model),
                    ('rf', rf_model)
                ], weights=[0.4, 0.6])
                
                model_path = self.ensemble_model_path
            else:
                self.learning_model = xgb.XGBRegressor(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    min_child_weight=3,
                    random_state=42,
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    early_stopping_rounds=15
                )
                model_path = self.model_path
            
            # Fit the model
            if hasattr(self.learning_model, 'fit'):
                if self.use_ensemble:
                    self.learning_model.fit(X_train, y_train)
                else:
                    self.learning_model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        verbose=False
                    )
            
            # Evaluate model performance
            y_pred = self.learning_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Model retrained - MSE: {mse:.4f}, R²: {r2:.4f}")
            print(f"Using {'ensemble' if self.use_ensemble else 'XGBoost'} model")
            
            # Save the updated model
            joblib.dump(self.learning_model, model_path)
            
            # Feature importance analysis
            self._analyze_feature_importance()
                    
        except Exception as e:
            print(f"Error during model retraining: {e}")

    def _analyze_feature_importance(self):
        """Analyze feature importance for the model"""
        try:
            feature_names = [
                'job_desc_length', 'resume_length', 'num_mandatory_skills', 
                'num_preferred_skills', 'has_linkedin', 'has_license',
                'location_match', 'required_experience', 'num_education_req',
                'resume_word_count', 'mandatory_skills_found', 'skill_match_ratio',
                'keyword_overlap'
            ]
            
            if self.use_ensemble:
                # Get importance from XGBoost component of ensemble
                if hasattr(self.learning_model.named_estimators_['xgb'], 'feature_importances_'):
                    importances = self.learning_model.named_estimators_['xgb'].feature_importances_
                else:
                    return
            else:
                if hasattr(self.learning_model, 'feature_importances_'):
                    importances = self.learning_model.feature_importances_
                else:
                    return
            
            feature_importance = dict(zip(feature_names, importances))
            print("Top 5 Feature Importances:")
            for feature, importance in sorted(feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {feature}: {importance:.4f}")
                
        except Exception as e:
            print(f"Error analyzing feature importance: {e}")

    def _get_learning_adjusted_score(self, job, candidate):
        """Get prediction with stability improvements"""
        try:
            vec = self._create_feature_vector(job, candidate)
            
            # Ensure we have the right number of features
            if len(vec) != 13:
                print(f"Warning: Feature vector has {len(vec)} features, expected 13")
                if len(vec) < 13:
                    vec.extend([0] * (13 - len(vec)))
                else:
                    vec = vec[:13]
            
            prediction = self.learning_model.predict([vec])[0]
            
            # Apply more conservative scaling for stability
            prediction = np.clip(prediction, 0, 1)  # Ensure bounds
            
            # Add slight smoothing to reduce variance
            if hasattr(self, '_last_predictions'):
                self._last_predictions.append(prediction)
                if len(self._last_predictions) > 5:
                    self._last_predictions.pop(0)
                # Use moving average for stability
                prediction = np.mean(self._last_predictions)
            else:
                self._last_predictions = [prediction]
            
            return prediction * 100
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return super().calculate_match_score(job, candidate.resume)["overall_match_percentage"]

    def calculate_enhanced_match_score(self, job, candidate):
        """Calculate enhanced match score with stability improvements"""
        req = self.extract_job_requirements(job)
        base = super().calculate_match_score(job, candidate.resume)
        
        score = {
            **base,
            "linkedin_verification": {},
            "license_verification": {},
            "location_match": self._calculate_location_match(req["location"], candidate.current_location),
            "mandatory_skills_match": self._validate_mandatory_skills(req["mandatory_skills"], candidate.resume),
        }
        
        # Add learning data for continuous improvement
        self._add_learning_data(job, candidate, score)
        
        # Get model prediction
        score["learning_adjusted_score"] = self._get_learning_adjusted_score(job, candidate)
        score["model_type"] = "ensemble" if self.use_ensemble else "xgboost"
        
        return score

    def _add_learning_data(self, job, candidate, score):
        """Add training example to learning dataset"""
        try:
            vec = self._create_feature_vector(job, candidate)
            target = score["overall_match_percentage"] / 100
            self.learning_data.append((vec, target))
            
            # Retrain model less frequently for stability
            if len(self.learning_data) % 25 == 0:  # Retrain every 25 samples
                self._retrain_model()
                
        except Exception as e:
            print(f"Error adding learning data: {e}")

    def find_top_candidates(self, job, candidates, top_n=10):
        """Find top candidates using stable scoring"""
        scores = []
        
        for c in candidates:
            try:
                score_data = self.calculate_enhanced_match_score(job, c)
                scores.append({"candidate": c, "score": score_data})
            except Exception as e:
                print(f"Error scoring candidate: {e}")
                continue
        
        # Sort by learning-adjusted score
        ranked_scores = sorted(scores, 
                             key=lambda x: x["score"]["learning_adjusted_score"], 
                             reverse=True)
        
        return ranked_scores[:top_n]

    def get_model_info(self):
        """Get information about the current model"""
        try:
            info = {
                "model_type": "Ensemble (XGBoost + Random Forest)" if self.use_ensemble else "XGBoost Regressor",
                "training_samples": len(self.learning_data),
                "model_path": self.ensemble_model_path if self.use_ensemble else self.model_path,
                "stability_threshold": self.stability_threshold,
                "use_ensemble": self.use_ensemble
            }
            
            if not self.use_ensemble and hasattr(self.learning_model, 'get_params'):
                params = self.learning_model.get_params()
                info["hyperparameters"] = {
                    "n_estimators": params.get("n_estimators", "N/A"),
                    "max_depth": params.get("max_depth", "N/A"),
                    "learning_rate": params.get("learning_rate", "N/A"),
                    "reg_alpha": params.get("reg_alpha", "N/A"),
                    "reg_lambda": params.get("reg_lambda", "N/A"),
                }
                
            return info
        except Exception as e:
            return {"error": str(e)}

    def switch_to_ensemble(self):
        """Manually switch to ensemble model for better stability"""
        self.use_ensemble = True
        print("Switched to ensemble mode for improved stability")
        if len(self.learning_data) > 20:
            self._retrain_model()

    def switch_to_xgboost(self):
        """Switch back to pure XGBoost (if stability is acceptable)"""
        self.use_ensemble = False
        print("Switched to XGBoost mode")
        if len(self.learning_data) > 20:
            self._retrain_model()

    def export_training_data(self, filepath="training_data.csv"):
        """Export training data to CSV for analysis"""
        if not self.learning_data:
            print("No training data available")
            return
            
        try:
            X, y = zip(*self.learning_data)
            feature_names = [
                'job_desc_length', 'resume_length', 'num_mandatory_skills', 
                'num_preferred_skills', 'has_linkedin', 'has_license',
                'location_match', 'required_experience', 'num_education_req',
                'resume_word_count', 'mandatory_skills_found', 'skill_match_ratio',
                'keyword_overlap'
            ]
            
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = y
            df.to_csv(filepath, index=False)
            print(f"Training data exported to {filepath}")
            
        except Exception as e:
            print(f"Error exporting training data: {e}")

    def __del__(self):
        """Cleanup selenium driver"""
        if hasattr(self, 'driver'):
            try:
                self.driver.quit()
            except:
                pass