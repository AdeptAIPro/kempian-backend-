"""
ML-Based Domain Classifier

Replaces rule-based domain classification with machine learning models.
Uses ensemble approach: RandomForest + optional BERT/transformer models.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")

# Try to import transformers for BERT
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. BERT features will be disabled.")

# Domain labels
DOMAIN_LABELS = ['technology', 'healthcare', 'finance', 'education', 'marketing', 'unknown']
DOMAIN_LABEL_MAP = {label: idx for idx, label in enumerate(DOMAIN_LABELS)}


class MLDomainClassifier:
    """
    Machine Learning-based Domain Classifier
    
    Uses ensemble approach:
    1. RandomForest with TF-IDF features (primary)
    2. Optional BERT/transformer model (if available)
    3. Fallback to rule-based if models not trained
    """
    
    def __init__(self, model_path: Optional[str] = None, use_bert: bool = False):
        """
        Initialize ML Domain Classifier
        
        Args:
            model_path: Path to saved model file (optional)
            use_bert: Whether to use BERT model (requires transformers library)
        """
        self.model_path = model_path or os.path.join("model", "domain_classifier.pkl")
        self.use_bert = use_bert and TRANSFORMERS_AVAILABLE
        self.is_trained = False
        
        # Initialize RandomForest classifier
        if SKLEARN_AVAILABLE:
            self.rf_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(DOMAIN_LABELS)
        else:
            self.rf_classifier = None
            self.vectorizer = None
            self.label_encoder = None
        
        # Initialize BERT model (optional)
        self.bert_model = None
        self.bert_tokenizer = None
        self.bert_pipeline = None
        
        if self.use_bert:
            try:
                # Use a lightweight BERT model for domain classification
                model_name = "distilbert-base-uncased"  # Faster than full BERT
                self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Note: For production, you'd fine-tune this on domain data
                # For now, we'll use it as a feature extractor
                logger.info("BERT model initialized (using as feature extractor)")
            except Exception as e:
                logger.warning(f"Failed to initialize BERT: {e}")
                self.use_bert = False
        
        # Fallback rule-based classifier
        self._init_fallback_classifier()
        
        # Load saved model if exists
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _init_fallback_classifier(self):
        """Initialize fallback rule-based classifier"""
        self.fallback_patterns = {
            'technology': [
                r'\b(?:python|java|javascript|react|node|aws|docker|kubernetes|machine learning|ai|ml|data science|backend|frontend|full.?stack|devops|cloud|azure|gcp|sql|database|api|microservices|agile|scrum)\b',
                r'\b(?:software|developer|engineer|programmer|coder|architect|tech|it|computer|programming|coding|development)\b'
            ],
            'healthcare': [
                r'\b(?:nurse|doctor|physician|medical|healthcare|hospital|clinic|patient|care|rn|md|phd|health|medicine|clinical|therapist|pharmacist|dentist|veterinary)\b',
                r'\b(?:icu|emergency|surgery|pediatric|cardiology|oncology|neurology|psychiatry|radiology|anesthesia)\b'
            ],
            'finance': [
                r'\b(?:finance|banking|investment|accounting|audit|tax|financial|analyst|advisor|consultant|trading|portfolio|risk|compliance|fintech)\b',
                r'\b(?:cpa|cfa|cma|frm|prm|actuary|underwriter|broker|trader|banker|accountant)\b'
            ],
            'education': [
                r'\b(?:teacher|professor|instructor|educator|academic|university|college|school|education|curriculum|pedagogy|research|phd|masters|bachelor)\b',
                r'\b(?:student|learning|teaching|training|development|coaching|mentoring|tutoring)\b'
            ],
            'marketing': [
                r'\b(?:marketing|advertising|brand|digital|social media|content|seo|sem|ppc|campaign|strategy|analytics|growth|sales|business development)\b',
                r'\b(?:marketer|advertiser|brand manager|content creator|social media manager|growth hacker|sales representative)\b'
            ]
        }
        
        # Compile patterns
        import re
        self.compiled_patterns = {}
        for domain, patterns in self.fallback_patterns.items():
            self.compiled_patterns[domain] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract features from text for ML model
        
        Args:
            text: Input text
            
        Returns:
            Feature vector
        """
        if not text:
            text = ""
        
        # TF-IDF features (primary)
        if self.vectorizer and self.is_trained:
            try:
                tfidf_features = self.vectorizer.transform([text]).toarray()[0]
            except Exception:
                tfidf_features = np.zeros(self.vectorizer.max_features if hasattr(self.vectorizer, 'max_features') else 5000)
        else:
            # Placeholder features if not trained
            tfidf_features = np.zeros(5000)
        
        # BERT features (optional, if available)
        bert_features = np.array([])
        if self.use_bert and self.bert_tokenizer:
            try:
                # Use BERT as feature extractor (simple approach)
                # In production, you'd fine-tune BERT on domain data
                inputs = self.bert_tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                # For now, use tokenizer output as simple features
                # In production, you'd use the model's hidden states
                bert_features = np.array([len(inputs['input_ids'][0])] * 10)  # Placeholder
            except Exception as e:
                logger.debug(f"BERT feature extraction failed: {e}")
                bert_features = np.zeros(10)
        
        # Combine features
        if len(bert_features) > 0:
            features = np.concatenate([tfidf_features, bert_features])
        else:
            features = tfidf_features
        
        return features
    
    def train(
        self,
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        validation_split: Optional[float] = None
    ):
        """
        Train the domain classifier
        
        Args:
            texts: List of text samples
            labels: List of domain labels (technology, healthcare, finance, etc.)
            test_size: Fraction of data to use for testing
            validation_split: Optional validation split
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available. Cannot train model.")
            return
        
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        
        if len(texts) < 10:
            logger.warning("Training data is very small. Model may not generalize well.")
        
        logger.info(f"Training domain classifier on {len(texts)} examples...")
        
        # Encode labels
        try:
            encoded_labels = self.label_encoder.transform(labels)
        except ValueError:
            # Fit label encoder if not already fitted
            self.label_encoder.fit(list(set(labels)))
            encoded_labels = self.label_encoder.transform(labels)
        
        # Vectorize texts
        logger.info("Vectorizing texts...")
        X = self.vectorizer.fit_transform(texts)
        
        # Split data
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
            )
        else:
            X_train, y_train = X, encoded_labels
            X_test, y_test = None, None
        
        # Train RandomForest
        logger.info("Training RandomForest classifier...")
        self.rf_classifier.fit(X_train, y_train)
        
        # Evaluate
        if X_test is not None:
            train_score = self.rf_classifier.score(X_train, y_train)
            test_score = self.rf_classifier.score(X_test, y_test)
            logger.info(f"Training accuracy: {train_score:.4f}")
            logger.info(f"Test accuracy: {test_score:.4f}")
        
        self.is_trained = True
        logger.info("Domain classifier training completed!")
    
    def classify_domain(self, text: str) -> Tuple[str, float]:
        """
        Classify domain of input text
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (domain, confidence_score)
        """
        if not text:
            return 'unknown', 0.0
        
        # Use trained ML model if available
        if self.is_trained and self.rf_classifier:
            try:
                # Extract features
                features = self.extract_features(text)
                
                # Reshape for sklearn (handle both sparse and dense)
                if hasattr(features, 'toarray'):
                    features = features.toarray()
                features = features.reshape(1, -1)
                
                # Predict
                if hasattr(self.rf_classifier, 'predict_proba'):
                    probabilities = self.rf_classifier.predict_proba(features)[0]
                    predicted_idx = np.argmax(probabilities)
                    confidence = float(probabilities[predicted_idx])
                    
                    # Get domain label
                    domain = self.label_encoder.inverse_transform([predicted_idx])[0]
                    
                    return domain, confidence
                else:
                    # Fallback to prediction only
                    predicted_idx = self.rf_classifier.predict(features)[0]
                    domain = self.label_encoder.inverse_transform([predicted_idx])[0]
                    return domain, 0.8  # Default confidence
            except Exception as e:
                logger.warning(f"ML classification failed: {e}. Using fallback.")
                return self._classify_fallback(text)
        else:
            # Use fallback rule-based classifier
            return self._classify_fallback(text)
    
    def _classify_fallback(self, text: str) -> Tuple[str, float]:
        """Fallback rule-based classification"""
        if not text:
            return 'unknown', 0.0
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, compiled_patterns in self.compiled_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in compiled_patterns:
                pattern_matches = len(pattern.findall(text_lower))
                if pattern_matches > 0:
                    matches += pattern_matches
                    score += min(pattern_matches * 0.1, 0.5)
            
            if matches > 0:
                domain_scores[domain] = min(score, 1.0)
        
        if not domain_scores:
            return 'unknown', 0.0
        
        # Return domain with highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[best_domain]
        
        return best_domain, confidence
    
    def should_filter_candidate(
        self,
        candidate_domain: str,
        query_domain: str,
        candidate_confidence: float,
        query_confidence: float
    ) -> bool:
        """
        Determine if candidate should be filtered out based on domain mismatch
        
        Args:
            candidate_domain: Candidate's domain
            query_domain: Query's domain
            candidate_confidence: Confidence in candidate domain classification
            query_confidence: Confidence in query domain classification
            
        Returns:
            True if candidate should be filtered out
        """
        # If domains are unknown, don't filter
        if candidate_domain == 'unknown' or query_domain == 'unknown':
            return False
        
        # If domains match, don't filter
        if candidate_domain == query_domain:
            return False
        
        # Enhanced filtering logic
        if candidate_confidence > 0.8 and query_confidence > 0.8 and candidate_domain != query_domain:
            return True
        
        # Filter if candidate has very high confidence in a different domain
        if candidate_confidence > 0.9 and candidate_domain != query_domain:
            return True
        
        # Don't filter if both have low confidence
        if candidate_confidence < 0.5 and query_confidence < 0.5:
            return False
        
        return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or not self.rf_classifier:
            return {}
        
        try:
            importances = self.rf_classifier.feature_importances_
            
            # Get feature names from vectorizer
            if hasattr(self.vectorizer, 'get_feature_names_out'):
                feature_names = self.vectorizer.get_feature_names_out()
            elif hasattr(self.vectorizer, 'get_feature_names'):
                feature_names = self.vectorizer.get_feature_names()
            else:
                return {}
            
            # Create feature importance dictionary
            importance_dict = dict(zip(feature_names, importances))
            
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            return dict(sorted_importance[:50])  # Top 50 features
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model to file"""
        if not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'rf_classifier': self.rf_classifier,
                    'vectorizer': self.vectorizer,
                    'label_encoder': self.label_encoder,
                    'is_trained': self.is_trained,
                    'use_bert': self.use_bert
                }, f)
            logger.info(f"Saved domain classifier to {path}")
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
                self.rf_classifier = data['rf_classifier']
                self.vectorizer = data['vectorizer']
                self.label_encoder = data['label_encoder']
                self.is_trained = data.get('is_trained', False)
                self.use_bert = data.get('use_bert', False)
            
            logger.info(f"Loaded domain classifier from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


# Global instance
_ml_domain_classifier = None


def get_ml_domain_classifier(model_path: Optional[str] = None, use_bert: bool = False) -> MLDomainClassifier:
    """Get or create global ML domain classifier instance"""
    global _ml_domain_classifier
    if _ml_domain_classifier is None:
        _ml_domain_classifier = MLDomainClassifier(model_path=model_path, use_bert=use_bert)
    return _ml_domain_classifier

