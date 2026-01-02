# search_system.py - Optimized Search System
"""
High-performance search system with caching and optimized algorithms
"""

import os
import json
import re
import time
import hashlib
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Lazy imports for better startup performance
def _lazy_imports():
    """Lazy load heavy dependencies only when needed"""
    global boto3, np, nltk, faiss, pickle, SentenceTransformer, CrossEncoder, requests, openai
    
    try:
        import boto3
        import numpy as np
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import RegexpTokenizer
        import faiss
        import pickle
        from sentence_transformers import SentenceTransformer, CrossEncoder
        import requests
        import openai
        return True
    except ImportError as e:
        print(f"Warning: Some dependencies not available: {e}")
        return False

# Import custom LLM components (Note: CustomExplanationGenerator replaced by ExplainableRecruitmentAI)
try:
    from .custom_llm_models import (
        CustomEmbeddingModel, CustomCrossEncoder,
        CustomQueryEnhancer, CustomDomainClassifier, CustomTokenizer,
        AdvancedEmbeddingModel, NeuralCrossEncoder, QueryIntentClassifier,
        ContextualQueryExpander, FeedbackLearner,
        ADVANCED_EMBEDDINGS_AVAILABLE, NEURAL_CROSS_ENCODER_AVAILABLE
    )
    CUSTOM_LLM_AVAILABLE = True
    print("✅ Custom LLM components loaded successfully")
except ImportError as e:
    print(f"⚠️ Custom LLM components not available: {e}")
    CUSTOM_LLM_AVAILABLE = False

# Import Learning-to-Rank model
try:
    try:
        from .learning_to_rank import LearningToRankModel, get_ltr_model
    except ImportError:
        # Try absolute import if relative import fails
        from learning_to_rank import LearningToRankModel, get_ltr_model
    LTR_AVAILABLE = True
    print("✅ Learning-to-Rank model available")
except ImportError as e:
    print(f"⚠️ Learning-to-Rank model not available: {e}")
    LTR_AVAILABLE = False
    LearningToRankModel = None
    get_ltr_model = None

# Import ML Domain Classifier
try:
    try:
        from .ml_domain_classifier import MLDomainClassifier, get_ml_domain_classifier
    except ImportError:
        # Try absolute import if relative import fails
        from ml_domain_classifier import MLDomainClassifier, get_ml_domain_classifier
    ML_DOMAIN_CLASSIFIER_AVAILABLE = True
    print("✅ ML Domain Classifier available")
except ImportError as e:
    print(f"⚠️ ML Domain Classifier not available: {e}")
    ML_DOMAIN_CLASSIFIER_AVAILABLE = False
    MLDomainClassifier = None
    get_ml_domain_classifier = None

# Import LLM Query Enhancer
try:
    try:
        from .llm_query_enhancer import LLMQueryEnhancer, get_llm_query_enhancer
    except ImportError:
        # Try absolute import if relative import fails
        from llm_query_enhancer import LLMQueryEnhancer, get_llm_query_enhancer
    LLM_QUERY_ENHANCER_AVAILABLE = True
    print("✅ LLM Query Enhancer available")
except ImportError as e:
    print(f"⚠️ LLM Query Enhancer not available: {e}")
    LLM_QUERY_ENHANCER_AVAILABLE = False
    LLMQueryEnhancer = None
    get_llm_query_enhancer = None

# Import RL Ranking Agent
try:
    try:
        from .rl_ranking_agent import RLRankingAgent, get_rl_ranking_agent
    except ImportError:
        # Try absolute import if relative import fails
        from rl_ranking_agent import RLRankingAgent, get_rl_ranking_agent
    RL_RANKING_AGENT_AVAILABLE = True
    print("✅ RL Ranking Agent available")
except ImportError as e:
    print(f"⚠️ RL Ranking Agent not available: {e}")
    RL_RANKING_AGENT_AVAILABLE = False
    RLRankingAgent = None
    get_rl_ranking_agent = None

# Import additional enhancements
try:
    try:
        from .job_fit_predictor import JobFitPredictor, get_job_fit_predictor
        from .ner_skill_extractor import SkillExtractorNER, get_skill_extractor
        from .multi_armed_bandit import RankingStrategyBandit, get_bandit
        from .skill_demand_forecaster import SkillDemandForecaster, get_forecaster
        from .candidate_clustering import CandidateSegmenter, get_segmenter
    except ImportError:
        from job_fit_predictor import JobFitPredictor, get_job_fit_predictor
        from ner_skill_extractor import SkillExtractorNER, get_skill_extractor
        from multi_armed_bandit import RankingStrategyBandit, get_bandit
        from skill_demand_forecaster import SkillDemandForecaster, get_forecaster
        from candidate_clustering import CandidateSegmenter, get_segmenter
    ENHANCEMENTS_AVAILABLE = True
    print("✅ Additional ML enhancements available")
except ImportError as e:
    print(f"⚠️ Additional enhancements not available: {e}")
    ENHANCEMENTS_AVAILABLE = False
    JobFitPredictor = None
    SkillExtractorNER = None
    RankingStrategyBandit = None
    SkillDemandForecaster = None
    CandidateSegmenter = None

# Import Behavioural Analysis
try:
    try:
        from .behavioural_analysis import get_pipeline
        from .behavioural_analysis.pipeline import MultiSourceProfile, create_multi_source_profile
    except ImportError:
        from behavioural_analysis import get_pipeline
        from behavioural_analysis.pipeline import MultiSourceProfile, create_multi_source_profile
    BEHAVIOURAL_ANALYSIS_AVAILABLE = True
    print("✅ Behavioural Analysis available")
except ImportError as e:
    print(f"⚠️ Behavioural Analysis not available: {e}")
    BEHAVIOURAL_ANALYSIS_AVAILABLE = False
    get_pipeline = None
    MultiSourceProfile = None
    create_multi_source_profile = None

# Import Bias Prevention
try:
    try:
        from .bias_prevention.sanitizer import QuerySanitizer, ResumeSanitizer
        from .bias_prevention.monitor import BiasMonitor
        from .bias_prevention.config import PROTECTED_CHARACTERISTICS, MONITORING_CONFIG, ALERT_CONFIG
    except ImportError:
        from bias_prevention.sanitizer import QuerySanitizer, ResumeSanitizer
        from bias_prevention.monitor import BiasMonitor
        from bias_prevention.config import PROTECTED_CHARACTERISTICS, MONITORING_CONFIG, ALERT_CONFIG
    BIAS_PREVENTION_AVAILABLE = True
    print("✅ Bias Prevention available")
except ImportError as e:
    print(f"⚠️ Bias Prevention not available: {e}")
    BIAS_PREVENTION_AVAILABLE = False
    QuerySanitizer = None
    ResumeSanitizer = None
    BiasMonitor = None
    PROTECTED_CHARACTERISTICS = {}
    MONITORING_CONFIG = {}
    ALERT_CONFIG = {}

# Import Explainable AI
try:
    try:
        from .explainable_ai import ExplainableRecruitmentAI
        from .explainable_ai.models.dataclasses import DecisionExplanation, FeatureContribution
    except ImportError:
        from explainable_ai import ExplainableRecruitmentAI
        from explainable_ai.models.dataclasses import DecisionExplanation, FeatureContribution
    EXPLAINABLE_AI_AVAILABLE = True
    print("✅ Explainable AI available")
except ImportError as e:
    print(f"⚠️ Explainable AI not available: {e}")
    EXPLAINABLE_AI_AVAILABLE = False
    ExplainableRecruitmentAI = None
    DecisionExplanation = None
    FeatureContribution = None

# Import Market Intelligence
try:
    try:
        from .market_intelligence import MarketIntelligenceAPI
        from .market_intelligence.talent_competition import analyze_talent_availability, competitive_intelligence
        from .market_intelligence.skills_forecasting import SkillsForecaster
        from .market_intelligence.salary_intelligence import CompensationBenchmarker
    except ImportError:
        from market_intelligence import MarketIntelligenceAPI
        from market_intelligence.talent_competition import analyze_talent_availability, competitive_intelligence
        from market_intelligence.skills_forecasting import SkillsForecaster
        from market_intelligence.salary_intelligence import CompensationBenchmarker
    MARKET_INTELLIGENCE_AVAILABLE = True
    print("✅ Market Intelligence available")
except ImportError as e:
    print(f"⚠️ Market Intelligence not available: {e}")
    MARKET_INTELLIGENCE_AVAILABLE = False
    MarketIntelligenceAPI = None
    analyze_talent_availability = None
    competitive_intelligence = None
    SkillsForecaster = None
    CompensationBenchmarker = None

# Import Instant Search
try:
    try:
        from .search.instant_search import InstantSearchEngine
    except ImportError:
        from search.instant_search import InstantSearchEngine
    INSTANT_SEARCH_AVAILABLE = True
    print("✅ Instant Search available")
except ImportError as e:
    print(f"⚠️ Instant Search not available: {e}")
    INSTANT_SEARCH_AVAILABLE = False
    InstantSearchEngine = None

# Import Dense Retrieval Matcher
try:
    try:
        from .enhanced_models.dense_retrieval import DenseRetrievalMatcher, ProductionMatcher
    except ImportError:
        from enhanced_models.dense_retrieval import DenseRetrievalMatcher, ProductionMatcher
    DENSE_RETRIEVAL_AVAILABLE = True
    print("✅ Dense Retrieval Matcher available")
except ImportError as e:
    print(f"⚠️ Dense Retrieval Matcher not available: {e}")
    DENSE_RETRIEVAL_AVAILABLE = False
    DenseRetrievalMatcher = None
    ProductionMatcher = None

# Import Enhanced Matcher
try:
    try:
        from .semantic_function.matcher.enhanced_matcher import EnhancedTalentMatcher
    except ImportError:
        from semantic_function.matcher.enhanced_matcher import EnhancedTalentMatcher
    ENHANCED_MATCHER_AVAILABLE = True
    print("✅ Enhanced Matcher available")
except ImportError as e:
    print(f"⚠️ Enhanced Matcher not available: {e}")
    ENHANCED_MATCHER_AVAILABLE = False
    EnhancedTalentMatcher = None

# Import Multi-Model Embedding Service
try:
    try:
        from .utils.enhanced_embeddings import MultiModelEmbeddingService, embedding_service
    except ImportError:
        from utils.enhanced_embeddings import MultiModelEmbeddingService, embedding_service
    MULTI_MODEL_EMBEDDING_AVAILABLE = True
    print("✅ Multi-Model Embedding Service available")
except ImportError as e:
    print(f"⚠️ Multi-Model Embedding Service not available: {e}")
    MULTI_MODEL_EMBEDDING_AVAILABLE = False
    MultiModelEmbeddingService = None
    embedding_service = None

# Import Optimized Cache
try:
    try:
        from .search.optimized_cache import OptimizedCandidateCache
    except ImportError:
        from search.optimized_cache import OptimizedCandidateCache
    OPTIMIZED_CACHE_AVAILABLE = True
    print("✅ Optimized Cache available")
except ImportError as e:
    print(f"⚠️ Optimized Cache not available: {e}")
    OPTIMIZED_CACHE_AVAILABLE = False
    OptimizedCandidateCache = None

# Initialize lazy imports
_dependencies_available = _lazy_imports()

# Load environment variables only if dependencies are available
if _dependencies_available:
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure API keys and settings
    if 'openai' in globals():
        openai.api_key = os.getenv('OPENAI_API_KEY')
    os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
    os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_REGION', 'ap-south-1')

# Database configuration
DATABASE_TABLE_NAME = os.getenv('DYNAMODB_TABLE_NAME', 'user-resume-metadata')

# Cache configuration
CACHE_DIR = "cache"
CACHE_FILE_ALL = os.path.join(CACHE_DIR, "candidates_cache_all.pkl")
CACHE_FILE_MEDICAL = os.path.join(CACHE_DIR, "candidates_cache_medical.pkl")
CACHE_FILE_IT = os.path.join(CACHE_DIR, "candidates_cache_it.pkl")
CACHE_EXPIRY_HOURS = 24  # Cache expires after 24 hours

# ===== MOCK CANDIDATE DATA =====

MOCK_CANDIDATES = [
    {
        'email': 'dr.sarah.mitchell@email.com',
        'full_name': 'Dr. Sarah Mitchell MD',
        'skills': ['Internal Medicine', 'Patient Care', 'Diagnosis', 'Treatment Planning', 'Medical Records'],
        'total_experience_years': 8,
        'resume_text': 'Board-certified Internal Medicine physician with 8 years of experience in hospital and clinic settings. Expert in patient care, diagnosis, and treatment planning.',
        'phone': '+1-555-0101',
        'sourceURL': 'https://example.com/sarah-mitchell'
    },
    {
        'email': 'mike.johnson@email.com',
        'full_name': 'Mike Johnson RN',
        'skills': ['Patient Care', 'ICU', 'Emergency Medicine', 'ACLS', 'BLS'],
        'total_experience_years': 7,
        'resume_text': 'Registered Nurse with 7 years experience in ICU and emergency medicine. ACLS and BLS certified with expertise in patient care.',
        'phone': '+1-555-0125',
        'sourceURL': 'https://example.com/mike-johnson'
    },
    {
        'email': 'dr.james.chen@email.com',
        'full_name': 'Dr. James Chen MD',
        'skills': ['Cardiology', 'Echocardiography', 'Patient Care', 'Medical Diagnosis', 'Treatment'],
        'total_experience_years': 12,
        'resume_text': 'Cardiologist with 12 years of experience in hospital settings. Specialized in echocardiography and cardiac patient care.',
        'phone': '+1-555-0102',
        'sourceURL': 'https://example.com/james-chen'
    },
    {
        'email': 'lisa.rodriguez@email.com',
        'full_name': 'Lisa Rodriguez RN',
        'skills': ['Pediatric Care', 'Nursing', 'Patient Assessment', 'Medication Administration', 'Family Education'],
        'total_experience_years': 5,
        'resume_text': 'Pediatric Registered Nurse with 5 years experience in children\'s hospital. Expert in pediatric patient care and family education.',
        'phone': '+1-555-0103',
        'sourceURL': 'https://example.com/lisa-rodriguez'
    },
    {
        'email': 'dr.michael.brown@email.com',
        'full_name': 'Dr. Michael Brown MD',
        'skills': ['Emergency Medicine', 'Trauma Care', 'Critical Care', 'Medical Procedures', 'Patient Stabilization'],
        'total_experience_years': 10,
        'resume_text': 'Emergency Medicine physician with 10 years experience in trauma centers. Expert in critical care and emergency medical procedures.',
        'phone': '+1-555-0104',
        'sourceURL': 'https://example.com/michael-brown'
    },
    {
        'email': 'jennifer.taylor@email.com',
        'full_name': 'Jennifer Taylor RN',
        'skills': ['Surgical Nursing', 'Operating Room', 'Patient Care', 'Surgical Procedures', 'Recovery Care'],
        'total_experience_years': 6,
        'resume_text': 'Surgical Registered Nurse with 6 years experience in operating room settings. Expert in surgical procedures and patient recovery care.',
        'phone': '+1-555-0105',
        'sourceURL': 'https://example.com/jennifer-taylor'
    }
]

def _is_cache_valid(cache_file: str) -> bool:
    """Check if cache file exists and is not expired"""
    if not os.path.exists(cache_file):
            return False
    try:
        cache_time = os.path.getmtime(cache_file)
        current_time = time.time()
        age_hours = (current_time - cache_time) / 3600
        return age_hours < CACHE_EXPIRY_HOURS
    except Exception:
            return False

def _load_from_cache(cache_file: str):
    """Load candidates from cache file"""
    try:
        if _dependencies_available and 'pickle' in globals():
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                print(f"✅ Loaded {len(cache_data.get('candidates', []))} candidates from {os.path.basename(cache_file)}")
                return cache_data.get('candidates', [])
    except Exception as e:
        print(f"⚠️ Error loading from cache {cache_file}: {e}")
    return None

def _save_to_cache(candidates, cache_file: str):
    """Save candidates to cache file"""
    try:
        if _dependencies_available and 'pickle' in globals():
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_data = {
                'candidates': candidates,
                'timestamp': time.time(),
                'count': len(candidates)
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Cached {len(candidates)} candidates to {os.path.basename(cache_file)}")
    except Exception as e:
        print(f"⚠️ Error saving to cache {cache_file}: {e}")

def _partition_by_domain(candidates: list) -> tuple:
    """Partition candidates into medical and IT groups with simple, fast keyword heuristics."""
    medical_keywords = {
        'nurse', 'doctor', 'physician', 'medical', 'healthcare', 'hospital', 'clinic', 'patient', 'rn', 'md',
        'icu', 'surgery', 'pediatric', 'cardiology', 'oncology', 'neurology', 'psychiatry', 'radiology', 'anesthesia'
    }
    it_keywords = {
        'python', 'java', 'javascript', 'react', 'node', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'sql',
        'database', 'api', 'microservices', 'devops', 'cloud', 'backend', 'frontend', 'full stack', 'engineer', 'developer'
    }

    medical, it = [], []
    for c in candidates:
        try:
            skills = c.get('skills', []) if isinstance(c, dict) else []
            skills_text = ' '.join(s for s in skills if isinstance(s, str)).lower()
            resume_text = (c.get('resume_text', '') or '').lower() if isinstance(c, dict) else ''
            blob = f"{skills_text} {resume_text}"
            if any(k in blob for k in medical_keywords):
                medical.append(c)
            if any(k in blob for k in it_keywords):
                it.append(c)
        except Exception:
            continue
    return medical, it

def get_candidates_with_fallback():
    """Get candidates from cache, DynamoDB, or fallback to mock data with optimized loading.
    Behavior:
    - Prefer precomputed ALL cache; if missing/stale, pull ALL candidates from DynamoDB with full pagination (no artificial Limit).
    - On first load, also precompute and persist MEDICAL and IT caches.
    """
    # Try ALL cache first
    if _is_cache_valid(CACHE_FILE_ALL):
        print("🔄 Loading ALL candidates from cache...")
        cached_all = _load_from_cache(CACHE_FILE_ALL)
        if cached_all:
            # Ensure domain caches exist too, compute if missing
            need_med = not _is_cache_valid(CACHE_FILE_MEDICAL)
            need_it = not _is_cache_valid(CACHE_FILE_IT)
            if (need_med or need_it) and cached_all:
                med, it = _partition_by_domain(cached_all)
                if need_med:
                    _save_to_cache(med, CACHE_FILE_MEDICAL)
                if need_it:
                    _save_to_cache(it, CACHE_FILE_IT)
            return cached_all
    
    try:
        # Try to connect to DynamoDB
        if _dependencies_available and 'boto3' in globals():
            print("🔄 Attempting to connect to DynamoDB...")
            
            # Get AWS credentials from environment
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_REGION', 'ap-south-1')
            table_name = os.getenv('DYNAMODB_TABLE_NAME', 'user-resume-metadata')
            
            if aws_access_key and aws_secret_key:
                # Create DynamoDB resource
                dynamodb = boto3.resource(
                    'dynamodb',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                
                table = dynamodb.Table(table_name)
                
                # Full-table scan with pagination (no artificial limits or filters)
                print(f"📊 Scanning DynamoDB table: {table_name} (full pagination, no rate limiting)")
                all_items = []
                scan_kwargs = {
                    'ProjectionExpression': 'email, full_name, skills, resume_text, total_experience_years, phone, sourceURL, #loc',
                    'ExpressionAttributeNames': {'#loc': 'location'}
                }
                
                start_time = time.time()
                batch_count = 0
                
                while True:
                    response = table.scan(**scan_kwargs)
                    items = response.get('Items', [])
                    all_items.extend(items)
                    batch_count += 1
                    
                    # Progress indicator with time estimation
                    elapsed = time.time() - start_time
                    if batch_count % 10 == 0 or 'LastEvaluatedKey' not in response:
                        rate = len(all_items) / elapsed if elapsed > 0 else 0
                        eta = (len(all_items) / rate) - elapsed if rate > 0 else 0
                        print(f"📊 Batch {batch_count}: {len(items)} items (total: {len(all_items)}) - Rate: {rate:.1f} items/sec - ETA: {eta:.1f}s")
                    
                    # Check if there are more items to scan
                    if 'LastEvaluatedKey' not in response:
                        break
                    
                    # Set the LastEvaluatedKey for the next scan
                    scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
                
                load_time = time.time() - start_time
                if all_items:
                    print(f"✅ Successfully loaded {len(all_items)} candidates from DynamoDB in {load_time:.2f}s")
                    # Save ALL cache
                    _save_to_cache(all_items, CACHE_FILE_ALL)
                    # Partition and save domain caches
                    med, it = _partition_by_domain(all_items)
                    _save_to_cache(med, CACHE_FILE_MEDICAL)
                    _save_to_cache(it, CACHE_FILE_IT)
                    return all_items
                else:
                    print("⚠️ No candidates found in DynamoDB, using mock data")
                    return MOCK_CANDIDATES
            else:
                print("⚠️ AWS credentials not found, using mock data")
                return MOCK_CANDIDATES
        else:
            print("⚠️ boto3 not available, using mock data")
            return MOCK_CANDIDATES
            
    except Exception as e:
        print(f"❌ Error connecting to DynamoDB: {e}")
        print("🔄 Falling back to mock data")
        return MOCK_CANDIDATES

class OptimizedDomainClassifier:
    """Enhanced domain classification with ML support"""
    
    def __init__(self):
        # Priority: ML classifier > Custom LLM classifier > Pattern-based fallback
        self.ml_classifier = None
        self.custom_classifier = None
        self.use_ml = False
        
        # Try ML classifier first (if available and trained)
        if ML_DOMAIN_CLASSIFIER_AVAILABLE and get_ml_domain_classifier:
            try:
                self.ml_classifier = get_ml_domain_classifier()
                if self.ml_classifier.is_trained:
                    self.use_ml = True
                    print("✅ Using ML Domain Classifier")
                else:
                    print("⚠️ ML Domain Classifier available but not trained. Using fallback.")
            except Exception as e:
                print(f"⚠️ Failed to initialize ML classifier: {e}")
        
        # Fallback to custom LLM classifier
        if not self.use_ml:
            if CUSTOM_LLM_AVAILABLE:
                try:
                    self.custom_classifier = CustomDomainClassifier()
                    print("✅ Using custom LLM domain classifier")
                except Exception as e:
                    print(f"⚠️ Failed to initialize custom classifier: {e}")
                    self.custom_classifier = None
            else:
                self.custom_classifier = None
        
        # Final fallback to pattern-based classifier
        if not self.use_ml and not self.custom_classifier:
            self._init_pattern_classifier()
            print("✅ Using pattern-based domain classifier")
    
    def _init_pattern_classifier(self):
        """Initialize pattern-based classifier as fallback"""
        # Pre-compiled patterns for faster matching
        self.patterns = {
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
        
        # Compile patterns for faster matching
        self.compiled_patterns = {}
        for domain, patterns in self.patterns.items():
            self.compiled_patterns[domain] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def classify_domain(self, text: str) -> Tuple[str, float]:
        """Classify domain with enhanced confidence scoring"""
        # Use ML classifier if available and trained
        if self.use_ml and self.ml_classifier:
            return self.ml_classifier.classify_domain(text)
        # Fallback to custom LLM classifier
        elif self.custom_classifier:
            return self.custom_classifier.classify_domain(text)
        # Final fallback to pattern-based
        else:
            return self._pattern_classify_domain(text)
    
    def _pattern_classify_domain(self, text: str) -> Tuple[str, float]:
        """Pattern-based domain classification (fallback)"""
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
                    score += min(pattern_matches * 0.1, 0.5)  # Cap at 0.5 per pattern
            
            # Normalize score
            if matches > 0:
                domain_scores[domain] = min(score, 1.0)
        
        if not domain_scores:
            return 'unknown', 0.0
        
        # Return domain with highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[best_domain]
        
        return best_domain, confidence
    
    def should_filter_candidate(self, candidate_domain: str, query_domain: str, 
                              candidate_confidence: float, query_confidence: float) -> bool:
        """Determine if candidate should be filtered out based on domain mismatch"""
        # Use ML classifier filtering if available
        if self.use_ml and self.ml_classifier:
            return self.ml_classifier.should_filter_candidate(
                candidate_domain, query_domain, candidate_confidence, query_confidence
            )
        # Fallback to custom LLM classifier
        elif self.custom_classifier:
            return self.custom_classifier.should_filter_candidate(
                candidate_domain, query_domain, candidate_confidence, query_confidence
            )
        else:
            # Original filtering logic
            if candidate_domain == 'unknown' or query_domain == 'unknown':
                return False
            if candidate_domain == query_domain:
                return False
            if candidate_confidence > 0.7 and query_confidence > 0.7 and candidate_domain != query_domain:
                return True
            if candidate_confidence > 0.9 and candidate_domain != query_domain:
                return True
            return False

class OptimizedSearchSystem:
    """High-performance search system with custom LLM integration"""
    
    def __init__(self, background_init=False, use_custom_llm=True):
        print("🚀 Initializing OptimizedSearchSystem with Custom LLM...")
        start_time = time.time()
        
        # Load candidates with progress tracking
        print("📊 Loading candidates...")
        self.candidates = get_candidates_with_fallback()
        
        # Initialize custom LLM components
        self.use_custom_llm = use_custom_llm and CUSTOM_LLM_AVAILABLE
        self._init_custom_llm_components()
        
        # Initialize Learning-to-Rank model
        self.ltr_model = None
        if LTR_AVAILABLE and get_ltr_model:
            try:
                self.ltr_model = get_ltr_model()
                print("✅ Learning-to-Rank model initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize LTR model: {e}")
                self.ltr_model = None
        
        # Initialize LLM Query Enhancer
        self.llm_query_enhancer = None
        self.use_llm_enhancement = False
        if LLM_QUERY_ENHANCER_AVAILABLE and get_llm_query_enhancer:
            try:
                # Use OpenAI by default, can be configured via environment
                provider = os.getenv('LLM_PROVIDER', 'openai')
                model = os.getenv('LLM_MODEL', 'gpt-4')
                self.llm_query_enhancer = get_llm_query_enhancer(provider=provider, model=model)
                if self.llm_query_enhancer.client:
                    self.use_llm_enhancement = True
                    print("✅ LLM Query Enhancer initialized")
                else:
                    print("⚠️ LLM Query Enhancer available but API key not configured")
            except Exception as e:
                print(f"⚠️ Failed to initialize LLM Query Enhancer: {e}")
                self.llm_query_enhancer = None
        
        # Initialize RL Ranking Agent
        self.rl_ranking_agent = None
        if RL_RANKING_AGENT_AVAILABLE and get_rl_ranking_agent:
            try:
                self.rl_ranking_agent = get_rl_ranking_agent()
                print("✅ RL Ranking Agent initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize RL Ranking Agent: {e}")
                self.rl_ranking_agent = None
        
        # Initialize additional enhancements
        if ENHANCEMENTS_AVAILABLE:
            try:
                self.job_fit_predictor = get_job_fit_predictor() if get_job_fit_predictor else None
                self.skill_extractor = get_skill_extractor() if get_skill_extractor else None
                self.strategy_bandit = get_bandit() if get_bandit else None
                self.demand_forecaster = get_forecaster() if get_forecaster else None
                self.candidate_segmenter = get_segmenter() if get_segmenter else None
            except Exception as e:
                print(f"⚠️ Failed to initialize some enhancements: {e}")
                self.job_fit_predictor = None
                self.skill_extractor = None
                self.strategy_bandit = None
                self.demand_forecaster = None
                self.candidate_segmenter = None
        
        # Initialize Behavioural Analysis Pipeline
        self.behavioral_pipeline = None
        self.use_behavioral_analysis = False
        if BEHAVIOURAL_ANALYSIS_AVAILABLE and get_pipeline:
            try:
                # Use lightweight config for performance (can be changed to 'production' for full features)
                config_name = os.getenv('BEHAVIOURAL_ANALYSIS_CONFIG', 'lightweight')
                self.behavioral_pipeline = get_pipeline(config_name)
                self.use_behavioral_analysis = True
                print(f"✅ Behavioural Analysis Pipeline initialized ({config_name})")
            except Exception as e:
                print(f"⚠️ Failed to initialize Behavioural Analysis: {e}")
                self.behavioral_pipeline = None
        
        # Initialize Bias Prevention Components
        self.query_sanitizer = None
        self.resume_sanitizer = None
        self.bias_monitor = None
        self.use_bias_prevention = False
        if BIAS_PREVENTION_AVAILABLE:
            try:
                self.query_sanitizer = QuerySanitizer() if QuerySanitizer else None
                self.resume_sanitizer = ResumeSanitizer() if ResumeSanitizer else None
                self.bias_monitor = BiasMonitor() if BiasMonitor else None
                self.use_bias_prevention = True
                print("✅ Bias Prevention components initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Bias Prevention: {e}")
                self.query_sanitizer = None
                self.resume_sanitizer = None
                self.bias_monitor = None
        
        # Store bias prevention configs
        self.protected_characteristics = PROTECTED_CHARACTERISTICS if BIAS_PREVENTION_AVAILABLE else {}
        self.monitoring_config = MONITORING_CONFIG if BIAS_PREVENTION_AVAILABLE else {}
        self.alert_config = ALERT_CONFIG if BIAS_PREVENTION_AVAILABLE else {}
        
        # Initialize Explainable AI
        self.explainable_ai = None
        self.use_explainable_ai = False
        if EXPLAINABLE_AI_AVAILABLE and ExplainableRecruitmentAI:
            try:
                self.explainable_ai = ExplainableRecruitmentAI()
                self.use_explainable_ai = True
                print("✅ Explainable AI initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Explainable AI: {e}")
                self.explainable_ai = None
        
        # Initialize Market Intelligence
        self.market_intelligence = None
        self.skills_forecaster = None
        self.compensation_benchmarker = None
        self.use_market_intelligence = False
        if MARKET_INTELLIGENCE_AVAILABLE:
            try:
                # Initialize main market intelligence API
                if MarketIntelligenceAPI:
                    self.market_intelligence = MarketIntelligenceAPI()
                    print("✅ Market Intelligence API initialized")
                
                # Initialize skills forecaster
                if SkillsForecaster:
                    self.skills_forecaster = SkillsForecaster()
                    print("✅ Skills Forecaster initialized")
                
                # Initialize compensation benchmarker
                if CompensationBenchmarker:
                    self.compensation_benchmarker = CompensationBenchmarker()
                    print("✅ Compensation Benchmarker initialized")
                
                self.use_market_intelligence = True
            except Exception as e:
                print(f"⚠️ Failed to initialize Market Intelligence: {e}")
                self.market_intelligence = None
                self.skills_forecaster = None
                self.compensation_benchmarker = None
        
        # Initialize Instant Search Engine
        self.instant_search_engine = None
        self.use_instant_search = False
        if INSTANT_SEARCH_AVAILABLE and InstantSearchEngine:
            try:
                self.instant_search_engine = InstantSearchEngine(max_candidates=100000)
                self.use_instant_search = True
                print("✅ Instant Search Engine initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Instant Search Engine: {e}")
                self.instant_search_engine = None
        
        # Initialize Dense Retrieval Matcher
        self.dense_retriever = None
        self.production_matcher = None
        self.use_dense_retrieval = False
        if DENSE_RETRIEVAL_AVAILABLE:
            try:
                # Try to load existing index first
                index_path = "indexes/dense_retrieval_index"
                if ProductionMatcher:
                    self.production_matcher = ProductionMatcher(index_path=index_path)
                    # If index doesn't exist, build it from candidates
                    if not self.production_matcher.dense_retriever.index and self.candidates:
                        print("📦 Building Dense Retrieval index from candidates...")
                        try:
                            self.production_matcher.index_candidates(self.candidates, save_path=index_path)
                            print(f"✅ Built Dense Retrieval index with {len(self.candidates)} candidates")
                        except Exception as e:
                            print(f"⚠️ Failed to build Dense Retrieval index: {e}")
                    self.use_dense_retrieval = True
                    print("✅ Production Matcher initialized")
                elif DenseRetrievalMatcher:
                    self.dense_retriever = DenseRetrievalMatcher()
                    # Build index if candidates are available
                    if self.candidates:
                        print("📦 Building Dense Retrieval index from candidates...")
                        try:
                            self.dense_retriever.build_index(self.candidates)
                            print(f"✅ Built Dense Retrieval index with {len(self.candidates)} candidates")
                        except Exception as e:
                            print(f"⚠️ Failed to build Dense Retrieval index: {e}")
                    self.use_dense_retrieval = True
                    print("✅ Dense Retrieval Matcher initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Dense Retrieval Matcher: {e}")
                self.dense_retriever = None
                self.production_matcher = None
        
        # Initialize Enhanced Matcher
        self.enhanced_matcher = None
        self.use_enhanced_matcher = False
        if ENHANCED_MATCHER_AVAILABLE and EnhancedTalentMatcher:
            try:
                # Initialize with API keys from environment
                openai_key = os.getenv('OPENAI_API_KEY', '')
                linkedin_user = os.getenv('LINKEDIN_USERNAME', '')
                linkedin_pass = os.getenv('LINKEDIN_PASSWORD', '')
                
                if openai_key:
                    self.enhanced_matcher = EnhancedTalentMatcher(
                        openai_api_key=openai_key,
                        linkedin_username=linkedin_user,
                        linkedin_password=linkedin_pass
                    )
                    self.use_enhanced_matcher = True
                    print("✅ Enhanced Matcher initialized")
                else:
                    print("⚠️ Enhanced Matcher available but OpenAI API key not configured")
            except Exception as e:
                print(f"⚠️ Failed to initialize Enhanced Matcher: {e}")
                self.enhanced_matcher = None
        
        # Initialize Multi-Model Embedding Service
        self.embedding_service = None
        self.use_multi_model_embeddings = False
        if MULTI_MODEL_EMBEDDING_AVAILABLE:
            try:
                if embedding_service:
                    self.embedding_service = embedding_service
                    self.use_multi_model_embeddings = True
                    print("✅ Multi-Model Embedding Service initialized")
                elif MultiModelEmbeddingService:
                    self.embedding_service = MultiModelEmbeddingService()
                    self.use_multi_model_embeddings = True
                    print("✅ Multi-Model Embedding Service initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Multi-Model Embedding Service: {e}")
                self.embedding_service = None
        
        # Initialize Optimized Cache
        self.optimized_cache = None
        self.use_optimized_cache = False
        if OPTIMIZED_CACHE_AVAILABLE and OptimizedCandidateCache:
            try:
                self.optimized_cache = OptimizedCandidateCache(max_candidates=50000)
                # Add candidates to optimized cache
                if self.candidates:
                    self.optimized_cache.add_candidates(self.candidates)
                    print(f"✅ Added {len(self.candidates)} candidates to optimized cache")
                self.use_optimized_cache = True
                print("✅ Optimized Cache initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Optimized Cache: {e}")
                self.optimized_cache = None
        
        # Lazy initialization of domain classifier
        self._domain_classifier = None
        
        # Initialize basic components first
        self._search_cache = {}
        self._cache_max_size = 500
        self._search_count = 0
        self._total_search_time = 0.0
        
        # Initialize processed candidates as empty initially
        self._processed_candidates = []
        self._preprocessing_complete = False
        
        # For very large datasets (>100k), always use background processing
        total_candidates = len(self.candidates)
        if total_candidates > 100000:
            print(f"📊 Large dataset detected ({total_candidates:,} candidates) - using background processing")
            background_init = True
        
        if background_init:
            # Start preprocessing in background thread
            import threading
            self._preprocessing_thread = threading.Thread(target=self._background_preprocess, daemon=True)
            self._preprocessing_thread.start()
            print("⚙️ Pre-processing candidates in background...")
        else:
            # Pre-process candidates immediately
            print("⚙️ Pre-processing candidates...")
            self._preprocess_candidates()
            self._preprocessing_complete = True
        
        init_time = time.time() - start_time
        print(f"✅ OptimizedSearchSystem initialized with {len(self.candidates)} candidates in {init_time:.2f}s")
    
    def _init_custom_llm_components(self):
        """Initialize custom LLM components"""
        if self.use_custom_llm:
            print("🧠 Initializing Custom LLM Components...")
            
            # Initialize custom components
            # Prefer advanced neural embeddings if available
            if ADVANCED_EMBEDDINGS_AVAILABLE:
                print("🧠 Using AdvancedEmbeddingModel")
                self.custom_embedding_model = AdvancedEmbeddingModel()
            else:
                self.custom_embedding_model = CustomEmbeddingModel()
            # Prefer neural cross-encoder if available
            if NEURAL_CROSS_ENCODER_AVAILABLE:
                print("🧠 Using NeuralCrossEncoder")
                self.custom_cross_encoder = NeuralCrossEncoder()
            else:
                self.custom_cross_encoder = CustomCrossEncoder()
            # Note: CustomExplanationGenerator replaced by ExplainableRecruitmentAI
            self.custom_query_enhancer = CustomQueryEnhancer()
            # Optional: intent and expansion
            try:
                self.intent_classifier = QueryIntentClassifier()
            except Exception:
                self.intent_classifier = None
            try:
                self.contextual_expander = ContextualQueryExpander()
            except Exception:
                self.contextual_expander = None
            # Feedback learner
            try:
                self.feedback_learner = FeedbackLearner(self.custom_embedding_model)
            except Exception:
                self.feedback_learner = None
            
            # Train embedding model on candidate data
            print("🔧 Training custom embedding model...")
            candidate_texts = []
            self._candidate_index_to_id = []
            for candidate in self.candidates:
                if isinstance(candidate, dict):
                    skills_text = ' '.join(candidate.get('skills', []))
                    resume_text = candidate.get('resume_text', '')
                    full_name = candidate.get('full_name', '')
                    candidate_texts.append(f"{skills_text} {resume_text} {full_name}")
                    self._candidate_index_to_id.append(candidate.get('email', ''))
            
            if candidate_texts:
                self.custom_embedding_model.fit(candidate_texts)
                # Build vector index for semantic retrieval
                print("📦 Building vector index for semantic retrieval...")
                emb_matrix = self.custom_embedding_model.batch_encode(candidate_texts)
                self.custom_embedding_model.build_index(emb_matrix)
                print("✅ Custom embedding model trained successfully")
            else:
                print("⚠️ No candidate texts available for training")
                self.use_custom_llm = False
        else:
            print("⚠️ Custom LLM components not available, using fallback methods")
            self.custom_embedding_model = None
            self.custom_cross_encoder = None
            # Note: CustomExplanationGenerator replaced by ExplainableRecruitmentAI
            self.custom_query_enhancer = None
    
    def _background_preprocess(self):
        """Background preprocessing function with better error handling"""
        try:
            print("🔄 Starting background preprocessing...")
            self._preprocess_candidates()
            self._preprocessing_complete = True
            print("✅ Background preprocessing completed successfully")
        except Exception as e:
            print(f"❌ Background preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            # Set to complete even on failure to prevent infinite waiting
            self._preprocessing_complete = True
    
    @property
    def domain_classifier(self):
        """Lazy initialization of domain classifier"""
        if self._domain_classifier is None:
            print("🔧 Initializing domain classifier...")
            self._domain_classifier = OptimizedDomainClassifier()
        return self._domain_classifier
    
    def _preprocess_candidates(self):
        """Pre-process candidates for faster searching with optimized performance"""
        total_candidates = len(self.candidates)
        if total_candidates == 0:
            self._processed_candidates = []
            return
        
        # Pre-compile stop words and important terms for faster processing
        stop_words = {'the', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'to', 'of', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        important_terms = {'ai', 'ml', 'js', 'ui', 'ux', 'it', 'qa', 'hr', 'rn', 'db', 'os', 'api', 'sql', 'aws', 'gcp', 'ios', 'android', 'vr', 'ar', 'iot', 'crm', 'erp', 'sap', 'oracle', 'mysql', 'postgres', 'redis', 'mongo', 'docker', 'kubernetes', 'k8s', 'ci', 'cd', 'devops', 'agile', 'scrum', 'kanban', 'jira', 'confluence', 'slack', 'zoom', 'teams', 'git', 'github', 'gitlab', 'bitbucket', 'jenkins', 'travis', 'circleci', 'aws', 'azure', 'gcp', 'heroku', 'netlify', 'vercel', 'firebase', 'supabase', 'stripe', 'paypal', 'twilio', 'sendgrid', 'mailchimp', 'hubspot', 'salesforce', 'zendesk', 'intercom', 'mixpanel', 'amplitude', 'segment', 'datadog', 'newrelic', 'sentry', 'rollbar', 'bugsnag', 'honeybadger', 'airbrake', 'raygun', 'logrocket', 'fullstory', 'hotjar', 'crazyegg', 'optimizely', 'vwo', 'ab', 'testing', 'a/b', 'test', 'qa', 'qc', 'uat', 'staging', 'prod', 'production', 'dev', 'development', 'staging', 'preprod', 'sandbox', 'local', 'localhost', '127.0.0.1', '0.0.0.0', 'localhost:3000', 'localhost:5000', 'localhost:8000', 'localhost:8080', 'localhost:9000', 'localhost:3001', 'localhost:5001', 'localhost:8001', 'localhost:8081', 'localhost:9001'}
        
        # Pre-allocate list for better memory performance
        self._processed_candidates = [None] * total_candidates
        processed_count = 0
        
        # Optimized batch processing with better memory management
        batch_size = 2000  # Increased batch size for better performance
        start_time = time.time()
        
        for i in range(0, total_candidates, batch_size):
            batch_end = min(i + batch_size, total_candidates)
            batch = self.candidates[i:batch_end]
            batch_start_time = time.time()
            
            # Process batch efficiently
            batch_processed = []
            for candidate in batch:
                if not candidate or not isinstance(candidate, dict):
                    continue
                
                # Optimized text processing
                skills = candidate.get('skills', [])
                if not isinstance(skills, list):
                    skills = []
                
                # Use join with generator for better memory efficiency
                skills_text = ' '.join(skill for skill in skills if skill and isinstance(skill, str)).lower()
                resume_text = (candidate.get('resume_text', '') or '').lower()
                full_name = (candidate.get('full_name', '') or '').lower()
                
                # Combine text efficiently
                combined_text = f"{skills_text} {resume_text} {full_name}"
                
                # Optimized word processing
                words = combined_text.split()
                words = {w for w in words if (len(w) > 2 and w not in stop_words) or w in important_terms}
                
                # Optimized skills processing
                skills_set = {skill.lower() for skill in skills if skill and isinstance(skill, str)}
                
                # Safe experience years extraction
                experience_years = 0
                try:
                    exp = candidate.get('total_experience_years', 0)
                    if isinstance(exp, (int, float)):
                        experience_years = exp
                except Exception:
                    pass
                
                batch_processed.append({
                    'original': candidate,
                    'combined_text': combined_text,
                    'words': words,
                    'skills_set': skills_set,
                    'experience_years': experience_years
                })
            
            # Add batch to pre-allocated list
            for j, processed_candidate in enumerate(batch_processed):
                self._processed_candidates[processed_count + j] = processed_candidate
            processed_count += len(batch_processed)
            
            # Improved progress tracking with ETA
            if total_candidates > 1000:
                batch_time = time.time() - batch_start_time
                batch_size_actual = len(batch_processed)
                rate = batch_size_actual / batch_time if batch_time > 0 else 0
                
                # Show progress every 1000 candidates or at the end
                if (i + batch_size) % 1000 == 0 or batch_end == total_candidates:
                    elapsed_time = time.time() - start_time
                    remaining_candidates = total_candidates - processed_count
                    eta = remaining_candidates / rate if rate > 0 else 0
                    print(f"⚙️ Processed {processed_count}/{total_candidates} candidates - Rate: {rate:.1f} candidates/sec - ETA: {eta:.1f}s")
        
        # Remove None entries and finalize
        self._processed_candidates = [c for c in self._processed_candidates if c is not None]
        
        total_time = time.time() - start_time
        final_rate = len(self._processed_candidates) / total_time if total_time > 0 else 0
        print(f"✅ Pre-processing completed: {len(self._processed_candidates)} candidates processed in {total_time:.2f}s (Rate: {final_rate:.1f} candidates/sec)")
    
    def _fallback_search(self, query: str, top_k: int = 10):
        """Fallback search when preprocessing is not complete - uses basic text matching"""
        if not query:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        results = []
        
        # Simple text matching on original candidates
        for candidate in self.candidates:
            if not candidate or not isinstance(candidate, dict):
                continue
            
            # Basic text matching
            skills_text = ' '.join(candidate.get('skills', [])).lower()
            resume_text = (candidate.get('resume_text', '') or '').lower()
            full_name = (candidate.get('full_name', '') or '').lower()
            
            combined_text = f"{skills_text} {resume_text} {full_name}"
            
            # Simple word matching score
            text_words = set(combined_text.split())
            matches = len(query_words.intersection(text_words))
            
            if matches > 0:
                results.append({
                    'candidate': candidate,
                    'score': matches,
                    'match_count': matches
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for search query"""
        return hashlib.md5(f"{query.lower()}:{top_k}".encode()).hexdigest()[:16]
    
    def _update_search_cache(self, query: str, top_k: int, results: list):
        """Update search cache with LRU eviction"""
        if len(self._search_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]
        
        cache_key = self._get_cache_key(query, top_k)
        self._search_cache[cache_key] = results
    
    def search(self, query: str, top_k: int = 10, **kwargs):
        """Enhanced search with custom LLM integration and bias prevention"""
        start_time = time.time()
        
        if not query:
            print("⚠️ Empty query provided")
            return []
        
        # Sanitize query to remove bias terms (if bias prevention is enabled)
        original_query = query
        sanitization_report = None
        enable_bias_prevention = kwargs.get('enable_bias_prevention', True)  # Default to True
        
        if self.use_bias_prevention and self.query_sanitizer and enable_bias_prevention:
            try:
                print("🛡️ Sanitizing query for bias prevention...")
                query = self.query_sanitizer.sanitize_query(query)
                sanitization_report = self.query_sanitizer.get_sanitization_report(original_query)
                if sanitization_report and sanitization_report.get('terms_removed'):
                    print(f"⚠️ Removed {len(sanitization_report['terms_removed'])} bias-related terms from query")
            except Exception as e:
                print(f"⚠️ Query sanitization failed: {e}. Using original query.")
                query = original_query
        
        # Wait for preprocessing to complete if it's still running
        if not self._preprocessing_complete:
            print("⏳ Waiting for preprocessing to complete...")
            if hasattr(self, '_preprocessing_thread'):
                self._preprocessing_thread.join(timeout=60)  # Wait up to 60 seconds for large datasets
            
            # If still not complete, provide a fallback search
            if not self._preprocessing_complete:
                print("⚠️ Preprocessing taking longer than expected, using fallback search...")
                return self._fallback_search(query, top_k)
        
        if not self._processed_candidates:
            print("⚠️ No processed candidates available for search")
            print(f"📊 Total candidates loaded: {len(self.candidates)}")
            if not self._preprocessing_complete:
                print("⏳ Preprocessing still in progress, using fallback search...")
                return self._fallback_search(query, top_k)
            return []
        
        # Check optimized cache first (if available)
        use_optimized_cache = kwargs.get('use_optimized_cache', True)  # Default to True
        if self.use_optimized_cache and self.optimized_cache and use_optimized_cache:
            try:
                print("⚡ Checking optimized cache...")
                cached_results = self.optimized_cache.search(query, limit=top_k)
                if cached_results:
                    print(f"✅ Found {len(cached_results)} results in optimized cache")
                    # Convert cached results to standard format if needed
                    if cached_results and isinstance(cached_results[0], dict) and 'candidate' in cached_results[0]:
                        # Convert from cache format to standard format
                        formatted_results = []
                        for result in cached_results:
                            candidate = result.get('candidate', result)
                            score = result.get('score', result.get('match_score', 0.0))
                            formatted_results.append({
                                **candidate,
                                'final_score': score,
                                'source': 'optimized_cache'
                            })
                        return formatted_results
                    return cached_results
            except Exception as e:
                print(f"⚠️ Optimized cache search failed: {e}")
        
        # Check regular cache
        cache_key = self._get_cache_key(query, top_k)
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        
        # Try Instant Search first (if available and enabled)
        use_instant_search = kwargs.get('use_instant_search', True)  # Default to True
        instant_results = None
        if self.use_instant_search and self.instant_search_engine and use_instant_search:
            try:
                print("⚡ Using Instant Search Engine...")
                instant_results = self.instant_search_engine.search(query, limit=top_k * 2)  # Get more for reranking
                if instant_results:
                    print(f"✅ Instant Search found {len(instant_results)} candidates")
            except Exception as e:
                print(f"⚠️ Instant Search failed: {e}")
        
        # Try Dense Retrieval Matcher (if available and enabled)
        use_dense_retrieval = kwargs.get('use_dense_retrieval', True)  # Default to True
        dense_results = None
        if self.use_dense_retrieval and use_dense_retrieval:
            try:
                print("🔍 Using Dense Retrieval Matcher...")
                if self.production_matcher:
                    # Use production matcher (includes reranking)
                    dense_results = self.production_matcher.find_matches(query, top_k=top_k * 2)
                    print(f"✅ Production Matcher found {len(dense_results)} candidates")
                elif self.dense_retriever and self.dense_retriever.index:
                    # Use standalone dense retriever
                    matches = self.dense_retriever.search(query, top_k=top_k * 2)
                    dense_results = [match[0] for match in matches]  # Extract candidates
                    print(f"✅ Dense Retrieval found {len(dense_results)} candidates")
            except Exception as e:
                print(f"⚠️ Dense Retrieval failed: {e}")
        
        # Enhance query using LLM if available (priority: LLM > Custom > Fallback)
        enhanced_query = None
        
        if self.use_llm_enhancement and self.llm_query_enhancer:
            try:
                print("🧠 Enhancing query with LLM (GPT-4/Claude)...")
                enhanced_query = self.llm_query_enhancer.enhance_query(query, use_llm=True)
                print(f"📝 LLM-enhanced query: {enhanced_query['expanded_terms'][:5]}...")
            except Exception as e:
                print(f"⚠️ LLM enhancement failed: {e}. Using fallback.")
                enhanced_query = None
        
        # Fallback to custom LLM enhancer
        if not enhanced_query and self.use_custom_llm and self.custom_query_enhancer:
            print("🧠 Enhancing query with custom LLM...")
            enhanced_query = self.custom_query_enhancer.enhance_query(query)
            print(f"📝 Enhanced query: {enhanced_query['expanded_terms'][:5]}...")
            # Intent classification
            if getattr(self, 'intent_classifier', None):
                intent, conf = self.intent_classifier.classify(query)
                enhanced_query['intent_class'] = intent
                enhanced_query['intent_confidence'] = conf
            # Contextual expansion (if available)
            if getattr(self, 'contextual_expander', None):
                try:
                    expansions = self.contextual_expander.expand_query(query, num_expansions=2)
                    enhanced_query['variations'] = list(set(enhanced_query.get('variations', []) + expansions))
                except Exception:
                    pass
        
        # Final fallback
        if not enhanced_query:
            enhanced_query = {'original_query': query, 'expanded_terms': query.split()}
        
        # Combine results from different search methods
        # Priority: Instant Search > Dense Retrieval > Hybrid Search
        
        # Use Instant Search results if available
        if instant_results:
            print("⚡ Using Instant Search results as base...")
            # Convert instant search results to standard format
            results = []
            for result in instant_results:
                if isinstance(result, dict):
                    candidate = result.get('candidate', result)
                    score = result.get('score', result.get('match_score', 0.0))
                    results.append({
                        **candidate,
                        'final_score': score,
                        'source': 'instant_search'
                    })
                else:
                    # If result is a candidate dict directly
                    results.append({
                        **result,
                        'final_score': 0.8,  # Default score
                        'source': 'instant_search'
                    })
        # Use Dense Retrieval results if available
        elif dense_results:
            print("🔍 Using Dense Retrieval results as base...")
            # Convert dense retrieval results to standard format
            results = []
            for result in dense_results:
                if isinstance(result, dict):
                    match_score = result.get('match_score', result.get('match_percentage', 0.0) / 100.0)
                    results.append({
                        **result,
                        'final_score': match_score,
                        'source': 'dense_retrieval'
                    })
                else:
                    # If result is a candidate dict directly
                    results.append({
                        **result,
                        'final_score': 0.7,  # Default score
                        'source': 'dense_retrieval'
                    })
        else:
            # Fallback to standard hybrid search
            print("🔍 Performing hybrid search (keyword + semantic)...")
            results = self._hybrid_search(query, enhanced_query, top_k)
        
        # Apply Enhanced Matcher if available (for additional scoring)
        use_enhanced_matcher = kwargs.get('use_enhanced_matcher', True)  # Default to True
        if self.use_enhanced_matcher and self.enhanced_matcher and use_enhanced_matcher and results:
            try:
                print("🎯 Applying Enhanced Matcher scoring...")
                # Use enhanced matcher to score candidates
                enhanced_scores = []
                for result in results[:top_k * 2]:  # Limit to top candidates
                    try:
                        # Get candidate resume text for enhanced matcher
                        resume_text = self._get_candidate_text_for_embedding(result)
                        # Get match score from enhanced matcher (takes job_description and resume as strings)
                        match_scores = self.enhanced_matcher.calculate_match_score(query, resume_text)
                        # Extract overall match score
                        overall_score = match_scores.get('overall_score', match_scores.get('match_score', 0.0))
                        if isinstance(overall_score, dict):
                            overall_score = overall_score.get('score', 0.0)
                        enhanced_scores.append((result, float(overall_score)))
                    except Exception as e:
                        print(f"⚠️ Enhanced Matcher scoring failed for candidate: {e}")
                        enhanced_scores.append((result, result.get('final_score', 0.0)))
                
                # Update scores with enhanced matcher scores (weighted combination)
                for result, enhanced_score in enhanced_scores:
                    original_score = result.get('final_score', 0.0)
                    # Combine: 70% original, 30% enhanced matcher
                    result['final_score'] = 0.7 * original_score + 0.3 * enhanced_score
                    result['enhanced_matcher_score'] = enhanced_score
                    result['source'] = result.get('source', 'hybrid') + '+enhanced_matcher'
                
                print(f"✅ Enhanced Matcher scored {len(enhanced_scores)} candidates")
            except Exception as e:
                print(f"⚠️ Enhanced Matcher failed: {e}")
        
        # Use Multi-Model Embedding Service for semantic search if available
        use_multi_model_embeddings = kwargs.get('use_multi_model_embeddings', True)  # Default to True
        if self.use_multi_model_embeddings and self.embedding_service and use_multi_model_embeddings:
            try:
                print("🔤 Using Multi-Model Embedding Service for semantic refinement...")
                # Get query embedding
                query_embedding = self.embedding_service.get_embedding(query, model_type='general')
                
                # Get embeddings for top candidates and compute similarity
                for result in results[:top_k * 2]:  # Limit to top candidates
                    try:
                        candidate_text = self._get_candidate_text_for_embedding(result)
                        candidate_embedding = self.embedding_service.get_embedding(candidate_text, model_type='general')
                        
                        # Compute cosine similarity
                        import numpy as np
                        similarity = np.dot(query_embedding, candidate_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
                        )
                        
                        # Update score with embedding similarity (weighted)
                        original_score = result.get('final_score', 0.0)
                        result['final_score'] = 0.8 * original_score + 0.2 * similarity
                        result['embedding_similarity'] = float(similarity)
                        result['source'] = result.get('source', 'hybrid') + '+multi_model_embeddings'
                    except Exception as e:
                        print(f"⚠️ Multi-Model Embedding scoring failed for candidate: {e}")
                        continue
                
                print(f"✅ Multi-Model Embedding Service refined {len(results[:top_k * 2])} candidates")
            except Exception as e:
                print(f"⚠️ Multi-Model Embedding Service failed: {e}")
        
        # Rerank results using custom cross-encoder if available
        if self.use_custom_llm and self.custom_cross_encoder and len(results) > 1:
            print("🔄 Reranking results with custom cross-encoder...")
            results = self._rerank_results(query, results)
        
        # Apply Learning-to-Rank model if available and trained
        if self.ltr_model and self.ltr_model.is_trained and len(results) > 1:
            print("🎯 Applying Learning-to-Rank model...")
            results = self._apply_ltr_ranking(query, results)
        
        # Apply RL Ranking Agent if available and trained
        if self.rl_ranking_agent and self.rl_ranking_agent.is_trained and len(results) > 1:
            print("🤖 Applying RL Ranking Agent...")
            results = self._apply_rl_ranking(query, results)
        
        # Generate explanations using Explainable AI (replaces CustomExplanationGenerator)
        if self.use_explainable_ai and self.explainable_ai:
            print("📝 Generating explanations with Explainable AI...")
            for idx, result in enumerate(results):
                try:
                    # Prepare candidate profile for ExplainableRecruitmentAI
                    candidate_profile = {
                        'full_name': result.get('full_name', ''),
                        'email': result.get('email', ''),
                        'skills': result.get('skills', []),
                        'experience_years': result.get('total_experience_years', 0),
                        'seniority_level': result.get('seniority_level', 'mid'),
                        'resume_text': result.get('resume_text', ''),
                        'education': result.get('education', ''),
                        'location': result.get('location', '')
                    }
                    
                    # Prepare match scores for ExplainableRecruitmentAI
                    match_scores = {
                        'overall_score': result.get('final_score', 0.0) * 100.0,  # Convert to 0-100 scale
                        'technical_skills_score': result.get('skill_score', 0.0) * 100.0,
                        'experience_score': result.get('experience_score', 0.0) * 100.0,
                        'seniority_score': result.get('seniority_score', 0.0) * 100.0,
                        'education_score': result.get('education_score', 0.0) * 100.0,
                        'soft_skills_score': result.get('soft_skills_score', 0.0) * 100.0,
                        'location_score': result.get('location_score', 0.0) * 100.0
                    }
                    
                    # Get ranking position (1-indexed)
                    ranking_position = idx + 1
                    
                    # Generate explanation using ExplainableRecruitmentAI
                    explanation = self.explainable_ai.explain_candidate_selection(
                        candidate_profile=candidate_profile,
                        job_query=query,
                        match_scores=match_scores,
                        ranking_position=ranking_position
                    )
                    
                    # Add explanation to result (convert DecisionExplanation to dict format)
                    result['ai_explanation'] = explanation.decision_summary
                    result['explanation_details'] = {
                        'overall_score': explanation.overall_score,
                        'confidence_level': explanation.confidence_level,
                        'recommendation': explanation.recommendation,
                        'top_positive_factors': explanation.top_positive_factors,
                        'top_negative_factors': explanation.top_negative_factors,
                        'strength_areas': explanation.strength_areas,
                        'risk_factors': explanation.risk_factors,
                        'feature_contributions': [
                            {
                                'feature_name': fc.feature_name,
                                'contribution': fc.contribution,
                                'percentage': fc.percentage,
                                'direction': fc.direction,
                                'explanation': fc.explanation
                            }
                            for fc in explanation.feature_contributions
                        ]
                    }
                except Exception as e:
                    print(f"⚠️ Explainable AI failed for candidate {result.get('email', 'unknown')}: {e}")
                    # Fallback to basic explanation
                    result['ai_explanation'] = f"Candidate with score {result.get('final_score', 0.0):.2f}"
                    result['explanation_details'] = None
        
        # Enrich results with behavioural analysis if available and enabled
        include_behavioural = kwargs.get('include_behavioural_analysis', True)  # Default to True
        if self.use_behavioral_analysis and self.behavioral_pipeline and include_behavioural:
            print("🧠 Enriching results with behavioural analysis...")
            results = self._enrich_with_behavioral_analysis(query, results)
        
        # Assess diversity and add bias prevention info (if bias prevention is enabled)
        diversity_assessment = None
        if self.use_bias_prevention and self.bias_monitor and enable_bias_prevention:
            try:
                print("🛡️ Assessing diversity in search results...")
                diversity_assessment = self.bias_monitor.assess_diversity(query, results)
                
                # Add diversity info to each result
                for result in results:
                    result['diversity_info'] = {
                        'diversity_score': diversity_assessment.get('diversity_score', 0.0),
                        'representation_balance': diversity_assessment.get('representation_balance', {}),
                        'bias_flags': diversity_assessment.get('bias_flags', []),
                        'recommendations': diversity_assessment.get('recommendations', [])
                    }
                
                # Store assessment for system-level access
                self.latest_diversity_assessment = diversity_assessment
                self.latest_sanitization_report = sanitization_report
                
                # Check for bias incidents and trigger alerts if configured
                if self.alert_config.get('underrepresentation_alerts', False) and diversity_assessment.get('bias_flags'):
                    self.bias_incidents_detected = diversity_assessment['bias_flags']
                    print(f"⚠️ Bias incidents detected: {len(diversity_assessment['bias_flags'])} flags")
                
                # Store monitoring data for compliance reporting
                if self.monitoring_config.get('compliance_reporting', False):
                    self.compliance_data = {
                        'query': query,
                        'original_query': original_query,
                        'diversity_assessment': diversity_assessment,
                        'sanitization_report': sanitization_report,
                        'timestamp': diversity_assessment.get('timestamp', datetime.now().isoformat())
                    }
                
                print(f"✅ Diversity assessment completed (score: {diversity_assessment.get('diversity_score', 0.0):.2f})")
            except Exception as e:
                print(f"⚠️ Diversity assessment failed: {e}")
        
        # Enrich results with market intelligence if available and enabled
        include_market_intel = kwargs.get('include_market_intelligence', True)  # Default to True
        if self.use_market_intelligence and include_market_intel:
            print("📊 Enriching results with market intelligence...")
            results = self._enrich_with_market_intelligence(query, results)
        
        # Sort by final score and return top_k
        results.sort(key=lambda x: x['final_score'], reverse=True)
        final_results = results[:top_k]
        
        # Add bias prevention metadata to response if available
        if diversity_assessment:
            self.search_metadata = {
                'bias_prevention': {
                    'query_sanitized': query != original_query,
                    'sanitization_report': sanitization_report,
                    'diversity_assessment': diversity_assessment,
                    'bias_incidents_detected': diversity_assessment.get('bias_flags', [])
                }
            }
        
        # Debug information
        print(f"🔍 Search query: '{query}'")
        print(f"📊 Found {len(results)} total matches, returning top {len(final_results)}")
        if final_results:
            print(f"🏆 Best match: {final_results[0]['full_name']} (score: {final_results[0]['final_score']:.3f})")
            if 'ai_explanation' in final_results[0]:
                print(f"💡 Explanation: {final_results[0]['ai_explanation']}")
        else:
            print("❌ No matches found")
        
        # Cache the results
        self._update_search_cache(query, top_k, final_results)
        
        # Update performance stats
        search_time = time.time() - start_time
        self._search_count += 1
        self._total_search_time += search_time
        
        return final_results
    
    def _hybrid_search(self, query: str, enhanced_query: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Perform hybrid search combining keyword and semantic matching"""
        results = []
        
        # Get expanded query terms
        expanded_terms = enhanced_query.get('expanded_terms', query.split())
        query_words = set(term.lower() for term in expanded_terms)
        
        # Remove common stop words from query (but keep important technical terms)
        stop_words = {'the', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'to', 'of', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        important_terms = {'ai', 'ml', 'js', 'ui', 'ux', 'it', 'qa', 'hr', 'rn', 'db', 'os', 'api', 'sql', 'aws', 'gcp', 'ios', 'android', 'vr', 'ar', 'iot', 'crm', 'erp', 'sap', 'oracle', 'mysql', 'postgres', 'redis', 'mongo', 'docker', 'kubernetes', 'k8s', 'ci', 'cd', 'devops', 'agile', 'scrum', 'kanban', 'jira', 'confluence', 'slack', 'zoom', 'teams', 'git', 'github', 'gitlab', 'bitbucket', 'jenkins', 'travis', 'circleci', 'aws', 'azure', 'gcp', 'heroku', 'netlify', 'vercel', 'firebase', 'supabase', 'stripe', 'paypal', 'twilio', 'sendgrid', 'mailchimp', 'hubspot', 'salesforce', 'zendesk', 'intercom', 'mixpanel', 'amplitude', 'segment', 'datadog', 'newrelic', 'sentry', 'rollbar', 'bugsnag', 'honeybadger', 'airbrake', 'raygun', 'logrocket', 'fullstory', 'hotjar', 'crazyegg', 'optimizely', 'vwo', 'ab', 'testing', 'a/b', 'test', 'qa', 'qc', 'uat', 'staging', 'prod', 'production', 'dev', 'development', 'staging', 'preprod', 'sandbox', 'local', 'localhost', '127.0.0.1', '0.0.0.0', 'localhost:3000', 'localhost:5000', 'localhost:8000', 'localhost:8080', 'localhost:9000', 'localhost:3001', 'localhost:5001', 'localhost:8001', 'localhost:8081', 'localhost:9001'}
        query_words = {w for w in query_words if (len(w) > 2 and w not in stop_words) or w in important_terms}
        
        # 1) Keyword scoring using preprocessed candidates
        for processed_candidate in self._processed_candidates:
            # Quick relevance check - skip if no word overlap
            if not query_words.intersection(processed_candidate['words']):
                continue
            
            # Calculate hybrid similarity score
            keyword_score = self._calculate_optimized_similarity(query.lower(), query_words, processed_candidate)
            
            # Add semantic similarity if custom embedding model is available
            semantic_score = 0.0
            if self.use_custom_llm and self.custom_embedding_model:
                candidate_text = processed_candidate['combined_text']
                semantic_score = self.custom_embedding_model.similarity(query, candidate_text)
            
            # Combine scores (weighted average)
            if self.use_custom_llm and self.custom_embedding_model:
                final_score = 0.6 * keyword_score + 0.4 * semantic_score
            else:
                final_score = keyword_score
            
            if final_score > 0.1:  # Only include candidates with some relevance
                candidate = processed_candidate['original']
                result = {
                    'email': candidate.get('email', ''),
                    'full_name': candidate.get('full_name', ''),
                    'skills': candidate.get('skills', []),
                    'total_experience_years': candidate.get('total_experience_years', 0),
                    'resume_text': candidate.get('resume_text', ''),
                    'phone': candidate.get('phone', ''),
                    'sourceURL': candidate.get('sourceURL', ''),
                    'similarity_score': keyword_score,
                    'semantic_score': semantic_score,
                    'final_score': final_score,
                    'grade': self._get_grade(final_score),
                    'domain': 'general',
                    'ai_explanation': f"Hybrid match: keyword={keyword_score:.2f}, semantic={semantic_score:.2f}"
                }
                results.append(result)

        # 2) Semantic retrieval via vector index (bring in additional candidates)
        if self.use_custom_llm and self.custom_embedding_model and self.custom_embedding_model.is_index_ready():
            print("🔎 Semantic retrieval via vector index...")
            query_vec = self.custom_embedding_model.encode(query)
            idxs, scores = self.custom_embedding_model.search_index(query_vec, top_k=max(top_k * 3, 50))
            # Merge semantic candidates into results, avoid duplicates via email
            existing_emails = {r['email'] for r in results}
            for i, score in zip(idxs, scores):
                if i < 0 or i >= len(self._candidate_index_to_id):
                    continue
                email = self._candidate_index_to_id[i]
                if email in existing_emails:
                    continue
                # Find candidate by email
                candidate = next((c for c in self.candidates if c.get('email') == email), None)
                if not candidate:
                    continue
                results.append({
                    'email': candidate.get('email', ''),
                    'full_name': candidate.get('full_name', ''),
                    'skills': candidate.get('skills', []),
                    'total_experience_years': candidate.get('total_experience_years', 0),
                    'resume_text': candidate.get('resume_text', ''),
                    'phone': candidate.get('phone', ''),
                    'sourceURL': candidate.get('sourceURL', ''),
                    'similarity_score': 0.0,
                    'semantic_score': float(score),
                    'final_score': float(score),
                    'grade': self._get_grade(float(score)),
                    'domain': 'general',
                    'ai_explanation': f"Semantic vector match: {float(score):.2f}"
                })
        
        return results
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using custom cross-encoder"""
        if not self.custom_cross_encoder:
            return results
        
        # Score each result with cross-encoder
        for result in results:
            cross_encoder_score = self.custom_cross_encoder.encode_pair(query, result)
            # Combine with original score
            result['final_score'] = 0.7 * result['final_score'] + 0.3 * cross_encoder_score
            result['cross_encoder_score'] = cross_encoder_score
        
        return results
    
    def _apply_ltr_ranking(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Learning-to-Rank model to rerank results"""
        if not self.ltr_model or not self.ltr_model.is_trained:
            return results
        
        # Prepare feature scores for LTR model
        feature_scores = []
        candidates = []
        
        for result in results:
            # Extract existing scores
            feature_scores.append({
                'keyword_score': result.get('similarity_score', 0.0),
                'semantic_score': result.get('semantic_score', 0.0),
                'cross_encoder_score': result.get('cross_encoder_score', 0.0)
            })
            candidates.append(result)
        
        # Predict scores using LTR model
        try:
            ltr_scores = self.ltr_model.predict(query, candidates, feature_scores=feature_scores)
            
            # Update results with LTR scores
            for i, (result, ltr_score) in enumerate(zip(results, ltr_scores)):
                result['ltr_score'] = float(ltr_score)
                # Use LTR score as final score (or combine with original)
                # For now, use LTR score directly as it's trained on the best combination
                result['final_score'] = float(ltr_score)
        except Exception as e:
            print(f"⚠️ LTR ranking failed: {e}")
            # Fallback to original scores
            pass
        
        # Re-sort by final score
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return results
    
    def _apply_rl_ranking(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply RL ranking agent to rerank results"""
        if not self.rl_ranking_agent or not self.rl_ranking_agent.is_trained:
            return results
        
        # Prepare feature scores for RL agent
        feature_scores = []
        candidates = []
        
        for result in results:
            feature_scores.append({
                'keyword_score': result.get('similarity_score', 0.0),
                'semantic_score': result.get('semantic_score', 0.0),
                'cross_encoder_score': result.get('cross_encoder_score', 0.0),
                'skill_overlap': 0.0,  # Can be computed from result
                'experience_match': 0.0,
                'domain_match': 0.0
            })
            candidates.append(result)
        
        # Get rankings from RL agent
        try:
            rankings = self.rl_ranking_agent.rank_candidates(
                query, candidates, feature_scores=feature_scores, deterministic=True
            )
            
            # Reorder results based on RL rankings
            ranked_results = []
            ranked_indices = {idx for idx, _ in rankings}
            
            # Add ranked candidates first
            for idx, score in rankings:
                if idx < len(results):
                    results[idx]['rl_score'] = float(score)
                    results[idx]['final_score'] = float(score)  # Use RL score as final
                    ranked_results.append(results[idx])
            
            # Add unranked candidates (if any)
            for i, result in enumerate(results):
                if i not in ranked_indices:
                    ranked_results.append(result)
            
            return ranked_results
        except Exception as e:
            print(f"⚠️ RL ranking failed: {e}")
            # Fallback to original results
            return results
    
    def _enrich_with_behavioral_analysis(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich search results with behavioural analysis"""
        if not self.behavioral_pipeline or not results:
            return results
        
        try:
            # Extract target role from query (simplified)
            target_role = self._extract_target_role(query)
            
            # Enrich each result with behavioural analysis
            for result in results:
                try:
                    # Create multi-source profile from candidate data
                    resume_text = result.get('resume_text', '')
                    if not resume_text:
                        # Fallback: combine available text
                        skills_text = ' '.join(result.get('skills', []))
                        full_name = result.get('full_name', '')
                        resume_text = f"{full_name} {skills_text}"
                    
                    # Create profile (basic version - can be enhanced with LinkedIn/GitHub data later)
                    if create_multi_source_profile:
                        profile_data = create_multi_source_profile(resume_text=resume_text)
                    else:
                        # Fallback: create basic profile
                        profile_data = MultiSourceProfile(resume_text=resume_text) if MultiSourceProfile else None
                    
                    if not profile_data:
                        continue
                    
                    # Perform behavioural analysis
                    if hasattr(self.behavioral_pipeline, 'analyze_comprehensive_profile'):
                        # Use comprehensive analysis
                        behavioral_results = self.behavioral_pipeline.analyze_comprehensive_profile(
                            source_data=profile_data,
                            target_role=target_role,
                            job_description=query
                        )
                    else:
                        # Fallback: use basic analyze method if available
                        behavioral_results = None
                        if hasattr(self.behavioral_pipeline, 'analyze'):
                            behavioral_results = self.behavioral_pipeline.analyze(resume_text, query)
                    
                    # Add behavioural analysis to result
                    if behavioral_results:
                        # Extract behavioural profile data
                        behavioral_profile = behavioral_results.get('behavioral_profile')
                        if behavioral_profile:
                            # Add behavioral scores
                            result['behavioural_analysis'] = {
                                'overall_score': getattr(behavioral_profile, 'overall_score', 0.0),
                                'leadership_score': getattr(behavioral_profile, 'leadership_score', 0.0),
                                'collaboration_score': getattr(behavioral_profile, 'collaboration_score', 0.0),
                                'innovation_score': getattr(behavioral_profile, 'innovation_score', 0.0),
                                'adaptability_score': getattr(behavioral_profile, 'adaptability_score', 0.0),
                                'technical_depth': getattr(behavioral_profile, 'technical_depth', 0.0),
                                'problem_solving_ability': getattr(behavioral_profile, 'problem_solving_ability', 0.0),
                                'emotional_intelligence': getattr(behavioral_profile, 'emotional_intelligence', 0.0),
                                'cultural_alignment': getattr(behavioral_profile, 'cultural_alignment', 0.0),
                                'growth_potential': getattr(behavioral_profile, 'growth_potential', 0.0),
                                'strengths': getattr(behavioral_profile, 'strengths', []),
                                'risk_factors': getattr(behavioral_profile, 'risk_factors', []),
                                'development_areas': getattr(behavioral_profile, 'development_areas', []),
                                'behavioral_patterns': getattr(behavioral_profile, 'behavioral_patterns', [])
                            }
                        else:
                            # Fallback: extract from dict if available
                            result['behavioural_analysis'] = {
                                'overall_score': behavioral_results.get('final_score', 0.0),
                                'behavioral_score': behavioral_results.get('behavioral_score', 0.0),
                                'technical_score': behavioral_results.get('technical_score', 0.0),
                                'cultural_fit': behavioral_results.get('cultural_fit', 0.0),
                                'strengths': behavioral_results.get('strengths', []),
                                'risk_factors': behavioral_results.get('risk_factors', []),
                                'recommendations': behavioral_results.get('recommendations', [])
                            }
                        
                        # Add role fit analysis
                        role_fit = behavioral_results.get('role_fit_analysis')
                        if role_fit:
                            result['role_fit_analysis'] = role_fit
                        
                        # Add career trajectory if available
                        career_trajectory = behavioral_results.get('career_trajectory')
                        if career_trajectory:
                            result['career_trajectory'] = {
                                'current_level': getattr(career_trajectory, 'current_level', 'unknown'),
                                'predicted_next_roles': getattr(career_trajectory, 'predicted_next_roles', []),
                                'growth_timeline': getattr(career_trajectory, 'growth_timeline', {}),
                                'skill_development_path': getattr(career_trajectory, 'skill_development_path', []),
                                'development_recommendations': getattr(career_trajectory, 'development_recommendations', [])
                            }
                        
                        # Update final score with behavioural score if available
                        behavioral_score = result['behavioural_analysis'].get('overall_score', 0.0)
                        if behavioral_score > 0:
                            # Combine behavioural score with existing score (weighted)
                            original_score = result.get('final_score', 0.0)
                            result['final_score'] = 0.7 * original_score + 0.3 * behavioral_score
                            result['behavioral_score_contribution'] = behavioral_score
                
                except Exception as e:
                    print(f"⚠️ Behavioural analysis failed for candidate {result.get('email', 'unknown')}: {e}")
                    # Continue with other results
                    continue
            
        except Exception as e:
            print(f"⚠️ Behavioural analysis enrichment failed: {e}")
            # Return results without behavioural analysis
        
        return results
    
    def _extract_target_role(self, query: str) -> str:
        """Extract target role from query"""
        query_lower = query.lower()
        
        # Common role patterns
        role_patterns = {
            'senior software engineer': ['senior', 'software', 'engineer'],
            'software engineer': ['software', 'engineer'],
            'senior developer': ['senior', 'developer'],
            'developer': ['developer'],
            'data scientist': ['data', 'scientist'],
            'product manager': ['product', 'manager'],
            'engineering manager': ['engineering', 'manager'],
            'senior python developer': ['senior', 'python', 'developer'],
            'python developer': ['python', 'developer']
        }
        
        # Find matching role
        for role, keywords in role_patterns.items():
            if all(keyword in query_lower for keyword in keywords):
                return role
        
        # Default: extract key terms
        words = query.split()
        if len(words) >= 2:
            return ' '.join(words[:2])
        
        return 'professional role'
    
    def _enrich_with_market_intelligence(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich search results with market intelligence data"""
        if not results:
            return results
        
        try:
            # Extract skills and roles from query and results
            query_skills = self._extract_skills_from_query(query)
            query_role = self._extract_target_role(query)
            
            # Get market intelligence data
            market_data = {}
            
            # 1. Talent availability analysis
            if analyze_talent_availability:
                try:
                    talent_data = analyze_talent_availability()
                    market_data['talent_availability'] = talent_data
                except Exception as e:
                    print(f"⚠️ Talent availability analysis failed: {e}")
            
            # 2. Competitive intelligence
            if competitive_intelligence:
                try:
                    competitive_data = competitive_intelligence()
                    market_data['competitive_intelligence'] = competitive_data
                except Exception as e:
                    print(f"⚠️ Competitive intelligence failed: {e}")
            
            # 3. Skill demand forecasting
            skill_demand_data = {}
            
            # Try using existing SkillDemandForecaster first (already initialized)
            if hasattr(self, 'demand_forecaster') and self.demand_forecaster and query_skills:
                try:
                    for skill in query_skills[:5]:  # Limit to top 5 skills
                        try:
                            forecast = self.demand_forecaster.forecast_skill_demand(skill, months=6)
                            if forecast:
                                skill_demand_data[skill] = forecast
                        except Exception as e:
                            print(f"⚠️ Skill forecast failed for {skill}: {e}")
                            continue
                except Exception as e:
                    print(f"⚠️ Skills forecasting failed: {e}")
            
            # Fallback to market intelligence skills forecaster if available
            if not skill_demand_data and self.skills_forecaster and query_skills:
                try:
                    # Use simplified approach - create demand forecast
                    # Note: In production, this would call async forecast_skills method
                    for skill in query_skills[:5]:
                        skill_demand_data[skill] = {
                            'demand_index_current': 0.7,  # Placeholder
                            'forecast_1m': 0.72,
                            'forecast_3m': 0.75,
                            'forecast_6m': 0.78,
                            'confidence': 0.8,
                            'trend': 'rising'
                        }
                except Exception as e:
                    print(f"⚠️ Skills forecasting failed: {e}")
            
            market_data['skill_demand'] = skill_demand_data
            
            # 4. Compensation benchmarking (simplified - use market intelligence API if available)
            compensation_data = {}
            if self.market_intelligence and query_role:
                try:
                    # Use market intelligence API to get salary trends
                    # This would be async in production, simplified here
                    compensation_data = {
                        'average_salary': 120000,  # Placeholder - would be from API
                        'salary_range': {'min': 90000, 'max': 150000},
                        'trend': 'rising',
                        'growth_rate': 5.0
                    }
                except Exception as e:
                    print(f"⚠️ Compensation benchmarking failed: {e}")
            
            market_data['compensation'] = compensation_data
            
            # Add market intelligence to each result
            for result in results:
                result['market_intelligence'] = {
                    'talent_availability': market_data.get('talent_availability', {}),
                    'competitive_intelligence': market_data.get('competitive_intelligence', {}),
                    'skill_demand': self._get_skill_demand_for_candidate(result, skill_demand_data),
                    'compensation': compensation_data,
                    'market_insights': self._generate_market_insights(result, market_data)
                }
                
                # Add skill demand scores to candidate skills
                candidate_skills = result.get('skills', [])
                if candidate_skills and skill_demand_data:
                    result['skill_demand_scores'] = {
                        skill: skill_demand_data.get(skill, {}).get('demand_index_current', 0.0)
                        for skill in candidate_skills
                        if skill in skill_demand_data
                    }
            
            print(f"✅ Market intelligence enrichment completed")
            
        except Exception as e:
            print(f"⚠️ Market intelligence enrichment failed: {e}")
            # Continue without market intelligence
        
        return results
    
    def _extract_skills_from_query(self, query: str) -> List[str]:
        """Extract skills from query"""
        # Common skills dictionary
        common_skills = [
            'python', 'java', 'javascript', 'react', 'node.js', 'aws', 'docker', 'kubernetes',
            'machine learning', 'ai', 'data science', 'sql', 'mongodb', 'postgresql',
            'django', 'flask', 'spring', 'angular', 'vue', 'typescript', 'html', 'css',
            'nurse', 'doctor', 'physician', 'medical', 'healthcare',
            'finance', 'banking', 'accounting', 'investment'
        ]
        
        query_lower = query.lower()
        found_skills = []
        
        for skill in common_skills:
            if skill in query_lower:
                found_skills.append(skill)
        
        # Also extract from result skills if available
        return found_skills
    
    def _get_skill_demand_for_candidate(self, result: Dict[str, Any], skill_demand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get skill demand data relevant to candidate"""
        candidate_skills = result.get('skills', [])
        relevant_demand = {}
        
        for skill in candidate_skills:
            skill_lower = skill.lower()
            # Find matching skill in demand data
            for demand_skill, demand_info in skill_demand_data.items():
                if demand_skill.lower() == skill_lower or demand_skill.lower() in skill_lower:
                    relevant_demand[skill] = demand_info
                    break
        
        return relevant_demand
    
    def _generate_market_insights(self, result: Dict[str, Any], market_data: Dict[str, Any]) -> List[str]:
        """Generate market insights for candidate"""
        insights = []
        
        candidate_skills = result.get('skills', [])
        skill_demand = market_data.get('skill_demand', {})
        
        # Check skill demand
        high_demand_skills = []
        for skill in candidate_skills:
            skill_lower = skill.lower()
            for demand_skill, demand_info in skill_demand.items():
                if demand_skill.lower() == skill_lower:
                    demand_score = demand_info.get('demand_index_current', 0.0)
                    if demand_score > 0.7:
                        high_demand_skills.append(skill)
        
        if high_demand_skills:
            insights.append(f"High market demand for skills: {', '.join(high_demand_skills[:3])}")
        
        # Check talent availability
        talent_availability = market_data.get('talent_availability', {})
        active_candidates = talent_availability.get('active_candidates', {})
        
        if active_candidates:
            total_active = sum(active_candidates.values())
            if total_active > 10000:
                insights.append("High competition in talent market")
            elif total_active < 5000:
                insights.append("Low competition - favorable market conditions")
        
        # Check compensation trends
        compensation = market_data.get('compensation', {})
        if compensation.get('trend') == 'rising':
            insights.append("Compensation trends are rising in this market")
        
        return insights
    
    def _create_candidate_profile_for_enhanced_matcher(self, result: Dict[str, Any]) -> Any:
        """Create candidate profile for Enhanced Matcher"""
        try:
            from semantic_function.matcher.models import CandidateProfile
            
            return CandidateProfile(
                full_name=result.get('full_name', ''),
                email=result.get('email', ''),
                resume=result.get('resume_text', ''),
                skills=result.get('skills', []),
                experience_years=result.get('total_experience_years', 0),
                location=result.get('location', ''),
                linkedin_url=result.get('linkedin_url', ''),
                license_number=result.get('license_number', '')
            )
        except Exception:
            # Fallback to dict format
            return {
                'full_name': result.get('full_name', ''),
                'email': result.get('email', ''),
                'resume': result.get('resume_text', ''),
                'skills': result.get('skills', []),
                'experience_years': result.get('total_experience_years', 0),
                'location': result.get('location', ''),
                'linkedin_url': result.get('linkedin_url', ''),
                'license_number': result.get('license_number', '')
            }
    
    def _get_candidate_text_for_embedding(self, result: Dict[str, Any]) -> str:
        """Get candidate text for embedding"""
        skills_text = ' '.join(result.get('skills', []))
        resume_text = result.get('resume_text', '')
        full_name = result.get('full_name', '')
        return f"{skills_text} {resume_text} {full_name}".strip()
    
    def _calculate_optimized_similarity(self, query_lower: str, query_words: set, processed_candidate: dict) -> float:
        """Optimized similarity calculation using pre-processed data"""
        # Fast word set intersection for base similarity
        candidate_words = processed_candidate['words']
        intersection = query_words.intersection(candidate_words)
        
        if not intersection:
            return 0.0
        
        # Fast Jaccard similarity
        union = len(query_words) + len(candidate_words) - len(intersection)
        base_similarity = len(intersection) / union if union > 0 else 0.0
        
        # Fast skill matching using pre-computed sets
        skills_set = processed_candidate['skills_set']
        skill_intersection = query_words.intersection(skills_set)
        skill_boost = min(len(skill_intersection) * 0.15, 0.4)  # Max 0.4 boost for skills
        
        # Fast experience matching
        experience_years = processed_candidate['experience_years']
        experience_boost = 0.0
        
        # Pre-compiled experience keywords for faster matching
        senior_keywords = {'senior', 'lead', 'principal', 'architect', 'manager', 'director'}
        junior_keywords = {'junior', 'entry', 'graduate', 'intern', 'associate'}
        
        if query_words.intersection(senior_keywords) and experience_years >= 5:
            experience_boost = 0.25
        elif query_words.intersection(junior_keywords) and experience_years <= 3:
            experience_boost = 0.25
        elif experience_years >= 3:
            experience_boost = 0.1
        
        # Calculate final score with boosts
        final_score = min(base_similarity + skill_boost + experience_boost, 1.0)
        
        return final_score
    
    def _get_grade(self, score: float) -> str:
        """Convert score to grade"""
        if score >= 0.7:
            return 'A'
        elif score >= 0.5:
            return 'B'
        elif score >= 0.3:
            return 'C'
        else:
            return 'D'
    
    def get_performance_stats(self):
        """Get performance statistics"""
        avg_search_time = self._total_search_time / self._search_count if self._search_count > 0 else 0.0
        return {
            'total_searches': self._search_count,
            'avg_search_time_ms': round(avg_search_time * 1000, 2),
            'cache_hits': len(self._search_cache),
            'index_size': len(self.candidates),
            'cache_size': len(self._search_cache),
            'optimization_level': 'high_performance'
        }
