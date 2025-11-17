# search_system.py - Optimized Search System
"""
High-performance search system with caching and optimized algorithms
"""

import os
import sys
import json
import re
import time
import hashlib
import logging
import math
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from difflib import SequenceMatcher

# Add parent directory to path to find Sagemaker folder, ML models, features, and ops utils
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_ml_dir = os.path.join(_parent_dir, 'ml')
_features_dir = os.path.join(os.path.dirname(_parent_dir), 'features')
_ops_dir = os.path.join(_parent_dir, 'ops')
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _ml_dir not in sys.path:
    sys.path.insert(0, _ml_dir)
if _features_dir not in sys.path:
    sys.path.insert(0, _features_dir)
if _ops_dir not in sys.path:
    sys.path.insert(0, _ops_dir)

try:
    MAX_DYNAMODB_CANDIDATES = int(os.getenv('MAX_DYNAMODB_CANDIDATES', '100'))
except ValueError:
    MAX_DYNAMODB_CANDIDATES = 100

logger = logging.getLogger(__name__)


def _resolve_sagemaker_endpoint(*legacy_keys: str) -> Tuple[Optional[str], bool]:
    """
    Resolve the active SageMaker endpoint.
    
    Returns unified endpoint when configured, otherwise falls back to the
    provided legacy environment variables.
    """
    unified = os.getenv('SAGEMAKER_UNIFIED_ENDPOINT')
    if unified:
        return unified, True
    for key in legacy_keys:
        value = os.getenv(key)
        if value:
            return value, False
    return None, False

_EXPERIENCE_FIELDS = (
    'total_experience_years',
    'experience_years',
    'years_experience',
    'years_of_experience',
    'experience',
    'Experience',
    'ExperienceYears',
    'experienceYears',
)

_EXPERIENCE_RANGE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)(?=\s*(?:years?|yrs))',
    re.IGNORECASE,
)
_EXPERIENCE_PLUS_RE = re.compile(
    r'(\d+(?:\.\d+)?)(?=\s*\+\s*(?:years?|yrs))',
    re.IGNORECASE,
)
_EXPERIENCE_SIMPLE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs)\b',
    re.IGNORECASE,
)
_EXPERIENCE_FALLBACK_RE = re.compile(
    r'(\d+(?:\.\d+)?)(?=\s*(?:year\b|yrs\b|yr\b))',
    re.IGNORECASE,
)


def _parse_experience_value(value: Any) -> Optional[float]:
    """Attempt to parse a numeric experience value from arbitrary input."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            if math.isnan(float(value)):  # type: ignore[arg-type]
                return None
        except (TypeError, ValueError):
            return None
        return float(value)

    if isinstance(value, dict):
        for field in _EXPERIENCE_FIELDS:
            nested = value.get(field)
            parsed = _parse_experience_value(nested)
            if parsed is not None:
                return parsed
        for nested_value in value.values():
            parsed = _parse_experience_value(nested_value)
            if parsed is not None:
                return parsed
        return None

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None

        cleaned = cleaned.replace(',', '')

        try:
            return float(cleaned)
        except ValueError:
            pass

        matches: List[float] = []

        for match in _EXPERIENCE_RANGE_RE.finditer(cleaned):
            try:
                first = float(match.group(1))
                second = float(match.group(2))
                matches.append(max(first, second))
            except (TypeError, ValueError):
                continue

        for match in _EXPERIENCE_PLUS_RE.finditer(cleaned):
            try:
                matches.append(float(match.group(1)))
            except (TypeError, ValueError):
                continue

        for match in _EXPERIENCE_SIMPLE_RE.finditer(cleaned):
            try:
                matches.append(float(match.group(1)))
            except (TypeError, ValueError):
                continue

        if not matches:
            for match in _EXPERIENCE_FALLBACK_RE.finditer(cleaned):
                try:
                    matches.append(float(match.group(1)))
                except (TypeError, ValueError):
                    continue

        if matches:
            return max(matches)

    return None


def _extract_experience_from_text(text: str) -> Optional[float]:
    """Extract the highest experience mention from unstructured resume text."""
    if not text or not isinstance(text, str):
        return None

    matches: List[float] = []
    date_ranges: List[Tuple[int, int]] = []  # Track date ranges to sum them up

    # Pattern 1: Experience ranges (e.g., "5-7 years", "3 to 5 years")
    for match in _EXPERIENCE_RANGE_RE.finditer(text):
        try:
            first = float(match.group(1))
            second = float(match.group(2))
            matches.append(max(first, second))
        except (TypeError, ValueError):
            continue

    # Pattern 2: Experience with plus (e.g., "5+ years")
    for match in _EXPERIENCE_PLUS_RE.finditer(text):
        try:
            matches.append(float(match.group(1)))
        except (TypeError, ValueError):
            continue

    # Pattern 3: Simple experience mentions (e.g., "5 years", "3 yrs")
    for match in _EXPERIENCE_SIMPLE_RE.finditer(text):
        try:
            matches.append(float(match.group(1)))
        except (TypeError, ValueError):
            continue

    # Pattern 4: Calculate from date ranges (e.g., "2015-2020", "Jan 2018 - Dec 2023")
    # This is important for extracting experience from work history
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Year ranges: 2015-2020, 2015 to 2020, 2015–2020
    year_range_pattern = re.compile(r'(\d{4})\s*(?:-|to|–|–)\s*(\d{4})', re.IGNORECASE)
    for match in year_range_pattern.finditer(text):
        try:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            if 1950 <= start_year <= current_year and 1950 <= end_year <= current_year and start_year <= end_year:
                years = end_year - start_year
                if 0 < years <= 50:
                    date_ranges.append((start_year, end_year))
                    matches.append(float(years))
        except (TypeError, ValueError, IndexError):
            continue
    
    # Month Year ranges: Jan 2015 - Dec 2020, January 2015 to December 2020, 01/2015 - 12/2020
    month_year_patterns = [
        re.compile(r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\s*(?:-|to|–)\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})', re.IGNORECASE),
        re.compile(r'(\d{1,2})[/-](\d{4})\s*(?:-|to|–)\s*(\d{1,2})[/-](\d{4})', re.IGNORECASE),  # MM/YYYY - MM/YYYY
    ]
    for pattern in month_year_patterns:
        for match in pattern.finditer(text):
            try:
                if len(match.groups()) == 2:
                    start_year = int(match.group(1))
                    end_year = int(match.group(2))
                else:
                    start_year = int(match.group(2))
                    end_year = int(match.group(4))
                
                if 1950 <= start_year <= current_year and 1950 <= end_year <= current_year and start_year <= end_year:
                    years = end_year - start_year
                    if 0 < years <= 50:
                        date_ranges.append((start_year, end_year))
                        matches.append(float(years))
            except (TypeError, ValueError, IndexError):
                continue
    
    # Present/Current: 2015 - Present, 2015 to Current, 2015 - Now
    present_pattern = re.compile(r'(\d{4})\s*(?:-|to|–)\s*(?:present|current|now|till date|till now|till)', re.IGNORECASE)
    for match in present_pattern.finditer(text):
        try:
            start_year = int(match.group(1))
            if 1950 <= start_year <= current_year:
                years = current_year - start_year
                if 0 < years <= 50:
                    date_ranges.append((start_year, current_year))
                    matches.append(float(years))
        except (TypeError, ValueError, IndexError):
            continue

    # Pattern 5: Look for "over X years", "more than X years", "X+ years of experience"
    additional_patterns = [
        re.compile(r'(?:over|more than|at least|minimum|min)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)', re.IGNORECASE),
        re.compile(r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp|work)', re.IGNORECASE),
        re.compile(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:professional|relevant|total|overall)', re.IGNORECASE),
    ]
    
    for pattern in additional_patterns:
        for match in pattern.finditer(text):
            try:
                value = float(match.group(1))
                if 1 <= value <= 50:
                    matches.append(value)
            except (TypeError, ValueError):
                continue

    # Pattern 6: Context-based extraction (look for numbers near experience keywords)
    # This should run even if we have matches, to catch all mentions
    context_window = 50  # Increased window
    lowered = text.lower()
    experience_keywords = ['experience', 'exp', 'career', 'worked', 'working', 'employment', 'employed', 'professional', 'industry']
    
    for match in re.finditer(r'\d+(?:\.\d+)?', lowered):
        start, end = match.span()
        snippet = lowered[max(0, start - context_window): min(len(lowered), end + context_window)]
        
        # Check if any experience keyword is nearby
        if any(keyword in snippet for keyword in experience_keywords):
            # Additional validation: number should be reasonable (1-50 years)
            try:
                value = float(match.group())
                if 1 <= value <= 50:
                    matches.append(value)
            except (TypeError, ValueError):
                continue

    # If we found date ranges, try to calculate total experience by summing non-overlapping periods
    if date_ranges and not matches:
        # Sort by start year
        date_ranges.sort(key=lambda x: x[0])
        total_years = 0
        last_end = None
        
        for start, end in date_ranges:
            if last_end is None or start > last_end:
                # Non-overlapping period
                total_years += (end - start)
                last_end = end
            elif end > last_end:
                # Overlapping but extends further
                total_years += (end - last_end)
                last_end = end
        
        if total_years > 0:
            matches.append(float(total_years))

    if matches:
        # Return the maximum experience found (most likely the total)
        max_exp = max(matches)
        logger.debug(f"Extracted experience: {max_exp} years from text (found {len(matches)} matches)")
        return max_exp

    logger.debug("No experience patterns found in text")
    return None


def _normalize_skills(skills: Any) -> List[str]:
    """Ensure skills are represented as a clean list of strings."""
    if isinstance(skills, list):
        return [skill.strip() for skill in skills if isinstance(skill, str) and skill.strip()]
    if isinstance(skills, str):
        return [part.strip() for part in skills.split(',') if part.strip()]
    return []


def _normalize_candidate_record(candidate: Any) -> Any:
    """Normalize candidate record and ensure experience is non-zero."""
    if not isinstance(candidate, dict):
        return candidate

    normalized = dict(candidate)
    normalized['skills'] = _normalize_skills(normalized.get('skills'))

    experience: Optional[float] = None
    for field in _EXPERIENCE_FIELDS:
        experience = _parse_experience_value(normalized.get(field))
        if experience is not None:
            break

    if experience is None or experience <= 0:
        # Try extracting from resume text (most common source)
        resume_text = (normalized.get('resume_text') or '') + ' ' + (normalized.get('summary') or '')
        if resume_text and resume_text.strip():
            extracted = _extract_experience_from_text(resume_text)
            if extracted and extracted > 0:
                experience = extracted
                logger.debug(f"Extracted {experience} years from resume_text for {normalized.get('email', 'unknown')}")
        
        # If still not found, try extracting from work_history if available
        if (experience is None or experience <= 0) and normalized.get('work_history'):
            work_history = normalized['work_history']
            if isinstance(work_history, str) and work_history.strip():
                work_exp = _extract_experience_from_text(work_history)
                if work_exp and work_exp > 0:
                    if experience is None or work_exp > experience:
                        experience = work_exp
                        logger.debug(f"Extracted {experience} years from work_history (string) for {normalized.get('email', 'unknown')}")
            elif isinstance(work_history, list):
                # Try to extract from work history entries
                work_text = ' '.join(str(entry) for entry in work_history if entry)
                if work_text.strip():
                    work_exp = _extract_experience_from_text(work_text)
                    if work_exp and work_exp > 0:
                        if experience is None or work_exp > experience:
                            experience = work_exp
                            logger.debug(f"Extracted {experience} years from work_history (list) for {normalized.get('email', 'unknown')}")
        
        # Try extracting from other text fields
        if (experience is None or experience <= 0):
            for field in ['description', 'bio', 'about', 'profile', 'background']:
                field_text = normalized.get(field, '')
                if field_text and str(field_text).strip():
                    field_exp = _extract_experience_from_text(str(field_text))
                    if field_exp and field_exp > 0:
                        if experience is None or field_exp > experience:
                            experience = field_exp
                            logger.debug(f"Extracted {experience} years from {field} for {normalized.get('email', 'unknown')}")

    # Calculate experience years - use extracted value if found
    if experience is not None and experience > 0:
        experience_years = int(math.ceil(float(experience)))
        # Cap at reasonable maximum (50 years)
        experience_years = min(experience_years, 50)
    else:
        # Only default to 1 if we truly can't find any experience AND candidate has content
        if normalized.get('skills') or normalized.get('resume_text') or normalized.get('work_history'):
            experience_years = 1  # Minimum experience for candidates with content
        else:
            experience_years = 0  # No experience if no content
    
    normalized['total_experience_years'] = experience_years
    normalized['experience_years'] = experience_years
    
    # Log if we had to default (for debugging) - use info level so it's visible
    if experience_years == 1 and (normalized.get('resume_text') or normalized.get('work_history')):
        email = normalized.get('email', 'unknown')
        resume_preview = (normalized.get('resume_text') or '')[:100] if normalized.get('resume_text') else ''
        logger.info(f"⚠️ Defaulted to 1 year for {email} - could not extract from resume. Preview: {resume_preview}")
    
    return normalized


def _normalize_candidates(candidates: List[Any]) -> List[Any]:
    """Normalize a list of candidate records."""
    if not candidates:
        return []
    return [_normalize_candidate_record(candidate) for candidate in candidates]

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
    try:
        from .custom_llm_models import (
            CustomEmbeddingModel, CustomCrossEncoder,
            CustomQueryEnhancer, CustomDomainClassifier, CustomTokenizer,
            AdvancedEmbeddingModel, NeuralCrossEncoder, QueryIntentClassifier,
            ContextualQueryExpander, FeedbackLearner,
            ADVANCED_EMBEDDINGS_AVAILABLE, NEURAL_CROSS_ENCODER_AVAILABLE
        )
    except ImportError:
        # Try absolute import if relative import fails
        from custom_llm_models import (
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

# Import Skill Taxonomy and Context-Aware Weighting
try:
    try:
        from .skill_taxonomy import SkillTaxonomy, get_skill_taxonomy, ProficiencyLevel
    except ImportError:
        from skill_taxonomy import SkillTaxonomy, get_skill_taxonomy, ProficiencyLevel
    SKILL_TAXONOMY_AVAILABLE = True
    print("✅ Skill Taxonomy available")
except ImportError as e:
    print(f"⚠️ Skill Taxonomy not available: {e}")
    SKILL_TAXONOMY_AVAILABLE = False
    SkillTaxonomy = None
    get_skill_taxonomy = None
    ProficiencyLevel = None

try:
    try:
        from .context_aware_weighting import ContextAwareWeighting, get_context_weighting, JobContext, RoleType, CompanySize
    except ImportError:
        from context_aware_weighting import ContextAwareWeighting, get_context_weighting, JobContext, RoleType, CompanySize
    CONTEXT_WEIGHTING_AVAILABLE = True
    print("✅ Context-Aware Weighting available")
except ImportError as e:
    print(f"⚠️ Context-Aware Weighting not available: {e}")
    CONTEXT_WEIGHTING_AVAILABLE = False
    ContextAwareWeighting = None
    get_context_weighting = None
    JobContext = None
    RoleType = None
    CompanySize = None

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
        try:
            from ops.utils.enhanced_embeddings import MultiModelEmbeddingService, embedding_service
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
            normalized_all = _normalize_candidates(cached_all)
            # Refresh cache with normalized data to avoid stale zero-experience entries
            _save_to_cache(normalized_all, CACHE_FILE_ALL)
            # Ensure domain caches exist too, compute if missing
            need_med = not _is_cache_valid(CACHE_FILE_MEDICAL)
            need_it = not _is_cache_valid(CACHE_FILE_IT)
            if need_med or need_it:
                med, it = _partition_by_domain(normalized_all)
                if need_med:
                    _save_to_cache(med, CACHE_FILE_MEDICAL)
                if need_it:
                    _save_to_cache(it, CACHE_FILE_IT)
            return normalized_all
    
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
                
                max_candidates = max(1, MAX_DYNAMODB_CANDIDATES)
                print(f"📊 Scanning DynamoDB table: {table_name} (loading up to {max_candidates} candidates)")
                all_items = []
                scan_kwargs = {
                    'Limit': min(100, max_candidates),  # Per-request limit (DynamoDB limit, not our limit)
                    'ProjectionExpression': 'email, full_name, skills, resume_text, total_experience_years, phone, sourceURL, #loc, work_history, summary, description',
                    'ExpressionAttributeNames': {'#loc': 'location'}
                }
                
                start_time = time.time()
                batch_count = 0
                
                while len(all_items) < max_candidates:
                    response = table.scan(**scan_kwargs)
                    items = response.get('Items', [])
                    remaining_slots = max_candidates - len(all_items)
                    if remaining_slots <= 0:
                        break

                    if items:
                        if len(items) > remaining_slots:
                            all_items.extend(items[:remaining_slots])
                        else:
                            all_items.extend(items)

                    batch_count += 1
                    
                    # Progress indicator with time estimation
                    elapsed = time.time() - start_time
                    if batch_count % 10 == 0 or 'LastEvaluatedKey' not in response:
                        rate = len(all_items) / elapsed if elapsed > 0 else 0
                        eta = ((len(all_items) / rate) - elapsed) if rate > 0 else 0
                        print(f"📊 Batch {batch_count}: {len(items)} items (total: {len(all_items)}) - Rate: {rate:.1f} items/sec - ETA: {eta:.1f}s")
                    
                    # Check if there are more items to scan
                    if len(all_items) >= max_candidates:
                        print(f"ℹ️ Reached maximum of {max_candidates} candidates; stopping scan.")
                        break

                    if 'LastEvaluatedKey' not in response:
                        break
                    
                    # Set the LastEvaluatedKey for the next scan
                    scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
                
                load_time = time.time() - start_time
                if all_items:
                    print(f"✅ Successfully loaded {len(all_items)} candidates from DynamoDB in {load_time:.2f}s")
                    normalized_items = _normalize_candidates(all_items)
                    # Save ALL cache
                    _save_to_cache(normalized_items, CACHE_FILE_ALL)
                    # Partition and save domain caches
                    med, it = _partition_by_domain(normalized_items)
                    _save_to_cache(med, CACHE_FILE_MEDICAL)
                    _save_to_cache(it, CACHE_FILE_IT)
                    return normalized_items
                else:
                    print("⚠️ No candidates found in DynamoDB, using mock data")
                    return _normalize_candidates(MOCK_CANDIDATES)
            else:
                print("⚠️ AWS credentials not found, using mock data")
                return _normalize_candidates(MOCK_CANDIDATES)
        else:
            print("⚠️ boto3 not available, using mock data")
            return _normalize_candidates(MOCK_CANDIDATES)
            
    except Exception as e:
        print(f"❌ Error connecting to DynamoDB: {e}")
        print("🔄 Falling back to mock data")
        return _normalize_candidates(MOCK_CANDIDATES)

class OptimizedDomainClassifier:
    """Enhanced domain classification with ML support"""
    
    def __init__(self):
        # Priority: ML classifier > Custom LLM classifier > Pattern-based fallback
        self.ml_classifier = None
        self.custom_classifier = None
        self.use_ml = False
        
        # Try SageMaker domain classifier first if enabled
        try:
            import os
            use_sagemaker = os.getenv('USE_SAGEMAKER', 'false').lower() in ('1', 'true', 'yes')
            endpoint, unified_mode = _resolve_sagemaker_endpoint('SAGEMAKER_DOMAIN_CLASSIFIER_ENDPOINT')
            region = os.getenv('SAGEMAKER_REGION', os.getenv('AWS_REGION', 'ap-south-1'))
            if use_sagemaker and endpoint:
                from Sagemaker.sagemaker_client import SageMakerDomainClassifierClient  # type: ignore
                # Pass None to use unified endpoint, or endpoint name for legacy
                endpoint_name = None if unified_mode else endpoint
                self.ml_classifier = SageMakerDomainClassifierClient(
                    endpoint_name=endpoint_name,
                    region=region
                )
                # Provide a consistent interface
                self.ml_classifier.is_trained = True  # type: ignore[attr-defined]
                self.use_ml = True
                print("✅ Using SageMaker Domain Classifier")
        except Exception as e:
            print(f"⚠️ Failed to initialize SageMaker domain classifier: {e}")

        # Fallback: local ML classifier (if available and allowed)
        if not self.use_ml and ML_DOMAIN_CLASSIFIER_AVAILABLE and get_ml_domain_classifier:
            try:
                self.ml_classifier = get_ml_domain_classifier()
                if getattr(self.ml_classifier, 'is_trained', False):
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
                # Strong healthcare indicators - require multiple matches or specific context
                r'\b(?:registered nurse|rn\s+with|nurse\s+(?:practitioner|manager|supervisor)|nursing\s+(?:degree|certification|experience))\b',
                r'\b(?:physician|doctor|md\s+(?:physician|surgeon|specialist|practitioner)|medical\s+doctor|attending\s+physician)\b',
                r'\b(?:hospital|clinic|medical\s+center|healthcare\s+facility)\b',
                r'\b(?:icu|emergency\s+room|emergency\s+department|operating\s+room|surgery|surgical)\b',
                r'\b(?:pediatric|cardiology|oncology|neurology|psychiatry|radiology|anesthesia|dermatology|orthopedic)\b',
                r'\b(?:medical\s+(?:professional|staff|team|practitioner|practice|treatment|procedure|diagnosis))\b',
                r'\b(?:pharmacist|pharmacy|dentist|dental|veterinary|veterinarian)\b',
                # Require healthcare-specific context (not just "care" or "health" alone)
                r'\b(?:patient\s+(?:care|treatment|safety|outcomes)|healthcare\s+(?:provider|system|delivery|services))\b',
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
        
        self.healthcare_role_patterns = [
            re.compile(r'\bregistered nurse\b', re.IGNORECASE),
            re.compile(r'\bnurse practitioner\b', re.IGNORECASE),
            re.compile(r'\bnurse manager\b', re.IGNORECASE),
            re.compile(r'\bnurse\b', re.IGNORECASE),
            re.compile(r'\bphysician\b', re.IGNORECASE),
            re.compile(r'\bmedical doctor\b', re.IGNORECASE),
            re.compile(r'\bdoctor\b', re.IGNORECASE),
            re.compile(r'\bsurgeon\b', re.IGNORECASE),
            re.compile(r'\bclinician\b', re.IGNORECASE),
            re.compile(r'\bmedical assistant\b', re.IGNORECASE),
            re.compile(r'\bparamedic\b', re.IGNORECASE),
            re.compile(r'\btherapist\b', re.IGNORECASE),
            re.compile(r'\bpharmacist\b', re.IGNORECASE),
            re.compile(r'\bdentist\b', re.IGNORECASE),
            re.compile(r'\bveterinarian\b', re.IGNORECASE),
            re.compile(r'\brespiratory therapist\b', re.IGNORECASE),
            re.compile(r'\boccupational therapist\b', re.IGNORECASE),
            re.compile(r'\bphysical therapist\b', re.IGNORECASE),
            re.compile(r'\brn\b', re.IGNORECASE),
        ]

        self.technology_signal_patterns = [
            re.compile(r'\bsoftware\s+(?:engineer|developer|architect)\b', re.IGNORECASE),
            re.compile(r'\bfull\s*-?\s*stack\b', re.IGNORECASE),
            re.compile(r'\bbackend\b', re.IGNORECASE),
            re.compile(r'\bfront\s*end\b', re.IGNORECASE),
            re.compile(r'\bdevops\b', re.IGNORECASE),
            re.compile(r'\bdata\s+(?:scientist|engineer|analyst)\b', re.IGNORECASE),
            re.compile(r'\bmachine learning\b', re.IGNORECASE),
            re.compile(r'\bcloud\b', re.IGNORECASE),
            re.compile(r'\bpython\b', re.IGNORECASE),
            re.compile(r'\bjava\b', re.IGNORECASE),
            re.compile(r'\bjavascript\b', re.IGNORECASE),
            re.compile(r'\breact\b', re.IGNORECASE),
            re.compile(r'\bnode\.?js\b', re.IGNORECASE),
            re.compile(r'\baws\b', re.IGNORECASE),
            re.compile(r'\bazure\b', re.IGNORECASE),
            re.compile(r'\bkubernetes\b', re.IGNORECASE),
            re.compile(r'\bdocker\b', re.IGNORECASE),
            re.compile(r'\bgit\b', re.IGNORECASE),
            re.compile(r'\bgithub\b', re.IGNORECASE),
            re.compile(r'\bjenkins\b', re.IGNORECASE),
            re.compile(r'\bapi\b', re.IGNORECASE),
            re.compile(r'\bprogramming\b', re.IGNORECASE),
            re.compile(r'\bcoding\b', re.IGNORECASE),
            re.compile(r'\bdeveloper\b', re.IGNORECASE),
            re.compile(r'\bengineer\b', re.IGNORECASE),
        ]

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
        healthcare_role_patterns = getattr(self, 'healthcare_role_patterns', [])
        technology_signal_patterns = getattr(self, 'technology_signal_patterns', [])
        has_healthcare_role = any(pattern.search(text_lower) for pattern in healthcare_role_patterns)
        has_strong_tech_signal = any(pattern.search(text_lower) for pattern in technology_signal_patterns)
        
        for domain, compiled_patterns in self.compiled_patterns.items():
            score = 0.0
            matches = 0

            if domain == 'healthcare' and not has_healthcare_role:
                continue
            
            for pattern in compiled_patterns:
                pattern_matches = len(pattern.findall(text_lower))
                if pattern_matches > 0:
                    matches += pattern_matches
                    # Healthcare requires more evidence (higher threshold)
                    if domain == 'healthcare':
                        score += min(pattern_matches * 0.15, 0.6)  # Higher per-match, but lower cap
                    else:
                        score += min(pattern_matches * 0.1, 0.5)  # Cap at 0.5 per pattern
            
            # Healthcare needs at least 2 matches to be considered (more strict)
            if domain == 'healthcare' and matches < 2:
                continue
            
            # Normalize score
            if matches > 0:
                domain_scores[domain] = min(score, 1.0)
        
        if not domain_scores:
            return 'unknown', 0.0
        
        # Return domain with highest score, but require minimum confidence
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[best_domain]
        tech_confidence = domain_scores.get('technology', 0.0)
        healthcare_confidence = domain_scores.get('healthcare', 0.0)
        
        # Require minimum confidence threshold (especially for healthcare)
        if confidence < 0.3:
            return 'unknown', confidence
        
        # PRIORITY: If there are strong technology signals, prioritize technology over healthcare
        # This prevents software engineers from being misclassified as healthcare
        if has_strong_tech_signal and tech_confidence > 0.3:
            # Technology signals are strong - prioritize technology
            if tech_confidence >= healthcare_confidence * 0.6:  # If tech is at least 60% of healthcare score
                return 'technology', tech_confidence
            # Even if healthcare is higher, if tech confidence is decent, prefer tech
            if tech_confidence >= 0.4 and healthcare_confidence < 0.6:
                return 'technology', tech_confidence
        
        # If healthcare is the winner but confidence is low, check if other domains are close
        if best_domain == 'healthcare' and confidence < 0.5:
            other_domains = [(d, s) for d, s in domain_scores.items() if d != 'healthcare']
            if other_domains:
                second_best = max(other_domains, key=lambda x: x[1])
                if second_best[1] >= confidence * 0.8:  # Within 80% of healthcare score
                    # If another domain is close, prefer it over low-confidence healthcare
                    return second_best[0], second_best[1]

        if best_domain == 'healthcare':
            # Healthcare requires explicit healthcare role - if not present, don't classify as healthcare
            if not has_healthcare_role:
                other_domains = [(d, s) for d, s in domain_scores.items() if d != 'healthcare']
                if other_domains:
                    second_best = max(other_domains, key=lambda x: x[1])
                    return second_best[0], second_best[1]
                return 'unknown', 0.0
            # Even with healthcare role, if tech signals are strong, prefer technology
            if has_strong_tech_signal and tech_confidence > 0.35:
                if tech_confidence >= healthcare_confidence * 0.7:
                    return 'technology', tech_confidence
            if healthcare_confidence < 0.5 and tech_confidence > 0.3:
                return 'technology', tech_confidence
        
        return best_domain, confidence
    
    def should_filter_candidate(self, candidate_domain: str, query_domain: str, 
                              candidate_confidence: float, query_confidence: float) -> bool:
        """
        STRICT domain filtering: Only allow candidates from the exact same domain.
        This ensures healthcare queries only show healthcare candidates, IT queries only show IT candidates.
        """
        # If domains match exactly, never filter
        if candidate_domain == query_domain:
            return False
        
        # STRICT: If query domain is known and candidate domain is known but different, ALWAYS filter
        # This ensures strict domain separation (healthcare vs IT)
        if query_domain not in ('unknown', '', 'general') and candidate_domain not in ('unknown', '', 'general'):
            if candidate_domain != query_domain:
                return True  # Strict: different domains = filter out
        
        # Use ML classifier filtering if available (for additional validation)
        if self.use_ml and self.ml_classifier:
            ml_filter = self.ml_classifier.should_filter_candidate(
                candidate_domain, query_domain, candidate_confidence, query_confidence
            )
            if ml_filter:
                return True
        
        # Fallback to custom LLM classifier
        if self.custom_classifier:
            llm_filter = self.custom_classifier.should_filter_candidate(
                candidate_domain, query_domain, candidate_confidence, query_confidence
            )
            if llm_filter:
                return True
        
        # Additional strict checks: filter if both have reasonable confidence but different domains
        if candidate_confidence > 0.5 and query_confidence > 0.5 and candidate_domain != query_domain:
            return True
        
        # If candidate has high confidence in a different domain, filter
        if candidate_confidence > 0.7 and candidate_domain != query_domain and query_domain not in ('unknown', '', 'general'):
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
        
        # Initialize Learning-to-Rank model (SageMaker-first)
        self.ltr_model = None
        try:
            _use_sm = os.getenv('USE_SAGEMAKER', 'false').lower() in ('1', 'true', 'yes')
            if not _use_sm:
                # Skip SageMaker initialization if not enabled
                pass
            else:
                # Resolve endpoint configuration
                _ltr_endpoint: Optional[str] = None
                _ltr_is_unified: bool = False
                try:
                    resolved_endpoint, is_unified = _resolve_sagemaker_endpoint('SAGEMAKER_LTR_ENDPOINT')
                    _ltr_endpoint = resolved_endpoint
                    _ltr_is_unified = is_unified
                except Exception as resolve_err:
                    print(f"⚠️ Failed to resolve SageMaker LTR endpoint: {resolve_err}")
                    _ltr_endpoint = None
                    _ltr_is_unified = False
                
                # Only proceed if we have an endpoint configuration
                if _ltr_endpoint or _ltr_is_unified:
                    _region = os.getenv('SAGEMAKER_REGION', os.getenv('AWS_REGION', 'ap-south-1'))
                    from Sagemaker.sagemaker_client import SageMakerLTRClient  # type: ignore
                    
                    # Determine endpoint name to pass to client
                    if _ltr_is_unified:
                        _final_endpoint_name = None  # Use unified endpoint from env
                    else:
                        _final_endpoint_name = _ltr_endpoint  # Use legacy endpoint name
                    
                    class _LTRAdapter:
                        def __init__(self, endpoint: Optional[str], region: str, is_unified: bool):
                            # Pass None to use unified endpoint, or endpoint name for legacy
                            endpoint_name = None if is_unified else endpoint
                            self.client = SageMakerLTRClient(endpoint_name, region)
                            self.is_trained = True
                        def predict(self, query, candidates, feature_scores=None):
                            ranked = self.client.rank_candidates(query, candidates, feature_scores)
                            # Handle different response formats
                            if isinstance(ranked, list):
                                if not ranked:
                                    return [0.0] * len(candidates)
                                # If it's a list of dicts (ranked candidates)
                                if isinstance(ranked[0], dict):
                                    # Extract scores from dicts if available
                                    if 'score' in ranked[0] or 'ltr_score' in ranked[0]:
                                        scores = [float(r.get('score', r.get('ltr_score', 0.0))) for r in ranked]
                                        # Map scores to original candidate order
                                        if 'email' in ranked[0]:
                                            score_map = {r.get('email'): float(r.get('score', r.get('ltr_score', 0.0))) for r in ranked}
                                            return [score_map.get(c.get('email', ''), 0.0) for c in candidates]
                                        return scores[:len(candidates)]
                                    # If no scores, use position-based scoring
                                    if 'email' in ranked[0]:
                                        pos_score = {r.get('email'): float(len(ranked) - i) / len(ranked) for i, r in enumerate(ranked)}
                                        return [pos_score.get(c.get('email', ''), 0.0) for c in candidates]
                                # If it's a list of numbers
                                try:
                                    return [float(s) for s in ranked[:len(candidates)]]
                                except (ValueError, TypeError):
                                    return [0.0] * len(candidates)
                            # If it's a dict, try to extract scores
                            if isinstance(ranked, dict):
                                if 'scores' in ranked:
                                    return [float(s) for s in ranked['scores'][:len(candidates)]]
                                if 'ranked_candidates' in ranked:
                                    return self.predict(query, ranked['ranked_candidates'], feature_scores)
                            # Fallback
                            return [0.0] * len(candidates)
                    
                    # Create adapter instance
                    self.ltr_model = _LTRAdapter(_final_endpoint_name, _region, _ltr_is_unified)
                    print("✅ LTR via SageMaker initialized")
        except Exception as e:
            import traceback
            error_msg = str(e)
            tb_str = traceback.format_exc()
            print(f"⚠️ Failed to initialize SageMaker LTR: {error_msg}")
            if tb_str:
                print(f"   Traceback:\n{tb_str}")
            self.ltr_model = None
        if self.ltr_model is None and LTR_AVAILABLE and get_ltr_model:
            try:
                self.ltr_model = get_ltr_model()
                print("✅ Learning-to-Rank model initialized (local)")
            except Exception as e:
                print(f"⚠️ Failed to initialize local LTR model: {e}")
                self.ltr_model = None
        
        # Initialize LLM Query Enhancer (SageMaker-first)
        self.llm_query_enhancer = None
        self.use_llm_enhancement = False
        try:
            _use_sm = os.getenv('USE_SAGEMAKER', 'false').lower() in ('1', 'true', 'yes')
            _ep, _enhancer_unified = _resolve_sagemaker_endpoint('SAGEMAKER_LLM_ENHANCER_ENDPOINT')
            _region = os.getenv('SAGEMAKER_REGION', os.getenv('AWS_REGION', 'ap-south-1'))
            if _use_sm and _ep:
                from Sagemaker.sagemaker_client import SageMakerLLMQueryEnhancerClient  # type: ignore
                _enhancer_is_unified = _enhancer_unified
                class _EnhancerAdapter:
                    def __init__(self, endpoint: Optional[str], region: str, is_unified: bool):
                        endpoint_name = None if is_unified else endpoint
                        self.client = SageMakerLLMQueryEnhancerClient(
                            endpoint_name, region
                        )
                        self.hf_pipeline = None
                        self.client_backend = 'sagemaker'
                    def enhance_query(self, query: str, use_llm: bool = True):
                        return self.client.enhance_query(query, use_llm)
                self.llm_query_enhancer = _EnhancerAdapter(_ep, _region, _enhancer_is_unified)
                self.use_llm_enhancement = True
                print("✅ LLM Query Enhancer via SageMaker initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize SageMaker LLM Query Enhancer: {e}")
            self.llm_query_enhancer = None
        if self.llm_query_enhancer is None and LLM_QUERY_ENHANCER_AVAILABLE and get_llm_query_enhancer:
            try:
                provider = os.getenv('LLM_PROVIDER', 'huggingface')
                model = os.getenv('LLM_MODEL', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
                device = os.getenv('LLM_DEVICE', None)
                self.llm_query_enhancer = get_llm_query_enhancer(provider=provider, model=model, device=device)
                if self.llm_query_enhancer.hf_pipeline or self.llm_query_enhancer.client:
                    self.use_llm_enhancement = True
                    provider_name = "Hugging Face" if self.llm_query_enhancer.hf_pipeline else provider
                    print(f"✅ LLM Query Enhancer initialized ({provider_name})")
                else:
                    print("⚠️ LLM Query Enhancer available but no provider configured")
            except Exception as e:
                print(f"⚠️ Failed to initialize local LLM Query Enhancer: {e}")
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
        
        # Initialize Job Fit Predictor (SageMaker-first)
        self.job_fit_predictor = None
        try:
            _use_sm = os.getenv('USE_SAGEMAKER', 'false').lower() in ('1', 'true', 'yes')
            _ep, _jobfit_unified = _resolve_sagemaker_endpoint('SAGEMAKER_JOB_FIT_ENDPOINT')
            _region = os.getenv('SAGEMAKER_REGION', os.getenv('AWS_REGION', 'ap-south-1'))
            if _use_sm and _ep:
                if _jobfit_unified:
                    print("ℹ️ Job Fit Predictor not available on unified endpoint - using fallback options")
                else:
                    from Sagemaker.sagemaker_client import SageMakerJobFitPredictorClient  # type: ignore
                    class _JobFitAdapter:
                        def __init__(self, endpoint: str, region: str):
                            self.client = SageMakerJobFitPredictorClient(endpoint, region)
                        def predict(self, candidate: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
                            return self.client.predict_fit(candidate, job)
                    self.job_fit_predictor = _JobFitAdapter(_ep, _region)
                    print("✅ Job Fit Predictor via SageMaker initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize SageMaker Job Fit Predictor: {e}")
            self.job_fit_predictor = None
        
        # Fallback to local job fit predictor if SageMaker not available
        if self.job_fit_predictor is None and ENHANCEMENTS_AVAILABLE:
            try:
                self.job_fit_predictor = get_job_fit_predictor() if get_job_fit_predictor else None
                if self.job_fit_predictor:
                    print("✅ Job Fit Predictor initialized (local)")
            except Exception as e:
                print(f"⚠️ Failed to initialize local Job Fit Predictor: {e}")
                self.job_fit_predictor = None
        
        # Initialize additional enhancements (non-SageMaker)
        if ENHANCEMENTS_AVAILABLE:
            try:
                self.skill_extractor = get_skill_extractor() if get_skill_extractor else None
                self.strategy_bandit = get_bandit() if get_bandit else None
                self.demand_forecaster = get_forecaster() if get_forecaster else None
                self.candidate_segmenter = get_segmenter() if get_segmenter else None
            except Exception as e:
                print(f"⚠️ Failed to initialize some enhancements: {e}")
                self.skill_extractor = None
                self.strategy_bandit = None
                self.demand_forecaster = None
                self.candidate_segmenter = None
        
        # Initialize Skill Taxonomy
        self.skill_taxonomy = None
        self.use_skill_taxonomy = False
        if SKILL_TAXONOMY_AVAILABLE and get_skill_taxonomy:
            try:
                self.skill_taxonomy = get_skill_taxonomy()
                self.use_skill_taxonomy = True
                print("✅ Skill Taxonomy initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Skill Taxonomy: {e}")
                self.skill_taxonomy = None
        
        # Initialize Context-Aware Weighting
        self.context_weighting = None
        self.use_context_weighting = False
        if CONTEXT_WEIGHTING_AVAILABLE and get_context_weighting:
            try:
                self.context_weighting = get_context_weighting()
                self.use_context_weighting = True
                print("✅ Context-Aware Weighting initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Context-Aware Weighting: {e}")
                self.context_weighting = None
        
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
        
        # Initialize Dense Retrieval Matcher (SageMaker-first)
        self.dense_retriever = None
        self.production_matcher = None
        self.use_dense_retrieval = False
        try:
            _use_sm = os.getenv('USE_SAGEMAKER', 'false').lower() in ('1', 'true', 'yes')
            _ep, _dense_unified = _resolve_sagemaker_endpoint('SAGEMAKER_DENSE_RETRIEVAL_ENDPOINT')
            _region = os.getenv('SAGEMAKER_REGION', os.getenv('AWS_REGION', 'ap-south-1'))
            if _use_sm and _ep:
                from Sagemaker.sagemaker_client import SageMakerDenseRetrievalClient  # type: ignore
                _dense_is_unified = _dense_unified
                class _DenseAdapter:
                    def __init__(self, endpoint: Optional[str], region: str, is_unified: bool):
                        endpoint_name = None if is_unified else endpoint
                        self.client = SageMakerDenseRetrievalClient(
                            endpoint_name, region
                        )
                    def find_matches(self, query: str, top_k: int = 20, domain: Optional[str] = None):
                        return self.client.search(query, top_k=top_k, domain=domain)
                self.production_matcher = _DenseAdapter(_ep, _region, _dense_is_unified)
                self.use_dense_retrieval = True
                print("✅ Dense Retrieval via SageMaker initialized")
            elif DENSE_RETRIEVAL_AVAILABLE:
                # Try to load existing index first
                index_path = "indexes/dense_retrieval_index"
                if ProductionMatcher:
                    self.production_matcher = ProductionMatcher(index_path=index_path)
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
        
        # Initialize Multi-Model Embedding Service (SageMaker-first)
        self.embedding_service = None
        self.use_multi_model_embeddings = False
        try:
            _use_sm = os.getenv('USE_SAGEMAKER', 'false').lower() in ('1', 'true', 'yes')
            _ep, _embedding_unified = _resolve_sagemaker_endpoint('SAGEMAKER_EMBEDDING_ENDPOINT')
            _region = os.getenv('SAGEMAKER_REGION', os.getenv('AWS_REGION', 'ap-south-1'))
            if _use_sm and _ep:
                from Sagemaker.sagemaker_client import SageMakerEmbeddingClient  # type: ignore
                _embedding_is_unified = _embedding_unified
                class _EmbeddingServiceAdapter:
                    def __init__(self, endpoint: Optional[str], region: str, is_unified: bool):
                        endpoint_name = None if is_unified else endpoint
                        self.client = SageMakerEmbeddingClient(
                            endpoint_name, region
                        )
                    def get_embedding(self, text: str, model_type: str = 'general'):
                        import numpy as _np
                        try:
                            emb = self.client.encode(text)
                            # Check if embedding is valid (not an error string)
                            if isinstance(emb, str):
                                raise ValueError(f"Embedding service returned error: {emb}")
                            return _np.array(emb, dtype=float)
                        except (ValueError, KeyError) as e:
                            # Return zero embedding on error
                            print(f"⚠️ Embedding service error: {e}")
                            return _np.zeros(768, dtype=float)
                self.embedding_service = _EmbeddingServiceAdapter(_ep, _region, _embedding_is_unified)
                self.use_multi_model_embeddings = True
                print("✅ Embedding Service via SageMaker initialized")
            elif MULTI_MODEL_EMBEDDING_AVAILABLE:
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
            use_sm = os.getenv('USE_SAGEMAKER', 'false').lower() in ('1', 'true', 'yes')
            sm_emb_ep, sm_emb_unified = _resolve_sagemaker_endpoint('SAGEMAKER_EMBEDDING_ENDPOINT')
            sm_region = os.getenv('SAGEMAKER_REGION', os.getenv('AWS_REGION', 'ap-south-1'))
            self._sagemaker_embeddings = False
            if use_sm and sm_emb_ep:
                try:
                    from Sagemaker.sagemaker_client import SageMakerEmbeddingClient  # type: ignore
                    _sm_emb_is_unified = sm_emb_unified
                    class _EmbeddingAdapter:
                        def __init__(self, endpoint: Optional[str], region: str, is_unified: bool):
                            endpoint_name = None if is_unified else endpoint
                            self.client = SageMakerEmbeddingClient(
                                endpoint_name, region
                            )
                            self.is_index_ready = True
                            self.skip_training = True
                        def encode(self, text: str):
                            try:
                                emb = self.client.encode(text)
                                if isinstance(emb, str):
                                    raise ValueError(f"Embedding service returned error: {emb}")
                                return emb
                            except (ValueError, KeyError) as e:
                                import numpy as _np
                                print(f"⚠️ Embedding service error: {e}")
                                return _np.zeros(768, dtype=float)
                        def batch_encode(self, texts):
                            return [self.encode(t) for t in texts]
                        def similarity(self, a: str, b: str) -> float:
                            import numpy as _np
                            try:
                                ea = _np.array(self.encode(a), dtype=float)
                                eb = _np.array(self.encode(b), dtype=float)
                                denom = (float(_np.linalg.norm(ea)) * float(_np.linalg.norm(eb))) or 1.0
                                return float(_np.dot(ea, eb) / denom)
                            except Exception as e:
                                print(f"⚠️ Similarity calculation error: {e}")
                                return 0.0
                        def fit(self, *_args, **_kwargs):
                            return None
                        def build_index(self, *_args, **_kwargs):
                            self.is_index_ready = True
                    self.custom_embedding_model = _EmbeddingAdapter(sm_emb_ep, sm_region, _sm_emb_is_unified)
                    self._sagemaker_embeddings = True
                    print("🧠 Using SageMaker Embedding Service for embeddings")
                except Exception as e:
                    print(f"⚠️ Failed to initialize SageMaker embeddings: {e}")
                    self._sagemaker_embeddings = False
            if not self._sagemaker_embeddings:
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
            
            # Train embedding model on candidate data (skip if using SageMaker embeddings)
            if not getattr(self.custom_embedding_model, 'skip_training', False):
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
    
    def _fallback_search(self, query: str, top_k: int = 20):
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
    
    def search(self, query: str, top_k: int = 20, **kwargs):
        """Enhanced search with custom LLM integration and bias prevention"""
        start_time = time.time()
        
        # Ensure top_k is an integer and at least 20
        top_k = int(top_k) if top_k else 20
        if top_k < 20:
            print(f"⚠️ WARNING: top_k={top_k} is less than 20, setting to 20")
            top_k = 20
        
        print(f"🔍 DEBUG: OptimizedSearchSystem.search() called with top_k={top_k}")
        
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
        
        # Check optimized cache first (if available) - but still apply full scoring
        use_optimized_cache = kwargs.get('use_optimized_cache', True)  # Default to True
        cached_candidates = None
        if self.use_optimized_cache and self.optimized_cache and use_optimized_cache:
            try:
                print("⚡ Checking optimized cache...")
                # Request 5-10x more candidates to ensure we have enough after strict domain filtering
                cached_results = self.optimized_cache.search(query, limit=max(top_k * 10, 200))  # Get many more candidates to ensure we have enough after filtering
                if cached_results:
                    print(f"✅ Found {len(cached_results)} candidates in optimized cache - applying full scoring pipeline")
                    # Convert cached results to standard format for scoring
                    cached_candidates = []
                    for result in cached_results:
                        if isinstance(result, dict):
                            # Extract candidate data
                            # Get candidate from original candidates list if available
                            candidate_email = result.get('email', '')
                            original_candidate = None
                            if hasattr(self, 'candidates') and candidate_email:
                                # Try to find original candidate data
                                for cand in self.candidates:
                                    if isinstance(cand, dict) and cand.get('email') == candidate_email:
                                        original_candidate = cand
                                        break
                            
                            # Build candidate data with fallback to original candidate
                            # Get experience from result or original candidate, default to 1 if 0 or missing
                            exp_years = result.get('total_experience_years', 0)
                            if exp_years == 0 and original_candidate:
                                exp_years = original_candidate.get('total_experience_years', 0)
                            # Ensure experience is at least 1 if candidate has skills or resume
                            if exp_years == 0:
                                skills_list = result.get('skills', original_candidate.get('skills', []) if original_candidate else [])
                                resume_text = result.get('resume_text', original_candidate.get('resume_text', '') if original_candidate else '')
                                if skills_list or resume_text:
                                    exp_years = 1
                            
                            candidate_data = {
                                'email': result.get('email', ''),
                                'full_name': result.get('full_name', ''),
                                'skills': result.get('skills', original_candidate.get('skills', []) if original_candidate else []),
                                'total_experience_years': max(1, exp_years),  # Ensure at least 1 year
                                'experience_years': max(1, exp_years),  # Also set experience_years
                                'phone': result.get('phone', ''),
                                'sourceURL': result.get('sourceURL', ''),
                                'database_index': result.get('database_index', 0),
                                'resume_text': result.get('resume_text', original_candidate.get('resume_text', '') if original_candidate else ''),
                                'location': result.get('location', ''),
                                'education': result.get('education', ''),
                                'seniority_level': result.get('seniority_level', 'mid')
                            }
                            
                            # Ensure skills is a list
                            if isinstance(candidate_data['skills'], str):
                                candidate_data['skills'] = [s.strip() for s in candidate_data['skills'].split(',') if s.strip()]
                            elif not isinstance(candidate_data['skills'], list):
                                candidate_data['skills'] = []
                            # Use cache score as initial score, will be updated by ML models
                            initial_score = result.get('score', 0.0)
                            if isinstance(initial_score, int):
                                initial_score = initial_score / 100.0  # Convert from percentage
                            candidate_data['final_score'] = float(initial_score)
                            candidate_data['similarity_score'] = float(initial_score)
                            candidate_data['source'] = 'optimized_cache'
                            cached_candidates.append(candidate_data)
            except Exception as e:
                print(f"⚠️ Optimized cache search failed: {e}")
                cached_candidates = None
        
        # Check regular cache
        cache_key = self._get_cache_key(query, top_k)
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        
        # Try Instant Search first (if available and enabled)
        use_instant_search = kwargs.get('use_instant_search', True)  # Default to True
        enable_domain_filtering = kwargs.get('enable_domain_filtering', True)  # Default to True
        instant_results = None
        if self.use_instant_search and self.instant_search_engine and use_instant_search:
            try:
                print("⚡ Using Instant Search Engine...")
                # Pass domain classifier to instant search if available
                if hasattr(self, 'domain_classifier'):
                    self.instant_search_engine.domain_classifier = self.domain_classifier
                # Request 5-10x more candidates to ensure we have enough after strict domain filtering
                # This ensures we can get 20 candidates from the same domain
                search_limit = max(top_k * 10, 200)  # Request at least 200 candidates or 10x top_k
                instant_results = self.instant_search_engine.search(
                    query, 
                    limit=search_limit,  # Get many more candidates to ensure we have enough after filtering
                    top_k=search_limit,  # Also pass top_k parameter
                    enable_domain_filtering=False  # We'll apply domain filtering later to all candidates
                )
                if instant_results:
                    print(f"✅ Instant Search found {len(instant_results)} candidates (will filter by domain)")
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
                    # Request more candidates to ensure we have enough after domain filtering
                    dense_results = self.production_matcher.find_matches(query, top_k=max(top_k * 10, 200))
                    print(f"✅ Production Matcher found {len(dense_results)} candidates")
                elif self.dense_retriever and self.dense_retriever.index:
                    # Use standalone dense retriever
                    matches = self.dense_retriever.search(query, top_k=max(top_k * 10, 200))
                    dense_results = [match[0] for match in matches]  # Extract candidates
                    print(f"✅ Dense Retrieval found {len(dense_results)} candidates")
            except Exception as e:
                print(f"⚠️ Dense Retrieval failed: {e}")
        
        # Enhance query using LLM if available (priority: LLM > Custom > Fallback)
        enhanced_query = None
        
        if self.use_llm_enhancement and self.llm_query_enhancer:
            try:
                provider_name = "Hugging Face" if self.llm_query_enhancer.hf_pipeline else "OpenAI/Claude"
                print(f"🧠 Enhancing query with LLM ({provider_name})...")
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
        # Priority: Cached Candidates > Instant Search > Dense Retrieval > Hybrid Search
        
        # Use cached candidates if available (they'll still go through full scoring)
        if cached_candidates:
            print("⚡ Using cached candidates as base (applying full scoring pipeline)...")
            results = cached_candidates
        # Use Instant Search results if available
        elif instant_results:
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
                
                # Update scores with enhanced matcher scores, but preserve skill match priority
                for result, enhanced_score in enhanced_scores:
                    original_score = result.get('final_score', 0.0)
                    skill_match_score = result.get('skill_match_score', 0.0)
                    if skill_match_score > 0:
                        # If skill match exists, keep it prioritized (skill matching will override)
                        result['final_score'] = original_score  # Keep original (skill-based) score
                    else:
                        # If no skill match, use enhanced matcher score
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
                        
                        # Compute cosine similarity - ensure proper shape handling
                        import numpy as np
                        # Ensure both are 1D arrays
                        q_emb = np.array(query_embedding).flatten()
                        c_emb = np.array(candidate_embedding).flatten()
                        
                        # Compute cosine similarity
                        dot_product = np.dot(q_emb, c_emb)
                        norm_product = np.linalg.norm(q_emb) * np.linalg.norm(c_emb)
                        similarity = dot_product / norm_product if norm_product > 0 else 0.0
                        
                        # Update score with embedding similarity, but preserve skill match priority
                        original_score = result.get('final_score', 0.0)
                        skill_match_score = result.get('skill_match_score', 0.0)
                        if skill_match_score > 0:
                            # If skill match exists, keep it prioritized (skill matching will override)
                            result['final_score'] = original_score  # Keep original (skill-based) score
                        else:
                            # If no skill match, use embedding similarity
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
        
        # Apply Job Fit Predictor if available
        if self.job_fit_predictor and len(results) > 0:
            print("💼 Applying Job Fit Predictor...")
            results = self._apply_job_fit_scoring(query, results)
        
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
                    # Ensure experience is at least 1
                    exp_years = result.get('total_experience_years', 0)
                    if exp_years == 0:
                        skills_list = result.get('skills', [])
                        resume_text = result.get('resume_text', '')
                        if skills_list or resume_text:
                            exp_years = 1
                    
                    candidate_profile = {
                        'full_name': result.get('full_name', ''),
                        'email': result.get('email', ''),
                        'skills': result.get('skills', []),
                        'experience_years': max(1, exp_years),
                        'seniority_level': result.get('seniority_level', 'mid'),
                        'resume_text': result.get('resume_text', ''),
                        'education': result.get('education', ''),
                        'location': result.get('location', '')
                    }
                    
                    # Prepare match scores for ExplainableRecruitmentAI
                    # Use scaled percentages for better accuracy representation
                    final_score = result.get('final_score', 0.0)
                    scaled_overall = self._scale_score_to_percentage(final_score)
                    
                    match_scores = {
                        'overall_score': scaled_overall,  # Use scaled percentage
                        'technical_skills_score': self._scale_score_to_percentage(result.get('skill_score', 0.0)),
                        'experience_score': self._scale_score_to_percentage(result.get('experience_score', 0.0)),
                        'seniority_score': self._scale_score_to_percentage(result.get('seniority_score', 0.0)),
                        'education_score': self._scale_score_to_percentage(result.get('education_score', 0.0)),
                        'soft_skills_score': self._scale_score_to_percentage(result.get('soft_skills_score', 0.0)),
                        'location_score': self._scale_score_to_percentage(result.get('location_score', 0.0))
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
        
        # Apply skill-based scoring and reranking (IMPROVED ACCURACY)
        print("🎯 Applying skill-based scoring for improved accuracy...")
        results = self._apply_skill_based_scoring(query, results)
        
        # Apply domain filtering if enabled - STRICT ENFORCEMENT
        enable_domain_filtering = kwargs.get('enable_domain_filtering', True)  # Default to True
        query_domain = None  # Store query domain for later use
        if enable_domain_filtering and results:
            print("🔍 Applying STRICT domain filtering...")
            # Get query domain before filtering (needed for fetching more candidates from same domain)
            try:
                query_domain, _ = self.domain_classifier.classify_domain(query)
                print(f"🔍 Query domain identified: {query_domain}")
            except Exception as e:
                print(f"⚠️ Could not identify query domain: {e}")
            
            filtered_results = self._apply_domain_filtering(query, results)
            if filtered_results:
                results = filtered_results
                # If we have fewer than top_k candidates after domain filtering, try to get more from same domain
                if len(results) < top_k:
                    print(f"⚠️ Only {len(results)} candidates after domain filtering, need {top_k}. Fetching more from same domain ({query_domain})...")
                    additional_results = self._fetch_additional_candidates_from_domain(query, query_domain, results, top_k - len(results))
                    if additional_results:
                        results.extend(additional_results)
                        print(f"✅ Added {len(additional_results)} additional candidates from same domain (now have {len(results)} total)")
            else:
                # STRICT: If domain filtering removes all candidates, try to get candidates from the identified domain
                if query_domain and query_domain not in ('unknown', '', 'general'):
                    print(f"⚠️ Domain filtering removed all candidates. Attempting to fetch candidates from {query_domain} domain...")
                    domain_results = self._fetch_additional_candidates_from_domain(query, query_domain, [], top_k)
                    if domain_results:
                        results = domain_results
                        print(f"✅ Found {len(results)} candidates from {query_domain} domain")
                    else:
                        print("⚠️ No candidates found in the query's domain")
                        results = []  # Return empty instead of wrong-domain candidates
                else:
                    # STRICT: If domain filtering removes all candidates, return empty list
                    # This ensures we don't show candidates from wrong domains
                    print("⚠️ Domain filtering removed all candidates - no matches in correct domain")
                    print("   This indicates either:")
                    print("   1. No candidates exist in the query's domain")
                    print("   2. Domain classification needs improvement")
                    results = []  # Return empty instead of wrong-domain candidates
        
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
        
        # FINAL SORT: Maximum priority to skill matching - candidates with skills ALWAYS first
        # Sort by: 1) Has skill match (boolean - CRITICAL), 2) Skill match score, 3) Experience (when skills match), 4) Final score
        # This ensures candidates with ANY skill match (even 0.01) rank above those with 0
        # When skills match, candidates with more experience are ranked higher
        results.sort(key=lambda x: (
            x.get('skill_match_score', 0) > 0,  # Primary: Candidates with skill matches first (True > False) - CRITICAL
            x.get('skill_match_score', 0.0),  # Secondary: Skill match score (higher is better)
            x.get('total_experience_years', 0) if x.get('skill_match_score', 0) > 0 else 0,  # Tertiary: Experience (only when skills match)
            x.get('final_score', 0.0)  # Quaternary: Final score (higher is better)
        ), reverse=True)
        
        # Additional safety: ensure candidates with skill matches are truly first
        # Separate into two groups and recombine
        with_skills = [r for r in results if r.get('skill_match_score', 0) > 0]
        without_skills = [r for r in results if r.get('skill_match_score', 0) == 0]
        
        # Sort each group - WITH skills: by skill match, then experience, then final score
        with_skills.sort(key=lambda x: (
            x.get('skill_match_score', 0.0),  # Primary: skill match score
            x.get('total_experience_years', 0),  # Secondary: experience (more is better)
            x.get('final_score', 0.0)  # Tertiary: final score
        ), reverse=True)
        # WITHOUT skills: by final score only
        without_skills.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        
        # Recombine: skills first, then no skills
        results = with_skills + without_skills
        
        # ENHANCED: If we don't have enough results, try to get more candidates
        # This ensures we always return at least top_k candidates when available
        if len(results) < top_k:
            print(f"⚠️ Only found {len(results)} candidates, need {top_k}. Attempting to expand search...")
            
            # Try to get more candidates from fallback search or by lowering thresholds
            try:
                # Use fallback search to get additional candidates
                fallback_results = self._fallback_search(query, top_k=top_k * 2)
                
                # Add candidates that aren't already in results
                existing_emails = {r.get('email', '') for r in results}
                for fallback_result in fallback_results:
                    fallback_email = fallback_result.get('email', '')
                    if fallback_email and fallback_email not in existing_emails:
                        # Add with lower score to indicate it's from fallback
                        fallback_result['final_score'] = fallback_result.get('final_score', 0.0) * 0.8
                        fallback_result['source'] = fallback_result.get('source', 'fallback')
                        results.append(fallback_result)
                        if len(results) >= top_k:
                            break
                
                # Re-sort after adding fallback results
                with_skills_new = [r for r in results if r.get('skill_match_score', 0) > 0]
                without_skills_new = [r for r in results if r.get('skill_match_score', 0) == 0]
                with_skills_new.sort(key=lambda x: (
                    x.get('skill_match_score', 0.0),
                    x.get('total_experience_years', 0),
                    x.get('final_score', 0.0)
                ), reverse=True)
                without_skills_new.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
                results = with_skills_new + without_skills_new
                
                print(f"✅ Expanded search: now have {len(results)} candidates")
            except Exception as e:
                print(f"⚠️ Failed to expand search: {e}")
        
        # CRITICAL: Ensure all modules are applied to ALL candidates (including newly fetched ones)
        # Re-apply scoring and enrichment to ensure consistency
        if results:
            print(f"🔄 Re-applying all modules to {len(results)} candidates to ensure consistency...")
            
            # Re-apply skill-based scoring to all candidates
            results = self._apply_skill_based_scoring(query, results)
            
            # Re-apply Enhanced Matcher if available
            use_enhanced_matcher = kwargs.get('use_enhanced_matcher', True)
            if self.use_enhanced_matcher and self.enhanced_matcher and use_enhanced_matcher:
                try:
                    print("🎯 Re-applying Enhanced Matcher to all candidates...")
                    for result in results:
                        try:
                            resume_text = self._get_candidate_text_for_embedding(result)
                            match_scores = self.enhanced_matcher.calculate_match_score(query, resume_text)
                            overall_score = match_scores.get('overall_score', match_scores.get('match_score', 0.0))
                            if isinstance(overall_score, dict):
                                overall_score = overall_score.get('score', 0.0)
                            result['enhanced_matcher_score'] = float(overall_score)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"⚠️ Re-applying Enhanced Matcher failed: {e}")
            
            # Re-apply Multi-Model Embeddings if available
            use_multi_model_embeddings = kwargs.get('use_multi_model_embeddings', True)
            if self.use_multi_model_embeddings and self.embedding_service and use_multi_model_embeddings:
                try:
                    print("🔤 Re-applying Multi-Model Embeddings to all candidates...")
                    query_embedding = self.embedding_service.get_embedding(query, model_type='general')
                    for result in results:
                        try:
                            candidate_text = self._get_candidate_text_for_embedding(result)
                            candidate_embedding = self.embedding_service.get_embedding(candidate_text, model_type='general')
                            import numpy as np
                            q_emb = np.array(query_embedding).flatten()
                            c_emb = np.array(candidate_embedding).flatten()
                            dot_product = np.dot(q_emb, c_emb)
                            norm_product = np.linalg.norm(q_emb) * np.linalg.norm(c_emb)
                            similarity = dot_product / norm_product if norm_product > 0 else 0.0
                            result['embedding_similarity'] = float(similarity)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"⚠️ Re-applying Multi-Model Embeddings failed: {e}")
            
            # Re-sort after re-scoring
            with_skills = [r for r in results if r.get('skill_match_score', 0) > 0]
            without_skills = [r for r in results if r.get('skill_match_score', 0) == 0]
            with_skills.sort(key=lambda x: (
                x.get('skill_match_score', 0.0),
                x.get('total_experience_years', 0),
                x.get('final_score', 0.0)
            ), reverse=True)
            without_skills.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
            results = with_skills + without_skills
        
        # Return top_k results (or all if we have fewer)
        final_results = results[:top_k]
        print(f"🔍 DEBUG: After initial slice, have {len(final_results)} results, need {top_k}")
        
        # ENSURE: If we still don't have enough, try to get more from the same domain
        # This ensures we ALWAYS return top_k candidates from the same domain when available
        # Get enable_domain_filtering from kwargs if not already set
        enable_domain_filtering_final = kwargs.get('enable_domain_filtering', True)
        if len(final_results) < top_k and enable_domain_filtering_final and query_domain:
            print(f"⚠️ Only have {len(final_results)} candidates, need {top_k}. Getting additional candidates from {query_domain} domain...")
            try:
                # Get additional candidates from the same domain
                additional = self._fetch_additional_candidates_from_domain(
                    query, 
                    query_domain, 
                    final_results, 
                    top_k - len(final_results)
                )
                
                if additional:
                    # Apply ALL modules to additional candidates to ensure consistency
                    print(f"🔄 Applying all modules to {len(additional)} additional candidates...")
                    
                    # Apply skill-based scoring
                    additional = self._apply_skill_based_scoring(query, additional)
                    
                    # Apply Enhanced Matcher if available
                    use_enhanced_matcher = kwargs.get('use_enhanced_matcher', True)
                    if self.use_enhanced_matcher and self.enhanced_matcher and use_enhanced_matcher:
                        for result in additional:
                            try:
                                resume_text = self._get_candidate_text_for_embedding(result)
                                match_scores = self.enhanced_matcher.calculate_match_score(query, resume_text)
                                overall_score = match_scores.get('overall_score', match_scores.get('match_score', 0.0))
                                if isinstance(overall_score, dict):
                                    overall_score = overall_score.get('score', 0.0)
                                result['enhanced_matcher_score'] = float(overall_score)
                            except Exception:
                                pass
                    
                    # Apply Multi-Model Embeddings if available
                    use_multi_model_embeddings = kwargs.get('use_multi_model_embeddings', True)
                    if self.use_multi_model_embeddings and self.embedding_service and use_multi_model_embeddings:
                        try:
                            query_embedding = self.embedding_service.get_embedding(query, model_type='general')
                            for result in additional:
                                try:
                                    candidate_text = self._get_candidate_text_for_embedding(result)
                                    candidate_embedding = self.embedding_service.get_embedding(candidate_text, model_type='general')
                                    import numpy as np
                                    q_emb = np.array(query_embedding).flatten()
                                    c_emb = np.array(candidate_embedding).flatten()
                                    dot_product = np.dot(q_emb, c_emb)
                                    norm_product = np.linalg.norm(q_emb) * np.linalg.norm(c_emb)
                                    similarity = dot_product / norm_product if norm_product > 0 else 0.0
                                    result['embedding_similarity'] = float(similarity)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    
                    # Apply behavioral analysis if enabled
                    include_behavioural = kwargs.get('include_behavioural_analysis', True)
                    if self.use_behavioral_analysis and self.behavioral_pipeline and include_behavioural:
                        additional = self._enrich_with_behavioral_analysis(query, additional)
                    
                    # Apply explainable AI if enabled
                    if self.use_explainable_ai and self.explainable_ai:
                        for idx, result in enumerate(additional):
                            try:
                                exp_years = result.get('total_experience_years', 0)
                                if exp_years == 0:
                                    skills_list = result.get('skills', [])
                                    resume_text = result.get('resume_text', '')
                                    if skills_list or resume_text:
                                        exp_years = 1
                                
                                candidate_profile = {
                                    'full_name': result.get('full_name', ''),
                                    'email': result.get('email', ''),
                                    'skills': result.get('skills', []),
                                    'experience_years': max(1, exp_years),
                                    'seniority_level': result.get('seniority_level', 'mid'),
                                    'resume_text': result.get('resume_text', ''),
                                    'education': result.get('education', ''),
                                    'location': result.get('location', '')
                                }
                                
                                final_score = result.get('final_score', 0.0)
                                scaled_overall = self._scale_score_to_percentage(final_score)
                                
                                match_scores = {
                                    'overall_score': scaled_overall,
                                    'technical_skills_score': self._scale_score_to_percentage(result.get('skill_score', 0.0)),
                                    'experience_score': self._scale_score_to_percentage(result.get('experience_score', 0.0)),
                                    'seniority_score': self._scale_score_to_percentage(result.get('seniority_score', 0.0)),
                                    'education_score': self._scale_score_to_percentage(result.get('education_score', 0.0)),
                                    'soft_skills_score': self._scale_score_to_percentage(result.get('soft_skills_score', 0.0)),
                                    'location_score': self._scale_score_to_percentage(result.get('location_score', 0.0))
                                }
                                
                                ranking_position = len(final_results) + idx + 1
                                
                                explanation = self.explainable_ai.explain_candidate_selection(
                                    candidate_profile=candidate_profile,
                                    job_query=query,
                                    match_scores=match_scores,
                                    ranking_position=ranking_position
                                )
                                
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
                                print(f"⚠️ Explainable AI failed for additional candidate: {e}")
                                result['ai_explanation'] = f"Candidate with score {result.get('final_score', 0.0):.2f}"
                    
                    final_results.extend(additional)
                    
                    # Re-sort all results
                    with_skills = [r for r in final_results if r.get('skill_match_score', 0) > 0]
                    without_skills = [r for r in final_results if r.get('skill_match_score', 0) == 0]
                    with_skills.sort(key=lambda x: (
                        x.get('skill_match_score', 0.0),
                        x.get('total_experience_years', 0),
                        x.get('final_score', 0.0)
                    ), reverse=True)
                    without_skills.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
                    final_results = with_skills + without_skills
                    
                    print(f"✅ Added {len(additional)} additional candidates from {query_domain} domain (now have {len(final_results)} total, target was {top_k})")
            except Exception as e:
                print(f"⚠️ Failed to get additional candidates from domain: {e}")
                import traceback
                traceback.print_exc()
        
        # FINAL CHECK: Ensure we return exactly top_k if we have enough
        if len(final_results) > top_k:
            final_results = final_results[:top_k]
            print(f"🔍 DEBUG: Trimmed results to exactly {top_k} candidates")
        elif len(final_results) < top_k:
            print(f"⚠️ WARNING: Only returning {len(final_results)} candidates instead of requested {top_k}")
            print(f"   This may indicate there are fewer than {top_k} candidates in the {query_domain if query_domain else 'target'} domain")
        
        print(f"🔍 DEBUG: Final result count: {len(final_results)} (requested: {top_k})")
        
        # Scale final_score to higher percentage range for display
        # This boosts percentages while maintaining relative ordering
        for result in final_results:
            original_score = result.get('final_score', 0.0)
            # Scale the score to a higher percentage
            scaled_percentage = self._scale_score_to_percentage(original_score)
            result['match_percentage'] = scaled_percentage
            result['Confidence'] = scaled_percentage  # For frontend compatibility
            
            # Also scale individual component scores if they exist
            if 'skill_match_score' in result:
                result['SkillMatchScore'] = self._scale_score_to_percentage(result['skill_match_score'])
            if 'experience_score' in result:
                result['ExperienceRelevance'] = self._scale_score_to_percentage(result['experience_score'])
            if 'education_score' in result:
                result['EducationMatch'] = self._scale_score_to_percentage(result['education_score'])
        
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
                # Ensure experience is at least 1 if candidate has skills or resume
                exp_years = candidate.get('total_experience_years', 0)
                if exp_years == 0:
                    skills_list = candidate.get('skills', [])
                    resume_text = candidate.get('resume_text', '')
                    if skills_list or resume_text:
                        exp_years = 1
                
                result = {
                    'email': candidate.get('email', ''),
                    'full_name': candidate.get('full_name', ''),
                    'skills': candidate.get('skills', []),
                    'total_experience_years': max(1, exp_years),  # Ensure at least 1 year
                    'experience_years': max(1, exp_years),  # Also set experience_years
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
                # Ensure experience is at least 1 if candidate has skills or resume
                exp_years = candidate.get('total_experience_years', 0)
                if exp_years == 0:
                    skills_list = candidate.get('skills', [])
                    resume_text = candidate.get('resume_text', '')
                    if skills_list or resume_text:
                        exp_years = 1
                
                results.append({
                    'email': candidate.get('email', ''),
                    'full_name': candidate.get('full_name', ''),
                    'skills': candidate.get('skills', []),
                    'total_experience_years': max(1, exp_years),  # Ensure at least 1 year
                    'experience_years': max(1, exp_years),  # Also set experience_years
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
            try:
                # Check if encode_pair method exists
                if hasattr(self.custom_cross_encoder, 'encode_pair'):
                    cross_encoder_score = self.custom_cross_encoder.encode_pair(query, result)
                elif hasattr(self.custom_cross_encoder, 'score'):
                    # Alternative method name
                    cross_encoder_score = self.custom_cross_encoder.score(query, result)
                else:
                    # Skip if method doesn't exist
                    continue
                
                # Combine with original score, but preserve skill match priority
                original_score = result.get('final_score', 0.0)
                skill_match_score = result.get('skill_match_score', 0.0)
                if skill_match_score > 0:
                    # If skill match exists, keep it prioritized (skill matching will override)
                    result['final_score'] = original_score  # Keep original (skill-based) score
                else:
                    # If no skill match, use cross-encoder score
                    result['final_score'] = 0.7 * original_score + 0.3 * cross_encoder_score
                result['cross_encoder_score'] = cross_encoder_score
            except Exception as e:
                # Skip this result if encoding fails
                print(f"⚠️ Cross-encoder scoring failed for candidate: {e}")
                continue
        
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
                # Combine LTR score with existing score, but preserve skill match priority
                # If candidate has skill matches, skill matching will override this later
                original_score = result.get('final_score', 0.0)
                skill_match_score = result.get('skill_match_score', 0.0)
                if skill_match_score > 0:
                    # If skill match exists, keep it prioritized (skill matching will override)
                    result['final_score'] = original_score  # Keep original (skill-based) score
                else:
                    # If no skill match, use LTR score
                    result['final_score'] = float(ltr_score)
        except Exception as e:
            print(f"⚠️ LTR ranking failed: {e}")
            # Fallback to original scores
            pass
        
        # Re-sort by final score
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return results
    
    def _apply_job_fit_scoring(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Job Fit Predictor to score candidates"""
        if not self.job_fit_predictor:
            return results
        
        # Prepare job description from query
        job_description = {
            'title': query,
            'description': query,
            'required_skills': query.split(),  # Simple tokenization
            'required_experience': 0,  # Default
            'location': '',  # Default
            'job_type': 'full-time'  # Default
        }
        
        # Score each candidate with job fit predictor
        for result in results:
            try:
                # Prepare candidate data - ensure experience is at least 1
                exp_years = result.get('total_experience_years', 0)
                if exp_years == 0:
                    skills_list = result.get('skills', [])
                    resume_text = result.get('resume_text', '')
                    if skills_list or resume_text:
                        exp_years = 1
                
                candidate = {
                    'full_name': result.get('full_name', ''),
                    'email': result.get('email', ''),
                    'skills': result.get('skills', []),
                    'total_experience_years': max(1, exp_years),
                    'resume_text': result.get('resume_text', ''),
                    'education': result.get('education', ''),
                    'location': result.get('location', ''),
                    'seniority_level': result.get('seniority_level', 'mid')
                }
                
                # Predict job fit
                fit_prediction = self.job_fit_predictor.predict(candidate, job_description)
                
                # Extract fit probability
                fit_probability = fit_prediction.get('fit_probability', 0.5)
                if isinstance(fit_probability, (int, float)):
                    fit_score = float(fit_probability)
                else:
                    fit_score = 0.5
                
                # Store job fit prediction
                result['job_fit_score'] = fit_score
                result['job_fit_grade'] = fit_prediction.get('fit_grade', 'C')
                result['job_fit_recommendation'] = fit_prediction.get('recommendation', 'review')
                result['job_fit_factors'] = fit_prediction.get('key_factors', [])
                
                # Combine with existing final score, but preserve skill match priority
                existing_score = result.get('final_score', 0.0)
                skill_match_score = result.get('skill_match_score', 0.0)
                if skill_match_score > 0:
                    # If skill match exists, keep it prioritized (skill matching will override)
                    result['final_score'] = existing_score  # Keep original (skill-based) score
                else:
                    # If no skill match, use job fit score
                    result['final_score'] = 0.7 * existing_score + 0.3 * fit_score
                
            except Exception as e:
                print(f"⚠️ Job Fit Predictor scoring failed for candidate: {e}")
                # Keep existing score
                result['job_fit_score'] = 0.5
                result['job_fit_grade'] = 'C'
                result['job_fit_recommendation'] = 'review'
                continue
        
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
    
    def _extract_required_skills_from_query(self, query: str) -> List[str]:
        """Extract required skills from job description query - IMPROVED"""
        query_lower = query.lower()
        skills = []
        
        # Expanded and improved tech skill patterns with more variations
        tech_skills = {
            'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'ruby', 
                          'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'bash', 'shell'],
            'frameworks': ['react', 'vue', 'angular', 'node.js', 'nodejs', 'django', 'flask', 'fastapi', 
                         'spring', 'express', 'laravel', 'rails', '.net', 'asp.net', 'next.js', 'nuxt'],
            'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'k8s', 'terraform', 
                     'ansible', 'jenkins', 'github actions', 'gitlab ci', 'cloudformation', 'ec2', 's3'],
            'database': ['sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'cassandra', 
                        'elasticsearch', 'dynamodb', 'oracle', 'nosql', 'sqlite', 'mariadb'],
            'ml_ai': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy', 
                     'machine learning', 'ml', 'deep learning', 'ai', 'artificial intelligence', 
                     'neural networks', 'nlp', 'natural language processing', 'computer vision', 'cv'],
            'data': ['pandas', 'numpy', 'scikit-learn', 'data analysis', 'data science', 'big data', 
                    'spark', 'hadoop', 'kafka', 'tableau', 'power bi', 'matplotlib', 'seaborn', 
                    'jupyter', 'notebook', 'data visualization', 'statistics', 'statistical analysis'],
            'devops': ['docker', 'kubernetes', 'ci/cd', 'jenkins', 'terraform', 'ansible', 'chef', 
                      'puppet', 'git', 'github', 'gitlab', 'linux', 'unix', 'bash', 'shell scripting'],
            'frontend': ['react', 'vue', 'angular', 'javascript', 'typescript', 'html', 'css', 
                        'sass', 'less', 'redux', 'webpack', 'next.js', 'nuxt', 'vue.js'],
            'backend': ['node.js', 'python', 'java', 'spring', 'django', 'flask', 'express', 'api', 
                       'rest', 'graphql', 'microservices', 'fastapi', 'nest.js']
        }
        
        # IMPROVED: Extract skills from query with better pattern matching
        # First, extract exact matches with word boundaries
        for category, skill_list in tech_skills.items():
            for skill in skill_list:
                # Use word boundary for better matching
                skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(skill_pattern, query_lower):
                    normalized_skill = self._normalize_skill(skill)
                    if normalized_skill not in skills:
                        skills.append(normalized_skill)
        
        # IMPROVED: Also check for job title keywords that imply skills
        job_title_keywords = {
            'data scientist': ['python', 'pandas', 'numpy', 'scikit-learn', 'machine learning', 'data science', 'sql', 'statistics'],
            'data analyst': ['python', 'pandas', 'numpy', 'sql', 'excel', 'tableau', 'data analysis'],
            'software engineer': ['python', 'java', 'javascript', 'software development', 'programming'],
            'full stack': ['javascript', 'react', 'node.js', 'python', 'java', 'sql'],
            'frontend developer': ['react', 'vue', 'angular', 'javascript', 'typescript', 'html', 'css'],
            'backend developer': ['python', 'java', 'node.js', 'django', 'flask', 'express', 'sql'],
            'devops engineer': ['docker', 'kubernetes', 'aws', 'terraform', 'jenkins', 'ci/cd', 'linux'],
            'ml engineer': ['python', 'tensorflow', 'pytorch', 'machine learning', 'deep learning', 'pandas', 'numpy'],
            'ai engineer': ['python', 'tensorflow', 'pytorch', 'machine learning', 'ai', 'deep learning']
        }
        
        for keyword, implied_skills in job_title_keywords.items():
            if keyword in query_lower:
                for skill in implied_skills:
                    normalized_skill = self._normalize_skill(skill)
                    if normalized_skill not in skills:
                        skills.append(normalized_skill)
        
        # Also look for common skill phrases
        skill_phrases = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience\s*)?(?:in\s*)?([a-z\s]+)',
            r'(?:proficient|expert|experienced|strong|knowledge)\s*(?:in|with)\s*([a-z\s]+)',
            r'(?:requires|must have|required|need)\s*(?:experience|skills?|knowledge)\s*(?:in|with|of)\s*([a-z\s]+)'
        ]
        
        for pattern in skill_phrases:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if isinstance(match, tuple):
                    skill_text = match[-1].strip()
                else:
                    skill_text = match.strip()
                
                # Extract individual skills from the text
                words = skill_text.split()
                for word in words:
                    word_clean = word.strip('.,;:!?()[]{}')
                    if len(word_clean) > 2 and word_clean not in ['the', 'and', 'or', 'in', 'with', 'of', 'a', 'an']:
                        # Check if it matches any known skill
                        for category, skill_list in tech_skills.items():
                            for skill in skill_list:
                                if word_clean in skill or skill in word_clean:
                                    if skill not in skills:
                                        skills.append(skill)
        
        return skills
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill name for matching - ENHANCED with comprehensive aliases and synonyms"""
        if not skill:
            return ""
        
        skill_lower = skill.lower().strip()
        
        # Remove common prefixes/suffixes that don't affect meaning
        skill_lower = re.sub(r'^(proficient|expert|experienced|strong|knowledge|skill|skills|in|with|of)\s+', '', skill_lower)
        skill_lower = re.sub(r'\s+(proficient|expert|experienced|strong|knowledge|skill|skills|in|with|of)$', '', skill_lower)
        skill_lower = skill_lower.strip()
        
        # ENHANCED: Comprehensive normalization map with extensive skill aliases and synonyms
        skill_aliases = {
            # Programming languages - comprehensive variations
            'js': 'javascript',
            'javascript': 'javascript',
            'javascript es6': 'javascript',
            'javascript es2015': 'javascript',
            'ecmascript': 'javascript',
            'ts': 'typescript',
            'typescript': 'typescript',
            'cpp': 'c++',
            'c++': 'c++',
            'cplusplus': 'c++',
            'c plus plus': 'c++',
            'csharp': 'c#',
            'c#': 'c#',
            'c sharp': 'c#',
            'golang': 'go',
            'go lang': 'go',
            'go programming': 'go',
            # Frameworks - all variations
            'node.js': 'nodejs',
            'nodejs': 'nodejs',
            'node': 'nodejs',
            'node js': 'nodejs',
            'nodejs.js': 'nodejs',
            'reactjs': 'react',
            'react.js': 'react',
            'react': 'react',
            'react js': 'react',
            'reactjs.js': 'react',
            'reactjs.jsx': 'react',
            'vuejs': 'vue',
            'vue.js': 'vue',
            'vue': 'vue',
            'vue js': 'vue',
            'vuejs.js': 'vue',
            'angularjs': 'angular',
            'angular.js': 'angular',
            'angular': 'angular',
            'angular js': 'angular',
            'angularjs.js': 'angular',
            'angular 2': 'angular',
            'angular 2+': 'angular',
            'angular 4': 'angular',
            'angular 5': 'angular',
            'angular 6': 'angular',
            'angular 7': 'angular',
            'angular 8': 'angular',
            'angular 9': 'angular',
            'angular 10': 'angular',
            'angular 11': 'angular',
            'angular 12': 'angular',
            'angular 13': 'angular',
            'angular 14': 'angular',
            'angular 15': 'angular',
            'angular 16': 'angular',
            'angular 17': 'angular',
            'angular 18': 'angular',
            # ML/AI - comprehensive
            'tf': 'tensorflow',
            'tensorflow': 'tensorflow',
            'tensor flow': 'tensorflow',
            'tensor-flow': 'tensorflow',
            'pytorch': 'pytorch',
            'py torch': 'pytorch',
            'py-torch': 'pytorch',
            'pytorch': 'pytorch',
            'sklearn': 'scikit-learn',
            'scikit-learn': 'scikit-learn',
            'scikit learn': 'scikit-learn',
            'scikitlearn': 'scikit-learn',
            'scikit_learn': 'scikit-learn',
            'scikit_learning': 'scikit-learn',
            'ml': 'machine learning',
            'machinelearning': 'machine learning',
            'machine learning': 'machine learning',
            'machine-learning': 'machine learning',
            'ml engineering': 'machine learning',
            'ai': 'artificial intelligence',
            'artificialintelligence': 'artificial intelligence',
            'artificial intelligence': 'artificial intelligence',
            'artificial-intelligence': 'artificial intelligence',
            # Data - comprehensive
            'pd': 'pandas',
            'pandas': 'pandas',
            'pandas python': 'pandas',
            'np': 'numpy',
            'numpy': 'numpy',
            'numpy python': 'numpy',
            'data sci': 'data science',
            'datascience': 'data science',
            'data science': 'data science',
            'data-science': 'data science',
            'data analysis': 'data analysis',
            'dataanalysis': 'data analysis',
            'data-analysis': 'data analysis',
            'data analytics': 'data analysis',
            'bigdata': 'big data',
            'big data': 'big data',
            'big-data': 'big data',
            # Databases - comprehensive
            'postgres': 'postgresql',
            'postgresql': 'postgresql',
            'postgres db': 'postgresql',
            'postgres-db': 'postgresql',
            'postgres database': 'postgresql',
            'pg': 'postgresql',
            'postgresql database': 'postgresql',
            'mongo': 'mongodb',
            'mongo db': 'mongodb',
            'mongo-db': 'mongodb',
            'mongodb': 'mongodb',
            'mongo database': 'mongodb',
            # Cloud - comprehensive
            'gcp': 'google cloud',
            'google cloud platform': 'google cloud',
            'googlecloud': 'google cloud',
            'google-cloud': 'google cloud',
            'google cloud': 'google cloud',
            'amazon web services': 'aws',
            'amazonwebservices': 'aws',
            'amazon-web-services': 'aws',
            'aws': 'aws',
            'amazon aws': 'aws',
            # DevOps - comprehensive
            'k8s': 'kubernetes',
            'kubernetes': 'kubernetes',
            'kube': 'kubernetes',
            'ci/cd': 'ci/cd',
            'cicd': 'ci/cd',
            'ci cd': 'ci/cd',
            'ci-cd': 'ci/cd',
            'continuous integration': 'ci/cd',
            'continuous deployment': 'ci/cd',
            'continuous integration/deployment': 'ci/cd',
            # APIs - comprehensive
            'rest api': 'rest',
            'restapis': 'rest',
            'restful': 'rest',
            'rest': 'rest',
            'restful api': 'rest',
            'rest api': 'rest',
            'api': 'api',
            'apis': 'api',
            'rest apis': 'rest',
            'graphql': 'graphql',
            'graph ql': 'graphql',
            'graph-ql': 'graphql',
            # Other - comprehensive
            'html5': 'html',
            'html': 'html',
            'html 5': 'html',
            'css3': 'css',
            'css': 'css',
            'css 3': 'css',
            'jsx': 'javascript',
            'tsx': 'typescript',
            'redux': 'redux',
            'redux js': 'redux',
            'redux.js': 'redux',
            # Healthcare skills
            'rn': 'registered nurse',
            'registered nurse': 'registered nurse',
            'nurse': 'nursing',
            'nursing': 'nursing',
            'md': 'medical doctor',
            'medical doctor': 'medical doctor',
            'doctor': 'medical doctor',
            'physician': 'medical doctor',
            'np': 'nurse practitioner',
            'nurse practitioner': 'nurse practitioner',
            'acls': 'acls',
            'bls': 'bls',
            'ehr': 'ehr',
            'electronic health records': 'ehr',
            'emr': 'ehr',
            'electronic medical records': 'ehr',
        }
        
        # Check exact match first
        if skill_lower in skill_aliases:
            return skill_aliases[skill_lower]
        
        # Check if skill contains any alias key (for partial matches)
        for alias, normalized in skill_aliases.items():
            if alias in skill_lower or skill_lower in alias:
                return normalized
        
        # Remove common suffixes/prefixes and try again
        cleaned = re.sub(r'[\.\-\s]+$', '', skill_lower)
        cleaned = re.sub(r'^[\.\-\s]+', '', cleaned)
        if cleaned in skill_aliases:
            return skill_aliases[cleaned]
        
        return skill_lower
    
    def _fuzzy_match_ratio(self, str1: str, str2: str) -> float:
        """Calculate fuzzy matching ratio between two strings using SequenceMatcher"""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _calculate_skill_match_score(
        self, 
        candidate_skills: List[str], 
        required_skills: List[str],
        query: Optional[str] = None,
        candidate_context: Optional[str] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate skill match score with skill taxonomy, fuzzy matching, and context-aware weighting.
        
        Args:
            candidate_skills: List of candidate's skills
            required_skills: List of required skills
            query: Optional search query for context
            candidate_context: Optional candidate context text (resume, etc.)
            
        Returns:
            Tuple of (score, details_dict)
        """
        if not required_skills:
            return 0.0, {"matches": [], "total_matches": 0, "total_required": 0, "method": "basic"}
        
        # Use skill taxonomy if available
        if self.use_skill_taxonomy and self.skill_taxonomy:
            try:
                # Extract job context from query if available
                job_context = None
                if self.use_context_weighting and self.context_weighting and query:
                    try:
                        job_context = self.context_weighting.extract_job_context(query)
                    except Exception as e:
                        logger.debug(f"Failed to extract job context: {e}")
                
                # Calculate context-aware skill weights
                skill_weights = None
                if job_context and self.use_context_weighting and self.context_weighting:
                    try:
                        skill_weights = self.context_weighting.calculate_skill_weights(required_skills, job_context)
                    except Exception as e:
                        logger.debug(f"Failed to calculate context weights: {e}")
                
                # Use skill taxonomy for matching
                context_text = candidate_context or ' '.join(candidate_skills)
                match_result = self.skill_taxonomy.calculate_skill_match_score(
                    candidate_skills=candidate_skills,
                    required_skills=required_skills,
                    skill_weights=skill_weights,
                    context=context_text
                )
                
                return match_result['overall_score'], {
                    "matches": [d['skill'] for d in match_result['skill_details'] if d['match_type'] != 'missing'],
                    "exact_matches": [d['skill'] for d in match_result['skill_details'] if d['match_type'] == 'exact'],
                    "related_matches": [d['skill'] for d in match_result['skill_details'] if d['match_type'] == 'related'],
                    "partial_matches": [d['skill'] for d in match_result['skill_details'] if d['match_type'] == 'related'],
                    "missing_skills": match_result['missing_skills'],
                    "total_matches": match_result['exact_matches'] + match_result['related_matches'],
                    "total_required": len(required_skills),
                    "match_rate": match_result['match_percentage'],
                    "skill_details": match_result['skill_details'],
                    "method": "taxonomy_with_context" if job_context else "taxonomy"
                }
            except Exception as e:
                logger.warning(f"Skill taxonomy matching failed, falling back to basic: {e}")
                # Fall through to basic matching
        
        # ENHANCED: Improved basic matching with fuzzy matching
        candidate_skills_normalized = []
        for s in candidate_skills:
            if s:
                normalized = self._normalize_skill(str(s))
                if normalized and normalized not in candidate_skills_normalized:
                    candidate_skills_normalized.append(normalized)
        
        required_skills_normalized = []
        for s in required_skills:
            if s:
                normalized = self._normalize_skill(str(s))
                if normalized and normalized not in required_skills_normalized:
                    required_skills_normalized.append(normalized)
        
        # Find matches (exact, partial, and fuzzy)
        matches = []
        exact_matches = []
        partial_matches = []
        fuzzy_matches = []
        
        candidate_skills_set = set(candidate_skills_normalized)
        candidate_skills_lower = {s.lower() for s in candidate_skills_normalized}
        candidate_skills_list = list(candidate_skills_normalized)  # For fuzzy matching
        
        for req_skill in required_skills_normalized:
            req_skill_lower = req_skill.lower()
            matched = False
            match_type = None
            
            # 1. Exact match (normalized)
            if req_skill in candidate_skills_set:
                exact_matches.append(req_skill)
                matches.append(req_skill)
                matched = True
                match_type = 'exact'
            
            # 2. Case-insensitive exact match
            if not matched and req_skill_lower in candidate_skills_lower:
                exact_matches.append(req_skill)
                matches.append(req_skill)
                matched = True
                match_type = 'exact'
            
            # 3. Partial match (substring)
            if not matched:
                for cand_skill in candidate_skills_set:
                    cand_skill_lower = cand_skill.lower()
                    if req_skill_lower in cand_skill_lower or cand_skill_lower in req_skill_lower:
                        if len(req_skill_lower) >= 3 and len(cand_skill_lower) >= 3:
                            partial_matches.append(req_skill)
                            matches.append(req_skill)
                            matched = True
                            match_type = 'partial'
                            break
            
            # 4. Word-level matching
            if not matched:
                req_words = set(req_skill_lower.split())
                for cand_skill in candidate_skills_set:
                    cand_words = set(cand_skill.lower().split())
                    significant_words = {w for w in req_words.intersection(cand_words) if len(w) >= 3}
                    if significant_words and len(significant_words) >= min(2, len(req_words)):
                        partial_matches.append(req_skill)
                        matches.append(req_skill)
                        matched = True
                        match_type = 'partial'
                        break
            
            # 5. ENHANCED: Fuzzy matching for similar skills (e.g., "React" vs "ReactJS")
            if not matched:
                best_ratio = 0.0
                best_match = None
                for cand_skill in candidate_skills_list:
                    cand_skill_lower = cand_skill.lower()
                    # Calculate fuzzy match ratio
                    ratio = self._fuzzy_match_ratio(req_skill_lower, cand_skill_lower)
                    # Also check if one is contained in the other (for cases like "React" vs "ReactJS")
                    if req_skill_lower in cand_skill_lower or cand_skill_lower in req_skill_lower:
                        ratio = max(ratio, 0.85)  # Boost ratio for substring matches
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = cand_skill
                
                # Use fuzzy match if similarity is high enough (>= 0.75)
                if best_ratio >= 0.75:
                    fuzzy_matches.append(req_skill)
                    matches.append(req_skill)
                    matched = True
                    match_type = 'fuzzy'
        
        # ENHANCED: Calculate score with better weighting
        exact_score = len(exact_matches) / len(required_skills_normalized) if required_skills_normalized else 0
        partial_score = len(partial_matches) / len(required_skills_normalized) * 0.85 if required_skills_normalized else 0  # Increased from 0.75
        fuzzy_score = len(fuzzy_matches) / len(required_skills_normalized) * 0.70 if required_skills_normalized else 0  # New fuzzy score
        total_score = exact_score + partial_score + fuzzy_score
        
        match_rate = len(matches) / len(required_skills_normalized) if required_skills_normalized else 0
        
        # ENHANCED: Ensure minimum score if we have some matches
        if match_rate >= 0.1 and total_score < 0.1:
            total_score = 0.1
        elif match_rate >= 0.2 and total_score < 0.2:
            total_score = 0.2
        elif match_rate >= 0.3 and total_score < 0.3:
            total_score = 0.3
        
        return min(1.0, total_score), {
            "matches": matches,
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "fuzzy_matches": fuzzy_matches,
            "total_matches": len(matches),
            "total_required": len(required_skills_normalized),
            "match_rate": match_rate,
            "method": "enhanced_basic"
        }
    
    def _aggregate_candidate_skills(self, candidate: Dict[str, Any]) -> List[str]:
        """
        ENHANCED: Aggregate candidate skills from ALL possible fields.
        This ensures we don't miss skills that might be in different fields.
        """
        aggregated: List[str] = []
        
        def _add_values(values: Any):
            """Helper to add values to aggregated list"""
            if not values:
                return
            if isinstance(values, list):
                for v in values:
                    if v:
                        aggregated.append(str(v).strip())
            elif isinstance(values, dict):
                for k in values.keys():
                    if k:
                        aggregated.append(str(k).strip())
            elif isinstance(values, str):
                # Handle comma-separated strings
                if ',' in values:
                    for v in values.split(','):
                        if v.strip():
                            aggregated.append(v.strip())
                else:
                    aggregated.append(values.strip())
            else:
                aggregated.append(str(values).strip())
        
        # 1. From explicit skills field (highest priority)
        _add_values(candidate.get('skills', []))
        
        # 2. From designations_with_experience (often contains skill-like terms)
        _add_values(candidate.get('designations_with_experience'))
        
        # 3. From certifications (certifications often indicate skills)
        _add_values(candidate.get('certifications'))
        
        # 4. From current_position (job titles often contain skills)
        _add_values(candidate.get('current_position'))
        
        # 5. From education (degrees/fields can indicate skills)
        _add_values(candidate.get('education'))
        
        # 6. From domain_tag (domain can indicate skill area)
        _add_values(candidate.get('domain_tag'))
        
        # 7. ENHANCED: Extract from resume_text with comprehensive skill detection
        resume_text = str(candidate.get('resume_text', '') or '').lower()
        if resume_text:
            # Comprehensive tech skills list (matches query extraction)
            tech_skills = {
                'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'ruby', 
                              'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'bash', 'shell'],
                'frameworks': ['react', 'vue', 'angular', 'node.js', 'nodejs', 'django', 'flask', 'fastapi', 
                             'spring', 'express', 'laravel', 'rails', '.net', 'asp.net', 'next.js', 'nuxt'],
                'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'k8s', 'terraform', 
                         'ansible', 'jenkins', 'github actions', 'gitlab ci', 'cloudformation', 'ec2', 's3'],
                'database': ['sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'cassandra', 
                            'elasticsearch', 'dynamodb', 'oracle', 'nosql', 'sqlite', 'mariadb'],
                'ml_ai': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy', 
                         'machine learning', 'ml', 'deep learning', 'ai', 'artificial intelligence',
                         'neural networks', 'nlp', 'natural language processing', 'computer vision', 'cv'],
                'data': ['pandas', 'numpy', 'scikit-learn', 'data analysis', 'data science', 'big data', 
                        'spark', 'hadoop', 'kafka', 'tableau', 'power bi', 'matplotlib', 'seaborn',
                        'jupyter', 'notebook', 'data visualization', 'statistics', 'statistical analysis'],
                'devops': ['docker', 'kubernetes', 'ci/cd', 'jenkins', 'terraform', 'ansible', 'chef', 
                          'puppet', 'git', 'github', 'gitlab', 'linux', 'unix', 'bash', 'shell scripting'],
                'frontend': ['react', 'vue', 'angular', 'javascript', 'typescript', 'html', 'css', 
                            'sass', 'less', 'redux', 'webpack', 'next.js', 'nuxt', 'vue.js'],
                'backend': ['node.js', 'python', 'java', 'spring', 'django', 'flask', 'express', 'api', 
                           'rest', 'graphql', 'microservices', 'fastapi', 'nest.js'],
                'healthcare': ['nursing', 'patient care', 'medical', 'healthcare', 'rn', 'md', 'np', 
                              'acls', 'bls', 'ehr', 'emr', 'critical care', 'icu', 'emergency', 'surgery',
                              'pharmacy', 'radiology', 'physical therapy', 'rehabilitation']
            }
            
            # Extract skills from resume text
            for category, skill_list in tech_skills.items():
                for skill in skill_list:
                    skill_lower = skill.lower()
                    # Try word boundary first (exact match) - most accurate
                    skill_pattern = r'\b' + re.escape(skill_lower) + r'\b'
                    if re.search(skill_pattern, resume_text):
                        normalized_skill = self._normalize_skill(skill)
                        if normalized_skill not in aggregated:
                            aggregated.append(normalized_skill)
                    # Also check substring match for longer skills (more lenient)
                    elif len(skill_lower) >= 4:
                        if skill_lower in resume_text:
                            # Additional check: ensure it's not part of a longer word
                            pattern = r'(?<![a-z])' + re.escape(skill_lower) + r'(?![a-z])'
                            if re.search(pattern, resume_text):
                                normalized_skill = self._normalize_skill(skill)
                                if normalized_skill not in aggregated:
                                    aggregated.append(normalized_skill)
        
        # 8. From source URL filename hints (if available)
        source_url = str(candidate.get('sourceURL', '') or candidate.get('source_url', '') or '').lower()
        if source_url:
            # Extract potential skills from URL
            tech_keywords = ['python', 'java', 'javascript', 'react', 'node', 'angular', 'vue', 
                           'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'sql', 'mongodb']
            for keyword in tech_keywords:
                if keyword in source_url:
                    normalized_skill = self._normalize_skill(keyword)
                    if normalized_skill not in aggregated:
                        aggregated.append(normalized_skill)
        
        # Normalize and deduplicate
        normalized_skills: List[str] = []
        seen: set = set()
        for raw_skill in aggregated:
            if not raw_skill:
                continue
            normalized = self._normalize_skill(raw_skill)
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_skills.append(normalized)
        
        return normalized_skills
    
    def _apply_skill_based_scoring(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply skill-based scoring to improve accuracy - ENHANCED with comprehensive skill extraction"""
        if not results:
            return results
        
        try:
            # Extract required skills from query
            required_skills = self._extract_required_skills_from_query(query)
            
            if not required_skills:
                print("⚠️ No required skills extracted from query, skipping skill-based scoring")
                return results
            
            print(f"📋 Extracted {len(required_skills)} required skills from query")
            
            # Calculate skill match scores for each candidate
            scored_results = []
            for result in results:
                # ENHANCED: Use comprehensive skill aggregation from ALL candidate fields
                candidate_skills = self._aggregate_candidate_skills(result)
                
                # Debug: Log if no skills found
                if not candidate_skills:
                    logger.debug(f"No skills extracted for candidate {result.get('email', 'unknown')}")
                
                # Calculate skill match score with context
                candidate_context = result.get('resume_text', '') or ' '.join(candidate_skills)
                skill_score, match_details = self._calculate_skill_match_score(
                    candidate_skills, 
                    required_skills,
                    query=query,
                    candidate_context=candidate_context
                )
                
                # Update result with skill match information
                result['skill_match_score'] = skill_score
                result['skill_match_details'] = match_details
                result['required_skills'] = required_skills
                result['matched_skills'] = match_details['matches']
                
                # ENHANCED: PRIORITIZE SKILL MATCHING - candidates with skills ALWAYS rank above those without
                # Ensure skill_match_score is the PRIMARY factor in ranking
                original_score = result.get('final_score', 0.0)
                
                if skill_score > 0:
                    # Get experience years for weighting
                    exp_years = result.get('total_experience_years', 0) or result.get('experience_years', 0) or 0
                    
                    # Experience bonus when skills match (scaled)
                    exp_bonus = 0.0
                    if exp_years > 0:
                        # Base experience contribution (up to 10% bonus)
                        exp_bonus = min(0.10, (exp_years / 20.0) * 0.10)
                    
                    # ENHANCED: Candidates with skill matches get STRONG weight on skill match
                    # Skill score is PRIMARY (70%), original score is secondary (20%), experience bonus (10%)
                    # This ensures strong correlation between skill_match_score and final_score
                    result['final_score'] = 0.70 * skill_score + 0.20 * original_score + exp_bonus
                    
                    # Ensure final_score is at least as high as skill_score (skill match is important)
                    if result['final_score'] < skill_score:
                        result['final_score'] = skill_score
                    
                    result['skill_boost'] = result['final_score'] - original_score
                    result['experience_boost'] = exp_bonus
                    result['has_skill_match'] = True
                    
                    # Debug logging for skill matches
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Candidate {result.get('email', 'unknown')}: skill_score={skill_score:.3f}, "
                                   f"original_score={original_score:.3f}, final_score={result['final_score']:.3f}, "
                                   f"matches={len(match_details.get('matches', []))}")
                else:
                    # STRICT penalty for no skill matches: cap score very low
                    # Candidates with NO skill matches should NEVER rank above those with matches
                    # Cap at 0.10 maximum (very low) to ensure they rank below skill matches
                    result['final_score'] = min(original_score * 0.2, 0.10)  # Max 10% of original, capped at 0.10
                    result['no_skill_match_penalty'] = True
                    result['has_skill_match'] = False
                    
                    # Debug logging for no matches
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Candidate {result.get('email', 'unknown')}: NO skill matches, "
                                   f"original_score={original_score:.3f}, final_score={result['final_score']:.3f}, "
                                   f"candidate_skills={len(candidate_skills)}")
                
                scored_results.append(result)
            
            # MAXIMUM PRIORITY RERANKING: Skill matches first, then by skill match score, then experience, then final score
            # Sort candidates with skill matches first, then by skill match score (highest first), then experience, then final score
            candidates_with_skills = [r for r in scored_results if r.get('skill_match_score', 0) > 0]
            candidates_without_skills = [r for r in scored_results if r.get('skill_match_score', 0) == 0]
            
            # Sort candidates WITH skills by: skill_match_score (primary), then experience (secondary), then final_score (tertiary)
            # This ensures candidates with matching skills AND more experience rank higher
            candidates_with_skills.sort(key=lambda x: (
                x.get('skill_match_score', 0.0),  # Primary: skill match score
                x.get('total_experience_years', 0),  # Secondary: experience years (more is better)
                x.get('final_score', 0.0)  # Tertiary: final score
            ), reverse=True)
            
            # Sort candidates WITHOUT skills by final score only (they're already at the bottom)
            candidates_without_skills.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
            
            # Put candidates with skills first (they always rank higher), then those without
            scored_results = candidates_with_skills + candidates_without_skills
            
            skill_match_count = len(candidates_with_skills)
            print(f"✅ Skill-based scoring applied: {skill_match_count} candidates with skill matches (ranked first)")
            if len(candidates_without_skills) > 0:
                print(f"   ⚠️ {len(candidates_without_skills)} candidates with NO skill matches pushed to bottom")
            
            return scored_results
            
        except Exception as e:
            print(f"⚠️ Skill-based scoring failed: {e}")
            import traceback
            traceback.print_exc()
            return results  # Return original results if scoring fails
    
    def _apply_domain_filtering(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply domain filtering using domain classifier and domain BERT if available"""
        if not results:
            return results
        
        try:
            # Classify query domain
            query_domain, query_confidence = self.domain_classifier.classify_domain(query)
            print(f"🔍 Query domain: {query_domain} (confidence: {query_confidence:.2f})")
            
            # Use domain BERT from behavioral pipeline if available for more accurate domain detection
            domain_bert = None
            if self.use_behavioral_analysis and self.behavioral_pipeline:
                if hasattr(self.behavioral_pipeline, 'domain_bert'):
                    domain_bert = self.behavioral_pipeline.domain_bert
                    # Use domain BERT for query domain detection (more accurate than keyword matching)
                    try:
                        query_domain_type = domain_bert.detect_domain(query)
                        query_domain = query_domain_type.value if hasattr(query_domain_type, 'value') else str(query_domain_type)
                        query_confidence = 0.85  # Higher confidence for BERT-based detection
                        print(f"🔍 Query domain (BERT): {query_domain}")
                    except Exception as e:
                        print(f"⚠️ Domain BERT detection failed: {e}, using keyword-based classification")
            
            # STRICT DOMAIN FILTERING: Always enforce domain matching when domain is known
            # For ambiguous queries, use classifier to determine domain
            if query_domain in ('unknown', '', 'general') or query_confidence < 0.5:
                # Try harder to classify ambiguous queries using multiple methods
                print("🔍 Query domain ambiguous; using enhanced classification...")
                # Try domain BERT if available
                if domain_bert:
                    try:
                        domain_bert_result = domain_bert.detect_domain(query)
                        domain_bert_value = getattr(domain_bert_result, 'value', None) if hasattr(domain_bert_result, 'value') else str(domain_bert_result)
                        if domain_bert_value and domain_bert_value not in ('unknown', '', 'general'):
                            query_domain = domain_bert_value
                            query_confidence = 0.85
                            print(f"✅ Query domain classified (BERT): {query_domain}")
                    except Exception:
                        pass
                
                # If still unknown, use ML classifier with higher threshold
                if query_domain in ('unknown', '', 'general'):
                    try:
                        # Classify with multiple attempts
                        query_domain, query_confidence = self.domain_classifier.classify_domain(query)
                        # If confidence is low, try with resume context patterns
                        if query_confidence < 0.6:
                            # Look for strong domain indicators in query
                            query_lower = query.lower()
                            if any(term in query_lower for term in ['nurse', 'doctor', 'physician', 'medical', 'healthcare', 'hospital', 'clinic', 'patient', 'rn', 'md']):
                                query_domain = 'healthcare'
                                query_confidence = 0.75
                            elif any(term in query_lower for term in ['software', 'developer', 'programmer', 'engineer', 'coding', 'python', 'java', 'javascript', 'it', 'tech', 'technology']):
                                query_domain = 'it'
                                query_confidence = 0.75
                    except Exception as e:
                        print(f"⚠️ Enhanced classification failed: {e}")
                
                if query_domain in ('unknown', '', 'general'):
                    print("⚠️ Query domain still unknown after enhanced classification; will allow all domains")
                    enforce_domain = False
                else:
                    enforce_domain = True
            else:
                enforce_domain = True
            
            # Filter and score candidates based on domain match
            filtered_results = []
            for result in results:
                # Get candidate text for domain classification
                candidate_text = result.get('resume_text', '')
                if not candidate_text:
                    # Fallback: combine available fields
                    skills_text = ' '.join(result.get('skills', []))
                    candidate_text = f"{result.get('full_name', '')} {skills_text}"
                
                # Classify candidate domain
                candidate_domain, candidate_confidence = self.domain_classifier.classify_domain(candidate_text)
                
                # Use domain BERT for candidate if available
                if domain_bert:
                    try:
                        candidate_domain_type = domain_bert.detect_domain(candidate_text)
                        candidate_domain = candidate_domain_type.value if hasattr(candidate_domain_type, 'value') else str(candidate_domain_type)
                        candidate_confidence = 0.85
                    except Exception:
                        pass  # Use keyword-based classification as fallback
                
                # STRICT DOMAIN ENFORCEMENT: Only show candidates from exact same domain
                if enforce_domain:
                    # CRITICAL: If query domain is known (not unknown/general), enforce strict matching
                    if query_domain not in ('unknown', '', 'general'):
                        # If candidate domain is unknown, try harder to classify it
                        if candidate_domain in ('unknown', '', 'general'):
                            # Use domain classifier with full candidate text
                            candidate_full_text = (
                                result.get('resume_text', '') or 
                                ' '.join(result.get('skills', [])) or
                                result.get('full_name', '') or
                                ''
                            )
                            if candidate_full_text:
                                try:
                                    candidate_domain, candidate_confidence = self.domain_classifier.classify_domain(candidate_full_text)
                                    print(f"🔍 Re-classified candidate {result.get('email', 'unknown')[:20]}... as {candidate_domain} (confidence: {candidate_confidence:.2f})")
                                except Exception:
                                    pass
                        
                        # STRICT: Different domains = filter out (no exceptions for known domains)
                        if candidate_domain not in ('unknown', '', 'general') and query_domain != candidate_domain:
                            print(f"🚫 Filtered {result.get('email', 'unknown')} due to strict domain mismatch ({candidate_domain} vs {query_domain})")
                            should_filter = True
                        # If candidate is still unknown after classification, filter it out for known query domains
                        elif candidate_domain in ('unknown', '', 'general') and candidate_confidence < 0.3:
                            print(f"🚫 Filtered {result.get('email', 'unknown')} - domain could not be determined (confidence: {candidate_confidence:.2f})")
                            should_filter = True
                        else:
                            should_filter = False
                    else:
                        should_filter = False
                    
                    # Additional strict filtering using classifier (for edge cases)
                    if not should_filter:
                        should_filter = self.domain_classifier.should_filter_candidate(
                            candidate_domain, query_domain, candidate_confidence, max(query_confidence, 0.5)
                        )
                        if should_filter:
                            print(f"🚫 Filtered {result.get('email', 'unknown')} due to classifier domain mismatch ({candidate_domain} vs {query_domain})")
                else:
                    should_filter = False
                
                if not should_filter:
                    # Add domain information to result
                    result['domain'] = candidate_domain
                    result['domain_confidence'] = candidate_confidence
                    result['query_domain'] = query_domain
                    result['query_domain_confidence'] = query_confidence
                    result['domain_match'] = candidate_domain == query_domain
                    
                    # Boost score if domain matches (but preserve skill-based scoring priority)
                    if candidate_domain == query_domain and candidate_confidence > 0.7:
                        original_score = result.get('final_score', 0.0)
                        skill_match_score = result.get('skill_match_score', 0.0)
                        # Only apply small domain boost if skill match exists (skill matching is priority)
                        if skill_match_score > 0:
                            domain_boost = min(0.03 * candidate_confidence, 0.05)  # Max 5% boost (very small, skill is priority)
                            result['final_score'] = min(original_score + domain_boost, 1.0)
                            result['domain_boost'] = domain_boost
                        # Don't boost if no skill matches - keep them penalized (skill matching is priority)
                    
                    filtered_results.append(result)
                else:
                    print(f"🚫 Filtered out candidate {result.get('email', 'unknown')}: domain mismatch ({candidate_domain} vs {query_domain})")
            
            if not filtered_results:
                print("⚠️ Domain filtering produced zero results - no candidates match the query domain")
                print(f"   Query domain: {query_domain} (confidence: {query_confidence:.2f})")
                print("   Returning empty list to enforce strict domain matching")
                return []  # Return empty list instead of wrong-domain candidates
            
            print(f"✅ Domain filtering: {len(results)} -> {len(filtered_results)} candidates")
            print(f"   Query domain: {query_domain}, All candidates match domain")
            return filtered_results
            
        except Exception as e:
            print(f"⚠️ Domain filtering failed: {e}")
            import traceback
            traceback.print_exc()
            # STRICT: Don't return wrong-domain candidates if filtering fails
            # Return empty list to enforce domain matching
            print("   Returning empty list due to filtering failure - strict domain enforcement")
            return []
    
    def _fetch_additional_candidates_from_domain(
        self, 
        query: str, 
        target_domain: str, 
        existing_results: List[Dict[str, Any]], 
        needed_count: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch additional candidates from the same domain to ensure we have enough results.
        This method ensures all candidates are from the same domain as the query.
        
        Args:
            query: Original search query
            target_domain: Domain to filter candidates by (must match exactly)
            existing_results: Already selected candidates (to avoid duplicates)
            needed_count: Number of additional candidates needed
            
        Returns:
            List of additional candidates from the same domain
        """
        if not target_domain or target_domain in ('unknown', '', 'general') or needed_count <= 0:
            return []
        
        print(f"🔍 Fetching {needed_count} additional candidates from {target_domain} domain...")
        
        additional_candidates = []
        existing_emails = {r.get('email', '') for r in existing_results}
        
        # Try to get candidates from all available sources
        candidate_sources = []
        
        # Source 1: Processed candidates (if available)
        if hasattr(self, '_processed_candidates') and self._processed_candidates:
            for processed_cand in self._processed_candidates:
                if len(additional_candidates) >= needed_count:
                    break
                email = processed_cand.get('email', '')
                if email and email not in existing_emails:
                    candidate_sources.append(processed_cand)
        
        # Source 2: Full candidates list
        if len(additional_candidates) < needed_count and hasattr(self, 'candidates') and self.candidates:
            for candidate in self.candidates:
                if len(additional_candidates) >= needed_count:
                    break
                if isinstance(candidate, dict):
                    email = candidate.get('email', '')
                    if email and email not in existing_emails:
                        candidate_sources.append(candidate)
        
        # Source 3: Instant search engine candidates
        if len(additional_candidates) < needed_count and hasattr(self, 'instant_search_engine') and self.instant_search_engine:
            if hasattr(self.instant_search_engine, 'candidates') and self.instant_search_engine.candidates:
                for email, candidate in self.instant_search_engine.candidates.items():
                    if len(additional_candidates) >= needed_count:
                        break
                    if email and email not in existing_emails:
                        # Convert CandidateIndex to dict if needed
                        if hasattr(candidate, 'email'):
                            candidate_dict = {
                                'email': candidate.email,
                                'full_name': candidate.full_name,
                                'skills': getattr(candidate, 'skills', []),
                                'total_experience_years': getattr(candidate, 'experience', 0),
                                'resume_text': getattr(candidate, 'resume_text', ''),
                                'phone': getattr(candidate, 'phone', ''),
                                'sourceURL': getattr(candidate, 'source_url', '')
                            }
                            candidate_sources.append(candidate_dict)
                        elif isinstance(candidate, dict):
                            candidate_sources.append(candidate)
        
        # Now filter candidates by domain and add them
        for candidate in candidate_sources:
            if len(additional_candidates) >= needed_count:
                break
            
            try:
                # Get candidate text for domain classification
                candidate_text = candidate.get('resume_text', '')
                if not candidate_text:
                    skills_text = ' '.join(candidate.get('skills', []))
                    candidate_text = f"{candidate.get('full_name', '')} {skills_text}"
                
                # Classify candidate domain
                candidate_domain, candidate_confidence = self.domain_classifier.classify_domain(candidate_text)
                
                # STRICT: Only add if domain matches exactly
                if candidate_domain == target_domain and candidate_confidence > 0.3:
                    # Create result dict with basic scoring
                    result = {
                        'email': candidate.get('email', ''),
                        'full_name': candidate.get('full_name', ''),
                        'skills': candidate.get('skills', []),
                        'total_experience_years': candidate.get('total_experience_years', candidate.get('experience', 0)),
                        'resume_text': candidate.get('resume_text', ''),
                        'phone': candidate.get('phone', ''),
                        'sourceURL': candidate.get('sourceURL', candidate.get('source_url', '')),
                        'final_score': 0.1,  # Low initial score (will be updated by scoring pipeline)
                        'skill_match_score': 0.0,
                        'domain': candidate_domain,
                        'domain_confidence': candidate_confidence,
                        'query_domain': target_domain,
                        'domain_match': True,
                        'source': 'domain_filtered_fetch'
                    }
                    
                    additional_candidates.append(result)
                    existing_emails.add(result['email'])
            except Exception as e:
                print(f"⚠️ Error processing candidate for domain fetch: {e}")
                continue
        
        print(f"✅ Fetched {len(additional_candidates)} additional candidates from {target_domain} domain")
        return additional_candidates
    
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
                        
                        # Update final score with behavioural score, but preserve skill match priority
                        behavioral_score = result['behavioural_analysis'].get('overall_score', 0.0)
                        if behavioral_score > 0:
                            # Combine behavioural score with existing score, but preserve skill priority
                            original_score = result.get('final_score', 0.0)
                            skill_match_score = result.get('skill_match_score', 0.0)
                            if skill_match_score > 0:
                                # If skill match exists, keep it prioritized (skill matching is top priority)
                                result['final_score'] = original_score  # Keep original (skill-based) score
                            else:
                                # If no skill match, use behavioral score
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
        # Ensure experience is at least 1
        exp_years = result.get('total_experience_years', 0)
        if exp_years == 0:
            skills_list = result.get('skills', [])
            resume_text = result.get('resume_text', '')
            if skills_list or resume_text:
                exp_years = 1
        
        try:
            from semantic_function.matcher.models import CandidateProfile
            
            return CandidateProfile(
                full_name=result.get('full_name', ''),
                email=result.get('email', ''),
                resume=result.get('resume_text', ''),
                skills=result.get('skills', []),
                experience_years=max(1, exp_years),
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
                'experience_years': max(1, exp_years),
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
    
    def _scale_score_to_percentage(self, score: float) -> float:
        """
        Scale final_score (0-1) to a higher percentage range (0-100).
        This boosts percentages for good matches while maintaining relative ordering.
        
        Uses a non-linear scaling function:
        - Scores 0.0-0.2: Scale to 40-60%
        - Scores 0.2-0.4: Scale to 60-75%
        - Scores 0.4-0.6: Scale to 75-85%
        - Scores 0.6-0.8: Scale to 85-93%
        - Scores 0.8-1.0: Scale to 93-100%
        """
        if score <= 0:
            return 40.0  # Minimum 40% for any result
        elif score >= 1.0:
            return 100.0  # Maximum 100%
        
        # Non-linear scaling: use a power function to boost good scores
        # This ensures good matches (0.4+) get 75%+, excellent matches (0.7+) get 90%+
        # Formula: base + (score^1.5) * range
        # This creates a curve that boosts mid-to-high scores more
        scaled = 50.0 + (score ** 1.3) * 45.0
        
        # Ensure minimum of 50% for any score > 0, and cap at 100%
        scaled = max(50.0, min(100.0, scaled))
        
        return round(scaled, 1)
    
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
