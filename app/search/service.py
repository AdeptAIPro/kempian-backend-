import os
import sys
import boto3
import logging
import time
import json
import numpy as np
import re
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
from importlib import import_module
from app.simple_logger import get_logger
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
from app.cache import search_cache, cache_manager
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Add adeptai components to path for imports
def _append_vendor_path(path: str) -> bool:
    """Append path to sys.path if it exists and isn't already included."""
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)
        return True
    return os.path.exists(path)

current_dir = os.path.dirname(__file__)
adeptai_master_path = os.path.join(current_dir, 'adeptai_master')
adeptai_components_path = os.path.join(current_dir, 'adeptai_components')

has_master = _append_vendor_path(adeptai_master_path)
has_components = _append_vendor_path(adeptai_components_path)

AdeptAIImportError = ImportError

ADEPTAI_IMPORT_BASES = []
if has_master:
    ADEPTAI_IMPORT_BASES.append('app.search.adeptai_master')
if has_components:
    ADEPTAI_IMPORT_BASES.append('app.search.adeptai_components')
if not ADEPTAI_IMPORT_BASES:
    ADEPTAI_IMPORT_BASES.append('app.search.adeptai_components')

def _import_adeptai_module(module_path: str):
    """Import module from adeptai_master first, then adeptai_components."""
    last_error = None
    for base in ADEPTAI_IMPORT_BASES:
        full_path = f"{base}.{module_path}"
        try:
            return import_module(full_path)
        except ImportError as err:
            last_error = err
    raise last_error or ImportError(f"Module {module_path} not found in AdeptAI packages")

def _import_attr(module_path: str, attr_name: str):
    """Import attribute from AdeptAI module, returning None if missing."""
    try:
        module = _import_adeptai_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as err:
        logger.warning(f"AdeptAI component '{module_path}.{attr_name}' unavailable: {err}")
        return None

# Setup logging
logger = get_logger("search")

# Setup AWS
REGION = 'ap-south-1'

# Configuration for large dataset handling
MAX_CANDIDATES_SMALL_DATASET = int(os.getenv('MAX_CANDIDATES_SMALL', 10000))
MAX_CANDIDATES_MEDIUM_DATASET = int(os.getenv('MAX_CANDIDATES_MEDIUM', 50000))
MAX_CANDIDATES_LARGE_DATASET = int(os.getenv('MAX_CANDIDATES_LARGE', 100000))
CACHE_VALIDITY_HOURS = int(os.getenv('CACHE_VALIDITY_HOURS', 1))  # How long to use cached index
FORCE_FULL_LOAD = os.getenv('FORCE_FULL_LOAD', 'false').lower() == 'true'  # Force load all candidates

# Domain labels for precise filtering
HEALTHCARE_DOMAIN_LABELS = {
    'healthcare', 'health care', 'medical', 'medicine', 'clinical', 'hospital',
    'med-surg', 'nursing', 'patient care', 'physician', 'doctor', 'nurse',
    'rn', 'lpn', 'cna', 'therapist'
}

IT_DOMAIN_LABELS = {
    'it/tech', 'technology', 'tech', 'software', 'engineering', 'developer', 'it'
}

# Performance optimization settings
ENABLE_PARALLEL_PROCESSING = os.getenv('ENABLE_PARALLEL_PROCESSING', 'true').lower() == 'true'
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))  # Number of parallel workers
VECTOR_CACHE_SIZE = int(os.getenv('VECTOR_CACHE_SIZE', 1000))  # Cache size for vectors
ENABLE_SMART_FILTERING = os.getenv('ENABLE_SMART_FILTERING', 'true').lower() == 'true'
ENABLE_EMBEDDING_CACHE = os.getenv('ENABLE_EMBEDDING_CACHE', 'true').lower() == 'true'

# ULTRA-FAST MODE: Load minimal candidates for immediate results
ULTRA_FAST_MODE = os.getenv('ULTRA_FAST_MODE', 'true').lower() == 'true'
FAST_LOAD_CANDIDATES = int(os.getenv('FAST_LOAD_CANDIDATES', 20000))  # Load 20000 candidates to ensure enough for domain filtering

# Global caches for performance
_vector_cache = {}
_tfidf_vectorizer = None
_tfidf_matrix = None
_candidate_vectors = None

# Initialize DynamoDB only if credentials are available
try:
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        dynamodb = boto3.resource('dynamodb', region_name=REGION,
                                  aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                  aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
        table = dynamodb.Table('resume_metadata')
        feedback_table = dynamodb.Table('resume_feedback')
    else:
        dynamodb = None
        table = None
        feedback_table = None
except Exception as e:
    logger.warning(f"Could not initialize DynamoDB: {e}")
    dynamodb = None
    table = None
    feedback_table = None

# Import the ORIGINAL AdeptAI algorithm components (from adeptbackend/)
try:
    # Import the main enhanced recruitment search system
    EnhancedRecruitmentSearchSystem = _import_attr('enhanced_recruitment_search', 'EnhancedRecruitmentSearchSystem')
    CandidateProfile = _import_attr('enhanced_recruitment_search', 'CandidateProfile')
    SkillExtractor = _import_attr('enhanced_recruitment_search', 'SkillExtractor')
    MemoryOptimizedEmbeddingSystem = _import_attr('enhanced_recruitment_search', 'MemoryOptimizedEmbeddingSystem')
   
    EnhancedCandidateMatchingSystem = _import_attr('enhanced_candidate_matcher', 'EnhancedCandidateMatchingSystem')
    MatchScore = _import_attr('enhanced_candidate_matcher', 'MatchScore')
   
    AdvancedJobQueryParser = _import_attr('advanced_query_parser', 'AdvancedJobQueryParser')
    ParsedJobQuery = _import_attr('advanced_query_parser', 'ParsedJobQuery')
    JobRequirement = _import_attr('advanced_query_parser', 'JobRequirement')
   
    OptimizedSearchSystem = _import_attr('search.fast_search', 'OptimizedSearchSystem')
    FastSearchSystem = _import_attr('search.ultra_fast_search', 'FastSearchSystem')
    PerformanceMonitor = _import_attr('search.performance', 'PerformanceMonitor')
    EmbeddingCache = _import_attr('search.cache', 'EmbeddingCache')
   
    BatchProcessor = _import_attr('utils.batch_processor', 'BatchProcessor')
    UtilsEmbeddingCache = _import_attr('utils.caching', 'EmbeddingCache')
    AdvancedSkillMatcher = _import_attr('utils.precision_scorer', 'AdvancedSkillMatcher')
    NaturalLanguageQueryParser = _import_attr('utils.query_parser', 'NaturalLanguageQueryParser')
    MultiModelEmbeddingService = _import_attr('utils.enhanced_embeddings', 'MultiModelEmbeddingService')
   
    DomainIntegration = _import_attr('domain_integration', 'DomainIntegration')
    BiasPrevention = _import_attr('bias_prevention', 'BiasPrevention')
   
    ORIGINAL_ALGORITHM_AVAILABLE = True
    logger.info("Original AdeptAI algorithm imported successfully")
   
except ImportError as e:
    logger.warning(f"Original AdeptAI algorithm not available: {e}")
    ORIGINAL_ALGORITHM_AVAILABLE = False

# Fallback algorithm (simplified version)
@dataclass
class FallbackMatchScore:
    """Fallback matching score"""
    overall_score: float
    confidence: float
    match_explanation: str

# Optimized utility functions
@lru_cache(maxsize=VECTOR_CACHE_SIZE)
def _get_text_hash(text: str) -> str:
    """Get hash for text caching"""
    return hashlib.md5(text.encode()).hexdigest()

def _is_meaningless_query(query: str) -> bool:
    """
    Check if a query is meaningless (contains only random characters, no valid words).
    Returns True if the query should be rejected.
    """
    if not query or not query.strip():
        return True
    
    query = query.strip()
    
    # Check if query is too short (less than 3 characters)
    if len(query) < 3:
        return True
    
    # Extract all words from the query
    words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
    
    # If no valid words found, it's meaningless
    if not words:
        return True
    
    # Check if query contains mostly random characters (no common English words)
    # Common English words that might appear in job descriptions
    common_words = {
        'the', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'to', 'of', 'a', 'an',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'must', 'can', 'may', 'might',
        'job', 'position', 'role', 'candidate', 'experience', 'skills', 'required',
        'developer', 'engineer', 'nurse', 'doctor', 'manager', 'analyst', 'specialist',
        'python', 'java', 'javascript', 'react', 'sql', 'aws', 'docker', 'kubernetes',
        'rn', 'lpn', 'cna', 'bachelor', 'master', 'degree', 'certification', 'years'
    }
    
    # Check if any word is a common English word or looks like a valid word
    # A word is considered valid if:
    # 1. It's in the common words list, OR
    # 2. It has at least 3 characters and contains vowels (likely a real word)
    # 3. It doesn't have excessive consonant clusters (random text indicator)
    valid_word_count = 0
    for word in words:
        if word in common_words:
            valid_word_count += 1
        elif len(word) >= 3:
            # Check if word contains vowels (likely a real word)
            has_vowels = any(char in word for char in 'aeiou')
            
            # Reject words with no vowels at all (strong indicator of random text)
            if not has_vowels:
                continue
            
            # Check if word has reasonable consonant-vowel pattern
            # Random strings usually have too many consonants in a row
            consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyz]{4,}', word)
            
            # Reject words with long consonant clusters (more than 4 consonants in a row)
            if len(consonant_clusters) > 0:
                continue
            
            # Check vowel-to-consonant ratio - valid words usually have reasonable balance
            vowel_count = sum(1 for char in word if char in 'aeiou')
            consonant_count = sum(1 for char in word if char in 'bcdfghjklmnpqrstvwxyz')
            total_letters = vowel_count + consonant_count
            
            if total_letters > 0:
                vowel_ratio = vowel_count / total_letters
                # Valid English words typically have 30-50% vowels
                # If vowel ratio is too low (< 20%), it's likely random text
                if vowel_ratio >= 0.20:
                    valid_word_count += 1
    
    # If less than 30% of words are valid, consider it meaningless
    # Also require at least 1 valid word
    if len(words) > 0:
        valid_ratio = valid_word_count / len(words)
        if valid_ratio < 0.3 or valid_word_count == 0:
            logger.warning(f"Query rejected as meaningless: '{query}' (valid words: {valid_word_count}/{len(words)}, ratio: {valid_ratio:.2%})")
            return True
    
    # Additional check: if query is very long but has very few valid words, reject it
    if len(query) > 50 and valid_word_count < 2:
        logger.warning(f"Query rejected as meaningless: '{query}' (long query with too few valid words)")
        return True
    
    return False

def _initialize_tfidf_vectorizer(candidates: List[Dict]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Initialize TF-IDF vectorizer with candidate data"""
    global _tfidf_vectorizer, _tfidf_matrix
   
    if _tfidf_vectorizer is not None and _tfidf_matrix is not None:
        return _tfidf_vectorizer, _tfidf_matrix
   
    logger.info("Initializing TF-IDF vectorizer for fast similarity calculations...")
   
    # Prepare text data
    def _to_text(value: Any) -> str:
        try:
            if value is None:
                return ''
            if isinstance(value, list):
                return ' '.join(str(x) for x in value if x is not None)
            return str(value)
        except Exception:
            return ''
    texts = []
    for candidate in candidates:
        # Handle skills properly - convert to string if it's a list
        skills = candidate.get('skills', [])
        skills_text = _to_text(skills)
       
        text_parts = [
            _to_text(candidate.get('resume_text', '')),
            _to_text(candidate.get('experience', '')),
            skills_text,
            _to_text(candidate.get('education', '')),
            _to_text(candidate.get('summary', ''))
        ]
        combined_text = ' '.join(part for part in text_parts if part)
        texts.append(combined_text)
   
    # Initialize TF-IDF vectorizer with optimized parameters
    _tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit features for speed
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        sublinear_tf=True  # Use sublinear term frequency scaling
    )
   
    # Fit and transform
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(texts)
   
    logger.info(f"TF-IDF vectorizer initialized with {_tfidf_matrix.shape[1]} features")
    return _tfidf_vectorizer, _tfidf_matrix

# Soft skills that should be excluded from technical matching
SOFT_SKILLS = {
    'communication', 'leadership', 'teamwork', 'problem solving', 'analytical thinking',
    'creative', 'adaptable', 'collaborative', 'mentoring', 'project management',
    'time management', 'organization', 'multitasking', 'attention to detail',
    'critical thinking', 'decision making', 'negotiation', 'conflict resolution',
    'presentation', 'public speaking', 'interpersonal', 'emotional intelligence',
    'customer service', 'customer relations', 'client relations', 'stakeholder management',
    'work ethic', 'reliability', 'punctuality', 'professionalism', 'integrity',
    'flexibility', 'resilience', 'stress management', 'self-motivated', 'proactive',
    'active listening', 'empathy', 'patience', 'tolerance', 'cultural awareness',
    'diversity', 'inclusion', 'team player', 'cooperation', 'coordination',
    'delegation', 'supervision', 'training', 'coaching', 'mentoring',
    'strategic thinking', 'vision', 'innovation', 'entrepreneurship', 'initiative',
    # Additional common soft skills
    'business development', 'sales', 'marketing', 'customer care', 'customer support',
    'relationship building', 'networking', 'persuasion', 'influence', 'motivation',
    'change management', 'people management', 'team building', 'conflict management',
    'verbal communication', 'written communication', 'listening skills', 'feedback',
    'adaptability', 'versatility', 'open-mindedness', 'curiosity', 'learning agility'
}

def _is_technical_skill(skill: str) -> bool:
    """Check if a skill is a technical skill (not a soft skill)"""
    if not skill:
        return False
    
    skill_lower = skill.lower().strip()
    
    # Check against soft skills list
    if skill_lower in SOFT_SKILLS:
        return False
    
    # Check if skill contains soft skill keywords
    soft_keywords = ['communication', 'leadership', 'teamwork', 'problem solving', 
                     'management', 'interpersonal', 'customer service', 'presentation',
                     'business development', 'sales', 'marketing', 'customer care',
                     'relationship', 'networking', 'persuasion', 'influence']
    if any(keyword in skill_lower for keyword in soft_keywords):
        # But allow technical management terms
        technical_management = ['project management', 'product management', 'program management',
                               'system management', 'database management', 'network management',
                               'data management', 'content management', 'configuration management',
                               'version control', 'release management', 'deployment management']
        if not any(tech_term in skill_lower for tech_term in technical_management):
            return False
    
    # Technical skills typically contain:
    # - Programming languages, frameworks, tools, technologies
    # - Certifications, protocols, standards
    # - Technical methodologies
    return True

def _filter_technical_skills_only(skills: List[str]) -> List[str]:
    """Filter skills to keep only technical skills, removing soft skills"""
    if not skills:
        return []
    
    technical_skills = []
    for skill in skills:
        if isinstance(skill, str):
            skill_clean = skill.strip()
            if skill_clean and _is_technical_skill(skill_clean):
                technical_skills.append(skill_clean)
    
    return technical_skills

def _get_healthcare_skill_variations():
    """Get mapping of healthcare skill abbreviations to full forms and variations"""
    return {
        # Nursing credentials
        'rn': ['rn', 'registered nurse', 'registered nursing', 'r.n.', 'r.n'],
        'registered nurse': ['rn', 'registered nurse', 'registered nursing', 'r.n.', 'r.n'],
        'lpn': ['lpn', 'licensed practical nurse', 'lp nurse', 'l.p.n.', 'l.p.n'],
        'licensed practical nurse': ['lpn', 'licensed practical nurse', 'lp nurse', 'l.p.n.', 'l.p.n'],
        'cna': ['cna', 'certified nursing assistant', 'nursing assistant', 'c.n.a.', 'c.n.a'],
        'certified nursing assistant': ['cna', 'certified nursing assistant', 'nursing assistant', 'c.n.a.', 'c.n.a'],
        'np': ['np', 'nurse practitioner', 'aprn', 'advanced practice registered nurse', 'n.p.', 'n.p'],
        'nurse practitioner': ['np', 'nurse practitioner', 'aprn', 'advanced practice registered nurse', 'n.p.', 'n.p'],
        # Certifications
        'bls': ['bls', 'basic life support', 'b.l.s.', 'b.l.s'],
        'basic life support': ['bls', 'basic life support', 'b.l.s.', 'b.l.s'],
        'acls': ['acls', 'advanced cardiac life support', 'a.c.l.s.', 'a.c.l.s'],
        'advanced cardiac life support': ['acls', 'advanced cardiac life support', 'a.c.l.s.', 'a.c.l.s'],
        'cpr': ['cpr', 'cardiopulmonary resuscitation', 'c.p.r.', 'c.p.r'],
        'cardiopulmonary resuscitation': ['cpr', 'cardiopulmonary resuscitation', 'c.p.r.', 'c.p.r'],
        'pals': ['pals', 'pediatric advanced life support', 'p.a.l.s.', 'p.a.l.s'],
        'pediatric advanced life support': ['pals', 'pediatric advanced life support', 'p.a.l.s.', 'p.a.l.s'],
        'tncc': ['tncc', 'trauma nursing core course', 't.n.c.c.', 't.n.c.c'],
        'trauma nursing core course': ['tncc', 'trauma nursing core course', 't.n.c.c.', 't.n.c.c'],
        # Units/Departments
        'icu': ['icu', 'intensive care unit', 'critical care', 'i.c.u.', 'i.c.u'],
        'intensive care unit': ['icu', 'intensive care unit', 'critical care', 'i.c.u.', 'i.c.u'],
        'ccu': ['ccu', 'cardiac care unit', 'coronary care', 'c.c.u.', 'c.c.u'],
        'er': ['er', 'emergency room', 'emergency department', 'ed', 'e.r.', 'e.r'],
        'emergency room': ['er', 'emergency room', 'emergency department', 'ed', 'e.r.', 'e.r'],
        'emergency department': ['er', 'emergency room', 'emergency department', 'ed', 'e.r.', 'e.r'],
        'ed': ['er', 'emergency room', 'emergency department', 'ed', 'e.r.', 'e.r'],
        'med-surg': ['med-surg', 'medical surgical', 'medical-surgical', 'med surg'],
        'nicu': ['nicu', 'neonatal intensive care', 'n.i.c.u.', 'n.i.c.u'],
        'picu': ['picu', 'pediatric intensive care', 'p.i.c.u.', 'p.i.c.u'],
        # Systems
        'ehr': ['ehr', 'emr', 'electronic health record', 'electronic medical record', 'e.h.r.', 'e.m.r.'],
        'emr': ['ehr', 'emr', 'electronic health record', 'electronic medical record', 'e.h.r.', 'e.m.r.'],
        'electronic health record': ['ehr', 'emr', 'electronic health record', 'electronic medical record', 'e.h.r.', 'e.m.r.'],
        'electronic medical record': ['ehr', 'emr', 'electronic health record', 'electronic medical record', 'e.h.r.', 'e.m.r.'],
    }

def _normalize_skill_for_matching(skill: str) -> List[str]:
    """Normalize a skill to get all its variations for matching"""
    skill_lower = skill.lower().strip()
    variations = _get_healthcare_skill_variations()
   
    # Check if skill is in variations map
    for key, variants in variations.items():
        if skill_lower == key or skill_lower in variants:
            return variants
   
    # Return the skill itself and common variations
    normalized = [skill_lower]
   
    # Remove punctuation variations
    normalized.append(skill_lower.replace('.', '').replace('-', ' ').replace('_', ' '))
    normalized.append(skill_lower.replace(' ', '').replace('-', ''))
   
    return list(set(normalized))

def _skills_match(skill1: str, skill2: str) -> bool:
    """Check if two skills match, considering healthcare variations"""
    skill1_variations = _normalize_skill_for_matching(skill1)
    skill2_variations = _normalize_skill_for_matching(skill2)
   
    # Check for exact match in variations
    for v1 in skill1_variations:
        for v2 in skill2_variations:
            if v1 == v2:
                return True
            # Check substring match (but be careful with short abbreviations)
            # For healthcare abbreviations (2-4 chars), allow exact match only
            if len(v1) >= 2 and len(v2) >= 2:
                # For short skills (2-4 chars), require exact match
                if len(v1) <= 4 and len(v2) <= 4:
                    if v1 == v2:
                        return True
                # For longer skills, allow substring match
                elif len(v1) >= 3 and len(v2) >= 3:
                    if v1 in v2 or v2 in v1:
                        return True
   
    return False

def _order_skills_with_required_first(candidate_skills: List[str], job_required_skills: List[str]) -> List[str]:
    """Order candidate skills so that required/matched skills appear first"""
    if not job_required_skills or not candidate_skills:
        return candidate_skills if isinstance(candidate_skills, list) else []
   
    # Normalize to lists
    if isinstance(candidate_skills, str):
        candidate_skills = [s.strip() for s in candidate_skills.split(',')]
    if isinstance(job_required_skills, str):
        job_required_skills = [s.strip() for s in job_required_skills.split(',')]
   
    candidate_skills = [str(s).strip() for s in candidate_skills if s]
    job_required_skills = [str(s).strip() for s in job_required_skills if s]
   
    # Separate matched and unmatched skills
    matched_skills = []
    unmatched_skills = []
    matched_indices = set()
   
    # Find skills that match required skills (with healthcare-aware matching)
    for i, candidate_skill in enumerate(candidate_skills):
        is_matched = False
       
        for required_skill in job_required_skills:
            # Use healthcare-aware matching
            if _skills_match(candidate_skill, required_skill):
                matched_skills.append(candidate_skill)
                matched_indices.add(i)
                is_matched = True
                break
       
        if not is_matched:
            unmatched_skills.append(candidate_skill)
   
    # Return matched skills first, then unmatched skills
    return matched_skills + unmatched_skills

def _calculate_fast_similarity(query: str, candidate_text: str, vectorizer: TfidfVectorizer, matrix: np.ndarray, candidate_idx: int) -> float:
    """Calculate fast similarity using TF-IDF"""
    try:
        # Transform query
        query_vector = vectorizer.transform([query])
       
        # Get candidate vector
        candidate_vector = matrix[candidate_idx:candidate_idx+1]
       
        # Calculate cosine similarity
        similarity = cosine_similarity(query_vector, candidate_vector)[0][0]
        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating fast similarity: {e}")
        return 0.0

def _smart_filter_candidates(candidates: List[Dict], job_description: str, top_k: int = 200) -> List[Dict]:
    """Smart pre-filtering to reduce computation"""
    if not ENABLE_SMART_FILTERING or len(candidates) <= top_k:
        return candidates
   
    logger.info(f"Applying smart filtering to {len(candidates)} candidates...")
   
    # Extract key terms from job description
    job_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
    job_keywords = {kw for kw in job_keywords if len(kw) > 2}
   
    # Score candidates based on keyword matches
    scored_candidates = []
    for i, candidate in enumerate(candidates):
        # Handle skills properly - convert to string if it's a list
        skills = candidate.get('skills', [])
        if isinstance(skills, list):
            skills_text = ' '.join(str(s) for s in skills if s is not None)
        else:
            skills_text = str(skills) if skills else ''
       
        def _safe_text(v: Any) -> str:
            if v is None:
                return ''
            if isinstance(v, list):
                return ' '.join(str(x) for x in v if x is not None)
            return str(v)
       
        candidate_text = ' '.join([
            _safe_text(candidate.get('resume_text', '')),
            _safe_text(candidate.get('experience', '')),
            skills_text,
            _safe_text(candidate.get('education', '')),
            _safe_text(candidate.get('summary', ''))
        ]).lower()
       
        # Count keyword matches
        matches = sum(1 for keyword in job_keywords if keyword in candidate_text)
        score = matches / len(job_keywords) if job_keywords else 0
       
        scored_candidates.append((score, i, candidate))
   
    # Sort by score and take top candidates
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    filtered_candidates = [candidate for _, _, candidate in scored_candidates[:top_k]]
   
    logger.info(f"Smart filtering reduced candidates from {len(candidates)} to {len(filtered_candidates)}")
    return filtered_candidates

def _parallel_candidate_scoring(candidates: List[Dict], job_description: str, vectorizer: TfidfVectorizer, matrix: np.ndarray) -> List[Tuple[int, float]]:
    """Parallel candidate scoring for better performance"""
    if not ENABLE_PARALLEL_PROCESSING or len(candidates) < 50:
        # Use sequential processing for small datasets
        scores = []
        for i, candidate in enumerate(candidates):
            score = _calculate_fast_similarity(job_description, '', vectorizer, matrix, i)
            scores.append((i, score))
        return scores
   
    logger.info(f"Using parallel processing for {len(candidates)} candidates...")
   
    def score_candidate(args):
        i, candidate, job_desc, vec, mat = args
        score = _calculate_fast_similarity(job_desc, '', vec, mat, i)
        return (i, score)
   
    # Prepare arguments for parallel processing
    args_list = [(i, candidate, job_description, vectorizer, matrix) for i, candidate in enumerate(candidates)]
   
    # Use ThreadPoolExecutor for I/O bound tasks
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        scores = list(executor.map(score_candidate, args_list))
   
    return scores

def _normalize_to_list(value: Any) -> List[str]:
    """Convert value to a clean list of strings"""
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if item]
    if isinstance(value, str):
        parts = re.split(r'[;\n,]', value)
        return [part.strip() for part in parts if part.strip()]
    return []

def _parse_experience_years(value: Any) -> int:
    """Extract numeric experience in years from various value types"""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        try:
            return max(int(round(float(value))), 0)
        except Exception:
            return 0
    if isinstance(value, str):
        match = re.search(r'\d+(?:\.\d+)?', value)
        if match:
            try:
                return max(int(round(float(match.group(0)))), 0)
            except Exception:
                return 0
    return 0

def _normalize_candidate_profile(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize candidate profile fields from mixed data sources"""
    contact_sources: List[Dict[str, Any]] = []
    for key in ('contact', 'contactInfo', 'contact_info', 'contactDetails', 'contact_details'):
        value = candidate.get(key)
        if isinstance(value, dict):
            contact_sources.append(value)

    contact: Dict[str, Any] = {}
    for source in contact_sources:
        for key, value in source.items():
            if value is not None and key not in contact:
                contact[key] = value

    def _first_non_empty(*values: Any) -> str:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
            elif isinstance(value, (int, float)):
                return str(value)
        return ''

    # Enhanced name extraction with more field variations
    full_name = _first_non_empty(
        candidate.get('full_name'),
        candidate.get('FullName'),
        candidate.get('fullName'),
        candidate.get('name'),
        candidate.get('Name'),
        candidate.get('candidate_name'),
        candidate.get('candidateName'),
        candidate.get('CandidateName'),
        candidate.get('applicant_name'),
        candidate.get('applicantName')
    )
    email = _first_non_empty(
        candidate.get('email'),
        candidate.get('Email'),
        candidate.get('primary_email'),
        candidate.get('primaryEmail'),
        candidate.get('contact_email'),
        candidate.get('contactEmail'),
        contact.get('email'),
        contact.get('primaryEmail'),
        contact.get('emailAddress')
    )
    if not email:
        for key in ('emails', 'emailAddresses', 'email_addresses'):
            emails_value = candidate.get(key) or contact.get(key)
            if isinstance(emails_value, list):
                email = _first_non_empty(*emails_value)
                if email:
                    break
    # Enhanced phone extraction with more field variations
    phone = _first_non_empty(
        candidate.get('phone'),
        candidate.get('Phone'),
        candidate.get('phone_number'),
        candidate.get('phoneNumber'),
        candidate.get('PhoneNumber'),
        candidate.get('primary_phone'),
        candidate.get('primaryPhone'),
        candidate.get('PrimaryPhone'),
        candidate.get('mobile'),
        candidate.get('Mobile'),
        candidate.get('mobile_number'),
        candidate.get('mobileNumber'),
        candidate.get('contact_number'),
        candidate.get('contactNumber'),
        candidate.get('ContactNumber'),
        contact.get('phone'),
        contact.get('Phone'),
        contact.get('mobile'),
        contact.get('contactNumber'),
        contact.get('phoneNumber')
    )
   
    # Enhanced location extraction with more field variations
    location = _first_non_empty(
        candidate.get('location'),
        candidate.get('Location'),
        candidate.get('current_location'),
        candidate.get('currentLocation'),
        candidate.get('CurrentLocation'),
        candidate.get('city'),
        candidate.get('City'),
        candidate.get('address'),
        candidate.get('Address'),
        candidate.get('residence'),
        candidate.get('Residence'),
        candidate.get('location_city'),
        candidate.get('locationCity')
    )
    skills = _normalize_to_list(candidate.get('skills') or candidate.get('Skills'))
   
    # Enhanced education extraction with more field variations
    education = _first_non_empty(
        candidate.get('education'),
        candidate.get('Education'),
        candidate.get('qualification'),
        candidate.get('Qualification'),
        candidate.get('qualifications'),
        candidate.get('Qualifications'),
        candidate.get('degree'),
        candidate.get('Degree'),
        candidate.get('degrees'),
        candidate.get('Degrees'),
        candidate.get('educational_background'),
        candidate.get('educationalBackground'),
        candidate.get('academic_background'),
        candidate.get('academicBackground')
    )
    education = education if isinstance(education, str) else str(education) if education else ''
   
    designations = _normalize_to_list(
        candidate.get('designations_with_experience') or
        candidate.get('designationsWithExperience') or
        candidate.get('designations')
    )
    certifications = _normalize_to_list(candidate.get('certifications') or candidate.get('Certifications'))
    total_exp_value = (
        candidate.get('total_experience_years') or
        candidate.get('experience_years') or
        candidate.get('Experience') or
        candidate.get('experience')
    )
    total_experience_years = _parse_experience_years(total_exp_value)
    domain_tag = candidate.get('domain_tag') or candidate.get('category') or candidate.get('domain') or ''
    current_position = (
        candidate.get('current_position') or
        candidate.get('currentRole') or
        candidate.get('Position') or
        candidate.get('title') or
        candidate.get('Title') or
        ''
    )
    filename = candidate.get('filename') or candidate.get('resume_filename') or candidate.get('resumeFileName') or ''
    source_url = (
        candidate.get('sourceURL') or
        candidate.get('sourceUrl') or
        candidate.get('source_url') or
        candidate.get('resumeUrl') or
        candidate.get('ResumeFile') or
        ''
    )
    # Enhanced resume text extraction with more field variations
    resume_text = _first_non_empty(
        candidate.get('resume_text'),
        candidate.get('resumeText'),
        candidate.get('ResumeText'),
        candidate.get('resume'),
        candidate.get('Resume'),
        candidate.get('summary'),
        candidate.get('Summary'),
        candidate.get('profile_summary'),
        candidate.get('profileSummary'),
        candidate.get('bio'),
        candidate.get('Bio'),
        candidate.get('description'),
        candidate.get('Description')
    )
    education = education if isinstance(education, str) else str(education)
    return {
        'full_name': full_name,
        'email': email,
        'phone': phone,
        'skills': skills,
        'education': education,
        'designations_with_experience': designations,
        'certifications': certifications,
        'total_experience_years': total_experience_years,
        'domain_tag': domain_tag,
        'current_position': current_position,
        'location': location,
        'filename': filename,
        'source_url': source_url,
        'resume_text': resume_text
    }

class FallbackAlgorithm:
    """Optimized fallback algorithm with enhanced performance"""
   
    def __init__(self):
        self.candidates = []
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._candidate_vectors = None
        logger.info("Optimized fallback algorithm initialized")
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate candidates based on email and full_name combination."""
        if not candidates:
            return candidates
        
        seen = set()
        deduplicated = []
        duplicates_removed = 0
        
        for candidate in candidates:
            # Get unique identifier: email + full_name combination
            email = (
                candidate.get('email') or 
                candidate.get('Email') or 
                candidate.get('email_address') or
                ''
            ).strip().lower()
            
            full_name = (
                candidate.get('full_name') or 
                candidate.get('FullName') or 
                candidate.get('name') or 
                candidate.get('Name') or
                ''
            ).strip().lower()
            
            # Create unique key from email and name
            # If email exists, use email as primary identifier
            # If no email, use name + phone as fallback
            if email and email not in ('unknown', 'n/a', ''):
                unique_key = email
            elif full_name and full_name not in ('unknown', ''):
                # Use name + phone as fallback identifier
                phone = (
                    candidate.get('phone') or 
                    candidate.get('Phone') or 
                    ''
                ).strip().lower()
                unique_key = f"{full_name}_{phone}" if phone else full_name
            else:
                # Last resort: use source_url or filename if available
                source_url = candidate.get('source_url') or candidate.get('sourceUrl') or candidate.get('sourceURL') or ''
                filename = candidate.get('filename') or ''
                unique_key = source_url or filename or f"unknown_{len(seen)}"
            
            # Check if we've seen this candidate before
            if unique_key not in seen:
                seen.add(unique_key)
                deduplicated.append(candidate)
            else:
                duplicates_removed += 1
                logger.debug(f"Removed duplicate candidate: {full_name or 'Unknown'} ({email or 'no email'})")
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate candidates. Remaining: {len(deduplicated)}")
        
        return deduplicated
   
    def _get_suggested_candidates(self, job_description: str, top_k: int = 5) -> List[Dict]:
        """Get suggested candidates when no matches found - using only technical skills"""
        try:
            logger.info(f"Getting suggested candidates for: {job_description[:100]}...")
            
            # Extract ONLY technical skills from job description
            all_skills = self._extract_skills_from_job(job_description)
            technical_skills_only = _filter_technical_skills_only(all_skills)
            
            logger.info(f"Technical skills extracted: {technical_skills_only}")
            
            if not technical_skills_only or not self.candidates:
                # If no technical skills or no candidates, return empty
                return []
            
            # Score candidates based on technical skills only
            scored_candidates = []
            for candidate in self.candidates[:1000]:  # Limit to first 1000 for performance
                candidate_skills = candidate.get('skills', [])
                if isinstance(candidate_skills, str):
                    candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                
                # Filter candidate skills to technical only
                candidate_technical_skills = _filter_technical_skills_only(candidate_skills)
                
                # Calculate technical skill match
                matches = 0
                for tech_skill in technical_skills_only:
                    tech_skill_lower = tech_skill.lower()
                    for candidate_tech in candidate_technical_skills:
                        candidate_tech_lower = candidate_tech.lower()
                        if tech_skill_lower in candidate_tech_lower or candidate_tech_lower in tech_skill_lower:
                            matches += 1
                            break
                
                if matches > 0:
                    match_ratio = matches / len(technical_skills_only)
                    scored_candidates.append((match_ratio, candidate))
            
            # Sort by match ratio and return top candidates
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            suggested = [candidate for _, candidate in scored_candidates[:top_k]]
            
            logger.info(f"Found {len(suggested)} suggested candidates based on technical skills")
            return suggested
            
        except Exception as e:
            logger.error(f"Error getting suggested candidates: {e}")
            return []

    def keyword_search(self, job_description, top_k=20):
        """Optimized keyword-based search with TF-IDF"""
        if not job_description:
            return [], "No job description provided"
       
        # Validate query - reject meaningless queries early
        if _is_meaningless_query(job_description):
            logger.warning(f"Meaningless query rejected in keyword_search: '{job_description[:100]}...'")
            return [], "No candidates found. Please provide a valid job description with meaningful keywords."
       
        start_time = time.time()
        logger.info(f"Starting optimized keyword search for: {job_description[:100]}...")
       
        # Extract required skills and detect domain FIRST
        # Filter to technical skills only
        job_keywords = re.findall(r'\b\w+\b', job_description.lower())
        job_domain = self.detect_domain(job_keywords)
        job_domain_lower = (job_domain or '').strip().lower()
        all_job_skills = self._extract_skills_from_job(job_description)
        # Filter to technical skills only for matching
        job_required_skills = _filter_technical_skills_only(all_job_skills)
       
        logger.info(f"[DEBUG] Job Analysis - Domain: {job_domain_lower}, Required skills: {job_required_skills}")
        logger.info(f"[DEBUG] Job description snippet: {job_description[:200]}...")
       
        # Apply smart filtering first
        # Filter to many more candidates for domain filtering (domain filtering removes most candidates)
        if job_domain_lower and job_domain_lower != 'general':
            # For domain filtering, we need 10x more candidates
            filter_top_k = top_k * 10
            logger.info(f"Domain filtering active: Pre-filtering to {filter_top_k} candidates")
        else:
            filter_top_k = top_k * 5
       
        filtered_candidates = _smart_filter_candidates(self.candidates, job_description, filter_top_k)
       
        # STRICT PRE-FILTERING: Reject candidates with zero relevant skills or wrong domain
        logger.info(f"[DEBUG] Starting strict pre-filtering. Initial candidates: {len(filtered_candidates)}")
        logger.info(f"[DEBUG] Pre-filter conditions - Has required skills: {bool(job_required_skills)}, Domain: {job_domain_lower}")
       
        if job_required_skills or job_domain_lower:
            strictly_filtered = []
            rejected_count = 0
            rejected_domain_mismatch = 0
            rejected_skill_mismatch = 0
            rejected_non_healthcare_skills = 0
           
            for idx, candidate in enumerate(filtered_candidates):
                candidate_name = candidate.get('full_name') or candidate.get('FullName') or candidate.get('name', f'Candidate_{idx}')
                candidate_skills_raw = candidate.get('skills', [])
                if isinstance(candidate_skills_raw, str):
                    candidate_skills_raw = [s.strip() for s in candidate_skills_raw.split(',')]
                candidate_skills_lower = [str(s).lower().strip() for s in candidate_skills_raw if s]
               
                logger.debug(f"[DEBUG] Processing candidate {idx+1}/{len(filtered_candidates)}: {candidate_name}")
                logger.debug(f"[DEBUG] Candidate skills: {candidate_skills_lower[:10]}")  # First 10 skills
                # Check domain mismatch (strict rejection)
                if job_domain_lower and job_domain_lower != 'general':
                    candidate_domain = (
                        candidate.get('domain_tag') or
                        candidate.get('category') or
                        candidate.get('domain') or
                        ''
                    ).strip().lower()
                   
                    logger.debug(f"[DEBUG] Candidate domain: {candidate_domain} vs Job domain: {job_domain_lower}")
                   
                    # Strict domain rejection for healthcare jobs
                    if job_domain_lower == 'healthcare':
                        # Reject IT/Tech candidates for healthcare jobs
                        if candidate_domain in {'it/tech', 'technology', 'tech', 'it', 'software', 'engineering'}:
                            rejected_count += 1
                            rejected_domain_mismatch += 1
                            logger.warning(f"[DEBUG] REJECTED {candidate_name}: IT/Tech domain ({candidate_domain}) for healthcare job")
                            continue
                        # Also check candidate skills for domain mismatch
                        candidate_skills = candidate.get('skills', [])
                        if isinstance(candidate_skills, str):
                            candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                        candidate_skills_lower = [s.lower() for s in candidate_skills if s]
                       
                        # STRICT: For healthcare jobs, require ACTUAL clinical/nursing skills, not just generic "healthcare"
                        # Non-clinical skills that should be rejected for clinical healthcare roles
                        non_clinical_skills = {
                            'accounting', 'account', 'accounts receivable', 'accounts payable',
                            'customer service', 'customer care', 'customer support',
                            'data entry', 'data processing', 'administrative',
                            'sales', 'marketing', 'finance', 'hr', 'human resources',
                            'receptionist', 'secretary', 'office management', 'billing',
                            'insurance', 'claims processing', 'medical billing', 'medical coding',
                            'communication'  # Generic communication is not a clinical skill
                        }
                       
                        # ACTUAL clinical/nursing skills required for healthcare jobs
                        clinical_healthcare_skills = {
                            'rn', 'registered nurse', 'nurse', 'nursing', 'lpn', 'cna',
                            'patient care', 'clinical', 'clinical care', 'direct patient care',
                            'icu', 'ccu', 'er', 'emergency', 'med-surg', 'medical surgical',
                            'bls', 'acls', 'cpr', 'pals', 'tncc',
                            'medication administration', 'med administration', 'med pass',
                            'charting', 'clinical documentation', 'care plan', 'care planning',
                            'case management', 'discharge planning', 'vital signs', 'vitals',
                            'phlebotomy', 'blood draw', 'iv therapy', 'intravenous',
                            'wound care', 'dressing change', 'infection control',
                            'ehr', 'emr', 'epic', 'cerner', 'meditech',
                            'therapist', 'occupational therapy', 'physical therapy', 'respiratory therapy',
                            'physician', 'doctor', 'surgeon', 'anesthesiologist'
                        }
                       
                        # Check if candidate has actual clinical skills
                        has_clinical_skills = any(
                            clinical_skill in ' '.join(candidate_skills_lower)
                            for clinical_skill in clinical_healthcare_skills
                        )
                       
                        # Check if candidate has non-clinical skills
                        has_non_clinical = any(
                            non_clinical in ' '.join(candidate_skills_lower)
                            for non_clinical in non_clinical_skills
                        )
                       
                        # Count clinical vs non-clinical skills
                        clinical_count = sum(1 for skill in candidate_skills_lower
                                            if any(cs in skill for cs in clinical_healthcare_skills))
                        non_clinical_count = sum(1 for skill in candidate_skills_lower
                                                if any(ncs in skill for ncs in non_clinical_skills))
                       
                        logger.debug(f"[DEBUG] Healthcare check - Clinical skills: {clinical_count}, Non-clinical: {non_clinical_count}, Has clinical: {has_clinical_skills}")
                       
                        # REJECT if:
                        # 1. No clinical skills at all, OR
                        # 2. Has non-clinical skills and clinical skills count is less than non-clinical count
                        if not has_clinical_skills:
                            rejected_count += 1
                            rejected_non_healthcare_skills += 1
                            logger.warning(f"[DEBUG] REJECTED {candidate_name}: No clinical skills for healthcare job (skills: {candidate_skills_lower[:5]})")
                            continue
                        elif has_non_clinical and clinical_count < non_clinical_count:
                            rejected_count += 1
                            rejected_non_healthcare_skills += 1
                            logger.warning(f"[DEBUG] REJECTED {candidate_name}: More non-clinical skills ({non_clinical_count}) than clinical skills ({clinical_count}) for healthcare job")
                            continue
                   
                    # Strict domain rejection for IT/Tech jobs
                    if job_domain_lower in {'it/tech', 'technology', 'tech'}:
                        if candidate_domain in {'healthcare', 'medical', 'nursing'}:
                            # Only reject if they have NO tech skills
                            candidate_skills = candidate.get('skills', [])
                            if isinstance(candidate_skills, str):
                                candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                            candidate_skills_lower = [s.lower() for s in candidate_skills if s]
                           
                            tech_indicators = {'python', 'java', 'javascript', 'react', 'sql', 'aws', 'docker',
                                             'software', 'developer', 'programming', 'coding'}
                            has_tech_skills = any(indicator in ' '.join(candidate_skills_lower) for indicator in tech_indicators)
                           
                            if not has_tech_skills:
                                rejected_count += 1
                                continue
               
                # Check skill match - STRICT: Only match on TECHNICAL skills, reject soft skills
                if job_required_skills:
                    candidate_skills = candidate.get('skills', [])
                    if isinstance(candidate_skills, str):
                        candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                    
                    # CRITICAL: Filter candidate skills to TECHNICAL ONLY - exclude soft skills
                    candidate_technical_skills = _filter_technical_skills_only(candidate_skills)
                    candidate_technical_skills_lower = [s.lower() for s in candidate_technical_skills if s]
                   
                    # If candidate has NO technical skills, reject immediately
                    if not candidate_technical_skills_lower:
                        rejected_count += 1
                        rejected_skill_mismatch += 1
                        logger.warning(f"[DEBUG] REJECTED {candidate_name}: No technical skills found (only soft skills)")
                        continue
                   
                    # Calculate skill match ratio using ONLY technical skills
                    job_skills_lower = [s.lower() for s in job_required_skills]
                    matches = sum(1 for skill in job_skills_lower if any(skill in cs or cs in skill for cs in candidate_technical_skills_lower))
                    skill_match_ratio = matches / len(job_skills_lower) if job_skills_lower else 0.0
                   
                    matched_skills = [skill for skill in job_skills_lower if any(skill in cs or cs in skill for cs in candidate_technical_skills_lower)]
                   
                    logger.debug(f"[DEBUG] Technical skill match - Required: {job_skills_lower}, Candidate technical: {candidate_technical_skills_lower[:5]}, Matched: {matched_skills}, Ratio: {skill_match_ratio:.2%}")
                   
                    # STRICT: For exact matches, require at least 50% technical skill match
                    # This ensures we only show candidates with substantial technical skill overlap
                    skill_threshold = 0.50  # Require 50% technical skill match for exact matches
                   
                    # REJECT if technical skill match is below threshold
                    if skill_match_ratio < skill_threshold:
                        rejected_count += 1
                        rejected_skill_mismatch += 1
                        logger.warning(f"[DEBUG] REJECTED {candidate_name}: Insufficient technical skill match ({skill_match_ratio:.2%} < {skill_threshold:.0%}, matched: {matched_skills})")
                        continue
                   
                    logger.info(f"[DEBUG] ACCEPTED {candidate_name}: Technical skill match {skill_match_ratio:.2%} (matched: {matched_skills})")
               
                strictly_filtered.append(candidate)
           
            filtered_candidates = strictly_filtered
            logger.info(f"[DEBUG] Strict pre-filtering COMPLETE:")
            logger.info(f"[DEBUG]   - Total rejected: {rejected_count}")
            logger.info(f"[DEBUG]   - Rejected (domain mismatch): {rejected_domain_mismatch}")
            logger.info(f"[DEBUG]   - Rejected (non-healthcare skills): {rejected_non_healthcare_skills}")
            logger.info(f"[DEBUG]   - Rejected (insufficient skill match): {rejected_skill_mismatch}")
            logger.info(f"[DEBUG]   - Remaining candidates: {len(filtered_candidates)}")
           
            if len(filtered_candidates) == 0:
                logger.warning(f"[DEBUG] WARNING: All candidates were filtered out! Relaxing filters and retrying...")
                # Fallback: Relax filtering - accept candidates with any skill match or domain match
                relaxed_filtered = []
                for candidate in _smart_filter_candidates(self.candidates, job_description, top_k * 20):
                    candidate_name = candidate.get('full_name') or candidate.get('FullName', 'Unknown Candidate')
                    candidate_skills = candidate.get('skills', [])
                    if isinstance(candidate_skills, str):
                        candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                    candidate_skills_lower = [s.lower() for s in candidate_skills if s]
                   
                    # Accept if candidate has ANY matching skill
                    if job_required_skills:
                        job_skills_lower = [s.lower() for s in job_required_skills]
                        has_any_match = any(skill in ' '.join(candidate_skills_lower) or any(skill in cs or cs in skill for cs in candidate_skills_lower)
                                          for skill in job_skills_lower)
                        if has_any_match:
                            relaxed_filtered.append(candidate)
                            logger.debug(f"[DEBUG] RELAXED ACCEPT {candidate_name}: Has at least one matching skill")
                    else:
                        # No required skills, accept all
                        relaxed_filtered.append(candidate)
               
                filtered_candidates = relaxed_filtered[:top_k * 5]  # Limit to reasonable number
                logger.info(f"[DEBUG] Relaxed filtering found {len(filtered_candidates)} candidates")
        else:
            logger.info(f"[DEBUG] Skipping strict pre-filtering (no required skills or domain detected)")
       
        # If still no candidates, use original filtered list from smart filtering
        if len(filtered_candidates) == 0:
            logger.warning(f"[DEBUG] Still no candidates after relaxed filtering. Using smart-filtered candidates.")
            filtered_candidates = _smart_filter_candidates(self.candidates, job_description, top_k * 5)
       
        # Initialize TF-IDF if not already done
        if len(filtered_candidates) > 0:
            if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
                self._tfidf_vectorizer, self._tfidf_matrix = _initialize_tfidf_vectorizer(filtered_candidates)
           
            # Calculate similarities in parallel
            scores = _parallel_candidate_scoring(filtered_candidates, job_description, self._tfidf_vectorizer, self._tfidf_matrix)
        else:
            logger.error(f"[DEBUG] No candidates available for scoring!")
            scores = []
       
        # Sort by score and take top candidates
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:top_k]]
       
        # Build results
        results = []
        for idx in top_indices:
            candidate = filtered_candidates[idx]
            score = scores[idx][1] * 100  # Convert to percentage
           
            # FINAL SAFETY CHECK: Verify TECHNICAL skill match before including in results
            if job_required_skills:
                candidate_skills = candidate.get('skills', [])
                if isinstance(candidate_skills, str):
                    candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                
                # Filter to technical skills only
                candidate_technical_skills = _filter_technical_skills_only(candidate_skills)
                candidate_technical_skills_lower = [str(s).lower().strip() for s in candidate_technical_skills if s]
                job_skills_lower = [str(s).lower().strip() for s in job_required_skills if s]
               
                # Calculate final TECHNICAL skill match
                matches = sum(1 for skill in job_skills_lower if any(skill in cs or cs in skill for cs in candidate_technical_skills_lower))
                final_skill_match = matches / len(job_skills_lower) if job_skills_lower else 0.0
               
                # Skip candidates with zero TECHNICAL skill match
                if final_skill_match == 0.0:
                    candidate_name = candidate.get('full_name') or candidate.get('FullName') or candidate.get('name', 'Unknown')
                    logger.warning(f"[DEBUG] FINAL CHECK: Skipping {candidate_name} - zero technical skill match (candidate technical skills: {candidate_technical_skills_lower[:5]}, required: {job_skills_lower[:5]})")
                    continue
                else:
                    matched_skills = [skill for skill in job_skills_lower if any(skill in cs or cs in skill for cs in candidate_technical_skills_lower)]
                    logger.debug(f"[DEBUG] FINAL CHECK: Including candidate with {final_skill_match:.2%} technical skill match (matched: {matched_skills})")
       
            # Enhanced scoring with additional factors
            enhanced_score = self._calculate_enhanced_score(candidate, job_description, score)
            match_percentage = round(min(enhanced_score, 80.0), 1)  # Cap at 80% maximum
            grade = self.get_grade(enhanced_score)

            normalized = _normalize_candidate_profile(candidate)
            domain_tag = normalized['domain_tag'] or self._detect_category(candidate)
            experience_years = normalized['total_experience_years']
            experience_display = str(experience_years) if experience_years else str(
                candidate.get('Experience') or candidate.get('experience') or experience_years
            )
            skills = normalized['skills'] or (candidate.get('skills') if isinstance(candidate.get('skills'), list) else [])
            if isinstance(skills, str):
                skills = _normalize_to_list(skills)
           
            # CRITICAL: Filter skills to TECHNICAL ONLY for display - remove soft skills
            skills = _filter_technical_skills_only(skills)
           
            # Order skills with required skills first
            if job_required_skills and skills:
                skills = _order_skills_with_required_first(skills, job_required_skills)
           
            top_skills = skills[:5] if isinstance(skills, list) else []
            certifications = normalized['certifications']
            designations = normalized['designations_with_experience']
            current_position = normalized['current_position']
            # Enhanced location resolution with fallback
            location = normalized['location'] or candidate.get('location') or candidate.get('Location') or candidate.get('current_location') or candidate.get('city') or 'Unknown'
            resume_text = normalized['resume_text'] or candidate.get('resume_text', '')
            source_url = normalized['source_url']
            if not source_url:
                source_url = candidate.get('source_url', '')
            filename = normalized['filename']

            contact_bundle: Dict[str, Any] = {}
            for key in ('contact', 'contactInfo', 'contact_info', 'contactDetails', 'contact_details'):
                value = candidate.get(key)
                if isinstance(value, dict):
                    for c_key, c_val in value.items():
                        if c_val is not None and c_key not in contact_bundle:
                            contact_bundle[c_key] = c_val

            def _resolve_email() -> str:
                email_candidates = [
                    normalized['email'],
                    candidate.get('email'),
                    candidate.get('Email'),
                    candidate.get('email_address'),
                    candidate.get('EmailAddress'),
                    candidate.get('primary_email'),
                    candidate.get('primaryEmail'),
                    candidate.get('primary_email_address'),
                    candidate.get('primaryEmailAddress'),
                    candidate.get('contact_email'),
                    candidate.get('contactEmail'),
                    candidate.get('work_email'),
                    candidate.get('workEmail'),
                    candidate.get('personal_email'),
                    candidate.get('personalEmail'),
                    candidate.get('business_email'),
                    candidate.get('businessEmail'),
                    contact_bundle.get('email'),
                    contact_bundle.get('Email'),
                    contact_bundle.get('emailAddress'),
                    contact_bundle.get('EmailAddress'),
                    contact_bundle.get('primaryEmail'),
                    contact_bundle.get('primary_email'),
                    contact_bundle.get('workEmail'),
                    contact_bundle.get('personalEmail'),
                ]
                for val in email_candidates:
                    if isinstance(val, str):
                        stripped = val.strip()
                        if stripped and stripped.lower() != 'unknown':
                            return stripped
                for key in ('emails', 'Emails', 'emailAddresses', 'EmailAddresses', 'email_addresses', 'contact_emails'):
                    emails_value = candidate.get(key) or contact_bundle.get(key)
                    if isinstance(emails_value, list):
                        for email_entry in emails_value:
                            if isinstance(email_entry, str):
                                stripped = email_entry.strip()
                                if stripped and stripped.lower() != 'unknown':
                                    return stripped
                return 'unknown'

            def _resolve_phone() -> str:
                phone_candidates = [
                    normalized['phone'],
                    candidate.get('phone'),
                    candidate.get('Phone'),
                    candidate.get('phone_number'),
                    candidate.get('phoneNumber'),
                    candidate.get('PhoneNumber'),
                    candidate.get('primary_phone'),
                    candidate.get('primaryPhone'),
                    candidate.get('PrimaryPhone'),
                    candidate.get('mobile'),
                    candidate.get('Mobile'),
                    candidate.get('mobile_number'),
                    candidate.get('mobileNumber'),
                    candidate.get('contact_number'),
                    candidate.get('contactNumber'),
                    candidate.get('ContactNumber'),
                    contact_bundle.get('phone'),
                    contact_bundle.get('Phone'),
                    contact_bundle.get('mobile'),
                    contact_bundle.get('Mobile'),
                    contact_bundle.get('contactNumber'),
                    contact_bundle.get('phoneNumber')
                ]
                for val in phone_candidates:
                    if isinstance(val, str):
                        stripped = val.strip()
                        if stripped and stripped.lower() not in ('unknown', 'not provided', 'n/a', 'na', ''):
                            return stripped
                    elif isinstance(val, (int, float)):
                        return str(int(val))
                return 'Not provided'

            email_value = _resolve_email()
            phone_value = _resolve_phone()
           
            # Enhanced name resolution with comprehensive fallback
            resolved_name = (
                normalized['full_name'] or
                candidate.get('full_name') or
                candidate.get('FullName') or
                candidate.get('fullName') or
                candidate.get('name') or
                candidate.get('Name') or
                candidate.get('candidate_name') or
                'Unknown'
            )
           
            results.append({
                'email': email_value,
                'full_name': resolved_name,
                'FullName': resolved_name,
                'phone': phone_value,
                'match_percentage': match_percentage,
                'Score': match_percentage,
                'grade': grade,
                'Grade': grade,
                'category': domain_tag,
                'domain_tag': domain_tag,
                'domain': domain_tag,
                'skills': skills,
                'Skills': skills,
                'top_skills': top_skills,
                'experience_years': experience_years,
                'total_experience_years': experience_years,
                'Experience': experience_display,
                'experience': experience_display,
                'designations_with_experience': designations,
                'designationsWithExperience': designations,
                'certifications': certifications,
                'Certifications': certifications,
                'education': normalized['education'],
                'Education': normalized['education'],
                'current_position': current_position,
                'title': current_position or candidate.get('title') or candidate.get('Title'),
                'location': location,
                'filename': filename,
                'source_url': source_url,
                'sourceUrl': source_url,
                'sourceURL': source_url,
                'resume_text': resume_text,
                'resumeText': resume_text,
                'match_explanation': f"TF-IDF similarity: {round(score, 1)}% + enhancements",
                'MatchExplanation': f"TF-IDF similarity: {round(score, 1)}% + enhancements",
                'contactInfo': {
                    'email': email_value,
                    'phone': phone_value
                }
            })
       
        # Remove duplicates before returning
        results = self._deduplicate_candidates(results)
        
        elapsed_time = time.time() - start_time
        
        # If no results found, get suggested candidates based on technical skills only
        if len(results) == 0:
            logger.info("No matches found. Getting suggested candidates based on technical skills only...")
            suggested_candidates = self._get_suggested_candidates(job_description, top_k=5)
            
                # Format suggested candidates with enhanced information
            for candidate in suggested_candidates:
                normalized = _normalize_candidate_profile(candidate)
                # Filter skills to technical only
                all_skills = normalized.get('skills', [])
                technical_skills = _filter_technical_skills_only(all_skills)
                
                # Calculate similarity score for suggested candidates (0-30% range)
                similarity_score = self._calculate_suggestion_similarity(job_description, technical_skills)
                
                # Create result entry for suggested candidate
                suggested_result = {
                    'email': normalized.get('email', 'unknown'),
                    'full_name': normalized.get('full_name', 'Unknown'),
                    'FullName': normalized.get('full_name', 'Unknown'),
                    'phone': normalized.get('phone', 'Not provided'),
                    'match_percentage': similarity_score,  # Suggested candidates get similarity score
                    'Score': similarity_score,
                    'grade': 'Suggested',
                    'Grade': 'Suggested',
                    'category': normalized.get('domain_tag', 'General'),
                    'domain_tag': normalized.get('domain_tag', 'General'),
                    'skills': technical_skills,  # Only technical skills
                    'Skills': technical_skills,
                    'experience_years': normalized.get('total_experience_years', 0),
                    'location': normalized.get('location', 'Unknown'),
                    'is_suggested': True,  # Flag to indicate this is a suggestion
                    'suggestion_reason': f'No exact matches found. Showing candidate with {similarity_score:.1f}% technical skill similarity.',
                    'similarity_breakdown': self._get_similarity_breakdown(job_description, technical_skills)
                }
                results.append(suggested_result)
            
            logger.info(f"Optimized keyword search completed in {elapsed_time:.2f}s, found {len(results)} suggested candidates")
            return results, f"No exact matches found. Showing {len(results)} suggested candidates with similar technical skills."
        
        logger.info(f"Optimized keyword search completed in {elapsed_time:.2f}s, found {len(results)} unique candidates")
        return results, f"Found {len(results)} candidates using optimized algorithm in {elapsed_time:.2f}s"
   
    def extract_keywords(self, text):
        """Extract keywords from text"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
   
    def calculate_simple_score(self, candidate, keywords):
        """Calculate simple matching score"""
        candidate_text = f"{candidate.get('full_name', '')} {candidate.get('resume_text', '')} {' '.join(candidate.get('skills', []))}"
        candidate_text = candidate_text.lower()
       
        matches = sum(1 for keyword in keywords if keyword in candidate_text)
        return min(100, (matches / len(keywords)) * 100) if keywords else 0
   
    def get_grade(self, score):
        """Get grade based on score"""
        if score >= 80:
            return "Grade A"
        elif score >= 60:
            return "Grade B"
        elif score >= 40:
            return "Grade C"
        else:
            return "Grade D"
   
    def detect_domain(self, keywords):
        """Detect domain based on keywords - same logic as SearchService"""
        if not keywords:
            return "General"
       
        # Enhanced domain detection logic with comprehensive keywords
        tech_keywords = {
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node', 'express',
            'spring', 'hibernate', 'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'nosql', 'firebase',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'ci/cd',
            'git', 'github', 'gitlab', 'bitbucket', 'api', 'rest', 'graphql', 'json', 'xml',
            'frontend', 'backend', 'fullstack', 'devops', 'sre', 'networking', 'tcp/ip', 'dns', 'http',
            'https', 'cloud', 'serverless', 'lambda', 'ec2', 's3', 'vpc', 'rds', 'cloudfront', 'elastic beanstalk',
            'machine learning', 'deep learning', 'nlp', 'computer vision', 'neural networks',
            'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy', 'data science',
            'data engineering', 'etl', 'big data', 'spark', 'hadoop', 'kafka', 'airflow', 'mlops',
            'cybersecurity', 'penetration testing', 'vulnerability assessment', 'firewall', 'ids', 'ips',
            'ethical hacking', 'encryption', 'security analyst', 'information security', 'siem',
            'system administrator', 'linux', 'bash', 'powershell', 'windows server', 'active directory',
            'virtualization', 'vmware', 'hyper-v', 'containers', 'microservices', 'service mesh',
            'api gateway', 'load balancer', 'reverse proxy', 'ssl', 'tls', 'certificates',
            'observability', 'monitoring', 'logging', 'prometheus', 'grafana', 'elasticsearch', 'splunk',
            'software engineer', 'developer', 'programmer', 'software architect', 'technical lead',
            'qa', 'quality assurance', 'test automation', 'selenium', 'cypress', 'junit', 'testng',
            'scrum', 'agile', 'kanban', 'jira', 'confluence', 'project management', 'product management',
            'ux', 'ui', 'design systems', 'responsive design', 'accessibility', 'wireframes',
            'figma', 'adobe xd', 'version control', 'containerization', 'release engineering',
            'refactoring', 'code review', 'clean code', 'oop', 'functional programming',
            'data structures', 'algorithms', 'system design', 'design patterns', 'low-level design',
            'high-level design', 'load testing', 'performance tuning', 'scalability', 'reliability',
            'availability', 'fault tolerance', 'disaster recovery', 'backup', 'integration testing',
            'unit testing', 'acceptance testing', 'regression testing', 'a/b testing',
            'command line', 'cli tools', 'dev environment', 'ide', 'intellij', 'vscode', 'eclipse',
            'visual studio', 'notepad++', 'debugging', 'profiling', 'deployment', 'release',
            'continuous integration', 'continuous deployment', 'infrastructure as code', 'cloud-native',
            'edge computing', 'blockchain', 'web3', 'solidity', 'smart contracts', 'crypto', 'bitcoin',
            'ethereum', 'iot', 'robotics', 'embedded systems', 'firmware', 'raspberry pi', 'arduino',
            'low-code', 'no-code', 'platform engineering', 'data lake', 'data warehouse',
            'business intelligence', 'tableau', 'power bi', 'lookml', 'metabase', 'snowflake',
            'dbt', 'redshift', 'athena', 'clickhouse', 'databricks', 'elastic stack', 'logstash',
            'kibana', 'telemetry', 'incident management', 'on-call', 'sla', 'slo', 'slis'
        }
       
        healthcare_keywords = {
            'nursing', 'rn', 'lpn', 'nurse practitioner', 'registered nurse', 'licensed practical nurse',
            'medical', 'clinical', 'hospital', 'patient', 'care', 'healthcare', 'doctor', 'physician',
            'surgeon', 'therapist', 'occupational therapist', 'physical therapist', 'speech therapist',
            'pharmacist', 'pharmacy', 'technician', 'lab tech', 'x-ray tech', 'radiologic technologist',
            'ultrasound technician', 'sonographer', 'radiology', 'mri', 'ct scan', 'imaging', 'diagnostics',
            'treatment', 'medication', 'medication administration', 'dosage', 'prescription', 'drug',
            'charting', 'electronic health record', 'emr', 'ehr', 'epic', 'cerner', 'meditech',
            'vitals', 'blood pressure', 'pulse', 'respiration', 'temperature', 'oxygen saturation',
            'icu', 'ccu', 'ward', 'inpatient', 'outpatient', 'emergency', 'er', 'triage',
            'ambulance', 'paramedic', 'emt', 'first responder', 'bcls', 'acls', 'cpr',
            'infection control', 'aseptic technique', 'sterilization', 'hand hygiene',
            'wound care', 'dressing change', 'surgical wound', 'incision', 'suturing',
            'patient care', 'patient safety', 'discharge planning', 'care coordination',
            'home healthcare', 'visiting nurse', 'hospice', 'palliative care',
            'nursing home', 'long-term care', 'assisted living', 'geriatrics',
            'pediatrics', 'neonatal', 'nicu', 'labor and delivery', 'ob/gyn', 'midwife',
            'mental health', 'psychiatry', 'psychology', 'behavioral health',
            'substance abuse', 'addiction treatment', 'detox', 'rehab', 'counseling',
            'social work', 'case management', 'insurance', 'medicare', 'medicaid',
            'claims', 'billing', 'medical coding', 'icd-10', 'cpt', 'hipaa',
            'compliance', 'regulatory', 'quality assurance', 'joint commission',
            'clinical documentation', 'progress notes', 'care plan', 'assessment',
            'evaluation', 'diagnosis', 'disease management', 'chronic care',
            'diabetes management', 'hypertension', 'cardiology', 'oncology',
            'nephrology', 'pulmonology', 'neurology', 'gastroenterology',
            'hepatology', 'endocrinology', 'orthopedics', 'pain management',
            'anesthesiology', 'surgery', 'pre-op', 'post-op', 'perioperative',
            'scrub nurse', 'circulating nurse', 'anesthetist', 'surgical tech',
            'occupational health', 'industrial medicine', 'telemedicine', 'telehealth',
            'virtual care', 'remote monitoring', 'health informatics', 'biomedical',
            'clinical trials', 'research nurse', 'principal investigator',
            'institutional review board', 'data collection', 'public health',
            'epidemiology', 'vaccination', 'immunization', 'screening', 'prevention',
            'wellness', 'health education', 'nutrition', 'dietician', 'fitness',
            'rehabilitation', 'physical medicine', 'mobility', 'assistive devices',
            'wheelchair', 'prosthetics', 'orthotics', 'speech pathology',
            'medical assistant', 'certified nursing assistant', 'cna',
            'medical receptionist', 'healthcare administrator', 'medical records',
            'healthcare provider', 'healthcare professional', 'continuing education',
            'board certification', 'state license', 'clinical rotation',
            'nursing school', 'preceptorship', 'internship', 'residency',
            'fellowship', 'grand rounds', 'rounding', 'consultation', 'handoff',
            'multidisciplinary team', 'interprofessional', 'patient satisfaction',
            'patient rights', 'informed consent', 'advanced directive', 'dnr',
            'code blue', 'rapid response', 'falls risk', 'pressure ulcer', 'restraints', 'cna', 'word', 'wound', 'care', 'charting'
        }
       
        # Convert keywords to lowercase for comparison
        keywords_lower = [kw.lower() for kw in keywords]
       
        # Count matches
        tech_count = 0
        healthcare_count = 0
       
        for keyword in keywords_lower:
            # Check for tech keywords
            if any(tech_kw in keyword or keyword in tech_kw for tech_kw in tech_keywords):
                tech_count += 1
            # Check for healthcare keywords
            if any(health_kw in keyword or keyword in health_kw for health_kw in healthcare_keywords):
                healthcare_count += 1
       
        # Determine domain based on counts
        if healthcare_count > tech_count and healthcare_count > 0:
            return "Healthcare"
        elif tech_count > healthcare_count and tech_count > 0:
            return "IT/Tech"
        else:
            return "General"

    def semantic_match(self, job_description, use_gpt4_reranking=True):
        """Semantic match method for fallback algorithm"""
        # Validate query - reject meaningless queries early
        if _is_meaningless_query(job_description):
            logger.warning(f"Meaningless query rejected in FallbackAlgorithm.semantic_match: '{job_description[:100] if job_description else ''}...'")
            return {
                'results': [],
                'summary': 'No candidates found. Please provide a valid job description with meaningful keywords.',
                'total_candidates': 0,
                'search_query': job_description,
                'algorithm_used': 'Fallback Algorithm',
                'fallback': True,
                'query_rejected': True
            }
        
        try:
            logger.info(f"[DEBUG] FallbackAlgorithm.semantic_match() called")
            logger.info(f"[DEBUG] This will call keyword_search() which HAS strict pre-filtering")
            logger.info(f"Fallback algorithm: Performing semantic match for query: {job_description[:100]}...")
           
            # Use keyword search as fallback (THIS HAS STRICT PRE-FILTERING)
            results, summary = self.keyword_search(job_description, top_k=20)
            logger.info(f"[DEBUG] keyword_search returned {len(results)} results")
           
            return {
                'results': results,
                'summary': summary,
                'total_candidates': len(results),
                'search_query': job_description,
                'algorithm_used': 'Fallback Algorithm',
                'fallback': True
            }
           
        except Exception as e:
            logger.error(f"Fallback algorithm error: {e}", exc_info=True)
            return {
                'results': [],
                'summary': f"Fallback search failed: {str(e)}",
                'total_candidates': 0,
                'search_query': job_description,
                'algorithm_used': 'Fallback Algorithm',
                'fallback': True,
                'error': True
            }
   
    def _calculate_enhanced_score(self, candidate, job_description, base_score):
        """Calculate enhanced score with additional factors"""
        try:
            # Base TF-IDF score
            enhanced_score = base_score
           
            # Skill matching bonus
            candidate_skills = candidate.get('skills', [])
            job_skills = self._extract_skills_from_job(job_description)
            skill_match_ratio = self._calculate_skill_match_ratio(candidate_skills, job_skills)
            enhanced_score += skill_match_ratio * 10  # Up to 10% bonus
           
            # Experience matching
            experience_bonus = self._calculate_experience_bonus(candidate, job_description)
            enhanced_score += experience_bonus * 5  # Up to 5% bonus
           
            # Location matching
            location_bonus = self._calculate_location_bonus(candidate, job_description)
            enhanced_score += location_bonus * 3  # Up to 3% bonus
           
            # Education level bonus
            education_bonus = self._calculate_education_bonus(candidate, job_description)
            enhanced_score += education_bonus * 2  # Up to 2% bonus

            # Domain alignment bonus/penalty
            domain_adjustment = self._calculate_domain_alignment(candidate, job_description)
            enhanced_score += domain_adjustment
           
            return max(0, min(enhanced_score, 80))  # Keep score between 0 and 80 (capped at 80%)
           
        except Exception as e:
            logger.error(f"Error calculating enhanced score: {e}")
            return base_score
   
    def _extract_skills_from_job(self, job_description):
        """Extract skills from job description with enhanced healthcare and critical skill detection"""
        if not job_description:
            return []
       
        job_desc_lower = job_description.lower()
        skills = set()
       
        # Common technical skills
        tech_skill_patterns = [
            r'\b(python|java|javascript|react|angular|vue|node\.?js|typescript)\b',
            r'\b(aws|azure|gcp|docker|kubernetes|terraform)\b',
            r'\b(sql|mysql|postgresql|mongodb|redis)\b',
            r'\b(git|github|gitlab|jenkins|ci/cd)\b',
            r'\b(machine learning|ml|ai|artificial intelligence|data science)\b',
            r'\b(agile|scrum|kanban|devops|microservices)\b',
        ]
       
        # Healthcare and clinical terminology (ENHANCED)
        healthcare_skill_patterns = [
            # Critical nursing credentials
            r'\b(rn|registered nurse|registered nursing)\b',
            r'\b(lpn|licensed practical nurse|lp nurse)\b',
            r'\b(cna|certified nursing assistant|nursing assistant)\b',
            r'\b(nurse practitioner|np|aprn|advanced practice registered nurse)\b',
            # Specialized units
            r'\b(icu|intensive care unit|critical care)\b',
            r'\b(ccu|cardiac care unit|coronary care)\b',
            r'\b(er|emergency room|emergency department|ed)\b',
            r'\b(med-surg|medical surgical|medical-surgical)\b',
            r'\b(ltc|long-term care|long term care)\b',
            r'\b(nicu|neonatal intensive care)\b',
            r'\b(picu|pediatric intensive care)\b',
            # Critical certifications
            r'\b(bls|basic life support)\b',
            r'\b(acls|advanced cardiac life support)\b',
            r'\b(cpr|cardiopulmonary resuscitation)\b',
            r'\b(pals|pediatric advanced life support)\b',
            r'\b(tncc|trauma nursing core course)\b',
            # Clinical skills
            r'\b(patient care|clinical care|direct patient care)\b',
            r'\b(medication administration|med administration|med pass)\b',
            r'\b(charting|documentation|clinical documentation)\b',
            r'\b(care plan|careplanning|care planning)\b',
            r'\b(case management|discharge planning)\b',
            r'\b(vital signs|vitals|monitoring)\b',
            r'\b(phlebotomy|blood draw|venipuncture)\b',
            r'\b(iv therapy|intravenous|iv insertion)\b',
            # Healthcare systems
            r'\b(ehr|emr|electronic health record|electronic medical record)\b',
            r'\b(epic|cerner|meditech|allscripts|athenahealth)\b',
            # Healthcare roles and settings
            r'\b(healthcare|health care|medical|hospital|clinic|clinical)\b',
            r'\b(nursing|nurse|patient safety|patient advocacy)\b',
            r'\b(therapist|occupational therapy|physical therapy|respiratory therapy)\b',
        ]
       
        # Extract all skills
        for pattern in tech_skill_patterns + healthcare_skill_patterns:
            matches = re.findall(pattern, job_desc_lower)
            if isinstance(matches, list):
                skills.update(matches)
            elif matches:
                skills.add(matches)
       
        # Also extract from "Requirements:" section more explicitly
        requirements_section = re.search(r'(?:requirements?|must have|required)[\s:]+([^\.]+)', job_desc_lower, re.IGNORECASE)
        if requirements_section:
            req_text = requirements_section.group(1)
            # Look for certifications and licenses
            cert_patterns = [
                r'\b(rn|registered nurse)\b',
                r'\b(bls|acls|cpr|pals|tncc)\b',
                r'\b(license|licensed|certification|certified)\b',
            ]
            for pattern in cert_patterns:
                matches = re.findall(pattern, req_text)
                if matches:
                    skills.update(matches)
       
        # Normalize and expand healthcare skill variations
        normalized_skills = set()
        healthcare_variations = _get_healthcare_skill_variations()
       
        for skill in skills:
            skill_lower = skill.lower().strip()
            # Check if this skill has known variations
            found_variation = False
            for key, variants in healthcare_variations.items():
                if skill_lower == key or skill_lower in variants:
                    # Add the canonical form (first in the list) and the original
                    normalized_skills.add(variants[0])  # Add canonical form
                    normalized_skills.add(skill_lower)  # Keep original
                    found_variation = True
                    break
           
            if not found_variation:
                # Keep non-healthcare skills as-is
                normalized_skills.add(skill_lower)
       
        return list(normalized_skills)

    def _calculate_domain_alignment(self, candidate: Dict[str, Any], job_description: str) -> float:
        """Boost or penalize candidates based on domain alignment with the job description."""
        try:
            # Determine job domain from description keywords
            job_keywords = re.findall(r'\b\w+\b', job_description.lower())
            job_domain = self.detect_domain(job_keywords)
            job_domain_lower = (job_domain or '').strip().lower()

            if not job_domain_lower or job_domain_lower == 'general':
                return 0.0

            # Determine candidate domain
            candidate_domain = (
                candidate.get('domain_tag') or
                candidate.get('category') or
                candidate.get('domain') or
                ''
            )
            candidate_domain_lower = str(candidate_domain).strip().lower()

            if not candidate_domain_lower or candidate_domain_lower == 'general':
                candidate_domain_lower = str(self._detect_category(candidate) or '').strip().lower()

            if job_domain_lower == candidate_domain_lower:
                # Strong reward for perfect alignment
                return 12.0

            if job_domain_lower == 'healthcare':
                # STRICT: Heavily penalize IT-focused candidates for healthcare roles
                if candidate_domain_lower in {'it/tech', 'technology', 'tech', 'it', 'software', 'engineering'}:
                    return -50.0  # Very large penalty to push them out of results
            if job_domain_lower in {'it/tech', 'technology', 'tech'}:
                if candidate_domain_lower in {'healthcare', 'medical', 'nursing'}:
                    return -30.0  # Large penalty for non-tech candidates in tech jobs

            # Penalty for other mismatches
                    return -10.0

        except Exception as e:
            logger.error(f"Error calculating domain alignment: {e}")
            return 0.0
   
    def _calculate_skill_match_ratio(self, candidate_skills, job_skills):
        """Calculate skill match ratio with improved matching logic"""
        if not job_skills:
            return 0.0
       
        if not candidate_skills:
            return 0.0
       
        # Normalize candidate skills
        if isinstance(candidate_skills, str):
            candidate_skills = [s.strip() for s in candidate_skills.split(',')]
        candidate_skills_lower = [str(skill).lower().strip() for skill in candidate_skills if skill]
       
        # Normalize job skills
        if isinstance(job_skills, str):
            job_skills = [s.strip() for s in job_skills.split(',')]
        job_skills_lower = [str(skill).lower().strip() for skill in job_skills if skill]
       
        # Improved matching: check for exact match, substring match, or keyword match
        matches = 0
        for job_skill in job_skills_lower:
            # Exact match
            if job_skill in candidate_skills_lower:
                matches += 1
                continue
           
            # Substring match (e.g., "rn" matches "registered nurse")
            if any(job_skill in cs or cs in job_skill for cs in candidate_skills_lower):
                matches += 1
                continue
           
            # Keyword match for multi-word skills (e.g., "registered nurse" matches "rn")
            job_keywords = job_skill.split()
            if len(job_keywords) > 1:
                # Check if all keywords appear in candidate skills
                if all(any(kw in cs for cs in candidate_skills_lower) for kw in job_keywords if len(kw) > 2):
                    matches += 1
       
        return matches / len(job_skills_lower) if job_skills_lower else 0.0
   
    def _calculate_experience_bonus(self, candidate, job_description):
        """Calculate experience bonus with enhanced accuracy"""
        try:
            # Use the enhanced experience match function for consistency
            return self._calculate_experience_match(candidate, job_description)
               
        except Exception as e:
            logger.error(f"Error calculating experience bonus: {e}")
            return 0.0
   
    def _calculate_location_bonus(self, candidate, job_description):
        """Calculate location bonus with enhanced accuracy"""
        try:
            # Use the enhanced location match function for consistency
            return self._calculate_location_match(candidate, job_description)
           
        except Exception as e:
            logger.error(f"Error calculating location bonus: {e}")
            return 0.0
   
    def _calculate_education_bonus(self, candidate, job_description):
        """Calculate education bonus with enhanced accuracy"""
        try:
            # Extract education from multiple candidate fields
            education_sources = [
                candidate.get('education'),
                candidate.get('Education'),
                candidate.get('qualification'),
                candidate.get('qualifications'),
                candidate.get('degree'),
                candidate.get('degrees')
            ]
           
            education_text = ''
            for source in education_sources:
                if source:
                    if isinstance(source, list):
                        education_text += ' '.join(str(e) for e in source) + ' '
                    else:
                        education_text += str(source) + ' '
           
            education_text = education_text.lower().strip()
           
            if not education_text:
                return 0.2  # Low score if no education data
           
            # Extract education requirements from job description
            job_desc_lower = job_description.lower()
           
            # Degree level requirements
            degree_keywords = {
                'phd': ['phd', 'ph.d', 'doctorate', 'doctoral', 'd.phil'],
                'master': ['master', 'masters', 'ms', 'm.s', 'mba', 'm.sc', 'm.a', 'm.ed'],
                'bachelor': ['bachelor', 'bachelors', 'bs', 'b.s', 'ba', 'b.a', 'b.sc', 'b.tech', 'b.e', 'bachelor\'s'],
                'associate': ['associate', 'a.a', 'a.s', 'diploma'],
                'degree': ['degree', 'graduate', 'undergraduate', 'college']
            }
           
            # Check what degree level is required
            required_level = None
            for level, keywords in degree_keywords.items():
                if any(keyword in job_desc_lower for keyword in keywords):
                    required_level = level
                    break
           
            # Check candidate's education level
            candidate_level = None
            candidate_level_score = {}
           
            for level, keywords in degree_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in education_text)
                if matches > 0:
                    candidate_level_score[level] = matches
                    if candidate_level is None or level in ['phd', 'master', 'bachelor']:
                        candidate_level = level
           
            # Scoring logic
            if required_level is None:
                # No specific requirement - check if candidate has any degree
                if candidate_level:
                    return 0.8  # Good score for having a degree
            else:
                    return 0.4  # Lower score if no degree mentioned
           
            # Match required level
            level_hierarchy = {'phd': 4, 'master': 3, 'bachelor': 2, 'associate': 1, 'degree': 1}
            required_rank = level_hierarchy.get(required_level, 0)
            candidate_rank = level_hierarchy.get(candidate_level, 0)
           
            if candidate_rank >= required_rank:
                if candidate_rank == required_rank:
                    return 1.0  # Exact match
                else:
                    return 0.95  # Overqualified (still good)
            elif candidate_rank >= required_rank - 1:
                return 0.70  # Close (e.g., bachelor when master required)
            else:
                return 0.40  # Below requirement
           
        except Exception as e:
            logger.error(f"Error calculating education bonus: {e}")
            return 0.3
   
    def _detect_category(self, candidate):
        """Detect candidate category based on skills and experience"""
        try:
            skills = candidate.get('skills', [])
            experience = candidate.get('experience', '')
            education = candidate.get('education', '')
           
            combined_text = f"{' '.join(skills)} {experience} {education}".lower()
           
            # Healthcare keywords
            healthcare_keywords = ['healthcare', 'medical', 'nursing', 'doctor', 'physician', 'hospital', 'clinic', 'patient', 'health', 'medicine', 'pharmacy', 'therapist', 'counselor']
            if any(keyword in combined_text for keyword in healthcare_keywords):
                return 'Healthcare'
           
            # Finance keywords
            finance_keywords = ['finance', 'banking', 'accounting', 'financial', 'investment', 'trading', 'audit', 'tax', 'budget', 'revenue', 'profit']
            if any(keyword in combined_text for keyword in finance_keywords):
                return 'Finance'
           
            # Education keywords
            education_keywords = ['education', 'teaching', 'teacher', 'professor', 'instructor', 'academic', 'university', 'college', 'school', 'student', 'learning']
            if any(keyword in combined_text for keyword in education_keywords):
                return 'Education'
           
            # Technology keywords (default)
            tech_keywords = ['software', 'programming', 'development', 'engineering', 'technology', 'computer', 'data', 'analyst', 'developer', 'engineer']
            if any(keyword in combined_text for keyword in tech_keywords):
                return 'IT/Tech'
           
            return 'IT/Tech'  # Default category
           
        except Exception as e:
            logger.error(f"Error detecting category: {e}")
            return 'IT/Tech'

class AdeptAIMastersAlgorithm:
    """Main algorithm class that uses the original AdeptAI implementation"""
   
    def __init__(self):
        self.performance_stats = {
            'total_searches': 0,
            'avg_response_time': 0,
            'original_algorithm_used': 0,
            'fallback_used': 0
        }
       
        # Initialize the original algorithm if available
        if ORIGINAL_ALGORITHM_AVAILABLE:
            try:
                # Initialize the enhanced recruitment search system
                self.enhanced_system = EnhancedRecruitmentSearchSystem()
               
                # Initialize performance monitoring
                self.performance_monitor = PerformanceMonitor()
               
                # Initialize caching system
                self.embedding_cache = EmbeddingCache()
               
                # Initialize batch processor
                self.batch_processor = BatchProcessor()
               
                # Initialize advanced utils
                self.skill_matcher = AdvancedSkillMatcher()
                self.query_parser = NaturalLanguageQueryParser()
                self.multi_model_embeddings = MultiModelEmbeddingService()
                self.utils_cache = UtilsEmbeddingCache()
               
                # Initialize additional components (if available)
                self.domain_integration = DomainIntegration() if DomainIntegration else None
                self.bias_prevention = BiasPrevention() if BiasPrevention else None
               
                logger.info("Original AdeptAI algorithm initialized successfully")
               
            except Exception as e:
                logger.error(f"Error initializing original algorithm: {e}")
                self.enhanced_system = None
        else:
            self.enhanced_system = None
            logger.warning("Original algorithm not available, using fallback")
       
        # Initialize fallback algorithm
        self.fallback_algorithm = FallbackAlgorithm()
       
        # Load candidates from DynamoDB if available
        # Use background loading for large datasets
        if os.getenv('BACKGROUND_LOADING', 'false').lower() == 'true':
            self._load_candidates_background()
        else:
            self._load_candidates_from_dynamodb()
   
    def _load_candidates_from_dynamodb(self):
        """Load candidates from DynamoDB with optimizations for large datasets"""
        if table:
            try:
                # Check if we have a cached index first (only if not forcing full load)
                if not FORCE_FULL_LOAD and self._check_cached_index():
                    logger.info("Using cached candidate index - skipping DynamoDB reload")
                    return
               
                # Get total count first to estimate progress
                total_count = self._get_total_candidate_count()
                logger.info(f"Estimated total candidates in DynamoDB: {total_count:,}")
               
                # For very large datasets, use sampling or limits
                max_candidates = self._get_max_candidates_to_load(total_count)
                if max_candidates < total_count:
                    logger.info(f"Large dataset detected. Loading sample of {max_candidates:,} candidates out of {total_count:,}")
               
                candidates = self._load_candidates_optimized(max_candidates)
               
                if not candidates:
                    logger.warning("No candidates loaded from DynamoDB")
                    return
               
                logger.info(f"Successfully loaded {len(candidates):,} candidates from DynamoDB")
               
                # Store candidates in fallback algorithm
                self.fallback_algorithm.candidates = candidates
               
                # If original algorithm is available, index the candidates
                if self.enhanced_system and candidates:
                    try:
                        logger.info("Starting to index candidates in enhanced system...")
                        self.enhanced_system.index_candidates(candidates)
                        logger.info("Candidates indexed in original algorithm")
                    except Exception as e:
                        logger.error(f"Error indexing candidates: {e}")
                       
            except Exception as e:
                logger.error(f"Error loading candidates from DynamoDB: {e}")
        else:
            logger.warning("DynamoDB not available, using empty candidate list")
   
    def _check_cached_index(self):
        """Check if we have a recent cached index"""
        try:
            import os
            import time
           
            # Check if index files exist and are recent (less than 1 hour old)
            index_files = [
                "enhanced_search_index.faiss",
                "enhanced_search_index_candidates.pkl",
                "enhanced_search_index_embeddings.npy"
            ]
           
            for file_path in index_files:
                if not os.path.exists(file_path):
                    return False
               
                # Check if file is older than 1 hour
                if time.time() - os.path.getmtime(file_path) > 3600:
                    return False
           
            return True
        except:
            return False
   
    def _get_total_candidate_count(self):
        """Get approximate total count of candidates with pagination"""
        # Use fast approximate counting with in-memory cache; fall back to scan as last resort
        try:
            import time
            global _COUNT_CACHE
        except Exception:
            _COUNT_CACHE = None  # type: ignore

        try:
            now_ts = time.time()
            # 2-hour TTL cache
            if isinstance(globals().get('_COUNT_CACHE'), dict):
                cached = globals()['_COUNT_CACHE']
                if now_ts - cached.get('ts', 0) < 7200 and cached.get('value', 0) > 0:
                    return int(cached['value'])

            # Fast path: DynamoDB DescribeTable ItemCount (eventually consistent)
            try:
                client = table.meta.client
                desc = client.describe_table(TableName=table.name)
                item_count = int(desc.get('Table', {}).get('ItemCount', 0))
                globals()['_COUNT_CACHE'] = {'value': item_count, 'ts': now_ts}
                logger.info(f"Estimated total candidates in DynamoDB (DescribeTable): {item_count:,}")
                return item_count
            except Exception as e:
                logger.warning(f"DescribeTable ItemCount failed, falling back to paginated count: {e}")

            # Slow fallback (rarely used)
            total_count = 0
            last_evaluated_key = None
            logger.info("Counting total candidates in DynamoDB (fallback scan)...")
            while True:
                scan_params = {'Select': 'COUNT'}
                if last_evaluated_key:
                    scan_params['ExclusiveStartKey'] = last_evaluated_key
                response = table.scan(**scan_params)
                page_count = response.get('Count', 0)
                total_count += page_count
                logger.info(f"Count page: {page_count} candidates (Total so far: {total_count})")
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            globals()['_COUNT_CACHE'] = {'value': total_count, 'ts': now_ts}
            logger.info(f"Total candidates in DynamoDB: {total_count:,}")
            return total_count
        except Exception as e:
            logger.error(f"Error counting candidates: {e}")
            return 0
   
    def _get_max_candidates_to_load(self, total_count):
        """Determine how many candidates to load based on dataset size - ULTRA-FAST MODE"""
        # Super-fast first response: smaller sample for the first request only
        try:
            global _FIRST_REQUEST_DONE
        except Exception:
            _FIRST_REQUEST_DONE = False  # type: ignore

        if not globals().get('_FIRST_REQUEST_DONE', False):
            sample_size = 100
            logger.info(f"First-request FAST START: loading {sample_size} candidates for immediate response")
            return sample_size
        # If force full load is enabled, load everything
        if FORCE_FULL_LOAD:
            logger.info("FORCE_FULL_LOAD enabled - loading all candidates")
            return total_count
       
        # ULTRA-FAST MODE: Load minimal candidates for immediate results
        if ULTRA_FAST_MODE:
            sample_size = FAST_LOAD_CANDIDATES
            logger.info(f"ULTRA-FAST MODE: Loading only {sample_size:,} candidates for immediate results (out of {total_count:,})")
            return sample_size
       
        # ULTRA-FAST LOADING: Load enough candidates to ensure 20 domain-specific results
        # Domain filtering can remove 80-90% of candidates, so we need to load many more
        if total_count > 100000:
            # For very large datasets (>100k), load 20k to ensure enough domain matches
            sample_size = 20000
            logger.info(f"Ultra-large dataset ({total_count:,} candidates) - Loading {sample_size:,} for domain-filtered results")
            return sample_size
        elif total_count > 50000:
            # For large datasets (50k-100k), load 15k
            sample_size = 15000
            logger.info(f"Large dataset ({total_count:,} candidates) - Loading {sample_size:,} for domain-filtered results")
            return sample_size
        elif total_count > 20000:
            # For medium-large datasets (20k-50k), load 10k
            sample_size = 10000
            logger.info(f"Medium-large dataset ({total_count:,} candidates) - Loading {sample_size:,} for domain-filtered results")
            return sample_size
        elif total_count > 10000:
            # For medium datasets (10k-20k), load 8k
            sample_size = 8000
            logger.info(f"Medium dataset ({total_count:,} candidates) - Loading {sample_size:,} for domain-filtered results")
            return sample_size
        else:
            # For small datasets, load all
            logger.info(f"Small dataset ({total_count:,} candidates) - loading all")
            return total_count
   
    def _load_candidates_optimized(self, max_candidates):
        """Load candidates with optimizations"""
        candidates = []
        last_evaluated_key = None
        page_count = 0
        loaded_count = 0
       
        logger.info(f"Starting optimized candidate loading... Target: {max_candidates:,} candidates")
       
        while True:
            page_count += 1
           
            # Prepare scan parameters with limit - OPTIMIZED FOR SPEED
            # Use larger page sizes for faster loading
            page_size = min(5000, max_candidates - loaded_count) if max_candidates > 0 else 5000
            scan_params = {
                'Limit': page_size
            }
            if last_evaluated_key:
                scan_params['ExclusiveStartKey'] = last_evaluated_key
           
            logger.info(f"Page {page_count}: Scanning with limit {scan_params['Limit']}")
           
            # Scan with pagination
            response = table.scan(**scan_params)
           
            # Add items from this page
            page_items = response.get('Items', [])
            candidates.extend(page_items)
            loaded_count += len(page_items)
           
            logger.info(f"Page {page_count}: Loaded {len(page_items)} candidates (Total: {loaded_count:,})")
           
            # Progress logging every 10 pages
            if page_count % 10 == 0:
                if max_candidates > 0:
                    progress = (loaded_count / max_candidates) * 100
                    logger.info(f"Progress: {loaded_count:,}/{max_candidates:,} candidates ({progress:.1f}%)")
                else:
                    logger.info(f"Progress: {loaded_count:,} candidates loaded")
           
            # Check if we've reached our limit (only if max_candidates > 0)
            if max_candidates > 0 and loaded_count >= max_candidates:
                logger.info(f"Reached target limit of {max_candidates:,} candidates")
                break
           
            # Check if there are more pages
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                logger.info("No more pages available in DynamoDB")
                break
       
        logger.info(f"Final result: Loaded {len(candidates):,} candidates across {page_count} pages")
        return candidates
   
    def _load_candidates_background(self):
        """Load candidates in background thread for non-blocking startup"""
        def background_loader():
            try:
                logger.info("Starting background candidate loading...")
                self._load_candidates_from_dynamodb()
                logger.info("Background candidate loading completed")
            except Exception as e:
                logger.error(f"Background loading failed: {e}")
       
        # Start background thread
        thread = threading.Thread(target=background_loader, daemon=True)
        thread.start()
       
        # For immediate use, load a larger sample to ensure enough candidates for domain filtering
        if table:
            try:
                logger.info("Loading initial sample for immediate use...")
                # Load more candidates initially to ensure we have enough after domain filtering
                response = table.scan(Limit=10000)  # Load 10000 for immediate use to ensure domain filtering has enough candidates
                sample_candidates = response.get('Items', [])
                self.fallback_algorithm.candidates = sample_candidates
                logger.info(f"Loaded {len(sample_candidates)} candidates for immediate use")
            except Exception as e:
                logger.error(f"Error loading sample candidates: {e}")
   
    def get_grade(self, score):
        """Get grade based on score"""
        if score >= 80:
            return "Grade A"
        elif score >= 60:
            return "Grade B"
        elif score >= 40:
            return "Grade C"
        else:
            return "Grade D"
   
    def extract_keywords(self, text):
        """Enhanced keyword extraction with better parsing and relevance scoring"""
        if not text:
            return []
       
        try:
            # Use the original algorithm's keyword extraction
            if self.enhanced_system:
                # Enhanced keyword extraction with better parsing
                keywords = self._extract_enhanced_keywords(text)
                return keywords
            else:
                # Fallback keyword extraction
                return self.fallback_algorithm.extract_keywords(text)
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
   
    def _extract_enhanced_keywords(self, text):
        """Enhanced keyword extraction with better parsing and relevance scoring"""
        if not text:
            return []
       
        # Convert to lowercase for processing
        text_lower = text.lower()
       
        # Enhanced stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'shall', 'ought', 'need', 'dare', 'used', 'able', 'about', 'above', 'across', 'after',
            'against', 'along', 'among', 'around', 'before', 'behind', 'below', 'beneath', 'beside',
            'between', 'beyond', 'during', 'except', 'inside', 'outside', 'through', 'throughout',
            'under', 'underneath', 'until', 'upon', 'within', 'without', 'up', 'down', 'out', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'now', 'also', 'back', 'even', 'still', 'well', 'much', 'many', 'little', 'less',
            'least', 'more', 'most', 'better', 'best', 'worse', 'worst', 'good', 'bad', 'great',
            'small', 'large', 'big', 'long', 'short', 'high', 'low', 'new', 'old', 'young', 'first',
            'last', 'next', 'previous', 'early', 'late', 'fast', 'slow', 'quick', 'easy', 'hard',
            'difficult', 'simple', 'complex', 'important', 'necessary', 'possible', 'available',
            'ready', 'sure', 'certain', 'clear', 'obvious', 'different', 'same', 'similar',
            'various', 'several', 'multiple', 'single', 'double', 'triple', 'half', 'quarter',
            'full', 'empty', 'complete', 'incomplete', 'finished', 'unfinished', 'done', 'undone',
            'open', 'closed', 'free', 'busy', 'available', 'unavailable', 'present', 'absent',
            'here', 'there', 'everywhere', 'nowhere', 'somewhere', 'anywhere', 'always', 'never',
            'sometimes', 'often', 'usually', 'rarely', 'seldom', 'frequently', 'occasionally',
            'recently', 'lately', 'soon', 'immediately', 'quickly', 'slowly', 'carefully',
            'easily', 'hardly', 'barely', 'almost', 'nearly', 'quite', 'rather', 'pretty',
            'fairly', 'somewhat', 'slightly', 'completely', 'totally', 'entirely', 'partly',
            'partially', 'mostly', 'mainly', 'primarily', 'especially', 'particularly',
            'specifically', 'generally', 'usually', 'normally', 'typically', 'commonly',
            'frequently', 'regularly', 'constantly', 'continuously', 'permanently',
            'temporarily', 'briefly', 'shortly', 'eventually', 'finally', 'ultimately',
            'initially', 'originally', 'previously', 'earlier', 'later', 'afterwards',
            'meanwhile', 'simultaneously', 'together', 'separately', 'individually',
            'collectively', 'jointly', 'independently', 'alone', 'together', 'apart',
            'close', 'near', 'far', 'away', 'here', 'there', 'everywhere', 'nowhere'
        }
       
        # Extract words using regex with better pattern
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text_lower)
       
        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
       
        # Extract technical terms and phrases
        technical_phrases = self._extract_technical_phrases(text_lower)
       
        # Extract job-related terms
        job_terms = self._extract_job_terms(text_lower)
       
        # Extract skill-related terms
        skill_terms = self._extract_skill_terms(text_lower)
       
        # Combine all keywords and remove duplicates
        all_keywords = filtered_words + technical_phrases + job_terms + skill_terms
        unique_keywords = list(dict.fromkeys(all_keywords))  # Preserve order while removing duplicates
       
        # Score keywords by relevance
        scored_keywords = self._score_keywords(unique_keywords, text_lower)
       
        # Return top keywords sorted by relevance
        return scored_keywords[:50]  # Limit to top 50 most relevant keywords
   
    def _extract_technical_phrases(self, text):
        """Extract technical phrases and compound terms"""
        technical_patterns = [
            r'machine learning|ml|artificial intelligence|ai',
            r'data science|data engineering|data analysis',
            r'cloud computing|cloud platform|cloud services',
            r'web development|web application|web service',
            r'mobile development|mobile app|mobile application',
            r'database design|database management|database administration',
            r'software engineering|software development|software architecture',
            r'system administration|system design|system architecture',
            r'network security|cybersecurity|information security',
            r'project management|agile methodology|scrum master',
            r'user experience|ux design|ui design',
            r'quality assurance|qa testing|test automation',
            r'devops|ci/cd|continuous integration|continuous deployment',
            r'microservices|api development|rest api|graphql',
            r'containerization|docker|kubernetes|orchestration',
            r'big data|data pipeline|etl|data warehouse',
            r'business intelligence|bi|data visualization',
            r'blockchain|cryptocurrency|distributed systems',
            r'iot|internet of things|embedded systems',
            r'augmented reality|ar|virtual reality|vr'
        ]
       
        phrases = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
        return phrases
   
    def _extract_job_terms(self, text):
        """Extract job-related terms and titles"""
        job_patterns = [
            r'senior|sr\.|lead|principal|staff|architect',
            r'junior|jr\.|entry-level|associate|graduate',
            r'mid-level|intermediate|experienced',
            r'manager|director|head|chief|vp|vice president',
            r'analyst|specialist|expert|consultant|advisor',
            r'engineer|developer|programmer|coder',
            r'designer|architect|consultant|advisor',
            r'coordinator|administrator|supervisor',
            r'researcher|scientist|technician',
            r'intern|trainee|apprentice'
        ]
       
        terms = []
        for pattern in job_patterns:
            matches = re.findall(pattern, text)
            terms.extend(matches)
        return terms
   
    def _extract_skill_terms(self, text):
        """Extract skill-related terms"""
        skill_patterns = [
            r'python|java|javascript|typescript|c\+\+|c#|go|rust|php|ruby|swift|kotlin|scala',
            r'react|angular|vue|svelte|next\.js|nuxt\.js|express\.js|django|flask|fastapi|spring|laravel|rails',
            r'mysql|postgresql|mongodb|redis|elasticsearch|cassandra|dynamodb|oracle|sql server|sqlite',
            r'aws|azure|gcp|google cloud|heroku|digitalocean|vercel',
            r'docker|kubernetes|jenkins|gitlab ci|github actions|terraform|ansible|chef|puppet',
            r'tensorflow|pytorch|scikit-learn|pandas|numpy|opencv|hugging face|transformers|langchain',
            r'jest|pytest|junit|selenium|cypress|mocha|chai',
            r'git|github|gitlab|bitbucket|svn',
            r'linux|unix|windows|macos|ubuntu|centos|debian',
            r'html|css|sass|scss|less|bootstrap|tailwind|material-ui'
        ]
       
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text)
            skills.extend(matches)
        return skills
   
    def _score_keywords(self, keywords, text):
        """Score keywords by relevance and frequency"""
        keyword_scores = {}
       
        for keyword in keywords:
            # Count frequency in text
            frequency = text.count(keyword)
           
            # Base score from frequency
            score = frequency
           
            # Boost technical terms
            if any(tech in keyword for tech in ['python', 'java', 'javascript', 'react', 'angular', 'vue', 'aws', 'docker', 'kubernetes']):
                score *= 2.0
           
            # Boost job-related terms
            if any(job in keyword for job in ['senior', 'lead', 'principal', 'architect', 'manager', 'director']):
                score *= 1.5
           
            # Boost skill terms
            if any(skill in keyword for skill in ['development', 'engineering', 'design', 'analysis', 'management']):
                score *= 1.3
           
            # Boost experience terms
            if any(exp in keyword for exp in ['years', 'experience', 'expertise', 'proficiency']):
                score *= 1.2
           
            keyword_scores[keyword] = score
       
        # Sort by score and return keywords
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, score in sorted_keywords]
   
    def detect_domain(self, keywords):
        """Detect domain based on keywords"""
        if not keywords:
            return "General"
       
        # Enhanced domain detection logic with more comprehensive keywords
        tech_keywords = {
    'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node', 'express',
    'spring', 'hibernate', 'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'nosql', 'firebase',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'ci/cd',
    'git', 'github', 'gitlab', 'bitbucket', 'api', 'rest', 'graphql', 'json', 'xml',
    'frontend', 'backend', 'fullstack', 'devops', 'sre', 'networking', 'tcp/ip', 'dns', 'http',
    'https', 'cloud', 'serverless', 'lambda', 'ec2', 's3', 'vpc', 'rds', 'cloudfront', 'elastic beanstalk',
    'machine learning', 'deep learning', 'nlp', 'computer vision', 'neural networks',
    'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy', 'data science',
    'data engineering', 'etl', 'big data', 'spark', 'hadoop', 'kafka', 'airflow', 'mlops',
    'cybersecurity', 'penetration testing', 'vulnerability assessment', 'firewall', 'ids', 'ips',
    'ethical hacking', 'encryption', 'security analyst', 'information security', 'siem',
    'system administrator', 'linux', 'bash', 'powershell', 'windows server', 'active directory',
    'virtualization', 'vmware', 'hyper-v', 'containers', 'microservices', 'service mesh',
    'api gateway', 'load balancer', 'reverse proxy', 'ssl', 'tls', 'certificates',
    'observability', 'monitoring', 'logging', 'prometheus', 'grafana', 'elasticsearch', 'splunk',
    'software engineer', 'developer', 'programmer', 'software architect', 'technical lead',
    'qa', 'quality assurance', 'test automation', 'selenium', 'cypress', 'junit', 'testng',
    'scrum', 'agile', 'kanban', 'jira', 'confluence', 'project management', 'product management',
    'ux', 'ui', 'design systems', 'responsive design', 'accessibility', 'wireframes',
    'figma', 'adobe xd', 'version control', 'containerization', 'release engineering',
    'refactoring', 'code review', 'clean code', 'oop', 'functional programming',
    'data structures', 'algorithms', 'system design', 'design patterns', 'low-level design',
    'high-level design', 'load testing', 'performance tuning', 'scalability', 'reliability',
    'availability', 'fault tolerance', 'disaster recovery', 'backup', 'integration testing',
    'unit testing', 'acceptance testing', 'regression testing', 'a/b testing',
    'command line', 'cli tools', 'dev environment', 'ide', 'intellij', 'vscode', 'eclipse',
    'visual studio', 'notepad++', 'debugging', 'profiling', 'deployment', 'release',
    'continuous integration', 'continuous deployment', 'infrastructure as code', 'cloud-native',
    'edge computing', 'blockchain', 'web3', 'solidity', 'smart contracts', 'crypto', 'bitcoin',
    'ethereum', 'iot', 'robotics', 'embedded systems', 'firmware', 'raspberry pi', 'arduino',
    'low-code', 'no-code', 'platform engineering', 'data lake', 'data warehouse',
    'business intelligence', 'tableau', 'power bi', 'lookml', 'metabase', 'snowflake',
    'dbt', 'redshift', 'athena', 'clickhouse', 'databricks', 'elastic stack', 'logstash',
    'kibana', 'telemetry', 'incident management', 'on-call', 'sla', 'slo', 'slis'
}

        healthcare_keywords = {
    'nursing', 'rn', 'lpn', 'nurse practitioner', 'registered nurse', 'licensed practical nurse',
    'medical', 'clinical', 'hospital', 'patient', 'care', 'healthcare', 'doctor', 'physician',
    'surgeon', 'therapist', 'occupational therapist', 'physical therapist', 'speech therapist',
    'pharmacist', 'pharmacy', 'technician', 'lab tech', 'x-ray tech', 'radiologic technologist',
    'ultrasound technician', 'sonographer', 'radiology', 'mri', 'ct scan', 'imaging', 'diagnostics',
    'treatment', 'medication', 'medication administration', 'dosage', 'prescription', 'drug',
    'charting', 'electronic health record', 'emr', 'ehr', 'epic', 'cerner', 'meditech',
    'vitals', 'blood pressure', 'pulse', 'respiration', 'temperature', 'oxygen saturation',
    'icu', 'ccu', 'ward', 'inpatient', 'outpatient', 'emergency', 'er', 'triage',
    'ambulance', 'paramedic', 'emt', 'first responder', 'bcls', 'acls', 'cpr',
    'infection control', 'aseptic technique', 'sterilization', 'hand hygiene',
    'wound care', 'dressing change', 'surgical wound', 'incision', 'suturing',
    'patient care', 'patient safety', 'discharge planning', 'care coordination',
    'home healthcare', 'visiting nurse', 'hospice', 'palliative care',
    'nursing home', 'long-term care', 'assisted living', 'geriatrics',
    'pediatrics', 'neonatal', 'nicu', 'labor and delivery', 'ob/gyn', 'midwife',
    'mental health', 'psychiatry', 'psychology', 'behavioral health',
    'substance abuse', 'addiction treatment', 'detox', 'rehab', 'counseling',
    'social work', 'case management', 'insurance', 'medicare', 'medicaid',
    'claims', 'billing', 'medical coding', 'icd-10', 'cpt', 'hipaa',
    'compliance', 'regulatory', 'quality assurance', 'joint commission',
    'clinical documentation', 'progress notes', 'care plan', 'assessment',
    'evaluation', 'diagnosis', 'disease management', 'chronic care',
    'diabetes management', 'hypertension', 'cardiology', 'oncology',
    'nephrology', 'pulmonology', 'neurology', 'gastroenterology',
    'hepatology', 'endocrinology', 'orthopedics', 'pain management',
    'anesthesiology', 'surgery', 'pre-op', 'post-op', 'perioperative',
    'scrub nurse', 'circulating nurse', 'anesthetist', 'surgical tech',
    'occupational health', 'industrial medicine', 'telemedicine', 'telehealth',
    'virtual care', 'remote monitoring', 'health informatics', 'biomedical',
    'clinical trials', 'research nurse', 'principal investigator',
    'institutional review board', 'data collection', 'public health',
    'epidemiology', 'vaccination', 'immunization', 'screening', 'prevention',
    'wellness', 'health education', 'nutrition', 'dietician', 'fitness',
    'rehabilitation', 'physical medicine', 'mobility', 'assistive devices',
    'wheelchair', 'prosthetics', 'orthotics', 'speech pathology',
    'medical assistant', 'certified nursing assistant', 'cna',
    'medical receptionist', 'healthcare administrator', 'medical records',
    'healthcare provider', 'healthcare professional', 'continuing education',
    'board certification', 'state license', 'clinical rotation',
    'nursing school', 'preceptorship', 'internship', 'residency',
    'fellowship', 'grand rounds', 'rounding', 'consultation', 'handoff',
    'multidisciplinary team', 'interprofessional', 'patient satisfaction',
    'patient rights', 'informed consent', 'advanced directive', 'dnr',
    'code blue', 'rapid response', 'falls risk', 'pressure ulcer', 'restraints','cna','word','wound' 'care','charting'
}

        # Oil & Gas keywords (focused list)
        oil_gas_keywords = {
            'oil', 'gas', 'petroleum', 'crude', 'refinery', 'refining', 'drilling', 'rig', 'offshore',
            'onshore', 'upstream', 'downstream', 'midstream', 'exploration', 'production', 'reservoir',
            'well', 'wellhead', 'pipeline', 'petrochemical', 'lng', 'natural gas', 'shale', 'fracking',
            'oilfield', 'oil field', 'oil platform', 'platform', 'drilling engineer', 'petroleum engineer',
            'reservoir engineer', 'production engineer', 'process engineer', 'pipeline engineer',
            'oil and gas', 'oil & gas', 'energy sector', 'energy industry', 'hydrocarbon',
            'distillation', 'cracking', 'fractionation', 'compressor', 'turbine', 'flare',
            'well completion', 'well intervention', 'workover', 'wireline', 'mud logging',
            'blowout preventer', 'bop', 'derrick', 'drill string', 'drill bit', 'casing',
            'gas processing', 'gas treatment', 'sweetening', 'dehydration', 'sulfur recovery',
            'cogeneration', 'cogen', 'gas turbine', 'steam turbine', 'boiler', 'heat exchanger',
            'reactor', 'furnace', 'catalyst', 'reformer', 'isomerization', 'alkylation',
            'corrosion', 'inspection', 'ndt', 'api', 'asme', 'ansi', 'astm', 'iso',
            'hse', 'health safety environment', 'process safety', 'psm', 'hazop', 'hazid',
            'safety', 'environmental compliance', 'regulatory compliance', 'epc', 'epcm',
            'project management', 'construction', 'fabrication', 'welding', 'commissioning',
            'turnaround', 'maintenance', 'reliability', 'rotating equipment', 'static equipment',
            'vessel', 'column', 'tower', 'pump', 'compressor', 'turbine', 'motor', 'dcs',
            'scada', 'plc', 'instrumentation', 'control valve', 'process control', 'apc',
            'hysys', 'aspen plus', 'proii', 'unisim', 'petrosim', 'olga', 'pipesim',
            'eclipse', 'cmg', 'petrel', 'pi system', 'wonderware', 'maximo', 'sap pm'
        }
        
        # Marketing keywords (including social media, digital marketing, content marketing)
        marketing_keywords = {
            'marketing', 'marketer', 'marketing manager', 'marketing director', 'marketing coordinator',
            'digital marketing', 'online marketing', 'internet marketing', 'e-marketing',
            'social media', 'social media manager', 'social media coordinator', 'social media specialist',
            'social media marketing', 'smm', 'social media strategy', 'social media content',
            'facebook', 'instagram', 'twitter', 'linkedin', 'tiktok', 'youtube', 'pinterest', 'snapchat',
            'content marketing', 'content creator', 'content strategist', 'content writer',
            'seo', 'search engine optimization', 'sem', 'search engine marketing', 'ppc', 'pay per click',
            'google ads', 'adwords', 'bing ads', 'display advertising', 'banner ads', 'display ads',
            'email marketing', 'email campaign', 'newsletter', 'mailchimp', 'constant contact',
            'brand management', 'brand manager', 'branding', 'brand strategy', 'brand development',
            'public relations', 'pr', 'publicist', 'media relations', 'press release', 'press kit',
            'advertising', 'advertising manager', 'advertising agency', 'creative director',
            'copywriter', 'copywriting', 'creative writing', 'ad copy', 'marketing copy',
            'market research', 'market analyst', 'consumer research', 'focus group', 'survey',
            'analytics', 'marketing analytics', 'google analytics', 'web analytics', 'data analytics',
            'crm', 'customer relationship management', 'salesforce', 'hubspot', 'marketing automation',
            'campaign management', 'marketing campaign', 'advertising campaign', 'promotional campaign',
            'event marketing', 'event management', 'trade show', 'exhibition', 'conference',
            'influencer marketing', 'influencer', 'affiliate marketing', 'partnership marketing',
            'product marketing', 'product manager', 'product launch', 'go-to-market',
            'demand generation', 'lead generation', 'lead nurturing', 'conversion optimization',
            'growth marketing', 'growth hacker', 'growth strategy', 'user acquisition',
            'retention marketing', 'customer retention', 'loyalty program', 'customer loyalty',
            'community management', 'community manager', 'online community', 'forum moderation',
            'graphic design', 'visual design', 'adobe creative suite', 'photoshop', 'illustrator',
            'video marketing', 'video production', 'video editing', 'youtube marketing',
            'podcast marketing', 'podcast production', 'audio content', 'streaming',
            'mobile marketing', 'app marketing', 'mobile advertising', 'sms marketing',
            'local marketing', 'local seo', 'google my business', 'local advertising',
            'b2b marketing', 'business to business', 'b2c marketing', 'business to consumer',
            'marketing communications', 'marcom', 'integrated marketing', 'omnichannel marketing',
            'marketing strategy', 'strategic marketing', 'marketing planning', 'marketing budget',
            'roi', 'return on investment', 'marketing roi', 'campaign roi', 'conversion rate',
            'ctr', 'click through rate', 'cpc', 'cost per click', 'cpm', 'cost per mille',
            'cpa', 'cost per acquisition', 'lifetime value', 'ltv', 'customer lifetime value',
            'kpi', 'key performance indicator', 'marketing metrics', 'performance marketing'
        }
        
        # Networking keywords (IT networking and business networking)
        networking_keywords = {
            'networking', 'network engineer', 'network administrator', 'network specialist',
            'network technician', 'network architect', 'network security', 'network infrastructure',
            'cisco', 'ccna', 'ccnp', 'ccie', 'juniper', 'arista', 'palo alto', 'fortinet',
            'tcp/ip', 'osi model', 'routing', 'switching', 'vlan', 'vpn', 'wan', 'lan', 'man',
            'firewall', 'load balancer', 'network monitoring', 'network troubleshooting',
            'dns', 'dhcp', 'subnetting', 'ip addressing', 'ipv4', 'ipv6', 'bgp', 'ospf', 'eigrp',
            'ethernet', 'wireless', 'wifi', '802.11', 'bluetooth', 'network protocols',
            'network design', 'network planning', 'network optimization', 'network performance',
            'sd-wan', 'software defined networking', 'sdn', 'nfv', 'network function virtualization',
            'cloud networking', 'aws networking', 'azure networking', 'gcp networking',
            'business networking', 'professional networking', 'networking events', 'networking skills',
            'relationship building', 'stakeholder management', 'partnership development'
        }
        
        # Sales keywords
        sales_keywords = {
            'sales', 'sales representative', 'sales rep', 'sales executive', 'sales manager',
            'sales director', 'account executive', 'account manager', 'account representative',
            'business development', 'bd', 'business development manager', 'bdr', 'sdr',
            'sales development', 'inside sales', 'outside sales', 'field sales', 'territory sales',
            'retail sales', 'wholesale sales', 'b2b sales', 'b2c sales', 'enterprise sales',
            'saas sales', 'software sales', 'solution sales', 'consultative selling',
            'relationship selling', 'relationship management', 'customer relationship',
            'lead generation', 'prospecting', 'cold calling', 'cold emailing', 'outreach',
            'qualification', 'lead qualification', 'opportunity qualification', 'bant',
            'pipeline management', 'sales pipeline', 'forecasting', 'sales forecast',
            'quota', 'sales quota', 'target', 'sales target', 'revenue target',
            'closing', 'deal closing', 'negotiation', 'contract negotiation', 'pricing',
            'presentation', 'sales presentation', 'demo', 'product demonstration',
            'objection handling', 'objection management', 'customer objections',
            'crm', 'salesforce', 'hubspot', 'pipedrive', 'zoho', 'sales crm',
            'sales process', 'sales methodology', 'sandler', 'spin selling', 'challenger sale',
            'customer success', 'account management', 'key account management', 'kam',
            'upselling', 'cross-selling', 'retention', 'customer retention', 'churn reduction'
        }
        
        # Finance & Accounting keywords
        finance_keywords = {
            'finance', 'financial', 'accounting', 'accountant', 'cpa', 'certified public accountant',
            'financial analyst', 'financial planning', 'financial reporting', 'financial modeling',
            'bookkeeping', 'accounts payable', 'accounts receivable', 'ap', 'ar', 'general ledger',
            'trial balance', 'balance sheet', 'income statement', 'cash flow', 'p&l', 'profit and loss',
            'audit', 'auditing', 'internal audit', 'external audit', 'auditor', 'sox', 'sarbanes oxley',
            'tax', 'taxation', 'tax preparation', 'tax planning', 'tax compliance', 'irs',
            'payroll', 'payroll processing', 'payroll management', 'payroll administration',
            'budgeting', 'budget', 'budget planning', 'budget analysis', 'forecasting', 'financial forecast',
            'cost accounting', 'cost analysis', 'cost control', 'variance analysis',
            'treasury', 'treasury management', 'cash management', 'liquidity management',
            'investment', 'investment analysis', 'portfolio management', 'asset management',
            'risk management', 'financial risk', 'credit risk', 'market risk', 'operational risk',
            'compliance', 'regulatory compliance', 'financial compliance', 'gaap', 'ifrs',
            'erp', 'sap', 'oracle financials', 'quickbooks', 'xero', 'sage', 'financial software',
            'controller', 'financial controller', 'cfo', 'chief financial officer', 'finance director'
        }
        
        # HR & Human Resources keywords
        hr_keywords = {
            'human resources', 'hr', 'hr manager', 'hr director', 'hr specialist', 'hr coordinator',
            'recruiting', 'recruitment', 'recruiter', 'talent acquisition', 'talent management',
            'hiring', 'onboarding', 'employee onboarding', 'orientation', 'employee orientation',
            'training', 'employee training', 'learning and development', 'l&d', 'training and development',
            'performance management', 'performance review', 'performance appraisal', 'employee evaluation',
            'compensation', 'compensation and benefits', 'c&b', 'payroll', 'salary administration',
            'benefits', 'employee benefits', 'health benefits', 'retirement benefits', '401k',
            'employee relations', 'labor relations', 'union relations', 'collective bargaining',
            'policy', 'hr policy', 'employee handbook', 'policy development', 'compliance',
            'employment law', 'labor law', 'workplace compliance', 'eeoc', 'ada', 'fmla',
            'diversity', 'inclusion', 'diversity and inclusion', 'd&i', 'equity', 'dei',
            'organizational development', 'od', 'change management', 'organizational change',
            'talent development', 'succession planning', 'career development', 'employee development',
            'engagement', 'employee engagement', 'satisfaction', 'employee satisfaction', 'retention',
            'hris', 'human resources information system', 'workday', 'bamboo', 'adp', 'paycom',
            'hris', 'ats', 'applicant tracking system', 'taleo', 'greenhouse', 'lever'
        }
        
        # Operations keywords
        operations_keywords = {
            'operations', 'operations manager', 'operations director', 'operations coordinator',
            'supply chain', 'supply chain management', 'scm', 'logistics', 'logistics coordinator',
            'procurement', 'purchasing', 'buyer', 'sourcing', 'vendor management', 'supplier management',
            'inventory', 'inventory management', 'inventory control', 'warehouse', 'warehousing',
            'distribution', 'distribution center', 'fulfillment', 'order fulfillment',
            'quality', 'quality control', 'qc', 'quality assurance', 'qa', 'quality management',
            'process improvement', 'process optimization', 'lean', 'six sigma', 'kaizen',
            'project management', 'pmp', 'project manager', 'program manager', 'portfolio manager',
            'facilities', 'facilities management', 'facility manager', 'maintenance', 'facility maintenance',
            'production', 'production manager', 'manufacturing', 'manufacturing manager',
            'scheduling', 'production scheduling', 'workforce planning', 'capacity planning',
            'kpi', 'key performance indicator', 'metrics', 'operational metrics', 'dashboard',
            'erp', 'enterprise resource planning', 'sap', 'oracle', 'erp system'
        }
        
        # Convert keywords to lowercase for comparison and create a combined text
        keywords_lower = [kw.lower() for kw in keywords]
        combined_text = ' '.join(keywords_lower)
       
        # Count matches with improved logic (exact matches and phrase matches)
        tech_count = 0
        healthcare_count = 0
        oil_gas_count = 0
        marketing_count = 0
        networking_count = 0
        sales_count = 0
        finance_count = 0
        hr_count = 0
        operations_count = 0
       
        # Function to check if keyword appears in text (word boundary aware)
        def keyword_matches(text, keyword):
            """Check if keyword matches in text with word boundary awareness."""
            # Exact word match
            if keyword == text:
                return True
            # Word boundary match
            import re
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                return True
            # Substring match (for multi-word keywords)
            if keyword in text or text in keyword:
                # Only count if keyword is substantial (at least 3 chars)
                if len(keyword) >= 3:
                    return True
            return False
       
        for keyword in keywords_lower:
            # Check for tech keywords
            for tech_kw in tech_keywords:
                if keyword_matches(keyword, tech_kw) or keyword_matches(combined_text, tech_kw):
                    tech_count += 1
                    break
           
            # Check for healthcare keywords
            for health_kw in healthcare_keywords:
                if keyword_matches(keyword, health_kw) or keyword_matches(combined_text, health_kw):
                    healthcare_count += 1
                    break
            
            # Check for oil & gas keywords
            for og_kw in oil_gas_keywords:
                if keyword_matches(keyword, og_kw) or keyword_matches(combined_text, og_kw):
                    oil_gas_count += 1
                    break
            
            # Check for marketing keywords
            for mk_kw in marketing_keywords:
                if keyword_matches(keyword, mk_kw) or keyword_matches(combined_text, mk_kw):
                    marketing_count += 1
                    break
            
            # Check for networking keywords
            for net_kw in networking_keywords:
                if keyword_matches(keyword, net_kw) or keyword_matches(combined_text, net_kw):
                    networking_count += 1
                    break
            
            # Check for sales keywords
            for sales_kw in sales_keywords:
                if keyword_matches(keyword, sales_kw) or keyword_matches(combined_text, sales_kw):
                    sales_count += 1
                    break
            
            # Check for finance keywords
            for fin_kw in finance_keywords:
                if keyword_matches(keyword, fin_kw) or keyword_matches(combined_text, fin_kw):
                    finance_count += 1
                    break
            
            # Check for HR keywords
            for hr_kw in hr_keywords:
                if keyword_matches(keyword, hr_kw) or keyword_matches(combined_text, hr_kw):
                    hr_count += 1
                    break
            
            # Check for operations keywords
            for ops_kw in operations_keywords:
                if keyword_matches(keyword, ops_kw) or keyword_matches(combined_text, ops_kw):
                    operations_count += 1
                    break
        
        # Also check for multi-word phrases in the combined text
        # This helps catch phrases like "social media manager", "oil and gas engineer"
        for tech_kw in tech_keywords:
            if len(tech_kw.split()) > 1 and tech_kw in combined_text:
                tech_count += 0.5  # Half weight for phrase matches
        
        for health_kw in healthcare_keywords:
            if len(health_kw.split()) > 1 and health_kw in combined_text:
                healthcare_count += 0.5
        
        for og_kw in oil_gas_keywords:
            if len(og_kw.split()) > 1 and og_kw in combined_text:
                oil_gas_count += 0.5
        
        for mk_kw in marketing_keywords:
            if len(mk_kw.split()) > 1 and mk_kw in combined_text:
                marketing_count += 0.5
        
        for net_kw in networking_keywords:
            if len(net_kw.split()) > 1 and net_kw in combined_text:
                networking_count += 0.5
        
        for sales_kw in sales_keywords:
            if len(sales_kw.split()) > 1 and sales_kw in combined_text:
                sales_count += 0.5
        
        for fin_kw in finance_keywords:
            if len(fin_kw.split()) > 1 and fin_kw in combined_text:
                finance_count += 0.5
        
        for hr_kw in hr_keywords:
            if len(hr_kw.split()) > 1 and hr_kw in combined_text:
                hr_count += 0.5
        
        for ops_kw in operations_keywords:
            if len(ops_kw.split()) > 1 and ops_kw in combined_text:
                operations_count += 0.5
       
        # Log the detection results for debugging
        logger.info(f"Domain detection - Tech: {tech_count}, Healthcare: {healthcare_count}, Oil & Gas: {oil_gas_count}, "
                   f"Marketing: {marketing_count}, Networking: {networking_count}, Sales: {sales_count}, "
                   f"Finance: {finance_count}, HR: {hr_count}, Operations: {operations_count}")
       
        # Determine domain with improved logic (check all domains and return the one with highest count)
        domain_counts = {
            'Oil & Gas': oil_gas_count,
            'Marketing': marketing_count,
            'Networking': networking_count,
            'Sales': sales_count,
            'Finance': finance_count,
            'HR': hr_count,
            'Operations': operations_count,
            'Healthcare': healthcare_count,
            'IT/Tech': tech_count
        }
        
        # Find the domain with the highest count
        max_count = max(domain_counts.values())
        if max_count > 0:
            # Get all domains with the max count (in case of ties)
            top_domains = [domain for domain, count in domain_counts.items() if count == max_count]
            
            # Priority order for tie-breaking
            priority_order = ['Oil & Gas', 'Marketing', 'Networking', 'Sales', 'Finance', 'HR', 
                            'Operations', 'Healthcare', 'IT/Tech']
            
            # Return the highest priority domain among the top domains
            for domain in priority_order:
                if domain in top_domains:
                    logger.info(f"Domain classified as {domain}")
                    return domain
        
            logger.info("Domain classified as General")
            return "General"
   
    def semantic_similarity(self, text1, text2):
        """Enhanced semantic similarity calculation with multiple algorithms for maximum accuracy"""
        if not text1 or not text2:
            return 0.0
       
        try:
            # Convert to lowercase for processing
            text1_lower = text1.lower()
            text2_lower = text2.lower()
           
            # Calculate multiple similarity metrics
            jaccard_sim = self._calculate_jaccard_similarity(text1_lower, text2_lower)
            cosine_sim = self._calculate_cosine_similarity(text1_lower, text2_lower)
            keyword_sim = self._calculate_keyword_similarity(text1_lower, text2_lower)
            phrase_sim = self._calculate_phrase_similarity(text1_lower, text2_lower)
           
            # Calculate n-gram similarity for better phrase matching
            bigram_sim = self._calculate_ngram_similarity(text1_lower, text2_lower, n=2)
            trigram_sim = self._calculate_ngram_similarity(text1_lower, text2_lower, n=3)
           
            # Calculate weighted term frequency similarity
            tf_sim = self._calculate_tf_similarity(text1_lower, text2_lower)
           
            # Enhanced weighted combination (optimized for accuracy)
            weights = {
                'jaccard': 0.20,
                'cosine': 0.20,
                'keyword': 0.20,
                'phrase': 0.15,
                'bigram': 0.10,
                'trigram': 0.08,
                'tf': 0.07
            }
           
            final_similarity = (
                jaccard_sim * weights['jaccard'] +
                cosine_sim * weights['cosine'] +
                keyword_sim * weights['keyword'] +
                phrase_sim * weights['phrase'] +
                bigram_sim * weights['bigram'] +
                trigram_sim * weights['trigram'] +
                tf_sim * weights['tf']
            )
           
            # Apply boost for very high individual scores
            if max(jaccard_sim, cosine_sim, keyword_sim) >= 0.9:
                final_similarity = min(final_similarity * 1.05, 1.0)  # 5% boost
           
            return min(final_similarity, 1.0)  # Cap at 1.0
           
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
   
    def _calculate_ngram_similarity(self, text1, text2, n=2):
        """Calculate n-gram similarity"""
        if not text1 or not text2:
            return 0.0
       
        def get_ngrams(text, n):
            words = text.split()
            ngrams = []
            for i in range(len(words) - n + 1):
                ngrams.append(' '.join(words[i:i+n]))
            return set(ngrams)
       
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
       
        if not ngrams1 or not ngrams2:
            return 0.0
       
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
       
        return len(intersection) / len(union) if union else 0.0
   
    def _calculate_tf_similarity(self, text1, text2):
        """Calculate term frequency weighted similarity"""
        from collections import Counter
       
        if not text1 or not text2:
            return 0.0
       
        words1 = text1.split()
        words2 = text2.split()
       
        if not words1 or not words2:
            return 0.0
       
        # Calculate term frequencies
        tf1 = Counter(words1)
        tf2 = Counter(words2)
       
        # Normalize by document length
        len1 = len(words1)
        len2 = len(words2)
       
        # Calculate weighted similarity
        common_words = set(words1) & set(words2)
        if not common_words:
            return 0.0
       
        similarity = 0.0
        for word in common_words:
            # Use minimum TF as similarity contribution
            tf_score = min(tf1[word] / len1, tf2[word] / len2)
            similarity += tf_score
       
        # Normalize by average document length
        avg_len = (len1 + len2) / 2
        return min(similarity / avg_len if avg_len > 0 else 0.0, 1.0)
   
    def _calculate_jaccard_similarity(self, text1, text2):
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
   
    def _calculate_cosine_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts"""
        from collections import Counter
        import math
       
        # Create word frequency vectors
        words1 = Counter(text1.split())
        words2 = Counter(text2.split())
       
        # Get all unique words
        all_words = set(words1.keys()) | set(words2.keys())
       
        # Create vectors
        vec1 = [words1.get(word, 0) for word in all_words]
        vec2 = [words2.get(word, 0) for word in all_words]
       
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
       
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
   
    def _calculate_keyword_similarity(self, text1, text2):
        """Calculate similarity based on important keywords"""
        # Extract keywords from both texts
        keywords1 = self._extract_enhanced_keywords(text1)
        keywords2 = self._extract_enhanced_keywords(text2)
       
        if not keywords1 or not keywords2:
            return 0.0
       
        # Calculate overlap of top keywords
        top_keywords1 = set(keywords1[:20])  # Top 20 keywords
        top_keywords2 = set(keywords2[:20])
       
        intersection = top_keywords1.intersection(top_keywords2)
        union = top_keywords1.union(top_keywords2)
        return len(intersection) / len(union) if union else 0.0
   
    def _calculate_phrase_similarity(self, text1, text2):
        """Calculate similarity based on technical phrases"""
        # Extract technical phrases from both texts
        phrases1 = self._extract_technical_phrases(text1)
        phrases2 = self._extract_technical_phrases(text2)
       
        if not phrases1 or not phrases2:
            return 0.0
       
        # Calculate phrase overlap
        phrases1_set = set(phrases1)
        phrases2_set = set(phrases2)
       
        intersection = phrases1_set.intersection(phrases2_set)
        union = phrases1_set.union(phrases2_set)
        return len(intersection) / len(union) if union else 0.0
   
    def _identify_critical_skills(self, job_description, job_skills):
        """Identify critical/required skills from job description"""
        if not job_description or not job_skills:
            return set()
       
        job_desc_lower = job_description.lower()
        critical_skills = set()
       
        # Look for phrases that indicate required/critical skills
        critical_patterns = [
            r'(?:required|must have|mandatory|essential|critical|necessary)[\s:]+([^\.\n]+)',
            r'(?:strong|expert|proficient|extensive)[\s]+(?:experience|knowledge|skills?)[\s]+(?:in|with|of)[\s:]+([^\.\n]+)',
            r'(?:minimum|at least)[\s]+([^\.\n]+?)(?:years?|experience)',
        ]
       
        for pattern in critical_patterns:
            matches = re.findall(pattern, job_desc_lower, re.IGNORECASE)
            for match in matches:
                # Extract skills from the match
                skills = re.split(r'[,;|•\n•\-\–\—]', match)
                for skill in skills:
                    skill = skill.strip().lower()
                    if skill and len(skill) > 2:
                        # Check if this skill is in our job_skills list
                        for js in job_skills:
                            if skill in js.lower() or js.lower() in skill:
                                critical_skills.add(js.lower())
       
        # Also mark first 3-5 skills as critical (they're usually most important)
        if job_skills:
            for skill in job_skills[:5]:
                critical_skills.add(skill.lower())
       
        return critical_skills
   
    def _fuzzy_match_skills(self, candidate_skills, job_skills, threshold=0.70, critical_skills=None):
        """Fuzzy matching for skills with enhanced similarity threshold and weighted scoring for maximum accuracy"""
        if not candidate_skills or not job_skills:
            return 0.0
       
        if not job_skills:
            return 0.0
       
        if critical_skills is None:
            critical_skills = set()
       
        matches = 0
        very_strong_matches = 0
        strong_matches = 0
        critical_matches = 0
        total_match_score = 0.0
        total_job_skills = len(job_skills)
       
        # Normalize all skills to lowercase for better matching
        candidate_skills_normalized = [s.lower().strip() for s in candidate_skills if s]
        job_skills_normalized = [s.lower().strip() for s in job_skills if s]
        critical_skills_normalized = {s.lower().strip() for s in critical_skills}
       
        # Weight skills by importance (earlier in list = more important, exponential decay)
        for idx, job_skill in enumerate(job_skills_normalized):
            best_match = 0.0
            best_candidate_skill = None
            is_critical = job_skill in critical_skills_normalized
           
            # Try exact match first (fastest)
            if job_skill in candidate_skills_normalized:
                best_match = 1.0
                best_candidate_skill = job_skill
            else:
                # Try fuzzy matching
                for candidate_skill in candidate_skills_normalized:
                    similarity = self._calculate_skill_similarity(job_skill, candidate_skill)
                    if similarity > best_match:
                        best_match = similarity
                        best_candidate_skill = candidate_skill
           
            # Enhanced weight calculation
            # Critical skills get 1.5x weight
            # Top 5 skills get full weight, then exponential decay
            if is_critical:
                base_weight = 1.5  # Critical skills are 50% more important
            elif idx < 5:
                base_weight = 1.0
            else:
                base_weight = 0.92 ** (idx - 4)  # Slower decay for better coverage
           
            weight = base_weight
            weighted_score = best_match * weight
            total_match_score += weighted_score
           
            # Count matches with different thresholds
            if best_match >= 0.95:
                very_strong_matches += 1
                strong_matches += 1
                matches += 1
                if is_critical:
                    critical_matches += 1
            elif best_match >= 0.85:
                strong_matches += 1
                matches += 1
                if is_critical:
                    critical_matches += 1
            elif best_match >= threshold:
                matches += 1
                if is_critical:
                    critical_matches += 1
       
        # Calculate match ratio and average similarity
        match_ratio = matches / total_job_skills if total_job_skills > 0 else 0.0
        avg_similarity = total_match_score / total_job_skills if total_job_skills > 0 else 0.0
       
        # Calculate weighted average similarity (accounting for skill importance and critical skills)
        total_weight = 0.0
        for i in range(total_job_skills):
            job_skill = job_skills_normalized[i] if i < len(job_skills_normalized) else ''
            is_critical = job_skill in critical_skills_normalized
            if is_critical:
                base_weight = 1.5  # Critical skills get 1.5x weight
            elif i < 5:
                base_weight = 1.0
            else:
                base_weight = 0.92 ** (i - 4)
            total_weight += base_weight
       
        weighted_avg = total_match_score / total_weight if total_weight > 0 and total_job_skills > 0 else 0.0
       
        # Enhanced scoring with multiple factors
        very_strong_ratio = very_strong_matches / total_job_skills if total_job_skills > 0 else 0.0
        strong_ratio = strong_matches / total_job_skills if total_job_skills > 0 else 0.0
       
        # Calculate critical skill match ratio
        total_critical = len(critical_skills_normalized) if critical_skills_normalized else 0
        critical_match_ratio = critical_matches / total_critical if total_critical > 0 else 0.0
       
        final_score = (
            match_ratio * 0.45 +           # Match ratio (primary)
            weighted_avg * 0.35 +          # Weighted average similarity (increased)
            avg_similarity * 0.10 +        # Overall average
            very_strong_ratio * 0.05 +     # Bonus for very strong matches
            critical_match_ratio * 0.05    # Bonus for critical skill matches
        )
       
        # Enhanced boost system for high-quality matches
        # Critical skills boost
        if total_critical > 0 and critical_match_ratio >= 0.8:  # 80%+ critical skills matched
            final_score *= 1.15  # 15% boost for matching critical skills
        elif total_critical > 0 and critical_match_ratio >= 0.6:  # 60%+ critical skills matched
            final_score *= 1.10  # 10% boost
       
        if very_strong_matches >= total_job_skills * 0.6:  # 60%+ very strong matches
            final_score *= 1.12  # 12% boost
        elif very_strong_matches >= total_job_skills * 0.5:  # 50%+ very strong matches
            final_score *= 1.08  # 8% boost
        elif strong_matches >= total_job_skills * 0.7:  # 70%+ strong matches
            final_score *= 1.06  # 6% boost
        elif match_ratio >= 0.9:  # 90%+ matches
            final_score *= 1.05  # 5% boost
        elif match_ratio >= 0.8:  # 80%+ matches
            final_score *= 1.03  # 3% boost
       
        # Enhanced penalty system for poor matches
        # Penalty for missing critical skills
        if total_critical > 0 and critical_match_ratio < 0.3:  # Less than 30% critical skills matched
            final_score *= 0.70  # 30% penalty for missing critical skills
        elif total_critical > 0 and critical_match_ratio < 0.5:  # Less than 50% critical skills matched
            final_score *= 0.85  # 15% penalty
       
        if match_ratio < 0.2:
            final_score *= 0.75  # 25% penalty for very poor matches
        elif match_ratio < 0.3:
            final_score *= 0.85  # 15% penalty for poor matches
        elif match_ratio < 0.5:
            final_score *= 0.92  # 8% penalty for below average matches
       
        return min(final_score, 1.0)
   
    def _calculate_skill_similarity(self, skill1, skill2):
        """Calculate similarity between two skills using multiple methods"""
        if not skill1 or not skill2:
            return 0.0
       
        skill1_lower = skill1.lower().strip()
        skill2_lower = skill2.lower().strip()
       
        # Exact match
        if skill1_lower == skill2_lower:
            return 1.0
       
        # Check for common variations (significantly expanded for better matching)
        variations = {
            # Programming languages (expanded)
            'javascript': ['js', 'ecmascript', 'node.js', 'nodejs', 'node', 'es6', 'es2015', 'es7', 'es8', 'es9', 'es10', 'esnext', 'jsx', 'tsx'],
            'python': ['py', 'python3', 'python 3', 'python2', 'python 2', 'django', 'flask', 'fastapi', 'pylons', 'pyramid', 'tornado'],
            'java': ['java ee', 'j2ee', 'j2se', 'j2me', 'spring', 'spring boot', 'spring mvc', 'spring framework', 'jsp', 'servlet', 'jpa', 'hibernate'],
            'typescript': ['ts', 'typescript 3', 'tsx', 'ts-node', 'typescript 4', 'typescript 5'],
            'c++': ['cpp', 'c plus plus', 'cxx', 'cplusplus', 'c with classes'],
            'c#': ['csharp', 'dotnet', '.net', 'dot net', 'asp.net', 'asp net', 'entity framework', 'ef core'],
            'go': ['golang', 'go lang', 'go programming'],
            'rust': ['rustlang', 'rust programming'],
            'php': ['php5', 'php7', 'php8', 'laravel', 'symfony', 'codeigniter', 'zend'],
            'ruby': ['ruby on rails', 'rails', 'ror', 'sinatra'],
            'swift': ['swiftui', 'swift 5', 'swift 4'],
            'kotlin': ['kotlin android', 'kotlin multiplatform'],
            'scala': ['scala 2', 'scala 3', 'akka', 'play framework'],
            # Frameworks (expanded)
            'react': ['reactjs', 'react.js', 'react native', 'reactjs', 'react hooks', 'redux', 'mobx', 'next.js', 'nextjs'],
            'angular': ['angularjs', 'angular.js', 'angular 2+', 'angular 2', 'angular 4', 'angular 5', 'angular 6+', 'ng', 'angular cli'],
            'vue': ['vuejs', 'vue.js', 'vue 2', 'vue 3', 'nuxt', 'nuxt.js', 'nuxtjs', 'vuex'],
            'node.js': ['nodejs', 'node', 'express', 'nest', 'nestjs', 'koa', 'hapi', 'fastify'],
            'django': ['django rest framework', 'drf', 'django orm'],
            'flask': ['flask restful', 'flask sqlalchemy'],
            # Cloud & DevOps (expanded)
            'aws': ['amazon web services', 'ec2', 's3', 'lambda', 'cloudformation', 'rds', 'dynamodb', 'sns', 'sqs', 'ses', 'cloudfront', 'route53', 'vpc', 'iam', 'cloudwatch', 'ecs', 'eks', 'fargate'],
            'azure': ['microsoft azure', 'azure cloud', 'azure functions', 'azure devops', 'azure sql', 'azure storage'],
            'gcp': ['google cloud', 'google cloud platform', 'gce', 'gcs', 'gke', 'cloud functions', 'cloud run'],
            'docker': ['containerization', 'containers', 'docker compose', 'dockerfile', 'docker swarm'],
            'kubernetes': ['k8s', 'container orchestration', 'kube', 'kubectl', 'helm', 'istio'],
            'terraform': ['tf', 'terraform cloud', 'terraform enterprise'],
            'ansible': ['ansible tower', 'ansible awx'],
            'jenkins': ['ci/cd', 'continuous integration', 'continuous deployment', 'jenkins pipeline', 'jenkinsfile'],
            # Databases (expanded)
            'sql': ['mysql', 'postgresql', 'postgres', 'database', 'sql server', 'oracle', 'sqlite', 'mariadb', 'mssql', 'tsql', 'plsql'],
            'mongodb': ['mongo', 'nosql', 'mongodb atlas', 'mongoose'],
            'postgresql': ['postgres', 'pg', 'postgresql 12', 'postgresql 13', 'postgresql 14'],
            'redis': ['redis cache', 'redis cluster'],
            'elasticsearch': ['elastic', 'elk stack', 'elastic stack', 'kibana', 'logstash'],
            # ML/AI (expanded)
            'machine learning': ['ml', 'artificial intelligence', 'ai', 'deep learning', 'neural networks', 'neural net', 'nn', 'dl'],
            'tensorflow': ['tf', 'tensor flow', 'tensorflow 2', 'tensorflow lite'],
            'pytorch': ['pytorch', 'torch', 'torchvision'],
            'pandas': ['pd', 'pandas dataframe'],
            'numpy': ['np', 'numerical python'],
            'scikit-learn': ['sklearn', 'scikit learn', 'sk learn'],
            # Tools (expanded)
            'git': ['version control', 'github', 'gitlab', 'bitbucket', 'svn', 'subversion', 'mercurial', 'hg'],
            'github': ['github actions', 'github ci', 'github workflows'],
            'gitlab': ['gitlab ci', 'gitlab runner'],
            'jira': ['atlassian jira', 'jira software'],
            'confluence': ['atlassian confluence'],
            'postman': ['postman api', 'newman'],
            'rest api': ['rest', 'restful', 'restful api', 'rest api design'],
            'graphql': ['gql', 'graph ql', 'apollo', 'relay'],
            # Healthcare specific (expanded)
            'epic': ['epic systems', 'epic emr', 'epic ehr', 'epic mychart', 'epic hyperspace'],
            'cerner': ['cerner powerchart', 'cerner ehr', 'cerner millennium', 'cerner hedis'],
            'hl7': ['health level 7', 'hl7 fhir', 'fhir', 'hl7 v2', 'hl7 v3'],
            'hipaa': ['hipaa compliance', 'health information privacy', 'hipaa regulations'],
            'emr': ['electronic medical records', 'ehr', 'electronic health records', 'electronic patient records'],
            'medical coding': ['icd-10', 'cpt codes', 'medical billing', 'icd10', 'cpt', 'hcpcs', 'drg'],
            'nursing': ['rn', 'registered nurse', 'bsn', 'msn', 'dnp', 'lpn', 'lvn', 'cna', 'certified nursing assistant'],
            'pharmacy': ['pharmd', 'pharmacist', 'pharmaceutical', 'pharmacy tech', 'pharmacy technician'],
            'patient care': ['patient safety', 'patient advocacy', 'patient education', 'patient assessment'],
            'clinical documentation': ['charting', 'progress notes', 'soap notes', 'clinical notes', 'medical documentation'],
        }
       
        # Check if skills are variations of each other
        for main_skill, variants in variations.items():
            if (skill1_lower == main_skill and skill2_lower in variants) or \
               (skill2_lower == main_skill and skill1_lower in variants):
                return 0.9
           
            if skill1_lower in variants and skill2_lower in variants:
                return 0.8
       
        # Calculate string similarity using Levenshtein distance
        return self._calculate_string_similarity(skill1_lower, skill2_lower)
   
    def _calculate_string_similarity(self, str1, str2):
        """Calculate string similarity using enhanced Levenshtein distance with better handling"""
        if len(str1) < len(str2):
            str1, str2 = str2, str1
       
        if len(str2) == 0:
            return 0.0
       
        # Check for substring match (one skill contains the other)
        if str2 in str1:
            # Calculate similarity based on how much of the longer string is covered
            return len(str2) / len(str1) if len(str1) > 0 else 0.0
        elif str1 in str2:
            return len(str1) / len(str2) if len(str2) > 0 else 0.0
       
        # Check for word-level similarity (for multi-word skills)
        words1 = set(str1.split())
        words2 = set(str2.split())
        if words1 and words2:
            word_intersection = words1.intersection(words2)
            word_union = words1.union(words2)
            word_similarity = len(word_intersection) / len(word_union) if word_union else 0.0
            if word_similarity > 0.5:  # If significant word overlap, boost the score
                # Use word similarity as base, then refine with Levenshtein
                base_score = word_similarity * 0.7
            else:
                base_score = 0.0
        else:
            base_score = 0.0
       
        # Calculate Levenshtein distance
        matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
       
        for i in range(len(str1) + 1):
            matrix[i][0] = i
        for j in range(len(str2) + 1):
            matrix[0][j] = j
       
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i-1] == str2[j-1]:
                    cost = 0
                else:
                    cost = 1
               
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )
       
        distance = matrix[len(str1)][len(str2)]
        max_len = max(len(str1), len(str2))
        levenshtein_similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0
       
        # Combine word similarity and Levenshtein similarity
        if base_score > 0:
            final_similarity = max(base_score, levenshtein_similarity * 0.3 + base_score * 0.7)
        else:
            final_similarity = levenshtein_similarity
       
        return min(final_similarity, 1.0)
   
    def _extract_skills_from_description(self, job_description):
        """Extract skills from job description with enhanced accuracy and comprehensive coverage"""
        if not job_description:
            return []
       
        job_desc_lower = job_description.lower()
        extracted_skills = []
       
        # Comprehensive technical skills patterns (expanded significantly)
        skill_patterns = [
            # Programming languages (expanded)
            r'\b(python|java|javascript|typescript|go|rust|c\+\+|c#|ruby|php|swift|kotlin|scala|r|matlab|perl|bash|powershell|dart|elixir|erlang|haskell|clojure|f#|vb\.net|objective-c|delphi)\b',
            # Frontend frameworks (expanded)
            r'\b(react|angular|vue|django|flask|spring|express|laravel|rails|asp\.net|node\.js|next\.js|nuxt|svelte|ember|backbone|meteor|gatsby|remix|sveltekit)\b',
            # Backend frameworks
            r'\b(fastapi|fastify|koa|hapi|nest|adonis|phoenix|gin|echo|fiber|chi|gorilla|mux)\b',
            # Databases (expanded)
            r'\b(mysql|postgresql|mongodb|cassandra|redis|oracle|sql server|dynamodb|elasticsearch|mariadb|sqlite|neo4j|influxdb|couchdb|rethinkdb|firebase|firestore|cosmosdb|cockroachdb)\b',
            # Cloud & DevOps (expanded)
            r'\b(aws|azure|gcp|docker|kubernetes|jenkins|terraform|ansible|ci/cd|git|github|gitlab|bitbucket|circleci|travis|bamboo|teamcity|gitlab ci|github actions|azure devops)\b',
            # Cloud services
            r'\b(ec2|s3|lambda|rds|dynamodb|cloudfront|route53|vpc|iam|cloudformation|cloudwatch|sns|sqs|ses|ecs|eks|fargate|lightsail|elastic beanstalk)\b',
            # Data & ML (expanded)
            r'\b(machine learning|ml|artificial intelligence|ai|tensorflow|pytorch|pandas|numpy|spark|hadoop|scikit-learn|keras|xgboost|lightgbm|catboost|opencv|nltk|spacy|transformers)\b',
            # Tools & Others (expanded)
            r'\b(jira|confluence|slack|agile|scrum|rest api|graphql|microservices|api development|postman|insomnia|swagger|openapi|soap|grpc|webhooks)\b',
            # Testing frameworks
            r'\b(jest|mocha|chai|cypress|selenium|pytest|junit|testng|rspec|cucumber|playwright|puppeteer|karma|jasmine)\b',
            # Healthcare specific (expanded)
            r'\b(epic|epic systems|cerner|hl7|hipaa|emr|ehr|electronic medical records|electronic health records|allscripts|athenahealth|meditech|nextgen)\b',
            r'\b(medical coding|icd-10|cpt codes|medical billing|clinical documentation|patient care|hims|health information management)\b',
            r'\b(nursing|rn|registered nurse|bsn|msn|pharmacy|pharmd|pharmacist|physician|doctor|md|pa|physician assistant)\b',
            r'\b(healthcare|health care|medical|hospital|clinical|health information|healthcare it|telemedicine|telehealth)\b',
            # Additional healthcare skills
            r'\b(patient safety|quality improvement|care coordination|discharge planning|medication administration|vital signs|charting|documentation)\b',
        ]
       
        for pattern in skill_patterns:
            matches = re.findall(pattern, job_desc_lower)
            if isinstance(matches, list):
                extracted_skills.extend(matches)
            else:
                extracted_skills.append(matches)
       
        # Enhanced extraction from structured sections
        skill_section_patterns = [
            r'(?:required|must have|proficient in|experience with|knowledge of|familiar with|expertise in)[\s:]+([^\.\n]+)',
            r'(?:skills?|technologies?|tools?|frameworks?)[\s:]+([^\.\n]+)',
            r'(?:proficiency|competency|expertise)[\s:]+([^\.\n]+)',
            r'(?:working knowledge|hands-on experience|strong background)[\s:]+([^\.\n]+)',
        ]
       
        for pattern in skill_section_patterns:
            matches = re.findall(pattern, job_desc_lower, re.IGNORECASE)
            for match in matches:
                # Split by common delimiters and clean
                skills = re.split(r'[,;|•\n•\-\–\—]', match)
                for skill in skills:
                    skill = skill.strip()
                    # Remove common prefixes/suffixes
                    skill = re.sub(r'^(years?|yrs?|experience|proficient|expert|strong|good|basic|advanced|intermediate)\s+', '', skill, flags=re.IGNORECASE)
                    skill = re.sub(r'\s+(years?|yrs?|experience|proficient|expert|strong|good|basic|advanced|intermediate)$', '', skill, flags=re.IGNORECASE)
                    if len(skill) > 2 and skill not in ['and', 'or', 'the', 'a', 'an']:
                        extracted_skills.append(skill)
       
        # Extract skills from bullet points or lists
        bullet_pattern = r'[•\-\*]\s*([a-z\s]+(?:\.js|\.net|\.jsx|\.tsx|\.py|\.java|\.cpp|\.sql|api|sdk|sdk|ui|ux)?)'
        bullet_matches = re.findall(bullet_pattern, job_desc_lower)
        extracted_skills.extend([m.strip() for m in bullet_matches if len(m.strip()) > 2])
       
        # Remove duplicates and normalize
        normalized_skills = []
        seen = set()
        for skill in extracted_skills:
            skill_lower = skill.lower().strip()
            if skill_lower and skill_lower not in seen and len(skill_lower) > 1:
                normalized_skills.append(skill_lower)
                seen.add(skill_lower)
       
        # Remove very common words that aren't skills
        stop_skills = {'years', 'experience', 'required', 'preferred', 'must', 'have', 'with', 'and', 'or', 'the', 'a', 'an'}
        normalized_skills = [s for s in normalized_skills if s not in stop_skills]
       
        logger.debug(f"Extracted {len(normalized_skills)} skills from job description")
        return normalized_skills
   
    def _calculate_domain_alignment(self, candidate, job_description):
        """Calculate domain/industry alignment score"""
        try:
            # Detect domain from job description
            job_keywords = re.findall(r'\b\w+\b', job_description.lower())
            job_domain = self.detect_domain(job_keywords)
            job_domain_lower = (job_domain or '').strip().lower()
           
            if not job_domain_lower or job_domain_lower == 'general':
                return 0.7  # Neutral score if no specific domain
           
            # Get candidate domain
            candidate_domain = candidate.get('domain', '') or candidate.get('domain_tag', '') or candidate.get('industry', '')
            candidate_domain_lower = str(candidate_domain).strip().lower()
           
            if not candidate_domain_lower:
                # Try to infer from skills/experience
                candidate_text = (
                    candidate.get('resumeText', '') or
                    candidate.get('experience', '') or
                    candidate.get('summary', '') or ''
                ).lower()
               
                candidate_keywords = re.findall(r'\b\w+\b', candidate_text)
                inferred_domain = self.detect_domain(candidate_keywords)
                candidate_domain_lower = (inferred_domain or '').strip().lower()
           
            if not candidate_domain_lower:
                return 0.5  # Neutral if can't determine
           
            # Exact match
            if candidate_domain_lower == job_domain_lower:
                return 1.0
           
            # Healthcare variations
            healthcare_terms = ['healthcare', 'health care', 'medical', 'hospital', 'clinical', 'health']
            if job_domain_lower in healthcare_terms and any(term in candidate_domain_lower for term in healthcare_terms):
                return 1.0
           
            # Tech variations
            tech_terms = ['tech', 'technology', 'it', 'software', 'engineering', 'developer']
            if job_domain_lower in tech_terms and any(term in candidate_domain_lower for term in tech_terms):
                return 1.0
           
            # Partial match
            if job_domain_lower in candidate_domain_lower or candidate_domain_lower in job_domain_lower:
                return 0.8
           
            return 0.4  # Low score for different domains
           
        except Exception as e:
            logger.error(f"Error calculating domain alignment: {e}")
            return 0.5
   
    def _calculate_title_match(self, candidate, job_description):
        """Calculate job title/role match score"""
        try:
            # Extract candidate title
            candidate_title = (
                candidate.get('title', '') or
                candidate.get('current_title', '') or
                candidate.get('job_title', '') or
                candidate.get('position', '') or ''
            ).lower()
           
            if not candidate_title:
                return 0.5  # Neutral if no title
           
            # Extract job title from description
            job_desc_lower = job_description.lower()
           
            # Common title patterns
            title_patterns = [
                r'(?:looking for|seeking|hiring|position of|role of|title:)\s+([a-z\s]+?)(?:developer|engineer|manager|analyst|specialist|director|lead|architect)',
                r'(senior|junior|mid-level)?\s*(software|backend|frontend|full.?stack|data|devops|ml|ai|cloud)?\s*(developer|engineer|architect|manager|analyst|scientist|specialist)',
            ]
           
            job_titles = []
            for pattern in title_patterns:
                matches = re.findall(pattern, job_desc_lower)
                if matches:
                    if isinstance(matches[0], tuple):
                        job_titles.extend([' '.join(m).strip() for m in matches])
                    else:
                        job_titles.extend([m.strip() for m in matches])
           
            if not job_titles:
                return 0.6  # Neutral if can't extract job title
           
            # Check for exact or partial matches
            for job_title in job_titles:
                if job_title in candidate_title or candidate_title in job_title:
                    return 1.0
               
                # Check for key role words
                role_words = ['developer', 'engineer', 'manager', 'analyst', 'architect', 'scientist', 'specialist']
                candidate_role_words = [w for w in role_words if w in candidate_title]
                job_role_words = [w for w in role_words if w in job_title]
               
                if candidate_role_words and job_role_words:
                    if set(candidate_role_words) & set(job_role_words):
                        return 0.85  # Same role type
           
            return 0.4  # Low match
           
        except Exception as e:
            logger.error(f"Error calculating title match: {e}")
            return 0.5
   
    def _calculate_experience_relevance(self, candidate, job_description):
        """Calculate how relevant the candidate's experience content is to the job"""
        try:
            # Get candidate experience text
            experience_text = (
                candidate.get('experience', '') or
                candidate.get('work_experience', '') or
                candidate.get('employment_history', '') or
                candidate.get('resumeText', '') or
                ''
            ).lower()
           
            if not experience_text:
                return 0.3  # Low score if no experience text
           
            # Extract key terms from job description
            job_keywords = self._extract_enhanced_keywords(job_description)
            job_skills = self._extract_skills_from_description(job_description)
           
            # Extract important phrases from job description
            important_phrases = self._extract_important_phrases(job_description)
           
            # Count how many job keywords appear in candidate experience
            keyword_matches = sum(1 for keyword in job_keywords[:30] if keyword in experience_text)
            keyword_match_ratio = keyword_matches / min(30, len(job_keywords)) if job_keywords else 0
           
            # Count how many job skills appear in candidate experience
            skill_matches = sum(1 for skill in job_skills if skill in experience_text)
            skill_match_ratio = skill_matches / len(job_skills) if job_skills else 0
           
            # Count how many important phrases appear
            phrase_matches = sum(1 for phrase in important_phrases if phrase.lower() in experience_text)
            phrase_match_ratio = phrase_matches / len(important_phrases) if important_phrases else 0
           
            # Weighted combination
            relevance_score = (
                keyword_match_ratio * 0.4 +
                skill_match_ratio * 0.4 +
                phrase_match_ratio * 0.2
            )
           
            return min(relevance_score, 1.0)
           
        except Exception as e:
            logger.error(f"Error calculating experience relevance: {e}")
            return 0.5
   
    def _extract_important_phrases(self, text):
        """Extract important phrases from text"""
        if not text:
            return []
       
        text_lower = text.lower()
        phrases = []
       
        # Extract technical phrases (2-3 word combinations)
        # Common patterns like "machine learning", "cloud computing", "patient care"
        phrase_patterns = [
            r'\b(machine learning|deep learning|artificial intelligence|data science|cloud computing)\b',
            r'\b(patient care|clinical documentation|electronic health records|health information)\b',
            r'\b(software development|web development|mobile development|full stack|backend|frontend)\b',
            r'\b(agile methodology|scrum master|project management|team leadership)\b',
            r'\b(api development|microservices|container orchestration|continuous integration)\b',
        ]
       
        for pattern in phrase_patterns:
            matches = re.findall(pattern, text_lower)
            phrases.extend(matches)
       
        # Also extract noun phrases (adjective + noun patterns)
        noun_phrase_pattern = r'\b(?:[a-z]+ ){1,2}(?:system|platform|application|service|framework|tool|technology)\b'
        noun_phrases = re.findall(noun_phrase_pattern, text_lower)
        phrases.extend(noun_phrases)
       
        return list(set(phrases))  # Remove duplicates
   
    def _calculate_keyword_density(self, candidate, job_description):
        """Calculate how many important keywords from job description appear in candidate profile"""
        try:
            # Get all candidate text
            candidate_text = (
                candidate.get('resumeText', '') or
                candidate.get('experience', '') or
                candidate.get('summary', '') or
                candidate.get('profile_summary', '') or
                candidate.get('description', '') or
                ''
            ).lower()
           
            if not candidate_text:
                return 0.2  # Low score if no candidate text
           
            # Extract important keywords from job description (excluding common words)
            job_keywords = self._extract_enhanced_keywords(job_description)
           
            # Filter out very common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                         'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could'}
           
            important_keywords = [kw for kw in job_keywords if kw not in stop_words and len(kw) > 3]
           
            if not important_keywords:
                return 0.5  # Neutral if no important keywords
           
            # Count matches (top 50 most important keywords)
            top_keywords = important_keywords[:50]
            matches = sum(1 for keyword in top_keywords if keyword in candidate_text)
           
            # Calculate density score
            density_score = matches / len(top_keywords) if top_keywords else 0
           
            # Bonus for high density
            if density_score >= 0.7:
                density_score = min(density_score * 1.1, 1.0)  # 10% bonus for high density
           
            return min(density_score, 1.0)
           
        except Exception as e:
            logger.error(f"Error calculating keyword density: {e}")
            return 0.5
   
    def _calculate_certification_match(self, candidate, job_description):
        """Calculate certification and license match score"""
        try:
            # Get candidate certifications
            certifications = []
            cert_sources = [
                candidate.get('certifications'),
                candidate.get('Certifications'),
                candidate.get('licenses'),
                candidate.get('Licenses'),
                candidate.get('credentials'),
            ]
           
            for source in cert_sources:
                if source:
                    if isinstance(source, list):
                        certifications.extend([str(c).lower().strip() for c in source])
                    elif isinstance(source, str):
                        certifications.extend([c.strip().lower() for c in re.split(r'[,;|]', source)])
           
            # Also check in resume text
            resume_text = (candidate.get('resumeText', '') or '').lower()
            cert_patterns = [
                r'\b(certified|certification|cert|certificate|license|licensed|credential)\s+([a-z\s]+)',
                r'\b([a-z\s]+)\s+(certified|certification|cert|certificate)',
            ]
           
            for pattern in cert_patterns:
                matches = re.findall(pattern, resume_text)
                for match in matches:
                    if isinstance(match, tuple):
                        cert_name = ' '.join(match).strip()
                    else:
                        cert_name = match.strip()
                    if len(cert_name) > 3:
                        certifications.append(cert_name)
           
            certifications = list(set(certifications))
           
            if not certifications:
                return 0.3  # Low score if no certifications
           
            # Extract required certifications from job description
            job_desc_lower = job_description.lower()
            required_certs = []
           
            # Common certification patterns
            cert_keywords = [
                'certified', 'certification', 'certificate', 'license', 'licensed',
                'credential', 'certification required', 'must have', 'preferred'
            ]
           
            # Extract certification mentions
            for keyword in cert_keywords:
                if keyword in job_desc_lower:
                    # Try to extract certification name
                    pattern = rf'{keyword}[:\s]+([a-z\s]+?)(?:\.|,|;|required|preferred)'
                    matches = re.findall(pattern, job_desc_lower)
                    required_certs.extend([m.strip() for m in matches if len(m.strip()) > 3])
           
            if not required_certs:
                return 0.6  # Neutral if no specific certification requirements
           
            # Check for matches
            matches = 0
            for req_cert in required_certs:
                for candidate_cert in certifications:
                    # Exact or partial match
                    if req_cert in candidate_cert or candidate_cert in req_cert:
                        matches += 1
                        break
                    # Check for common abbreviations
                    if self._calculate_string_similarity(req_cert, candidate_cert) > 0.8:
                        matches += 1
                        break
           
            match_ratio = matches / len(required_certs) if required_certs else 0
           
            return min(match_ratio, 1.0)
           
        except Exception as e:
            logger.error(f"Error calculating certification match: {e}")
            return 0.4
   
    def _enhance_candidate_scoring(self, candidate, job_description):
        """Enhanced candidate scoring with advanced matching algorithms for maximum accuracy (85-95%)"""
        try:
            # Extract skills from candidate (multiple sources with enhanced parsing)
            candidate_skills = []
            skill_sources = [
                candidate.get('skills'),
                candidate.get('Skills'),
                candidate.get('technical_skills'),
                candidate.get('Technical Skills'),
                candidate.get('core_skills'),
                candidate.get('proficiencies'),
                candidate.get('competencies'),
                candidate.get('expertise'),
            ]
           
            # Also extract from resume text and experience
            resume_text = (
                candidate.get('resumeText', '') or
                candidate.get('experience', '') or
                candidate.get('summary', '') or ''
            )
           
            for source in skill_sources:
                if source:
                    if isinstance(source, list):
                        candidate_skills.extend([str(s).lower().strip() for s in source])
                    elif isinstance(source, str):
                        # Split by common delimiters
                        candidate_skills.extend([s.strip().lower() for s in re.split(r'[,;|•\n•\-\–\—]', source)])
           
            # Extract skills from resume text using patterns
            if resume_text:
                resume_lower = resume_text.lower()
                # Extract technical skills mentioned in text
                tech_skill_patterns = [
                    r'\b(python|java|javascript|typescript|react|angular|vue|node\.js|aws|docker|kubernetes|sql|mongodb|postgresql)\b',
                    r'\b(machine learning|ml|ai|tensorflow|pytorch|pandas|numpy)\b',
                    r'\b(epic|cerner|hl7|hipaa|emr|ehr|nursing|pharmacy|medical coding)\b',
                ]
                for pattern in tech_skill_patterns:
                    matches = re.findall(pattern, resume_lower)
                    candidate_skills.extend([m.lower() if isinstance(m, str) else m[0].lower() for m in matches])
           
            # Clean and normalize skills
            normalized_skills = []
            seen = set()
            for skill in candidate_skills:
                skill = skill.strip().lower()
                # Remove common prefixes/suffixes
                skill = re.sub(r'^(years?|yrs?|experience|proficient|expert|strong|good|basic|advanced|intermediate)\s+', '', skill)
                skill = re.sub(r'\s+(years?|yrs?|experience|proficient|expert|strong|good|basic|advanced|intermediate)$', '', skill)
                if skill and len(skill) > 1 and skill not in seen:
                    normalized_skills.append(skill)
                    seen.add(skill)
           
            candidate_skills = normalized_skills
           
            # Extract skills from job description with enhanced extraction
            job_keywords = self._extract_enhanced_keywords(job_description)
            job_skills = self._extract_skills_from_description(job_description)
           
            # Calculate enhanced semantic similarity
            candidate_text = (
                candidate.get('resumeText', '') or
                candidate.get('experience', '') or
                candidate.get('summary', '') or
                candidate.get('profile_summary', '') or
                candidate.get('description', '') or
                ''
            )
           
            base_similarity = self.semantic_similarity(candidate_text, job_description)
           
            # Identify critical/required skills from job description
            critical_skills = self._identify_critical_skills(job_description, job_skills)
           
            # Enhanced fuzzy skill matching with optimized threshold for maximum accuracy
            # Lower threshold (0.65) allows for better fuzzy matching while maintaining accuracy
            skill_match_score = self._fuzzy_match_skills(candidate_skills, job_skills, threshold=0.65, critical_skills=critical_skills)
           
            # Calculate experience match (enhanced)
            experience_match = self._calculate_experience_match(candidate, job_description)
           
            # Calculate experience relevance (not just years, but relevant experience)
            experience_relevance = self._calculate_experience_relevance(candidate, job_description)
           
            # Calculate location match (enhanced)
            location_match = self._calculate_location_match(candidate, job_description)
           
            # Calculate education match (enhanced)
            education_match = self._calculate_education_bonus(candidate, job_description)
           
            # Calculate domain alignment
            domain_alignment = self._calculate_domain_alignment(candidate, job_description)
           
            # Calculate title/role match
            title_match = self._calculate_title_match(candidate, job_description)
           
            # Calculate keyword density match (how many important keywords appear)
            keyword_density = self._calculate_keyword_density(candidate, job_description)
           
            # Calculate certification/license match
            certification_match = self._calculate_certification_match(candidate, job_description)
           
            # Enhanced weighted final score (optimized for maximum accuracy)
            # Weights carefully tuned for precision
            final_score = (
                skill_match_score * 0.32 +         # Skills (most critical - increased weight)
                base_similarity * 0.22 +          # Semantic similarity (foundation)
                experience_match * 0.15 +         # Experience years match
                experience_relevance * 0.10 +     # Relevant experience content
                domain_alignment * 0.08 +         # Domain match (critical for accuracy)
                keyword_density * 0.06 +          # Keyword density (important terms)
                title_match * 0.04 +              # Title/role match
                education_match * 0.02 +          # Education
                certification_match * 0.01 +     # Certifications/licenses
                location_match * 0.00              # Location (minimal weight, flexible)
            )
           
            # Advanced boost/penalty system for maximum accuracy
            critical_scores = {
                'skills': skill_match_score,
                'experience': experience_match,
                'domain': domain_alignment,
                'semantic': base_similarity
            }
           
            # Count strong matches (>= 0.85) and very strong matches (>= 0.95)
            strong_matches = sum(1 for score in critical_scores.values() if score >= 0.85)
            very_strong_matches = sum(1 for score in critical_scores.values() if score >= 0.95)
           
            # Progressive boost system
            if very_strong_matches >= 3:
                final_score *= 1.15  # 15% boost for exceptional matches
            elif very_strong_matches >= 2:
                final_score *= 1.12  # 12% boost for very strong matches
            elif strong_matches >= 3:
                final_score *= 1.10  # 10% boost for multiple strong matches
            elif strong_matches >= 2:
                final_score *= 1.07  # 7% boost for good matches
           
            # Penalty system for poor matches
            weak_critical = sum(1 for score in critical_scores.values() if score < 0.3)
            if weak_critical >= 2:
                final_score *= 0.60  # 40% penalty for multiple weak critical areas
            elif weak_critical >= 1:
                if skill_match_score < 0.2 or base_similarity < 0.25:
                    final_score *= 0.70  # 30% penalty for poor skill or semantic match
           
            # Bonus for perfect skill match
            if skill_match_score >= 0.95:
                final_score *= 1.05  # 5% bonus for near-perfect skill match
           
            # Bonus for perfect domain match
            if domain_alignment >= 0.95:
                final_score *= 1.03  # 3% bonus for perfect domain match
           
            # Penalty for domain mismatch (critical for accuracy)
            if domain_alignment < 0.4:
                final_score *= 0.75  # 25% penalty for domain mismatch
           
            # Ensure minimum quality threshold
            if final_score < 0.15:  # Very low scores
                final_score *= 0.5  # Further reduce very poor matches
           
            return min(final_score * 100, 100)  # Convert to percentage and cap at 100
           
        except Exception as e:
            logger.error(f"Error in enhanced candidate scoring: {e}")
            return 0.0
   
    def _calculate_experience_match(self, candidate, job_description):
        """Calculate experience match score with enhanced accuracy"""
        try:
            # Enhanced patterns to extract required experience
            exp_patterns = [
                r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:relevant\s*)?experience',
                r'(\d+)\+?\s*years?\s*(?:in|with|of)',
                r'minimum\s*(?:of\s*)?(\d+)\s*years?',
                r'at\s*least\s*(\d+)\s*years?',
                r'(\d+)\+?\s*years?\s*(?:professional|work)',
                r'(\d+)\+?\s*y\.?o\.?e\.?',  # years of experience abbreviation
                r'(\d+)\+?\s*yrs?\.?',  # yrs abbreviation
            ]
           
            required_years = 0
            for pattern in exp_patterns:
                match = re.search(pattern, job_description.lower())
                if match:
                    required_years = int(match.group(1))
                    break
           
            if required_years == 0:
                return 1.0  # No specific requirement - don't penalize
           
            # Enhanced candidate experience extraction
            candidate_exp = 0
            exp_sources = [
                candidate.get('total_experience_years'),
                candidate.get('experience_years'),
                candidate.get('Experience'),
                candidate.get('experience'),
                candidate.get('years_of_experience'),
                candidate.get('yoe')
            ]
           
            for exp_source in exp_sources:
                if exp_source is not None:
                    if isinstance(exp_source, (int, float)):
                        candidate_exp = int(exp_source)
                        break
                    elif isinstance(exp_source, str):
                        # Extract all numbers and take the largest (likely total years)
                        numbers = re.findall(r'\d+(?:\.\d+)?', exp_source)
                        if numbers:
                            candidate_exp = max(int(float(n)) for n in numbers)
                            break
           
            if candidate_exp == 0:
                return 0.3  # Penalize if no experience data available
           
            # Enhanced scoring with more nuanced tiers
            if candidate_exp >= required_years * 1.2:
                return 1.0  # Overqualified - full score
            elif candidate_exp >= required_years:
                return 1.0  # Meets requirement exactly
            elif candidate_exp >= required_years * 0.9:
                return 0.95  # Very close (90%+)
            elif candidate_exp >= required_years * 0.8:
                return 0.85  # Close (80%+)
            elif candidate_exp >= required_years * 0.7:
                return 0.70  # Moderate (70%+)
            elif candidate_exp >= required_years * 0.6:
                return 0.55  # Below requirement (60%+)
            elif candidate_exp >= required_years * 0.5:
                return 0.40  # Significantly below (50%+)
            else:
                return 0.20  # Too low experience
               
        except Exception as e:
            logger.error(f"Error calculating experience match: {e}")
            return 0.5
   
    def _calculate_location_match(self, candidate, job_description):
        """Calculate location match score with enhanced accuracy"""
        try:
            candidate_location = candidate.get('location', '').lower()
            job_desc_lower = job_description.lower()
           
            if not candidate_location:
                return 0.3  # Penalize if no location info
           
            # Check for remote work keywords
            remote_keywords = ['remote', 'work from home', 'wfh', 'distributed', 'work anywhere',
                             'location independent', 'virtual', 'telecommute', 'telecommuting']
            is_remote_job = any(keyword in job_desc_lower for keyword in remote_keywords)
           
            if is_remote_job:
                # Check if candidate is open to remote
                candidate_remote_indicators = ['remote', 'anywhere', 'open to remote', 'wfh',
                                              'work from home', 'distributed']
                if any(indicator in candidate_location for indicator in candidate_remote_indicators):
                    return 1.0  # Perfect match for remote
                # If job is remote, location is less critical
                return 0.8  # High score for remote jobs
           
            # Check for hybrid work
            hybrid_keywords = ['hybrid', 'flexible', 'part remote', 'part-time remote']
            is_hybrid_job = any(keyword in job_desc_lower for keyword in hybrid_keywords)
           
            if is_hybrid_job:
                return 0.85  # High score for hybrid (location less critical)
           
            # Extract specific location from job description
            location_patterns = [
                r'(?:in|at|based in|located in|office in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*([A-Z]{2})',  # City, State
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+office',
            ]
           
            job_locations = []
            for pattern in location_patterns:
                matches = re.findall(pattern, job_description)
                if matches:
                    if isinstance(matches[0], tuple):
                        job_locations.extend([loc.lower() for loc in matches[0] if loc])
                    else:
                        job_locations.extend([m.lower() for m in matches])
           
            # Normalize candidate location
            candidate_parts = [part.strip() for part in candidate_location.split(',')]
            candidate_city = candidate_parts[0].lower() if candidate_parts else ''
            candidate_state = candidate_parts[1].lower() if len(candidate_parts) > 1 else ''
            candidate_country = candidate_parts[2].lower() if len(candidate_parts) > 2 else ''
           
            # Exact match
            if candidate_location in job_desc_lower or any(part in job_desc_lower for part in candidate_parts):
                return 1.0
           
            # City match
            if candidate_city and any(candidate_city in loc or loc in candidate_city for loc in job_locations):
                return 0.95
           
            # State/region match
            us_states = {
                'california': 'ca', 'texas': 'tx', 'florida': 'fl', 'new york': 'ny',
                'pennsylvania': 'pa', 'illinois': 'il', 'ohio': 'oh', 'georgia': 'ga',
                'north carolina': 'nc', 'michigan': 'mi', 'new jersey': 'nj', 'virginia': 'va',
                'washington': 'wa', 'arizona': 'az', 'massachusetts': 'ma', 'tennessee': 'tn',
                'indiana': 'in', 'missouri': 'mo', 'maryland': 'md', 'wisconsin': 'wi'
            }
           
            candidate_state_code = us_states.get(candidate_state, candidate_state[:2] if len(candidate_state) >= 2 else '')
           
            if candidate_state_code:
                if candidate_state_code in job_desc_lower or candidate_state in job_desc_lower:
                    return 0.80  # State match
           
            # Country match
            if candidate_country and candidate_country in job_desc_lower:
                return 0.70  # Country match
           
            # Partial match (substring)
            for job_loc in job_locations:
                if job_loc in candidate_location or candidate_location in job_loc:
                    return 0.65
           
            # No match - penalize but don't eliminate
            return 0.25
           
        except Exception as e:
            logger.error(f"Error calculating location match: {e}")
            return 0.5
   
    def _calculate_education_bonus(self, candidate, job_description):
        """Calculate education bonus with enhanced accuracy"""
        try:
            # Extract education from multiple candidate fields
            education_sources = [
                candidate.get('education'),
                candidate.get('Education'),
                candidate.get('qualification'),
                candidate.get('qualifications'),
                candidate.get('degree'),
                candidate.get('degrees')
            ]
           
            education_text = ''
            for source in education_sources:
                if source:
                    if isinstance(source, list):
                        education_text += ' '.join(str(e) for e in source) + ' '
                    else:
                        education_text += str(source) + ' '
           
            education_text = education_text.lower().strip()
           
            if not education_text:
                return 0.2  # Low score if no education data
           
            # Extract education requirements from job description
            job_desc_lower = job_description.lower()
           
            # Degree level requirements
            degree_keywords = {
                'phd': ['phd', 'ph.d', 'doctorate', 'doctoral', 'd.phil'],
                'master': ['master', 'masters', 'ms', 'm.s', 'mba', 'm.sc', 'm.a', 'm.ed'],
                'bachelor': ['bachelor', 'bachelors', 'bs', 'b.s', 'ba', 'b.a', 'b.sc', 'b.tech', 'b.e', 'bachelor\'s'],
                'associate': ['associate', 'a.a', 'a.s', 'diploma'],
                'degree': ['degree', 'graduate', 'undergraduate', 'college']
            }
           
            # Check what degree level is required
            required_level = None
            for level, keywords in degree_keywords.items():
                if any(keyword in job_desc_lower for keyword in keywords):
                    required_level = level
                    break
           
            # Check candidate's education level
            candidate_level = None
            candidate_level_score = {}
           
            for level, keywords in degree_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in education_text)
                if matches > 0:
                    candidate_level_score[level] = matches
                    if candidate_level is None or level in ['phd', 'master', 'bachelor']:
                        candidate_level = level
           
            # Scoring logic
            if required_level is None:
                # No specific requirement - check if candidate has any degree
                if candidate_level:
                    return 0.8  # Good score for having a degree
                else:
                    return 0.4  # Lower score if no degree mentioned
           
            # Match required level
            level_hierarchy = {'phd': 4, 'master': 3, 'bachelor': 2, 'associate': 1, 'degree': 1}
            required_rank = level_hierarchy.get(required_level, 0)
            candidate_rank = level_hierarchy.get(candidate_level, 0)
           
            if candidate_rank >= required_rank:
                if candidate_rank == required_rank:
                    return 1.0  # Exact match
                else:
                    return 0.95  # Overqualified (still good)
            elif candidate_rank >= required_rank - 1:
                return 0.70  # Close (e.g., bachelor when master required)
            else:
                return 0.40  # Below requirement
           
        except Exception as e:
            logger.error(f"Error calculating education bonus: {e}")
            return 0.3
   
    def _get_suggested_candidates(self, job_description: str, top_k: int = 5) -> List[Dict]:
        """Get suggested candidates when no matches found - using only technical skills"""
        try:
            logger.info(f"Getting suggested candidates for: {job_description[:100]}...")
            
            # Extract ONLY technical skills from job description
            all_skills = self.fallback_algorithm._extract_skills_from_job(job_description)
            technical_skills_only = _filter_technical_skills_only(all_skills)
            
            logger.info(f"Technical skills extracted: {technical_skills_only}")
            
            # Get candidates from fallback algorithm
            candidates = getattr(self.fallback_algorithm, 'candidates', [])
            if not technical_skills_only or not candidates:
                return []
            
            # Score candidates based on technical skills only
            scored_candidates = []
            for candidate in candidates[:1000]:  # Limit to first 1000 for performance
                candidate_skills = candidate.get('skills', [])
                if isinstance(candidate_skills, str):
                    candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                
                # Filter candidate skills to technical only
                candidate_technical_skills = _filter_technical_skills_only(candidate_skills)
                
                # Calculate technical skill match
                matches = 0
                for tech_skill in technical_skills_only:
                    tech_skill_lower = tech_skill.lower()
                    for candidate_tech in candidate_technical_skills:
                        candidate_tech_lower = candidate_tech.lower()
                        if tech_skill_lower in candidate_tech_lower or candidate_tech_lower in tech_skill_lower:
                            matches += 1
                            break
                
                if matches > 0:
                    match_ratio = matches / len(technical_skills_only)
                    scored_candidates.append((match_ratio, candidate))
            
            # Sort by match ratio and return top candidates
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            suggested = [candidate for _, candidate in scored_candidates[:top_k]]
            
            logger.info(f"Found {len(suggested)} suggested candidates based on technical skills")
            return suggested
            
        except Exception as e:
            logger.error(f"Error getting suggested candidates: {e}")
            return []

    def keyword_search(self, job_description, top_k=20):
        """Advanced search using the original AdeptAI algorithm with caching"""
        # Validate query - reject meaningless queries early
        if _is_meaningless_query(job_description):
            logger.warning(f"Meaningless query rejected in AdeptAIMastersAlgorithm.keyword_search: '{job_description[:100] if job_description else ''}...'")
            return [], "No candidates found. Please provide a valid job description with meaningful keywords."
        
        start_time = time.time()
        self.performance_stats['total_searches'] += 1
       
        # Record search start in performance monitor
        self._record_performance(0, job_description, cache_hit=False)
       
        # Handle None job_description
        if job_description is None:
            job_description = ""
       
        job_keywords = re.findall(r'\b\w+\b', job_description.lower())
        job_domain = self.detect_domain(job_keywords)
        job_domain_lower = (job_domain or '').strip().lower()
       
        logger.info(f"Starting ORIGINAL AdeptAI search for: '{job_description[:100]}...' (target: {top_k} candidates)")
       
            # Check cache first
        try:
            cached_results = search_cache.get_search_results(job_description, {'top_k': top_k})
            if cached_results:
                logger.info("Cache hit for search query")
                # Remove duplicates from cached results
                cached_results = self._deduplicate_candidates(cached_results)
                # Sort cached results by match score before domain filtering
                try:
                    cached_results.sort(key=lambda x: x.get('match_percentage', x.get('Score', x.get('score', 0))), reverse=True)
                except Exception as e:
                    logger.debug(f"Error sorting cached results: {e}")
                filtered_results = self._apply_domain_filter(cached_results, job_domain_lower)
                # Sort again after domain filtering
                try:
                    filtered_results.sort(key=lambda x: x.get('match_percentage', x.get('Score', x.get('score', 0))), reverse=True)
                except Exception as e:
                    logger.debug(f"Error sorting filtered cached results: {e}")
                # Ensure we return exactly top_k candidates
                filtered_results = filtered_results[:top_k]
                response_time = time.time() - start_time
                self.performance_stats['avg_response_time'] = (
                    (self.performance_stats['avg_response_time'] * (self.performance_stats['total_searches'] - 1) + response_time) /
                    self.performance_stats['total_searches']
                )
                summary = f"Found {len(filtered_results)} candidates (cached"
                if job_domain_lower and job_domain_lower != 'general':
                    summary += f", domain: {job_domain}"
                summary += ")"
                return filtered_results, summary
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
       
        # Try original algorithm first
        if self.enhanced_system:
            try:
                logger.info("Using ORIGINAL AdeptAI algorithm...")
               
                # Use batch processing if available
                if hasattr(self, 'batch_processor') and self.batch_processor:
                    logger.info("Using batch processing for enhanced performance")
               
                # Request many more candidates for domain filtering to ensure we get 20 domain-matched results
                # For domain filtering, we need to request 5-10x more candidates since many will be filtered out
                if job_domain_lower and job_domain_lower != 'general':
                    search_top_k = top_k * 10  # Request 10x to ensure enough domain matches
                    logger.info(f"Domain filtering active: Requesting {search_top_k} candidates from algorithm (will filter to {top_k} domain-matched candidates)")
                else:
                    search_top_k = top_k
                    logger.info(f"No domain filter: Requesting {search_top_k} candidates from algorithm")
               
                # Extract required skills and detect domain FIRST for strict filtering
                # Use fallback algorithm's skill extraction method
                job_required_skills = self.fallback_algorithm._extract_skills_from_job(job_description)
                logger.info(f"[DEBUG] AdeptAIMastersAlgorithm - Domain: {job_domain_lower}, Required skills: {job_required_skills}")
               
                # Use the original algorithm's search method
                original_results = self.enhanced_system.search(job_description, top_k=search_top_k)
               
                if original_results:
                    # Format results for frontend
                    formatted_results = []
                    rejected_count = 0
                    rejected_domain = 0
                    rejected_skills = 0
                   
                    logger.info(f"[DEBUG] AdeptAIMastersAlgorithm: Processing {len(original_results)} results from original algorithm")
                   
                    for result in original_results:
                        formatted_result = self._format_original_result_for_frontend(result, job_description, job_required_skills)
                       
                        # STRICT PRE-FILTERING: Apply skill and domain filtering before adding to results
                        if job_required_skills or job_domain_lower:
                            candidate_name = formatted_result.get('full_name') or formatted_result.get('FullName', 'Unknown')
                           
                            # Check domain mismatch (strict rejection)
                            if job_domain_lower and job_domain_lower != 'general':
                                candidate_domain = (
                                    formatted_result.get('domain_tag') or
                                    formatted_result.get('category') or
                                    formatted_result.get('domain') or
                                    ''
                                ).strip().lower()
                               
                                # Strict domain rejection for healthcare jobs
                                if job_domain_lower == 'healthcare':
                                    if candidate_domain in {'it/tech', 'technology', 'tech', 'it', 'software', 'engineering'}:
                                        logger.warning(f"[DEBUG] REJECTED {candidate_name}: IT/Tech domain for healthcare job")
                                        rejected_count += 1
                                        rejected_domain += 1
                                        continue
                               
                                # Check candidate skills for domain mismatch
                                candidate_skills = formatted_result.get('skills', [])
                                if isinstance(candidate_skills, str):
                                    candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                                candidate_skills_lower = [str(s).lower().strip() for s in candidate_skills if s]
                               
                                # STRICT: For healthcare jobs, require ACTUAL clinical/nursing skills, not just generic "healthcare"
                                if job_domain_lower == 'healthcare':
                                    # Non-clinical skills that should be rejected for clinical healthcare roles
                                    non_clinical_skills = {
                                        'accounting', 'account', 'accounts receivable', 'accounts payable',
                                        'customer service', 'customer care', 'customer support',
                                        'data entry', 'data processing', 'administrative',
                                        'sales', 'marketing', 'finance', 'hr', 'human resources',
                                        'receptionist', 'secretary', 'office management', 'billing',
                                        'insurance', 'claims processing', 'medical billing', 'medical coding',
                                        'communication'  # Generic communication is not a clinical skill
                                    }
                                   
                                    # ACTUAL clinical/nursing skills required for healthcare jobs
                                    clinical_healthcare_skills = {
                                        'rn', 'registered nurse', 'nurse', 'nursing', 'lpn', 'cna',
                                        'patient care', 'clinical', 'clinical care', 'direct patient care',
                                        'icu', 'ccu', 'er', 'emergency', 'med-surg', 'medical surgical',
                                        'bls', 'acls', 'cpr', 'pals', 'tncc',
                                        'medication administration', 'med administration', 'med pass',
                                        'charting', 'clinical documentation', 'care plan', 'care planning',
                                        'case management', 'discharge planning', 'vital signs', 'vitals',
                                        'phlebotomy', 'blood draw', 'iv therapy', 'intravenous',
                                        'wound care', 'dressing change', 'infection control',
                                        'ehr', 'emr', 'epic', 'cerner', 'meditech',
                                        'therapist', 'occupational therapy', 'physical therapy', 'respiratory therapy',
                                        'physician', 'doctor', 'surgeon', 'anesthesiologist'
                                    }
                                   
                                    # Check if candidate has actual clinical skills
                                    has_clinical_skills = any(
                                        clinical_skill in ' '.join(candidate_skills_lower)
                                        for clinical_skill in clinical_healthcare_skills
                                    )
                                   
                                    # Check if candidate has non-clinical skills
                                    has_non_clinical = any(
                                        non_clinical in ' '.join(candidate_skills_lower)
                                        for non_clinical in non_clinical_skills
                                    )
                                   
                                    # Count clinical vs non-clinical skills
                                    clinical_count = sum(1 for skill in candidate_skills_lower
                                                        if any(cs in skill for cs in clinical_healthcare_skills))
                                    non_clinical_count = sum(1 for skill in candidate_skills_lower
                                                            if any(ncs in skill for ncs in non_clinical_skills))
                                   
                                    # REJECT if:
                                    # 1. No clinical skills at all, OR
                                    # 2. Has non-clinical skills and clinical skills count is less than non-clinical count
                                    if not has_clinical_skills:
                                        logger.warning(f"[DEBUG] REJECTED {candidate_name}: No clinical skills for healthcare job (skills: {candidate_skills_lower[:5]})")
                                        rejected_count += 1
                                        rejected_domain += 1
                                        continue
                                    elif has_non_clinical and clinical_count < non_clinical_count:
                                        logger.warning(f"[DEBUG] REJECTED {candidate_name}: More non-clinical skills ({non_clinical_count}) than clinical skills ({clinical_count}) for healthcare job")
                                        rejected_count += 1
                                        rejected_domain += 1
                                        continue
                           
                            # Check skill match - STRICT: Only match on TECHNICAL skills, reject soft skills
                            if job_required_skills:
                                candidate_skills = formatted_result.get('skills', [])
                                if isinstance(candidate_skills, str):
                                    candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                                
                                # CRITICAL: Filter candidate skills to TECHNICAL ONLY - exclude soft skills
                                candidate_technical_skills = _filter_technical_skills_only(candidate_skills)
                                candidate_technical_skills_lower = [str(s).lower().strip() for s in candidate_technical_skills if s]
                               
                                # If candidate has NO technical skills, reject immediately
                                if not candidate_technical_skills_lower:
                                    logger.warning(f"[DEBUG] REJECTED {candidate_name}: No technical skills found (only soft skills)")
                                    rejected_count += 1
                                    rejected_skills += 1
                                    continue
                               
                                # Calculate TECHNICAL skill match ratio
                                job_skills_lower = [str(s).lower().strip() for s in job_required_skills if s]
                                matches = sum(1 for skill in job_skills_lower if any(skill in cs or cs in skill for cs in candidate_technical_skills_lower))
                                skill_match_ratio = matches / len(job_skills_lower) if job_skills_lower else 0.0
                               
                                matched_skills = [skill for skill in job_skills_lower if any(skill in cs or cs in skill for cs in candidate_technical_skills_lower)]
                               
                                # STRICT: For exact matches, require at least 50% technical skill match
                                skill_threshold = 0.50  # Require 50% technical skill match for exact matches
                               
                                # REJECT if technical skill match is below threshold
                                if skill_match_ratio < skill_threshold:
                                    logger.warning(f"[DEBUG] REJECTED {candidate_name}: Insufficient technical skill match ({skill_match_ratio:.2%} < {skill_threshold:.0%}, matched: {matched_skills})")
                                    rejected_count += 1
                                    rejected_skills += 1
                                    continue
                               
                                logger.debug(f"[DEBUG] ACCEPTED {candidate_name}: Technical skill match {skill_match_ratio:.2%} (matched: {matched_skills})")
                       
                        # If we get here, candidate passed all filters
                        # Enhance with skill matching if available
                        if hasattr(self, 'skill_matcher') and self.skill_matcher:
                            try:
                                # Extract skills from the result
                                candidate_skills = result.get('skills', [])
                                if isinstance(candidate_skills, str):
                                    candidate_skills = [candidate_skills]
                               
                                # Calculate enhanced skill score
                                skill_score = self.skill_matcher.calculate_skill_match_score(
                                    candidate_skills,
                                    job_description
                                )
                                formatted_result['enhanced_skill_score'] = skill_score
                            except Exception as e:
                                logger.warning(f"Skill matching failed: {e}")
                       
                        formatted_results.append(formatted_result)
                   
                    logger.info(f"[DEBUG] AdeptAIMastersAlgorithm strict pre-filtering COMPLETE:")
                    logger.info(f"[DEBUG]   - Total rejected: {rejected_count}")
                    logger.info(f"[DEBUG]   - Rejected (domain mismatch): {rejected_domain}")
                    logger.info(f"[DEBUG]   - Rejected (insufficient skill match): {rejected_skills}")
                    logger.info(f"[DEBUG]   - Remaining candidates: {len(formatted_results)}")
                    
                    # Remove duplicates before further processing
                    formatted_results = self._deduplicate_candidates(formatted_results)
                    logger.info(f"[DEBUG] After deduplication: {len(formatted_results)} unique candidates")
                    
                    # Fallback: If all candidates were filtered out, relax filtering
                    if len(formatted_results) == 0:
                        logger.warning(f"[DEBUG] All candidates filtered out! Relaxing filters...")
                        # Re-process results with relaxed filtering - accept candidates with any skill match
                        for result in original_results:
                            try:
                                formatted_result = self._format_original_result_for_frontend(result, job_description, job_required_skills)
                                candidate_name = formatted_result.get('full_name') or formatted_result.get('FullName', 'Unknown')
                               
                                # Relaxed check: Accept if candidate has ANY matching skill or healthcare indicator
                                if job_required_skills:
                                    candidate_skills = formatted_result.get('skills', [])
                                    if isinstance(candidate_skills, str):
                                        candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                                    candidate_skills_lower = [str(s).lower().strip() for s in candidate_skills if s]
                                    job_skills_lower = [str(s).lower().strip() for s in job_required_skills if s]
                                   
                                    # Check for any skill match
                                    has_any_match = any(skill in ' '.join(candidate_skills_lower) or
                                                       any(skill in cs or cs in skill for cs in candidate_skills_lower)
                                                       for skill in job_skills_lower)
                                   
                                    # For healthcare, be more lenient - accept candidates with healthcare-related keywords
                                    if job_domain_lower == 'healthcare':
                                        healthcare_indicators = {'rn', 'nurse', 'nursing', 'medical', 'clinical', 'healthcare',
                                                                 'patient', 'hospital', 'clinic', 'icu', 'bls', 'acls', 'cpr',
                                                                 'lpn', 'cna', 'patient care', 'vitals', 'medication', 'charting',
                                                                 'therapist', 'pharmacy', 'physician', 'doctor', 'surgeon'}
                                        has_healthcare_indicator = any(indicator in ' '.join(candidate_skills_lower)
                                                                      for indicator in healthcare_indicators)
                                        # Also check resume text for healthcare keywords
                                        resume_text = formatted_result.get('resume_text', '') or formatted_result.get('resumeText', '')
                                        if isinstance(resume_text, str):
                                            resume_lower = resume_text.lower()
                                            has_healthcare_in_resume = any(indicator in resume_lower for indicator in healthcare_indicators)
                                        else:
                                            has_healthcare_in_resume = False
                                       
                                        if has_any_match or has_healthcare_indicator or has_healthcare_in_resume:
                                            formatted_results.append(formatted_result)
                                            logger.debug(f"[DEBUG] RELAXED ACCEPT {candidate_name}: Has matching skill or healthcare indicator")
                                    elif has_any_match:
                                        formatted_results.append(formatted_result)
                                        logger.debug(f"[DEBUG] RELAXED ACCEPT {candidate_name}: Has matching skill")
                                else:
                                    # No required skills, accept all
                                    formatted_results.append(formatted_result)
                            except Exception as e:
                                logger.debug(f"Error processing result in relaxed filtering: {e}")
                                continue
                       
                        logger.info(f"[DEBUG] Relaxed filtering found {len(formatted_results)} candidates")
                        
                        # Remove duplicates after relaxed filtering
                        formatted_results = self._deduplicate_candidates(formatted_results)
                        logger.info(f"[DEBUG] After relaxed filtering deduplication: {len(formatted_results)} unique candidates")
                        
                        # If still no results after relaxed filtering, use fallback algorithm
                        if len(formatted_results) == 0:
                            logger.warning(f"[DEBUG] Still no candidates after relaxed filtering. Using fallback algorithm...")
                            try:
                                fallback_results, fallback_summary = self.fallback_algorithm.keyword_search(job_description, top_k * 5)
                                # Format fallback results
                                for fallback_result in fallback_results:
                                    formatted_results.append(fallback_result)
                                # Remove duplicates after adding fallback results
                                formatted_results = self._deduplicate_candidates(formatted_results)
                                logger.info(f"[DEBUG] Fallback algorithm found {len(formatted_results)} unique candidates after deduplication")
                            except Exception as fallback_error:
                                logger.error(f"Fallback algorithm error: {fallback_error}")
                                logger.error(f"Error in semantic matching: {fallback_error}")
                   
                    # Sort by match score (highest first) before domain filtering
                    try:
                        formatted_results.sort(key=lambda x: x.get('match_percentage', x.get('Score', x.get('score', 0))), reverse=True)
                    except Exception as e:
                        logger.debug(f"Error sorting formatted results: {e}")
                   
                    self.performance_stats['original_algorithm_used'] += 1
                    response_time = time.time() - start_time
                    self.performance_stats['avg_response_time'] = (
                        (self.performance_stats['avg_response_time'] * (self.performance_stats['total_searches'] - 1) + response_time) /
                        self.performance_stats['total_searches']
                    )
                   
                    # Record performance metrics
                    self._record_performance(response_time, job_description, cache_hit=False)
                    
                    # Remove duplicates before domain filtering
                    formatted_results = self._deduplicate_candidates(formatted_results)
                    
                    filtered_results = self._apply_domain_filter(formatted_results, job_domain_lower)
                    # Sort again after domain filtering to ensure highest scores first
                    try:
                        filtered_results.sort(key=lambda x: x.get('match_percentage', x.get('Score', x.get('score', 0))), reverse=True)
                    except Exception as e:
                        logger.debug(f"Error sorting filtered results: {e}")
                    # Ensure we return exactly top_k candidates
                    filtered_results = filtered_results[:top_k]
                    summary = f"Found {len(filtered_results)} candidates using ORIGINAL AdeptAI algorithm"
                    if job_domain_lower and job_domain_lower != 'general':
                        summary += f" (domain: {job_domain})"
                    logger.info(f"Original algorithm search completed: {len(filtered_results)} results in {response_time:.2f}s")
                   
                    # If no results found, get suggested candidates
                    if len(filtered_results) == 0:
                        logger.info("No matches found. Getting suggested candidates based on technical skills only...")
                        suggested_candidates = self._get_suggested_candidates(job_description, top_k=5)
                        
                        # Format suggested candidates
                        for candidate in suggested_candidates:
                            try:
                                formatted_result = self._format_original_result_for_frontend(candidate, job_description, [])
                                # Filter skills to technical only
                                all_skills = formatted_result.get('skills', [])
                                technical_skills = _filter_technical_skills_only(all_skills)
                                formatted_result['skills'] = technical_skills
                                formatted_result['Skills'] = technical_skills
                                formatted_result['match_percentage'] = 0.0
                                formatted_result['Score'] = 0.0
                                formatted_result['is_suggested'] = True
                                formatted_result['suggestion_reason'] = 'No exact matches found. Showing candidates with similar technical skills.'
                                filtered_results.append(formatted_result)
                            except Exception as e:
                                logger.error(f"Error formatting suggested candidate: {e}")
                                continue
                        
                        summary = f"No exact matches found. Showing {len(filtered_results)} suggested candidates based on technical skills."
                    
                    # Cache the results
                    try:
                        search_cache.set_search_results(job_description, filtered_results, {'top_k': top_k})
                        logger.debug("Search results cached successfully")
                    except Exception as e:
                        logger.warning(f"Failed to cache search results: {e}")
                   
                    return filtered_results, summary
                else:
                    logger.warning("Original algorithm returned no results, falling back to fallback search")
            except Exception as e:
                logger.error(f"Original algorithm error: {e}")
                logger.info("Falling back to fallback search...")
       
        # Fallback to simple search
        logger.info("Using FALLBACK search (simple keyword matching)...")
        self.performance_stats['fallback_used'] += 1
       
        # Record fallback usage
        self._record_performance(0, job_description, cache_hit=False)
        # Request many more candidates for fallback to account for domain filtering
        if job_domain_lower and job_domain_lower != 'general':
            fallback_top_k = top_k * 10  # Request 10x for domain filtering
            logger.info(f"Domain filtering active: Requesting {fallback_top_k} candidates from fallback (will filter to {top_k} domain-matched)")
        else:
            fallback_top_k = top_k
        fallback_results, summary = self.fallback_algorithm.keyword_search(job_description, fallback_top_k)
        # Sort fallback results by match score before domain filtering
        try:
            fallback_results.sort(key=lambda x: x.get('match_percentage', x.get('Score', x.get('score', 0))), reverse=True)
        except Exception as e:
            logger.debug(f"Error sorting fallback results: {e}")
       
        # Remove duplicates before domain filtering
        fallback_results = self._deduplicate_candidates(fallback_results)
        
        filtered_results = self._apply_domain_filter(fallback_results, job_domain_lower)
        # Sort again after domain filtering to ensure highest scores first
        try:
            filtered_results.sort(key=lambda x: x.get('match_percentage', x.get('Score', x.get('score', 0))), reverse=True)
        except Exception as e:
            logger.debug(f"Error sorting filtered fallback results: {e}")
        # Ensure we return exactly top_k candidates
        filtered_results = filtered_results[:top_k]
        
        # If no results found, get suggested candidates
        if len(filtered_results) == 0:
            logger.info("No matches found. Getting suggested candidates based on technical skills only...")
            suggested_candidates = self._get_suggested_candidates(job_description, top_k=5)
            
            # Format suggested candidates using fallback algorithm
            for candidate in suggested_candidates:
                try:
                    # Use fallback algorithm's normalization
                    normalized = _normalize_candidate_profile(candidate)
                    # Filter skills to technical only
                    all_skills = normalized.get('skills', [])
                    technical_skills = _filter_technical_skills_only(all_skills)
                    
                    # Create result entry
                    suggested_result = {
                        'email': normalized.get('email', 'unknown'),
                        'full_name': normalized.get('full_name', 'Unknown'),
                        'FullName': normalized.get('full_name', 'Unknown'),
                        'phone': normalized.get('phone', 'Not provided'),
                        'match_percentage': 0.0,
                        'Score': 0.0,
                        'grade': 'Suggested',
                        'Grade': 'Suggested',
                        'category': normalized.get('domain_tag', 'General'),
                        'domain_tag': normalized.get('domain_tag', 'General'),
                        'skills': technical_skills,
                        'Skills': technical_skills,
                        'experience_years': normalized.get('total_experience_years', 0),
                        'location': normalized.get('location', 'Unknown'),
                        'is_suggested': True,
                        'suggestion_reason': 'No exact matches found. Showing candidates with similar technical skills.'
                    }
                    filtered_results.append(suggested_result)
                except Exception as e:
                    logger.error(f"Error formatting suggested candidate: {e}")
                    continue
            
            summary = f"No exact matches found. Showing {len(filtered_results)} suggested candidates based on technical skills."
        
        if job_domain_lower and job_domain_lower != 'general':
            summary = f"{summary} (domain: {job_domain}, filtered to {len(filtered_results)})"
        return filtered_results, summary
   
    def _record_performance(self, response_time: float, query: str, cache_hit: bool = False, error: bool = False):
        """Record search performance metrics, supporting both old/new monitor APIs."""
        monitor = getattr(self, 'performance_monitor', None)
        if not monitor:
            return
        try:
            if hasattr(monitor, 'record_search'):
                monitor.record_search(response_time, query, cache_hit)
            elif hasattr(monitor, 'record_query'):
                monitor.record_query(query, response_time, cache_hit=cache_hit, error=error)
        except Exception as perf_err:
            logger.debug(f"Performance monitor recording failed: {perf_err}")
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate candidates based on email and full_name combination."""
        if not candidates:
            return candidates
        
        seen = set()
        deduplicated = []
        duplicates_removed = 0
        
        for candidate in candidates:
            # Get unique identifier: email + full_name combination
            email = (
                candidate.get('email') or 
                candidate.get('Email') or 
                candidate.get('email_address') or
                ''
            ).strip().lower()
            
            full_name = (
                candidate.get('full_name') or 
                candidate.get('FullName') or 
                candidate.get('name') or 
                candidate.get('Name') or
                ''
            ).strip().lower()
            
            # Create unique key from email and name
            # If email exists, use email as primary identifier
            # If no email, use name + phone as fallback
            if email and email not in ('unknown', 'n/a', ''):
                unique_key = email
            elif full_name and full_name not in ('unknown', ''):
                # Use name + phone as fallback identifier
                phone = (
                    candidate.get('phone') or 
                    candidate.get('Phone') or 
                    ''
                ).strip().lower()
                unique_key = f"{full_name}_{phone}" if phone else full_name
            else:
                # Last resort: use source_url or filename if available
                source_url = candidate.get('source_url') or candidate.get('sourceUrl') or candidate.get('sourceURL') or ''
                filename = candidate.get('filename') or ''
                unique_key = source_url or filename or f"unknown_{len(seen)}"
            
            # Check if we've seen this candidate before
            if unique_key not in seen:
                seen.add(unique_key)
                deduplicated.append(candidate)
            else:
                duplicates_removed += 1
                logger.debug(f"Removed duplicate candidate: {full_name or 'Unknown'} ({email or 'no email'})")
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate candidates. Remaining: {len(deduplicated)}")
        
        return deduplicated
   
    def _apply_domain_filter(self, candidates: List[Dict[str, Any]], job_domain: str) -> List[Dict[str, Any]]:
        """Ensure returned candidates align with the job domain (e.g., healthcare-only). Returns ONLY domain-matched candidates sorted by match score."""
        # Remove duplicates first
        candidates = self._deduplicate_candidates(candidates)
        
        job_domain_lower = (job_domain or '').strip().lower()
        if not candidates or not job_domain_lower or job_domain_lower == 'general':
            # If no domain filter, sort by match score and return top candidates (up to 20)
            try:
                candidates.sort(key=lambda x: x.get('match_percentage', x.get('Score', x.get('score', 0))), reverse=True)
            except Exception as e:
                logger.debug(f"Error sorting candidates: {e}")
            return (candidates or [])[:20]
       
        # Strict domain filtering - only return candidates that match the domain
        filtered = [
            candidate for candidate in candidates
            if self._candidate_matches_target_domain(candidate, job_domain_lower)
        ]
       
        # Remove duplicates again after domain filtering (in case domain filter allows duplicates)
        filtered = self._deduplicate_candidates(filtered)
       
        # Sort by match score to get best domain-matched candidates first (highest score first)
        try:
            filtered.sort(key=lambda x: x.get('match_percentage', x.get('Score', x.get('score', 0))), reverse=True)
        except Exception as e:
            logger.debug(f"Error sorting filtered candidates: {e}")
       
        # Log domain filtering results
        logger.info(f"Domain filter '{job_domain}': {len(filtered)} unique candidates matched out of {len(candidates)} total")
       
        # If we don't have enough domain-filtered candidates, log warning but return only domain matches
        if len(filtered) < 20:
            logger.warning(f"Only {len(filtered)} candidates match domain '{job_domain}' (need 20). Returning only domain-matched candidates.")
            if len(filtered) == 0:
                logger.error(f"CRITICAL: No candidates match domain '{job_domain}'. This may indicate insufficient domain-tagged candidates in database.")
       
        # Return ONLY domain-matched candidates (up to 20), sorted by match score (highest first)
        return filtered[:20]
   
    def _candidate_matches_target_domain(self, candidate: Dict[str, Any], target_domain: str) -> bool:
        """Check if a candidate belongs to the requested domain with enhanced accuracy."""
        if not candidate:
            return False
       
        # First check explicit domain tags (most reliable)
        normalized_domain = str(
            candidate.get('domain_tag') or
            candidate.get('category') or
            candidate.get('domain') or
            candidate.get('industry') or
            ''
        ).strip().lower()
       
        if normalized_domain:
            # Exact match or healthcare domain match
            if target_domain == 'healthcare':
                if normalized_domain in HEALTHCARE_DOMAIN_LABELS:
                    return True
                # Also check for healthcare-related terms in the domain string
                healthcare_terms = ['health', 'medical', 'hospital', 'clinical', 'nursing', 'pharmacy', 'physician']
                if any(term in normalized_domain for term in healthcare_terms):
                    return True
           
            # IT/Tech domain match
            if target_domain in IT_DOMAIN_LABELS:
                if normalized_domain in IT_DOMAIN_LABELS:
                    return True
                # Also check for tech-related terms
                tech_terms = ['tech', 'it', 'software', 'engineering', 'developer', 'programming', 'computer']
                if any(term in normalized_domain for term in tech_terms):
                    return True
       
        # If no explicit domain, infer from candidate content
        inferred_domain = self._infer_candidate_domain(candidate)
        if not inferred_domain:
            return False
       
        # Match inferred domain
        if target_domain == 'healthcare':
            if inferred_domain in HEALTHCARE_DOMAIN_LABELS:
                return True
            # Additional check: if inferred is healthcare-related
            if 'health' in inferred_domain or 'medical' in inferred_domain:
                return True
       
        if target_domain in IT_DOMAIN_LABELS:
            if inferred_domain in IT_DOMAIN_LABELS:
                return True
            # Additional check: if inferred is tech-related
            if 'tech' in inferred_domain or 'it' in inferred_domain:
                return True
       
        return inferred_domain == target_domain
   
    def _infer_candidate_domain(self, candidate: Dict[str, Any]) -> str:
        """Infer candidate domain based on their skills and text fields with enhanced accuracy."""
        keywords = self._extract_candidate_keywords(candidate)
        if not keywords:
            return ''
       
        # Use enhanced domain detection (this already uses healthcare_keywords and tech_keywords internally)
        inferred = self.detect_domain(keywords)
        inferred_lower = (inferred or '').strip().lower()
       
        # Additional validation: check if keywords strongly indicate a domain
        keywords_text = ' '.join(keywords).lower()
       
        # Define keyword sets for validation (same as in detect_domain)
        healthcare_terms = ['healthcare', 'medical', 'nursing', 'doctor', 'physician', 'hospital',
                           'clinic', 'patient', 'health', 'medicine', 'pharmacy', 'therapist',
                           'counselor', 'rn', 'lpn', 'emr', 'ehr', 'epic', 'cerner', 'clinical']
        tech_terms = ['software', 'programming', 'development', 'engineering', 'technology',
                     'computer', 'data', 'analyst', 'developer', 'engineer', 'python', 'java',
                     'javascript', 'react', 'aws', 'docker', 'kubernetes']
       
        # Count healthcare indicators
        healthcare_indicators = sum(1 for term in healthcare_terms if term in keywords_text)
        # Count tech indicators
        tech_indicators = sum(1 for term in tech_terms if term in keywords_text)
       
        # If strong indicators exist, use them (override if detection was weak)
        if healthcare_indicators >= 3 and healthcare_indicators > tech_indicators * 1.5:
            return 'healthcare'
        elif tech_indicators >= 3 and tech_indicators > healthcare_indicators * 1.5:
            return 'it/tech'
       
        return inferred_lower
   
    def _extract_candidate_keywords(self, candidate: Dict[str, Any]) -> List[str]:
        """Aggregate candidate text/skills into keywords for domain checks."""
        keywords: List[str] = []
       
        skills = candidate.get('skills') or candidate.get('Skills') or []
        if isinstance(skills, str):
            skills = [skills]
        if isinstance(skills, list):
            keywords.extend([str(skill).lower() for skill in skills if skill])
       
        text_fields = [
            candidate.get('resume_text') or candidate.get('resumeText'),
            candidate.get('summary'),
            candidate.get('experience'),
            candidate.get('Experience'),
            candidate.get('current_position'),
            candidate.get('title'),
            candidate.get('education'),
            candidate.get('Education')
        ]
        for text in text_fields:
            if isinstance(text, str):
                keywords.extend(re.findall(r'\b\w+\b', text.lower()))
       
        return keywords
   
    def _format_original_result_for_frontend(self, original_result: Dict[str, Any], job_description: str = "", job_required_skills: List[str] = None) -> Dict[str, Any]:
        """Format original algorithm result for frontend compatibility with improved domain detection"""
        try:
            # Extract basic information
            normalized = _normalize_candidate_profile(original_result)
           
            # Enhanced email resolution with comprehensive fallback
            email = normalized['email']
            if not email or email.lower() == 'unknown':
                email = (
                    original_result.get('email') or
                    original_result.get('Email') or
                    original_result.get('primary_email') or
                    original_result.get('primaryEmail') or
                    original_result.get('contact_email') or
                    original_result.get('contactEmail') or
                    'unknown'
                )
                if email and email.lower() == 'unknown':
                    # Check in contact bundles
                    contact_bundle = original_result.get('contact') or original_result.get('contactInfo') or {}
                    email = contact_bundle.get('email') or contact_bundle.get('Email') or 'unknown'
           
            # Enhanced name resolution with comprehensive fallback
            full_name = (
                normalized['full_name'] or
                original_result.get('full_name') or
                original_result.get('FullName') or
                original_result.get('fullName') or
                original_result.get('name') or
                original_result.get('Name') or
                original_result.get('candidate_name') or
                'Unknown'
            )
           
            # Enhanced phone resolution with comprehensive fallback
            phone = normalized['phone']
            if not phone or phone.lower() in ('unknown', 'not provided', 'n/a'):
                phone = (
                    original_result.get('phone') or
                    original_result.get('Phone') or
                    original_result.get('phone_number') or
                    original_result.get('phoneNumber') or
                    original_result.get('primary_phone') or
                    original_result.get('primaryPhone') or
                    original_result.get('mobile') or
                    original_result.get('Mobile') or
                    'Not provided'
                )
                if phone and phone.lower() in ('unknown', 'not provided', 'n/a'):
                    # Check in contact bundles
                    contact_bundle = original_result.get('contact') or original_result.get('contactInfo') or {}
                    phone = contact_bundle.get('phone') or contact_bundle.get('Phone') or contact_bundle.get('mobile') or 'Not provided'
           
            # Enhanced location resolution with comprehensive fallback
            location = (
                normalized['location'] or
                original_result.get('location') or
                original_result.get('Location') or
                original_result.get('current_location') or
                original_result.get('city') or
                original_result.get('City') or
                'Unknown'
            )
            source_url = normalized['source_url'] or original_result.get('sourceURL') or original_result.get('sourceUrl') or original_result.get('source_url', '')
            resume_text = normalized['resume_text'] or original_result.get('resume_text', '')
            skills = normalized['skills'] or original_result.get('skills', [])
            if isinstance(skills, str):
                skills = _normalize_to_list(skills)
            elif not isinstance(skills, list):
                skills = []
           
            # CRITICAL: Filter skills to TECHNICAL ONLY for display - remove soft skills
            skills = _filter_technical_skills_only(skills)
           
            # Order skills with required skills first
            if skills:
                try:
                    # Use provided job_required_skills or extract from job_description
                    if job_required_skills is None and job_description and hasattr(self, 'fallback_algorithm') and self.fallback_algorithm:
                        job_required_skills = self.fallback_algorithm._extract_skills_from_job(job_description)
                        # Filter job required skills to technical only
                        job_required_skills = _filter_technical_skills_only(job_required_skills)
                   
                    if job_required_skills:
                        skills = _order_skills_with_required_first(skills, job_required_skills)
                except Exception as e:
                    logger.debug(f"Error ordering skills: {e}")
           
            # Calculate match percentage using enhanced scoring
            match_percentage = 0
            if 'overall_score' in original_result:
                match_percentage = original_result.get('overall_score', 0)
            elif 'match_percentage' in original_result:
                match_percentage = original_result.get('match_percentage', 0)
            elif 'Score' in original_result:
                match_percentage = original_result.get('Score', 0)
           
            # Ensure it's a number and round to 1 decimal place
            if isinstance(match_percentage, str):
                try:
                    match_percentage = float(match_percentage.replace('%', ''))
                except:
                    match_percentage = 0
            elif not isinstance(match_percentage, (int, float)):
                match_percentage = 0
           
            # Enhance scoring with fuzzy matching and better algorithms
            if match_percentage == 0:
                # Use enhanced candidate scoring if no score available
                match_percentage = self._enhance_candidate_scoring(original_result, job_description)
            else:
                # Enhance existing score with additional factors
                enhanced_score = self._enhance_candidate_scoring(original_result, job_description)
                # Blend original and enhanced scores (70% original, 30% enhanced)
                match_percentage = (match_percentage * 0.7) + (enhanced_score * 0.3)
           
            # Round the percentage to 1 decimal place and cap at 80% maximum
            match_percentage = round(min(match_percentage, 80.0), 1)
           
            # Get experience years
            experience_years = normalized['total_experience_years']
            if not experience_years:
                experience_years = _parse_experience_years(original_result.get('experience_years'))
            if not experience_years:
                experience_years = _parse_experience_years(original_result.get('Experience'))
            experience_display = str(experience_years) if experience_years else str(original_result.get('Experience') or original_result.get('experience') or 0)
           
            # Improved category detection using the enhanced domain detection
            category = normalized['domain_tag'] or original_result.get('category', '') or original_result.get('domain', '')
            if not category:
                # Extract keywords from skills, education, and experience for category detection
                skills_text = ' '.join(skills) if skills else ''
                education_text = original_result.get('education', '')
                experience_text = original_result.get('experience', '')
                combined_text = f"{skills_text} {education_text} {experience_text}".lower()
               
                # Use the enhanced domain detection method
                keywords = combined_text.split()
                category = self.detect_domain(keywords)
                logger.info(f"Detected category for {full_name}: {category}")
           
            # Get grade
            grade = original_result.get('grade', self.get_grade(match_percentage))
           
            # Get match explanation
            match_explanation = original_result.get('match_explanation', f"Match score: {match_percentage}%")
           
            # Get detailed calculation breakdown if available
            calculation_details = original_result.get('calculation_details', {})
            if not calculation_details and match_percentage > 0:
                # Generate calculation details from enhanced scoring
                calculation_details = self._get_calculation_details(original_result, job_description)
           
            certifications = normalized['certifications'] or original_result.get('certifications', [])
            if isinstance(certifications, str):
                certifications = _normalize_to_list(certifications)
            designations = normalized['designations_with_experience'] or original_result.get('designations_with_experience') or original_result.get('designations') or []
            if isinstance(designations, str):
                designations = _normalize_to_list(designations)
            current_position = normalized['current_position'] or original_result.get('current_position') or original_result.get('title') or original_result.get('Title') or ''
            filename = normalized['filename'] or original_result.get('filename') or original_result.get('resume_filename', '')
            education = normalized['education'] or original_result.get('education', '')

            return {
                'email': email,
                'FullName': full_name,  # Frontend expects FullName
                'full_name': full_name,
                'phone': phone,
                'Score': match_percentage,  # Frontend expects Score (rounded)
                'Grade': grade,  # Frontend expects Grade
                'category': category,  # Frontend expects category
                'domain_tag': category,
                'domain': category,
                'skills': skills,  # Frontend expects skills array
                'Skills': skills,  # Also provide Skills for compatibility
                'Experience': experience_display,  # Frontend expects Experience as string
                'experience': experience_display,  # Also provide experience for compatibility
                'experience_years': experience_years,
                'total_experience_years': experience_years,
                'location': location,
                'sourceUrl': source_url,  # Frontend expects sourceUrl
                'sourceURL': source_url,  # Also provide sourceURL for compatibility
                'source_url': source_url,
                'resumeText': resume_text,  # Frontend expects resumeText
                'resume_text': resume_text,
                'match_explanation': match_explanation,  # Backend field name
                'MatchExplanation': match_explanation,  # Frontend expects MatchExplanation
                'seniority_level': original_result.get('seniority_level', 'Mid-level'),
                'Education': education,  # Frontend expects Education
                'education': education,
                'Certifications': certifications,  # Frontend expects Certifications
                'certifications': certifications,
                'previous_roles': original_result.get('previous_roles', []),
                'industries': original_result.get('industries', []),
                'designations_with_experience': designations,
                'designationsWithExperience': designations,
                'current_position': current_position,
                'title': current_position or original_result.get('title') or original_result.get('Title'),
                'filename': filename,
                # Add additional fields that frontend might expect
                'name': full_name,  # Alternative name field
                'contactInfo': {
                    'email': email,
                    'phone': phone
                },
                'Confidence': self._calculate_confidence_score(match_percentage, original_result),
                'source': 'AI Matching',
                # Add domain-specific fields for better badge display
                'domain': category,
                'domain_confidence': 0.8 if category != 'General' else 0.5,
                # Add detailed calculation breakdown
                'calculation_details': calculation_details
            }
        except Exception as e:
            logger.error(f"Error formatting original result: {e}")
            return self._create_fallback_result(full_name or "Unknown")
   
    def _create_fallback_result(self, name: str) -> Dict[str, Any]:
        """Create a fallback result when formatting fails"""
        return {
            'email': 'unknown',
            'FullName': name,  # Frontend expects FullName
            'name': name,  # Alternative name field
            'phone': 'Not provided',
            'Score': 0,  # Frontend expects Score
            'Grade': 'D',  # Frontend expects Grade
            'category': 'IT/Tech',
            'skills': [],  # Frontend expects skills array
            'Skills': [],  # Also provide Skills for compatibility
            'Experience': '0',  # Frontend expects Experience as string
            'experience': '0',  # Also provide experience for compatibility
            'location': 'Unknown',
            'sourceUrl': '',  # Frontend expects sourceUrl
            'sourceURL': '',  # Also provide sourceURL for compatibility
            'resumeText': '',  # Frontend expects resumeText
            'match_explanation': 'Error processing result',
            'MatchExplanation': 'Error processing result',
            'seniority_level': 'Unknown',
            'Education': '',  # Frontend expects Education
            'Certifications': [],  # Frontend expects Certifications
            'previous_roles': [],
            'industries': [],
            'contactInfo': {
                'email': 'unknown',
                'phone': 'Not provided'
            },
            'Confidence': 50.0,  # Default confidence score for fallback results
            'source': 'AI Matching'
        }
   
    def get_enhanced_performance_stats(self):
        """Get comprehensive performance statistics"""
        stats = {
            'total_searches': self.performance_stats['total_searches'],
            'avg_response_time': round(self.performance_stats['avg_response_time'], 2),
            'original_algorithm_usage': self.performance_stats['original_algorithm_used'],
            'fallback_usage': self.performance_stats['fallback_used'],
            'original_algorithm_available': ORIGINAL_ALGORITHM_AVAILABLE
        }
       
        # Add performance monitor stats if available
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            try:
                monitor_stats = self.performance_monitor.get_current_status()
                stats.update({
                    'performance_monitor_stats': monitor_stats,
                    'cache_hit_rate': self.performance_monitor.get_cache_hit_rate(),
                    'total_indexing_operations': self.performance_monitor.get_total_indexing_operations()
                })
            except Exception as e:
                logger.warning(f"Could not get performance monitor stats: {e}")
       
        # Add cache stats if available
        if hasattr(self, 'embedding_cache') and self.embedding_cache:
            try:
                cache_stats = self.embedding_cache.get_stats()
                stats['embedding_cache_stats'] = cache_stats
            except Exception as e:
                logger.warning(f"Could not get cache stats: {e}")
       
        # Add batch processor stats if available
        if hasattr(self, 'batch_processor') and self.batch_processor:
            try:
                batch_stats = self.batch_processor.get_stats()
                stats['batch_processor_stats'] = batch_stats
            except Exception as e:
                logger.warning(f"Could not get batch processor stats: {e}")
        return stats
   
    def semantic_match(self, job_description, use_gpt4_reranking=True):
        """Optimized semantic matching with enhanced performance and accuracy"""
        # Validate query - reject meaningless queries early
        if _is_meaningless_query(job_description):
            logger.warning(f"Meaningless query rejected in AdeptAIMastersAlgorithm.semantic_match: '{job_description[:100] if job_description else ''}...'")
            return {
                'results': [],
                'summary': 'No candidates found. Please provide a valid job description with meaningful keywords.',
                'total_candidates': 0,
                'error': False,
                'query_rejected': True
            }
        
        start_time = time.time()
        logger.info(f"Starting optimized semantic match for: {job_description[:100]}...")
       
        try:
            # Use the original algorithm's semantic matching if available
            if self.enhanced_system:
                logger.info("Using enhanced algorithm for semantic matching")
               
                # Apply smart filtering to reduce computation
                if ENABLE_SMART_FILTERING and hasattr(self.fallback_algorithm, 'candidates'):
                    # Pre-filter to many more candidates to ensure we have enough after domain filtering
                    # Domain filtering can remove 80-90% of candidates, so we need a large pool
                    job_keywords = re.findall(r'\b\w+\b', job_description.lower())
                    job_domain = self.detect_domain(job_keywords)
                    job_domain_lower = (job_domain or '').strip().lower()
                   
                    if job_domain_lower and job_domain_lower != 'general':
                        prefilter_top_k = 500  # Pre-filter to 500 for domain filtering
                        logger.info(f"Domain filtering active: Pre-filtering to {prefilter_top_k} candidates")
                    else:
                        prefilter_top_k = 100  # Normal pre-filtering
                   
                    filtered_candidates = _smart_filter_candidates(
                        self.fallback_algorithm.candidates,
                        job_description,
                        top_k=prefilter_top_k
                    )
                    logger.info(f"Smart filtering reduced candidates to {len(filtered_candidates)}")
               
                # Use optimized keyword search - ensure we get 20 candidates
                results, summary = self.keyword_search(job_description, top_k=20)
                
                # Remove duplicates before post-processing
                results = self._deduplicate_candidates(results)
                
                # Post-process results for better accuracy
                results = self._post_process_results(results, job_description)
               
                # Sort by match score (highest first) before limiting
                try:
                    results.sort(key=lambda x: x.get('match_percentage', x.get('Score', x.get('score', 0))), reverse=True)
                except Exception as e:
                    logger.debug(f"Error sorting results in semantic_match: {e}")
               
                # Ensure we return exactly 20 candidates (or as many as available)
                results = results[:20] if results else []
               
                # Ensure accurate candidate counting
                total_candidates = len(results)
                elapsed_time = time.time() - start_time
               
                logger.info(f"Enhanced algorithm completed in {elapsed_time:.2f}s, found {total_candidates} candidates")
               
                return {
                    'results': results,
                    'summary': f"{summary} (Enhanced algorithm, {elapsed_time:.2f}s)",
                    'total_candidates': total_candidates,
                    'algorithm_used': 'Enhanced AdeptAI Algorithm',
                    'processing_time': elapsed_time
                }
            else:
                logger.info("Using optimized fallback algorithm for semantic matching")
                fallback_results = self.fallback_algorithm.semantic_match(job_description)
               
                # Post-process fallback results
                if fallback_results.get('results'):
                    # Remove duplicates before post-processing
                    fallback_results['results'] = self._deduplicate_candidates(fallback_results['results'])
                    fallback_results['results'] = self._post_process_results(fallback_results['results'], job_description)
                    # Sort by match score (highest first) before limiting
                    try:
                        fallback_results['results'].sort(key=lambda x: x.get('match_percentage', x.get('Score', x.get('score', 0))), reverse=True)
                    except Exception as e:
                        logger.debug(f"Error sorting fallback results: {e}")
                    # Ensure we return exactly 20 candidates (or as many as available)
                    fallback_results['results'] = fallback_results['results'][:20] if fallback_results['results'] else []
               
                # Ensure accurate candidate counting for fallback
                total_candidates = len(fallback_results['results']) if fallback_results['results'] else 0
                elapsed_time = time.time() - start_time
               
                logger.info(f"Optimized fallback algorithm completed in {elapsed_time:.2f}s, found {total_candidates} candidates")
               
                return {
                    'results': fallback_results['results'],
                    'summary': f"{fallback_results['summary']} (Optimized fallback, {elapsed_time:.2f}s)",
                    'total_candidates': total_candidates,
                    'algorithm_used': 'Optimized Fallback Algorithm',
                    'processing_time': elapsed_time
                }
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
            elapsed_time = time.time() - start_time
            return {
                'results': [],
                'summary': f"Error in semantic matching: {str(e)} ({elapsed_time:.2f}s)",
                'total_candidates': 0,
                'algorithm_used': 'Error',
                'processing_time': elapsed_time
            }
   
    def _post_process_results(self, results, job_description):
        """Post-process results for better accuracy and consistency, sorted by match score (highest first)"""
        try:
            if not results:
                return results
            
            # Remove duplicates first
            results = self._deduplicate_candidates(results)
           
            # Sort by match percentage/score for consistency (highest first)
            # Try multiple score field names to ensure we sort correctly
            def get_score(candidate):
                return candidate.get('match_percentage') or candidate.get('Score') or candidate.get('score') or candidate.get('MatchScore') or 0
           
            results.sort(key=get_score, reverse=True)
           
            # Add domain and confidence information
            for result in results:
                if 'category' not in result:
                    result['category'] = self._detect_category_from_result(result)
               
                if 'domain' not in result:
                    result['domain'] = result.get('category', 'IT/Tech')
               
                if 'domain_confidence' not in result:
                    result['domain_confidence'] = self._calculate_domain_confidence(result, job_description)
               
                # Ensure consistent field names
                if 'Score' not in result and 'match_percentage' in result:
                    result['Score'] = result['match_percentage']
               
                if 'Grade' not in result and 'grade' in result:
                    result['Grade'] = result['grade']
           
            return results
           
        except Exception as e:
            logger.error(f"Error in post-processing results: {e}")
            return results
   
    def _detect_category_from_result(self, result):
        """Detect category from result data"""
        try:
            skills = result.get('skills', []) or result.get('top_skills', [])
            experience = result.get('experience', '') or result.get('Experience', '')
            education = result.get('education', '')
           
            combined_text = f"{' '.join(skills)} {experience} {education}".lower()
           
            # Healthcare keywords
            healthcare_keywords = ['healthcare', 'medical', 'nursing', 'doctor', 'physician', 'hospital', 'clinic', 'patient', 'health', 'medicine', 'pharmacy', 'therapist', 'counselor']
            if any(keyword in combined_text for keyword in healthcare_keywords):
                return 'Healthcare'
           
            # Finance keywords
            finance_keywords = ['finance', 'banking', 'accounting', 'financial', 'investment', 'trading', 'audit', 'tax', 'budget', 'revenue', 'profit']
            if any(keyword in combined_text for keyword in finance_keywords):
                return 'Finance'
           
            # Education keywords
            education_keywords = ['education', 'teaching', 'teacher', 'professor', 'instructor', 'academic', 'university', 'college', 'school', 'student', 'learning']
            if any(keyword in combined_text for keyword in education_keywords):
                return 'Education'
           
            return 'IT/Tech'  # Default category
           
        except Exception as e:
            logger.error(f"Error detecting category from result: {e}")
            return 'IT/Tech'
   
    def _calculate_domain_confidence(self, result, job_description):
        """Calculate confidence in domain classification"""
        try:
            category = result.get('category', 'IT/Tech')
            skills = result.get('skills', []) or result.get('top_skills', [])
           
            # Count matching keywords for the detected category
            if category == 'Healthcare':
                keywords = ['healthcare', 'medical', 'nursing', 'doctor', 'physician', 'hospital', 'clinic', 'patient', 'health', 'medicine', 'pharmacy', 'therapist', 'counselor']
            elif category == 'Finance':
                keywords = ['finance', 'banking', 'accounting', 'financial', 'investment', 'trading', 'audit', 'tax', 'budget', 'revenue', 'profit']
            elif category == 'Education':
                keywords = ['education', 'teaching', 'teacher', 'professor', 'instructor', 'academic', 'university', 'college', 'school', 'student', 'learning']
            else:
                keywords = ['software', 'programming', 'development', 'engineering', 'technology', 'computer', 'data', 'analyst', 'developer', 'engineer']
           
            combined_text = f"{' '.join(skills)} {result.get('experience', '')} {result.get('education', '')}".lower()
            matches = sum(1 for keyword in keywords if keyword in combined_text)
           
            # Calculate confidence based on number of matches
            confidence = min(matches / len(keywords), 1.0) if keywords else 0.5
            return round(confidence * 100, 1)  # Return as percentage
           
        except Exception as e:
            logger.error(f"Error calculating domain confidence: {e}")
            return 50.0  # Default confidence
   
    def _calculate_confidence_score(self, match_percentage: float, candidate_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on match quality and data completeness"""
        try:
            # Base confidence from match percentage (0-80% maps to 0-100 confidence)
            # Higher match percentage = higher confidence
            match_confidence = (match_percentage / 80.0) * 100.0 if match_percentage > 0 else 0.0
           
            # Data completeness factor (0-20 points)
            completeness_score = 0.0
            required_fields = ['skills', 'experience', 'education', 'name']
            optional_fields = ['phone', 'email', 'location', 'certifications']
           
            # Check required fields
            for field in required_fields:
                value = candidate_data.get(field) or candidate_data.get(field.capitalize()) or candidate_data.get(field.title())
                if value and ((isinstance(value, str) and value.strip()) or (isinstance(value, list) and len(value) > 0)):
                    completeness_score += 3.0  # 3 points per required field
           
            # Check optional fields
            for field in optional_fields:
                value = candidate_data.get(field) or candidate_data.get(field.capitalize()) or candidate_data.get(field.title())
                if value and ((isinstance(value, str) and value.strip()) or (isinstance(value, list) and len(value) > 0)):
                    completeness_score += 2.0  # 2 points per optional field
           
            # Cap completeness at 20 points
            completeness_score = min(completeness_score, 20.0)
           
            # Calculate final confidence: 80% from match quality, 20% from data completeness
            final_confidence = (match_confidence * 0.8) + completeness_score
           
            # Ensure confidence is between 0 and 100
            final_confidence = max(0.0, min(100.0, final_confidence))
           
            return round(final_confidence, 1)
           
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            # Fallback: use match percentage as base confidence
            return round(min((match_percentage / 80.0) * 100.0, 100.0), 1) if match_percentage > 0 else 50.0
   
    def _get_calculation_details(self, candidate: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """Get detailed calculation breakdown for match score and confidence"""
        try:
            # Recalculate scores to get breakdown
            candidate_skills = []
            skill_sources = [
                candidate.get('skills'),
                candidate.get('Skills'),
                candidate.get('technical_skills'),
            ]
            for source in skill_sources:
                if source:
                    if isinstance(source, list):
                        candidate_skills.extend([str(s).lower().strip() for s in source])
                    elif isinstance(source, str):
                        candidate_skills.extend([s.strip().lower() for s in re.split(r'[,;|•\n]', source)])
           
            job_skills = self._extract_skills_from_description(job_description)
            candidate_text = (
                candidate.get('resumeText', '') or
                candidate.get('experience', '') or
                candidate.get('summary', '') or ''
            )
           
            base_similarity = self.semantic_similarity(candidate_text, job_description)
            skill_match_score = self._fuzzy_match_skills(candidate_skills, job_skills, threshold=0.65)
            experience_match = self._calculate_experience_match(candidate, job_description)
            experience_relevance = self._calculate_experience_relevance(candidate, job_description)
            domain_alignment = self._calculate_domain_alignment(candidate, job_description)
            keyword_density = self._calculate_keyword_density(candidate, job_description)
            title_match = self._calculate_title_match(candidate, job_description)
            education_match = self._calculate_education_bonus(candidate, job_description)
            certification_match = self._calculate_certification_match(candidate, job_description)
            location_match = self._calculate_location_match(candidate, job_description)
           
            # Calculate weighted contributions
            match_breakdown = {
                'skills_match': {
                    'score': round(skill_match_score * 100, 1),
                    'weight': 32,
                    'contribution': round(skill_match_score * 0.32 * 100, 1),
                    'description': 'Skills alignment with job requirements'
                },
                'semantic_similarity': {
                    'score': round(base_similarity * 100, 1),
                    'weight': 22,
                    'contribution': round(base_similarity * 0.22 * 100, 1),
                    'description': 'Overall profile similarity to job description'
                },
                'experience_match': {
                    'score': round(experience_match * 100, 1),
                    'weight': 15,
                    'contribution': round(experience_match * 0.15 * 100, 1),
                    'description': 'Years of experience alignment'
                },
                'experience_relevance': {
                    'score': round(experience_relevance * 100, 1),
                    'weight': 10,
                    'contribution': round(experience_relevance * 0.10 * 100, 1),
                    'description': 'Relevance of past experience to role'
                },
                'domain_alignment': {
                    'score': round(domain_alignment * 100, 1),
                    'weight': 8,
                    'contribution': round(domain_alignment * 0.08 * 100, 1),
                    'description': 'Industry/domain match (Healthcare, IT, etc.)'
                },
                'keyword_density': {
                    'score': round(keyword_density * 100, 1),
                    'weight': 6,
                    'contribution': round(keyword_density * 0.06 * 100, 1),
                    'description': 'Important keywords from job description'
                },
                'title_match': {
                    'score': round(title_match * 100, 1),
                    'weight': 4,
                    'contribution': round(title_match * 0.04 * 100, 1),
                    'description': 'Job title/role alignment'
                },
                'education': {
                    'score': round(education_match * 100, 1),
                    'weight': 2,
                    'contribution': round(education_match * 0.02 * 100, 1),
                    'description': 'Education level match'
                },
                'certifications': {
                    'score': round(certification_match * 100, 1),
                    'weight': 1,
                    'contribution': round(certification_match * 0.01 * 100, 1),
                    'description': 'Certifications and licenses'
                },
                'location': {
                    'score': round(location_match * 100, 1),
                    'weight': 0,
                    'contribution': round(location_match * 0.00 * 100, 1),
                    'description': 'Location preference (flexible)'
                }
            }
           
            # Calculate confidence breakdown
            match_percentage = sum(item['contribution'] for item in match_breakdown.values())
            match_confidence = (match_percentage / 80.0) * 100.0 if match_percentage > 0 else 0.0
           
            # Data completeness
            completeness_score = 0.0
            required_fields = ['skills', 'experience', 'education', 'name']
            optional_fields = ['phone', 'email', 'location', 'certifications']
           
            completeness_breakdown = {}
            for field in required_fields:
                value = candidate.get(field) or candidate.get(field.capitalize()) or candidate.get(field.title())
                has_field = value and ((isinstance(value, str) and value.strip()) or (isinstance(value, list) and len(value) > 0))
                completeness_breakdown[field] = {
                    'present': has_field,
                    'points': 3.0 if has_field else 0.0,
                    'required': True
                }
                if has_field:
                    completeness_score += 3.0
           
            for field in optional_fields:
                value = candidate.get(field) or candidate.get(field.capitalize()) or candidate.get(field.title())
                has_field = value and ((isinstance(value, str) and value.strip()) or (isinstance(value, list) and len(value) > 0))
                completeness_breakdown[field] = {
                    'present': has_field,
                    'points': 2.0 if has_field else 0.0,
                    'required': False
                }
                if has_field:
                    completeness_score += 2.0
           
            completeness_score = min(completeness_score, 20.0)
            final_confidence = (match_confidence * 0.8) + completeness_score
           
            return {
                'match_score_breakdown': match_breakdown,
                'confidence_breakdown': {
                    'match_quality': {
                        'score': round(match_confidence, 1),
                        'weight': 80,
                        'contribution': round(match_confidence * 0.8, 1),
                        'description': 'Based on match score quality'
                    },
                    'data_completeness': {
                        'score': round(completeness_score, 1),
                        'weight': 20,
                        'contribution': round(completeness_score, 1),
                        'description': 'Profile data completeness',
                        'field_details': completeness_breakdown
                    }
                },
                'total_match_score': round(match_percentage, 1),
                'total_confidence': round(final_confidence, 1)
            }
        except Exception as e:
            logger.error(f"Error getting calculation details: {e}")
            return {}

# Global instances
_algorithm_instance = None

def get_algorithm_instance():
    """Get or create the algorithm instance with error handling"""
    global _algorithm_instance
    try:
        if _algorithm_instance is None:
            logger.info("Initializing new algorithm instance...")
            _algorithm_instance = AdeptAIMastersAlgorithm()
            logger.info("Algorithm instance initialized successfully")
        return _algorithm_instance
    except Exception as e:
        logger.error(f"Failed to initialize algorithm instance: {e}", exc_info=True)
        # Return a minimal fallback instance
        if _algorithm_instance is None:
            logger.warning("Creating minimal fallback algorithm instance")
            _algorithm_instance = FallbackAlgorithm()
    return _algorithm_instance

def semantic_match(job_description, top_k=20):
    """Semantic matching function with error handling and accurate counting"""
    try:
        # Validate query - reject meaningless queries early
        if _is_meaningless_query(job_description):
            logger.warning(f"Meaningless query rejected: '{job_description[:100]}...'")
            return {
                'results': [],
                'summary': 'No candidates found. Please provide a valid job description with meaningful keywords.',
                'total_candidates': 0,
                'error': False,
                'query_rejected': True
            }
        
        logger.info(f"[DEBUG] ========== SEMANTIC MATCH STARTED ==========")
        logger.info(f"[DEBUG] Job description: {job_description[:200]}...")
        logger.info(f"[DEBUG] Requesting top_k={top_k} candidates")
       
        # Extract and log job requirements for debugging
        job_keywords = re.findall(r'\b\w+\b', job_description.lower())
        logger.info(f"[DEBUG] Job keywords extracted: {job_keywords[:20]}...")
       
        # Try optimized search system first (if available)
        try:
            from app.search.optimized_search_service import get_optimized_search_service, search_optimized
           
            # Check if optimized service is available and ready
            service = get_optimized_search_service()
            if service and service.is_ready():
                logger.info("[DEBUG] Using optimized parallel search system for semantic match")
                logger.warning("[DEBUG] NOTE: Strict pre-filtering is in FallbackAlgorithm.keyword_search - this path may not use it!")
                result = search_optimized(job_description, top_k=top_k, use_ultra_fast=True)
               
                # Apply accuracy enhancement
                if result.get('results'):
                    try:
                        from app.search.accuracy_enhancement_system import enhance_search_accuracy
                        logger.info("Applying accuracy enhancement to search results")
                        enhanced_results = enhance_search_accuracy(job_description, result['results'], top_k=top_k)
                        result['results'] = enhanced_results
                        result['accuracy_enhanced'] = True
                        logger.info(f"Accuracy enhancement applied to {len(enhanced_results)} results")
                    except Exception as e:
                        logger.warning(f"Accuracy enhancement failed: {e}, using original results")
               
                # Ensure total_candidates is always present and accurate
                if 'total_candidates' not in result:
                    result['total_candidates'] = len(result.get('results', []))
               
                # Log the accurate count
                logger.info(f"Optimized semantic match completed: {result.get('summary', 'No summary')}")
                logger.info(f"Total candidates found: {result.get('total_candidates', 0)}")
               
                return result
        except ImportError:
            logger.info("Optimized search system not available, falling back to scalable system")
        except Exception as e:
            logger.warning(f"Optimized search system error: {e}, falling back to scalable system")
       
        # Try scalable search system as fallback (if available)
        try:
            from app.search.integration_guide import get_scalable_integration
            scalable_integration = get_scalable_integration()
           
            if scalable_integration.is_initialized:
                logger.info("Using scalable search system for semantic match")
                result = scalable_integration.search_candidates(
                    job_description=job_description,
                    top_k=top_k,
                    search_type='hybrid'
                )
               
                # Ensure total_candidates is always present and accurate
                if 'total_candidates' not in result:
                    result['total_candidates'] = len(result.get('results', []))
               
                # Log the accurate count
                logger.info(f"Scalable semantic match completed: {result.get('summary', 'No summary')}")
                logger.info(f"Total candidates found: {result.get('total_candidates', 0)}")
               
                return result
        except ImportError:
            logger.info("Scalable search system not available, falling back to original algorithm")
        except Exception as e:
            logger.warning(f"Scalable search system error: {e}, falling back to original algorithm")
       
        # Fallback to original algorithm
        try:
            algorithm = get_algorithm_instance()
            logger.info("[DEBUG] Algorithm instance obtained successfully")
            logger.info(f"[DEBUG] Algorithm type: {type(algorithm).__name__}")
           
            # Check if it's FallbackAlgorithm (which has our strict filtering)
            if isinstance(algorithm, FallbackAlgorithm):
                logger.info("[DEBUG] Using FallbackAlgorithm - strict pre-filtering WILL BE APPLIED")
            else:
                logger.warning(f"[DEBUG] Using {type(algorithm).__name__} - strict pre-filtering may NOT be applied")
               
        except Exception as e:
            logger.error(f"Failed to get algorithm instance: {e}")
            # Return fallback results instead of failing
            return {
                'results': [],
                'summary': f"Search temporarily unavailable. Please try again later. Error: {str(e)}",
                'total_candidates': 0,
                'error': True,
                'fallback': True
            }
       
        # Try to perform the semantic match
        try:
            logger.info("[DEBUG] Calling algorithm.semantic_match()...")
            result = algorithm.semantic_match(job_description)
            logger.info(f"[DEBUG] Algorithm returned {len(result.get('results', []))} results")
           
            # Apply accuracy enhancement to fallback results
            if result.get('results'):
                try:
                    from app.search.accuracy_enhancement_system import enhance_search_accuracy
                    logger.info("Applying accuracy enhancement to fallback results")
                    enhanced_results = enhance_search_accuracy(job_description, result['results'], top_k=top_k)
                    result['results'] = enhanced_results
                    result['accuracy_enhanced'] = True
                    logger.info(f"Accuracy enhancement applied to {len(enhanced_results)} fallback results")
                except Exception as e:
                    logger.warning(f"Accuracy enhancement failed for fallback: {e}, using original results")
           
            # Ensure total_candidates is always present and accurate
            if 'total_candidates' not in result:
                result['total_candidates'] = len(result.get('results', []))
           
            # Log the accurate count
            logger.info(f"Semantic match completed successfully: {result.get('summary', 'No summary')}")
            logger.info(f"Total candidates found: {result.get('total_candidates', 0)}")
           
            return result
        except Exception as e:
            logger.error(f"Semantic match failed: {e}", exc_info=True)
            # Return fallback results instead of failing
            return {
                'results': [],
                'summary': f"Search failed. Please try again later. Error: {str(e)}",
                'total_candidates': 0,
                'error': True,
                'fallback': True
            }
           
    except Exception as e:
        logger.error(f"Unexpected error in semantic_match: {e}", exc_info=True)
        # Return fallback results instead of failing
        return {
            'results': [],
            'summary': f"Search temporarily unavailable. Please try again later. Error: {str(e)}",
            'total_candidates': 0,
            'error': True,
            'fallback': True
        }

def keyword_search(job_description, top_k=20):
    """Keyword search function"""
    # Validate query - reject meaningless queries early
    if _is_meaningless_query(job_description):
        logger.warning(f"Meaningless query rejected in keyword_search wrapper: '{job_description[:100] if job_description else ''}...'")
        return [], "No candidates found. Please provide a valid job description with meaningful keywords."
    
    algorithm = get_algorithm_instance()
    return algorithm.keyword_search(job_description, top_k)

def register_feedback(candidate_id, positive=True):
    """Register feedback for a candidate"""
    try:
        if feedback_table:
            feedback_table.put_item(Item={
                'candidate_id': candidate_id,
                'feedback': 'positive' if positive else 'negative',
                'timestamp': datetime.utcnow().isoformat()
            })
            logger.info(f"Feedback registered for candidate {candidate_id}")
        else:
            logger.warning("Feedback table not available")
    except Exception as e:

        logger.error(f"Error registering feedback: {e}")