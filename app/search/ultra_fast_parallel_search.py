"""
Ultra-Fast Parallel Search System
Implements parallel processing, advanced caching, and optimized algorithms for lightning-fast search results.
"""

import asyncio
import concurrent.futures
import time
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import redis
import json
import hashlib

logger = logging.getLogger(__name__)

HEALTHCARE_KEYWORDS = {
    'healthcare', 'medical', 'hospital', 'clinic', 'patient', 'nursing', 'nurse',
    'registered nurse', 'licensed practical nurse', 'lpn', 'rn', 'cna', 'care plan',
    'med-surg', 'med surge', 'icu', 'ccu', 'er', 'emergency', 'clinical', 'charting',
    'medication administration', 'phlebotomy', 'therapy', 'therapist', 'rehabilitation',
    'bcls', 'acls', 'cpr', 'vitals', 'home health', 'long-term care', 'ltc', 'case management',
    'patient care', 'health informatics', 'epic', 'cerner', 'meditech'
}

TECH_KEYWORDS = {
    'software', 'developer', 'development', 'engineering', 'engineer', 'programming',
    'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'node', 'devops',
    'aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 'ci/cd', 'api', 'database',
    'frontend', 'backend', 'fullstack', 'sql', 'data science', 'machine learning'
}


def _normalize_domain_label(domain: str) -> str:
    domain_lower = (domain or '').strip().lower()
    if domain_lower in {'healthcare', 'medical', 'clinical', 'nursing'}:
        return 'Healthcare'
    if domain_lower in {'it/tech', 'technology', 'tech', 'software', 'it'}:
        return 'IT/Tech'
    if domain_lower:
        return domain.strip()
    return 'General'


def detect_domain_from_text(text: str) -> str:
    if not text:
        return 'general'
    text_lower = text.lower()
    healthcare_count = sum(1 for keyword in HEALTHCARE_KEYWORDS if keyword in text_lower)
    tech_count = sum(1 for keyword in TECH_KEYWORDS if keyword in text_lower)
    if healthcare_count > tech_count and healthcare_count > 0:
        return 'healthcare'
    if tech_count > healthcare_count and tech_count > 0:
        return 'it/tech'
    return 'general'


def detect_candidate_domain(candidate: Dict[str, Any]) -> str:
    existing = (
        candidate.get('domain_tag') or
        candidate.get('category') or
        candidate.get('domain')
    )
    if existing:
        return detect_domain_from_text(str(existing))

    parts: List[str] = []
    title = candidate.get('title')
    if title:
        parts.append(str(title))
    experience = candidate.get('experience')
    if experience:
        parts.append(str(experience))
    education = candidate.get('education')
    if education:
        parts.append(str(education))
    skills = candidate.get('skills')
    if isinstance(skills, list):
        parts.extend(str(skill) for skill in skills if skill is not None)
    elif isinstance(skills, str):
        parts.append(skills)

    combined_text = ' '.join(parts)
    return detect_domain_from_text(combined_text)


def resolve_candidate_email(candidate: Dict[str, Any]) -> Optional[str]:
    email_fields = [
        'email', 'Email',
        'email_address', 'EmailAddress',
        'primary_email', 'primaryEmail', 'primary_email_address', 'primaryEmailAddress',
        'contact_email', 'contactEmail',
        'work_email', 'workEmail',
        'personal_email', 'personalEmail',
        'business_email', 'businessEmail'
    ]

    for field in email_fields:
        value = candidate.get(field)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped.lower() != 'unknown':
                return stripped

    contact_dicts = []
    for key in ('contact', 'Contact', 'contactInfo', 'contact_info', 'contactDetails', 'contact_details'):
        contact_value = candidate.get(key)
        if isinstance(contact_value, dict):
            contact_dicts.append(contact_value)

    for contact in contact_dicts:
        for field in email_fields + ['emailAddress', 'EmailAddress']:
            value = contact.get(field)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped and stripped.lower() != 'unknown':
                    return stripped
        for key in ('emails', 'Emails', 'emailAddresses', 'EmailAddresses', 'email_addresses'):
            emails_value = contact.get(key)
            if isinstance(emails_value, list):
                for email_entry in emails_value:
                    if isinstance(email_entry, str):
                        stripped = email_entry.strip()
                        if stripped and stripped.lower() != 'unknown':
                            return stripped

    for key in ('emails', 'Emails', 'emailAddresses', 'EmailAddresses', 'email_addresses'):
        emails_value = candidate.get(key)
        if isinstance(emails_value, list):
            for email_entry in emails_value:
                if isinstance(email_entry, str):
                    stripped = email_entry.strip()
                    if stripped and stripped.lower() != 'unknown':
                        return stripped

    return None

@dataclass
class SearchResult:
    candidate_id: str
    score: float
    match_reasons: List[str]
    candidate_data: Dict[str, Any]
    processing_time: float = 0.0

class ParallelEmbeddingProcessor:
    """Handles parallel embedding generation for maximum speed"""
    
    def __init__(self, embedding_service, max_workers: int = None):
        self.embedding_service = embedding_service
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.embedding_cache = {}
        self.cache_lock = threading.Lock()
        
    def encode_batch_parallel(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings in parallel batches for maximum speed"""
        start_time = time.time()
        
        # Check cache first
        cache_key = hashlib.md5('|'.join(texts).encode()).hexdigest()
        with self.cache_lock:
            if cache_key in self.embedding_cache:
                logger.info(f"Embedding cache hit for batch of {len(texts)} texts")
                return self.embedding_cache[cache_key]
        
        # Process in parallel batches
        embeddings = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Split texts into batches
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            # Submit all batches for parallel processing
            future_to_batch = {
                executor.submit(self._process_batch, batch): batch 
                for batch in batches
            }
            
            # Collect results in order
            batch_results = []
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_embeddings = future.result()
                    batch_results.append((future_to_batch[future], batch_embeddings))
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    batch_results.append((future_to_batch[future], np.array([])))
            
            # Sort by original order and concatenate
            batch_results.sort(key=lambda x: texts.index(x[0][0]) if x[0] else 0)
            for _, batch_embeddings in batch_results:
                if len(batch_embeddings) > 0:
                    embeddings.extend(batch_embeddings)
        
        if not embeddings:
            logger.warning("No embeddings generated, falling back to single processing")
            embeddings = [self.embedding_service.encode_single(text) for text in texts]
        
        result = np.array(embeddings)
        
        # Cache the result
        with self.cache_lock:
            self.embedding_cache[cache_key] = result
            # Limit cache size
            if len(self.embedding_cache) > 1000:
                # Remove oldest 20% of entries
                keys_to_remove = list(self.embedding_cache.keys())[:200]
                for key in keys_to_remove:
                    del self.embedding_cache[key]
        
        processing_time = time.time() - start_time
        logger.info(f"Parallel embedding generation completed: {len(texts)} texts in {processing_time:.2f}s")
        
        return result
    
    def _process_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        """Process a single batch of texts"""
        try:
            if len(batch_texts) == 1:
                return [self.embedding_service.encode_single(batch_texts[0])]
            else:
                return self.embedding_service.encode_batch(batch_texts)
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Fallback to individual processing
            return [self.embedding_service.encode_single(text) for text in batch_texts]

class UltraFastSearchEngine:
    """Ultra-fast search engine with parallel processing and advanced optimizations"""
    
    def __init__(self, embedding_service, redis_client=None):
        self.embedding_service = embedding_service
        self.redis_client = redis_client
        self.parallel_processor = ParallelEmbeddingProcessor(embedding_service)
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0.0,
            'parallel_processing_time': 0.0,
            'faiss_search_time': 0.0
        }
        
        # FAISS index and candidate data
        self.faiss_index = None
        self.candidate_data = {}
        self.candidate_embeddings = None
        self.candidate_ids = []
        
        # Threading locks
        self.index_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Search result cache
        self.search_cache = {}
        self.cache_max_size = 1000
        
    def initialize_with_candidates(self, candidates: Dict[str, Any]):
        """Initialize the search engine with candidate data"""
        start_time = time.time()
        logger.info(f"Initializing ultra-fast search engine with {len(candidates)} candidates")
        
        try:
            # Prepare candidate data
            self.candidate_data = candidates
            self.candidate_ids = list(candidates.keys())
            
            # Generate search texts for all candidates
            search_texts = []
            for candidate_id, candidate in candidates.items():
                search_text = self._generate_search_text(candidate)
                search_texts.append(search_text)
            
            # Generate embeddings in parallel
            logger.info("Generating embeddings in parallel...")
            embeddings = self.parallel_processor.encode_batch_parallel(search_texts)
            
            # Build FAISS index
            logger.info("Building FAISS index...")
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings.astype('float32'))
            self.candidate_embeddings = embeddings
            
            init_time = time.time() - start_time
            logger.info(f"Ultra-fast search engine initialized in {init_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            return False
    
    def _generate_search_text(self, candidate: Dict[str, Any]) -> str:
        """Generate searchable text from candidate data"""
        try:
            # Extract key information
            name = candidate.get('name', '')
            title = candidate.get('title', '')
            skills = candidate.get('skills', [])
            experience = candidate.get('experience', '')
            education = candidate.get('education', '')
            
            # Convert skills to string if it's a list
            if isinstance(skills, list):
                skills_text = ', '.join(skills)
            else:
                skills_text = str(skills)
            
            # Combine all information
            search_text = f"{name} {title} {skills_text} {experience} {education}"
            
            # Clean and truncate
            search_text = ' '.join(search_text.split())
            return search_text[:2000]  # Limit length for performance
            
        except Exception as e:
            logger.error(f"Error generating search text: {e}")
            return ""
    
    def search_ultra_fast(self, query: str, top_k: int = 20, use_cache: bool = True) -> List[SearchResult]:
        """Perform ultra-fast parallel search"""
        start_time = time.time()
        
        with self.stats_lock:
            self.stats['total_searches'] += 1
        
        try:
            # Check cache first
            if use_cache:
                cache_key = f"{query.lower()}_{top_k}"
                if cache_key in self.search_cache:
                    with self.stats_lock:
                        self.stats['cache_hits'] += 1
                    logger.info("Search cache hit")
                    return self.search_cache[cache_key]
            
            if not self.faiss_index or self.faiss_index.ntotal == 0:
                logger.warning("FAISS index not available")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_service.encode_single(query)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Perform FAISS search
            faiss_start = time.time()
            scores, indices = self.faiss_index.search(query_embedding, min(top_k * 2, self.faiss_index.ntotal))
            faiss_time = time.time() - faiss_start
            
            # Process results in parallel
            results = []
            job_domain = detect_domain_from_text(query)

            with ThreadPoolExecutor(max_workers=min(8, len(indices[0]))) as executor:
                future_to_idx = {
                    executor.submit(self._process_search_result, idx, score, query, job_domain): (idx, score)
                    for idx, score in zip(indices[0], scores[0])
                    if idx < len(self.candidate_ids)
                }
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing search result: {e}")
            
            # Sort by score and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
            
            # Cache results
            if use_cache:
                self.search_cache[cache_key] = results
                if len(self.search_cache) > self.cache_max_size:
                    # Remove oldest entries
                    keys_to_remove = list(self.search_cache.keys())[:200]
                    for key in keys_to_remove:
                        del self.search_cache[key]
            
            # Update stats
            search_time = time.time() - start_time
            with self.stats_lock:
                self.stats['avg_search_time'] = (
                    (self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + search_time) /
                    self.stats['total_searches']
                )
                self.stats['faiss_search_time'] = faiss_time
            
            logger.info(f"Ultra-fast search completed: {len(results)} results in {search_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Ultra-fast search failed: {e}")
            return []
    
    def _process_search_result(self, idx: int, score: float, query: str, job_domain: str) -> Optional[SearchResult]:
        """Process a single search result"""
        try:
            if idx >= len(self.candidate_ids):
                return None
            
            candidate_id = self.candidate_ids[idx]
            candidate = self.candidate_data[candidate_id]

            email_value = resolve_candidate_email(candidate)
            if email_value:
                candidate['email'] = email_value
                contact_info = candidate.get('contactInfo')
                if not isinstance(contact_info, dict):
                    contact_info = {}
                if not contact_info.get('email'):
                    contact_info['email'] = email_value
                candidate['contactInfo'] = contact_info

            candidate_domain = detect_candidate_domain(candidate)
            adjusted_score = float(score)

            job_domain_lower = (job_domain or 'general').lower()
            candidate_domain_lower = (candidate_domain or 'general').lower()

            if job_domain_lower == 'healthcare':
                if candidate_domain_lower == 'healthcare':
                    adjusted_score *= 1.2
                elif candidate_domain_lower in {'it/tech', 'technology', 'tech'}:
                    adjusted_score *= 0.55
                else:
                    adjusted_score *= 0.8
            elif job_domain_lower == 'it/tech':
                if candidate_domain_lower == 'it/tech':
                    adjusted_score *= 1.15
                elif candidate_domain_lower in {'healthcare', 'medical'}:
                    adjusted_score *= 0.75
            else:
                if candidate_domain_lower == job_domain_lower and job_domain_lower not in {'general', ''}:
                    adjusted_score *= 1.1

            adjusted_score = max(0.0, min(adjusted_score, 0.8))  # Cap at 0.8 (80% maximum)
            
            # Calculate match reasons
            match_reasons = self._calculate_match_reasons(candidate, query, adjusted_score)
            
            return SearchResult(
                candidate_id=candidate_id,
                score=float(adjusted_score),
                match_reasons=match_reasons,
                candidate_data=candidate,
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Error processing search result: {e}")
            return None
    
    def _calculate_match_reasons(self, candidate: Dict[str, Any], query: str, score: float) -> List[str]:
        """Calculate match reasons for a candidate"""
        reasons = []
        
        # Score-based reason
        if score > 0.9:
            reasons.append(f"Excellent match (Score: {score:.1%})")
        elif score > 0.8:
            reasons.append(f"Very good match (Score: {score:.1%})")
        elif score > 0.7:
            reasons.append(f"Good match (Score: {score:.1%})")
        else:
            reasons.append(f"Reasonable match (Score: {score:.1%})")
        
        # Skill-based reasons
        skills = candidate.get('skills', [])
        if isinstance(skills, list) and skills:
            query_lower = query.lower()
            matching_skills = [skill for skill in skills if skill.lower() in query_lower]
            if matching_skills:
                reasons.append(f"Skills match: {', '.join(matching_skills[:3])}")
        
        # Title-based reasons
        title = candidate.get('title', '')
        if title and any(word.lower() in query.lower() for word in title.split()):
            reasons.append(f"Title relevance: {title}")
        
        return reasons
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.stats_lock:
            return {
                'total_searches': self.stats['total_searches'],
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['total_searches']),
                'avg_search_time': self.stats['avg_search_time'],
                'faiss_search_time': self.stats['faiss_search_time'],
                'candidates_indexed': len(self.candidate_data),
                'cache_size': len(self.search_cache)
            }

# Global instance
_ultra_fast_engine = None

def get_ultra_fast_engine(embedding_service, redis_client=None):
    """Get or create the ultra-fast search engine instance"""
    global _ultra_fast_engine
    if _ultra_fast_engine is None:
        _ultra_fast_engine = UltraFastSearchEngine(embedding_service, redis_client)
    return _ultra_fast_engine

def initialize_ultra_fast_search(candidates: Dict[str, Any], embedding_service, redis_client=None):
    """Initialize the ultra-fast search system"""
    engine = get_ultra_fast_engine(embedding_service, redis_client)
    return engine.initialize_with_candidates(candidates)

def search_ultra_fast(query: str, top_k: int = 20, use_cache: bool = True):
    """Perform ultra-fast search"""
    engine = get_ultra_fast_engine(None)
    if engine is None:
        logger.error("Ultra-fast search engine not initialized")
        return []
    
    return engine.search_ultra_fast(query, top_k, use_cache)
