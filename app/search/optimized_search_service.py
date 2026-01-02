"""
Optimized Search Service
Integrates ultra-fast parallel search with existing system for maximum performance.
"""

import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from app.search.ultra_fast_parallel_search import (
    get_ultra_fast_engine, 
    initialize_ultra_fast_search,
    search_ultra_fast,
    detect_candidate_domain,
    detect_domain_from_text,
    _normalize_domain_label
)
from app.search.service import get_algorithm_instance
try:
    from app.search.adeptai_master.enhanced_recruitment_search import EnhancedRecruitmentSearchSystem
except ImportError:
    from app.search.adeptai_components.enhanced_recruitment_search import EnhancedRecruitmentSearchSystem

logger = logging.getLogger(__name__)

class OptimizedSearchService:
    """High-performance search service with parallel processing and intelligent fallbacks"""
    
    def __init__(self, embedding_service, redis_client=None):
        self.embedding_service = embedding_service
        self.redis_client = redis_client
        self.ultra_fast_engine = None
        self.fallback_algorithm = None
        self.initialization_lock = threading.Lock()
        self.is_initialized = False
        
        # Performance tracking
        self.stats = {
            'ultra_fast_searches': 0,
            'fallback_searches': 0,
            'total_searches': 0,
            'avg_ultra_fast_time': 0.0,
            'avg_fallback_time': 0.0,
            'initialization_time': 0.0
        }

    def _candidate_matches_domain(self, candidate: Dict[str, Any], job_domain: str) -> bool:
        if job_domain not in {'healthcare', 'it/tech'}:
            return True

        detected = detect_candidate_domain(candidate).lower()
        if detected == job_domain:
            return True

        combined_parts: List[str] = []
        for key in [
            'domain', 'domain_tag', 'category',
            'title', 'current_position', 'position', 'role',
            'experience', 'Experience', 'experience_summary',
            'education', 'Education',
            'certifications', 'Certifications'
        ]:
            value = candidate.get(key)
            if isinstance(value, str):
                combined_parts.append(value)
            elif isinstance(value, list):
                combined_parts.extend(str(item) for item in value if item is not None)

        skills = candidate.get('skills') or candidate.get('Skills')
        if isinstance(skills, list):
            combined_parts.extend(str(skill) for skill in skills if skill is not None)
        elif isinstance(skills, str):
            combined_parts.append(skills)

        combined_text = ' '.join(combined_parts)
        return detect_domain_from_text(combined_text) == job_domain

    def _filter_search_results_by_job_domain(self, results: List, job_domain: str) -> List:
        if job_domain not in {'healthcare', 'it/tech'}:
            return results

        filtered = [result for result in results if self._candidate_matches_domain(result.candidate_data, job_domain)]
        return filtered or results

    def _filter_formatted_results_by_job_domain(self, results: List[Dict[str, Any]], job_domain: str) -> List[Dict[str, Any]]:
        if job_domain not in {'healthcare', 'it/tech'}:
            return results

        filtered = [candidate for candidate in results if self._candidate_matches_domain(candidate, job_domain)]
        return filtered or results
        
    def initialize_async(self, candidates: Dict[str, Any]) -> bool:
        """Initialize the search service asynchronously"""
        def _init():
            try:
                logger.info(f"Initializing optimized search service with {len(candidates)} candidates")
                start_time = time.time()
                
                # Initialize ultra-fast engine
                self.ultra_fast_engine = get_ultra_fast_engine(self.embedding_service, self.redis_client)
                success = self.ultra_fast_engine.initialize_with_candidates(candidates)
                
                if success:
                    # Initialize fallback algorithm in parallel
                    try:
                        self.fallback_algorithm = get_algorithm_instance()
                        logger.info("Fallback algorithm initialized successfully")
                    except Exception as e:
                        logger.warning(f"Fallback algorithm initialization failed: {e}")
                
                init_time = time.time() - start_time
                self.stats['initialization_time'] = init_time
                self.is_initialized = success
                
                logger.info(f"Optimized search service initialized in {init_time:.2f}s")
                return success
                
            except Exception as e:
                logger.error(f"Failed to initialize optimized search service: {e}")
                return False
        
        # Run initialization in a separate thread to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_init)
            return future.result(timeout=30)  # 30 second timeout
    
    def search_optimized(self, query: str, top_k: int = 20, use_ultra_fast: bool = True) -> Dict[str, Any]:
        """Perform optimized search with intelligent fallbacks"""
        start_time = time.time()
        self.stats['total_searches'] += 1
        
        try:
            # Try ultra-fast search first if available and enabled
            if use_ultra_fast and self.ultra_fast_engine and self.is_initialized:
                try:
                    logger.info("🚀 Using ultra-fast parallel search...")
                    results = self.ultra_fast_engine.search_ultra_fast(query, top_k, use_cache=True)
                    
                    if results:
                        # Convert to expected format
                        formatted_results = self._format_ultra_fast_results(results, query)
                        
                        search_time = time.time() - start_time
                        self.stats['ultra_fast_searches'] += 1
                        self._update_avg_time('avg_ultra_fast_time', search_time)
                        
                        logger.info(f"Ultra-fast search completed: {len(formatted_results)} results in {search_time:.2f}s")
                        
                        return {
                            'results': formatted_results,
                            'summary': f"Found {len(formatted_results)} highly matched candidates using Ultra-Fast Parallel Algorithm ({search_time:.2f}s)",
                            'total_candidates': len(formatted_results),
                            'algorithm_used': 'Ultra-Fast Parallel Algorithm',
                            'processing_time': search_time,
                            'performance_boost': True
                        }
                    else:
                        logger.warning("Ultra-fast search returned no results, trying fallback")
                        
                except Exception as e:
                    logger.error(f"Ultra-fast search failed: {e}, trying fallback")
            
            # Fallback to original algorithm
            if self.fallback_algorithm:
                try:
                    logger.info("🔄 Using fallback algorithm...")
                    fallback_results = self.fallback_algorithm.semantic_match(query)
                    
                    search_time = time.time() - start_time
                    self.stats['fallback_searches'] += 1
                    self._update_avg_time('avg_fallback_time', search_time)
                    
                    logger.info(f"Fallback search completed in {search_time:.2f}s")
                    
                    # Add performance info to results
                    if isinstance(fallback_results, dict):
                        fallback_results['processing_time'] = search_time
                        fallback_results['algorithm_used'] = 'Enhanced Fallback Algorithm'
                        fallback_results['performance_boost'] = False

                        job_domain = detect_domain_from_text(query)
                        filtered_formatted = self._filter_formatted_results_by_job_domain(
                            fallback_results.get('results', []),
                            job_domain
                        )
                        fallback_results['results'] = filtered_formatted
                        fallback_results['total_candidates'] = len(filtered_formatted)
                    
                    return fallback_results
                    
                except Exception as e:
                    logger.error(f"Fallback search also failed: {e}")
            
            # Ultimate fallback - return empty results
            search_time = time.time() - start_time
            logger.error("All search methods failed")
            
            return {
                'results': [],
                'summary': f"Search temporarily unavailable. Please try again later. ({search_time:.2f}s)",
                'total_candidates': 0,
                'algorithm_used': 'None Available',
                'processing_time': search_time,
                'error': True
            }
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Search service error: {e}")
            
            return {
                'results': [],
                'summary': f"Search error: {str(e)} ({search_time:.2f}s)",
                'total_candidates': 0,
                'algorithm_used': 'Error',
                'processing_time': search_time,
                'error': True
            }
    
    def _format_ultra_fast_results(self, results: List, query: str) -> List[Dict[str, Any]]:
        """Format ultra-fast search results for frontend compatibility"""
        job_domain = detect_domain_from_text(query)
        filtered_results = self._filter_search_results_by_job_domain(results, job_domain)
        formatted_results = []
        
        for result in filtered_results:
            try:
                candidate_data = result.candidate_data
                
                # Calculate match percentage from score (capped at 80% maximum)
                match_percentage = min(80.0, result.score * 100)

                detected_domain = detect_candidate_domain(candidate_data)
                normalized_domain = _normalize_domain_label(detected_domain)

                # Derive domain confidence if not provided
                domain_confidence = candidate_data.get('domain_confidence')
                if domain_confidence is None:
                    if job_domain in {'healthcare', 'it/tech'}:
                        domain_confidence = 0.9 if normalized_domain.lower() == job_domain else 0.45
                    else:
                        domain_confidence = 0.75
                
                # Calculate confidence score based on match quality and data completeness
                confidence_score = self._calculate_confidence_score(match_percentage, candidate_data)
                
                # Format the result
                formatted_result = {
                    'id': result.candidate_id,
                    'name': candidate_data.get('name', 'Unknown Candidate'),
                    'title': candidate_data.get('title', ''),
                    'location': candidate_data.get('location', ''),
                    'skills': candidate_data.get('skills', []),
                    'experience': candidate_data.get('experience', ''),
                    'education': candidate_data.get('education', ''),
                    'match_percentage': match_percentage,
                    'Score': match_percentage,
                    'Confidence': confidence_score,
                    'match_reasons': result.match_reasons,
                    'source': candidate_data.get('source', 'Unknown'),
                    'sourceUrl': candidate_data.get('sourceUrl', ''),
                    'avatar': candidate_data.get('avatar', ''),
                    'crossSourceVerified': candidate_data.get('crossSourceVerified', False),
                    'crossSourceOccurrences': candidate_data.get('crossSourceOccurrences', 0),
                    'category': candidate_data.get('category') or candidate_data.get('domain_tag') or normalized_domain,
                    'domain': candidate_data.get('domain') or candidate_data.get('domain_tag') or normalized_domain,
                    'domain_confidence': domain_confidence,
                    'processing_time': result.processing_time
                }
                
                # Add additional fields if available
                for key in ['phone', 'email', 'linkedin', 'github', 'portfolio', 'certifications', 'languages', 'availability', 'salary_expectations', 'relocation']:
                    if key in candidate_data:
                        formatted_result[key] = candidate_data[key]
                
                formatted_results.append(formatted_result)
                
            except Exception as e:
                logger.error(f"Error formatting result: {e}")
                continue
        
        return formatted_results
    
    def _update_avg_time(self, stat_key: str, new_time: float):
        """Update average time statistics"""
        if stat_key in self.stats:
            current_avg = self.stats[stat_key]
            count = self.stats['ultra_fast_searches'] if 'ultra_fast' in stat_key else self.stats['fallback_searches']
            
            if count > 0:
                self.stats[stat_key] = ((current_avg * (count - 1)) + new_time) / count
            else:
                self.stats[stat_key] = new_time
    
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
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.stats.copy()
        
        if self.ultra_fast_engine:
            ultra_fast_stats = self.ultra_fast_engine.get_performance_stats()
            stats.update(ultra_fast_stats)
        
        return stats
    
    def is_ready(self) -> bool:
        """Check if the search service is ready to use"""
        return self.is_initialized and (self.ultra_fast_engine is not None or self.fallback_algorithm is not None)

# Global service instance
_optimized_service = None

def get_optimized_search_service(embedding_service=None, redis_client=None):
    """Get or create the optimized search service instance"""
    global _optimized_service
    if _optimized_service is None and embedding_service:
        _optimized_service = OptimizedSearchService(embedding_service, redis_client)
    return _optimized_service

def initialize_optimized_search(candidates: Dict[str, Any], embedding_service, redis_client=None):
    """Initialize the optimized search service"""
    service = get_optimized_search_service(embedding_service, redis_client)
    if service:
        return service.initialize_async(candidates)
    return False

def search_optimized(query: str, top_k: int = 20, use_ultra_fast: bool = True):
    """Perform optimized search"""
    service = get_optimized_search_service()
    if service and service.is_ready():
        return service.search_optimized(query, top_k, use_ultra_fast)
    else:
        logger.error("Optimized search service not available")
        return {
            'results': [],
            'summary': 'Search service not initialized',
            'total_candidates': 0,
            'error': True
        }
