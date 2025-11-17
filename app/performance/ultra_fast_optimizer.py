# Ultra-Fast Performance Optimizer for 1000+ Candidates
# Advanced optimizations for maximum speed and efficiency

import os
import time
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
import numpy as np
import faiss
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy import func, and_, or_, desc, asc, text
from app.simple_logger import get_logger
from app.models import db, CandidateProfile, CandidateSkill, CandidateEducation, CandidateExperience

logger = get_logger("ultra_fast_optimizer")

@dataclass
class UltraFastConfig:
    """Configuration for ultra-fast operations"""
    max_workers: int = 8
    batch_size: int = 200
    cache_size: int = 2000
    prefetch_size: int = 500
    connection_pool_size: int = 30
    enable_async: bool = True
    enable_compression: bool = True
    enable_memory_mapping: bool = True

class UltraFastCandidateProcessor:
    """Ultra-fast candidate processing with advanced optimizations"""
    
    def __init__(self, config: UltraFastConfig = None):
        self.config = config or UltraFastConfig()
        self.cache = {}
        self.prefetch_cache = {}
        self.connection_pool = None
        self.performance_stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time': 0,
            'peak_memory_usage': 0
        }
        
    @lru_cache(maxsize=1000)
    def get_candidate_by_id_cached(self, candidate_id: int) -> Optional[Dict]:
        """Cached candidate retrieval with LRU cache"""
        try:
            candidate = db.session.query(CandidateProfile).options(
                joinedload(CandidateProfile.skills),
                joinedload(CandidateProfile.education),
                joinedload(CandidateProfile.experience)
            ).filter(CandidateProfile.id == candidate_id).first()
            
            if candidate:
                return candidate.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting candidate {candidate_id}: {str(e)}")
            return None
    
    def get_candidates_ultra_fast(self, 
                                 page: int = 1, 
                                 per_page: int = 100,
                                 filters: Optional[Dict] = None,
                                 search_query: Optional[str] = None,
                                 sort_by: str = 'created_at',
                                 sort_order: str = 'desc') -> Dict[str, Any]:
        """
        Ultra-fast candidate retrieval with advanced optimizations
        """
        start_time = time.time()
        
        try:
            # Use raw SQL for maximum speed
            base_query = """
                SELECT cp.id, cp.full_name, cp.location, cp.experience_years, 
                       cp.summary, cp.is_public, cp.created_at
                FROM candidate_profiles cp
            """
            
            where_conditions = []
            params = {}
            
            # Apply filters with optimized SQL
            if filters:
                if 'experience_years_min' in filters:
                    where_conditions.append("cp.experience_years >= :exp_min")
                    params['exp_min'] = filters['experience_years_min']
                
                if 'experience_years_max' in filters:
                    where_conditions.append("cp.experience_years <= :exp_max")
                    params['exp_max'] = filters['experience_years_max']
                
                if 'location' in filters:
                    where_conditions.append("cp.location ILIKE :location")
                    params['location'] = f"%{filters['location']}%"
                
                if 'is_public' in filters:
                    where_conditions.append("cp.is_public = :is_public")
                    params['is_public'] = filters['is_public']
            
            # Apply search query
            if search_query:
                search_terms = search_query.split()
                search_conditions = []
                for i, term in enumerate(search_terms):
                    search_conditions.append(f"""
                        (cp.full_name ILIKE :search_term_{i} OR 
                         cp.summary ILIKE :search_term_{i} OR 
                         cp.location ILIKE :search_term_{i})
                    """)
                    params[f'search_term_{i}'] = f"%{term}%"
                
                if search_conditions:
                    where_conditions.append(f"({' OR '.join(search_conditions)})")
            
            # Build final query
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            # Add sorting
            if sort_by == 'created_at':
                order_clause = f"ORDER BY cp.created_at {'DESC' if sort_order == 'desc' else 'ASC'}"
            elif sort_by == 'experience_years':
                order_clause = f"ORDER BY cp.experience_years {'DESC' if sort_order == 'desc' else 'ASC'}"
            elif sort_by == 'full_name':
                order_clause = f"ORDER BY cp.full_name {'DESC' if sort_order == 'desc' else 'ASC'}"
            else:
                order_clause = "ORDER BY cp.created_at DESC"
            
            base_query += f" {order_clause}"
            
            # Get total count efficiently
            count_query = f"SELECT COUNT(*) FROM ({base_query}) as count_query"
            total_count = db.session.execute(text(count_query), params).scalar()
            
            # Add pagination
            offset = (page - 1) * per_page
            paginated_query = f"{base_query} LIMIT {per_page} OFFSET {offset}"
            
            # Execute query
            result = db.session.execute(text(paginated_query), params)
            candidates_data = []
            
            # Process results in batches for better memory usage
            batch_size = min(per_page, 50)
            for i in range(0, per_page, batch_size):
                batch_results = result.fetchmany(batch_size)
                
                for row in batch_results:
                    candidate_data = {
                        'id': row[0],
                        'full_name': row[1],
                        'location': row[2],
                        'experience_years': row[3],
                        'summary': row[4],
                        'is_public': row[5],
                        'created_at': row[6].isoformat() if row[6] else None
                    }
                    candidates_data.append(candidate_data)
            
            processing_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats['total_processed'] += len(candidates_data)
            self.performance_stats['avg_processing_time'] = (
                (self.performance_stats['avg_processing_time'] * 
                 (self.performance_stats['total_processed'] - len(candidates_data)) + 
                 processing_time) / self.performance_stats['total_processed']
            )
            
            logger.info(f"Ultra-fast retrieval: {len(candidates_data)} candidates in {processing_time:.3f}s")
            
            return {
                'candidates': candidates_data,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total_count,
                    'pages': (total_count + per_page - 1) // per_page
                },
                'performance': {
                    'processing_time': processing_time,
                    'candidates_per_second': len(candidates_data) / max(processing_time, 0.001),
                    'total_candidates': total_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ultra-fast retrieval: {str(e)}")
            raise
    
    def batch_process_ultra_fast(self, 
                                candidate_ids: List[int], 
                                process_func: callable,
                                batch_size: Optional[int] = None) -> List[Any]:
        """
        Ultra-fast batch processing with parallel execution
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        start_time = time.time()
        
        try:
            # Split into batches
            batches = [candidate_ids[i:i + batch_size] for i in range(0, len(candidate_ids), batch_size)]
            
            # Process batches in parallel
            results = []
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_batch = {
                    executor.submit(self._process_batch_ultra_fast, batch, process_func): batch 
                    for batch in batches
                }
                
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Error processing batch {batch}: {str(e)}")
                        results.extend([None] * len(batch))
            
            processing_time = time.time() - start_time
            logger.info(f"Ultra-fast batch processing: {len(results)} candidates in {processing_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ultra-fast batch processing: {str(e)}")
            raise
    
    def _process_batch_ultra_fast(self, candidate_ids: List[int], process_func: callable) -> List[Any]:
        """Process a single batch with ultra-fast optimizations"""
        try:
            # Use raw SQL for maximum speed
            ids_str = ','.join(map(str, candidate_ids))
            query = f"""
                SELECT cp.id, cp.full_name, cp.location, cp.experience_years, 
                       cp.summary, cp.is_public, cp.created_at
                FROM candidate_profiles cp
                WHERE cp.id IN ({ids_str})
            """
            
            result = db.session.execute(text(query))
            candidates = []
            
            for row in result:
                candidate_data = {
                    'id': row[0],
                    'full_name': row[1],
                    'location': row[2],
                    'experience_years': row[3],
                    'summary': row[4],
                    'is_public': row[5],
                    'created_at': row[6].isoformat() if row[6] else None
                }
                candidates.append(candidate_data)
            
            # Process each candidate
            results = []
            for candidate_data in candidates:
                try:
                    result = process_func(candidate_data)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing candidate {candidate_data['id']}: {str(e)}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return [None] * len(candidate_ids)
    
    def prefetch_candidates(self, candidate_ids: List[int]) -> None:
        """Prefetch candidates for faster subsequent access"""
        try:
            # Use raw SQL for prefetching
            ids_str = ','.join(map(str, candidate_ids))
            query = f"""
                SELECT cp.id, cp.full_name, cp.location, cp.experience_years, 
                       cp.summary, cp.is_public, cp.created_at
                FROM candidate_profiles cp
                WHERE cp.id IN ({ids_str})
            """
            
            result = db.session.execute(text(query))
            
            for row in result:
                candidate_data = {
                    'id': row[0],
                    'full_name': row[1],
                    'location': row[2],
                    'experience_years': row[3],
                    'summary': row[4],
                    'is_public': row[5],
                    'created_at': row[6].isoformat() if row[6] else None
                }
                self.prefetch_cache[row[0]] = candidate_data
            
            logger.info(f"Prefetched {len(candidate_ids)} candidates")
            
        except Exception as e:
            logger.error(f"Error prefetching candidates: {str(e)}")
    
    def get_candidate_ultra_fast(self, candidate_id: int) -> Optional[Dict]:
        """Ultra-fast single candidate retrieval"""
        # Check prefetch cache first
        if candidate_id in self.prefetch_cache:
            self.performance_stats['cache_hits'] += 1
            return self.prefetch_cache[candidate_id]
        
        # Check main cache
        if candidate_id in self.cache:
            self.performance_stats['cache_hits'] += 1
            return self.cache[candidate_id]
        
        # Use cached method
        self.performance_stats['cache_misses'] += 1
        candidate_data = self.get_candidate_by_id_cached(candidate_id)
        
        if candidate_data:
            self.cache[candidate_id] = candidate_data
            if len(self.cache) > self.config.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        return candidate_data
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get ultra-fast performance statistics"""
        cache_hit_rate = 0
        if self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'] > 0:
            cache_hit_rate = self.performance_stats['cache_hits'] / (
                self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
            )
        
        return {
            **self.performance_stats,
            'cache_hit_rate': cache_hit_rate,
            'prefetch_cache_size': len(self.prefetch_cache),
            'main_cache_size': len(self.cache)
        }
    
    def clear_caches(self):
        """Clear all caches"""
        self.cache.clear()
        self.prefetch_cache.clear()
        self.get_candidate_by_id_cached.cache_clear()
        logger.info("All caches cleared")

class UltraFastSearchEngine:
    """Ultra-fast search engine with advanced optimizations"""
    
    def __init__(self, config: UltraFastConfig = None):
        self.config = config or UltraFastConfig()
        self.faiss_index = None
        self.embedding_model = None
        self.search_cache = {}
        self.performance_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0
        }
    
    def initialize_search_engine(self):
        """Initialize the search engine with optimizations"""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            
            # Load optimized embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Build FAISS index with optimizations
            self._build_optimized_faiss_index()
            
            logger.info("Ultra-fast search engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing search engine: {str(e)}")
    
    def _build_optimized_faiss_index(self):
        """Build optimized FAISS index for maximum speed"""
        try:
            # Get all candidates with optimized query
            query = """
                SELECT cp.id, cp.full_name, cp.summary, cp.location,
                       GROUP_CONCAT(cs.skill_name) as skills,
                       GROUP_CONCAT(ce.institution) as education
                FROM candidate_profiles cp
                LEFT JOIN candidate_skills cs ON cp.id = cs.profile_id
                LEFT JOIN candidate_education ce ON cp.id = ce.profile_id
                GROUP BY cp.id
            """
            
            result = db.session.execute(text(query))
            candidates = []
            
            for row in result:
                candidate_data = {
                    'id': row[0],
                    'full_name': row[1] or '',
                    'summary': row[2] or '',
                    'location': row[3] or '',
                    'skills': row[4] or '',
                    'education': row[5] or ''
                }
                candidates.append(candidate_data)
            
            if not candidates:
                logger.warning("No candidates found for indexing")
                return
            
            # Generate embeddings in optimized batches
            texts = [self._generate_search_text(candidate) for candidate in candidates]
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=64,  # Optimized batch size
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Create optimized FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.faiss_index.add(embeddings)
            
            # Store candidate IDs
            self.candidate_ids = [c['id'] for c in candidates]
            
            logger.info(f"Optimized FAISS index built with {len(candidates)} candidates")
            
        except Exception as e:
            logger.error(f"Error building optimized FAISS index: {str(e)}")
            raise
    
    def _generate_search_text(self, candidate: Dict) -> str:
        """Generate optimized search text"""
        text_parts = [
            candidate.get('full_name', ''),
            candidate.get('summary', ''),
            candidate.get('location', ''),
            candidate.get('skills', ''),
            candidate.get('education', '')
        ]
        return ' '.join(filter(None, text_parts))
    
    def search_ultra_fast(self, 
                         query: str, 
                         top_k: int = 20,
                         use_cache: bool = True) -> List[Dict]:
        """Ultra-fast search with advanced optimizations"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{query.lower()}_{top_k}"
            if use_cache and cache_key in self.search_cache:
                self.performance_stats['cache_hits'] += 1
                return self.search_cache[cache_key]
            
            if not self.faiss_index or not self.embedding_model:
                logger.warning("Search engine not initialized, using fallback")
                return self._fallback_search(query, top_k)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search with optimized parameters
            scores, indices = self.faiss_index.search(
                query_embedding, 
                min(top_k * 2, len(self.candidate_ids))
            )
            
            # Get results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.candidate_ids):
                    results.append({
                        'candidate_id': self.candidate_ids[idx],
                        'score': float(score),
                        'match_reasons': [f"Similarity score: {score:.3f}"]
                    })
            
            # Sort by score and limit
            results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
            
            # Cache results
            if use_cache:
                self.search_cache[cache_key] = results
                if len(self.search_cache) > self.config.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.search_cache))
                    del self.search_cache[oldest_key]
            
            # Update stats
            search_time = time.time() - start_time
            self.performance_stats['total_searches'] += 1
            self.performance_stats['avg_search_time'] = (
                (self.performance_stats['avg_search_time'] * 
                 (self.performance_stats['total_searches'] - 1) + search_time) / 
                self.performance_stats['total_searches']
            )
            
            logger.info(f"Ultra-fast search: {len(results)} results in {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ultra-fast search: {str(e)}")
            return self._fallback_search(query, top_k)
    
    def _fallback_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback to database search"""
        try:
            search_terms = query.split()
            search_conditions = []
            params = {}
            
            for i, term in enumerate(search_terms):
                search_conditions.append(f"""
                    (cp.full_name ILIKE :search_term_{i} OR 
                     cp.summary ILIKE :search_term_{i} OR 
                     cp.location ILIKE :search_term_{i})
                """)
                params[f'search_term_{i}'] = f"%{term}%"
            
            where_clause = " OR ".join(search_conditions)
            query_sql = f"""
                SELECT cp.id, 
                       (CASE WHEN cp.full_name ILIKE :query THEN 1 ELSE 0 END +
                        CASE WHEN cp.summary ILIKE :query THEN 1 ELSE 0 END +
                        CASE WHEN cp.location ILIKE :query THEN 1 ELSE 0 END) as relevance
                FROM candidate_profiles cp
                WHERE {where_clause}
                ORDER BY relevance DESC, cp.created_at DESC
                LIMIT {top_k}
            """
            params['query'] = f"%{query}%"
            
            result = db.session.execute(text(query_sql), params)
            results = []
            
            for row in result:
                results.append({
                    'candidate_id': row[0],
                    'score': float(row[1]) / 3.0,  # Normalize to 0-1
                    'match_reasons': [f"Database search match for '{query}'"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fallback search: {str(e)}")
            return []

# Global instances
ultra_fast_processor = UltraFastCandidateProcessor()
ultra_fast_search_engine = UltraFastSearchEngine()
