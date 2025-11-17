# Optimized Search System for 1000+ Candidates
# High-performance search with caching, indexing, and batch processing

import os
import time
import json
import pickle
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import joinedload
from app.simple_logger import get_logger
from app.models import db, CandidateProfile, CandidateSkill
from .optimized_candidate_handler import OptimizedCandidateHandler

logger = get_logger("search_performance")

@dataclass
class SearchResult:
    """Search result with metadata"""
    candidate_id: int
    score: float
    match_reasons: List[str]
    candidate_data: Dict[str, Any]

class OptimizedSearchSystem:
    """High-performance search system for large candidate datasets"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 cache_size: int = 1000,
                 batch_size: int = 100):
        self.embedding_model = embedding_model
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.candidate_handler = OptimizedCandidateHandler()
        
        # Initialize components
        self.embedding_model_instance = None
        self.faiss_index = None
        self.candidate_embeddings = {}
        self.search_cache = {}
        self.candidate_ids = None
        self.performance_metrics = {
            'total_searches': 0,
            'cache_hits': 0,
            'average_search_time': 0,
            'total_candidates_indexed': 0
        }
        self._initialized = False
        
        # Don't initialize immediately - wait for first use
    
    def _ensure_initialized(self):
        """Ensure the system is initialized (lazy initialization)"""
        if not self._initialized:
            self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the search system components"""
        try:
            logger.info("Initializing optimized search system...")
            
            # Load embedding model
            self.embedding_model_instance = SentenceTransformer(self.embedding_model)
            
            # Build FAISS index
            self._build_faiss_index()
            
            self._initialized = True
            logger.info("Search system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing search system: {str(e)}")
            # Don't raise - allow fallback to database search
            self._initialized = False
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        try:
            logger.info("Building FAISS index...")
            
            # Get all candidate profiles
            candidates = db.session.query(CandidateProfile).options(
                joinedload(CandidateProfile.skills),
                joinedload(CandidateProfile.education),
                joinedload(CandidateProfile.experience)
            ).all()
            
            if not candidates:
                logger.warning("No candidates found for indexing")
                return
            
            # Generate embeddings in batches
            embeddings = []
            candidate_ids = []
            
            for i in range(0, len(candidates), self.batch_size):
                batch = candidates[i:i + self.batch_size]
                batch_texts = [self._generate_search_text(candidate) for candidate in batch]
                
                # Generate embeddings for batch
                batch_embeddings = self.embedding_model_instance.encode(
                    batch_texts, 
                    batch_size=32,
                    show_progress_bar=True
                )
                
                embeddings.append(batch_embeddings)
                candidate_ids.extend([c.id for c in batch])
                
                logger.info(f"Processed {min(i + self.batch_size, len(candidates))}/{len(candidates)} candidates")
            
            # Combine all embeddings
            all_embeddings = np.vstack(embeddings)
            
            # Create FAISS index
            dimension = all_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(all_embeddings)
            
            # Add embeddings to index
            self.faiss_index.add(all_embeddings)
            
            # Store candidate IDs for mapping
            self.candidate_ids = np.array(candidate_ids)
            
            self.performance_metrics['total_candidates_indexed'] = len(candidates)
            logger.info(f"FAISS index built with {len(candidates)} candidates")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise
    
    def _generate_search_text(self, candidate: CandidateProfile) -> str:
        """Generate searchable text for a candidate"""
        try:
            text_parts = []
            
            # Basic info
            if candidate.full_name:
                text_parts.append(candidate.full_name)
            
            if candidate.summary:
                text_parts.append(candidate.summary)
            
            if candidate.location:
                text_parts.append(candidate.location)
            
            # Skills
            if candidate.skills:
                skills_text = " ".join([skill.skill_name for skill in candidate.skills])
                text_parts.append(skills_text)
            
            # Education
            if candidate.education:
                education_text = " ".join([
                    f"{edu.institution} {edu.degree} {edu.field_of_study or ''}" 
                    for edu in candidate.education
                ])
                text_parts.append(education_text)
            
            # Experience
            if candidate.experience:
                experience_text = " ".join([
                    f"{exp.job_title} {exp.company} {exp.description or ''}" 
                    for exp in candidate.experience
                ])
                text_parts.append(experience_text)
            
            return " ".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error generating search text for candidate {candidate.id}: {str(e)}")
            return ""
    
    def search_candidates(self, 
                         query: str, 
                         top_k: int = 20,
                         filters: Optional[Dict] = None,
                         use_cache: bool = True) -> List[SearchResult]:
        """
        Search candidates with high performance
        """
        start_time = time.time()
        
        try:
            # Ensure system is initialized
            self._ensure_initialized()
            
            # Check cache first
            cache_key = self._generate_cache_key(query, top_k, filters)
            if use_cache and cache_key in self.search_cache:
                self.performance_metrics['cache_hits'] += 1
                logger.info("Cache hit for search query")
                return self.search_cache[cache_key]
            
            # Check if FAISS is available
            if not self._initialized or self.faiss_index is None:
                logger.warning("FAISS index not available, falling back to database search")
                return self._fallback_search(query, top_k, filters)
            
            # Generate query embedding
            query_embedding = self.embedding_model_instance.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Perform similarity search
            scores, indices = self.faiss_index.search(query_embedding, min(top_k * 2, len(self.candidate_ids)))
            
            # Get candidate IDs and scores
            candidate_scores = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.candidate_ids):
                    candidate_scores.append((int(self.candidate_ids[idx]), float(score)))
            
            # Apply filters if provided
            if filters:
                candidate_scores = self._apply_filters_to_results(candidate_scores, filters)
            
            # Get top results
            candidate_scores = candidate_scores[:top_k]
            
            # Get candidate data
            results = self._get_search_results(candidate_scores, query)
            
            # Cache results
            if use_cache:
                self.search_cache[cache_key] = results
                if len(self.search_cache) > self.cache_size:
                    # Remove oldest entries
                    oldest_key = next(iter(self.search_cache))
                    del self.search_cache[oldest_key]
            
            # Update metrics
            search_time = time.time() - start_time
            self.performance_metrics['total_searches'] += 1
            self.performance_metrics['average_search_time'] = (
                (self.performance_metrics['average_search_time'] * (self.performance_metrics['total_searches'] - 1) + search_time) 
                / self.performance_metrics['total_searches']
            )
            
            logger.info(f"Search completed in {search_time:.2f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return self._fallback_search(query, top_k, filters)
    
    def _generate_cache_key(self, query: str, top_k: int, filters: Optional[Dict]) -> str:
        """Generate cache key for search query"""
        key_data = {
            'query': query.lower().strip(),
            'top_k': top_k,
            'filters': filters or {}
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _apply_filters_to_results(self, 
                                 candidate_scores: List[Tuple[int, float]], 
                                 filters: Dict) -> List[Tuple[int, float]]:
        """Apply filters to search results"""
        try:
            if not filters:
                return candidate_scores
            
            # Get candidate IDs
            candidate_ids = [candidate_id for candidate_id, _ in candidate_scores]
            
            # Build filter query
            query = db.session.query(CandidateProfile.id)
            
            if 'experience_years_min' in filters:
                query = query.filter(CandidateProfile.experience_years >= filters['experience_years_min'])
            
            if 'experience_years_max' in filters:
                query = query.filter(CandidateProfile.experience_years <= filters['experience_years_max'])
            
            if 'location' in filters:
                query = query.filter(CandidateProfile.location.ilike(f"%{filters['location']}%"))
            
            if 'is_public' in filters:
                query = query.filter(CandidateProfile.is_public == filters['is_public'])
            
            # Apply candidate ID filter
            query = query.filter(CandidateProfile.id.in_(candidate_ids))
            
            # Get filtered IDs
            filtered_ids = set(row[0] for row in query.all())
            
            # Filter results
            filtered_scores = [
                (candidate_id, score) for candidate_id, score in candidate_scores
                if candidate_id in filtered_ids
            ]
            
            return filtered_scores
            
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return candidate_scores
    
    def _get_search_results(self, 
                           candidate_scores: List[Tuple[int, float]], 
                           query: str) -> List[SearchResult]:
        """Get search results with candidate data"""
        try:
            if not candidate_scores:
                return []
            
            # Get candidate IDs
            candidate_ids = [candidate_id for candidate_id, _ in candidate_scores]
            
            # Get candidates with relationships
            candidates = db.session.query(CandidateProfile).options(
                joinedload(CandidateProfile.skills),
                joinedload(CandidateProfile.education),
                joinedload(CandidateProfile.experience),
                joinedload(CandidateProfile.certifications),
                joinedload(CandidateProfile.projects)
            ).filter(CandidateProfile.id.in_(candidate_ids)).all()
            
            # Create mapping
            candidate_map = {c.id: c for c in candidates}
            
            # Build results
            results = []
            for candidate_id, score in candidate_scores:
                if candidate_id in candidate_map:
                    candidate = candidate_map[candidate_id]
                    match_reasons = self._generate_match_reasons(candidate, query)
                    
                    result = SearchResult(
                        candidate_id=candidate_id,
                        score=score,
                        match_reasons=match_reasons,
                        candidate_data=candidate.to_dict()
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting search results: {str(e)}")
            return []
    
    def _generate_match_reasons(self, candidate: CandidateProfile, query: str) -> List[str]:
        """Generate reasons why candidate matches the query"""
        reasons = []
        query_lower = query.lower()
        
        # Check name match
        if candidate.full_name and query_lower in candidate.full_name.lower():
            reasons.append(f"Name contains '{query}'")
        
        # Check summary match
        if candidate.summary and query_lower in candidate.summary.lower():
            reasons.append(f"Summary contains '{query}'")
        
        # Check skills match
        if candidate.skills:
            matching_skills = [
                skill.skill_name for skill in candidate.skills
                if query_lower in skill.skill_name.lower()
            ]
            if matching_skills:
                reasons.append(f"Skills: {', '.join(matching_skills[:3])}")
        
        # Check location match
        if candidate.location and query_lower in candidate.location.lower():
            reasons.append(f"Location: {candidate.location}")
        
        return reasons[:5]  # Limit to 5 reasons
    
    def _fallback_search(self, query: str, top_k: int, filters: Optional[Dict]) -> List[SearchResult]:
        """Fallback to database search if FAISS is not available"""
        try:
            logger.info("Using fallback database search")
            
            # Use the optimized candidate handler
            search_results = self.candidate_handler.get_candidates_optimized(
                page=1,
                per_page=top_k,
                filters=filters,
                search_query=query
            )
            
            # Convert to SearchResult format
            results = []
            for candidate_data in search_results['candidates']:
                result = SearchResult(
                    candidate_id=candidate_data['id'],
                    score=0.8,  # Default score for fallback
                    match_reasons=[f"Database search match for '{query}'"],
                    candidate_data=candidate_data
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fallback search: {str(e)}")
            return []
    
    def rebuild_index(self):
        """Rebuild the search index"""
        try:
            logger.info("Rebuilding search index...")
            self._build_faiss_index()
            logger.info("Search index rebuilt successfully")
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get search performance metrics"""
        return {
            **self.performance_metrics,
            'cache_size': len(self.search_cache),
            'index_size': self.faiss_index.ntotal if self.faiss_index else 0
        }
    
    def clear_cache(self):
        """Clear search cache"""
        self.search_cache.clear()
        logger.info("Search cache cleared")

# Global instance
search_system = OptimizedSearchSystem()
