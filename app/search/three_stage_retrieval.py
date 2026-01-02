"""
Three-Stage Retrieval Pipeline
Production-grade retrieval with strict pre-filters, FAISS, and cross-encoder re-ranking.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from app.search.production_skill_ontology import get_production_ontology
from app.search.enhanced_geocoding import get_enhanced_geocoding, EnhancedLocationData
from app.search.hybrid_embedding_service import get_hybrid_embedding_service
from app.search.ranking_feature_extractor import get_feature_extractor

logger = logging.getLogger(__name__)

@dataclass
class JobRequirements:
    """Structured job requirements"""
    job_id: str
    description: str
    domain: str
    
    # Skills
    required_skill_ids: List[str]
    preferred_skill_ids: List[str]
    skill_weights: Dict[str, float]
    
    # Experience
    required_experience_years: int
    
    # Certifications
    required_certifications: List[str]
    
    # Clearance/Visa
    required_clearance: Optional[str]
    required_visa: Optional[str]
    
    # Location
    location_data: EnhancedLocationData
    remote_eligible: bool
    timezone_requirement: Optional[int]  # Max offset hours
    
    # Seniority
    seniority_level: str


class ThreeStageRetrievalPipeline:
    """Complete three-stage retrieval pipeline"""
    
    def __init__(self):
        self.skill_ontology = get_production_ontology()
        self.geocoding = get_enhanced_geocoding()
        self.embedding_service = get_hybrid_embedding_service()
        self.feature_extractor = get_feature_extractor()
        
        # FAISS index (loaded separately)
        self.faiss_index = None
        self.candidate_embeddings = None
        self.candidate_ids = []
    
    def search(
        self,
        job_requirements: JobRequirements,
        candidates: List[Dict[str, Any]],
        top_k: int = 20,
        strict_required_skills: bool = True
    ) -> Dict[str, Any]:
        """
        Complete three-stage search pipeline
        
        Returns:
            {
                'results': List[Dict],
                'stage_timings': Dict[str, float],
                'candidates_per_stage': Dict[str, int]
            }
        """
        start_time = time.time()
        stage_timings = {}
        candidates_per_stage = {}
        
        # Stage 1: Pre-Filter (Exact Match)
        stage1_start = time.time()
        filtered_candidates = self._stage1_prefilter(
            candidates, job_requirements, strict_required_skills
        )
        stage_timings['stage1_prefilter'] = time.time() - stage1_start
        candidates_per_stage['after_stage1'] = len(filtered_candidates)
        
        if not filtered_candidates:
            return {
                'results': [],
                'stage_timings': stage_timings,
                'candidates_per_stage': candidates_per_stage,
                'total_time': time.time() - start_time
            }
        
        # Stage 2: FAISS Retrieval (Bi-Encoder)
        stage2_start = time.time()
        faiss_results = self._stage2_faiss_retrieval(
            job_requirements, filtered_candidates, top_k=200
        )
        stage_timings['stage2_faiss'] = time.time() - stage2_start
        candidates_per_stage['after_stage2'] = len(faiss_results)
        
        if not faiss_results:
            return {
                'results': [],
                'stage_timings': stage_timings,
                'candidates_per_stage': candidates_per_stage,
                'total_time': time.time() - start_time
            }
        
        # Stage 3: Cross-Encoder + Ranking
        stage3_start = time.time()
        final_results = self._stage3_rerank(
            job_requirements, faiss_results, filtered_candidates, top_k
        )
        stage_timings['stage3_rerank'] = time.time() - stage3_start
        candidates_per_stage['final'] = len(final_results)
        
        total_time = time.time() - start_time
        stage_timings['total'] = total_time
        
        return {
            'results': final_results,
            'stage_timings': stage_timings,
            'candidates_per_stage': candidates_per_stage,
            'total_time': total_time
        }
    
    def _stage1_prefilter(
        self,
        candidates: List[Dict],
        job_requirements: JobRequirements,
        strict_required_skills: bool
    ) -> List[Dict]:
        """Stage 1: Pre-filter with exact matching"""
        filtered = []
        
        for candidate in candidates:
            # 1. Certification filter
            if job_requirements.required_certifications:
                candidate_certs = set(
                    candidate.get('certifications', []) or []
                )
                required_certs = set(job_requirements.required_certifications)
                if not required_certs.issubset(candidate_certs):
                    continue
            
            # 2. Clearance/Visa filter
            if job_requirements.required_clearance:
                if candidate.get('clearance') != job_requirements.required_clearance:
                    continue
            
            if job_requirements.required_visa:
                if candidate.get('visa_status') != job_requirements.required_visa:
                    continue
            
            # 3. Remote eligibility filter
            candidate_location = candidate.get('location_data')
            if candidate_location:
                if isinstance(candidate_location, dict):
                    candidate_location = EnhancedLocationData(**candidate_location)
                
                if not job_requirements.remote_eligible:
                    if candidate_location.is_remote:
                        continue
            else:
                # No location data, skip if remote not eligible
                if not job_requirements.remote_eligible:
                    continue
            
            # 4. Skill pre-filter (30% threshold)
            if job_requirements.required_skill_ids:
                candidate_skill_ids = set(candidate.get('skill_ids', []) or [])
                required_skill_ids = set(job_requirements.required_skill_ids)
                
                if required_skill_ids:
                    skill_match_ratio = len(required_skill_ids & candidate_skill_ids) / len(required_skill_ids)
                    if skill_match_ratio < 0.3:
                        continue
            
            # 5. Location pre-filter (geohash)
            if candidate_location and not candidate_location.is_remote:
                job_location = job_requirements.location_data
                if job_location and not job_location.is_remote:
                    # Geohash-based pre-filter
                    job_geohash = job_location.geohash
                    candidate_geohash = candidate_location.geohash
                    
                    if not self._geohash_in_radius(candidate_geohash, job_geohash, radius_km=100):
                        continue
            
            filtered.append(candidate)
        
        logger.info(f"Stage 1: Filtered {len(candidates)} -> {len(filtered)} candidates")
        return filtered
    
    def _geohash_in_radius(self, candidate_geohash: str, job_geohash: str, radius_km: float) -> bool:
        """Check if candidate geohash is within radius of job geohash"""
        if candidate_geohash in ['remote', 'unknown'] or job_geohash in ['remote', 'unknown']:
            return True
        
        # Get neighbors
        neighbors = self.geocoding.get_geohash_neighbors(job_geohash, radius_km)
        
        # Check if candidate geohash matches any neighbor
        for neighbor in neighbors:
            if candidate_geohash.startswith(neighbor):
                return True
        
        return False
    
    def _stage2_faiss_retrieval(
        self,
        job_requirements: JobRequirements,
        candidates: List[Dict],
        top_k: int = 200
    ) -> List[Tuple[int, float]]:
        """Stage 2: FAISS retrieval using bi-encoder"""
        
        # Encode query
        query_embedding = self.embedding_service.encode_query(
            job_requirements.description,
            job_requirements.domain
        )
        
        if query_embedding is None:
            logger.warning("Could not encode query, using fallback")
            return []
        
        # Get candidate embeddings
        candidate_embeddings = []
        candidate_indices = []
        
        for i, candidate in enumerate(candidates):
            embedding = candidate.get('bi_encoder_embedding')
            if embedding is not None:
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                candidate_embeddings.append(embedding)
                candidate_indices.append(i)
        
        if not candidate_embeddings:
            logger.warning("No candidate embeddings available")
            return []
        
        candidate_embeddings = np.array(candidate_embeddings)
        
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        candidate_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Calculate similarities
        similarities = np.dot(candidate_norms, query_norm).flatten()
        
        # Get top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return (candidate_index, similarity_score)
        results = [
            (candidate_indices[int(idx)], float(similarities[idx]))
            for idx in top_indices
        ]
        
        logger.info(f"Stage 2: Retrieved {len(results)} candidates from FAISS")
        return results
    
    def _stage3_rerank(
        self,
        job_requirements: JobRequirements,
        faiss_results: List[Tuple[int, float]],
        candidates: List[Dict],
        top_k: int = 20
    ) -> List[Dict]:
        """Stage 3: Cross-encoder re-ranking + feature-based ranking"""
        
        # Get candidate texts for cross-encoder
        candidate_texts = []
        candidate_indices = []
        
        for idx, _ in faiss_results:
            candidate = candidates[idx]
            candidate_text = self._candidate_to_text(candidate)
            candidate_texts.append(candidate_text)
            candidate_indices.append(idx)
        
        # Cross-encoder scoring
        cross_scores = self.embedding_service.rerank_candidates(
            job_requirements.description,
            candidate_texts,
            top_k=len(candidate_texts)
        )
        
        # Create mapping: candidate_index -> cross_score
        cross_score_map = {}
        for rerank_idx, cross_score in cross_scores:
            original_idx = candidate_indices[rerank_idx]
            cross_score_map[original_idx] = cross_score
        
        # Extract features and calculate final scores
        ranked_results = []
        
        for idx, dense_score in faiss_results:
            candidate = candidates[idx]
            cross_score = cross_score_map.get(idx, 0.0)
            
            # Extract comprehensive features
            features = self.feature_extractor.extract_features(
                job_description=job_requirements.description,
                candidate=candidate,
                job_location=job_requirements.location_data.standardized_name if job_requirements.location_data else None,
                job_required_skills=job_requirements.required_skill_ids,
                job_preferred_skills=job_requirements.preferred_skill_ids,
                dense_similarity=dense_score,
                cross_encoder_score=cross_score
            )
            
            # Calculate final score (can use XGBoost model here)
            final_score = self._calculate_final_score(features, dense_score, cross_score)
            
            # Format result
            result = {
                'candidate': candidate,
                'matchScore': final_score,
                'Score': final_score,
                'dense_similarity': dense_score,
                'cross_encoder_score': cross_score,
                'features': features
            }
            
            ranked_results.append(result)
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x['matchScore'], reverse=True)
        
        logger.info(f"Stage 3: Re-ranked to {len(ranked_results)} final candidates")
        return ranked_results[:top_k]
    
    def _candidate_to_text(self, candidate: Dict) -> str:
        """Convert candidate to searchable text"""
        parts = []
        
        # Add resume text
        if candidate.get('resume_text'):
            parts.append(candidate['resume_text'])
        
        # Add experiences
        experiences = candidate.get('experiences', [])
        for exp in experiences:
            if isinstance(exp, dict):
                parts.append(f"{exp.get('title_normalized', '')} at {exp.get('company', '')}")
                if exp.get('achievements'):
                    parts.extend(exp['achievements'][:3])  # Top 3 achievements
        
        # Add skills
        skills = candidate.get('skills_raw', []) or candidate.get('skills', [])
        if skills:
            parts.append(', '.join(skills[:10]))  # Top 10 skills
        
        # Add education
        if candidate.get('education'):
            parts.append(candidate['education'])
        
        return ' '.join(parts)
    
    def _calculate_final_score(
        self,
        features: Dict[str, float],
        dense_score: float,
        cross_score: float
    ) -> float:
        """Calculate final ranking score"""
        
        # Weighted combination (before XGBoost model is trained)
        final_score = (
            0.35 * features.get('cross_encoder_score', cross_score) +
            0.25 * features.get('dense_similarity', dense_score) +
            0.15 * features.get('weighted_skill_match', 0) +
            0.10 * features.get('experience_match', 0) +
            0.05 * features.get('location_distance_score', 0) +
            0.05 * features.get('certification_match', 0) +
            0.05 * features.get('education_match', 0)
        )
        
        # Normalize to 0-100
        return min(100.0, max(0.0, final_score * 100))


# Global instance
_retrieval_pipeline = None

def get_retrieval_pipeline() -> ThreeStageRetrievalPipeline:
    """Get or create global retrieval pipeline instance"""
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = ThreeStageRetrievalPipeline()
    return _retrieval_pipeline

