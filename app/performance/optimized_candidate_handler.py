# Optimized Candidate Handler for 1000+ Candidates
# This module provides high-performance candidate processing and search capabilities

import os
import time
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import numpy as np
import faiss
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy import func, and_, or_, desc, asc
from app.simple_logger import get_logger
from app.models import db, CandidateProfile, CandidateSkill, CandidateEducation, CandidateExperience, CandidateCertification, CandidateProject

logger = get_logger("performance")

@dataclass
class PerformanceMetrics:
    """Track performance metrics for candidate operations"""
    total_candidates: int
    processing_time: float
    memory_usage: float
    cache_hits: int
    cache_misses: int
    database_queries: int
    search_time: float

class OptimizedCandidateHandler:
    """High-performance candidate handler for 1000+ candidates"""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache = {}
        self.metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
        
    def get_candidates_optimized(self, 
                                page: int = 1, 
                                per_page: int = 50,
                                filters: Optional[Dict] = None,
                                search_query: Optional[str] = None,
                                sort_by: str = 'created_at',
                                sort_order: str = 'desc') -> Dict[str, Any]:
        """
        Optimized candidate retrieval with pagination and filtering
        Handles 1000+ candidates efficiently
        """
        start_time = time.time()
        
        try:
            # Build base query with eager loading to avoid N+1 queries
            query = db.session.query(CandidateProfile).options(
                joinedload(CandidateProfile.skills),
                joinedload(CandidateProfile.education),
                joinedload(CandidateProfile.experience),
                joinedload(CandidateProfile.certifications),
                joinedload(CandidateProfile.projects)
            )
            
            # Apply filters
            if filters:
                query = self._apply_filters(query, filters)
            
            # Apply search query
            if search_query:
                query = self._apply_search_query(query, search_query)
            
            # Apply sorting
            query = self._apply_sorting(query, sort_by, sort_order)
            
            # Get total count efficiently
            total_count = query.count()
            
            # Apply pagination
            offset = (page - 1) * per_page
            candidates = query.offset(offset).limit(per_page).all()
            
            # Convert to dict format
            candidates_data = [candidate.to_dict() for candidate in candidates]
            
            processing_time = time.time() - start_time
            self.metrics.processing_time = processing_time
            self.metrics.total_candidates = total_count
            
            logger.info(f"Retrieved {len(candidates_data)} candidates in {processing_time:.2f}s")
            
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
                    'total_candidates': total_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving candidates: {str(e)}")
            raise
    
    def _apply_filters(self, query, filters: Dict) -> Any:
        """Apply database filters efficiently"""
        if 'experience_years_min' in filters:
            query = query.filter(CandidateProfile.experience_years >= filters['experience_years_min'])
        
        if 'experience_years_max' in filters:
            query = query.filter(CandidateProfile.experience_years <= filters['experience_years_max'])
        
        if 'location' in filters:
            query = query.filter(CandidateProfile.location.ilike(f"%{filters['location']}%"))
        
        if 'skills' in filters and filters['skills']:
            # Use EXISTS for better performance with large datasets
            skill_subquery = db.session.query(CandidateSkill.profile_id).filter(
                CandidateSkill.skill_name.in_(filters['skills'])
            ).subquery()
            query = query.filter(CandidateProfile.id.in_(skill_subquery))
        
        if 'is_public' in filters:
            query = query.filter(CandidateProfile.is_public == filters['is_public'])
        
        if 'visa_status' in filters:
            query = query.filter(CandidateProfile.visa_status == filters['visa_status'])
        
        return query
    
    def _apply_search_query(self, query, search_query: str) -> Any:
        """Apply full-text search efficiently"""
        search_terms = search_query.split()
        search_conditions = []
        
        for term in search_terms:
            term_condition = or_(
                CandidateProfile.full_name.ilike(f"%{term}%"),
                CandidateProfile.summary.ilike(f"%{term}%"),
                CandidateProfile.location.ilike(f"%{term}%")
            )
            search_conditions.append(term_condition)
        
        if search_conditions:
            query = query.filter(and_(*search_conditions))
        
        return query
    
    def _apply_sorting(self, query, sort_by: str, sort_order: str) -> Any:
        """Apply sorting efficiently"""
        if sort_by == 'created_at':
            if sort_order == 'desc':
                query = query.order_by(desc(CandidateProfile.created_at))
            else:
                query = query.order_by(asc(CandidateProfile.created_at))
        elif sort_by == 'experience_years':
            if sort_order == 'desc':
                query = query.order_by(desc(CandidateProfile.experience_years))
            else:
                query = query.order_by(asc(CandidateProfile.experience_years))
        elif sort_by == 'full_name':
            if sort_order == 'desc':
                query = query.order_by(desc(CandidateProfile.full_name))
            else:
                query = query.order_by(asc(CandidateProfile.full_name))
        
        return query
    
    def batch_process_candidates(self, 
                               candidate_ids: List[int], 
                               process_func: callable,
                               batch_size: Optional[int] = None) -> List[Any]:
        """
        Process candidates in batches for better memory management
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        results = []
        total_batches = (len(candidate_ids) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(candidate_ids)} candidates in {total_batches} batches")
        
        for i in range(0, len(candidate_ids), batch_size):
            batch = candidate_ids[i:i + batch_size]
            batch_results = self._process_batch(batch, process_func)
            results.extend(batch_results)
            
            # Log progress
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(candidate_ids)} candidates")
        
        return results
    
    def _process_batch(self, candidate_ids: List[int], process_func: callable) -> List[Any]:
        """Process a single batch of candidates"""
        try:
            # Get candidates with all relationships in one query
            candidates = db.session.query(CandidateProfile).options(
                joinedload(CandidateProfile.skills),
                joinedload(CandidateProfile.education),
                joinedload(CandidateProfile.experience),
                joinedload(CandidateProfile.certifications),
                joinedload(CandidateProfile.projects)
            ).filter(CandidateProfile.id.in_(candidate_ids)).all()
            
            # Process each candidate
            results = []
            for candidate in candidates:
                try:
                    result = process_func(candidate)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing candidate {candidate.id}: {str(e)}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return [None] * len(candidate_ids)
    
    def get_candidate_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about candidates"""
        try:
            stats = {}
            
            # Total candidates
            stats['total_candidates'] = db.session.query(CandidateProfile).count()
            
            # Public vs private
            stats['public_candidates'] = db.session.query(CandidateProfile).filter(
                CandidateProfile.is_public == True
            ).count()
            
            stats['private_candidates'] = stats['total_candidates'] - stats['public_candidates']
            
            # Experience distribution
            experience_stats = db.session.query(
                func.count(CandidateProfile.id),
                func.avg(CandidateProfile.experience_years),
                func.min(CandidateProfile.experience_years),
                func.max(CandidateProfile.experience_years)
            ).filter(CandidateProfile.experience_years.isnot(None)).first()
            
            if experience_stats[0] > 0:
                stats['experience'] = {
                    'count': experience_stats[0],
                    'average': float(experience_stats[1]) if experience_stats[1] else 0,
                    'min': experience_stats[2],
                    'max': experience_stats[3]
                }
            
            # Top skills
            top_skills = db.session.query(
                CandidateSkill.skill_name,
                func.count(CandidateSkill.id).label('count')
            ).group_by(CandidateSkill.skill_name).order_by(
                desc('count')
            ).limit(10).all()
            
            stats['top_skills'] = [
                {'skill': skill, 'count': count} 
                for skill, count in top_skills
            ]
            
            # Location distribution
            location_stats = db.session.query(
                CandidateProfile.location,
                func.count(CandidateProfile.id).label('count')
            ).filter(
                CandidateProfile.location.isnot(None),
                CandidateProfile.location != ''
            ).group_by(CandidateProfile.location).order_by(
                desc('count')
            ).limit(10).all()
            
            stats['top_locations'] = [
                {'location': location, 'count': count} 
                for location, count in location_stats
            ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting candidate statistics: {str(e)}")
            return {}
    
    def optimize_database_indexes(self) -> Dict[str, Any]:
        """Suggest database indexes for better performance"""
        indexes = [
            {
                'table': 'candidate_profiles',
                'columns': ['is_public', 'created_at'],
                'name': 'idx_candidate_public_created',
                'description': 'Index for filtering public candidates by creation date'
            },
            {
                'table': 'candidate_profiles',
                'columns': ['experience_years'],
                'name': 'idx_candidate_experience',
                'description': 'Index for filtering by experience years'
            },
            {
                'table': 'candidate_profiles',
                'columns': ['location'],
                'name': 'idx_candidate_location',
                'description': 'Index for location-based searches'
            },
            {
                'table': 'candidate_skills',
                'columns': ['skill_name'],
                'name': 'idx_candidate_skills_name',
                'description': 'Index for skill-based filtering'
            },
            {
                'table': 'candidate_skills',
                'columns': ['profile_id', 'skill_name'],
                'name': 'idx_candidate_skills_profile_skill',
                'description': 'Composite index for skill queries'
            }
        ]
        
        return {
            'suggested_indexes': indexes,
            'performance_impact': 'High - These indexes will significantly improve query performance for large datasets'
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'total_candidates_processed': self.metrics.total_candidates,
            'average_processing_time': self.metrics.processing_time,
            'cache_hit_rate': self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses),
            'database_queries': self.metrics.database_queries,
            'search_time': self.metrics.search_time
        }

# Global instance
candidate_handler = OptimizedCandidateHandler()
