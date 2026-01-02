"""
Background Search Initializer
Handles background initialization of search systems to improve first search performance.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio

from app.search.search_initializer import get_search_initializer, initialize_search_systems
try:
    from app.search.adeptai_master.enhanced_recruitment_search import EnhancedRecruitmentSearchSystem
except ImportError:
    from app.search.adeptai_components.enhanced_recruitment_search import EnhancedRecruitmentSearchSystem
from app.search.service import get_algorithm_instance

logger = logging.getLogger(__name__)

class BackgroundSearchInitializer:
    """Handles background initialization of search systems"""
    
    def __init__(self):
        self.initialization_thread = None
        self.is_initializing = False
        self.initialization_complete = False
        self.initialization_error = None
        
    def start_background_initialization(self, embedding_service, redis_client=None):
        """Start background initialization of search systems"""
        if self.is_initializing or self.initialization_complete:
            logger.info("Background initialization already running or complete")
            return
        
        logger.info("Starting background search system initialization...")
        self.is_initializing = True
        self.initialization_error = None
        
        # Start initialization in a separate thread
        self.initialization_thread = threading.Thread(
            target=self._background_initialization_worker,
            args=(embedding_service, redis_client),
            daemon=True
        )
        self.initialization_thread.start()
    
    def _background_initialization_worker(self, embedding_service, redis_client=None):
        """Background worker that initializes search systems"""
        try:
            logger.info("Background initialization worker started")
            
            # Get candidate data from the existing algorithm
            candidates = self._get_candidate_data()
            
            if not candidates:
                logger.warning("No candidate data available for initialization")
                self.initialization_error = "No candidate data available"
                return
            
            logger.info(f"Found {len(candidates)} candidates for initialization")
            
            # Initialize search systems
            success = initialize_search_systems(candidates, embedding_service, redis_client)
            
            if success:
                self.initialization_complete = True
                logger.info("✅ Background search system initialization completed successfully")
            else:
                self.initialization_error = "Initialization failed"
                logger.error("❌ Background search system initialization failed")
                
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Background initialization error: {e}")
        finally:
            self.is_initializing = False
    
    def _get_candidate_data(self) -> Dict[str, Any]:
        """Get candidate data from existing systems and normalize for downstream consumers."""
        try:
            algorithm = get_algorithm_instance()

            # Try enhanced system candidates first
            if hasattr(algorithm, 'enhanced_system') and algorithm.enhanced_system:
                raw_candidates = getattr(algorithm.enhanced_system, 'candidates', None)
                normalized = self._normalize_candidate_map(raw_candidates)
                if normalized:
                    logger.info(f"Retrieved {len(normalized)} candidates from enhanced system")
                    return normalized

            # Fallback algorithm candidates
            raw_candidates = getattr(algorithm, 'candidates', None)
            normalized = self._normalize_candidate_map(raw_candidates)
            if normalized:
                logger.info(f"Retrieved {len(normalized)} candidates from fallback algorithm")
                return normalized

            # Placeholder for direct table access (not implemented)
            if hasattr(algorithm, 'table') and algorithm.table:
                try:
                    logger.info("Attempting to load candidates from database...")
                    # Implement actual DB retrieval when available
                    return {}
                except Exception as e:
                    logger.warning(f"Could not load candidates from database: {e}")

            logger.warning("No candidate data found in any system")
            return {}

        except Exception as e:
            logger.error(f"Error getting candidate data: {e}")
            return {}

    def _normalize_candidate_map(self, candidates: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Convert candidate objects to plain dictionaries suitable for search initializers."""
        if not candidates:
            return {}

        normalized: Dict[str, Dict[str, Any]] = {}

        for candidate_id, candidate_obj in candidates.items():
            try:
                candidate_dict = self._normalize_candidate(candidate_obj)
                if candidate_dict:
                    normalized[candidate_id] = candidate_dict
            except Exception as exc:
                logger.error(f"Failed to normalize candidate {candidate_id}: {exc}")
                continue

        return normalized

    def _normalize_candidate(self, candidate_obj: Any) -> Dict[str, Any]:
        """Ensure the candidate is represented as a dictionary with expected search fields."""
        if candidate_obj is None:
            return {}

        candidate_dict: Dict[str, Any]

        if isinstance(candidate_obj, dict):
            candidate_dict = dict(candidate_obj)
        elif hasattr(candidate_obj, 'to_dict'):
            candidate_dict = dict(candidate_obj.to_dict())
        elif hasattr(candidate_obj, '__dict__'):
            candidate_dict = {
                key: value
                for key, value in vars(candidate_obj).items()
                if not key.startswith('_')
            }
        else:
            logger.warning(f"Unsupported candidate object type: {type(candidate_obj)}")
            return {}

        # Field normalization for downstream search engine expectations
        full_name = (
            candidate_dict.get('full_name')
            or candidate_dict.get('fullName')
            or candidate_dict.get('name')
            or candidate_dict.get('email')
            or 'Unknown Candidate'
        )
        candidate_dict.setdefault('name', full_name)

        title = candidate_dict.get('title')
        if not title:
            previous_roles = candidate_dict.get('previous_roles') or candidate_dict.get('previousRoles')
            if isinstance(previous_roles, list) and previous_roles:
                title = str(previous_roles[0])
            elif isinstance(previous_roles, str):
                title = previous_roles
            else:
                title = (
                    candidate_dict.get('current_position')
                    or candidate_dict.get('seniority_level')
                    or ''
                )
            candidate_dict['title'] = title

        skills = candidate_dict.get('skills') or candidate_dict.get('Skills') or []
        if isinstance(skills, str):
            skills = [skill.strip() for skill in skills.split(',') if skill.strip()]
        elif not isinstance(skills, list):
            skills = [str(skills)] if skills else []
        candidate_dict['skills'] = skills

        experience_value = (
            candidate_dict.get('experience')
            or candidate_dict.get('experience_years')
            or candidate_dict.get('Experience')
        )
        if isinstance(experience_value, (int, float)):
            candidate_dict['experience'] = f"{experience_value} years"
        elif isinstance(experience_value, str):
            candidate_dict['experience'] = experience_value
        else:
            candidate_dict['experience'] = ''

        education = candidate_dict.get('education') or candidate_dict.get('Education') or ''
        candidate_dict['education'] = education

        location = candidate_dict.get('location') or candidate_dict.get('Location') or ''
        candidate_dict['location'] = location

        # Harmonize source fields for frontend consistency
        if 'sourceUrl' not in candidate_dict and candidate_dict.get('source_url'):
            candidate_dict['sourceUrl'] = candidate_dict['source_url']
        candidate_dict.setdefault('source', candidate_dict.get('sourceUrl', 'EnhancedRecruitmentSearch'))

        # Ensure email key exists
        candidate_dict.setdefault('email', candidate_dict.get('Email') or candidate_dict.get('name'))

        return candidate_dict
    
    def get_initialization_status(self) -> Dict[str, Any]:
        """Get the current initialization status"""
        return {
            'is_initializing': self.is_initializing,
            'initialization_complete': self.initialization_complete,
            'initialization_error': self.initialization_error,
            'thread_alive': self.initialization_thread.is_alive() if self.initialization_thread else False
        }
    
    def wait_for_initialization(self, timeout: float = 30.0) -> bool:
        """Wait for initialization to complete"""
        if not self.initialization_thread:
            return False
        
        self.initialization_thread.join(timeout=timeout)
        return self.initialization_complete
    
    def force_initialization(self, embedding_service, redis_client=None) -> bool:
        """Force immediate initialization (blocking)"""
        try:
            logger.info("Forcing immediate search system initialization...")
            candidates = self._get_candidate_data()
            
            if not candidates:
                logger.error("No candidate data available for forced initialization")
                return False
            
            success = initialize_search_systems(candidates, embedding_service, redis_client)
            
            if success:
                self.initialization_complete = True
                logger.info("✅ Forced initialization completed successfully")
            else:
                logger.error("❌ Forced initialization failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Forced initialization error: {e}")
            return False

# Global background initializer instance
_background_initializer = None

def get_background_initializer():
    """Get or create the background initializer instance"""
    global _background_initializer
    if _background_initializer is None:
        _background_initializer = BackgroundSearchInitializer()
    return _background_initializer

def start_background_initialization(embedding_service, redis_client=None):
    """Start background initialization of search systems"""
    initializer = get_background_initializer()
    initializer.start_background_initialization(embedding_service, redis_client)

def get_initialization_status() -> Dict[str, Any]:
    """Get the current initialization status"""
    initializer = get_background_initializer()
    return initializer.get_initialization_status()

def wait_for_initialization(timeout: float = 30.0) -> bool:
    """Wait for initialization to complete"""
    initializer = get_background_initializer()
    return initializer.wait_for_initialization(timeout)

def force_initialization(embedding_service, redis_client=None) -> bool:
    """Force immediate initialization"""
    initializer = get_background_initializer()
    return initializer.force_initialization(embedding_service, redis_client)
