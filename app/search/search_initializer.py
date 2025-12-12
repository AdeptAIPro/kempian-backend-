"""
Search System Initializer
Handles initialization of the optimized search system with candidate data.
"""

import logging
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

from app.search.optimized_search_service import get_optimized_search_service, initialize_optimized_search
try:
    from app.search.adeptai_master.enhanced_recruitment_search import EnhancedRecruitmentSearchSystem
except ImportError:
    from app.search.adeptai_components.enhanced_recruitment_search import EnhancedRecruitmentSearchSystem
from app.search.service import get_algorithm_instance

logger = logging.getLogger(__name__)

class SearchSystemInitializer:
    """Handles initialization of search systems with candidate data"""
    
    def __init__(self):
        self.initialization_status = {
            'optimized_search': False,
            'fallback_algorithm': False,
            'initialization_time': 0.0,
            'candidate_count': 0,
            'last_initialization': None
        }
        self.initialization_lock = threading.Lock()
        self.is_initializing = False
    
    def initialize_search_systems(self, candidates: Dict[str, Any], embedding_service, redis_client=None) -> bool:
        """Initialize all search systems with candidate data"""
        with self.initialization_lock:
            if self.is_initializing:
                logger.info("Search system initialization already in progress")
                return False
            
            self.is_initializing = True
            start_time = time.time()
            
            try:
                logger.info(f"Initializing search systems with {len(candidates)} candidates")
                
                # Initialize optimized search system
                optimized_success = self._initialize_optimized_search(candidates, embedding_service, redis_client)
                
                # Initialize fallback algorithm
                fallback_success = self._initialize_fallback_algorithm(candidates)
                
                # Update status
                self.initialization_status.update({
                    'optimized_search': optimized_success,
                    'fallback_algorithm': fallback_success,
                    'initialization_time': time.time() - start_time,
                    'candidate_count': len(candidates),
                    'last_initialization': time.time()
                })
                
                success = optimized_success or fallback_success
                
                if success:
                    logger.info(f"Search systems initialized successfully in {self.initialization_status['initialization_time']:.2f}s")
                else:
                    logger.error("Failed to initialize any search systems")
                
                return success
                
            except Exception as e:
                logger.error(f"Search system initialization failed: {e}")
                return False
            finally:
                self.is_initializing = False
    
    def _initialize_optimized_search(self, candidates: Dict[str, Any], embedding_service, redis_client=None) -> bool:
        """Initialize the optimized search system"""
        try:
            logger.info("Initializing optimized search system...")
            success = initialize_optimized_search(candidates, embedding_service, redis_client)
            
            if success:
                logger.info("✅ Optimized search system initialized successfully")
            else:
                logger.warning("❌ Optimized search system initialization failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Optimized search initialization error: {e}")
            return False
    
    def _initialize_fallback_algorithm(self, candidates: Dict[str, Any]) -> bool:
        """Initialize the fallback algorithm"""
        try:
            logger.info("Initializing fallback algorithm...")
            
            # Get the algorithm instance
            algorithm = get_algorithm_instance()
            
            # If it's an enhanced system, try to initialize it with candidates
            if hasattr(algorithm, 'enhanced_system') and algorithm.enhanced_system:
                if hasattr(algorithm.enhanced_system, 'candidates'):
                    algorithm.enhanced_system.candidates = candidates
                    logger.info("Enhanced system candidates updated")
            
            # If it's a fallback algorithm, try to initialize it
            if hasattr(algorithm, 'candidates'):
                algorithm.candidates = candidates
                logger.info("Fallback algorithm candidates updated")
            
            logger.info("✅ Fallback algorithm initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Fallback algorithm initialization error: {e}")
            return False
    
    def get_initialization_status(self) -> Dict[str, Any]:
        """Get the current initialization status"""
        with self.initialization_lock:
            return self.initialization_status.copy()
    
    def is_ready(self) -> bool:
        """Check if any search system is ready"""
        with self.initialization_lock:
            return (self.initialization_status['optimized_search'] or 
                   self.initialization_status['fallback_algorithm'])
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance information about the search systems"""
        status = self.get_initialization_status()
        
        # Get optimized search stats if available
        optimized_stats = {}
        try:
            from app.search.optimized_search_service import get_optimized_search_service
            service = get_optimized_search_service()
            if service:
                optimized_stats = service.get_performance_stats()
        except Exception as e:
            logger.warning(f"Could not get optimized search stats: {e}")
        
        return {
            'initialization_status': status,
            'optimized_search_stats': optimized_stats,
            'is_ready': self.is_ready(),
            'recommendation': self._get_performance_recommendation()
        }
    
    def _get_performance_recommendation(self) -> str:
        """Get performance recommendations based on current status"""
        status = self.initialization_status
        
        if status['optimized_search']:
            return "🚀 Ultra-fast parallel search is active - expect 10-50x faster results!"
        elif status['fallback_algorithm']:
            return "⚡ Fallback algorithm is active - good performance with standard speed"
        else:
            return "⚠️ No search systems are ready - performance may be limited"

# Global initializer instance
_search_initializer = None

def get_search_initializer():
    """Get or create the search initializer instance"""
    global _search_initializer
    if _search_initializer is None:
        _search_initializer = SearchSystemInitializer()
    return _search_initializer

def initialize_search_systems(candidates: Dict[str, Any], embedding_service, redis_client=None) -> bool:
    """Initialize all search systems with candidate data"""
    initializer = get_search_initializer()
    return initializer.initialize_search_systems(candidates, embedding_service, redis_client)

def get_search_status() -> Dict[str, Any]:
    """Get the current search system status"""
    initializer = get_search_initializer()
    return initializer.get_performance_info()
