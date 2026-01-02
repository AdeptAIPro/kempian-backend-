# ✨ UPDATE THIS FILE
from .fast_search import OptimizedSearchSystem, get_optimized_search_system
from .performance import PerformanceMonitor
from .cache import EmbeddingCache

# ✨ ADD THESE NEW IMPORTS
try:
    from ..enhanced_recruitment_search import EnhancedRecruitmentSearchSystem
    ENHANCED_SEARCH_AVAILABLE = True
except ImportError:
    ENHANCED_SEARCH_AVAILABLE = False

__all__ = [
    'OptimizedSearchSystem', 
    'get_optimized_search_system', 
    'PerformanceMonitor', 
    'EmbeddingCache',
    'ENHANCED_SEARCH_AVAILABLE'
]

if ENHANCED_SEARCH_AVAILABLE:
    __all__.append('EnhancedRecruitmentSearchSystem')