# ✨ UPDATE THIS FILE
try:
    from .fast_search import OptimizedSearchSystem, get_optimized_search_system
    FAST_SEARCH_AVAILABLE = True
except ImportError:
    OptimizedSearchSystem = None
    get_optimized_search_system = None
    FAST_SEARCH_AVAILABLE = False

try:
    from .performance import PerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PerformanceMonitor = None
    PERFORMANCE_MONITOR_AVAILABLE = False

try:
    from .cache import EmbeddingCache
    CACHE_AVAILABLE = True
except ImportError:
    EmbeddingCache = None
    CACHE_AVAILABLE = False

# ✨ ADD THESE NEW IMPORTS
try:
    from ..enhanced_recruitment_search import EnhancedRecruitmentSearchSystem
    ENHANCED_SEARCH_AVAILABLE = True
except ImportError:
    ENHANCED_SEARCH_AVAILABLE = False
    EnhancedRecruitmentSearchSystem = None

__all__ = [
    'FAST_SEARCH_AVAILABLE',
    'PERFORMANCE_MONITOR_AVAILABLE',
    'CACHE_AVAILABLE',
    'ENHANCED_SEARCH_AVAILABLE'
]

if FAST_SEARCH_AVAILABLE:
    __all__.extend(['OptimizedSearchSystem', 'get_optimized_search_system'])

if PERFORMANCE_MONITOR_AVAILABLE:
    __all__.append('PerformanceMonitor')

if CACHE_AVAILABLE:
    __all__.append('EmbeddingCache')

if ENHANCED_SEARCH_AVAILABLE:
    __all__.append('EnhancedRecruitmentSearchSystem')