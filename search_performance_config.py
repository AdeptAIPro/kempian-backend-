#!/usr/bin/env python3
"""
Search performance configuration
"""

# Performance settings
PERFORMANCE_CONFIG = {
    # Maximum number of candidates to process for accuracy enhancement
    'MAX_CANDIDATES_FOR_ENHANCEMENT': 200,
    
    # Maximum number of candidates to process for initial search
    'MAX_CANDIDATES_FOR_SEARCH': 1000,
    
    # Enable caching for embeddings
    'ENABLE_EMBEDDING_CACHE': True,
    
    # Cache size limit
    'MAX_CACHE_SIZE': 1000,
    
    # Enable progress logging
    'ENABLE_PROGRESS_LOGGING': True,
    
    # Progress logging interval (every N candidates)
    'PROGRESS_LOG_INTERVAL': 50,
    
    # Use fast embedding model only
    'USE_FAST_EMBEDDING_ONLY': True,
    
    # Skip complex similarity calculations for better performance
    'SKIP_COMPLEX_SIMILARITY': True
}

def get_performance_config():
    """Get performance configuration"""
    return PERFORMANCE_CONFIG.copy()

def update_performance_config(**kwargs):
    """Update performance configuration"""
    global PERFORMANCE_CONFIG
    PERFORMANCE_CONFIG.update(kwargs)
    return PERFORMANCE_CONFIG
