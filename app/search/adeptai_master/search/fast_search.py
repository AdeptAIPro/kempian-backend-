"""
Fast Search Module - Wrapper for OptimizedSearchSystem
Provides compatibility layer for search subsystem
"""

import sys
import os

# Add parent directory to path to import search_system
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from search_system import OptimizedSearchSystem
except ImportError:
    # Fallback if import fails
    OptimizedSearchSystem = None


def get_optimized_search_system():
    """Get or create an instance of OptimizedSearchSystem"""
    if OptimizedSearchSystem is None:
        raise ImportError("OptimizedSearchSystem not available")
    return OptimizedSearchSystem()


__all__ = ['OptimizedSearchSystem', 'get_optimized_search_system']

