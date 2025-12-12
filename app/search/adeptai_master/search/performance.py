"""
Performance Monitor Module - Tracks and monitors search system performance
"""

import time
import logging
from typing import Dict, Optional, List
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for search operations"""
    total_queries: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    query_times: List[float] = field(default_factory=list)
    

class PerformanceMonitor:
    """Monitor and track search system performance"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.query_history = []
        self.max_history = 1000
        
    def record_query(self, query: str, response_time: float, cache_hit: bool = False, error: bool = False):
        """Record a query and its performance metrics"""
        self.metrics.total_queries += 1
        
        # Update response time statistics
        self.metrics.query_times.append(response_time)
        if len(self.metrics.query_times) > self.max_history:
            self.metrics.query_times.pop(0)
        
        # Update min/max
        if response_time < self.metrics.min_response_time:
            self.metrics.min_response_time = response_time
        if response_time > self.metrics.max_response_time:
            self.metrics.max_response_time = response_time
        
        # Update average
        total_time = sum(self.metrics.query_times)
        self.metrics.avg_response_time = total_time / len(self.metrics.query_times)
        
        # Update cache stats
        if cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
        
        # Update error count
        if error:
            self.metrics.errors += 1
        
        # Store query history
        self.query_history.append({
            'query': query,
            'response_time': response_time,
            'cache_hit': cache_hit,
            'timestamp': time.time()
        })
        
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        cache_hit_rate = 0.0
        if self.metrics.total_queries > 0:
            cache_hit_rate = (self.metrics.cache_hits / self.metrics.total_queries) * 100
        
        error_rate = 0.0
        if self.metrics.total_queries > 0:
            error_rate = (self.metrics.errors / self.metrics.total_queries) * 100
        
        return {
            'total_queries': self.metrics.total_queries,
            'avg_response_time_ms': round(self.metrics.avg_response_time * 1000, 2),
            'min_response_time_ms': round(self.metrics.min_response_time * 1000, 2) if self.metrics.min_response_time != float('inf') else 0,
            'max_response_time_ms': round(self.metrics.max_response_time * 1000, 2),
            'cache_hits': self.metrics.cache_hits,
            'cache_misses': self.metrics.cache_misses,
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'errors': self.metrics.errors,
            'error_rate_percent': round(error_rate, 2)
        }
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = PerformanceMetrics()
        self.query_history.clear()
        logger.info("Performance metrics reset")
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent query history"""
        return self.query_history[-limit:]


__all__ = ['PerformanceMonitor', 'PerformanceMetrics']

