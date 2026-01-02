# search/performance.py

import logging
from collections import deque, defaultdict, Counter
from app.simple_logger import get_logger
import time
from typing import List, Dict, Any, Tuple # Ensure Tuple is imported here!

logger = get_logger("search")

class PerformanceMonitor:
    """
    Monitors the performance of the search system.
    Tracks search times, cache hit rates, query statistics, and resource usage.
    """

    def __init__(self, history_size: int = 1000):
        self.search_times = deque(maxlen=history_size)  # Store recent search durations
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_searches = 0
        self.query_log = deque(maxlen=history_size)  # Store recent queries
        self.query_frequency = defaultdict(int)  # For tracking frequent queries
        self.resource_usage_log = deque(maxlen=history_size) # For CPU/Memory (if integrated)
        self.indexing_times = deque(maxlen=history_size) # For indexing performance
        self.total_indexing_operations = 0
        self.last_reset_time = time.time()
        logger.info("PerformanceMonitor initialized.")

    def record_search(self, duration: float, query: str, cache_hit: bool):
        """Record details of a single search operation."""
        self.search_times.append(duration)
        self.query_log.append((query, duration, time.time()))
        self.query_frequency[query.lower()] += 1
        self.total_searches += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        logger.debug(f"Search recorded: Duration={duration:.4f}s, Query='{query}', CacheHit={cache_hit}")

    def record_indexing(self, duration: float, operation: str = "full_index"):
        """Record details of an indexing operation."""
        self.indexing_times.append(duration)
        self.total_indexing_operations += 1
        logger.info(f"Indexing operation recorded: Type='{operation}', Duration={duration:.4f}s")

    def get_average_search_time(self) -> float:
        """Calculate the average search time over recorded history."""
        if not self.search_times:
            return 0.0
        return sum(self.search_times) / len(self.search_times)

    def get_cache_hit_rate(self) -> float:
        """Calculate the cache hit rate."""
        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations == 0:
            return 0.0
        return (self.cache_hits / total_cache_operations) * 100

    def get_total_searches(self) -> int:
        """Get the total number of searches recorded."""
        return self.total_searches

    def get_query_frequency(self, query: str) -> int:
        """Get the frequency of a specific query."""
        return self.query_frequency[query.lower()]

    def get_most_frequent_queries(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get the top N most frequent queries."""
        return Counter(self.query_frequency).most_common(top_n)

    def get_average_indexing_time(self) -> float:
        """Calculate the average indexing time over recorded history."""
        if not self.indexing_times:
            return 0.0
        return sum(self.indexing_times) / len(self.indexing_times)

    def get_total_indexing_operations(self) -> int:
        """Get the total number of indexing operations recorded."""
        return self.total_indexing_operations

    def reset_metrics(self):
        """Reset all performance metrics."""
        self.search_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_searches = 0
        self.query_log.clear()
        self.query_frequency.clear()
        self.resource_usage_log.clear()
        self.indexing_times.clear()
        self.total_indexing_operations = 0
        self.last_reset_time = time.time()
        logger.info("PerformanceMonitor metrics have been reset.")

    def get_current_status(self) -> Dict[str, Any]:
        """Return a dictionary of current performance metrics."""
        return {
            "total_searches": self.get_total_searches(),
            "average_search_time_ms": self.get_average_search_time() * 1000,
            "cache_hit_rate_percent": self.get_cache_hit_rate(),
            "most_frequent_queries": self.get_most_frequent_queries(),
            "total_indexing_operations": self.get_total_indexing_operations(),
            "average_indexing_time_ms": self.get_average_indexing_time() * 1000,
            "monitor_uptime_seconds": time.time() - self.last_reset_time
        }

    # Placeholder for resource usage monitoring (requires OS-specific libraries like psutil)
    def record_resource_usage(self, cpu_percent: float, memory_percent: float):
        """Record CPU and memory usage."""
        self.resource_usage_log.append({"timestamp": time.time(), "cpu": cpu_percent, "memory": memory_percent})
        logger.debug(f"Resource usage recorded: CPU={cpu_percent}%, Memory={memory_percent}%")

    def get_top_queries(self, top_n: int = 10) -> List[Tuple[str, int]]: # Corrected Tuple import
        """Get the top N most frequent queries from the log."""
        query_counts = Counter(item[0] for item in self.query_log)
        return query_counts.most_common(top_n)