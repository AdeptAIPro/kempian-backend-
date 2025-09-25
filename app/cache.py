"""
Redis Caching System for High Performance
Handles multi-level caching for search results, user profiles, and analytics
"""
import redis
import json
import hashlib
import os
from typing import Any, Optional, Union, List, Dict
from datetime import datetime, timedelta
from app.simple_logger import get_logger

logger = get_logger("cache")

class CacheManager:
    """High-performance Redis cache manager with multi-level caching"""
    
    def __init__(self):
        # Redis configuration
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Cache configuration (optimized for 2000+ users)
        self.default_ttl = 600   # 10 minutes (increased from 5)
        self.search_ttl = 1200   # 20 minutes (increased from 10)
        self.user_ttl = 3600     # 1 hour (increased from 30 minutes)
        self.analytics_ttl = 7200  # 2 hours (increased from 1 hour)
        
        # In-memory L1 cache for ultra-fast access (increased capacity)
        self.l1_cache = {}
        self.l1_max_size = 5000  # Increased from 1000
        self.l1_ttl = 300  # 5 minutes (increased from 1 minute)
        
        logger.info("Cache manager initialized successfully")
    
    def _generate_key(self, prefix: str, identifier: Union[str, int, Dict]) -> str:
        """Generate consistent cache key"""
        if isinstance(identifier, dict):
            # For complex objects, create hash
            identifier = hashlib.md5(json.dumps(identifier, sort_keys=True).encode()).hexdigest()
        return f"{prefix}:{identifier}"
    
    def _is_l1_valid(self, key: str) -> bool:
        """Check if L1 cache entry is still valid"""
        if key not in self.l1_cache:
            return False
        
        entry = self.l1_cache[key]
        if datetime.now() > entry['expires_at']:
            del self.l1_cache[key]
            return False
        
        return True
    
    def get(self, key: str, use_l1: bool = True) -> Optional[Any]:
        """Get value from cache with L1 and L2 fallback"""
        try:
            # L1 Cache (in-memory) - fastest
            if use_l1 and key in self.l1_cache and self._is_l1_valid(key):
                logger.debug(f"L1 cache hit for key: {key}")
                return self.l1_cache[key]['value']
            
            # L2 Cache (Redis) - fast
            value = self.redis_client.get(key)
            if value:
                try:
                    parsed_value = json.loads(value)
                    # Store in L1 cache for next time
                    if use_l1:
                        self.l1_cache[key] = {
                            'value': parsed_value,
                            'expires_at': datetime.now() + timedelta(seconds=self.l1_ttl)
                        }
                        # Clean up L1 cache if it gets too large
                        if len(self.l1_cache) > self.l1_max_size:
                            self._cleanup_l1_cache()
                    
                    logger.debug(f"L2 cache hit for key: {key}")
                    return parsed_value
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse cached value for key: {key}")
                    return None
            
            logger.debug(f"Cache miss for key: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, use_l1: bool = True) -> bool:
        """Set value in cache with L1 and L2 storage"""
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            # Store in L1 cache (in-memory)
            if use_l1:
                self.l1_cache[key] = {
                    'value': value,
                    'expires_at': datetime.now() + timedelta(seconds=min(ttl, self.l1_ttl))
                }
            
            # Store in L2 cache (Redis)
            serialized_value = json.dumps(value, default=str)
            self.redis_client.setex(key, ttl, serialized_value)
            
            logger.debug(f"Cache set for key: {key}, ttl: {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from both L1 and L2 cache"""
        try:
            # Remove from L1 cache
            if key in self.l1_cache:
                del self.l1_cache[key]
            
            # Remove from L2 cache
            result = self.redis_client.delete(key)
            
            logger.debug(f"Cache delete for key: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                result = self.redis_client.delete(*keys)
                logger.info(f"Deleted {result} keys matching pattern: {pattern}")
                return result
            return 0
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {str(e)}")
            return 0
    
    def _cleanup_l1_cache(self):
        """Clean up expired entries from L1 cache"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.l1_cache.items()
            if now > entry['expires_at']
        ]
        for key in expired_keys:
            del self.l1_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            redis_info = self.redis_client.info()
            return {
                'l1_cache_size': len(self.l1_cache),
                'l1_max_size': self.l1_max_size,
                'redis_used_memory': redis_info.get('used_memory_human', 'N/A'),
                'redis_connected_clients': redis_info.get('connected_clients', 0),
                'redis_keyspace_hits': redis_info.get('keyspace_hits', 0),
                'redis_keyspace_misses': redis_info.get('keyspace_misses', 0),
                'redis_hit_rate': self._calculate_hit_rate(redis_info)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}
    
    def _calculate_hit_rate(self, redis_info: Dict) -> float:
        """Calculate Redis hit rate"""
        hits = redis_info.get('keyspace_hits', 0)
        misses = redis_info.get('keyspace_misses', 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

# Specialized cache methods for different data types
class SearchCache:
    """Specialized caching for search results"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.ttl = 600  # 10 minutes
    
    def get_search_results(self, query: str, filters: Dict = None) -> Optional[List[Dict]]:
        """Get cached search results"""
        key = self.cache._generate_key("search", {"query": query, "filters": filters or {}})
        return self.cache.get(key)
    
    def set_search_results(self, query: str, results: List[Dict], filters: Dict = None) -> bool:
        """Cache search results"""
        key = self.cache._generate_key("search", {"query": query, "filters": filters or {}})
        return self.cache.set(key, results, self.ttl)
    
    def invalidate_search_cache(self):
        """Invalidate all search cache"""
        return self.cache.delete_pattern("search:*")

class UserCache:
    """Specialized caching for user profiles and data"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.ttl = 1800  # 30 minutes
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get cached user profile"""
        key = self.cache._generate_key("user_profile", user_id)
        return self.cache.get(key)
    
    def set_user_profile(self, user_id: str, profile: Dict) -> bool:
        """Cache user profile"""
        key = self.cache._generate_key("user_profile", user_id)
        return self.cache.set(key, profile, self.ttl)
    
    def get_user_analytics(self, user_id: str) -> Optional[Dict]:
        """Get cached user analytics"""
        key = self.cache._generate_key("user_analytics", user_id)
        return self.cache.get(key)
    
    def set_user_analytics(self, user_id: str, analytics: Dict) -> bool:
        """Cache user analytics"""
        key = self.cache._generate_key("user_analytics", user_id)
        return self.cache.set(key, analytics, self.ttl)
    
    def invalidate_user_cache(self, user_id: str):
        """Invalidate user-specific cache"""
        patterns = [
            f"user_profile:{user_id}",
            f"user_analytics:{user_id}"
        ]
        for pattern in patterns:
            self.cache.delete_pattern(pattern)

class AnalyticsCache:
    """Specialized caching for analytics and KPIs"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.ttl = 3600  # 1 hour
    
    def get_kpis(self, user_id: str = None) -> Optional[Dict]:
        """Get cached KPIs"""
        key = self.cache._generate_key("kpis", user_id or "global")
        return self.cache.get(key)
    
    def set_kpis(self, kpis: Dict, user_id: str = None) -> bool:
        """Cache KPIs"""
        key = self.cache._generate_key("kpis", user_id or "global")
        return self.cache.set(key, kpis, self.ttl)
    
    def get_candidate_stats(self, filters: Dict = None) -> Optional[Dict]:
        """Get cached candidate statistics"""
        key = self.cache._generate_key("candidate_stats", filters or {})
        return self.cache.get(key)
    
    def set_candidate_stats(self, stats: Dict, filters: Dict = None) -> bool:
        """Cache candidate statistics"""
        key = self.cache._generate_key("candidate_stats", filters or {})
        return self.cache.set(key, stats, self.ttl)

# Global cache instances
cache_manager = CacheManager()
search_cache = SearchCache(cache_manager)
user_cache = UserCache(cache_manager)
analytics_cache = AnalyticsCache(cache_manager)

# Cache warming functions
def warm_search_cache():
    """Pre-warm search cache with popular queries"""
    popular_queries = [
        "software engineer",
        "data scientist", 
        "product manager",
        "full stack developer",
        "machine learning engineer",
        "devops engineer",
        "frontend developer",
        "backend developer"
    ]
    
    logger.info("Starting search cache warming...")
    for query in popular_queries:
        # This would typically call the actual search function
        # For now, we'll just log the warming process
        logger.debug(f"Warming cache for query: {query}")
    
    logger.info("Search cache warming completed")

def warm_user_cache():
    """Pre-warm user cache with active users"""
    logger.info("Starting user cache warming...")
    # This would typically get active users and pre-load their profiles
    logger.info("User cache warming completed")

# Initialize cache warming
if __name__ == "__main__":
    warm_search_cache()
    warm_user_cache()
