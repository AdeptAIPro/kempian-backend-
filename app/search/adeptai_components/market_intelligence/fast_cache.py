"""
Fast Caching System for Market Intelligence

High-performance caching with Redis and in-memory fallback
"""

from __future__ import annotations

import json
import time
import hashlib
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory cache")


class FastCache:
    """High-performance caching system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", default_ttl: int = 300):
        self.default_ttl = default_ttl
        self.memory_cache = {}
        self.redis_client = None
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()  # Test connection
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory cache")
                self.redis_client = None
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key"""
        content = json.dumps(data, sort_keys=True)
        hash_key = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}:{hash_key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # Memory cache
                if key in self.memory_cache:
                    item = self.memory_cache[key]
                    if time.time() < item["expires_at"]:
                        return item["value"]
                    else:
                        del self.memory_cache[key]
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached value"""
        try:
            ttl = ttl or self.default_ttl
            
            if self.redis_client:
                return self.redis_client.setex(key, ttl, json.dumps(value))
            else:
                # Memory cache
                self.memory_cache[key] = {
                    "value": value,
                    "expires_at": time.time() + ttl
                }
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def get_or_set(self, key: str, factory_func, ttl: Optional[int] = None) -> Any:
        """Get from cache or set using factory function"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Generate value using factory function
        if asyncio.iscoroutinefunction(factory_func):
            value = await factory_func()
        else:
            value = factory_func()
        
        await self.set(key, value, ttl)
        return value
    
    async def batch_get(self, keys: list) -> Dict[str, Any]:
        """Get multiple values at once"""
        results = {}
        
        if self.redis_client:
            try:
                values = self.redis_client.mget(keys)
                for key, value in zip(keys, values):
                    if value:
                        results[key] = json.loads(value)
            except Exception as e:
                logger.error(f"Batch get error: {e}")
        else:
            # Memory cache
            for key in keys:
                if key in self.memory_cache:
                    item = self.memory_cache[key]
                    if time.time() < item["expires_at"]:
                        results[key] = item["value"]
                    else:
                        del self.memory_cache[key]
        
        return results
    
    async def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values at once"""
        try:
            ttl = ttl or self.default_ttl
            
            if self.redis_client:
                pipe = self.redis_client.pipeline()
                for key, value in items.items():
                    pipe.setex(key, ttl, json.dumps(value))
                pipe.execute()
                return True
            else:
                # Memory cache
                for key, value in items.items():
                    self.memory_cache[key] = {
                        "value": value,
                        "expires_at": time.time() + ttl
                    }
                return True
        except Exception as e:
            logger.error(f"Batch set error: {e}")
            return False
    
    def clear(self, pattern: str = "*") -> bool:
        """Clear cache"""
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                if pattern == "*":
                    self.memory_cache.clear()
                else:
                    # Simple pattern matching for memory cache
                    keys_to_delete = [k for k in self.memory_cache.keys() if pattern.replace("*", "") in k]
                    for key in keys_to_delete:
                        del self.memory_cache[key]
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    "type": "redis",
                    "connected": True,
                    "memory_used": info.get("used_memory_human", "N/A"),
                    "keys": info.get("db0", {}).get("keys", 0),
                    "hit_rate": "N/A"  # Would need to track this separately
                }
            else:
                return {
                    "type": "memory",
                    "connected": True,
                    "keys": len(self.memory_cache),
                    "memory_used": f"{len(str(self.memory_cache))} bytes"
                }
        except Exception as e:
            return {
                "type": "error",
                "connected": False,
                "error": str(e)
            }


# Global fast cache instance
fast_cache = FastCache()
