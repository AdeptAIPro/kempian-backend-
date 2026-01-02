import hashlib
import pickle
import os
import time
from typing import Any, Optional
from app.simple_logger import get_logger
import logging

logger = get_logger("search")

class EmbeddingCache:
    """
    Persistent cache for embeddings with TTL support
    """
    
    def __init__(self, cache_dir: str = "cache", max_size: int = 1000, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache for fast access
        self.memory_cache = {}
        
        # Load existing cache index
        self.index_file = os.path.join(cache_dir, "cache_index.pkl")
        self.cache_index = self._load_index()
        
        # Clean expired entries
        self._cleanup_expired()
    
    def _load_index(self) -> dict:
        """Load cache index from disk"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_index(self):
        """Save cache index to disk"""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.cache_index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key"""
        return hashlib.md5(f"{model_name}:{text}".encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, (timestamp, _) in self.cache_index.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _remove_entry(self, cache_key: str):
        """Remove cache entry"""
        # Remove from memory cache
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Remove from disk
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        
        # Remove from index
        if cache_key in self.cache_index:
            del self.cache_index[cache_key]
    
    def _manage_cache_size(self):
        """Ensure cache doesn't exceed max size"""
        if len(self.cache_index) <= self.max_size:
            return
        
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: x[1][0]
        )
        
        # Remove oldest entries
        entries_to_remove = len(self.cache_index) - self.max_size + 100  # Remove extra for buffer
        for i in range(min(entries_to_remove, len(sorted_entries))):
            cache_key = sorted_entries[i][0]
            self._remove_entry(cache_key)
        
        logger.info(f"Removed {entries_to_remove} old cache entries")
    
    def get(self, text: str, model_name: str) -> Optional[Any]:
        """Get embedding from cache"""
        cache_key = self._get_cache_key(text, model_name)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        if cache_key in self.cache_index:
            timestamp, _ = self.cache_index[cache_key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                self._remove_entry(cache_key)
                return None
            
            # Load from disk
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        embedding = pickle.load(f)
                    
                    # Add to memory cache
                    self.memory_cache[cache_key] = embedding
                    return embedding
                    
                except Exception as e:
                    logger.error(f"Failed to load cached embedding: {e}")
                    self._remove_entry(cache_key)
        
        return None
    
    def put(self, text: str, model_name: str, embedding: Any):
        """Store embedding in cache"""
        cache_key = self._get_cache_key(text, model_name)
        current_time = time.time()
        
        # Store in memory cache
        self.memory_cache[cache_key] = embedding
        
        # Store on disk
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            
            # Update index
            self.cache_index[cache_key] = (current_time, len(str(embedding)))
            
            # Manage cache size
            self._manage_cache_size()
            
            # Save index periodically
            if len(self.cache_index) % 50 == 0:
                self._save_index()
                
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
    
    def clear(self):
        """Clear all cache"""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache
        for cache_key in list(self.cache_index.keys()):
            self._remove_entry(cache_key)
        
        # Save empty index
        self._save_index()
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_size = sum(size for _, (_, size) in self.cache_index.items())
        
        return {
            'entries': len(self.cache_index),
            'memory_entries': len(self.memory_cache),
            'max_size': self.max_size,
            'total_size_bytes': total_size,
            'ttl_hours': self.ttl_seconds / 3600,
            'cache_dir': self.cache_dir
        }


class MemoryCache:
    """
    Simple in-memory cache for fast access
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
    
    def _evict_lru(self):
        """Remove least recently used items"""
        if len(self.cache) < self.max_size:
            return
        
        # Sort by access time
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        
        # Remove oldest 20% of items
        items_to_remove = max(1, len(sorted_items) // 5)
        for i in range(items_to_remove):
            key = sorted_items[i][0]
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Store item in cache"""
        # Evict if necessary
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)