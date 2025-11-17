import os
import pickle
import hashlib
import time
from typing import Any, Optional
from app.simple_logger import get_logger
import logging

logger = get_logger("search")

class EmbeddingCache:
    """Persistent cache for embeddings with memory management"""
    
    def __init__(self, cache_dir: str = "cache/embeddings", max_size: int = 10000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.memory_cache = {}
        self.access_times = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Initialized embedding cache at {cache_dir}")
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model"""
        return hashlib.md5(f"{model_name}:{text}".encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _manage_memory_cache_size(self):
        """Ensure memory cache doesn't exceed max size"""
        if len(self.memory_cache) <= self.max_size:
            return
        
        # Sort by access time (oldest first)
        sorted_items = sorted(
            self.access_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest 20% of items
        items_to_remove = len(sorted_items) // 5
        for i in range(items_to_remove):
            key = sorted_items[i][0]
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        logger.debug(f"Removed {items_to_remove} old cache entries")
    
    def get(self, text: str, model_name: str) -> Optional[Any]:
        """Get embedding from cache"""
        cache_key = self._get_cache_key(text, model_name)
        current_time = time.time()
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.access_times[cache_key] = current_time
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                
                # Add to memory cache
                self.memory_cache[cache_key] = embedding
                self.access_times[cache_key] = current_time
                
                # Manage cache size
                self._manage_memory_cache_size()
                
                return embedding
                
            except Exception as e:
                logger.error(f"Failed to load cached embedding: {e}")
                # Remove corrupted file
                try:
                    os.remove(cache_path)
                except:
                    pass
        
        return None
    
    def put(self, text: str, model_name: str, embedding: Any):
        """Store embedding in cache"""
        cache_key = self._get_cache_key(text, model_name)
        current_time = time.time()
        
        # Store in memory
        self.memory_cache[cache_key] = embedding
        self.access_times[cache_key] = current_time
        
        # Store on disk
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.error(f"Failed to cache embedding to disk: {e}")
        
        # Manage memory cache size
        self._manage_memory_cache_size()
    
    def clear(self):
        """Clear all cache"""
        # Clear memory cache
        self.memory_cache.clear()
        self.access_times.clear()
        
        # Clear disk cache
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}")
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        disk_files = 0
        total_disk_size = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    disk_files += 1
                    filepath = os.path.join(self.cache_dir, filename)
                    total_disk_size += os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
        
        return {
            'memory_entries': len(self.memory_cache),
            'disk_entries': disk_files,
            'max_size': self.max_size,
            'total_disk_size_mb': round(total_disk_size / (1024 * 1024), 2),
            'cache_dir': self.cache_dir
        }
    
    def cleanup_old_files(self, max_age_days: int = 7):
        """Remove old cache files"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        removed_count = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(self.cache_dir, filename)
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        removed_count += 1
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cache files")
        
        return removed_count