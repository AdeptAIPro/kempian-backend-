"""
Async utilities for AdeptAI application
Provides async processing capabilities for better performance
"""

import asyncio
import logging
from typing import Any, Callable, List, Optional, Dict, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps
import time

logger = logging.getLogger(__name__)


class AsyncProcessor:
    """Async processor for handling CPU and I/O intensive tasks"""
    
    def __init__(self, max_workers: int = 4, use_processes: bool = False):
        """
        Initialize async processor
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Whether to use processes instead of threads for CPU-intensive tasks
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        self._executor = None
    
    @property
    def executor(self):
        """Get or create executor"""
        if self._executor is None:
            self._executor = self.executor_class(max_workers=self.max_workers)
        return self._executor
    
    async def run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """Run function in executor asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def run_batch(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """Run multiple tasks concurrently"""
        coroutines = [self.run_in_executor(task, *args, **kwargs) for task in tasks]
        return await asyncio.gather(*coroutines, return_exceptions=True)
    
    def close(self):
        """Close the executor"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


class AsyncCache:
    """Async cache with TTL support"""
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize async cache
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() < entry['expires_at']:
                    return entry['value']
                else:
                    del self.cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        async with self._lock:
            ttl = ttl or self.default_ttl
            self.cache[key] = {
                'value': value,
                'expires_at': time.time() + ttl
            }
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()


class AsyncRateLimiter:
    """Async rate limiter with sliding window"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
    
    async def is_allowed(self) -> bool:
        """Check if request is allowed"""
        async with self._lock:
            now = time.time()
            # Remove old requests outside the window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.window_seconds]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    async def wait_if_needed(self) -> None:
        """Wait if rate limit is exceeded"""
        while not await self.is_allowed():
            await asyncio.sleep(0.1)


def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for async retry logic"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
            
            raise last_exception
        return wrapper
    return decorator


def async_timing(func):
    """Decorator for async function timing"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper


class AsyncBatchProcessor:
    """Process items in batches asynchronously"""
    
    def __init__(self, batch_size: int = 10, max_concurrent_batches: int = 3):
        """
        Initialize batch processor
        
        Args:
            batch_size: Number of items per batch
            max_concurrent_batches: Maximum concurrent batches
        """
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    async def process_batches(self, items: List[Any], process_func: Callable) -> List[Any]:
        """Process items in batches"""
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]
        
        async def process_batch(batch):
            async with self.semaphore:
                return await process_func(batch)
        
        batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results


# Global instances
async_processor = AsyncProcessor()
async_cache = AsyncCache()
async_rate_limiter = AsyncRateLimiter(max_requests=100, window_seconds=60)
async_batch_processor = AsyncBatchProcessor()


async def cleanup_async_resources():
    """Cleanup async resources"""
    async_processor.close()
    logger.info("Async resources cleaned up")
