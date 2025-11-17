import time
from collections import defaultdict
from app.simple_logger import get_logger
from datetime import datetime, timedelta
import logging

logger = get_logger(__name__.split('.')[-1])

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.max_requests = 10  # Max requests per window
        self.window_seconds = 60  # 1 minute window
    
    def is_allowed(self, user_id: int, request_type: str = "search") -> bool:
        """
        Check if user is allowed to make a request
        Returns True if allowed, False if rate limited
        """
        now = time.time()
        key = f"{user_id}_{request_type}"
        
        # Clean old requests outside the window
        self.requests[key] = [req_time for req_time in self.requests[key] 
                             if now - req_time < self.window_seconds]
        
        # Check if user has exceeded the limit
        if len(self.requests[key]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user {user_id}, request type: {request_type}")
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True
    
    def get_remaining_requests(self, user_id: int, request_type: str = "search") -> int:
        """Get remaining requests for user in current window"""
        now = time.time()
        key = f"{user_id}_{request_type}"
        
        # Clean old requests
        self.requests[key] = [req_time for req_time in self.requests[key] 
                             if now - req_time < self.window_seconds]
        
        return max(0, self.max_requests - len(self.requests[key]))
    
    def get_reset_time(self, user_id: int, request_type: str = "search") -> datetime:
        """Get time when rate limit resets for user"""
        key = f"{user_id}_{request_type}"
        if not self.requests[key]:
            return datetime.now()
        
        # Find the oldest request in the current window
        oldest_request = min(self.requests[key])
        return datetime.fromtimestamp(oldest_request + self.window_seconds)

# Global rate limiter instance
rate_limiter = RateLimiter()

def check_rate_limit(user_id: int, request_type: str = "search") -> tuple[bool, dict]:
    """
    Check if user is rate limited
    Returns (is_allowed, rate_limit_info)
    """
    is_allowed = rate_limiter.is_allowed(user_id, request_type)
    
    if is_allowed:
        remaining = rate_limiter.get_remaining_requests(user_id, request_type)
        reset_time = rate_limiter.get_reset_time(user_id, request_type)
        
        return True, {
            'remaining_requests': remaining,
            'reset_time': reset_time.isoformat(),
            'limit': rate_limiter.max_requests,
            'window_seconds': rate_limiter.window_seconds
        }
    else:
        reset_time = rate_limiter.get_reset_time(user_id, request_type)
        
        return False, {
            'error': 'Rate limit exceeded',
            'reset_time': reset_time.isoformat(),
            'limit': rate_limiter.max_requests,
            'window_seconds': rate_limiter.window_seconds
        }
