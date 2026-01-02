"""
Fast LLM Service with Connection Pooling and Optimized Performance

High-speed LLM operations with:
- Connection pooling
- Request batching
- Smart caching
- Fast fallbacks
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import aiohttp
from .fast_cache import fast_cache

logger = logging.getLogger(__name__)


@dataclass
class FastLLMRequest:
    """Optimized LLM request"""
    prompt: str
    system_prompt: str
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    cache_key: Optional[str] = None


@dataclass
class FastLLMResponse:
    """Optimized LLM response"""
    content: str
    model: str
    tokens_used: int
    cost: float
    processing_time: float
    cached: bool = False


class FastLLMService:
    """High-performance LLM service with optimizations"""
    
    def __init__(self):
        self.session = None
        self.connection_pool = None
        self.openai_api_key = None
        self.anthropic_api_key = None
        self._setup_apis()
    
    def _setup_apis(self):
        """Setup API keys"""
        import os
        from .claude_config import ClaudeConfig
        from .openai_config import OpenAIConfig
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or OpenAIConfig.API_KEY
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or ClaudeConfig.API_KEY
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=30,  # Total timeout
                connect=10,  # Connection timeout
                sock_read=20  # Socket read timeout
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "AdeptAI-MarketIntelligence/1.0"
                }
            )
        
        return self.session
    
    async def _call_openai_fast(self, request: FastLLMRequest) -> FastLLMResponse:
        """Fast OpenAI API call with optimizations"""
        if not self.openai_api_key:
            return self._fallback_response(request, "openai")
        
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            payload = {
                "model": request.model,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.prompt}
                ],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": False  # Disable streaming for speed
            }
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    usage = data.get("usage", {})
                    tokens_used = usage.get("total_tokens", 0)
                    cost = self._calculate_cost("openai", tokens_used)
                    
                    return FastLLMResponse(
                        content=content,
                        model=request.model,
                        tokens_used=tokens_used,
                        cost=cost,
                        processing_time=time.time() - start_time
                    )
                else:
                    logger.error(f"OpenAI API error: {response.status}")
                    return self._fallback_response(request, "openai")
                    
        except Exception as e:
            logger.error(f"OpenAI API call error: {e}")
            return self._fallback_response(request, "openai")
    
    async def _call_anthropic_fast(self, request: FastLLMRequest) -> FastLLMResponse:
        """Fast Anthropic API call with optimizations"""
        if not self.anthropic_api_key:
            return self._fallback_response(request, "anthropic")
        
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            # Use Claude Sonnet 4 model
            model = request.model if request.model else "claude-sonnet-4-20250514"
            
            payload = {
                "model": model,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "system": request.system_prompt,
                "messages": [{"role": "user", "content": request.prompt}]
            }
            
            headers = {
                "x-api-key": self.anthropic_api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["content"][0]["text"]
                    
                    usage = data.get("usage", {})
                    tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    cost = self._calculate_cost("anthropic", tokens_used)
                    
                    return FastLLMResponse(
                        content=content,
                        model=request.model,
                        tokens_used=tokens_used,
                        cost=cost,
                        processing_time=time.time() - start_time
                    )
                else:
                    logger.error(f"Anthropic API error: {response.status}")
                    return self._fallback_response(request, "anthropic")
                    
        except Exception as e:
            logger.error(f"Anthropic API call error: {e}")
            return self._fallback_response(request, "anthropic")
    
    def _calculate_cost(self, provider: str, tokens: int) -> float:
        """Calculate cost based on provider and tokens"""
        if provider == "openai":
            return (tokens / 1_000_000) * 0.15  # GPT-4o-mini pricing
        elif provider == "anthropic":
            return (tokens / 1_000_000) * 3.00  # Claude Sonnet 4 pricing
        return 0.0
    
    def _fallback_response(self, request: FastLLMRequest, provider: str) -> FastLLMResponse:
        """Fast fallback response when API is unavailable"""
        # Pre-generated responses for common patterns
        if "behavior" in request.prompt.lower():
            content = json.dumps({
                "behavior_analysis": {
                    "job_switch_probability": 0.65,
                    "salary_expectations": "moderate",
                    "remote_preference": 0.8,
                    "confidence": 0.75
                },
                "model": f"{provider}-fallback"
            })
        elif "market" in request.prompt.lower():
            content = json.dumps({
                "market_analysis": {
                    "trend_direction": "positive",
                    "confidence": 0.7,
                    "key_factors": ["salary_inflation", "skill_demand"]
                },
                "model": f"{provider}-fallback"
            })
        else:
            content = f"Fast fallback response from {provider}: {request.prompt[:100]}..."
        
        return FastLLMResponse(
            content=content,
            model=f"{request.model}-fallback",
            tokens_used=len(content) // 4,  # Rough token estimate
            cost=0.0,
            processing_time=0.01
        )
    
    async def generate_fast(self, request: FastLLMRequest) -> FastLLMResponse:
        """Generate response with caching and optimizations"""
        # Check cache first
        if request.cache_key:
            cached_response = await fast_cache.get(request.cache_key)
            if cached_response:
                return FastLLMResponse(**cached_response)
        
        # Generate response
        if "gpt" in request.model.lower():
            response = await self._call_openai_fast(request)
        elif "claude" in request.model.lower():
            response = await self._call_anthropic_fast(request)
        else:
            response = self._fallback_response(request, "unknown")
        
        # Cache response
        if request.cache_key and not response.cached:
            await fast_cache.set(
                request.cache_key,
                {
                    "content": response.content,
                    "model": response.model,
                    "tokens_used": response.tokens_used,
                    "cost": response.cost,
                    "processing_time": response.processing_time,
                    "cached": True
                },
                ttl=3600  # 1 hour cache
            )
        
        return response
    
    async def batch_generate(self, requests: List[FastLLMRequest]) -> List[FastLLMResponse]:
        """Batch generate multiple responses efficiently"""
        # Check cache for all requests first
        cache_keys = [req.cache_key for req in requests if req.cache_key]
        cached_responses = {}
        
        if cache_keys:
            cached_data = await fast_cache.batch_get(cache_keys)
            cached_responses = cached_data
        
        # Process requests
        tasks = []
        for i, request in enumerate(requests):
            if request.cache_key and request.cache_key in cached_responses:
                # Use cached response
                cached_data = cached_responses[request.cache_key]
                tasks.append(asyncio.create_task(
                    asyncio.sleep(0.001)  # Minimal delay
                ))
            else:
                # Generate new response
                tasks.append(self.generate_fast(request))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch generation error: {result}")
                responses.append(self._fallback_response(requests[i], "error"))
            elif isinstance(result, asyncio.Task) and result.done():
                # This was a cached response
                cached_data = cached_responses[requests[i].cache_key]
                responses.append(FastLLMResponse(**cached_data))
            else:
                responses.append(result)
        
        return responses
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "session_active": self.session is not None and not self.session.closed,
            "openai_available": bool(self.openai_api_key),
            "anthropic_available": bool(self.anthropic_api_key),
            "cache_stats": fast_cache.get_stats()
        }


# Global fast LLM service
fast_llm_service = FastLLMService()
