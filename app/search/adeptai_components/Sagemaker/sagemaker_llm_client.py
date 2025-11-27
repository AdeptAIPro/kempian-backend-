"""
SageMaker LLM Client
Unified client for interacting with SageMaker-hosted LLM endpoints
Supports multiple models and use cases with automatic routing and fallback
"""

import boto3
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available SageMaker model types"""
    QUERY_ENHANCER = "query_enhancer"
    BEHAVIORAL_ANALYZER = "behavioral_analyzer"
    MARKET_INTELLIGENCE = "market_intelligence"
    JOB_PARSER = "job_parser"
    EXPLANATION_GENERATOR = "explanation_generator"
    RESUME_SUMMARIZER = "resume_summarizer"
    QUESTION_GENERATOR = "question_generator"
    CANDIDATE_MATCHER = "candidate_matcher"


@dataclass
class LLMRequest:
    """Standardized LLM request structure"""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Standardized LLM response structure"""
    content: str
    model_used: str
    tokens_used: int
    processing_time: float
    confidence_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class SageMakerLLMClient:
    """
    Unified client for SageMaker LLM endpoints
    Supports multiple models with automatic routing, caching, and fallback
    """
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize SageMaker LLM Client
        
        Args:
            region_name: AWS region for SageMaker endpoints
            cache_enabled: Enable response caching
            cache_ttl: Cache time-to-live in seconds
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.region_name = region_name
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize SageMaker runtime client
        self.sagemaker_runtime = boto3.client(
            'sagemaker-runtime',
            region_name=region_name
        )
        
        # Initialize response cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Load endpoint configurations from environment
        self.endpoints = self._load_endpoint_configs()
        
        # Usage statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_latency": 0.0
        }
    
    def _load_endpoint_configs(self) -> Dict[ModelType, Optional[str]]:
        """Load endpoint names from environment variables"""
        return {
            ModelType.QUERY_ENHANCER: os.getenv(
                'SAGEMAKER_QUERY_ENHANCER_ENDPOINT',
                'adeptai-query-enhancer-v1'
            ),
            ModelType.BEHAVIORAL_ANALYZER: os.getenv(
                'SAGEMAKER_BEHAVIORAL_ENDPOINT',
                'adeptai-behavioral-analyzer-v1'
            ),
            ModelType.MARKET_INTELLIGENCE: os.getenv(
                'SAGEMAKER_MARKET_INTEL_ENDPOINT',
                'adeptai-market-intelligence-v1'
            ),
            ModelType.JOB_PARSER: os.getenv(
                'SAGEMAKER_JOB_PARSER_ENDPOINT',
                'adeptai-job-parser-v1'
            ),
            ModelType.EXPLANATION_GENERATOR: os.getenv(
                'SAGEMAKER_EXPLANATION_ENDPOINT',
                'adeptai-explanation-generator-v1'
            ),
            ModelType.RESUME_SUMMARIZER: os.getenv(
                'SAGEMAKER_RESUME_SUMMARIZER_ENDPOINT',
                'adeptai-resume-summarizer-v1'
            ),
            ModelType.QUESTION_GENERATOR: os.getenv(
                'SAGEMAKER_QUESTION_GENERATOR_ENDPOINT',
                'adeptai-question-generator-v1'
            ),
            ModelType.CANDIDATE_MATCHER: os.getenv(
                'SAGEMAKER_CANDIDATE_MATCHER_ENDPOINT',
                'adeptai-candidate-matcher-v1'
            ),
        }
    
    def _generate_cache_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model_type: ModelType,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key for request"""
        content = f"{prompt}|{system_prompt}|{model_type.value}|{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[LLMResponse]:
        """Get response from cache if available and not expired"""
        if not self.cache_enabled or cache_key not in self.cache:
            return None
        
        cached_item = self.cache[cache_key]
        if time.time() - cached_item['timestamp'] > self.cache_ttl:
            del self.cache[cache_key]
            return None
        
        self.stats["cache_hits"] += 1
        return cached_item['response']
    
    def _save_to_cache(self, cache_key: str, response: LLMResponse):
        """Save response to cache"""
        if self.cache_enabled:
            self.cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
            # Clean expired cache entries periodically
            if len(self.cache) > 1000:
                self._clean_cache()
    
    def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.cache.items()
            if current_time - item['timestamp'] > self.cache_ttl
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def invoke(
        self,
        model_type: ModelType,
        request: LLMRequest,
        use_cache: bool = True
    ) -> LLMResponse:
        """
        Invoke SageMaker endpoint with request
        
        Args:
            model_type: Type of model to invoke
            request: LLM request object
            use_cache: Whether to use cache
            
        Returns:
            LLMResponse object
            
        Raises:
            ValueError: If endpoint not configured
            Exception: If invocation fails
        """
        endpoint_name = self.endpoints.get(model_type)
        if not endpoint_name:
            raise ValueError(f"Endpoint not configured for {model_type.value}")
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(
                request.prompt,
                request.system_prompt,
                model_type,
                {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k
                }
            )
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Prepare payload for SageMaker
        payload = self._prepare_payload(request)
        
        # Invoke endpoint with retries
        start_time = time.time()
        response = None
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(payload),
                    CustomAttributes='accept_eula=true'  # For Llama models
                )
                break
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Endpoint invocation failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Endpoint invocation failed after {self.max_retries} attempts: {e}")
                    raise
        
        # Parse response
        processing_time = time.time() - start_time
        response_body = json.loads(response['Body'].read().decode('utf-8'))
        
        # Extract response content
        if isinstance(response_body, dict):
            # Handle different response formats
            content = response_body.get('generated_text', '') or \
                     response_body.get('outputs', [''])[0] if isinstance(response_body.get('outputs'), list) else '' or \
                     response_body.get('response', '')
            
            tokens_used = response_body.get('tokens_used', 0) or \
                         len(request.prompt.split()) + len(content.split())  # Fallback estimation
            
            confidence = response_body.get('confidence_score', 0.0)
        else:
            content = str(response_body)
            tokens_used = len(request.prompt.split()) + len(content.split())
            confidence = 0.0
        
        # Create response object
        llm_response = LLMResponse(
            content=content,
            model_used=endpoint_name,
            tokens_used=tokens_used,
            processing_time=processing_time,
            confidence_score=confidence,
            metadata=response_body.get('metadata')
        )
        
        # Update statistics
        self.stats["total_requests"] += 1
        self.stats["successful_requests"] += 1
        self.stats["total_tokens"] += tokens_used
        self.stats["total_latency"] += processing_time
        
        # Save to cache
        if use_cache:
            self._save_to_cache(cache_key, llm_response)
        
        return llm_response
    
    def _prepare_payload(self, request: LLMRequest) -> Dict[str, Any]:
        """Prepare payload for SageMaker endpoint"""
        payload = {
            "inputs": request.prompt,
            "parameters": {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "return_full_text": False
            }
        }
        
        if request.system_prompt:
            payload["system_prompt"] = request.system_prompt
        
        if request.stop_sequences:
            payload["parameters"]["stop"] = request.stop_sequences
        
        if request.metadata:
            payload["metadata"] = request.metadata
        
        return payload
    
    def batch_invoke(
        self,
        model_type: ModelType,
        requests: List[LLMRequest],
        use_cache: bool = True
    ) -> List[LLMResponse]:
        """
        Invoke endpoint with multiple requests (batch processing)
        
        Args:
            model_type: Type of model to invoke
            requests: List of LLM request objects
            use_cache: Whether to use cache
            
        Returns:
            List of LLMResponse objects
        """
        responses = []
        for request in requests:
            try:
                response = self.invoke(model_type, request, use_cache)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch request failed: {e}")
                # Create error response
                responses.append(LLMResponse(
                    content="",
                    model_used=self.endpoints.get(model_type, "unknown"),
                    tokens_used=0,
                    processing_time=0.0,
                    confidence_score=0.0,
                    metadata={"error": str(e)}
                ))
        
        return responses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        total_requests = self.stats["total_requests"]
        if total_requests == 0:
            return self.stats.copy()
        
        return {
            **self.stats,
            "avg_latency": self.stats["total_latency"] / total_requests,
            "success_rate": self.stats["successful_requests"] / total_requests,
            "cache_hit_rate": self.stats["cache_hits"] / total_requests,
            "avg_tokens_per_request": self.stats["total_tokens"] / total_requests
        }
    
    def clear_cache(self):
        """Clear all cached responses"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def health_check(self, model_type: ModelType) -> bool:
        """
        Check health of endpoint
        
        Args:
            model_type: Type of model to check
            
        Returns:
            True if endpoint is healthy, False otherwise
        """
        endpoint_name = self.endpoints.get(model_type)
        if not endpoint_name:
            return False
        
        try:
            # Send minimal test request
            test_request = LLMRequest(
                prompt="test",
                max_tokens=10,
                temperature=0.1
            )
            response = self.invoke(model_type, test_request, use_cache=False)
            return response.content is not None
        except Exception as e:
            logger.error(f"Health check failed for {model_type.value}: {e}")
            return False


# Global client instance
_sagemaker_client: Optional[SageMakerLLMClient] = None


def get_sagemaker_client() -> SageMakerLLMClient:
    """Get or create global SageMaker client instance"""
    global _sagemaker_client
    if _sagemaker_client is None:
        _sagemaker_client = SageMakerLLMClient()
    return _sagemaker_client


def initialize_sagemaker_client(**kwargs) -> SageMakerLLMClient:
    """Initialize global SageMaker client with custom configuration"""
    global _sagemaker_client
    _sagemaker_client = SageMakerLLMClient(**kwargs)
    return _sagemaker_client

