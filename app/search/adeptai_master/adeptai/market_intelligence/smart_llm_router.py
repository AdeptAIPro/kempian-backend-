"""
Smart LLM Router for Cost-Effective API Usage

Intelligently routes requests between OpenAI GPT-4o-mini and Claude Sonnet 4
based on task complexity and cost optimization
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from .fast_llm import FastLLMService, FastLLMRequest, FastLLMResponse
from .claude_config import ClaudeConfig
from .openai_config import OpenAIConfig

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for smart routing"""
    SIMPLE = "simple"           # Use GPT-4o-mini
    MODERATE = "moderate"       # Use GPT-4o-mini
    COMPLEX = "complex"         # Use Claude Sonnet 4
    CRITICAL = "critical"       # Use Claude Sonnet 4


class CostOptimization(Enum):
    """Cost optimization strategies"""
    MINIMIZE_COST = "minimize_cost"      # Prefer GPT-4o-mini
    BALANCE = "balance"                  # Balance cost and quality
    MAXIMIZE_QUALITY = "maximize_quality"  # Prefer Claude Sonnet 4


@dataclass
class SmartRoutingConfig:
    """Configuration for smart routing"""
    cost_optimization: CostOptimization = CostOptimization.BALANCE
    max_cost_per_request: float = 0.01  # $0.01 max cost per request
    quality_threshold: float = 0.8      # Minimum quality threshold
    use_caching: bool = True
    fallback_enabled: bool = True


class SmartLLMRouter:
    """Smart LLM router for cost-effective API usage"""
    
    def __init__(self, config: SmartRoutingConfig = None):
        self.config = config or SmartRoutingConfig()
        self.llm_service = FastLLMService()
        self.usage_stats = {
            "total_requests": 0,
            "openai_requests": 0,
            "claude_requests": 0,
            "total_cost": 0.0,
            "cache_hits": 0
        }
    
    def _analyze_task_complexity(self, prompt: str, system_prompt: str) -> TaskComplexity:
        """Analyze task complexity based on content"""
        
        # Keywords that indicate complex tasks
        complex_keywords = [
            "analyze", "comprehensive", "detailed", "nuanced", "complex",
            "strategic", "critical", "sophisticated", "advanced", "expert",
            "behavioral analysis", "market intelligence", "psychological",
            "multi-factor", "correlation", "synthesis"
        ]
        
        # Keywords that indicate simple tasks
        simple_keywords = [
            "basic", "simple", "quick", "summary", "overview", "list",
            "categorize", "classify", "filter", "screen", "check"
        ]
        
        content = (prompt + " " + system_prompt).lower()
        
        # Count complexity indicators
        complex_score = sum(1 for keyword in complex_keywords if keyword in content)
        simple_score = sum(1 for keyword in simple_keywords if keyword in content)
        
        # Determine complexity
        if complex_score > simple_score and complex_score >= 2:
            return TaskComplexity.COMPLEX
        elif complex_score > simple_score:
            return TaskComplexity.MODERATE
        elif simple_score > 0:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MODERATE
    
    def _select_model(self, complexity: TaskComplexity, estimated_tokens: int) -> str:
        """Select optimal model based on complexity and cost"""
        
        # Cost estimates (per 1M tokens)
        openai_cost = 0.15  # GPT-4o-mini
        claude_cost = 3.00  # Claude Sonnet 4
        
        estimated_cost_openai = (estimated_tokens / 1_000_000) * openai_cost
        estimated_cost_claude = (estimated_tokens / 1_000_000) * claude_cost
        
        # Apply cost optimization strategy
        if self.config.cost_optimization == CostOptimization.MINIMIZE_COST:
            # Always prefer GPT-4o-mini unless complexity is critical
            if complexity in [TaskComplexity.CRITICAL]:
                return "claude-sonnet-4-20250514"
            else:
                return "gpt-4o-mini"
        
        elif self.config.cost_optimization == CostOptimization.MAXIMIZE_QUALITY:
            # Always prefer Claude unless task is very simple
            if complexity == TaskComplexity.SIMPLE:
                return "gpt-4o-mini"
            else:
                return "claude-sonnet-4-20250514"
        
        else:  # BALANCE
            # Balance cost and quality
            if complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
                return "gpt-4o-mini"
            elif complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]:
                return "claude-sonnet-4-20250514"
            else:
                return "gpt-4o-mini"
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text) // 4
    
    async def route_request(self, 
                          prompt: str, 
                          system_prompt: str = "",
                          max_tokens: int = 1000,
                          temperature: float = 0.7,
                          cache_key: Optional[str] = None) -> FastLLMResponse:
        """Route request to optimal model"""
        
        # Analyze task complexity
        complexity = self._analyze_task_complexity(prompt, system_prompt)
        
        # Estimate tokens
        estimated_tokens = self._estimate_tokens(prompt + system_prompt) + max_tokens
        
        # Select optimal model
        model = self._select_model(complexity, estimated_tokens)
        
        # Create request
        request = FastLLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            cache_key=cache_key
        )
        
        # Generate response
        response = await self.llm_service.generate_fast(request)
        
        # Update stats
        self.usage_stats["total_requests"] += 1
        self.usage_stats["total_cost"] += response.cost
        
        if "gpt" in response.model.lower():
            self.usage_stats["openai_requests"] += 1
        elif "claude" in response.model.lower():
            self.usage_stats["claude_requests"] += 1
        
        if response.cached:
            self.usage_stats["cache_hits"] += 1
        
        # Log routing decision
        logger.info(f"Routed to {model} (complexity: {complexity.value}, cost: ${response.cost:.4f})")
        
        return response
    
    async def batch_route_requests(self, requests: List[Dict[str, Any]]) -> List[FastLLMResponse]:
        """Route multiple requests efficiently"""
        
        # Create FastLLMRequest objects
        llm_requests = []
        for req in requests:
            complexity = self._analyze_task_complexity(req["prompt"], req.get("system_prompt", ""))
            estimated_tokens = self._estimate_tokens(req["prompt"] + req.get("system_prompt", "")) + req.get("max_tokens", 1000)
            model = self._select_model(complexity, estimated_tokens)
            
            llm_request = FastLLMRequest(
                prompt=req["prompt"],
                system_prompt=req.get("system_prompt", ""),
                model=model,
                max_tokens=req.get("max_tokens", 1000),
                temperature=req.get("temperature", 0.7),
                cache_key=req.get("cache_key")
            )
            llm_requests.append(llm_request)
        
        # Process requests
        responses = await self.llm_service.batch_generate(llm_requests)
        
        # Update stats
        for response in responses:
            self.usage_stats["total_requests"] += 1
            self.usage_stats["total_cost"] += response.cost
            
            if "gpt" in response.model.lower():
                self.usage_stats["openai_requests"] += 1
            elif "claude" in response.model.lower():
                self.usage_stats["claude_requests"] += 1
            
            if response.cached:
                self.usage_stats["cache_hits"] += 1
        
        return responses
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics and cost breakdown"""
        total_requests = self.usage_stats["total_requests"]
        
        if total_requests == 0:
            return {
                "total_requests": 0,
                "total_cost": 0.0,
                "avg_cost_per_request": 0.0,
                "model_distribution": {"openai": 0, "claude": 0},
                "cache_hit_rate": 0.0
            }
        
        return {
            "total_requests": total_requests,
            "total_cost": round(self.usage_stats["total_cost"], 4),
            "avg_cost_per_request": round(self.usage_stats["total_cost"] / total_requests, 4),
            "model_distribution": {
                "openai": round(self.usage_stats["openai_requests"] / total_requests * 100, 1),
                "claude": round(self.usage_stats["claude_requests"] / total_requests * 100, 1)
            },
            "cache_hit_rate": round(self.usage_stats["cache_hits"] / total_requests, 3),
            "cost_breakdown": {
                "openai_cost": round(self.usage_stats["openai_requests"] * 0.00015, 4),  # Estimated
                "claude_cost": round(self.usage_stats["claude_requests"] * 0.003, 4)    # Estimated
            }
        }
    
    def get_cost_optimization_recommendations(self) -> List[str]:
        """Get cost optimization recommendations"""
        recommendations = []
        
        total_requests = self.usage_stats["total_requests"]
        if total_requests == 0:
            return ["No usage data available yet"]
        
        claude_ratio = self.usage_stats["claude_requests"] / total_requests
        
        if claude_ratio > 0.7:
            recommendations.append("High Claude usage detected. Consider using GPT-4o-mini for simple tasks to reduce costs.")
        
        if self.usage_stats["cache_hits"] / total_requests < 0.3:
            recommendations.append("Low cache hit rate. Enable more aggressive caching for repeated queries.")
        
        if self.usage_stats["total_cost"] / total_requests > 0.01:
            recommendations.append("High average cost per request. Review task complexity routing.")
        
        return recommendations


# Global smart router instance
smart_router = SmartLLMRouter()
