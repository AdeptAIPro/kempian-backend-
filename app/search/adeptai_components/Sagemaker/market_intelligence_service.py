"""
SageMaker Market Intelligence Service
Replaces OpenAI/Claude market intelligence with SageMaker-hosted models
Uses Llama 3.1 8B fine-tuned for market analysis
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from .sagemaker_llm_client import (
    SageMakerLLMClient,
    ModelType,
    LLMRequest,
    LLMResponse,
    get_sagemaker_client
)

logger = logging.getLogger(__name__)


@dataclass
class MarketIntelligence:
    """Market intelligence result structure"""
    talent_availability: Dict[str, Any]
    compensation_trends: Dict[str, Any]
    skill_evolution: Dict[str, Any]
    competitive_landscape: Dict[str, Any]
    economic_indicators: Dict[str, Any]
    behavioral_insights: Dict[str, Any]
    recommendations: List[str]
    confidence: float = 0.0
    timestamp: str = ""


class SageMakerMarketIntelligence:
    """
    Market intelligence service using SageMaker-hosted LLM
    Replaces OpenAI/Claude market intelligence
    """
    
    def __init__(
        self,
        client: Optional[SageMakerLLMClient] = None,
        model_name: str = "llama-3.1-8b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize SageMaker Market Intelligence Service
        
        Args:
            client: SageMaker LLM client (uses global client if None)
            model_name: Name of the model (for reference)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.client = client or get_sagemaker_client()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _create_analysis_prompt(self, market_data: Dict[str, Any]) -> str:
        """Create prompt for market intelligence analysis"""
        return f"""You are an expert market intelligence analyst specializing in recruitment and talent markets.

Analyze the following market data and provide comprehensive insights:

**Market Data:**
{json.dumps(market_data, indent=2)}

Provide a detailed market intelligence analysis in JSON format with the following structure:
{{
    "talent_availability": {{
        "active_candidates": 0,
        "passive_candidates": 0,
        "geographic_distribution": {{}},
        "mobility_likelihood": 0.0-1.0,
        "confidence_score": 0.0-1.0
    }},
    "compensation_trends": {{
        "salary_trends": {{}},
        "overall_inflation": 0.0,
        "market_competitiveness": 0.0-1.0,
        "confidence_score": 0.0-1.0
    }},
    "skill_evolution": {{
        "skill_evolution": {{}},
        "emerging_skills": [],
        "declining_skills": [],
        "confidence_score": 0.0-1.0
    }},
    "competitive_landscape": {{
        "industry_growth": {{}},
        "competitor_activity": {{}},
        "market_sentiment": 0.0-1.0,
        "confidence_score": 0.0-1.0
    }},
    "economic_indicators": {{
        "unemployment_rate": 0.0,
        "gdp_growth": 0.0,
        "inflation_rate": 0.0,
        "labor_force_participation": 0.0
    }},
    "behavioral_insights": {{
        "job_switch_probability": 0.0-1.0,
        "salary_expectations": "string",
        "remote_preference": 0.0-1.0,
        "confidence": 0.0-1.0
    }},
    "recommendations": ["recommendation1", "recommendation2", ...]
}}

Return ONLY the JSON object, no additional text."""
    
    def _parse_analysis_response(self, response: str) -> MarketIntelligence:
        """Parse LLM response into MarketIntelligence object"""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Create market intelligence object
            intelligence = MarketIntelligence(
                talent_availability=data.get('talent_availability', {}),
                compensation_trends=data.get('compensation_trends', {}),
                skill_evolution=data.get('skill_evolution', {}),
                competitive_landscape=data.get('competitive_landscape', {}),
                economic_indicators=data.get('economic_indicators', {}),
                behavioral_insights=data.get('behavioral_insights', {}),
                recommendations=data.get('recommendations', []),
                confidence=0.9,
                timestamp=datetime.now().isoformat()
            )
            
            return intelligence
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse market intelligence as JSON: {e}")
            logger.debug(f"Response: {response}")
            return self._create_fallback_intelligence()
        except Exception as e:
            logger.error(f"Error parsing market intelligence: {e}")
            return self._create_fallback_intelligence()
    
    def _create_fallback_intelligence(self) -> MarketIntelligence:
        """Create fallback market intelligence"""
        return MarketIntelligence(
            talent_availability={},
            compensation_trends={},
            skill_evolution={},
            competitive_landscape={},
            economic_indicators={},
            behavioral_insights={},
            recommendations=[],
            confidence=0.3,
            timestamp=datetime.now().isoformat()
        )
    
    def analyze_market(
        self,
        market_data: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze market intelligence using SageMaker LLM
        
        Args:
            market_data: Market data dictionary
            use_cache: Whether to use cache
            
        Returns:
            Dictionary with market intelligence
        """
        if not market_data:
            return self._create_fallback_intelligence().__dict__
        
        try:
            prompt = self._create_analysis_prompt(market_data)
            
            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt,
                system_prompt="You are an expert market intelligence analyst. Always return valid JSON only.",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                top_k=50
            )
            
            # Invoke SageMaker endpoint
            response = self.client.invoke(
                ModelType.MARKET_INTELLIGENCE,
                llm_request,
                use_cache=use_cache
            )
            
            # Parse response
            intelligence = self._parse_analysis_response(response.content)
            
            # Update confidence based on response quality
            intelligence.confidence = response.confidence_score if response.confidence_score > 0 else 0.9
            
            logger.info("Market intelligence analysis completed")
            return intelligence.__dict__
            
        except Exception as e:
            logger.error(f"SageMaker market intelligence analysis failed: {e}")
            return self._create_fallback_intelligence().__dict__
    
    def generate_insights(
        self,
        market_data: Dict[str, Any],
        focus_area: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate focused market insights
        
        Args:
            market_data: Market data dictionary
            focus_area: Optional focus area (e.g., "compensation", "skills", "talent")
            
        Returns:
            Dictionary with focused insights
        """
        intelligence = self.analyze_market(market_data)
        
        if focus_area:
            # Extract specific focus area
            focus_map = {
                "compensation": intelligence.get("compensation_trends", {}),
                "skills": intelligence.get("skill_evolution", {}),
                "talent": intelligence.get("talent_availability", {}),
                "competition": intelligence.get("competitive_landscape", {}),
                "economic": intelligence.get("economic_indicators", {})
            }
            
            return {
                "focus_area": focus_area,
                "insights": focus_map.get(focus_area, {}),
                "confidence": intelligence.get("confidence", 0.0),
                "timestamp": intelligence.get("timestamp", "")
            }
        
        return intelligence


# Global instance
_market_intelligence: Optional[SageMakerMarketIntelligence] = None


def get_market_intelligence() -> SageMakerMarketIntelligence:
    """Get or create global market intelligence instance"""
    global _market_intelligence
    if _market_intelligence is None:
        _market_intelligence = SageMakerMarketIntelligence()
    return _market_intelligence

