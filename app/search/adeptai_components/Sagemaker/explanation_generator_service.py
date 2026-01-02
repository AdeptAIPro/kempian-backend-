"""
SageMaker Explanation Generation Service
Uses Llama 3.1 8B fine-tuned for natural language explanations
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .sagemaker_llm_client import (
    SageMakerLLMClient,
    ModelType,
    LLMRequest,
    LLMResponse,
    get_sagemaker_client
)

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Explanation structure"""
    summary: str
    positive_factors: List[str]
    negative_factors: List[str]
    recommendation: str
    confidence_level: str
    reasoning: str
    risk_factors: List[str]
    strength_areas: List[str]
    confidence: float = 0.0


class SageMakerExplanationGenerator:
    """
    Explanation generation service using SageMaker-hosted LLM
    Generates natural language explanations for candidate selection decisions
    """
    
    def __init__(
        self,
        client: Optional[SageMakerLLMClient] = None,
        model_name: str = "llama-3.1-8b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """
        Initialize SageMaker Explanation Generator
        
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
    
    def _create_explanation_prompt(
        self,
        candidate_profile: Dict[str, Any],
        job_query: str,
        match_scores: Dict[str, float],
        ranking_position: int
    ) -> str:
        """Create prompt for explanation generation"""
        return f"""You are an expert recruitment AI assistant providing transparent explanations for candidate selection decisions.

**Candidate Profile:**
{json.dumps(candidate_profile, indent=2)}

**Job Query:**
{job_query}

**Match Scores:**
{json.dumps(match_scores, indent=2)}

**Ranking Position:** {ranking_position}

Generate a comprehensive, human-readable explanation for why this candidate was ranked at position {ranking_position}.

Provide your explanation in JSON format with the following structure:
{{
    "summary": "Brief summary of the candidate's fit (2-3 sentences)",
    "positive_factors": ["factor1", "factor2", ...],
    "negative_factors": ["factor1", "factor2", ...],
    "recommendation": "Strongly recommend|Recommend|Consider|Not recommended",
    "confidence_level": "Very High|High|Moderate|Low",
    "reasoning": "Detailed explanation of the decision (3-5 sentences)",
    "risk_factors": ["risk1", "risk2", ...],
    "strength_areas": ["strength1", "strength2", ...]
}}

Make the explanation clear, honest, and helpful for recruiters.
Return ONLY the JSON object, no additional text."""
    
    def _parse_explanation_response(self, response: str) -> Explanation:
        """Parse LLM response into Explanation object"""
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
            
            # Create explanation object
            explanation = Explanation(
                summary=data.get('summary', ''),
                positive_factors=data.get('positive_factors', []),
                negative_factors=data.get('negative_factors', []),
                recommendation=data.get('recommendation', 'Consider'),
                confidence_level=data.get('confidence_level', 'Moderate'),
                reasoning=data.get('reasoning', ''),
                risk_factors=data.get('risk_factors', []),
                strength_areas=data.get('strength_areas', []),
                confidence=0.9
            )
            
            return explanation
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse explanation as JSON: {e}")
            logger.debug(f"Response: {response}")
            return self._create_fallback_explanation()
        except Exception as e:
            logger.error(f"Error parsing explanation: {e}")
            return self._create_fallback_explanation()
    
    def _create_fallback_explanation(self) -> Explanation:
        """Create fallback explanation"""
        return Explanation(
            summary="Explanation unavailable due to processing error.",
            positive_factors=[],
            negative_factors=[],
            recommendation="Consider",
            confidence_level="Low",
            reasoning="Unable to generate detailed explanation at this time.",
            risk_factors=[],
            strength_areas=[],
            confidence=0.3
        )
    
    def generate_explanation(
        self,
        candidate_profile: Dict[str, Any],
        job_query: str,
        match_scores: Dict[str, float],
        ranking_position: int,
        use_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Generate explanation for candidate selection decision
        
        Args:
            candidate_profile: Candidate profile dictionary
            job_query: Job search query
            match_scores: Dictionary of match scores
            ranking_position: Candidate's ranking position
            use_cache: Whether to use cache (typically False for explanations)
            
        Returns:
            Dictionary with explanation
        """
        if not candidate_profile or not job_query:
            return self._create_fallback_explanation().__dict__
        
        try:
            prompt = self._create_explanation_prompt(
                candidate_profile,
                job_query,
                match_scores,
                ranking_position
            )
            
            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt,
                system_prompt="You are a transparent and helpful recruitment AI. Always return valid JSON only.",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                top_k=50
            )
            
            # Invoke SageMaker endpoint
            response = self.client.invoke(
                ModelType.EXPLANATION_GENERATOR,
                llm_request,
                use_cache=use_cache
            )
            
            # Parse response
            explanation = self._parse_explanation_response(response.content)
            
            # Update confidence based on response quality
            explanation.confidence = response.confidence_score if response.confidence_score > 0 else 0.9
            
            logger.info(f"Explanation generated for candidate at position {ranking_position}")
            return explanation.__dict__
            
        except Exception as e:
            logger.error(f"SageMaker explanation generation failed: {e}")
            return self._create_fallback_explanation().__dict__()
    
    def batch_generate_explanations(
        self,
        candidates: List[Dict[str, Any]],
        job_query: str,
        match_scores_list: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple candidates
        
        Args:
            candidates: List of candidate profile dictionaries
            job_query: Job search query
            match_scores_list: List of match scores dictionaries
            
        Returns:
            List of explanation dictionaries
        """
        results = []
        for i, (candidate, match_scores) in enumerate(zip(candidates, match_scores_list)):
            try:
                explanation = self.generate_explanation(
                    candidate_profile=candidate,
                    job_query=job_query,
                    match_scores=match_scores,
                    ranking_position=i + 1
                )
                results.append(explanation)
            except Exception as e:
                logger.error(f"Failed to generate explanation for candidate {i+1}: {e}")
                # Add fallback explanation
                fallback = self._create_fallback_explanation()
                results.append(fallback.__dict__)
        
        return results


# Global instance
_explanation_generator: Optional[SageMakerExplanationGenerator] = None


def get_explanation_generator() -> SageMakerExplanationGenerator:
    """Get or create global explanation generator instance"""
    global _explanation_generator
    if _explanation_generator is None:
        _explanation_generator = SageMakerExplanationGenerator()
    return _explanation_generator

