"""
SageMaker Behavioral Analysis Service
Replaces OpenAI/Claude behavioral analysis with SageMaker-hosted models
Uses Llama 3.1 70B fine-tuned for candidate behavioral analysis
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
class BehavioralProfile:
    """Behavioral profile structure"""
    candidate_id: str
    overall_score: float
    leadership_score: float
    collaboration_score: float
    innovation_score: float
    adaptability_score: float
    stability_score: float
    growth_potential: float
    emotional_intelligence: float
    stress_resilience: float
    communication_effectiveness: float
    technical_depth: float
    learning_agility: float
    problem_solving_ability: float
    cultural_alignment: float
    strengths: List[str]
    development_areas: List[str]
    behavioral_patterns: List[str]
    risk_factors: List[str]
    confidence: float = 0.0
    reasoning: str = ""


class SageMakerBehavioralAnalyzer:
    """
    Behavioral analysis service using SageMaker-hosted LLM
    Replaces OpenAI/Claude behavioral analysis
    """
    
    def __init__(
        self,
        client: Optional[SageMakerLLMClient] = None,
        model_name: str = "llama-3.1-70b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize SageMaker Behavioral Analyzer
        
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
    
    def _create_analysis_prompt(
        self,
        resume_text: str,
        job_description: str,
        career_history: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create prompt for behavioral analysis"""
        career_context = ""
        if career_history:
            career_context = f"\n\nCareer History:\n{json.dumps(career_history, indent=2)}"
        
        return f"""You are an expert behavioral psychologist and recruitment AI specialist analyzing candidate profiles for job fit.

Analyze the following candidate profile and job description to provide comprehensive behavioral insights.

**Resume Text:**
{resume_text[:3000]}...

**Job Description:**
{job_description[:2000]}...{career_context}

Provide a detailed behavioral analysis in JSON format with the following structure:
{{
    "overall_score": 0.0-1.0,
    "leadership_score": 0.0-1.0,
    "collaboration_score": 0.0-1.0,
    "innovation_score": 0.0-1.0,
    "adaptability_score": 0.0-1.0,
    "stability_score": 0.0-1.0,
    "growth_potential": 0.0-1.0,
    "emotional_intelligence": 0.0-1.0,
    "stress_resilience": 0.0-1.0,
    "communication_effectiveness": 0.0-1.0,
    "technical_depth": 0.0-1.0,
    "learning_agility": 0.0-1.0,
    "problem_solving_ability": 0.0-1.0,
    "cultural_alignment": 0.0-1.0,
    "strengths": ["strength1", "strength2", ...],
    "development_areas": ["area1", "area2", ...],
    "behavioral_patterns": ["pattern1", "pattern2", ...],
    "risk_factors": ["risk1", "risk2", ...],
    "reasoning": "Detailed explanation of the analysis..."
}}

Return ONLY the JSON object, no additional text."""
    
    def _parse_analysis_response(
        self,
        response: str,
        candidate_id: str
    ) -> BehavioralProfile:
        """Parse LLM response into BehavioralProfile object"""
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
            
            # Create behavioral profile
            profile = BehavioralProfile(
                candidate_id=candidate_id,
                overall_score=float(data.get('overall_score', 0.5)),
                leadership_score=float(data.get('leadership_score', 0.5)),
                collaboration_score=float(data.get('collaboration_score', 0.5)),
                innovation_score=float(data.get('innovation_score', 0.5)),
                adaptability_score=float(data.get('adaptability_score', 0.5)),
                stability_score=float(data.get('stability_score', 0.5)),
                growth_potential=float(data.get('growth_potential', 0.5)),
                emotional_intelligence=float(data.get('emotional_intelligence', 0.5)),
                stress_resilience=float(data.get('stress_resilience', 0.5)),
                communication_effectiveness=float(data.get('communication_effectiveness', 0.5)),
                technical_depth=float(data.get('technical_depth', 0.5)),
                learning_agility=float(data.get('learning_agility', 0.5)),
                problem_solving_ability=float(data.get('problem_solving_ability', 0.5)),
                cultural_alignment=float(data.get('cultural_alignment', 0.5)),
                strengths=data.get('strengths', []),
                development_areas=data.get('development_areas', []),
                behavioral_patterns=data.get('behavioral_patterns', []),
                risk_factors=data.get('risk_factors', []),
                confidence=0.9,
                reasoning=data.get('reasoning', '')
            )
            
            return profile
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse behavioral analysis as JSON: {e}")
            logger.debug(f"Response: {response}")
            return self._create_fallback_profile(candidate_id)
        except Exception as e:
            logger.error(f"Error parsing behavioral analysis: {e}")
            return self._create_fallback_profile(candidate_id)
    
    def _create_fallback_profile(self, candidate_id: str) -> BehavioralProfile:
        """Create fallback behavioral profile"""
        return BehavioralProfile(
            candidate_id=candidate_id,
            overall_score=0.5,
            leadership_score=0.5,
            collaboration_score=0.5,
            innovation_score=0.5,
            adaptability_score=0.5,
            stability_score=0.5,
            growth_potential=0.5,
            emotional_intelligence=0.5,
            stress_resilience=0.5,
            communication_effectiveness=0.5,
            technical_depth=0.5,
            learning_agility=0.5,
            problem_solving_ability=0.5,
            cultural_alignment=0.5,
            strengths=[],
            development_areas=[],
            behavioral_patterns=[],
            risk_factors=[],
            confidence=0.3,
            reasoning="Fallback analysis - LLM analysis unavailable"
        )
    
    def analyze_behavior(
        self,
        resume_text: str,
        job_description: str,
        career_history: Optional[Dict[str, Any]] = None,
        candidate_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze candidate behavior using SageMaker LLM
        
        Args:
            resume_text: Candidate resume text
            job_description: Job description text
            career_history: Optional career history data
            candidate_id: Optional candidate identifier
            
        Returns:
            Dictionary with behavioral profile
        """
        if not resume_text or not job_description:
            return self._create_fallback_profile(candidate_id or "unknown").__dict__
        
        candidate_id = candidate_id or f"candidate_{hash(resume_text[:100])}"
        
        try:
            prompt = self._create_analysis_prompt(
                resume_text,
                job_description,
                career_history
            )
            
            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt,
                system_prompt="You are an expert behavioral psychologist. Always return valid JSON only.",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                top_k=50
            )
            
            # Invoke SageMaker endpoint
            response = self.client.invoke(
                ModelType.BEHAVIORAL_ANALYZER,
                llm_request,
                use_cache=False  # Behavioral analysis is typically unique per candidate
            )
            
            # Parse response
            profile = self._parse_analysis_response(
                response.content,
                candidate_id
            )
            
            # Update confidence based on response quality
            profile.confidence = response.confidence_score if response.confidence_score > 0 else 0.9
            
            logger.info(f"Behavioral analysis completed for candidate: {candidate_id}")
            return profile.__dict__
            
        except Exception as e:
            logger.error(f"SageMaker behavioral analysis failed: {e}")
            return self._create_fallback_profile(candidate_id).__dict__
    
    def batch_analyze(
        self,
        candidates: List[Dict[str, Any]],
        job_description: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple candidates
        
        Args:
            candidates: List of candidate dictionaries with 'resume_text' and optional 'career_history'
            job_description: Job description text
            
        Returns:
            List of behavioral profile dictionaries
        """
        results = []
        for candidate in candidates:
            try:
                profile = self.analyze_behavior(
                    resume_text=candidate.get('resume_text', ''),
                    job_description=job_description,
                    career_history=candidate.get('career_history'),
                    candidate_id=candidate.get('candidate_id')
                )
                results.append(profile)
            except Exception as e:
                logger.error(f"Failed to analyze candidate: {e}")
                # Add fallback profile
                fallback = self._create_fallback_profile(
                    candidate.get('candidate_id', 'unknown')
                )
                results.append(fallback.__dict__)
        
        return results


# Global instance
_behavioral_analyzer: Optional[SageMakerBehavioralAnalyzer] = None


def get_behavioral_analyzer() -> SageMakerBehavioralAnalyzer:
    """Get or create global behavioral analyzer instance"""
    global _behavioral_analyzer
    if _behavioral_analyzer is None:
        _behavioral_analyzer = SageMakerBehavioralAnalyzer()
    return _behavioral_analyzer

