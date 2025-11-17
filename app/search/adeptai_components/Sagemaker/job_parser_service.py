"""
SageMaker Job Description Parser Service
Uses Llama 3.1 8B fine-tuned for structured job description extraction
"""

import json
import logging
import re
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
class JobRequirements:
    """Structured job requirements"""
    mandatory_skills: List[str]
    preferred_skills: List[str]
    required_experience_years: int
    education_level: str
    education_field: Optional[str]
    location: str
    remote_allowed: bool
    salary_range: Optional[Dict[str, Any]]
    seniority_level: str
    domain: str
    certifications: List[str]
    languages: List[str]
    travel_required: bool
    confidence: float = 0.0


class SageMakerJobParser:
    """
    Job description parser using SageMaker-hosted LLM
    Extracts structured requirements from job descriptions
    """
    
    def __init__(
        self,
        client: Optional[SageMakerLLMClient] = None,
        model_name: str = "llama-3.1-8b-instruct",
        temperature: float = 0.3,
        max_tokens: int = 512
    ):
        """
        Initialize SageMaker Job Parser
        
        Args:
            client: SageMaker LLM client (uses global client if None)
            model_name: Name of the model (for reference)
            temperature: Sampling temperature (lower for structured output)
            max_tokens: Maximum tokens to generate
        """
        self.client = client or get_sagemaker_client()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _create_extraction_prompt(self, job_description: str) -> str:
        """Create prompt for job requirement extraction"""
        return f"""You are an expert recruitment AI assistant specializing in extracting structured requirements from job descriptions.

Extract all requirements from the following job description:

**Job Description:**
{job_description[:4000]}...

Return ONLY a valid JSON object with the following structure:
{{
    "mandatory_skills": ["skill1", "skill2", ...],
    "preferred_skills": ["skill1", "skill2", ...],
    "required_experience_years": 0,
    "education_level": "High School|Bachelor's|Master's|PhD|None",
    "education_field": "field name or null",
    "location": "city, state or 'Remote'",
    "remote_allowed": true/false,
    "salary_range": {{
        "min": 0,
        "max": 0,
        "currency": "USD"
    }} or null,
    "seniority_level": "Entry|Junior|Mid|Senior|Lead|Principal|Executive",
    "domain": "Technology|Healthcare|Finance|Education|Marketing|Other",
    "certifications": ["cert1", "cert2", ...],
    "languages": ["language1", "language2", ...],
    "travel_required": true/false
}}

Extract information accurately. If information is not available, use null for optional fields or empty arrays for lists.
Return ONLY the JSON object, no additional text or explanation."""
    
    def _parse_extraction_response(self, response: str) -> JobRequirements:
        """Parse LLM response into JobRequirements object"""
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
            
            # Extract salary range
            salary_range = data.get('salary_range')
            if salary_range and isinstance(salary_range, dict):
                salary_range = {
                    "min": int(salary_range.get('min', 0)),
                    "max": int(salary_range.get('max', 0)),
                    "currency": salary_range.get('currency', 'USD')
                }
            else:
                salary_range = None
            
            # Create job requirements object
            requirements = JobRequirements(
                mandatory_skills=data.get('mandatory_skills', []),
                preferred_skills=data.get('preferred_skills', []),
                required_experience_years=int(data.get('required_experience_years', 0)),
                education_level=data.get('education_level', 'None'),
                education_field=data.get('education_field'),
                location=data.get('location', ''),
                remote_allowed=bool(data.get('remote_allowed', False)),
                salary_range=salary_range,
                seniority_level=data.get('seniority_level', 'Mid'),
                domain=data.get('domain', 'Other'),
                certifications=data.get('certifications', []),
                languages=data.get('languages', []),
                travel_required=bool(data.get('travel_required', False)),
                confidence=0.9
            )
            
            return requirements
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse job requirements as JSON: {e}")
            logger.debug(f"Response: {response}")
            return self._create_fallback_requirements()
        except Exception as e:
            logger.error(f"Error parsing job requirements: {e}")
            return self._create_fallback_requirements()
    
    def _create_fallback_requirements(self) -> JobRequirements:
        """Create fallback job requirements using simple pattern matching"""
        return JobRequirements(
            mandatory_skills=[],
            preferred_skills=[],
            required_experience_years=0,
            education_level='None',
            education_field=None,
            location='',
            remote_allowed=False,
            salary_range=None,
            seniority_level='Mid',
            domain='Other',
            certifications=[],
            languages=[],
            travel_required=False,
            confidence=0.3
        )
    
    def parse_job_description(
        self,
        job_description: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Parse job description to extract structured requirements
        
        Args:
            job_description: Job description text
            use_cache: Whether to use cache
            
        Returns:
            Dictionary with structured job requirements
        """
        if not job_description or not job_description.strip():
            return self._create_fallback_requirements().__dict__
        
        try:
            prompt = self._create_extraction_prompt(job_description)
            
            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt,
                system_prompt="You are an expert at extracting structured data. Always return valid JSON only.",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                top_k=50
            )
            
            # Invoke SageMaker endpoint
            response = self.client.invoke(
                ModelType.JOB_PARSER,
                llm_request,
                use_cache=use_cache
            )
            
            # Parse response
            requirements = self._parse_extraction_response(response.content)
            
            # Update confidence based on response quality
            requirements.confidence = response.confidence_score if response.confidence_score > 0 else 0.9
            
            logger.info("Job description parsed successfully")
            return requirements.__dict__
            
        except Exception as e:
            logger.error(f"SageMaker job parsing failed: {e}")
            return self._create_fallback_requirements().__dict__
    
    def batch_parse(
        self,
        job_descriptions: List[str],
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Parse multiple job descriptions
        
        Args:
            job_descriptions: List of job description texts
            use_cache: Whether to use cache
            
        Returns:
            List of job requirements dictionaries
        """
        results = []
        for job_description in job_descriptions:
            try:
                requirements = self.parse_job_description(job_description, use_cache)
                results.append(requirements)
            except Exception as e:
                logger.error(f"Failed to parse job description: {e}")
                # Add fallback requirements
                fallback = self._create_fallback_requirements()
                results.append(fallback.__dict__)
        
        return results


# Global instance
_job_parser: Optional[SageMakerJobParser] = None


def get_job_parser() -> SageMakerJobParser:
    """Get or create global job parser instance"""
    global _job_parser
    if _job_parser is None:
        _job_parser = SageMakerJobParser()
    return _job_parser

