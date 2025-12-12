"""
SageMaker Service
Connects to SageMaker endpoint for inference
"""

import boto3
import json
import os
import logging
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)

class SageMakerLLMService:
    """Service for SageMaker LLM inference"""
    
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME", "kempian-llm-endpoint")
        
        # Initialize SageMaker runtime client
        try:
            self.client = boto3.client(
                'sagemaker-runtime',
                region_name=self.region
            )
            logger.info(f"SageMaker service initialized: {self.endpoint_name} in {self.region}")
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker client: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text using SageMaker endpoint"""
        try:
            # Build full prompt with context
            full_prompt = prompt
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                full_prompt = f"{context_str}\n\n{prompt}"
            
            # Prepare payload
            payload = {
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Invoke endpoint
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            # Parse response
            result = json.loads(response['Body'].read())
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"SageMaker inference error: {e}")
            raise
    
    def extract_job_from_requirement(self, requirement: str) -> Dict[str, Any]:
        """Extract structured job data"""
        system_prompt = """You are an expert job posting extractor. Extract structured job data from requirements. Return ONLY valid JSON with these fields: title, description, location, company_name, employment_type, experience_level, skills (array), benefits, requirements, responsibilities, salary_min, salary_max, currency. Do not include markdown or commentary."""
        
        prompt = f"""{system_prompt}

Requirement: {requirement}

Extract and return JSON:"""
        
        try:
            response = self.generate(prompt, max_tokens=512, temperature=0.7)
            
            # Parse JSON
            json_start = response.find('{')
            json_end = response.rfind('}')
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end + 1]
                parsed = json.loads(json_str)
                
                return {
                    "success": True,
                    "job": {
                        "title": parsed.get("title", "Untitled Role"),
                        "description": parsed.get("description", requirement),
                        "location": parsed.get("location"),
                        "company_name": parsed.get("company_name"),
                        "employment_type": parsed.get("employment_type"),
                        "experience_level": parsed.get("experience_level"),
                        "skills": parsed.get("skills", []),
                        "benefits": parsed.get("benefits"),
                        "requirements": parsed.get("requirements"),
                        "responsibilities": parsed.get("responsibilities"),
                        "salary_min": parsed.get("salary_min"),
                        "salary_max": parsed.get("salary_max"),
                        "currency": parsed.get("currency", "USD"),
                    }
                }
        except Exception as e:
            logger.error(f"Error extracting job: {e}")
            return {
                "success": False,
                "error": str(e),
                "job": {
                    "title": requirement[:60],
                    "description": requirement,
                    "skills": []
                }
            }
    
    def analyze_candidates(
        self,
        message: str,
        candidates: List[Dict],
        job_description: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Analyze candidates"""
        # Build prompt (same as local service)
        candidate_count = len(candidates)
        candidate_details = ""
        
        for i, candidate in enumerate(candidates[:10]):
            name = candidate.get("FullName") or candidate.get("name") or "Unknown"
            skills = candidate.get("skills") or candidate.get("Skills") or []
            experience = candidate.get("experience") or candidate.get("Experience") or "Unknown"
            
            candidate_details += f"\nCandidate {i+1}: {name} - Skills: {', '.join(skills) if isinstance(skills, list) else skills} - Experience: {experience}"
        
        system_prompt = f"""You are an expert HR AI assistant. You have access to {candidate_count} candidates.

Job Description: {job_description or 'Not provided'}

Candidates:
{candidate_details}

Provide intelligent analysis based on the user's query."""
        
        if conversation_history:
            history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in conversation_history[-5:]])
            prompt = f"{system_prompt}\n\nConversation History:\n{history_text}\n\nUser Query: {message}\n\nProvide analysis:"
        else:
            prompt = f"{system_prompt}\n\nUser Query: {message}\n\nProvide analysis:"
        
        try:
            response = self.generate(prompt, max_tokens=1024, temperature=0.7)
            
            return {
                "success": True,
                "response": response,
                "suggestions": self._extract_suggestions(response),
                "confidence": self._calculate_confidence(response),
                "analysis_type": self._determine_analysis_type(message)
            }
        except Exception as e:
            logger.error(f"Error analyzing candidates: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error analyzing candidates.",
                "suggestions": [],
                "confidence": 0,
                "analysis_type": "general"
            }
    
    def _extract_suggestions(self, response: str) -> List[str]:
        """Extract suggestions from response"""
        suggestions = []
        lines = response.split('\n')
        
        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith(('•', '-', '*')) or trimmed.startswith(('1.', '2.', '3.')):
                suggestions.append(trimmed)
        
        return suggestions[:5]
    
    def _calculate_confidence(self, response: str) -> int:
        """Calculate confidence score"""
        confidence = 70
        if len(response) > 200:
            confidence += 10
        if 'specific' in response.lower() or 'example' in response.lower():
            confidence += 10
        return min(confidence, 95)
    
    def _determine_analysis_type(self, message: str) -> str:
        """Determine analysis type"""
        lower_message = message.lower()
        if 'compare' in lower_message or 'vs' in lower_message:
            return 'comparison'
        if 'recommend' in lower_message or 'best' in lower_message:
            return 'recommendation'
        if 'specific' in lower_message or 'candidate' in lower_message:
            return 'specific'
        return 'general'

