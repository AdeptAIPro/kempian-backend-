"""
ChatGPT Fallback Service
Fallback to OpenAI ChatGPT when SageMaker is unavailable
"""

import os
import logging
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. ChatGPT fallback disabled.")


class ChatGPTFallbackService:
    """Fallback service using OpenAI ChatGPT API"""
    
    def __init__(self):
        """Initialize ChatGPT fallback service"""
        self.client = None
        self.available = False
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not installed. Install with: pip install openai")
            return
        
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
            self.available = True
            logger.info(f"ChatGPT fallback service initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatGPT fallback: {e}")
            self.available = False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text using ChatGPT"""
        if not self.available or not self.client:
            raise Exception("ChatGPT fallback service not available")
        
        try:
            # Build full prompt with context
            full_prompt = prompt
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                full_prompt = f"{context_str}\n\n{prompt}"
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"ChatGPT API error: {e}")
            raise
    
    def extract_job_from_requirement(self, requirement: str) -> Dict[str, Any]:
        """Extract structured job data from requirement"""
        system_prompt = """You are an expert job posting extractor. Extract structured job data from requirements. Return ONLY valid JSON with these fields: title, description, location, company_name, employment_type, experience_level, skills (array), benefits, requirements, responsibilities, salary_min, salary_max, currency. Do not include markdown or commentary."""
        
        prompt = f"""{system_prompt}

Requirement: {requirement}

Extract and return JSON:"""
        
        try:
            response = self.generate(prompt, max_tokens=512, temperature=0.7)
            
            # Parse JSON from response
            import json
            json_start = response.find('{')
            json_end = response.rfind('}')
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end + 1]
            else:
                json_str = response
            
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
            logger.error(f"Error extracting job with ChatGPT: {e}")
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
        """Analyze candidates based on user query"""
        # Build system prompt
        candidate_count = len(candidates)
        candidate_details = ""
        
        for i, candidate in enumerate(candidates[:10]):  # Limit to first 10 for prompt
            name = candidate.get("FullName") or candidate.get("name") or candidate.get("fullName") or "Unknown"
            skills = candidate.get("skills") or candidate.get("Skills") or candidate.get("technicalSkills") or []
            experience = candidate.get("experience") or candidate.get("Experience") or "Unknown"
            
            candidate_details += f"\nCandidate {i+1}: {name} - Skills: {', '.join(skills) if isinstance(skills, list) else skills} - Experience: {experience}"
        
        system_prompt = f"""You are an expert HR AI assistant specializing in talent acquisition and candidate analysis. You have access to {candidate_count} candidates.

Job Description: {job_description or 'Not provided'}

Candidates:
{candidate_details}

Provide intelligent, conversational analysis based on the user's query. Be helpful, professional, and insightful."""
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history[-5:])
        
        messages.append({"role": "user", "content": message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "response": response_text,
                "suggestions": self._extract_suggestions(response_text),
                "confidence": self._calculate_confidence(response_text),
                "analysis_type": self._determine_analysis_type(message)
            }
        except Exception as e:
            logger.error(f"Error analyzing candidates with ChatGPT: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error analyzing candidates. Please try again.",
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
            if trimmed.startswith('•') or trimmed.startswith('-') or trimmed.startswith('*') or \
               trimmed.startswith(('1.', '2.', '3.', '4.', '5.')) or \
               'suggest' in trimmed.lower() or 'recommend' in trimmed.lower():
                suggestions.append(trimmed)
        
        return suggestions[:5]
    
    def _calculate_confidence(self, response: str) -> int:
        """Calculate confidence score"""
        confidence = 70
        
        if len(response) > 200:
            confidence += 10
        if 'specific' in response.lower() or 'example' in response.lower():
            confidence += 10
        if 'recommend' in response.lower() or 'suggest' in response.lower():
            confidence += 5
        if 'because' in response.lower() or 'since' in response.lower():
            confidence += 5
        
        return min(confidence, 95)
    
    def _determine_analysis_type(self, message: str) -> str:
        """Determine analysis type from message"""
        lower_message = message.lower()
        
        if 'compare' in lower_message or 'vs' in lower_message or 'difference' in lower_message:
            return 'comparison'
        if 'recommend' in lower_message or 'best' in lower_message or 'top' in lower_message:
            return 'recommendation'
        if 'specific' in lower_message or 'particular' in lower_message or 'candidate' in lower_message:
            return 'specific'
        return 'general'
    
    def is_available(self) -> bool:
        """Check if ChatGPT fallback is available"""
        return self.available

