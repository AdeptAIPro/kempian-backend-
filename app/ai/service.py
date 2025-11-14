"""
AI Service for Kempian Platform
Handles communication with Ollama and Llama 3 model
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class AIService:
    """AI Service for handling Ollama integration and AI operations"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize AI Service
        
        Args:
            ollama_url: URL of the Ollama server (default: localhost:11434)
        """
        self.ollama_url = ollama_url
        self.model_name = "llama3:8b"
        self.default_temperature = 0.7
        self.default_max_tokens = 1000
        
    def is_ollama_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            return []
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate AI response using Ollama
        
        Args:
            prompt: The input prompt
            context: Additional context for the AI (user data, job info, etc.)
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dict containing the AI response and metadata
        """
        try:
            # Check if Ollama is available
            if not self.is_ollama_available():
                return {
                    "success": False,
                    "error": "Ollama server is not available",
                    "response": "I'm currently unavailable. Please try again later."
                }
            
            # Prepare the full prompt with context
            full_prompt = self._build_prompt(prompt, context)
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature or self.default_temperature,
                    "num_predict": max_tokens or self.default_max_tokens,
                    "num_ctx": 2048,  # Reduce context window to save memory
                    "num_gpu": 0,     # Force CPU usage to save GPU memory
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "model": data.get("model", self.model_name),
                    "created_at": datetime.now().isoformat(),
                    "context_used": context is not None,
                    "tokens_generated": len(data.get("response", "").split()),
                    "metadata": {
                        "temperature": temperature or self.default_temperature,
                        "max_tokens": max_tokens or self.default_max_tokens,
                        "stream": stream
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Ollama API error: {response.status_code}",
                    "response": "I encountered an error processing your request."
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout",
                "response": "The request took too long to process. Please try again."
            }
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an unexpected error. Please try again."
            }
    
    def _build_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build the full prompt with context for the AI
        
        Args:
            prompt: User's input prompt
            context: Additional context (user info, job data, etc.)
            
        Returns:
            Formatted prompt string
        """
        # Base system prompt for Kempian AI
        system_prompt = """You are Kempian AI, an intelligent assistant for the Kempian job management platform. 
You help users with job-related tasks, career advice, resume optimization, and general job search assistance.

Key capabilities:
- Resume analysis and optimization suggestions
- Job matching and recommendations
- Career guidance and advice
- Interview preparation tips
- Salary negotiation strategies
- Industry insights and trends

Always be helpful, professional, and encouraging. Provide specific, actionable advice when possible."""

        # Add context if provided
        if context:
            context_str = self._format_context(context)
            system_prompt += f"\n\nCurrent context:\n{context_str}"
        
        # Combine system prompt with user input
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nKempian AI:"
        
        return full_prompt
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context data for the AI prompt"""
        context_parts = []
        
        if "user_profile" in context:
            user = context["user_profile"]
            context_parts.append(f"User Profile: {user.get('name', 'Unknown')} - {user.get('role', 'Job Seeker')}")
            if "experience" in user:
                context_parts.append(f"Experience: {user['experience']}")
            if "skills" in user:
                context_parts.append(f"Skills: {', '.join(user['skills'])}")
        
        if "job_data" in context:
            job = context["job_data"]
            context_parts.append(f"Job Context: {job.get('title', 'Unknown Position')} at {job.get('company', 'Unknown Company')}")
            if "requirements" in job:
                context_parts.append(f"Requirements: {job['requirements']}")
        
        if "conversation_history" in context:
            history = context["conversation_history"]
            context_parts.append(f"Recent conversation: {history}")
        
        return "\n".join(context_parts)
    
    def analyze_resume(self, resume_text: str, job_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze resume and provide optimization suggestions
        
        Args:
            resume_text: The resume content to analyze
            job_description: Optional job description to match against
            
        Returns:
            Analysis results and suggestions
        """
        prompt = f"Please analyze this resume and provide optimization suggestions"
        if job_description:
            prompt += f" specifically for this job: {job_description}"
        
        prompt += f"\n\nResume content:\n{resume_text[:2000]}"  # Limit resume length
        
        context = {
            "task_type": "resume_analysis",
            "job_description": job_description
        }
        
        return self.generate_response(prompt, context)
    
    def generate_job_recommendations(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate job recommendations based on user profile
        
        Args:
            user_profile: User's skills, experience, preferences
            
        Returns:
            Job recommendations and reasoning
        """
        prompt = f"Based on this user profile, suggest relevant job opportunities and career paths"
        
        context = {
            "user_profile": user_profile,
            "task_type": "job_recommendations"
        }
        
        return self.generate_response(prompt, context)
    
    def prepare_interview_questions(self, job_title: str, company: str, user_skills: List[str]) -> Dict[str, Any]:
        """
        Generate interview questions and preparation tips
        
        Args:
            job_title: The position being interviewed for
            company: Company name
            user_skills: User's relevant skills
            
        Returns:
            Interview questions and preparation advice
        """
        prompt = f"Generate interview questions and preparation tips for a {job_title} position at {company}"
        
        context = {
            "job_data": {
                "title": job_title,
                "company": company
            },
            "user_profile": {
                "skills": user_skills
            },
            "task_type": "interview_preparation"
        }
        
        return self.generate_response(prompt, context)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get the health status of the AI service"""
        ollama_available = self.is_ollama_available()
        models = self.get_available_models()
        
        return {
            "service_status": "healthy" if ollama_available else "unhealthy",
            "ollama_available": ollama_available,
            "model_loaded": self.model_name in [model.get("name", "") for model in models],
            "available_models": [model.get("name", "") for model in models],
            "timestamp": datetime.now().isoformat()
        }
