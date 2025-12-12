"""
Kempian LLM Service
Handles inference with SageMaker (required) and ChatGPT fallback
"""

import logging
import os
from typing import Dict, Optional, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMService:
    """Service for custom LLM inference - SageMaker only with ChatGPT fallback"""
    
    def __init__(self):
        """Initialize LLM service - SageMaker required, ChatGPT fallback"""
        self.model_version = os.getenv("LLM_MODEL_VERSION", "v1.0")
        self.sagemaker_service = None
        self.chatgpt_fallback = None
        self.model_loaded = False
        
        # Initialize SageMaker service (required)
        try:
            from .sagemaker_service import SageMakerLLMService
            self.sagemaker_service = SageMakerLLMService()
            self.model_loaded = True
            logger.info("SageMaker LLM service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker service: {e}")
            logger.warning("SageMaker is required but failed. Initializing ChatGPT fallback.")
            self.model_loaded = False
        
        # Initialize ChatGPT fallback
        try:
            from .chatgpt_fallback import ChatGPTFallbackService
            self.chatgpt_fallback = ChatGPTFallbackService()
            if self.chatgpt_fallback.is_available():
                logger.info("ChatGPT fallback service initialized")
            else:
                logger.warning("ChatGPT fallback not available (missing API key)")
        except Exception as e:
            logger.warning(f"ChatGPT fallback not available: {e}")
            self.chatgpt_fallback = None
        
        # Validate that at least one service is available
        if not self.model_loaded and (not self.chatgpt_fallback or not self.chatgpt_fallback.is_available()):
            logger.error("CRITICAL: Neither SageMaker nor ChatGPT fallback is available!")
            raise RuntimeError(
                "LLM service initialization failed. "
                "SageMaker is required. Set USE_SAGEMAKER=true and configure AWS credentials. "
                "Alternatively, set OPENAI_API_KEY for fallback."
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text from prompt - SageMaker first, ChatGPT fallback"""
        # Try SageMaker first
        if self.sagemaker_service and self.model_loaded:
            try:
                return self.sagemaker_service.generate(prompt, max_tokens, temperature, context)
            except Exception as e:
                logger.warning(f"SageMaker generation failed: {e}. Falling back to ChatGPT.")
        
        # Fallback to ChatGPT
        if self.chatgpt_fallback and self.chatgpt_fallback.is_available():
            try:
                logger.info("Using ChatGPT fallback for generation")
                return self.chatgpt_fallback.generate(prompt, max_tokens, temperature, context)
            except Exception as e:
                logger.error(f"ChatGPT fallback also failed: {e}")
                raise Exception(f"Both SageMaker and ChatGPT fallback failed. Last error: {e}")
        
        raise Exception("No LLM service available. Configure SageMaker or OpenAI API key.")
    
    def extract_job_from_requirement(self, requirement: str) -> Dict[str, Any]:
        """Extract structured job data from requirement - SageMaker first, ChatGPT fallback"""
        # Try SageMaker first
        if self.sagemaker_service and self.model_loaded:
            try:
                return self.sagemaker_service.extract_job_from_requirement(requirement)
            except Exception as e:
                logger.warning(f"SageMaker job extraction failed: {e}. Falling back to ChatGPT.")
        
        # Fallback to ChatGPT
        if self.chatgpt_fallback and self.chatgpt_fallback.is_available():
            try:
                logger.info("Using ChatGPT fallback for job extraction")
                return self.chatgpt_fallback.extract_job_from_requirement(requirement)
            except Exception as e:
                logger.error(f"ChatGPT fallback also failed: {e}")
                return {
                    "success": False,
                    "error": f"Both SageMaker and ChatGPT failed. Last error: {e}",
                    "job": {
                        "title": requirement[:60],
                        "description": requirement,
                        "skills": []
                    }
                }
        
        return {
            "success": False,
            "error": "No LLM service available",
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
        """Analyze candidates based on user query - SageMaker first, ChatGPT fallback"""
        # Try SageMaker first
        if self.sagemaker_service and self.model_loaded:
            try:
                return self.sagemaker_service.analyze_candidates(
                    message, candidates, job_description, conversation_history
                )
            except Exception as e:
                logger.warning(f"SageMaker candidate analysis failed: {e}. Falling back to ChatGPT.")
        
        # Fallback to ChatGPT
        if self.chatgpt_fallback and self.chatgpt_fallback.is_available():
            try:
                logger.info("Using ChatGPT fallback for candidate analysis")
                return self.chatgpt_fallback.analyze_candidates(
                    message, candidates, job_description, conversation_history
                )
            except Exception as e:
                logger.error(f"ChatGPT fallback also failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "response": "I encountered an error analyzing candidates. Please try again.",
                    "suggestions": [],
                    "confidence": 0,
                    "analysis_type": "general"
                }
        
        return {
            "success": False,
            "error": "No LLM service available",
            "response": "LLM service is not available. Please configure SageMaker or OpenAI API key.",
            "suggestions": [],
            "confidence": 0,
            "analysis_type": "general"
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        sagemaker_status = "available" if (self.sagemaker_service and self.model_loaded) else "unavailable"
        chatgpt_status = "available" if (self.chatgpt_fallback and self.chatgpt_fallback.is_available()) else "unavailable"
        
        overall_status = "healthy" if (self.model_loaded or (self.chatgpt_fallback and self.chatgpt_fallback.is_available())) else "unhealthy"
        
        return {
            "status": overall_status,
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "sagemaker": sagemaker_status,
            "chatgpt_fallback": chatgpt_status,
            "primary_service": "sagemaker" if self.model_loaded else ("chatgpt" if chatgpt_status == "available" else "none"),
            "timestamp": datetime.now().isoformat()
        }

