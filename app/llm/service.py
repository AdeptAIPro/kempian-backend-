"""
Kempian LLM Service
Handles inference with custom fine-tuned model
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import logging
import os
from typing import Dict, Optional, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMService:
    """Service for custom LLM inference"""
    
    def __init__(self):
        """Initialize LLM service"""
        # Check if using SageMaker
        self.use_sagemaker = os.getenv("USE_SAGEMAKER", "false").lower() == "true"
        
        if self.use_sagemaker:
            try:
                from .sagemaker_service import SageMakerLLMService
                self.sagemaker_service = SageMakerLLMService()
                self.model_loaded = True  # SageMaker handles loading
                logger.info("Using SageMaker for LLM inference")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize SageMaker service: {e}. Falling back to local model.")
                self.use_sagemaker = False
        
        # Local model configuration
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_version = os.getenv("LLM_MODEL_VERSION", "v1.0")
        
        # Model configuration
        self.config = {
            "base_model": os.getenv("LLM_MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2"),
            "device": os.getenv("LLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
            "load_in_4bit": os.getenv("LLM_LOAD_IN_4BIT", "true").lower() == "true",
            "max_length": int(os.getenv("LLM_MAX_LENGTH", "2048")),
        }
        
        # Try to load model on initialization (optional, can be lazy loaded)
        try:
            self.load_model()
        except Exception as e:
            logger.warning(f"Could not load model on init: {e}. Will load on first request.")
    
    def load_model(self):
        """Load the model and tokenizer"""
        if self.model_loaded:
            return
        
        try:
            logger.info(f"Loading LLM model from: {self.config['base_model']}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["base_model"],
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Setup quantization if enabled
            quantization_config = None
            if self.config["load_in_4bit"] and self.config["device"] == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization")
            
            # Load model
            if quantization_config:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config["base_model"],
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config["base_model"],
                    torch_dtype=torch.float16 if self.config["device"] == "cuda" else torch.float32,
                    device_map="auto" if self.config["device"] == "cuda" else None,
                    trust_remote_code=True
                )
                if self.config["device"] != "cuda":
                    self.model = self.model.to(self.config["device"])
            
            self.model_loaded = True
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text from prompt"""
        if self.use_sagemaker:
            return self.sagemaker_service.generate(prompt, max_tokens, temperature, context)
        
        if not self.model_loaded:
            self.load_model()
        
        try:
            # Build full prompt with context if provided
            full_prompt = prompt
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                full_prompt = f"{context_str}\n\n{prompt}"
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config["max_length"] - max_tokens
            )
            
            if self.config["device"] == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.config["device"]) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(full_prompt):
                generated_text = generated_text[len(full_prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def extract_job_from_requirement(self, requirement: str) -> Dict[str, Any]:
        """Extract structured job data from requirement"""
        if self.use_sagemaker:
            return self.sagemaker_service.extract_job_from_requirement(requirement)
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
        """Analyze candidates based on user query"""
        if self.use_sagemaker:
            return self.sagemaker_service.analyze_candidates(
                message, candidates, job_description, conversation_history
            )
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
        
        # Add conversation history if provided
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
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            "status": "healthy" if self.model_loaded else "loading",
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "device": self.config["device"],
            "model_path": self.config["base_model"],
            "timestamp": datetime.now().isoformat()
        }

