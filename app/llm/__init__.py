"""
Kempian Custom LLM Module
Custom LLM service to replace ChatGPT
"""

from .service import LLMService
from .routes import llm_bp

__all__ = ['LLMService', 'llm_bp']

