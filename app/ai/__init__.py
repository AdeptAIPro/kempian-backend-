# AI Module for Kempian Platform
# This module handles AI integration with Ollama and Llama 3

from .service import AIService
from .routes import ai_bp

__all__ = ['AIService', 'ai_bp']
