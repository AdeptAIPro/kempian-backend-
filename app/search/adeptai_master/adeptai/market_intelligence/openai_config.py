"""
OpenAI API Configuration

Configuration for OpenAI API integration with the provided API key
"""

import os
from typing import Dict, Any


class OpenAIConfig:
    """OpenAI API configuration"""
    
    # OpenAI API Key (loaded from environment for security)
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Model configuration
    MODEL = "gpt-4o-mini"  # Cost-effective model
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    
    # API endpoints
    BASE_URL = "https://api.openai.com/v1"
    CHAT_COMPLETIONS_ENDPOINT = f"{BASE_URL}/chat/completions"
    
    # Headers
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get OpenAI configuration"""
        return {
            "api_key": cls.API_KEY,
            "model": cls.MODEL,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
            "base_url": cls.BASE_URL,
            "headers": cls.HEADERS.copy()
        }
    
    @classmethod
    def test_connection(cls) -> bool:
        """Test if OpenAI API is accessible"""
        try:
            import aiohttp
            import asyncio
            
            async def test():
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": cls.MODEL,
                        "max_tokens": 50,
                        "messages": [{"role": "user", "content": "Hello"}]
                    }
                    
                    async with session.post(
                        cls.CHAT_COMPLETIONS_ENDPOINT,
                        headers=cls.HEADERS,
                        json=payload
                    ) as response:
                        return response.status == 200
            
            return asyncio.run(test())
        except Exception:
            return False
    
    @classmethod
    def get_usage_info(cls) -> Dict[str, Any]:
        """Get OpenAI API usage information"""
        return {
            "model": cls.MODEL,
            "cost_per_1m_tokens": 0.15,  # GPT-4o-mini pricing
            "max_tokens": cls.MAX_TOKENS,
            "api_status": "configured",
            "test_passed": cls.test_connection()
        }


# Set environment variable for other modules
os.environ["OPENAI_API_KEY"] = OpenAIConfig.API_KEY
