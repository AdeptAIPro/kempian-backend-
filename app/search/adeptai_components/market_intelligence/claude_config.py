"""
Claude API Configuration

Configuration for Claude API integration with the provided API key
"""

import os
from typing import Dict, Any


class ClaudeConfig:
    """Claude API configuration"""
    
    # Claude API Key (loaded from environment for security)
    API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Model configuration
    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7
    
    # API endpoints
    BASE_URL = "https://api.anthropic.com/v1"
    MESSAGES_ENDPOINT = f"{BASE_URL}/messages"
    
    # Headers
    HEADERS = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get Claude configuration"""
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
        """Test if Claude API is accessible"""
        try:
            import aiohttp
            import asyncio
            
            async def test():
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": cls.MODEL,
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hello"}]
                    }
                    
                    async with session.post(
                        cls.MESSAGES_ENDPOINT,
                        headers=cls.HEADERS,
                        json=payload
                    ) as response:
                        return response.status == 200
            
            return asyncio.run(test())
        except Exception:
            return False
    
    @classmethod
    def get_usage_info(cls) -> Dict[str, Any]:
        """Get Claude API usage information"""
        return {
            "model": cls.MODEL,
            "cost_per_1m_tokens": 3.00,  # Claude Sonnet 4 pricing
            "max_tokens": cls.MAX_TOKENS,
            "api_status": "configured",
            "test_passed": cls.test_connection()
        }


# Set environment variable for other modules
os.environ["ANTHROPIC_API_KEY"] = ClaudeConfig.API_KEY
