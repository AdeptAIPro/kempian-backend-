"""
API Configuration for Market Intelligence

Environment variables required for external API integrations
"""

import os
from typing import Dict, Any


class APIConfig:
    """Configuration for all external APIs"""
    
    # LLM APIs
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Salary & Compensation APIs
    GLASSDOOR_API_KEY = os.getenv("GLASSDOOR_API_KEY")
    PAYSCALE_API_KEY = os.getenv("PAYSCALE_API_KEY")
    LEVELS_FYI_API_KEY = os.getenv("LEVELS_FYI_API_KEY")
    
    # Government Data APIs
    BLS_API_KEY = os.getenv("BLS_API_KEY")
    CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    
    # Job Market APIs
    INDEED_API_KEY = os.getenv("INDEED_API_KEY")
    LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
    LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
    
    # Company Data APIs
    CRUNCHBASE_API_KEY = os.getenv("CRUNCHBASE_API_KEY")
    CLEARBIT_API_KEY = os.getenv("CLEARBIT_API_KEY")
    
    # Technology APIs
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    STACKOVERFLOW_API_KEY = os.getenv("STACKOVERFLOW_API_KEY")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM API configuration"""
        return {
            "openai_api_key": cls.OPENAI_API_KEY,
            "anthropic_api_key": cls.ANTHROPIC_API_KEY,
            "openai_available": bool(cls.OPENAI_API_KEY),
            "anthropic_available": bool(cls.ANTHROPIC_API_KEY)
        }
    
    @classmethod
    def get_data_apis_config(cls) -> Dict[str, Any]:
        """Get data APIs configuration"""
        return {
            "salary_apis": {
                "glassdoor": bool(cls.GLASSDOOR_API_KEY),
                "payscale": bool(cls.PAYSCALE_API_KEY),
                "levels_fyi": bool(cls.LEVELS_FYI_API_KEY)
            },
            "government_apis": {
                "bls": bool(cls.BLS_API_KEY),
                "census": bool(cls.CENSUS_API_KEY),
                "fred": bool(cls.FRED_API_KEY)
            },
            "job_market_apis": {
                "indeed": bool(cls.INDEED_API_KEY),
                "linkedin": bool(cls.LINKEDIN_CLIENT_ID and cls.LINKEDIN_CLIENT_SECRET)
            },
            "company_apis": {
                "crunchbase": bool(cls.CRUNCHBASE_API_KEY),
                "clearbit": bool(cls.CLEARBIT_API_KEY)
            },
            "tech_apis": {
                "github": bool(cls.GITHUB_TOKEN),
                "stackoverflow": bool(cls.STACKOVERFLOW_API_KEY)
            }
        }
    
    @classmethod
    def validate_required_apis(cls) -> Dict[str, Any]:
        """Validate that required APIs are configured"""
        llm_config = cls.get_llm_config()
        data_config = cls.get_data_apis_config()
        
        return {
            "llm_apis_ready": llm_config["openai_available"] or llm_config["anthropic_available"],
            "data_apis_ready": any(
                any(api.values()) for api in data_config.values()
            ),
            "recommendations": cls._get_setup_recommendations(llm_config, data_config)
        }
    
    @classmethod
    def _get_setup_recommendations(cls, llm_config: Dict, data_config: Dict) -> list:
        """Get setup recommendations based on current configuration"""
        recommendations = []
        
        if not llm_config["openai_available"] and not llm_config["anthropic_available"]:
            recommendations.append("Set up at least one LLM API (OpenAI or Anthropic) for AI functionality")
        
        if not any(data_config["salary_apis"].values()):
            recommendations.append("Configure salary APIs (Glassdoor, PayScale, or Levels.fyi) for compensation data")
        
        if not any(data_config["government_apis"].values()):
            recommendations.append("Set up government APIs (BLS, Census, or FRED) for economic data")
        
        if not any(data_config["job_market_apis"].values()):
            recommendations.append("Configure job market APIs (Indeed or LinkedIn) for talent supply data")
        
        return recommendations


# Environment variables template
ENV_TEMPLATE = """
# Copy this to your .env file and fill in your API keys

# LLM APIs (Required for AI functionality)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Salary & Compensation APIs (Recommended)
GLASSDOOR_API_KEY=your_glassdoor_api_key
PAYSCALE_API_KEY=your_payscale_api_key
LEVELS_FYI_API_KEY=your_levels_fyi_api_key

# Government Data APIs (Recommended)
BLS_API_KEY=your_bls_api_key
CENSUS_API_KEY=your_census_api_key
FRED_API_KEY=your_fred_api_key

# Job Market APIs (Recommended)
INDEED_API_KEY=your_indeed_api_key
LINKEDIN_CLIENT_ID=your_linkedin_client_id
LINKEDIN_CLIENT_SECRET=your_linkedin_client_secret

# Company Data APIs (Optional)
CRUNCHBASE_API_KEY=your_crunchbase_api_key
CLEARBIT_API_KEY=your_clearbit_api_key

# Technology APIs (Optional)
GITHUB_TOKEN=your_github_token
STACKOVERFLOW_API_KEY=your_stackoverflow_api_key

# Configuration
RATE_LIMIT_PER_MINUTE=60
CACHE_TTL_HOURS=24
"""
