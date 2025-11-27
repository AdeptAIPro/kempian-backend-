"""
Configuration for Market Intelligence module
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MarketIntelligenceConfig:
    """Configuration class for Market Intelligence module"""
    
    # API Keys for external data sources
    glassdoor_api_key: Optional[str] = None
    indeed_api_key: Optional[str] = None
    linkedin_api_key: Optional[str] = None
    payscale_api_key: Optional[str] = None
    salary_com_api_key: Optional[str] = None
    
    # Data collection settings
    rate_limit_delay: float = 1.0  # seconds between requests
    max_retries: int = 3
    timeout: int = 30  # seconds
    
    # Analysis settings
    default_analysis_period_months: int = 12
    min_data_points_for_analysis: int = 5
    confidence_threshold: float = 0.7
    
    # Caching settings
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    max_cache_size: int = 1000
    
    # Forecast settings
    enable_forecasts: bool = True
    forecast_periods: List[int] = None  # Will be set to [1, 3, 6, 12] if None
    
    # Data sources to use
    enabled_salary_sources: List[str] = None  # Will be set to all if None
    enabled_skill_sources: List[str] = None  # Will be set to all if None
    
    # Industry and location filters
    default_industries: List[str] = None
    default_locations: List[str] = None
    
    def __post_init__(self):
        """Set default values after initialization"""
        if self.forecast_periods is None:
            self.forecast_periods = [1, 3, 6, 12]  # months
        
        if self.enabled_salary_sources is None:
            self.enabled_salary_sources = [
                "glassdoor", "indeed", "linkedin", "payscale", "salary_com"
            ]
        
        if self.enabled_skill_sources is None:
            self.enabled_skill_sources = [
                "linkedin_jobs", "indeed_jobs", "github_trends", "stack_overflow"
            ]
        
        if self.default_industries is None:
            self.default_industries = [
                "technology", "finance", "healthcare", "education", "manufacturing"
            ]
        
        if self.default_locations is None:
            self.default_locations = [
                "Global", "San Francisco", "New York", "Seattle", "Boston"
            ]
    
    @classmethod
    def from_env(cls) -> 'MarketIntelligenceConfig':
        """Create configuration from environment variables"""
        return cls(
            glassdoor_api_key=os.getenv("GLASSDOOR_API_KEY"),
            indeed_api_key=os.getenv("INDEED_API_KEY"),
            linkedin_api_key=os.getenv("LINKEDIN_API_KEY"),
            payscale_api_key=os.getenv("PAYSCALE_API_KEY"),
            salary_com_api_key=os.getenv("SALARY_COM_API_KEY"),
            rate_limit_delay=float(os.getenv("MI_RATE_LIMIT_DELAY", "1.0")),
            max_retries=int(os.getenv("MI_MAX_RETRIES", "3")),
            timeout=int(os.getenv("MI_TIMEOUT", "30")),
            default_analysis_period_months=int(os.getenv("MI_ANALYSIS_PERIOD_MONTHS", "12")),
            min_data_points_for_analysis=int(os.getenv("MI_MIN_DATA_POINTS", "5")),
            confidence_threshold=float(os.getenv("MI_CONFIDENCE_THRESHOLD", "0.7")),
            enable_caching=os.getenv("MI_ENABLE_CACHING", "true").lower() == "true",
            cache_ttl_hours=int(os.getenv("MI_CACHE_TTL_HOURS", "24")),
            max_cache_size=int(os.getenv("MI_MAX_CACHE_SIZE", "1000")),
            enable_forecasts=os.getenv("MI_ENABLE_FORECASTS", "true").lower() == "true"
        )


# Global configuration instance
config = MarketIntelligenceConfig.from_env()


# Data source configurations
SALARY_SOURCE_CONFIGS = {
    "glassdoor": {
        "base_url": "https://api.glassdoor.com/api/api.htm",
        "rate_limit": 0.5,
        "requires_auth": True,
        "data_fields": ["salary", "company", "location", "position"]
    },
    "indeed": {
        "base_url": "https://indeed-indeed.p.rapidapi.com",
        "rate_limit": 0.3,
        "requires_auth": True,
        "data_fields": ["salary", "company", "location", "position"]
    },
    "linkedin": {
        "base_url": "https://api.linkedin.com/v2",
        "rate_limit": 0.4,
        "requires_auth": True,
        "data_fields": ["salary", "company", "location", "position"]
    },
    "payscale": {
        "base_url": "https://www.payscale.com/api",
        "rate_limit": 0.6,
        "requires_auth": False,
        "data_fields": ["salary", "company", "location", "position"]
    },
    "salary_com": {
        "base_url": "https://www.salary.com/api",
        "rate_limit": 0.5,
        "requires_auth": False,
        "data_fields": ["salary", "company", "location", "position"]
    }
}

SKILL_SOURCE_CONFIGS = {
    "linkedin_jobs": {
        "base_url": "https://api.linkedin.com/v2",
        "rate_limit": 0.3,
        "requires_auth": True,
        "data_fields": ["skills", "job_count", "demand_score"]
    },
    "indeed_jobs": {
        "base_url": "https://indeed-indeed.p.rapidapi.com",
        "rate_limit": 0.2,
        "requires_auth": True,
        "data_fields": ["skills", "job_count", "demand_score"]
    },
    "github_trends": {
        "base_url": "https://api.github.com",
        "rate_limit": 0.1,
        "requires_auth": False,
        "data_fields": ["repositories", "stars", "forks", "trending"]
    },
    "stack_overflow": {
        "base_url": "https://api.stackexchange.com/2.3",
        "rate_limit": 0.1,
        "requires_auth": False,
        "data_fields": ["questions", "tags", "trending"]
    }
}

# Industry-specific configurations
INDUSTRY_CONFIGS = {
    "technology": {
        "key_skills": ["Python", "JavaScript", "Machine Learning", "AWS", "Docker"],
        "salary_multiplier": 1.2,
        "growth_rate": 8.5,
        "competition_level": "high"
    },
    "finance": {
        "key_skills": ["SQL", "Python", "Leadership", "Analytics", "Risk Management"],
        "salary_multiplier": 1.15,
        "growth_rate": 5.2,
        "competition_level": "medium"
    },
    "healthcare": {
        "key_skills": ["Leadership", "Communication", "Machine Learning", "Data Analysis"],
        "salary_multiplier": 1.05,
        "growth_rate": 12.3,
        "competition_level": "medium"
    },
    "education": {
        "key_skills": ["Communication", "Leadership", "Teaching", "Curriculum Development"],
        "salary_multiplier": 0.9,
        "growth_rate": 3.0,
        "competition_level": "low"
    },
    "manufacturing": {
        "key_skills": ["Process Improvement", "Leadership", "Quality Control", "Automation"],
        "salary_multiplier": 0.95,
        "growth_rate": 2.5,
        "competition_level": "medium"
    }
}

# Location-specific configurations
LOCATION_CONFIGS = {
    "San Francisco": {
        "salary_multiplier": 1.4,
        "job_density": "high",
        "competition_level": "very_high"
    },
    "New York": {
        "salary_multiplier": 1.3,
        "job_density": "high",
        "competition_level": "very_high"
    },
    "Seattle": {
        "salary_multiplier": 1.2,
        "job_density": "high",
        "competition_level": "high"
    },
    "Boston": {
        "salary_multiplier": 1.15,
        "job_density": "medium",
        "competition_level": "high"
    },
    "Los Angeles": {
        "salary_multiplier": 1.1,
        "job_density": "medium",
        "competition_level": "medium"
    },
    "Chicago": {
        "salary_multiplier": 1.05,
        "job_density": "medium",
        "competition_level": "medium"
    },
    "Austin": {
        "salary_multiplier": 1.0,
        "job_density": "medium",
        "competition_level": "medium"
    },
    "Denver": {
        "salary_multiplier": 0.95,
        "job_density": "low",
        "competition_level": "low"
    },
    "Atlanta": {
        "salary_multiplier": 0.9,
        "job_density": "low",
        "competition_level": "low"
    },
    "Dallas": {
        "salary_multiplier": 0.85,
        "job_density": "low",
        "competition_level": "low"
    },
    "Global": {
        "salary_multiplier": 1.0,
        "job_density": "medium",
        "competition_level": "medium"
    }
}
