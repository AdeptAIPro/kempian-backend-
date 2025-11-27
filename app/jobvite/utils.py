"""
Shared utilities for Jobvite integration.
"""

import os
from typing import Dict, Literal

EnvironmentType = Literal["stage", "prod"]
EnvironmentDisplayType = Literal["Stage", "Production"]

def normalize_environment(env: str) -> EnvironmentType:
    """
    Normalize environment string to canonical DB value.
    
    Accepts: "Stage", "Production", "stage", "prod", "staging", "production"
    Returns: "stage" or "prod"
    
    Raises ValueError if invalid.
    """
    env_lower = env.lower().strip()
    
    if env_lower in ("stage", "staging"):
        return "stage"
    elif env_lower in ("prod", "production"):
        return "prod"
    else:
        raise ValueError(f"Invalid environment: {env}. Must be 'Stage'/'Production' or 'stage'/'prod'")

def denormalize_environment(env: EnvironmentType) -> EnvironmentDisplayType:
    """
    Convert DB environment value to human-readable format for API responses.
    
    Args:
        env: "stage" or "prod"
    
    Returns:
        "Stage" or "Production"
    """
    if env == "stage":
        return "Stage"
    elif env == "prod":
        return "Production"
    else:
        raise ValueError(f"Invalid environment: {env}")

def get_base_urls(environment: EnvironmentType) -> Dict[str, str]:
    """
    Get base URLs for Jobvite APIs based on environment.
    
    Args:
        environment: "stage" or "prod"
    
    Returns:
        Dict with 'v2' and 'onboarding' keys
    """
    if environment == "stage":
        return {
            'v2': os.getenv('JOBVITE_V2_BASE_URL_STAGE', 'https://api-stage.jobvite.com/v2'),
            'onboarding': os.getenv('JOBVITE_ONBOARDING_BASE_URL_STAGE', 'https://onboarding-stage.jobvite.com/api')
        }
    else:  # prod
        return {
            'v2': os.getenv('JOBVITE_V2_BASE_URL_PROD', 'https://api.jobvite.com/v2'),
            'onboarding': os.getenv('JOBVITE_ONBOARDING_BASE_URL_PROD', 'https://onboarding.jobvite.com/api')
        }

