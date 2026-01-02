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
    
    Note: Default URLs use /api/v2 format per Jobvite API specification.
    """
    from app.simple_logger import get_logger
    logger = get_logger("jobvite_utils")
    
    if environment == "stage":
        # Correct Stage URL per Jobvite documentation: api.jvistg2.com
        # NOT api-stg.jobvite.com (which doesn't resolve - NXDOMAIN)
        default_v2 = 'https://api.jvistg2.com/api/v2'
        v2_url = os.getenv('JOBVITE_V2_BASE_URL_STAGE', default_v2)
        if v2_url != default_v2:
            logger.info(f"Using custom JOBVITE_V2_BASE_URL_STAGE: {v2_url}")
        else:
            logger.debug(f"Using default stage URL: {v2_url}")
        return {
            'v2': v2_url,
            'onboarding': os.getenv('JOBVITE_ONBOARDING_BASE_URL_STAGE', 'https://api.jvistg2.com/api/v2')
        }
    else:  # prod
        default_v2 = 'https://api.jobvite.com/api/v2'
        v2_url = os.getenv('JOBVITE_V2_BASE_URL_PROD', default_v2)
        if v2_url != default_v2:
            logger.info(f"Using custom JOBVITE_V2_BASE_URL_PROD: {v2_url}")
        else:
            logger.debug(f"Using default prod URL: {v2_url}")
        return {
            'v2': v2_url,
            'onboarding': os.getenv('JOBVITE_ONBOARDING_BASE_URL_PROD', 'https://onboarding.jobvite.com/api')
        }

