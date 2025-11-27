"""
Hugging Face Integration Utilities
Helper functions for integrating Hugging Face models with SageMaker services
"""

import os
import logging
from typing import Dict, Any, Optional
from .huggingface_model_manager import get_model_manager, initialize_model_manager
from .huggingface_models_config import (
    ModelUseCase,
    get_model_for_use_case,
    get_huggingface_token,
    DEFAULT_HF_TOKEN
)

logger = logging.getLogger(__name__)


def setup_huggingface_for_use_case(use_case: ModelUseCase) -> Dict[str, Any]:
    """
    Set up Hugging Face model for a specific use case
    
    Args:
        use_case: Model use case
        
    Returns:
        Dictionary with model configuration
    """
    # Get model configuration
    model_config = get_model_for_use_case(use_case, priority=1)
    
    if not model_config:
        raise ValueError(f"No model configuration found for {use_case.value}")
    
    # Initialize model manager
    model_manager = initialize_model_manager(token=get_huggingface_token())
    
    # Download model if needed
    model_path = model_manager.get_model_path(use_case, priority=1)
    
    return {
        "model_id": model_config.model_id,
        "model_path": model_path,
        "instance_type": model_config.instance_type,
        "max_tokens": model_config.max_tokens,
        "temperature": model_config.temperature,
        "use_case": use_case.value,
        "description": model_config.description
    }


def get_huggingface_model_id(use_case: ModelUseCase, priority: int = 1) -> str:
    """
    Get Hugging Face model ID for a use case
    
    Args:
        use_case: Model use case
        priority: Model priority
        
    Returns:
        Hugging Face model ID
    """
    model_config = get_model_for_use_case(use_case, priority)
    if not model_config:
        raise ValueError(f"No model found for {use_case.value} with priority {priority}")
    
    return model_config.model_id


def configure_endpoint_for_huggingface(
    use_case: ModelUseCase,
    endpoint_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Configure SageMaker endpoint for Hugging Face model
    
    Args:
        use_case: Model use case
        endpoint_name: Optional endpoint name
        
    Returns:
        Configuration dictionary for endpoint deployment
    """
    model_config = get_model_for_use_case(use_case, priority=1)
    
    if not model_config:
        raise ValueError(f"No model configuration found for {use_case.value}")
    
    # Generate endpoint name if not provided
    if not endpoint_name:
        endpoint_name = f"adeptai-{use_case.value.replace('_', '-')}-v1"
    
    return {
        "endpoint_name": endpoint_name,
        "model_id": model_config.model_id,
        "instance_type": model_config.instance_type,
        "max_tokens": model_config.max_tokens,
        "temperature": model_config.temperature,
        "huggingface_token": get_huggingface_token(),
        "environment": {
            "HF_MODEL_ID": model_config.model_id,
            "HUGGINGFACE_TOKEN": get_huggingface_token(),
            "MODEL_PATH": f"hf://{model_config.model_id}"
        }
    }


def list_suitable_models_for_use_case(use_case: ModelUseCase) -> list:
    """
    List all suitable models for a use case
    
    Args:
        use_case: Model use case
        
    Returns:
        List of model information dictionaries
    """
    model_manager = get_model_manager()
    return model_manager.list_available_models(use_case)


# Environment variable configuration
def setup_huggingface_environment():
    """Set up environment variables for Hugging Face integration"""
    if 'HUGGINGFACE_TOKEN' not in os.environ:
        os.environ['HUGGINGFACE_TOKEN'] = DEFAULT_HF_TOKEN
    
    # Set Hugging Face cache directory
    if 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = '/opt/ml/model/hf_cache'
    
    # Set Hugging Face hub cache
    if 'TRANSFORMERS_CACHE' not in os.environ:
        os.environ['TRANSFORMERS_CACHE'] = '/opt/ml/model/hf_cache'


# Use case to model ID mapping for quick access
USE_CASE_TO_MODEL_ID = {
    ModelUseCase.QUERY_ENHANCEMENT: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ModelUseCase.BEHAVIORAL_ANALYSIS: "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ModelUseCase.MARKET_INTELLIGENCE: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ModelUseCase.JOB_PARSING: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ModelUseCase.EXPLANATION_GENERATION: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ModelUseCase.RESUME_SUMMARIZATION: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ModelUseCase.QUESTION_GENERATION: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ModelUseCase.CANDIDATE_MATCHING: "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def get_default_model_id(use_case: ModelUseCase) -> str:
    """Get default model ID for a use case"""
    return USE_CASE_TO_MODEL_ID.get(use_case, "meta-llama/Meta-Llama-3.1-8B-Instruct")

