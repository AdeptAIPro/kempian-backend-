"""
Hugging Face Models Configuration
Suitable models for each use case from Hugging Face Hub
Uses provided token: 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
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ModelUseCase(Enum):
    """Model use case enumeration"""
    QUERY_ENHANCEMENT = "query_enhancement"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    MARKET_INTELLIGENCE = "market_intelligence"
    JOB_PARSING = "job_parsing"
    EXPLANATION_GENERATION = "explanation_generation"
    RESUME_SUMMARIZATION = "resume_summarization"
    QUESTION_GENERATION = "question_generation"
    CANDIDATE_MATCHING = "candidate_matching"


@dataclass
class HuggingFaceModel:
    """Hugging Face model configuration"""
    model_id: str
    use_case: ModelUseCase
    description: str
    size: str  # Model size (e.g., "8B", "70B")
    instance_type: str  # Recommended SageMaker instance type
    max_tokens: int
    temperature: float
    priority: int  # 1 = primary, 2 = alternative, 3 = fallback
    notes: Optional[str] = None


# Hugging Face Models Registry
# Using latest and most suitable models from Hugging Face Hub

HUGGINGFACE_MODELS: Dict[ModelUseCase, List[HuggingFaceModel]] = {
    ModelUseCase.QUERY_ENHANCEMENT: [
        HuggingFaceModel(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_case=ModelUseCase.QUERY_ENHANCEMENT,
            description="Meta Llama 3.1 8B Instruct - Excellent for query enhancement and expansion",
            size="8B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.3,
            priority=1,
            notes="Fast, efficient, excellent instruction following"
        ),
        HuggingFaceModel(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            use_case=ModelUseCase.QUERY_ENHANCEMENT,
            description="Mistral 7B Instruct - Cost-effective alternative",
            size="7B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.3,
            priority=2,
            notes="Apache 2.0 license, good performance"
        ),
        HuggingFaceModel(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            use_case=ModelUseCase.QUERY_ENHANCEMENT,
            description="Qwen 2.5 7B Instruct - Multilingual support",
            size="7B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.3,
            priority=2,
            notes="Strong multilingual capabilities"
        ),
        HuggingFaceModel(
            model_id="microsoft/Phi-3-medium-4k-instruct",
            use_case=ModelUseCase.QUERY_ENHANCEMENT,
            description="Phi-3 Medium - Fast and efficient for structured tasks",
            size="14B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.3,
            priority=3,
            notes="Optimized for structured output"
        ),
    ],
    
    ModelUseCase.BEHAVIORAL_ANALYSIS: [
        HuggingFaceModel(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            use_case=ModelUseCase.BEHAVIORAL_ANALYSIS,
            description="Meta Llama 3.1 70B Instruct - Best for complex behavioral analysis",
            size="70B",
            instance_type="ml.g5.12xlarge",
            max_tokens=1024,
            temperature=0.7,
            priority=1,
            notes="Deep reasoning, excellent for nuanced analysis"
        ),
        HuggingFaceModel(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            use_case=ModelUseCase.BEHAVIORAL_ANALYSIS,
            description="Mixtral 8x7B Instruct - Mixture of experts, balanced performance",
            size="47B",
            instance_type="ml.g5.12xlarge",
            max_tokens=1024,
            temperature=0.7,
            priority=2,
            notes="Mixture of experts architecture, good balance"
        ),
        HuggingFaceModel(
            model_id="Qwen/Qwen2.5-72B-Instruct",
            use_case=ModelUseCase.BEHAVIORAL_ANALYSIS,
            description="Qwen 2.5 72B Instruct - Large model with multilingual support",
            size="72B",
            instance_type="ml.g5.12xlarge",
            max_tokens=1024,
            temperature=0.7,
            priority=2,
            notes="Strong reasoning capabilities"
        ),
        HuggingFaceModel(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_case=ModelUseCase.BEHAVIORAL_ANALYSIS,
            description="Meta Llama 3.1 8B Instruct - Faster alternative",
            size="8B",
            instance_type="ml.g5.2xlarge",
            max_tokens=1024,
            temperature=0.7,
            priority=3,
            notes="Faster but less nuanced than 70B"
        ),
    ],
    
    ModelUseCase.MARKET_INTELLIGENCE: [
        HuggingFaceModel(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_case=ModelUseCase.MARKET_INTELLIGENCE,
            description="Meta Llama 3.1 8B Instruct - Good for market analysis",
            size="8B",
            instance_type="ml.g5.2xlarge",
            max_tokens=2048,
            temperature=0.7,
            priority=1,
            notes="Fast enough for real-time analysis"
        ),
        HuggingFaceModel(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            use_case=ModelUseCase.MARKET_INTELLIGENCE,
            description="Mistral 7B Instruct - Cost-effective alternative",
            size="7B",
            instance_type="ml.g5.2xlarge",
            max_tokens=2048,
            temperature=0.7,
            priority=2,
            notes="Good for structured data extraction"
        ),
        HuggingFaceModel(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            use_case=ModelUseCase.MARKET_INTELLIGENCE,
            description="Qwen 2.5 7B Instruct - Multilingual market analysis",
            size="7B",
            instance_type="ml.g5.2xlarge",
            max_tokens=2048,
            temperature=0.7,
            priority=2,
            notes="Multilingual support for global markets"
        ),
    ],
    
    ModelUseCase.JOB_PARSING: [
        HuggingFaceModel(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_case=ModelUseCase.JOB_PARSING,
            description="Meta Llama 3.1 8B Instruct - Excellent for structured extraction",
            size="8B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.3,
            priority=1,
            notes="Fast and accurate structured extraction"
        ),
        HuggingFaceModel(
            model_id="microsoft/Phi-3-medium-4k-instruct",
            use_case=ModelUseCase.JOB_PARSING,
            description="Phi-3 Medium - Optimized for structured output",
            size="14B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.3,
            priority=2,
            notes="Specifically designed for structured tasks"
        ),
        HuggingFaceModel(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            use_case=ModelUseCase.JOB_PARSING,
            description="Mistral 7B Instruct - Good structured extraction",
            size="7B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.3,
            priority=2,
            notes="Reliable structured output"
        ),
    ],
    
    ModelUseCase.EXPLANATION_GENERATION: [
        HuggingFaceModel(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_case=ModelUseCase.EXPLANATION_GENERATION,
            description="Meta Llama 3.1 8B Instruct - Good for natural language explanations",
            size="8B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.7,
            priority=1,
            notes="Natural language generation"
        ),
        HuggingFaceModel(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            use_case=ModelUseCase.EXPLANATION_GENERATION,
            description="Mistral 7B Instruct - Alternative for explanations",
            size="7B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.7,
            priority=2,
            notes="Good conversational abilities"
        ),
    ],
    
    ModelUseCase.RESUME_SUMMARIZATION: [
        HuggingFaceModel(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_case=ModelUseCase.RESUME_SUMMARIZATION,
            description="Meta Llama 3.1 8B Instruct - Good for summarization",
            size="8B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.5,
            priority=1,
            notes="Fast summarization"
        ),
        HuggingFaceModel(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            use_case=ModelUseCase.RESUME_SUMMARIZATION,
            description="Mistral 7B Instruct - Alternative for summarization",
            size="7B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.5,
            priority=2,
            notes="Efficient summarization"
        ),
    ],
    
    ModelUseCase.QUESTION_GENERATION: [
        HuggingFaceModel(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_case=ModelUseCase.QUESTION_GENERATION,
            description="Meta Llama 3.1 8B Instruct - Good for question generation",
            size="8B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.8,
            priority=1,
            notes="Creative question generation"
        ),
        HuggingFaceModel(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            use_case=ModelUseCase.QUESTION_GENERATION,
            description="Mistral 7B Instruct - Alternative for questions",
            size="7B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.8,
            priority=2,
            notes="Good question variety"
        ),
    ],
    
    ModelUseCase.CANDIDATE_MATCHING: [
        HuggingFaceModel(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_case=ModelUseCase.CANDIDATE_MATCHING,
            description="Meta Llama 3.1 8B Instruct - Semantic matching",
            size="8B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.5,
            priority=1,
            notes="Good semantic understanding"
        ),
        HuggingFaceModel(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            use_case=ModelUseCase.CANDIDATE_MATCHING,
            description="Mistral 7B Instruct - Alternative matching",
            size="7B",
            instance_type="ml.g5.xlarge",
            max_tokens=512,
            temperature=0.5,
            priority=2,
            notes="Efficient matching"
        ),
    ],
}


def get_model_for_use_case(
    use_case: ModelUseCase,
    priority: int = 1
) -> Optional[HuggingFaceModel]:
    """
    Get model configuration for a specific use case
    
    Args:
        use_case: Model use case
        priority: Model priority (1 = primary, 2 = alternative, 3 = fallback)
        
    Returns:
        HuggingFaceModel configuration or None
    """
    models = HUGGINGFACE_MODELS.get(use_case, [])
    for model in models:
        if model.priority == priority:
            return model
    # Return first model if priority not found
    return models[0] if models else None


def get_all_models_for_use_case(use_case: ModelUseCase) -> List[HuggingFaceModel]:
    """
    Get all models for a specific use case
    
    Args:
        use_case: Model use case
        
    Returns:
        List of HuggingFaceModel configurations
    """
    return HUGGINGFACE_MODELS.get(use_case, [])


def get_huggingface_token() -> str:
    """Get Hugging Face token from environment"""
    return os.environ.get('HUGGINGFACE_TOKEN', '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')


# Default Hugging Face token
DEFAULT_HF_TOKEN = "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"

