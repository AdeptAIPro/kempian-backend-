"""
SageMaker LLM Services
Complete replacement for OpenAI and Claude APIs
Integrated with Hugging Face Hub for latest models
"""

from .sagemaker_llm_client import (
    SageMakerLLMClient,
    ModelType,
    LLMRequest,
    LLMResponse,
    get_sagemaker_client,
    initialize_sagemaker_client
)

from .query_enhancer_service import (
    SageMakerQueryEnhancer,
    QueryEnhancement,
    get_query_enhancer
)

from .behavioral_analyzer_service import (
    SageMakerBehavioralAnalyzer,
    BehavioralProfile,
    get_behavioral_analyzer
)

from .market_intelligence_service import (
    SageMakerMarketIntelligence,
    MarketIntelligence,
    get_market_intelligence
)

from .job_parser_service import (
    SageMakerJobParser,
    JobRequirements,
    get_job_parser
)

from .explanation_generator_service import (
    SageMakerExplanationGenerator,
    Explanation,
    get_explanation_generator
)

# Hugging Face Integration
from .huggingface_models_config import (
    ModelUseCase,
    get_model_for_use_case,
    get_huggingface_token,
    HUGGINGFACE_MODELS
)

from .huggingface_model_manager import (
    HuggingFaceModelManager,
    get_model_manager,
    initialize_model_manager
)

from .huggingface_integration import (
    setup_huggingface_for_use_case,
    get_huggingface_model_id,
    configure_endpoint_for_huggingface,
    list_suitable_models_for_use_case,
    setup_huggingface_environment,
    get_default_model_id
)

__all__ = [
    # Client
    'SageMakerLLMClient',
    'ModelType',
    'LLMRequest',
    'LLMResponse',
    'get_sagemaker_client',
    'initialize_sagemaker_client',
    
    # Query Enhancement
    'SageMakerQueryEnhancer',
    'QueryEnhancement',
    'get_query_enhancer',
    
    # Behavioral Analysis
    'SageMakerBehavioralAnalyzer',
    'BehavioralProfile',
    'get_behavioral_analyzer',
    
    # Market Intelligence
    'SageMakerMarketIntelligence',
    'MarketIntelligence',
    'get_market_intelligence',
    
    # Job Parsing
    'SageMakerJobParser',
    'JobRequirements',
    'get_job_parser',
    
    # Explanation Generation
    'SageMakerExplanationGenerator',
    'Explanation',
    'get_explanation_generator',
    
    # Hugging Face Integration
    'ModelUseCase',
    'get_model_for_use_case',
    'get_huggingface_token',
    'HUGGINGFACE_MODELS',
    'HuggingFaceModelManager',
    'get_model_manager',
    'initialize_model_manager',
    'setup_huggingface_for_use_case',
    'get_huggingface_model_id',
    'configure_endpoint_for_huggingface',
    'list_suitable_models_for_use_case',
    'setup_huggingface_environment',
    'get_default_model_id',
]

