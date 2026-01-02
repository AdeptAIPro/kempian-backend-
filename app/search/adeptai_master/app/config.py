import os
import secrets
from typing import List, Optional, Dict
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings using Pydantic for validation and type safety"""
    
    # Pydantic v2 settings: allow extra env keys and load .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Security
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="Secret key for session management")
    session_cookie_secure: bool = Field(default=True, description="Use secure cookies")
    
    # CORS
    cors_allowed_origins: str = Field(
        default="http://localhost:3000", 
        description="Allowed CORS origins (comma-separated)"
    )
    
    # Rate Limiting
    ratelimit_default: str = Field(default="100/minute", description="Default rate limit")
    ratelimit_storage_uri: str = Field(default="memory://", description="Rate limit storage URI")
    
    # Request Configuration
    max_content_length: int = Field(default=10485760, description="Max request size in bytes (10MB)")
    
    # Search Configuration
    search_index_path: str = Field(default="enhanced_search_index", description="Path to search index")
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key ID", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret access key", alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="ap-south-1", description="AWS region", alias="AWS_REGION")
    
    # DynamoDB Configuration
    dynamodb_table_name: str = Field(default="user-resume-metadata", description="Main DynamoDB table name", alias="DYNAMODB_TABLE_NAME")
    dynamodb_feedback_table: str = Field(default="career-user-table", description="Feedback table name", alias="DYNAMODB_FEEDBACK_TABLE")
    
    # SageMaker Configuration
    use_sagemaker: bool = Field(default=False, description="Use SageMaker endpoints instead of local models", alias="USE_SAGEMAKER")
    sagemaker_region: str = Field(default="ap-south-1", description="AWS region for SageMaker endpoints", alias="SAGEMAKER_REGION")
    
    # SageMaker Endpoint Names (set these after deploying models in Jupyter Lab)
    sagemaker_domain_classifier_endpoint: Optional[str] = Field(
        default=None, 
        description="SageMaker endpoint name for ML Domain Classifier",
        alias="SAGEMAKER_DOMAIN_CLASSIFIER_ENDPOINT"
    )
    sagemaker_ltr_endpoint: Optional[str] = Field(
        default=None,
        description="SageMaker endpoint name for Learning to Rank model",
        alias="SAGEMAKER_LTR_ENDPOINT"
    )
    sagemaker_dense_retrieval_endpoint: Optional[str] = Field(
        default=None,
        description="SageMaker endpoint name for Dense Retrieval model",
        alias="SAGEMAKER_DENSE_RETRIEVAL_ENDPOINT"
    )
    sagemaker_llm_enhancer_endpoint: Optional[str] = Field(
        default=None,
        description="SageMaker endpoint name for LLM Query Enhancer",
        alias="SAGEMAKER_LLM_ENHANCER_ENDPOINT"
    )
    sagemaker_job_fit_endpoint: Optional[str] = Field(
        default=None,
        description="SageMaker endpoint name for Job Fit Predictor",
        alias="SAGEMAKER_JOB_FIT_ENDPOINT"
    )
    sagemaker_embedding_endpoint: Optional[str] = Field(
        default=None,
        description="SageMaker endpoint name for Embedding Service",
        alias="SAGEMAKER_EMBEDDING_ENDPOINT"
    )
    
    # Hugging Face Configuration (preferred for local models)
    huggingface_token: str = Field(
        default="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",
        description="Hugging Face token for accessing models",
        alias="HUGGINGFACE_TOKEN"
    )
    huggingface_model: str = Field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        description="Hugging Face model ID for query enhancement",
        alias="HUGGINGFACE_MODEL"
    )
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key", alias="ANTHROPIC_API_KEY")
    
    # LLM Configuration
    llm_provider: str = Field(default="huggingface", description="Default LLM provider (huggingface, openai, anthropic, auto)")
    llm_device: Optional[str] = Field(default=None, description="Device for Hugging Face models (cuda, cpu, or None for auto)")
    
    # Model Configuration
    model_type: str = Field(default="lightgbm", description="ML model type")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    structured_logging: bool = Field(default=True, description="Enable structured (JSON) logging in production")
    
    # Tracing / Observability
    tracing_request_id_headers: str = Field(
        default="X-Request-ID,X-Trace-ID,X-Correlation-ID,Traceparent",
        description="Comma-separated list of headers to pull request IDs from"
    )
    expose_config_endpoint: bool = Field(default=True, description="Expose a read-only config endpoint for FE (non-sensitive)")
    
    # Application Configuration
    debug: bool = Field(default=False, description="Debug mode")
    testing: bool = Field(default=False, description="Testing mode")
    
    # Feature Flags (Subsystems)
    enable_instant_search: bool = Field(default=True, description="Enable instant search subsystem")
    enable_behavioural_analysis: bool = Field(default=True, description="Enable behavioural analysis subsystem")
    enable_bias_prevention: bool = Field(default=True, description="Enable bias prevention subsystem")
    enable_explainable_ai: bool = Field(default=True, description="Enable explainable AI subsystem")
    enable_market_intelligence: bool = Field(default=True, description="Enable market intelligence subsystem")
    
    def get_cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list"""
        return [origin.strip() for origin in self.cors_allowed_origins.split(',') if origin.strip()]
    
    def get_tracing_headers_list(self) -> List[str]:
        """Get tracing headers as a list (in priority order)"""
        return [h.strip() for h in self.tracing_request_id_headers.split(',') if h.strip()]
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Return feature flags as a dictionary for easy export"""
        return {
            'instant_search': self.enable_instant_search,
            'behavioural_analysis': self.enable_behavioural_analysis,
            'bias_prevention': self.enable_bias_prevention,
            'explainable_ai': self.enable_explainable_ai,
            'market_intelligence': self.enable_market_intelligence,
        }
    
    def get_sagemaker_config(self) -> Dict[str, Optional[str]]:
        """Return SageMaker endpoint configuration as a dictionary"""
        return {
            'domain_classifier_endpoint': self.sagemaker_domain_classifier_endpoint,
            'ltr_endpoint': self.sagemaker_ltr_endpoint,
            'dense_retrieval_endpoint': self.sagemaker_dense_retrieval_endpoint,
            'llm_enhancer_endpoint': self.sagemaker_llm_enhancer_endpoint,
            'job_fit_endpoint': self.sagemaker_job_fit_endpoint,
            'embedding_endpoint': self.sagemaker_embedding_endpoint,
        }
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    @validator('max_content_length')
    def validate_max_content_length(cls, v):
        if v < 1024:  # At least 1KB
            raise ValueError('max_content_length must be at least 1024 bytes')
        if v > 100 * 1024 * 1024:  # Max 100MB
            raise ValueError('max_content_length cannot exceed 100MB')
        return v
    
    # Note: legacy Config removed to be compatible with Pydantic v2 model_config


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)"""
    return Settings()


# Legacy AppConfig for Flask compatibility
class AppConfig:
    """Legacy configuration class for Flask compatibility"""
    
    def __init__(self):
        settings = get_settings()
        
        # Security
        self.SECRET_KEY = settings.secret_key
        self.SESSION_COOKIE_SECURE = settings.session_cookie_secure
        
        # CORS
        self.CORS_ALLOWED_ORIGINS = settings.get_cors_origins_list()
        
        # Limits
        self.RATELIMIT_DEFAULT = settings.ratelimit_default
        self.RATELIMIT_STORAGE_URI = settings.ratelimit_storage_uri
        
        # Request size
        self.MAX_CONTENT_LENGTH = settings.max_content_length
        
        # Search/Index
        self.SEARCH_INDEX_PATH = settings.search_index_path
