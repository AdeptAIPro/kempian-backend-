"""
Custom exceptions for AdeptAI application
Provides specific error types for better error handling and debugging
"""

from typing import Optional, Dict, Any


class AdeptAIError(Exception):
    """Base exception for all AdeptAI errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class ConfigurationError(AdeptAIError):
    """Raised when there's a configuration issue"""
    pass


class ServiceInitializationError(AdeptAIError):
    """Raised when a service fails to initialize"""
    pass


class SearchError(AdeptAIError):
    """Base class for search-related errors"""
    pass


class SearchSystemUnavailableError(SearchError):
    """Raised when search system is not available"""
    pass


class SearchIndexError(SearchError):
    """Raised when there's an issue with the search index"""
    pass


class SearchQueryError(SearchError):
    """Raised when there's an issue with the search query"""
    pass


class MLModelError(AdeptAIError):
    """Base class for ML model-related errors"""
    pass


class ModelLoadingError(MLModelError):
    """Raised when a model fails to load"""
    pass


class ModelPredictionError(MLModelError):
    """Raised when model prediction fails"""
    pass


class EmbeddingError(MLModelError):
    """Raised when embedding generation fails"""
    pass


class DatabaseError(AdeptAIError):
    """Base class for database-related errors"""
    pass


class DynamoDBConnectionError(DatabaseError):
    """Raised when DynamoDB connection fails"""
    pass


class DynamoDBQueryError(DatabaseError):
    """Raised when DynamoDB query fails"""
    pass


class DynamoDBTableNotFoundError(DatabaseError):
    """Raised when DynamoDB table is not found"""
    pass


class AWSConfigurationError(AdeptAIError):
    """Raised when AWS configuration is invalid"""
    pass


class AWSCredentialsError(AWSConfigurationError):
    """Raised when AWS credentials are invalid or missing"""
    pass


class AnalysisError(AdeptAIError):
    """Base class for analysis-related errors"""
    pass


class BehavioralAnalysisError(AnalysisError):
    """Raised when behavioral analysis fails"""
    pass


class BiasDetectionError(AnalysisError):
    """Raised when bias detection fails"""
    pass


class ExplainableAIError(AnalysisError):
    """Raised when explainable AI analysis fails"""
    pass


class ValidationError(AdeptAIError):
    """Raised when input validation fails"""
    pass


class APIError(AdeptAIError):
    """Base class for API-related errors"""
    pass


class RateLimitExceededError(APIError):
    """Raised when rate limit is exceeded"""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails"""
    pass


class AuthorizationError(APIError):
    """Raised when authorization fails"""
    pass


class ResourceNotFoundError(APIError):
    """Raised when a requested resource is not found"""
    pass


class ExternalServiceError(AdeptAIError):
    """Base class for external service errors"""
    pass


class OpenAIServiceError(ExternalServiceError):
    """Raised when OpenAI service fails"""
    pass


class MarketIntelligenceError(ExternalServiceError):
    """Raised when market intelligence service fails"""
    pass


class CacheError(AdeptAIError):
    """Raised when cache operations fail"""
    pass


class FileProcessingError(AdeptAIError):
    """Raised when file processing fails"""
    pass


class DataProcessingError(AdeptAIError):
    """Raised when data processing fails"""
    pass


# Error mapping for HTTP status codes
ERROR_STATUS_MAPPING = {
    ConfigurationError: 500,
    ServiceInitializationError: 503,
    SearchSystemUnavailableError: 503,
    SearchIndexError: 500,
    SearchQueryError: 400,
    ModelLoadingError: 500,
    ModelPredictionError: 500,
    EmbeddingError: 500,
    DynamoDBConnectionError: 503,
    DynamoDBQueryError: 500,
    DynamoDBTableNotFoundError: 404,
    AWSCredentialsError: 500,
    BehavioralAnalysisError: 500,
    BiasDetectionError: 500,
    ExplainableAIError: 500,
    ValidationError: 400,
    RateLimitExceededError: 429,
    AuthenticationError: 401,
    AuthorizationError: 403,
    ResourceNotFoundError: 404,
    OpenAIServiceError: 502,
    MarketIntelligenceError: 502,
    CacheError: 500,
    FileProcessingError: 400,
    DataProcessingError: 400,
}


def get_http_status_code(exception: AdeptAIError) -> int:
    """Get HTTP status code for an exception"""
    return ERROR_STATUS_MAPPING.get(type(exception), 500)


def create_error_response(exception: AdeptAIError) -> Dict[str, Any]:
    """Create a standardized error response"""
    return {
        "error": {
            "code": exception.error_code,
            "message": exception.message,
            "details": exception.details,
            "type": type(exception).__name__
        }
    }
