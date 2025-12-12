from flask import Flask, jsonify, request
import logging
from typing import Dict, Any, Tuple

from .exceptions import (
    AdeptAIError, 
    get_http_status_code, 
    create_error_response,
    ConfigurationError,
    ServiceInitializationError,
    SearchSystemUnavailableError,
    ValidationError,
    RateLimitExceededError,
    ResourceNotFoundError
)

logger = logging.getLogger(__name__)


def register_error_handlers(app: Flask) -> None:
    """Register comprehensive error handlers for the Flask application"""
    
    @app.errorhandler(AdeptAIError)
    def handle_adeptai_error(error: AdeptAIError) -> Tuple[Dict[str, Any], int]:
        """Handle custom AdeptAI errors"""
        status_code = get_http_status_code(error)
        logger.error(f"AdeptAI Error: {error.error_code} - {error.message}", extra={
            'error_code': error.error_code,
            'details': error.details
        })
        return create_error_response(error), status_code
    
    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError) -> Tuple[Dict[str, Any], int]:
        """Handle validation errors with detailed feedback"""
        logger.warning(f"Validation Error: {error.message}")
        return create_error_response(error), 400
    
    @app.errorhandler(ServiceInitializationError)
    def handle_service_error(error: ServiceInitializationError) -> Tuple[Dict[str, Any], int]:
        """Handle service initialization errors"""
        logger.error(f"Service Initialization Error: {error.message}")
        return create_error_response(error), 503
    
    @app.errorhandler(SearchSystemUnavailableError)
    def handle_search_unavailable(error: SearchSystemUnavailableError) -> Tuple[Dict[str, Any], int]:
        """Handle search system unavailable errors"""
        logger.error(f"Search System Unavailable: {error.message}")
        return create_error_response(error), 503
    
    @app.errorhandler(ConfigurationError)
    def handle_configuration_error(error: ConfigurationError) -> Tuple[Dict[str, Any], int]:
        """Handle configuration errors"""
        logger.error(f"Configuration Error: {error.message}")
        return create_error_response(error), 500
    
    @app.errorhandler(400)
    def handle_400(err) -> Tuple[Dict[str, Any], int]:
        """Handle bad request errors"""
        logger.warning(f"Bad Request: {str(err)}")
        return _error_response(400, "bad_request", str(err))
    
    @app.errorhandler(404)
    def handle_404(err) -> Tuple[Dict[str, Any], int]:
        """Handle not found errors"""
        logger.warning(f"Not Found: {str(err)}")
        return _error_response(404, "not_found", "The requested resource was not found")
    
    @app.errorhandler(429)
    def handle_429(err) -> Tuple[Dict[str, Any], int]:
        """Handle rate limit errors"""
        logger.warning(f"Rate Limited: {str(err)}")
        return _error_response(429, "rate_limited", "Too many requests")
    
    @app.errorhandler(500)
    def handle_500(err) -> Tuple[Dict[str, Any], int]:
        """Handle internal server errors"""
        logger.error(f"Internal Server Error: {str(err)}")
        return _error_response(500, "server_error", "An unexpected error occurred")
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(err: Exception) -> Tuple[Dict[str, Any], int]:
        """Handle unexpected errors"""
        logger.exception(f"Unexpected Error: {str(err)}")
        return _error_response(500, "unexpected_error", "An unexpected error occurred")


def _error_response(status: int, code: str, message: str) -> Tuple[Dict[str, Any], int]:
    """Create a standardized error response"""
    trace_id = request.headers.get("X-Request-ID")
    response = {
        "error": {
            "code": code,
            "message": message,
            "status": status
        }
    }
    
    if trace_id:
        response["error"]["trace_id"] = trace_id
    
    return response, status


