"""
Logging Middleware for Flask Application
Provides request/response logging, performance monitoring, and error tracking.
"""

import time
import uuid
from flask import request, g, current_app, jsonify
from app.logging_config import get_logger
from functools import wraps
from .logging_config import get_logger, log_request_info, log_response_info, log_performance, log_error

def setup_logging_middleware(app):
    """Setup logging middleware for Flask app"""
    
    @app.before_request
    def before_request():
        """Log request start and setup request context"""
        g.start_time = time.time()
        g.request_id = str(uuid.uuid4())[:8]
        
        # Log request info
        log_request_info()
        
        # Add CORS headers for logging
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200

    @app.after_request
    def after_request(response):
        """Log response and performance metrics"""
        if hasattr(g, 'start_time'):
            g.response_time = time.time() - g.start_time
            
            # Log performance metrics
            log_performance(
                f"{request.method} {request.endpoint or 'unknown'}",
                g.response_time,
                status_code=response.status_code,
                content_length=response.content_length or 0
            )
        
        # Log response info
        log_response_info(response)
        
        return response

    @app.errorhandler(Exception)
    def handle_exception(e):
        """Global exception handler with logging"""
        logger = get_logger('error')
        
        # Log the error
        log_error(e, f"Unhandled exception in {request.endpoint or 'unknown'}")
        
        # Return appropriate response
        if current_app.debug:
            return jsonify({
                'error': str(e),
                'type': type(e).__name__,
                'request_id': getattr(g, 'request_id', 'unknown')
            }), 500
        else:
            return jsonify({
                'error': 'Internal server error',
                'request_id': getattr(g, 'request_id', 'unknown')
            }), 500

    @app.errorhandler(404)
    def handle_404(e):
        """Handle 404 errors with logging"""
        logger = get_logger('access')
        logger.warning(f"404 Not Found: {request.method} {request.url}")
        
        return jsonify({
            'error': 'Not found',
            'path': request.path,
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 404

    @app.errorhandler(405)
    def handle_405(e):
        """Handle 405 Method Not Allowed with logging"""
        logger = get_logger('access')
        logger.warning(f"405 Method Not Allowed: {request.method} {request.url}")
        
        return jsonify({
            'error': 'Method not allowed',
            'method': request.method,
            'path': request.path,
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 405

    @app.errorhandler(400)
    def handle_400(e):
        """Handle 400 Bad Request with logging"""
        logger = get_logger('access')
        logger.warning(f"400 Bad Request: {request.method} {request.url}")
        
        return jsonify({
            'error': 'Bad request',
            'path': request.path,
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 400

    @app.errorhandler(401)
    def handle_401(e):
        """Handle 401 Unauthorized with logging"""
        logger = get_logger('security')
        logger.warning(f"401 Unauthorized: {request.method} {request.url}")
        
        return jsonify({
            'error': 'Unauthorized',
            'path': request.path,
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 401

    @app.errorhandler(403)
    def handle_403(e):
        """Handle 403 Forbidden with logging"""
        logger = get_logger('security')
        logger.warning(f"403 Forbidden: {request.method} {request.url}")
        
        return jsonify({
            'error': 'Forbidden',
            'path': request.path,
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 403

    @app.errorhandler(429)
    def handle_429(e):
        """Handle 429 Too Many Requests with logging"""
        logger = get_logger('security')
        logger.warning(f"429 Rate Limited: {request.method} {request.url}")
        
        return jsonify({
            'error': 'Too many requests',
            'path': request.path,
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 429

    @app.errorhandler(500)
    def handle_500(e):
        """Handle 500 Internal Server Error with logging"""
        logger = get_logger('error')
        logger.error(f"500 Internal Server Error: {request.method} {request.url}")
        
        return jsonify({
            'error': 'Internal server error',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 500

def log_api_call(func):
    """Decorator to log API calls with detailed information"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger('api')
        start_time = time.time()
        
        # Get request context
        method = request.method
        endpoint = request.endpoint or func.__name__
        url = request.url
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', 'Unknown')
        user_id = getattr(g, 'user_id', None)
        request_id = getattr(g, 'request_id', 'unknown')
        
        # Log request details
        logger.info(f"API call started: {method} {endpoint}")
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log successful response
            status_code = 200 if result else 500
            logger.info(f"API call completed: {method} {endpoint} - {status_code} in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            log_error(e, f"API call {method} {endpoint}")
            
            # Re-raise the exception
            raise
            
    return wrapper

def log_database_operation(operation_type: str):
    """Decorator to log database operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('database')
            start_time = time.time()
            
            logger.debug(f"Database operation started: {operation_type}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.debug(f"Database operation completed: {operation_type} in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                log_error(e, f"Database operation {operation_type}")
                raise
                
        return wrapper
    return decorator

def log_authentication_event(event_type: str):
    """Decorator to log authentication events"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('auth')
            start_time = time.time()
            
            # Get request context
            ip_address = request.remote_addr
            user_agent = request.headers.get('User-Agent', 'Unknown')
            
            logger.info(f"Authentication event: {event_type}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful authentication
                logger.info(f"Authentication successful: {event_type} in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log failed authentication
                logger.warning(f"Authentication failed: {event_type} - {str(e)}")
                log_error(e, f"Authentication {event_type}")
                
                raise
                
        return wrapper
    return decorator

def log_admin_operation(operation_type: str):
    """Decorator to log admin operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('admin')
            start_time = time.time()
            
            # Get request context
            user_id = getattr(g, 'user_id', None)
            ip_address = request.remote_addr
            
            logger.info(f"Admin operation started: {operation_type}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(f"Admin operation completed: {operation_type} in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                log_error(e, f"Admin operation {operation_type}")
                raise
                
        return wrapper
    return decorator

def log_search_operation(operation_type: str):
    """Decorator to log search operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('search')
            start_time = time.time()
            
            # Get request context
            user_id = getattr(g, 'user_id', None)
            
            logger.info(f"Search operation started: {operation_type}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log search performance
                log_performance(f"Search {operation_type}", duration, user_id=user_id)
                
                logger.info(f"Search operation completed: {operation_type} in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                log_error(e, f"Search operation {operation_type}")
                raise
                
        return wrapper
    return decorator

def log_stripe_operation(operation_type: str):
    """Decorator to log Stripe operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('stripe')
            start_time = time.time()
            
            # Get request context
            user_id = getattr(g, 'user_id', None)
            
            logger.info(f"Stripe operation started: {operation_type}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(f"Stripe operation completed: {operation_type} in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                log_error(e, f"Stripe operation {operation_type}")
                raise
                
        return wrapper
    return decorator

# Utility functions for manual logging
def log_user_action(user_id: str, action: str, details: dict = None):
    """Log user actions"""
    logger = get_logger('app')
    
    logger.info(
        f"User action: {action}",
        extra={
            'user_id': user_id,
            'action': action,
            'details': details or {},
            'ip_address': request.remote_addr if request else None
        }
    )

def log_system_event(event: str, details: dict = None):
    """Log system events"""
    logger = get_logger('app')
    
    logger.info(
        f"System event: {event}",
        extra={
            'event': event,
            'details': details or {},
            'system_event': True
        }
    )

def log_business_metric(metric_name: str, value: float, unit: str = None, details: dict = None):
    """Log business metrics"""
    logger = get_logger('performance')
    
    logger.info(
        f"Business metric: {metric_name} = {value}",
        extra={
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit,
            'business_metric': True,
            'details': details or {}
        }
    )
