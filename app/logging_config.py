"""
Comprehensive Logging Configuration for Kempian Backend
Provides centralized logging setup with file rotation, structured logging, and performance monitoring.
"""

import os
import sys
import logging
from app.logging_config import get_logger
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any
import json
import traceback
from functools import wraps
from flask import request, g, current_app
import time
import io

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'endpoint'):
            log_entry['endpoint'] = record.endpoint
        if hasattr(record, 'method'):
            log_entry['method'] = record.method
        if hasattr(record, 'ip_address'):
            log_entry['ip_address'] = record.ip_address
        if hasattr(record, 'response_time'):
            log_entry['response_time'] = record.response_time
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        if hasattr(record, 'stack_trace'):
            log_entry['stack_trace'] = record.stack_trace
        if hasattr(record, 'database_query'):
            log_entry['database_query'] = record.database_query
        if hasattr(record, 'performance_metrics'):
            log_entry['performance_metrics'] = record.performance_metrics
            
        return json.dumps(log_entry, ensure_ascii=False)

class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that safely handles Unicode encoding errors"""
    
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stderr
        super().__init__(stream)
    
    def emit(self, record):
        """Emit a record, handling Unicode encoding errors gracefully"""
        try:
            msg = self.format(record)
            stream = self.stream
            # Try to write with UTF-8 encoding, fallback to error handling
            try:
                if hasattr(stream, 'buffer'):
                    # For binary streams, encode to UTF-8
                    stream.buffer.write(msg.encode('utf-8', errors='replace'))
                    stream.buffer.write(self.terminator.encode('utf-8'))
                else:
                    # For text streams, try to write directly
                    # If it fails, replace problematic characters
                    try:
                        stream.write(msg + self.terminator)
                    except UnicodeEncodeError:
                        # Replace problematic characters and try again
                        safe_msg = msg.encode('utf-8', errors='replace').decode('utf-8')
                        stream.write(safe_msg + self.terminator)
            except (UnicodeEncodeError, AttributeError):
                # Final fallback: replace all problematic characters
                safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Format the message with colors
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        record.name = f"{log_color}{record.name}{reset_color}"
        
        try:
            return super().format(record)
        except UnicodeEncodeError:
            # If formatting fails due to Unicode, sanitize the message
            msg = record.getMessage()
            safe_msg = msg.encode('utf-8', errors='replace').decode('utf-8')
            record.msg = safe_msg
            record.args = ()
            return super().format(record)

class KempianLogger:
    """Centralized logger for Kempian backend"""
    
    def __init__(self, app=None):
        self.app = app
        self.loggers = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors and Unicode safety
        console_handler = SafeStreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handlers with rotation
        self._setup_file_handlers(log_dir)
        
        # Setup specific loggers
        self._setup_application_loggers()
        
    def _setup_file_handlers(self, log_dir):
        """Setup file handlers with rotation"""
        
        # Main application log
        app_log_file = os.path.join(log_dir, 'kempian_app.log')
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        app_handler.setLevel(logging.DEBUG)
        app_formatter = JSONFormatter()
        app_handler.setFormatter(app_formatter)
        
        # Error log
        error_log_file = os.path.join(log_dir, 'kempian_errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(app_formatter)
        
        # Access log
        access_log_file = os.path.join(log_dir, 'kempian_access.log')
        access_handler = logging.handlers.RotatingFileHandler(
            access_log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        access_handler.setLevel(logging.INFO)
        access_handler.setFormatter(app_formatter)
        
        # Performance log
        perf_log_file = os.path.join(log_dir, 'kempian_performance.log')
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(app_formatter)
        
        # Database log
        db_log_file = os.path.join(log_dir, 'kempian_database.log')
        db_handler = logging.handlers.RotatingFileHandler(
            db_log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        db_handler.setLevel(logging.DEBUG)
        db_handler.setFormatter(app_formatter)
        
        # Security log
        security_log_file = os.path.join(log_dir, 'kempian_security.log')
        security_handler = logging.handlers.RotatingFileHandler(
            security_log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(app_formatter)
        
        # Store handlers for specific loggers
        self.handlers = {
            'app': app_handler,
            'error': error_handler,
            'access': access_handler,
            'performance': perf_handler,
            'database': db_handler,
            'security': security_handler
        }
        
    def _setup_application_loggers(self):
        """Setup specific loggers for different components"""
        
        # Application logger
        app_logger = get_logger("kempian.app")
        app_logger.addHandler(self.handlers['app'])
        app_logger.addHandler(self.handlers['error'])
        app_logger.setLevel(logging.DEBUG)
        
        # Access logger
        access_logger = get_logger("kempian.access")
        access_logger.addHandler(self.handlers['access'])
        access_logger.setLevel(logging.INFO)
        
        # Performance logger
        perf_logger = get_logger("kempian.performance")
        perf_logger.addHandler(self.handlers['performance'])
        perf_logger.setLevel(logging.INFO)
        
        # Database logger
        db_logger = get_logger("kempian.database")
        db_logger.addHandler(self.handlers['database'])
        db_logger.setLevel(logging.DEBUG)
        
        # Security logger
        security_logger = get_logger("kempian.security")
        security_logger.addHandler(self.handlers['security'])
        security_logger.setLevel(logging.WARNING)
        
        # API logger
        api_logger = get_logger("kempian.api")
        api_logger.addHandler(self.handlers['app'])
        api_logger.addHandler(self.handlers['access'])
        api_logger.setLevel(logging.INFO)
        
        # Auth logger
        auth_logger = get_logger("kempian.auth")
        auth_logger.addHandler(self.handlers['app'])
        auth_logger.addHandler(self.handlers['security'])
        auth_logger.setLevel(logging.INFO)
        
        # Search logger
        search_logger = get_logger("kempian.search")
        search_logger.addHandler(self.handlers['app'])
        search_logger.addHandler(self.handlers['performance'])
        search_logger.setLevel(logging.INFO)
        
        # Admin logger
        admin_logger = get_logger("kempian.admin")
        admin_logger.addHandler(self.handlers['app'])
        admin_logger.addHandler(self.handlers['security'])
        admin_logger.setLevel(logging.INFO)
        
        # Stripe logger
        stripe_logger = get_logger("kempian.stripe")
        stripe_logger.addHandler(self.handlers['app'])
        stripe_logger.addHandler(self.handlers['error'])
        stripe_logger.setLevel(logging.INFO)
        
        # Store loggers
        self.loggers = {
            'app': app_logger,
            'access': access_logger,
            'performance': perf_logger,
            'database': db_logger,
            'security': security_logger,
            'api': api_logger,
            'auth': auth_logger,
            'search': search_logger,
            'admin': admin_logger,
            'stripe': stripe_logger
        }
        
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name"""
        if name in self.loggers:
            return self.loggers[name]
        else:
            # Create a new logger if not found
            logger = logging.getLogger(f'kempian.{name}')
            logger.addHandler(self.handlers['app'])
            logger.setLevel(logging.INFO)
            self.loggers[name] = logger
            return logger

# Global logger instance (lazy initialization)
_kempian_logger = None

def get_kempian_logger():
    """Get or create the global logger instance"""
    global _kempian_logger
    if _kempian_logger is None:
        _kempian_logger = KempianLogger()
    return _kempian_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return get_kempian_logger().get_logger(name)

def log_request_info():
    """Log request information"""
    logger = get_logger('access')
    
    # Get request info
    method = request.method
    endpoint = request.endpoint or 'unknown'
    url = request.url
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    # Get user info if available
    user_id = getattr(g, 'user_id', None)
    
    # Generate request ID
    request_id = getattr(g, 'request_id', f"req_{int(time.time() * 1000)}")
    g.request_id = request_id
    
    logger.info(
        f"Request started: {method} {url}",
        extra={
            'request_id': request_id,
            'method': method,
            'endpoint': endpoint,
            'url': url,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'user_id': user_id
        }
    )

def log_response_info(response):
    """Log response information"""
    logger = get_logger('access')
    
    # Get response info
    status_code = response.status_code
    content_length = response.content_length or 0
    
    # Get request info
    request_id = getattr(g, 'request_id', 'unknown')
    method = request.method
    endpoint = request.endpoint or 'unknown'
    url = request.url
    user_id = getattr(g, 'user_id', None)
    
    # Calculate response time
    response_time = getattr(g, 'response_time', 0)
    
    logger.info(
        f"Request completed: {method} {url} - {status_code}",
        extra={
            'request_id': request_id,
            'method': method,
            'endpoint': endpoint,
            'url': url,
            'status_code': status_code,
            'content_length': content_length,
            'response_time': response_time,
            'user_id': user_id
        }
    )
    
    return response

def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics"""
    logger = get_logger('performance')
    
    logger.info(
        f"Performance: {operation} completed in {duration:.3f}s",
        extra={
            'operation': operation,
            'duration': duration,
            'performance_metrics': kwargs
        }
    )

def log_database_query(query: str, duration: float = None, **kwargs):
    """Log database queries"""
    logger = get_logger('database')
    
    logger.debug(
        f"Database query executed",
        extra={
            'database_query': query,
            'duration': duration,
            **kwargs
        }
    )

def log_security_event(event_type: str, message: str, **kwargs):
    """Log security events"""
    logger = get_logger('security')
    
    logger.warning(
        f"Security event: {event_type} - {message}",
        extra={
            'event_type': event_type,
            'security_event': True,
            **kwargs
        }
    )

def log_error(error: Exception, context: str = None, **kwargs):
    """Log errors with full context"""
    logger = get_logger('error')
    
    error_type = type(error).__name__
    error_message = str(error)
    stack_trace = traceback.format_exc()
    
    logger.error(
        f"Error in {context or 'unknown context'}: {error_type} - {error_message}",
        extra={
            'error_type': error_type,
            'error_message': error_message,
            'stack_trace': stack_trace,
            'context': context,
            **kwargs
        }
    )

def log_api_call(endpoint: str, method: str, status_code: int, duration: float, **kwargs):
    """Log API calls"""
    logger = get_logger('api')
    
    logger.info(
        f"API call: {method} {endpoint} - {status_code}",
        extra={
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'duration': duration,
            **kwargs
        }
    )

# Decorators for automatic logging
def log_function_call(logger_name: str = 'app'):
    """Decorator to log function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = time.time()
            
            try:
                logger.debug(f"Calling {func.__name__}")
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"Function {func.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_error(e, f"Function {func.__name__}")
                raise
                
        return wrapper
    return decorator

def log_api_endpoint(logger_name: str = 'api'):
    """Decorator to log API endpoints"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = time.time()
            
            # Log request
            method = request.method
            endpoint = request.endpoint or func.__name__
            user_id = getattr(g, 'user_id', None)
            
            logger.info(f"API endpoint called: {method} {endpoint}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log response
                status_code = 200 if result else 500
                log_api_call(endpoint, method, status_code, duration, user_id=user_id)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_error(e, f"API endpoint {endpoint}")
                raise
                
        return wrapper
    return decorator

# Initialize logging when module is imported
if __name__ != '__main__':
    try:
        get_kempian_logger()._setup_logging()
    except Exception as e:
        print(f"Warning: Could not initialize logging: {e}")
