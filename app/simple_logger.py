"""
Simple Logger for Kempian Backend
A lightweight logging module without circular dependencies
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

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

class SafeFormatter(logging.Formatter):
    """Formatter that safely handles Unicode encoding errors"""
    
    def format(self, record):
        try:
            return super().format(record)
        except UnicodeEncodeError:
            # If formatting fails due to Unicode, sanitize the message
            msg = record.getMessage()
            safe_msg = msg.encode('utf-8', errors='replace').decode('utf-8')
            record.msg = safe_msg
            record.args = ()
            return super().format(record)

class SimpleLogger:
    """Simple logger for Kempian backend"""
    
    def __init__(self):
        self.loggers = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup basic logging configuration"""
        
        # Create logs directory
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        
        # Console handler with Unicode safety
        console_handler = SafeStreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = SafeFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handlers
        self._setup_file_handlers(log_dir)
    
    def _setup_file_handlers(self, log_dir):
        """Setup file handlers"""
        
        # Main app log
        app_log_file = log_dir / 'kempian_app.log'
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        app_handler.setLevel(logging.DEBUG)
        app_formatter = SafeFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        app_handler.setFormatter(app_formatter)
        
        # Error log
        error_log_file = log_dir / 'kempian_errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(app_formatter)
        
        # Access log
        access_log_file = log_dir / 'kempian_access.log'
        access_handler = logging.handlers.RotatingFileHandler(
            access_log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        access_handler.setLevel(logging.INFO)
        access_handler.setFormatter(app_formatter)
        
        # Store handlers
        self.handlers = {
            'app': app_handler,
            'error': error_handler,
            'access': access_handler
        }
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name"""
        if name not in self.loggers:
            logger = logging.getLogger(f'kempian.{name}')
            
            # Add appropriate handlers
            if name in ['auth', 'admin', 'security']:
                logger.addHandler(self.handlers['error'])
            elif name in ['access', 'api']:
                logger.addHandler(self.handlers['access'])
            else:
                logger.addHandler(self.handlers['app'])
            
            logger.setLevel(logging.INFO)
            self.loggers[name] = logger
        
        return self.loggers[name]

# Global logger instance
_simple_logger = None

def get_simple_logger():
    """Get the global simple logger instance"""
    global _simple_logger
    if _simple_logger is None:
        _simple_logger = SimpleLogger()
    return _simple_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return get_simple_logger().get_logger(name)

# Convenience functions
def log_info(message: str, logger_name: str = 'app', **kwargs):
    """Log info message"""
    logger = get_logger(logger_name)
    logger.info(message, extra=kwargs)

def log_error(message: str, logger_name: str = 'app', **kwargs):
    """Log error message"""
    logger = get_logger(logger_name)
    logger.error(message, extra=kwargs)

def log_warning(message: str, logger_name: str = 'app', **kwargs):
    """Log warning message"""
    logger = get_logger(logger_name)
    logger.warning(message, extra=kwargs)

def log_debug(message: str, logger_name: str = 'app', **kwargs):
    """Log debug message"""
    logger = get_logger(logger_name)
    logger.debug(message, extra=kwargs)
