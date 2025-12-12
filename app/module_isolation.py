"""
Module Isolation System
Ensures each backend module works independently and crashes don't affect other modules.
"""

import time
import threading
from typing import Dict, Optional, Callable, Any, Tuple
from functools import wraps
from flask import Blueprint, jsonify, request, g
from app.simple_logger import get_logger
from enum import Enum

logger = get_logger("module_isolation")


class ModuleStatus(Enum):
    """Module health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"


class ModuleHealth:
    """Tracks health status of a module"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.status = ModuleStatus.HEALTHY
        self.error_count = 0
        self.success_count = 0
        self.last_error_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.total_requests = 0
        self.lock = threading.Lock()
        
        # Circuit breaker settings
        self.failure_threshold = 5  # Open circuit after 5 failures
        self.success_threshold = 2  # Close circuit after 2 successes
        self.timeout = 60  # Timeout in seconds before trying again
        self.circuit_open_time: Optional[float] = None
        
    def record_success(self):
        """Record a successful request"""
        with self.lock:
            self.success_count += 1
            self.total_requests += 1
            self.last_success_time = time.time()
            
            # If circuit is open and we have enough successes, close it
            if self.status == ModuleStatus.CIRCUIT_OPEN:
                if self.success_count >= self.success_threshold:
                    self.status = ModuleStatus.HEALTHY
                    self.error_count = 0
                    self.circuit_open_time = None
                    logger.info(f"Module {self.module_name} circuit breaker closed - service restored")
            elif self.error_count > 0:
                # Reset error count on success
                self.error_count = 0
                if self.status == ModuleStatus.DEGRADED:
                    self.status = ModuleStatus.HEALTHY
                    
    def record_error(self):
        """Record an error"""
        with self.lock:
            self.error_count += 1
            self.total_requests += 1
            self.last_error_time = time.time()
            
            # Update status based on error rate
            error_rate = self.error_count / max(self.total_requests, 1)
            
            if error_rate > 0.5 and self.error_count >= self.failure_threshold:
                self.status = ModuleStatus.CIRCUIT_OPEN
                self.circuit_open_time = time.time()
                logger.warning(f"Module {self.module_name} circuit breaker opened - too many failures")
            elif error_rate > 0.2:
                self.status = ModuleStatus.DEGRADED
                logger.warning(f"Module {self.module_name} is degraded - error rate: {error_rate:.2%}")
            else:
                self.status = ModuleStatus.HEALTHY
                
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        with self.lock:
            if self.status == ModuleStatus.CIRCUIT_OPEN:
                # Check if timeout has passed
                if self.circuit_open_time and (time.time() - self.circuit_open_time) > self.timeout:
                    # Try to close circuit (half-open state)
                    self.status = ModuleStatus.DEGRADED
                    logger.info(f"Module {self.module_name} circuit breaker entering half-open state")
                    return False
                return True
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get current health status"""
        with self.lock:
            return {
                "module": self.module_name,
                "status": self.status.value,
                "error_count": self.error_count,
                "success_count": self.success_count,
                "total_requests": self.total_requests,
                "error_rate": self.error_count / max(self.total_requests, 1),
                "last_error_time": self.last_error_time,
                "last_success_time": self.last_success_time,
                "circuit_open": self.status == ModuleStatus.CIRCUIT_OPEN
            }


class ModuleIsolationManager:
    """Manages isolation and health monitoring for all modules"""
    
    def __init__(self):
        self.modules: Dict[str, ModuleHealth] = {}
        self.lock = threading.Lock()
        
    def get_module_health(self, module_name: str) -> ModuleHealth:
        """Get or create module health tracker"""
        with self.lock:
            if module_name not in self.modules:
                self.modules[module_name] = ModuleHealth(module_name)
            return self.modules[module_name]
            
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all modules"""
        with self.lock:
            return {name: health.get_status() for name, health in self.modules.items()}
            
    def reset_module(self, module_name: str):
        """Reset a module's health status"""
        with self.lock:
            if module_name in self.modules:
                self.modules[module_name] = ModuleHealth(module_name)
                logger.info(f"Module {module_name} health status reset")


# Global module isolation manager
_isolation_manager = ModuleIsolationManager()


def get_isolation_manager() -> ModuleIsolationManager:
    """Get the global module isolation manager"""
    return _isolation_manager


def isolated_route(module_name: str):
    """
    Decorator to isolate a route handler - catches all errors and prevents crashes
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            health = _isolation_manager.get_module_health(module_name)
            
            # Check circuit breaker
            if health.is_circuit_open():
                logger.warning(f"Module {module_name} circuit breaker is open - rejecting request")
                return jsonify({
                    'error': f'{module_name} service is temporarily unavailable',
                    'status': 'circuit_open',
                    'module': module_name
                }), 503
                
            try:
                # Execute the route handler
                result = func(*args, **kwargs)
                
                # Record success
                health.record_success()
                
                return result
                
            except Exception as e:
                # Record error
                health.record_error()
                
                # Handle database errors specifically
                from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
                if isinstance(e, (SQLAlchemyError, OperationalError, IntegrityError)):
                    logger.error(
                        f"Database error in {module_name} module route {func.__name__}: {str(e)}",
                        exc_info=True,
                        extra={
                            'module_name': module_name,
                            'route': func.__name__,
                            'error_type': 'database_error'
                        }
                    )
                    # Rollback any pending transaction
                    try:
                        from app import db
                        db.session.rollback()
                    except Exception:
                        pass
                else:
                    logger.error(
                        f"Error in {module_name} module route {func.__name__}: {str(e)}",
                        exc_info=True,
                        extra={
                            'module_name': module_name,
                            'route': func.__name__,
                            'error_type': type(e).__name__
                        }
                    )
                
                # Return error response without crashing
                return jsonify({
                    'error': f'An error occurred in {module_name} module',
                    'message': str(e) if g.get('debug', False) else 'Internal server error',
                    'module': module_name,
                    'status': health.status.value
                }), 500
                
        return wrapper
    return decorator


def safe_blueprint_register(app, blueprint: Blueprint, url_prefix: str, module_name: str) -> bool:
    """
    Safely register a blueprint with error isolation.
    Returns True if successful, False otherwise.
    """
    health = _isolation_manager.get_module_health(module_name)
    
    try:
        # Add handlers BEFORE registering the blueprint
        @blueprint.before_request
        def _ensure_module_available():
            if health.is_circuit_open():
                logger.warning(f"Module {module_name} circuit breaker open - rejecting request to {request.path}")
                return jsonify({
                    'error': f'{module_name} service is temporarily unavailable',
                    'status': 'circuit_open',
                    'module': module_name
                }), 503
            return None
        
        # Add error handler for this blueprint - catches all exceptions
        @blueprint.errorhandler(Exception)
        def handle_blueprint_error(e):
            health.record_error()
            
            # Handle database errors specifically
            from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
            if isinstance(e, (SQLAlchemyError, OperationalError, IntegrityError)):
                logger.error(
                    f"Database error in {module_name} blueprint: {str(e)}",
                    exc_info=True,
                    extra={'module_name': module_name, 'error_type': 'database_error'}
                )
                # Rollback any pending transaction
                try:
                    from app import db
                    db.session.rollback()
                except Exception:
                    pass
            else:
                logger.error(
                    f"Error in {module_name} blueprint: {str(e)}",
                    exc_info=True,
                    extra={'module_name': module_name, 'error_type': type(e).__name__}
                )
            
            return jsonify({
                'error': f'An error occurred in {module_name} module',
                'message': str(e) if app.debug else 'Internal server error',
                'module': module_name
            }), 500
            
        # Add 404 handler for this blueprint
        @blueprint.errorhandler(404)
        def handle_blueprint_404(e):
            return jsonify({
                'error': 'Not found',
                'module': module_name,
                'path': request.path
            }), 404
        
        @blueprint.after_request
        def _record_success(response):
            if response.status_code < 500:
                health.record_success()
            return response
        
        # Now register the blueprint with all handlers attached
        app.register_blueprint(blueprint, url_prefix=url_prefix)
            
        health.record_success()
        logger.info(f"Successfully registered {module_name} blueprint at {url_prefix}")
        return True
        
    except Exception as e:
        health.record_error()
        logger.error(
            f"Failed to register {module_name} blueprint: {str(e)}",
            exc_info=True,
            extra={'module_name': module_name, 'url_prefix': url_prefix}
        )
        return False


def safe_import_and_register(app, module_path: str, blueprint_name: str, 
                             url_prefix: str, module_name: str, 
                             required: bool = False) -> Optional[Blueprint]:
    """
    Safely import and register a blueprint module.
    Returns the blueprint if successful, None otherwise.
    """
    health = _isolation_manager.get_module_health(module_name)
    
    try:
        # Import the module
        module = __import__(module_path, fromlist=[blueprint_name])
        blueprint = getattr(module, blueprint_name)
        
        if not isinstance(blueprint, Blueprint):
            raise ValueError(f"{blueprint_name} is not a Blueprint instance")
            
        # Register with isolation
        if safe_blueprint_register(app, blueprint, url_prefix, module_name):
            return blueprint
        else:
            if required:
                raise RuntimeError(f"Failed to register required module {module_name}")
            return None
            
    except ImportError as e:
        health.record_error()
        if required:
            logger.error(f"Required module {module_name} not found: {str(e)}")
            raise
        else:
            logger.warning(f"Optional module {module_name} not available: {str(e)}")
            return None
    except Exception as e:
        health.record_error()
        logger.error(
            f"Error importing/registering {module_name}: {str(e)}",
            exc_info=True,
            extra={'module_name': module_name, 'module_path': module_path}
        )
        if required:
            raise
        return None


def wrap_blueprint_routes(blueprint: Blueprint, module_name: str):
    """
    Wrap all routes in a blueprint with isolation decorator.
    This is a fallback for routes that weren't decorated individually.
    """
    # Note: This is tricky because Flask doesn't expose a clean way to wrap existing routes
    # The isolated_route decorator should be used on individual routes instead
    pass

