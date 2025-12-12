"""
Admin Activity Logging Decorator
Automatically logs admin activities when they access admin routes
"""

import time
import json
from functools import wraps
from flask import request, g, jsonify
from app.services.admin_activity_logger import AdminActivityLogger
from app.simple_logger import get_logger

logger = get_logger("admin_activity_decorator")

def log_admin_activity(action_name=None, include_request_data=False):
    """
    Decorator to automatically log admin activities
    
    Args:
        action_name: Custom name for the action (defaults to endpoint name)
        include_request_data: Whether to include request data in logs
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            status_code = 200
            error_message = None
            
            try:
                # Get admin information from Flask g (set by admin_auth decorator)
                admin_email = getattr(g, 'admin_email', None)
                admin_id = getattr(g, 'admin_id', None)
                admin_role = getattr(g, 'admin_role', None)
                tenant_id = getattr(g, 'tenant_id', None)
                
                if not admin_email:
                    logger.warning("Admin activity logging called without admin context")
                    return f(*args, **kwargs)
                
                # Determine action name
                action = action_name or f"{request.method} {request.endpoint}"
                
                # Prepare request data if requested
                request_data = None
                if include_request_data:
                    try:
                        if request.is_json:
                            request_data = request.get_json()
                        elif request.form:
                            request_data = dict(request.form)
                        elif request.args:
                            request_data = dict(request.args)
                    except Exception as e:
                        logger.warning(f"Could not capture request data: {e}")
                
                # Execute the original function
                result = f(*args, **kwargs)
                
                # Calculate response time
                response_time_ms = int((time.time() - start_time) * 1000)
                
                # Get status code from result if it's a tuple
                if isinstance(result, tuple) and len(result) >= 2:
                    status_code = result[1] if isinstance(result[1], int) else 200
                
                # Log the activity
                AdminActivityLogger.log_admin_action(
                    admin_email=admin_email,
                    admin_id=admin_id,
                    admin_role=admin_role,
                    action=action,
                    endpoint=request.endpoint,
                    method=request.method,
                    request_data=request_data,
                    status_code=status_code,
                    response_time_ms=response_time_ms,
                    tenant_id=tenant_id
                )
                
                return result
                
            except Exception as e:
                # Calculate response time even for errors
                response_time_ms = int((time.time() - start_time) * 1000)
                error_message = str(e)
                status_code = 500
                
                # Log the error activity
                admin_email = getattr(g, 'admin_email', None)
                admin_id = getattr(g, 'admin_id', None)
                admin_role = getattr(g, 'admin_role', None)
                tenant_id = getattr(g, 'tenant_id', None)
                
                if admin_email:
                    AdminActivityLogger.log_admin_action(
                        admin_email=admin_email,
                        admin_id=admin_id,
                        admin_role=admin_role,
                        action=action_name or f"{request.method} {request.endpoint}",
                        endpoint=request.endpoint,
                        method=request.method,
                        request_data=request_data if include_request_data else None,
                        status_code=status_code,
                        response_time_ms=response_time_ms,
                        error_message=error_message,
                        tenant_id=tenant_id
                    )
                
                # Re-raise the exception
                raise
        
        return decorated_function
    return decorator

def log_admin_login_activity(f):
    """Special decorator for admin login activities"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        try:
            # Execute the original function
            result = f(*args, **kwargs)
            
            # Check if login was successful (status code 200)
            if isinstance(result, tuple) and len(result) >= 2:
                status_code = result[1] if isinstance(result[1], int) else 200
            else:
                status_code = 200
            
            if status_code == 200:
                # Extract admin information from the result
                try:
                    if isinstance(result, tuple) and len(result) >= 1:
                        response_data = result[0]
                        if hasattr(response_data, 'get_json'):
                            data = response_data.get_json()
                        else:
                            data = response_data
                        
                        user_data = data.get('user', {})
                        admin_email = user_data.get('email')
                        admin_id = user_data.get('id')
                        admin_role = user_data.get('role')
                        tenant_id = user_data.get('tenant_id')
                        
                        if admin_email and admin_role in ['admin', 'owner']:
                            # Log the admin login
                            AdminActivityLogger.log_admin_login(
                                admin_email=admin_email,
                                admin_id=admin_id,
                                admin_role=admin_role,
                                tenant_id=tenant_id
                            )
                            
                            # Store admin info in Flask g for future requests
                            g.admin_email = admin_email
                            g.admin_id = admin_id
                            g.admin_role = admin_role
                            g.tenant_id = tenant_id
                            
                except Exception as e:
                    logger.warning(f"Could not extract admin info from login response: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in admin login activity logging: {e}")
            # Re-raise the exception
            raise
        
        finally:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.debug(f"Admin login activity processed in {response_time_ms}ms")
    
    return decorated_function

def log_admin_logout_activity(f):
    """Special decorator for admin logout activities"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Get admin information from Flask g
            admin_email = getattr(g, 'admin_email', None)
            session_id = getattr(g, 'admin_session_id', None)
            
            # Execute the original function
            result = f(*args, **kwargs)
            
            # Log logout if admin was logged in
            if admin_email:
                AdminActivityLogger.log_admin_logout(
                    admin_email=admin_email,
                    session_id=session_id
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in admin logout activity logging: {e}")
            # Re-raise the exception
            raise
    
    return decorated_function
