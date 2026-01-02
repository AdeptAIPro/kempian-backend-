"""
Utility for logging user activities across the platform
"""
import json
import time
from functools import wraps
from flask import request, g
from app.models import UserActivityLog, db
from app.utils import get_current_user_flexible
from app.simple_logger import get_logger

logger = get_logger("user_activity")


def log_user_activity(activity_type, action=None, resource_type=None, resource_id=None, 
                     include_request_data=False, sanitize_keys=None):
    """
    Decorator to log user activities
    
    Args:
        activity_type: Type of activity (e.g., 'login', 'search', 'upload', 'view')
        action: Specific action performed (optional)
        resource_type: Type of resource being acted upon (optional)
        resource_id: ID of the resource (optional)
        include_request_data: Whether to include request data in log
        sanitize_keys: List of keys to remove from request data for security
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = None
            error_message = None
            success = True
            
            # Get current user
            current_user = get_current_user_flexible()
            user_email = current_user.get('email') if current_user else None
            user_id = None
            user_role = None
            
            if user_email:
                from app.models import User
                user = User.query.filter_by(email=user_email).first()
                if user:
                    user_id = user.id
                    user_role = user.role
            
            # Get request details
            ip_address = request.remote_addr
            user_agent = request.headers.get('User-Agent')
            endpoint = request.path
            method = request.method
            
            # Get request data if needed
            request_data = None
            if include_request_data:
                try:
                    if request.is_json:
                        request_data_dict = request.get_json() or {}
                    elif request.form:
                        request_data_dict = dict(request.form)
                    else:
                        request_data_dict = {}
                    
                    # Sanitize sensitive data
                    if sanitize_keys:
                        for key in sanitize_keys:
                            if key in request_data_dict:
                                request_data_dict[key] = "***REDACTED***"
                    
                    # Remove None values and limit size
                    request_data_dict = {k: v for k, v in request_data_dict.items() if v is not None}
                    if request_data_dict:
                        request_data = json.dumps(request_data_dict)[:5000]  # Limit size
                except Exception as e:
                    logger.warning(f"Failed to capture request data: {e}")
            
            # Get session ID if available
            session_id = request.headers.get('X-Session-Id') or g.get('session_id')
            
            # Get tenant ID if available
            tenant_id = g.get('tenant_id')
            
            try:
                # Execute the function
                response = f(*args, **kwargs)
                
                # Get status code from response
                if hasattr(response, 'status_code'):
                    status_code = response.status_code
                elif isinstance(response, tuple) and len(response) > 1:
                    status_code = response[1] if isinstance(response[1], int) else 200
                else:
                    status_code = 200
                
                # Check if it's an error response
                if status_code and status_code >= 400:
                    success = False
                    try:
                        if isinstance(response, tuple) and len(response) > 0:
                            response_data = response[0]
                            if hasattr(response_data, 'get_json'):
                                error_data = response_data.get_json()
                                error_message = error_data.get('error', 'Unknown error')[:500]
                    except:
                        pass
                
                return response
                
            except Exception as e:
                success = False
                error_message = str(e)[:500]
                status_code = 500
                raise
                
            finally:
                # Calculate response time
                response_time_ms = int((time.time() - start_time) * 1000)
                
                # Mark as logged to prevent duplicate logging by middleware
                g.activity_logged = True
                
                # Log the activity
                try:
                    if user_id:  # Only log if we have a valid user
                        activity_log = UserActivityLog(
                            user_email=user_email,
                            user_id=user_id,
                            user_role=user_role,
                            activity_type=activity_type,
                            action=action,
                            endpoint=endpoint,
                            method=method,
                            resource_type=resource_type,
                            resource_id=str(resource_id) if resource_id else None,
                            ip_address=ip_address,
                            user_agent=user_agent,
                            request_data=request_data,
                            status_code=status_code,
                            response_time_ms=response_time_ms,
                            tenant_id=tenant_id,
                            session_id=session_id,
                            error_message=error_message,
                            success=success
                        )
                        db.session.add(activity_log)
                        db.session.commit()
                except Exception as e:
                    logger.error(f"Failed to log user activity: {e}")
                    db.session.rollback()
        
        return wrapper
    return decorator


def log_user_action(user_id, user_email, user_role, activity_type, action=None, 
                   endpoint=None, method=None, resource_type=None, resource_id=None,
                   ip_address=None, user_agent=None, status_code=None, success=True,
                   error_message=None, request_data=None, tenant_id=None, session_id=None):
    """
    Direct function to log a user activity without using decorator
    
    Useful for logging activities that don't happen in route handlers
    """
    try:
        activity_log = UserActivityLog(
            user_email=user_email,
            user_id=user_id,
            user_role=user_role,
            activity_type=activity_type,
            action=action,
            endpoint=endpoint or (request.path if request else None),
            method=method or (request.method if request else None),
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id else None,
            ip_address=ip_address or (request.remote_addr if request else None),
            user_agent=user_agent or (request.headers.get('User-Agent') if request else None),
            request_data=request_data,
            status_code=status_code,
            response_time_ms=None,
            tenant_id=tenant_id,
            session_id=session_id,
            error_message=error_message,
            success=success
        )
        db.session.add(activity_log)
        db.session.commit()
    except Exception as e:
        logger.error(f"Failed to log user action: {e}")
        db.session.rollback()


def auto_log_user_activity(response):
    """
    Automatically log user activity for all authenticated requests.
    This should be called from Flask's after_request hook.
    
    Note: This will NOT log if the route already uses @log_user_activity decorator
    to avoid duplicate logging. The decorator sets g.activity_logged = True.
    
    Args:
        response: Flask response object
        
    Returns:
        The response object (unchanged)
    """
    try:
        # Skip if already logged by decorator
        if g.get('activity_logged', False):
            return response
        
        # Skip logging for certain paths
        path = request.path or ''
        
        # Skip public endpoints
        public_prefixes = (
            '/', '/health', '/public', '/auth/login', '/auth/register', 
            '/webhook', '/checkout', '/api/linkedin', '/api/health',
            '/admin/user-activity-logs', '/admin/user-activity-logs/stats',  # Skip logging the logging endpoints themselves
            '/admin/user-activity-logs/users'  # Skip logging the stats endpoints
        )
        
        for prefix in public_prefixes:
            if path == prefix or path.startswith(prefix + '/'):
                return response
        
        # Skip OPTIONS requests
        if request.method == 'OPTIONS':
            return response
        
        # Get current user
        current_user = get_current_user_flexible()
        if not current_user:
            return response
        
        user_email = current_user.get('email')
        if not user_email:
            return response
        
        # Get user from database
        from app.models import User
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return response
        
        user_id = user.id
        user_role = user.role
        
        # Get request details
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent')
        endpoint = request.path
        method = request.method
        
        # Get status code from response
        status_code = None
        if hasattr(response, 'status_code'):
            status_code = response.status_code
        elif isinstance(response, tuple) and len(response) > 1:
            status_code = response[1] if isinstance(response[1], int) else 200
        else:
            status_code = 200
        
        # Determine success
        success = status_code < 400 if status_code else True
        
        # Get error message if any
        error_message = None
        if not success and hasattr(response, 'get_json'):
            try:
                error_data = response.get_json()
                if error_data and isinstance(error_data, dict):
                    error_message = error_data.get('error', 'Unknown error')[:500]
            except:
                pass
        
        # Get response time from g if available
        response_time_ms = None
        if hasattr(g, 'response_time'):
            response_time_ms = int(g.response_time * 1000)
        elif hasattr(g, 'start_time'):
            import time
            response_time_ms = int((time.time() - g.start_time) * 1000)
        
        # Determine activity type from endpoint and method
        activity_type = _infer_activity_type(endpoint, method)
        
        # Get session ID if available
        session_id = request.headers.get('X-Session-Id') or g.get('session_id')
        
        # Get tenant ID if available
        tenant_id = g.get('tenant_id')
        
        # Log the activity
        try:
            activity_log = UserActivityLog(
                user_email=user_email,
                user_id=user_id,
                user_role=user_role,
                activity_type=activity_type,
                action=None,  # Can be set by specific decorators if needed
                endpoint=endpoint,
                method=method,
                resource_type=None,  # Can be inferred from endpoint if needed
                resource_id=None,
                ip_address=ip_address,
                user_agent=user_agent,
                request_data=None,  # Don't log request data by default for performance
                status_code=status_code,
                response_time_ms=response_time_ms,
                tenant_id=tenant_id,
                session_id=session_id,
                error_message=error_message,
                success=success
            )
            db.session.add(activity_log)
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to auto-log user activity: {e}")
            db.session.rollback()
        
    except Exception as e:
        # Don't let logging errors break the request
        logger.error(f"Error in auto_log_user_activity: {e}")
    
    return response


def _infer_activity_type(endpoint, method):
    """
    Infer activity type from endpoint and HTTP method.
    
    Args:
        endpoint: The request endpoint path
        method: HTTP method (GET, POST, etc.)
        
    Returns:
        Activity type string
    """
    endpoint_lower = endpoint.lower()
    
    # Login/logout
    if 'login' in endpoint_lower:
        return 'login'
    if 'logout' in endpoint_lower:
        return 'logout'
    
    # Search
    if 'search' in endpoint_lower:
        return 'search'
    
    # Upload/download
    if 'upload' in endpoint_lower:
        return 'upload'
    if 'download' in endpoint_lower:
        return 'download'
    
    # CRUD operations based on method
    if method == 'GET':
        return 'view'
    elif method == 'POST':
        return 'create'
    elif method == 'PUT' or method == 'PATCH':
        return 'update'
    elif method == 'DELETE':
        return 'delete'
    
    # Default
    return 'api_call'