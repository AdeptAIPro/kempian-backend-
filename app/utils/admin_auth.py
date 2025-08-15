import logging
from functools import wraps
from flask import request, jsonify
from app.models import User
from app.utils import get_current_user_flexible

logger = logging.getLogger(__name__)

def require_admin_auth(f):
    """Decorator to require admin authentication for admin routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Get current user from JWT token or custom auth token
            current_user = get_current_user_flexible()
            if not current_user:
                logger.warning("Admin route accessed without authentication")
                return jsonify({'error': 'Authentication required'}), 401
            
            user_email = current_user.get('email')
            if not user_email:
                logger.warning("Admin route accessed without valid email")
                return jsonify({'error': 'Valid user email required'}), 401
            
            # Check if user exists in database and has admin role
            user = User.query.filter_by(email=user_email).first()
            if not user:
                logger.warning(f"Admin route accessed by non-existent user: {user_email}")
                return jsonify({'error': 'User not found'}), 404
            
            # Check if user has admin or owner role
            if user.role not in ['admin', 'owner']:
                logger.warning(f"Admin route accessed by non-admin user: {user_email} (role: {user.role})")
                return jsonify({'error': 'Admin access required'}), 403
            
            logger.info(f"Admin route accessed by authorized user: {user_email} (role: {user.role})")
            return f(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in admin authentication: {e}")
            return jsonify({'error': 'Authentication error'}), 500
    
    return decorated_function
