from flask import Blueprint, jsonify, request
from app.models import User, db
from app.utils.unlimited_quota_production import (
    production_unlimited_quota_manager as unlimited_quota_manager, 
    is_unlimited_quota_user,
    get_unlimited_quota_info
)
from app.utils.admin_auth import require_admin_auth
import logging

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/unlimited-quota/users', methods=['GET'])
@require_admin_auth
def list_unlimited_quota_users():
    """List all unlimited quota users (admin only)"""
    try:
        users = unlimited_quota_manager.list_unlimited_users()
        stats = unlimited_quota_manager.get_stats()
        
        return jsonify({
            'users': users,
            'stats': stats,
            'message': 'Unlimited quota users retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing unlimited quota users: {e}")
        return jsonify({'error': 'Failed to retrieve unlimited quota users'}), 500

@admin_bp.route('/unlimited-quota/users', methods=['POST'])
@require_admin_auth
def add_unlimited_quota_user():
    """Add a new unlimited quota user (admin only)"""
    try:
        data = request.get_json()
        email = data.get('email')
        reason = data.get('reason', 'Admin granted')
        added_by = data.get('added_by', 'admin')
        quota_limit = data.get('quota_limit', -1)
        daily_limit = data.get('daily_limit', -1)
        monthly_limit = data.get('monthly_limit', -1)
        expires = data.get('expires')
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        # Check if user exists in database
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found in database'}), 404
        
        # Add unlimited quota
        success = unlimited_quota_manager.add_unlimited_user(
            email=email,
            reason=reason,
            added_by=added_by,
            quota_limit=quota_limit,
            daily_limit=daily_limit,
            monthly_limit=monthly_limit,
            expires=expires
        )
        
        if success:
            return jsonify({
                'message': f'Unlimited quota added for {email}',
                'user': get_unlimited_quota_info(email)
            }), 201
        else:
            return jsonify({'error': 'Failed to add unlimited quota'}), 500
            
    except Exception as e:
        logger.error(f"Error adding unlimited quota user: {e}")
        return jsonify({'error': 'Failed to add unlimited quota user'}), 500

@admin_bp.route('/unlimited-quota/users/<email>', methods=['PUT'])
@require_admin_auth
def update_unlimited_quota_user(email):
    """Update unlimited quota user settings (admin only)"""
    try:
        data = request.get_json()
        updates = {}
        
        # Only allow updating specific fields
        allowed_fields = ['quota_limit', 'daily_limit', 'monthly_limit', 'reason', 'expires', 'active']
        for field in allowed_fields:
            if field in data:
                updates[field] = data[field]
        
        if not updates:
            return jsonify({'error': 'No valid fields to update'}), 400
        
        updated_by = data.get('updated_by', 'admin')
        
        # Update user quota
        success = unlimited_quota_manager.update_user_quota(
            email=email,
            updates=updates,
            updated_by=updated_by
        )
        
        if success:
            return jsonify({
                'message': f'Unlimited quota updated for {email}',
                'user': get_unlimited_quota_info(email)
            }), 200
        else:
            return jsonify({'error': 'Failed to update unlimited quota'}), 500
            
    except Exception as e:
        logger.error(f"Error updating unlimited quota user: {e}")
        return jsonify({'error': 'Failed to update unlimited quota user'}), 500

@admin_bp.route('/unlimited-quota/users/<email>', methods=['DELETE'])
@require_admin_auth
def remove_unlimited_quota_user(email):
    """Remove unlimited quota for a user (admin only)"""
    try:
        removed_by = request.json.get('removed_by', 'admin') if request.json else 'admin'
        
        # Remove unlimited quota
        success = unlimited_quota_manager.remove_unlimited_user(
            email=email,
            removed_by=removed_by
        )
        
        if success:
            return jsonify({
                'message': f'Unlimited quota removed for {email}'
            }), 200
        else:
            return jsonify({'error': 'Failed to remove unlimited quota'}), 500
            
    except Exception as e:
        logger.error(f"Error removing unlimited quota user: {e}")
        return jsonify({'error': 'Failed to remove unlimited quota user'}), 500

@admin_bp.route('/unlimited-quota/users/<email>', methods=['GET'])
@require_admin_auth
def get_unlimited_quota_user(email):
    """Get unlimited quota info for a specific user (admin only)"""
    try:
        user_info = get_unlimited_quota_info(email)
        
        if not user_info:
            return jsonify({'error': 'User not found or no unlimited quota'}), 404
        
        return jsonify({
            'user': user_info,
            'message': 'Unlimited quota user info retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting unlimited quota user info: {e}")
        return jsonify({'error': 'Failed to get unlimited quota user info'}), 500

@admin_bp.route('/unlimited-quota/stats', methods=['GET'])
@require_admin_auth
def get_unlimited_quota_stats():
    """Get statistics about unlimited quota users (admin only)"""
    try:
        stats = unlimited_quota_manager.get_stats()
        
        return jsonify({
            'stats': stats,
            'message': 'Unlimited quota stats retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting unlimited quota stats: {e}")
        return jsonify({'error': 'Failed to get unlimited quota stats'}), 500
