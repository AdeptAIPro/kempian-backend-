"""
Routes for users to view their own activity logs
"""
from flask import Blueprint, jsonify, request
from app.models import UserActivityLog, User, db
from app.utils import get_current_user_flexible
from app.simple_logger import get_logger
from datetime import datetime, timedelta
from sqlalchemy import desc, and_, or_

logger = get_logger("user_activity_logs")

activity_logs_bp = Blueprint("user_activity_logs", __name__)


def _get_current_user():
    """Helper to get current authenticated user"""
    current_user = get_current_user_flexible()
    if not current_user or not current_user.get('email'):
        return None, jsonify({'error': 'Unauthorized'}), 401
    
    user = User.query.filter_by(email=current_user['email']).first()
    if not user:
        return None, jsonify({'error': 'User not found'}), 404
    
    return user, None, None


@activity_logs_bp.route('/activity-logs', methods=['GET'])
def get_user_activity_logs():
    """Get activity logs for the current user"""
    user, error_response, status = _get_current_user()
    if error_response:
        return error_response, status
    
    # Get query parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    activity_type = request.args.get('activity_type', None)
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    search = request.args.get('search', None)
    
    # Build query
    query = UserActivityLog.query.filter_by(user_id=user.id)
    
    # Filter by activity type
    if activity_type:
        query = query.filter(UserActivityLog.activity_type == activity_type)
    
    # Filter by date range
    if start_date:
        try:
            start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            query = query.filter(UserActivityLog.created_at >= start_datetime)
        except ValueError:
            return jsonify({'error': 'Invalid start_date format'}), 400
    
    if end_date:
        try:
            end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            query = query.filter(UserActivityLog.created_at <= end_datetime)
        except ValueError:
            return jsonify({'error': 'Invalid end_date format'}), 400
    
    # Search in action, endpoint, or resource_type
    if search:
        search_filter = or_(
            UserActivityLog.action.ilike(f'%{search}%'),
            UserActivityLog.endpoint.ilike(f'%{search}%'),
            UserActivityLog.resource_type.ilike(f'%{search}%')
        )
        query = query.filter(search_filter)
    
    # Order by created_at descending (newest first)
    query = query.order_by(desc(UserActivityLog.created_at))
    
    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    # Get statistics
    total_logs = UserActivityLog.query.filter_by(user_id=user.id).count()
    today_logs = UserActivityLog.query.filter(
        and_(
            UserActivityLog.user_id == user.id,
            UserActivityLog.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        )
    ).count()
    
    # Get activity type counts
    activity_counts = db.session.query(
        UserActivityLog.activity_type,
        db.func.count(UserActivityLog.id).label('count')
    ).filter_by(user_id=user.id).group_by(UserActivityLog.activity_type).all()
    
    return jsonify({
        'logs': [log.to_dict() for log in pagination.items],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': pagination.total,
            'pages': pagination.pages
        },
        'statistics': {
            'total_logs': total_logs,
            'today_logs': today_logs,
            'activity_type_counts': {activity_type: count for activity_type, count in activity_counts}
        }
    }), 200


@activity_logs_bp.route('/activity-logs/stats', methods=['GET'])
def get_user_activity_stats():
    """Get activity statistics for the current user"""
    user, error_response, status = _get_current_user()
    if error_response:
        return error_response, status
    
    # Get date range (default to last 30 days)
    days = request.args.get('days', 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get activity counts by type
    activity_counts = db.session.query(
        UserActivityLog.activity_type,
        db.func.count(UserActivityLog.id).label('count')
    ).filter(
        and_(
            UserActivityLog.user_id == user.id,
            UserActivityLog.created_at >= start_date
        )
    ).group_by(UserActivityLog.activity_type).all()
    
    # Get daily activity for the period
    daily_activity = db.session.query(
        db.func.date(UserActivityLog.created_at).label('date'),
        db.func.count(UserActivityLog.id).label('count')
    ).filter(
        and_(
            UserActivityLog.user_id == user.id,
            UserActivityLog.created_at >= start_date
        )
    ).group_by(db.func.date(UserActivityLog.created_at)).order_by('date').all()
    
    # Get success/failure ratio
    success_count = UserActivityLog.query.filter(
        and_(
            UserActivityLog.user_id == user.id,
            UserActivityLog.created_at >= start_date,
            UserActivityLog.success == True
        )
    ).count()
    
    failure_count = UserActivityLog.query.filter(
        and_(
            UserActivityLog.user_id == user.id,
            UserActivityLog.created_at >= start_date,
            UserActivityLog.success == False
        )
    ).count()
    
    return jsonify({
        'activity_type_counts': {activity_type: count for activity_type, count in activity_counts},
        'daily_activity': [{'date': str(date), 'count': count} for date, count in daily_activity],
        'success_rate': {
            'success': success_count,
            'failure': failure_count,
            'total': success_count + failure_count,
            'rate': (success_count / (success_count + failure_count) * 100) if (success_count + failure_count) > 0 else 0
        }
    }), 200

