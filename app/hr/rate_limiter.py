from flask import Blueprint, jsonify
from app.auth_utils import get_current_user_flexible, get_current_user
from app.utils.rate_limiter import check_rate_limit
from app.models import User


hr_rate_limiter_bp = Blueprint('hr_rate_limiter', __name__)


@hr_rate_limiter_bp.route('/status', methods=['GET'])
def rate_limiter_status():
    user = get_current_user_flexible() or get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    db_user = User.query.filter_by(email=user.get('email')).first()
    user_id = db_user.id if db_user else 0
    allowed, info = check_rate_limit(user_id, request_type='hr')
    return jsonify({'allowed': allowed, **info}), 200


