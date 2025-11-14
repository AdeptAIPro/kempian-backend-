from flask import Blueprint, jsonify
from app.auth_utils import get_current_user_flexible, get_current_user
from app.models import User


hr_leave_bp = Blueprint('hr_leave', __name__)


@hr_leave_bp.route('/', methods=['GET'])
def list_leave_items():
    user = get_current_user_flexible() or get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    db_user = User.query.filter_by(email=user.get('email')).first()
    return jsonify({'leave': [], 'user_id': db_user.id if db_user else None}), 200


