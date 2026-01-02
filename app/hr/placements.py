from flask import Blueprint, jsonify
from app.auth_utils import get_current_user_flexible, get_current_user
from app.models import JobApplication


hr_placements_bp = Blueprint('hr_placements', __name__)


@hr_placements_bp.route('/', methods=['GET'])
def list_placements():
    user = get_current_user_flexible() or get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    # Interpret placements as hired applications
    hired = JobApplication.query.filter_by(status='hired').order_by(JobApplication.updated_at.desc()).limit(50).all()
    return jsonify({'placements': [a.to_dict() for a in hired]}), 200


