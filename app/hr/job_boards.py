from flask import Blueprint, jsonify
from app.auth_utils import get_current_user_flexible, get_current_user
from app.models import Job


hr_job_boards_bp = Blueprint('hr_job_boards', __name__)


@hr_job_boards_bp.route('/', methods=['GET'])
def list_job_boards():
    user = get_current_user_flexible() or get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    jobs = Job.query.filter_by(is_public=True, status='active').order_by(Job.created_at.desc()).limit(20).all()
    return jsonify({'jobBoards': [j.to_dict() for j in jobs]}), 200


