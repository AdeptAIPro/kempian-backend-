from flask import Blueprint, jsonify
from app.auth_utils import get_current_user_flexible, get_current_user
from app.models import User, CandidateProfile


hr_documents_bp = Blueprint('hr_documents', __name__)


@hr_documents_bp.route('/', methods=['GET'])
def list_documents():
    user = get_current_user_flexible() or get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'documents': []}), 200
    profile = CandidateProfile.query.filter_by(user_id=db_user.id).first()
    docs = []
    if profile and profile.resume_s3_key:
        docs.append({
            'type': 'resume',
            's3_key': profile.resume_s3_key,
            'filename': profile.resume_filename,
            'uploaded_at': profile.resume_upload_date.isoformat() if profile.resume_upload_date else None
        })
    return jsonify({'documents': docs}), 200


