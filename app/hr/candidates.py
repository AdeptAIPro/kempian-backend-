from flask import Blueprint, request, jsonify
from app.db import db
from app.models import CandidateProfile, CandidateSkill
from app.auth_utils import get_current_user, get_current_user_flexible, get_current_tenant_id


hr_candidates_bp = Blueprint('hr_candidates', __name__)


def _require_user():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


@hr_candidates_bp.route('/', methods=['GET'])
def list_candidates():
    user, error = _require_user()
    if error:
        return error

    # Scope by tenant if available; otherwise return user's own profile
    tenant_id = get_current_tenant_id()
    if tenant_id:
        profiles = CandidateProfile.query.join(CandidateProfile.user).filter_by(tenant_id=tenant_id).order_by(CandidateProfile.created_at.desc()).all()
    else:
        from app.models import User
        db_user = User.query.filter_by(email=user['email']).first()
        if not db_user:
            return jsonify({'candidates': []}), 200
        profiles = CandidateProfile.query.filter_by(user_id=db_user.id).all()

    return jsonify({'candidates': [p.to_dict() for p in profiles]}), 200


@hr_candidates_bp.route('/<int:candidate_id>', methods=['GET'])
def get_candidate(candidate_id):
    user, error = _require_user()
    if error:
        return error

    profile = CandidateProfile.query.get(candidate_id)
    if not profile:
        return jsonify({'error': 'Candidate not found'}), 404

    return jsonify(profile.to_dict()), 200


@hr_candidates_bp.route('/', methods=['POST'])
def create_candidate():
    user, error = _require_user()
    if error:
        return error

    from app.models import User
    db_user = User.query.filter_by(email=user['email']).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json() or {}
    full_name = data.get('fullName') or data.get('full_name')
    if not full_name:
        return jsonify({'error': 'fullName is required'}), 400

    profile = CandidateProfile(
        user_id=db_user.id,
        full_name=full_name,
        phone=data.get('phone'),
        location=data.get('location'),
        summary=data.get('summary'),
        experience_years=data.get('experienceYears') or data.get('experience_years')
    )
    db.session.add(profile)
    db.session.flush()

    skills = data.get('skills') or []
    if isinstance(skills, list):
        for skill_name in skills:
            db.session.add(CandidateSkill(profile_id=profile.id, skill_name=str(skill_name)))

    db.session.commit()
    return jsonify(profile.to_dict()), 201


@hr_candidates_bp.route('/<int:candidate_id>', methods=['PUT', 'PATCH'])
def update_candidate(candidate_id):
    user, error = _require_user()
    if error:
        return error

    profile = CandidateProfile.query.get(candidate_id)
    if not profile:
        return jsonify({'error': 'Candidate not found'}), 404

    data = request.get_json() or {}
    if 'fullName' in data or 'full_name' in data:
        profile.full_name = data.get('fullName') or data.get('full_name') or profile.full_name
    if 'phone' in data:
        profile.phone = data.get('phone')
    if 'location' in data:
        profile.location = data.get('location')
    if 'summary' in data:
        profile.summary = data.get('summary')
    if 'experienceYears' in data or 'experience_years' in data:
        profile.experience_years = data.get('experienceYears') or data.get('experience_years')

    # Replace skills if provided
    if 'skills' in data and isinstance(data['skills'], list):
        # delete old
        for s in list(profile.skills):
            db.session.delete(s)
        for skill_name in data['skills']:
            db.session.add(CandidateSkill(profile_id=profile.id, skill_name=str(skill_name)))

    db.session.commit()
    return jsonify(profile.to_dict()), 200


@hr_candidates_bp.route('/<int:candidate_id>', methods=['DELETE'])
def delete_candidate(candidate_id):
    user, error = _require_user()
    if error:
        return error

    profile = CandidateProfile.query.get(candidate_id)
    if not profile:
        return jsonify({'error': 'Candidate not found'}), 404

    db.session.delete(profile)
    db.session.commit()
    return jsonify({'message': 'Candidate deleted successfully'}), 200


