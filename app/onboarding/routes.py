from flask import Blueprint, request, jsonify
from app.models import db, User, OnboardingFlag, OnboardingSubmission
from app.utils import get_current_user, get_current_user_flexible
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger("onboarding")

onboarding_bp = Blueprint('onboarding', __name__)

def _get_authed_user():
    u = get_current_user_flexible()
    if not u or not u.get('email'):
        return None, jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(email=u['email']).first()
    if not user:
        return None, jsonify({'error': 'User not found'}), 404
    return user, None, None

@onboarding_bp.route('/onboarding/status', methods=['GET'])
def onboarding_status():
    user, resp, code = _get_authed_user()
    if resp:
        return resp, code
    flag = OnboardingFlag.query.filter_by(user_id=user.id).first()
    required = bool(flag.required) if flag else False
    completed = bool(flag.completed) if flag else False
    latest = OnboardingSubmission.query.filter_by(user_id=user.id).order_by(OnboardingSubmission.created_at.desc()).first()
    return jsonify({
        'required': required and not completed,
        'completed': completed,
        'data': latest.data if latest else None
    })

@onboarding_bp.route('/onboarding/submit', methods=['POST'])
def onboarding_submit():
    user, resp, code = _get_authed_user()
    if resp:
        return resp, code
    payload = request.get_json() or {}
    submission = OnboardingSubmission(user_id=user.id, data=payload, created_at=datetime.utcnow())
    db.session.add(submission)
    # upsert flag
    flag = OnboardingFlag.query.filter_by(user_id=user.id).first()
    if not flag:
        flag = OnboardingFlag(user_id=user.id, required=False, completed=True)
        db.session.add(flag)
    else:
        flag.completed = True
        flag.required = False
        flag.updated_at = datetime.utcnow()
    db.session.commit()
    # Send a thank-you email to the user (best-effort)
    try:
        from app.emails.smtp import send_onboarding_thanks_email_smtp
        contact = payload.get('contactName') or payload.get('full_name') or (user.first_name or '')
        company_name = payload.get('companyName') or payload.get('company_name') or (user.company_name or '')
        services = payload.get('services')
        highlights = []
        if isinstance(services, list) and services:
            highlights.append('Services of interest: ' + ', '.join([str(s) for s in services]))
        if payload.get('goals'):
            highlights.append('Goals: ' + str(payload.get('goals'))[:200])
        send_onboarding_thanks_email_smtp(user.email, contact, company_name, highlights)
    except Exception as e:
        logger.warning(f"Onboarding thank-you email skipped: {e}")
    
    # Send notification email to support@kempian.ai with all onboarding details (best-effort)
    try:
        from app.emails.smtp import send_onboarding_notification_to_support_smtp
        contact_name = payload.get('contactName') or payload.get('full_name') or (user.first_name or '')
        company_name = payload.get('companyName') or payload.get('company_name') or (user.company_name or '')
        send_onboarding_notification_to_support_smtp(
            user.email,
            contact_name,
            company_name,
            payload
        )
    except Exception as e:
        logger.warning(f"Onboarding notification email to support skipped: {e}")
    
    return jsonify({'ok': True, 'submission_id': submission.id}), 201

@onboarding_bp.route('/onboarding/require', methods=['POST'])
def onboarding_require():
    # Admin/owner only
    admin = get_current_user()
    if not admin or admin.get('custom:role') not in ['admin','owner']:
        return jsonify({'error': 'Forbidden'}), 403
    data = request.get_json() or {}
    email = data.get('email')
    required = bool(data.get('required', True))
    if not email:
        return jsonify({'error': 'email required'}), 400
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    flag = OnboardingFlag.query.filter_by(user_id=user.id).first()
    if not flag:
        flag = OnboardingFlag(user_id=user.id, required=required, completed=False)
        db.session.add(flag)
    else:
        flag.required = required
        if required:
            flag.completed = False
        flag.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'ok': True}), 200

@onboarding_bp.route('/onboarding/delete', methods=['DELETE','OPTIONS'])
def onboarding_delete():
    """Delete all onboarding submissions and reset the flag for the current user"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        resp = jsonify({'ok': True})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'DELETE, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
        return resp, 200

    user, resp, code = _get_authed_user()
    if resp:
        return resp, code
    
    try:
        # Delete all submissions for this user
        OnboardingSubmission.query.filter_by(user_id=user.id).delete()
        
        # Reset or delete the flag
        flag = OnboardingFlag.query.filter_by(user_id=user.id).first()
        if flag:
            flag.completed = False
            flag.required = False
            flag.updated_at = datetime.utcnow()
        else:
            # Create a new flag if it doesn't exist (shouldn't happen, but be safe)
            flag = OnboardingFlag(user_id=user.id, required=False, completed=False)
            db.session.add(flag)
        
        db.session.commit()
        logger.info(f"Deleted onboarding data for user {user.email}")
        resp_json = jsonify({'ok': True, 'message': 'Onboarding data deleted successfully'})
        resp_json.headers['Access-Control-Allow-Origin'] = '*'
        return resp_json, 200
    except Exception as e:
        logger.error(f"Failed to delete onboarding data: {e}")
        db.session.rollback()
        resp_err = jsonify({'error': 'Failed to delete onboarding data'})
        resp_err.headers['Access-Control-Allow-Origin'] = '*'
        return resp_err, 500


