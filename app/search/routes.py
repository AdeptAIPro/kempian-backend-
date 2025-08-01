import logging
from flask import Blueprint, request, jsonify
from app.models import Tenant, User, Plan, JDSearchLog, TenantAlert, db
from app.emails.ses import send_quota_alert_email
from datetime import datetime
import jwt
import os
from .service import semantic_match, register_feedback
from app.utils.trial_manager import check_and_increment_trial_search, get_user_trial_status, create_user_trial

logger = logging.getLogger(__name__)

search_bp = Blueprint('search', __name__)

# Helper: decode JWT and get tenant/user

def get_jwt_payload():
    auth = request.headers.get('Authorization', None)
    if not auth:
        return None
    token = auth.split(' ')[1]
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload
    except Exception as e:
        logger.error(f"Error decoding JWT: {str(e)}", exc_info=True)
        return None

def get_user_from_jwt(payload):
    """
    Helper function to get user from JWT payload, handling tenant_id=0 cases
    """
    if not payload:
        return None, None
    
    tenant_id = int(payload.get('custom:tenant_id', 0))
    email = payload.get('email')
    
    if not email:
        return None, None
    
    # Handle users with tenant_id=0 (legacy users or JWT issues)
    if tenant_id == 0:
        # Find user by email only
        user = User.query.filter_by(email=email).first()
        if user:
            tenant_id = user.tenant_id
            logger.info(f'Fixed tenant_id for {email}: {user.tenant_id}')
            return user, tenant_id
        else:
            # User doesn't exist in database, create them
            logger.info(f'Creating missing user: {email}')
            user, tenant_id = create_missing_user(payload)
            return user, tenant_id
    else:
        # Ensure user exists in DB with the specified tenant_id
        user = User.query.filter_by(email=email, tenant_id=tenant_id).first()
        if not user:
            logger.error(f'User not found: email={email}, tenant_id={tenant_id}')
            return None, None
        return user, tenant_id

def create_missing_user(payload):
    """
    Create a user in the database if they exist in Cognito but not in DB
    """
    try:
        email = payload.get('email')
        sub = payload.get('sub')
        first_name = payload.get('given_name', '')
        last_name = payload.get('family_name', '')
        role = payload.get('custom:role', 'job_seeker')
        user_type = payload.get('custom:user_type', role)
        
        # Find Starter plan
        starter_plan = Plan.query.filter_by(name="Starter").first()
        if not starter_plan:
            logger.error("❌ Starter plan not found")
            return None, None
        
        # Create tenant
        tenant = Tenant(
            plan_id=starter_plan.id,
            stripe_customer_id="",
            stripe_subscription_id="",
            status="active"
        )
        db.session.add(tenant)
        db.session.commit()
        
        # Create user
        db_user = User(tenant_id=tenant.id, email=email, role="owner", user_type=user_type)
        db.session.add(db_user)
        db.session.commit()
        
        logger.info(f'✅ Created missing user: {email} with tenant_id: {tenant.id}')
        return db_user, tenant.id
        
    except Exception as e:
        logger.error(f'Failed to create missing user: {e}')
        return None, None

@search_bp.route('', methods=['POST'])
def jd_search():
    try:
        payload = get_jwt_payload()
        if not payload:
            logger.error('Unauthorized: No JWT payload')
            return jsonify({'error': 'Unauthorized: No JWT payload'}), 403
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user is on trial
        trial_status = get_user_trial_status(user.id)
        if trial_status:
            # User is on trial, check trial limits
            if not trial_status['is_valid']:
                return jsonify({
                    'error': 'Trial expired. Please upgrade to continue using the service.',
                    'trial_expired': True,
                    'upgrade_required': True
                }), 403
            
            if not trial_status['can_search_today']:
                return jsonify({
                    'error': f'Daily search limit reached ({trial_status["searches_used_today"]}/5). Please upgrade for unlimited searches.',
                    'daily_limit_reached': True,
                    'searches_used_today': trial_status['searches_used_today'],
                    'upgrade_required': True
                }), 429
            
            # Increment trial search count
            can_search, message = check_and_increment_trial_search(user.id)
            if not can_search:
                return jsonify({'error': message, 'trial_error': True}), 429
        
        # For non-trial users, check regular quota
        if not trial_status or not trial_status['is_valid']:
            now = datetime.utcnow()
            first_of_month = datetime(now.year, now.month, 1)
            month_str = now.strftime('%Y-%m')
            
            tenant = Tenant.query.get(tenant_id)
            if tenant:
                plan = Plan.query.get(tenant.plan_id)
            if not plan:
                logger.error(f'Plan not found for tenant_id={tenant_id}')
                return jsonify({'error': f'Plan not found for tenant_id={tenant_id}'}), 400
            
            count = JDSearchLog.query.filter(
                JDSearchLog.tenant_id == tenant_id,
                JDSearchLog.searched_at >= first_of_month
            ).count()
            quota = plan.jd_quota_per_month
            if count + 1 > quota:
                logger.error(f'Monthly quota exceeded: used={count}, quota={quota}, tenant_id={tenant_id}')
                return jsonify({'error': 'Monthly quota exceeded.', 'used': count, 'quota': quota, 'remaining': 0}), 429
            
            # 80% alert
            percent = int(((count + 1) / quota) * 100)
            if percent >= 80:
                alert = TenantAlert.query.filter_by(tenant_id=tenant_id, alert_type='quota_80', alert_month=month_str).first()
                if not alert:
                    owner = User.query.filter_by(tenant_id=tenant_id, role='owner').first()
                    if owner:
                        logger.warning(f"Quota alert: {percent}% used for tenant {tenant_id}, owner: {owner.email}")
                    alert = TenantAlert(tenant_id=tenant_id, alert_type='quota_80', alert_month=month_str)
                    db.session.add(alert)
                    db.session.commit()
            
            # Log search for non-trial users
            log = JDSearchLog(tenant_id=tenant_id, user_id=user.id)
            db.session.add(log)
            db.session.commit()
        
        # Call semantic matching - handle both 'query' and 'job_description' parameters
        request_data = request.get_json()
        job_desc = request_data.get('query') or request_data.get('job_description')
        
        if not job_desc:
            logger.error('No job description or query provided in request')
            return jsonify({'error': 'No job description provided'}), 400
        
        logger.info(f"Processing search for job description: {job_desc[:100]}...")
        results = semantic_match(job_desc)
        
        # Log search results for debugging
        results_count = len(results.get('results', [])) if results else 0
        logger.info(f"Search completed. Found {results_count} candidates for query: {job_desc[:50]}...")
        
        # Add trial info to response if user is on trial
        if trial_status and trial_status['is_valid']:
            results['trial_info'] = {
                'searches_used_today': trial_status['searches_used_today'],
                'days_remaining': trial_status['days_remaining'],
                'is_trial': True
            }
        
        return jsonify(results), 200
    except Exception as e:
        logger.error(f"Error in /search: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@search_bp.route('/quota', methods=['GET'])
def get_quota():
    try:
        payload = get_jwt_payload()
        if not payload:
            logger.error('Unauthorized: No JWT payload')
            return jsonify({'error': 'Unauthorized: No JWT payload'}), 403
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user is on trial
        trial_status = get_user_trial_status(user.id)
        if trial_status and trial_status['is_valid']:
            # Return trial quota info
            return jsonify({
                'quota': 5,  # 5 searches per day
                'used': trial_status['searches_used_today'],
                'remaining': max(0, 5 - trial_status['searches_used_today']),
                'is_trial': True,
                'days_remaining': trial_status['days_remaining'],
                'trial_end_date': trial_status['trial_end_date'].isoformat() if trial_status['trial_end_date'] else None
            }), 200
        
        # For non-trial users, return regular quota info
        now = datetime.utcnow()
        first_of_month = datetime(now.year, now.month, 1)
        tenant = Tenant.query.get(tenant_id)
        plan = Plan.query.get(tenant.plan_id) if tenant else None
        if not plan:
            logger.error(f'Plan not found for tenant_id={tenant_id}')
            return jsonify({'error': f'Plan not found for tenant_id={tenant_id}'}), 400
        
        count = JDSearchLog.query.filter(
            JDSearchLog.tenant_id == tenant_id,
            JDSearchLog.searched_at >= first_of_month
        ).count()
        quota = plan.jd_quota_per_month
        remaining = max(0, quota - count)
        
        return jsonify({
            'quota': quota, 
            'used': count, 
            'remaining': remaining,
            'is_trial': False
        }), 200
    except Exception as e:
        logger.error(f"Error in /search/quota: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@search_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for a candidate to improve future matching"""
    try:
        payload = get_jwt_payload()
        if not payload:
            logger.error('Unauthorized: No JWT payload')
            return jsonify({'error': 'Unauthorized: No JWT payload'}), 403
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        candidate_id = data.get('candidate_id') or data.get('email')
        is_positive = data.get('feedback') == 'good' or data.get('is_positive', False)
        
        if not candidate_id:
            return jsonify({'error': 'Missing candidate_id or email'}), 400
        
        # Register feedback
        register_feedback(candidate_id, positive=is_positive)
        
        logger.info(f'Feedback registered for candidate {candidate_id}: {"positive" if is_positive else "negative"}')
        
        return jsonify({
            'status': 'success',
            'message': f'Feedback registered successfully for candidate {candidate_id}'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /search/feedback: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500