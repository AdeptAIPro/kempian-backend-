import logging
from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import Tenant, User, Plan, JDSearchLog, TenantAlert, UserTrial, db
from app.emails.ses import send_quota_alert_email
from datetime import datetime
import jwt
import os
from .service import semantic_match, register_feedback
from app.utils.trial_manager import check_and_increment_trial_search, get_user_trial_status, create_user_trial
from app.utils.unlimited_quota_production import is_unlimited_quota_user, get_unlimited_quota_info
from app.auth_utils import decode_jwt  # Hardened JWT decoder

logger = get_logger("search")

search_bp = Blueprint('search', __name__)

# Helper: decode JWT and get tenant/user

def get_jwt_payload():
    auth = request.headers.get('Authorization', None)
    if not auth:
        logger.error("No Authorization header found")
        return None

    if not isinstance(auth, str) or not auth.startswith('Bearer '):
        logger.error(f"Invalid Authorization header format: {auth}")
        return None

    parts = auth.split(' ')
    if len(parts) != 2 or not parts[1]:
        logger.error("Malformed Authorization header")
        return None

    token = parts[1]
    try:
        payload = decode_jwt(token)
        if not payload:
            logger.error("JWT verification failed or token expired")
            return None
        if 'email' not in payload:
            logger.error("JWT payload missing email claim")
            return None
        return payload
    except Exception as e:
        logger.error(f"Error verifying JWT: {str(e)}", exc_info=True)
        return None

def get_user_from_jwt(payload):
    """
    Helper function to get user from JWT payload, handling tenant_id=0 cases
    """
    if not payload:
        logger.error("get_user_from_jwt: No payload provided")
        return None, None
    
    tenant_id = int(payload.get('custom:tenant_id', 0))
    email = payload.get('email')
    
    logger.info(f"get_user_from_jwt: email={email}, tenant_id={tenant_id}")
    
    if not email:
        logger.error("get_user_from_jwt: No email in payload")
        return None, None
    
    # Handle users with tenant_id=0 (legacy users or JWT issues)
    if tenant_id == 0:
        logger.info(f"get_user_from_jwt: tenant_id is 0, searching by email only: {email}")
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
        logger.info(f"get_user_from_jwt: searching for user with email={email}, tenant_id={tenant_id}")
        # Ensure user exists in DB with the specified tenant_id
        user = User.query.filter_by(email=email, tenant_id=tenant_id).first()
        if not user:
            logger.error(f'User not found: email={email}, tenant_id={tenant_id}')
            return None, None
        logger.info(f"get_user_from_jwt: found user {user.id} with tenant_id {tenant_id}")
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
        
        logger.info(f"create_missing_user: Creating user {email} with role {role}, user_type {user_type}")
        
        # Find Free Trial plan
        free_trial_plan = Plan.query.filter_by(name="Free Trial").first()
        if not free_trial_plan:
            logger.error("Free Trial plan not found")
            return None, None
        
        logger.info(f"create_missing_user: Found Free Trial plan with ID {free_trial_plan.id}")
        
        # Create tenant
        tenant = Tenant(
            plan_id=free_trial_plan.id,
            stripe_customer_id="",
            stripe_subscription_id="",
            status="active"
        )
        
        logger.info(f"create_missing_user: Creating tenant with plan_id {free_trial_plan.id}")
        db.session.add(tenant)
        db.session.flush()  # Get the tenant ID without committing
        
        logger.info(f"create_missing_user: Tenant created with ID {tenant.id}")
        
        # Create user
        db_user = User(
            tenant_id=tenant.id,
            email=email,
            role=role,
            user_type=user_type
        )
        
        logger.info(f"create_missing_user: Creating user with tenant_id {tenant.id}")
        db.session.add(db_user)
        db.session.flush()  # Get the user ID without committing
        
        logger.info(f"create_missing_user: User created with ID {db_user.id}")
        
        # Create user trial
        logger.info(f"create_missing_user: Creating trial for user {db_user.id}")
        trial = create_user_trial(db_user.id)
        if not trial:
            logger.error(f"create_missing_user: Failed to create trial for user {db_user.id}")
            db.session.rollback()
            return None, None
        
        logger.info(f"create_missing_user: Trial created successfully")
        
        # Commit all changes
        db.session.commit()
        logger.info(f'Created missing user: {email} with tenant_id: {tenant.id}')
        return db_user, tenant.id
        
    except Exception as e:
        logger.error(f'Failed to create missing user: {e}', exc_info=True)
        db.session.rollback()
        return None, None

@search_bp.route('', methods=['POST'])
def jd_search():
    try:
        payload = get_jwt_payload()
        if not payload:
            logger.error('Unauthorized: No JWT payload')
            return jsonify({'error': 'Unauthorized: No JWT payload'}), 403
        
        # Add debug logging
        logger.info(f"JWT payload received: email={payload.get('email')}, tenant_id={payload.get('custom:tenant_id')}")
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            logger.error(f'User not found for payload: {payload}')
            return jsonify({'error': 'User not found'}), 404
        
        logger.info(f"User found: {user.email}, tenant_id: {tenant_id}, user_id: {user.id}")
        
        # Check if user has unlimited quota
        is_unlimited = is_unlimited_quota_user(user.email)
        
        logger.info(f"User {user.email} - Unlimited: {is_unlimited}")
        
        if is_unlimited:
            logger.info(f"User {user.email} has unlimited quota - skipping quota checks")
        else:
            # Check trial status and quota
            trial_status = get_user_trial_status(user.id)
            if not trial_status:
                # No trial found, create one
                logger.info(f"Creating trial for user {user.email}")
                create_user_trial(user.id)
                trial_status = get_user_trial_status(user.id)
            
            # Get the actual trial object for quota checking
            trial = UserTrial.query.filter_by(user_id=user.id).first()
            if not trial:
                logger.error(f"No trial found for user {user.email}")
                return jsonify({'error': 'Trial not found'}), 404
            
            # Check if user can search today using the improved quota logic
            if not trial.can_search_today():
                # Get detailed quota status
                quota_status = trial.get_daily_quota_status()
                
                if not trial.is_trial_valid():
                    reason = "Trial expired"
                    error_details = {
                        'error': reason,
                        'error_type': 'trial_expired',
                        'trial_end_date': trial_status.get('trial_end_date'),
                        'days_expired': abs(trial_status.get('days_remaining', 0)),
                        'trial_duration': 7,  # 7 days free trial
                        'quota': 0,
                        'used': 0,
                        'remaining': 0,
                        'is_trial': True,
                        'upgrade_required': True
                    }
                elif quota_status['searches_used_today'] >= 5:
                    reason = "Daily search limit reached (5 searches)"
                    error_details = {
                        'error': reason,
                        'error_type': 'daily_limit_reached',
                        'trial_duration': 7,  # 7 days free trial
                        'quota': 5,
                        'used': quota_status['searches_used_today'],
                        'remaining': 0,
                        'is_trial': True,
                        'upgrade_required': False,
                        'reset_time': 'tomorrow',
                        'last_search_date': quota_status['last_search_date'].isoformat() if quota_status['last_search_date'] and hasattr(quota_status['last_search_date'], 'isoformat') and callable(getattr(quota_status['last_search_date'], 'isoformat', None)) else None,
                        'is_new_day': quota_status['is_new_day']
                    }
                else:
                    reason = "Cannot search today"
                    error_details = {
                        'error': reason,
                        'error_type': 'search_blocked',
                        'trial_duration': 7,  # 7 days free trial
                        'quota': 5,
                        'used': quota_status['searches_used_today'],
                        'remaining': quota_status['searches_remaining_today'],
                        'is_trial': True,
                        'upgrade_required': False,
                        'last_search_date': quota_status['last_search_date'].isoformat() if quota_status['last_search_date'] and hasattr(quota_status['last_search_date'], 'isoformat') and callable(getattr(quota_status['last_search_date'], 'isoformat', None)) else None,
                        'is_new_day': quota_status['is_new_day']
                    }
                
                logger.warning(f"User {user.email} cannot search: {reason}. Quota status: {quota_status}")
                return jsonify(error_details), 403
                
            logger.info(f"User {user.email} can search. Daily quota: {trial.searches_used_today}/5")
        
        # Get search parameters
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Accept both 'query' and 'job_description' parameters for compatibility
        job_description = data.get('job_description') or data.get('query', '')
        if not job_description:
            return jsonify({'error': 'Job description is required. Please provide either "query" or "job_description" parameter.'}), 400
        
        # Log the search request AND increment trial count in ONE operation
        try:
            # Create search log
            search_log = JDSearchLog(
                user_id=user.id,
                tenant_id=tenant_id,
                job_description=job_description[:1000],  # Limit length
                searched_at=datetime.utcnow()
            )
            db.session.add(search_log)
            
            # Increment trial search count in the same transaction
            if not is_unlimited and trial:
                trial.increment_search_count()
            
            db.session.commit()
            logger.info(f"Search logged and trial count incremented for user {user.email}")
        except Exception as e:
            logger.error(f"Failed to log search or increment trial count: {e}")
            db.session.rollback()
            return jsonify({'error': 'Failed to process search request'}), 500
        
        # Perform semantic search
        try:
            logger.info(f"Starting semantic search for user {user.email} with query: {job_description[:100]}...")
            results = semantic_match(job_description)
            logger.info(f"Search completed successfully for user {user.email}, found {len(results.get('results', []))} results")
            return jsonify(results)
        except Exception as e:
            logger.error(f"Semantic search failed for user {user.email}: {e}", exc_info=True)
            return jsonify({'error': 'Search failed. Please try again.'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in search endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@search_bp.route('/performance-stats', methods=['GET'])
def get_performance_stats():
    """Get comprehensive performance statistics for all AdeptAI components"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 403
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get enhanced performance stats
        from .service import AdeptAIMastersAlgorithm
        algorithm = AdeptAIMastersAlgorithm()
        stats = algorithm.get_enhanced_performance_stats()
        
        return jsonify({
            'success': True,
            'performance_stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f'Error getting performance stats: {e}')
        return jsonify({'error': 'Failed to get performance stats'}), 500

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
        
        # Check if user has unlimited quota
        is_unlimited = is_unlimited_quota_user(user.email)
        
        logger.info(f"User {user.email} - Unlimited: {is_unlimited}")
        
        if is_unlimited:
            unlimited_info = get_unlimited_quota_info(user.email)
            return jsonify({
                'quota': -1,  # -1 indicates unlimited
                'used': 0,
                'remaining': -1,
                'is_trial': False,
                'is_unlimited': True,
                'unlimited_reason': unlimited_info.get('reason', 'Unlimited Quota'),
                'unlimited_added_by': unlimited_info.get('added_by', 'system'),
                'unlimited_added_date': unlimited_info.get('added_date'),
                'message': f"Unlimited quota - {unlimited_info.get('reason', 'Unlimited Quota')}"
            }), 200
        
        # Check if user is on trial
        trial_status = get_user_trial_status(user.id)
        if trial_status and trial_status['is_valid']:
            try:
                # Return trial quota info
                trial_end_date = trial_status['trial_end_date']
                # Handle different date formats safely
                if trial_end_date and hasattr(trial_end_date, 'isoformat') and callable(getattr(trial_end_date, 'isoformat', None)):
                    trial_end_date_str = trial_end_date.isoformat()
                elif isinstance(trial_end_date, str):
                    trial_end_date_str = trial_end_date
                else:
                    trial_end_date_str = None
                    
                return jsonify({
                    'quota': 5,  # 5 searches per day
                    'used': trial_status['searches_used_today'],
                    'remaining': max(0, 5 - trial_status['searches_used_today']),
                    'is_trial': True,
                    'trial_duration': 7,  # 7 days free trial
                    'days_remaining': trial_status['days_remaining'],
                    'trial_end_date': trial_end_date_str,
                    'can_search_today': trial_status['can_search_today']
                }), 200
            except Exception as e:
                logger.error(f"Error formatting trial quota response for user {user.email}: {str(e)}")
                # Fallback response
                return jsonify({
                    'quota': 5,
                    'used': trial_status.get('searches_used_today', 0),
                    'remaining': max(0, 5 - trial_status.get('searches_used_today', 0)),
                    'is_trial': True,
                    'trial_duration': 7,
                    'days_remaining': trial_status.get('days_remaining', 0),
                    'trial_end_date': None,
                    'can_search_today': trial_status.get('can_search_today', False)
                }), 200
        elif trial_status and not trial_status['is_valid']:
            try:
                # Trial expired
                trial_end_date = trial_status['trial_end_date']
                # Handle different date formats safely
                if trial_end_date and hasattr(trial_end_date, 'isoformat') and callable(getattr(trial_end_date, 'isoformat', None)):
                    trial_end_date_str = trial_end_date.isoformat()
                elif isinstance(trial_end_date, str):
                    trial_end_date_str = trial_end_date
                else:
                    trial_end_date_str = None
                    
                return jsonify({
                    'quota': 0,  # No quota when trial expired
                    'used': 0,
                    'remaining': 0,
                    'is_trial': True,
                    'trial_expired': True,
                    'trial_duration': 7,  # 7 days free trial
                    'trial_end_date': trial_end_date_str,
                    'days_expired': abs(trial_status['days_remaining']),
                    'can_search_today': False,
                    'upgrade_required': True,
                    'message': 'Your 7-day free trial has expired. Upgrade to continue searching.'
                }), 200
            except Exception as e:
                logger.error(f"Error formatting expired trial response for user {user.email}: {str(e)}")
                # Fallback response
                return jsonify({
                    'quota': 0,
                    'used': 0,
                    'remaining': 0,
                    'is_trial': True,
                    'trial_expired': True,
                    'trial_duration': 7,
                    'trial_end_date': None,
                    'days_expired': 0,
                    'can_search_today': False,
                    'upgrade_required': True,
                    'message': 'Your 7-day free trial has expired. Upgrade to continue searching.'
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