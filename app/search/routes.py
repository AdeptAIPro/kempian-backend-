
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

logger = get_logger("search")

# Import scalable search system
try:
    from .integration_guide import get_scalable_integration
    SCALABLE_SYSTEM_AVAILABLE = True
    logger.info("Scalable search system available")
except ImportError as e:
    SCALABLE_SYSTEM_AVAILABLE = False
    logger.warning(f"Scalable search system not available: {e}")

# Initialize scalable search system
if SCALABLE_SYSTEM_AVAILABLE:
    try:
        scalable_integration = get_scalable_integration()
        if not scalable_integration.is_initialized:
            logger.info("Initializing scalable search system...")
            scalable_integration.initialize_system()
            logger.info("Scalable search system initialized successfully")
        else:
            logger.info("Scalable search system already initialized")

        # Kick off a lightweight warmup in background to keep models in RAM
        try:
            import threading
            def _warmup():
                try:
                    from .service import semantic_match
                    semantic_match("warmup: preload models and caches")
                except Exception as _e:
                    logger.warning(f"Warmup failed (non-fatal): {_e}")
            threading.Thread(target=_warmup, daemon=True).start()
            logger.info("Started background warmup thread for search models")
        except Exception as e:
            logger.warning(f"Failed to start warmup thread: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize scalable search system: {e}")
        SCALABLE_SYSTEM_AVAILABLE = False

search_bp = Blueprint('search', __name__)

# Helper: decode JWT and get tenant/user

def get_jwt_payload():
    auth = request.headers.get('Authorization', None)
    if not auth:
        logger.error("No Authorization header found")
        return None
    
    # Check if it's a Bearer token
    if not auth.startswith('Bearer '):
        logger.error(f"Invalid Authorization header format: {auth}")
        return None
    
    try:
        token = auth.split(' ')[1]
        if not token or len(token.split('.')) != 3:
            logger.error(f"Invalid JWT token format: {token}")
            return None
        
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
        logger.error("get_user_from_jwt: No payload provided")
        return None, None
    
    tenant_id = int(payload.get('custom:tenant_id', 0))
    email = payload.get('email')
    
    logger.debug(f"get_user_from_jwt: email={email}, tenant_id={tenant_id}")
    
    if not email:
        logger.error("get_user_from_jwt: No email in payload")
        return None, None
    
    # Always try to find user by email first (more reliable)
    user = User.query.filter_by(email=email).first()
    
    if user:
        # User exists, use their actual tenant_id
        actual_tenant_id = user.tenant_id
        logger.debug(f"get_user_from_jwt: found user {user.id} with actual tenant_id {actual_tenant_id}")
        
        # If JWT has wrong tenant_id, log the mismatch but use the correct one
        if tenant_id != 0 and tenant_id != actual_tenant_id:
            logger.warning(f"JWT tenant_id mismatch: JWT={tenant_id}, DB={actual_tenant_id} for {email}")
        
        return user, actual_tenant_id
    else:
        # User doesn't exist in database, create them
        logger.info(f'Creating missing user: {email}')
        user, tenant_id = create_missing_user(payload)
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
        logger.debug(f"JWT payload received: email={payload.get('email')}, tenant_id={payload.get('custom:tenant_id')}")
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            logger.error(f'User not found for payload: {payload}')
            return jsonify({'error': 'User not found'}), 404
        
        logger.debug(f"User found: {user.email}, tenant_id: {tenant_id}, user_id: {user.id}")
        
        # Check if user has unlimited quota
        is_unlimited = is_unlimited_quota_user(user.email)
        
        logger.debug(f"User {user.email} - Unlimited: {is_unlimited}")
        
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
            
            # Enhanced logging for debugging
            result_count = len(results.get('results', []))
            total_candidates = results.get('total_candidates', result_count)
            algorithm_used = results.get('algorithm_used', 'Unknown')
            
            logger.info(f"Search completed successfully for user {user.email}")
            logger.info(f"Results: {result_count} candidates returned, {total_candidates} total candidates")
            logger.info(f"Algorithm used: {algorithm_used}")
            
            # Log domain distribution if available
            if results.get('results'):
                domain_counts = {}
                for result in results['results']:
                    domain = result.get('category', result.get('domain', 'Unknown'))
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                logger.info(f"Domain distribution: {domain_counts}")
            
            return jsonify(results)
        except Exception as e:
            logger.error(f"Semantic search failed for user {user.email}: {e}", exc_info=True)
            return jsonify({'error': 'Search failed. Please try again.'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in search endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@search_bp.route('/performance-stats', methods=['GET'])
def get_performance_stats():
    """Get comprehensive performance statistics for all search systems"""
    try:
        # Get optimized search stats
        optimized_stats = {}
        try:
            from app.search.optimized_search_service import get_optimized_search_service
            service = get_optimized_search_service()
            if service:
                optimized_stats = service.get_performance_stats()
        except Exception as e:
            logger.warning(f"Could not get optimized search stats: {e}")
        
        # Get search system status
        search_status = {}
        try:
            from app.search.search_initializer import get_search_status
            search_status = get_search_status()
        except Exception as e:
            logger.warning(f"Could not get search status: {e}")
        
        # Get background initialization status
        init_status = {}
        try:
            from app.search.background_initializer import get_initialization_status
            init_status = get_initialization_status()
        except Exception as e:
            logger.warning(f"Could not get initialization status: {e}")
        
        # Get accuracy enhancement stats
        accuracy_stats = {}
        try:
            from app.search.accuracy_enhancement_system import get_accuracy_enhancement_system
            accuracy_system = get_accuracy_enhancement_system()
            if accuracy_system:
                accuracy_stats = accuracy_system.get_performance_stats()
        except Exception as e:
            logger.warning(f"Could not get accuracy enhancement stats: {e}")
        
        # Combine all stats
        performance_data = {
            'timestamp': time.time(),
            'optimized_search': optimized_stats,
            'search_systems': search_status,
            'initialization': init_status,
            'accuracy_enhancement': accuracy_stats,
            'recommendations': _get_performance_recommendations(optimized_stats, search_status)
        }
        
        return jsonify(performance_data)
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        return jsonify({'error': 'Failed to get performance stats'}), 500

@search_bp.route('/test-accuracy', methods=['POST'])
def test_accuracy_enhancement():
    """Test accuracy enhancement system with sample data"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        candidates = data.get('candidates', [])
        
        if not query or not candidates:
            return jsonify({'error': 'Query and candidates are required'}), 400
        
        # Apply accuracy enhancement
        from app.search.accuracy_enhancement_system import enhance_search_accuracy
        
        start_time = time.time()
        enhanced_results = enhance_search_accuracy(query, candidates, top_k=20)
        processing_time = time.time() - start_time
        
        # Calculate accuracy improvement
        original_scores = [c.get('match_percentage', 0) for c in candidates]
        enhanced_scores = [c.get('accuracy_score', 0) for c in enhanced_results]
        
        avg_original = sum(original_scores) / len(original_scores) if original_scores else 0
        avg_enhanced = sum(enhanced_scores) / len(enhanced_scores) if enhanced_scores else 0
        improvement = avg_enhanced - avg_original
        
        return jsonify({
            'success': True,
            'query': query,
            'original_candidates': len(candidates),
            'enhanced_candidates': len(enhanced_results),
            'processing_time': processing_time,
            'accuracy_improvement': improvement,
            'avg_original_score': avg_original,
            'avg_enhanced_score': avg_enhanced,
            'results': enhanced_results[:10]  # Return top 10 for testing
        })
        
    except Exception as e:
        logger.error(f"Error testing accuracy enhancement: {e}")
        return jsonify({'error': 'Failed to test accuracy enhancement'}), 500

def _get_performance_recommendations(optimized_stats, search_status):
    """Get performance recommendations based on current stats"""
    recommendations = []
    
    # Check if ultra-fast search is active
    if optimized_stats.get('total_searches', 0) > 0:
        avg_time = optimized_stats.get('avg_search_time', 0)
        if avg_time < 1.0:
            recommendations.append(" Ultra-fast search is performing excellently!")
        elif avg_time < 3.0:
            recommendations.append(" Search performance is good")
        else:
            recommendations.append(" Search performance could be improved")
    
    # Check cache hit rate
    cache_hit_rate = optimized_stats.get('cache_hit_rate', 0)
    if cache_hit_rate > 0.5:
        recommendations.append(" Cache is working effectively")
    elif cache_hit_rate > 0.2:
        recommendations.append(" Cache is providing some benefit")
    else:
        recommendations.append(" Consider enabling more caching")
    
    # Check if systems are ready
    if search_status.get('is_ready', False):
        recommendations.append(" Search systems are ready")
    else:
        recommendations.append(" Some search systems may not be ready")
    
    return recommendations

@search_bp.route('/performance-stats-old', methods=['GET'])
def get_performance_stats_old():
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

@search_bp.route('/scalable/status', methods=['GET'])
def get_scalable_status():
    """Get status of the scalable search system"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 403
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if not SCALABLE_SYSTEM_AVAILABLE:
            return jsonify({
                'available': False,
                'message': 'Scalable search system not available'
            }), 200
        
        try:
            scalable_integration = get_scalable_integration()
            status = scalable_integration.get_system_status()
            return jsonify({
                'available': True,
                'initialized': scalable_integration.is_initialized,
                'status': status,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        except Exception as e:
            return jsonify({
                'available': True,
                'initialized': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        
    except Exception as e:
        logger.error(f"Error getting scalable status: {e}")
        return jsonify({'error': str(e)}), 500

@search_bp.route('/scalable/initialize', methods=['POST'])
def initialize_scalable_system():
    """Initialize the scalable search system"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 403
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if not SCALABLE_SYSTEM_AVAILABLE:
            return jsonify({
                'success': False,
                'message': 'Scalable search system not available'
            }), 400
        
        try:
            scalable_integration = get_scalable_integration()
            if scalable_integration.is_initialized:
                return jsonify({
                    'success': True,
                    'message': 'Scalable search system already initialized',
                    'initialized': True
                }), 200
            
            # Initialize the system
            scalable_integration.initialize_system()
            
            return jsonify({
                'success': True,
                'message': 'Scalable search system initialized successfully',
                'initialized': True,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            logger.error(f"Failed to initialize scalable system: {e}")
            return jsonify({
                'success': False,
                'message': f'Failed to initialize: {str(e)}',
                'initialized': False
            }), 500
        
    except Exception as e:
        logger.error(f"Error initializing scalable system: {e}")
        return jsonify({'error': str(e)}), 500

@search_bp.route('/scalable/trigger-indexing', methods=['POST'])
def trigger_indexing():
    """Trigger background indexing of candidates"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 403
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if not SCALABLE_SYSTEM_AVAILABLE:
            return jsonify({
                'success': False,
                'message': 'Scalable search system not available'
            }), 400
        
        try:
            scalable_integration = get_scalable_integration()
            if not scalable_integration.is_initialized:
                return jsonify({
                    'success': False,
                    'message': 'Scalable search system not initialized'
                }), 400
            
            # Trigger indexing
            scalable_integration.trigger_indexing()
            
            return jsonify({
                'success': True,
                'message': 'Background indexing triggered successfully',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            logger.error(f"Failed to trigger indexing: {e}")
            return jsonify({
                'success': False,
                'message': f'Failed to trigger indexing: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f"Error triggering indexing: {e}")
        return jsonify({'error': str(e)}), 500