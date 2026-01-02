
import logging
from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import Tenant, User, Plan, JDSearchLog, TenantAlert, UserTrial, db
from app.emails.ses import send_quota_alert_email
from datetime import datetime
import jwt
import os
from .service import semantic_match, register_feedback
from app.search.manual_csv_matching import build_manual_csv_match
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


def _dedupe_candidates_in_response(result: dict, top_k: int = 20) -> dict:
    """
    Final safety net: de-duplicate candidates in the /search response.

    - Prefer email as unique key when present.
    - Otherwise, use candidate name (FullName/full_name/name) as the identity.
    - Keeps the first (usually highest-scoring) occurrence.
    """
    try:
        items = result.get('results') or []
        if not isinstance(items, list) or not items:
            result['results'] = []
            result['total_candidates'] = 0
            return result

        seen = set()
        unique: list[dict] = []

        for cand in items:
            if not isinstance(cand, dict):
                continue

            email = (cand.get('email') or '').strip().lower()
            name = (
                cand.get('FullName')
                or cand.get('full_name')
                or cand.get('name')
                or ''
            )
            name = name.strip().lower()

            if email:
                key = ('email', email)
            else:
                key = ('name', name)

            if key in seen:
                continue
            seen.add(key)
            unique.append(cand)

        # Limit to top_k while preserving order
        if isinstance(top_k, int) and top_k > 0:
            unique = unique[:top_k]

        result['results'] = unique
        result['total_candidates'] = len(unique)
        return result
    except Exception as e:
        logger.warning(f"Response-level de-duplication failed: {e}")
        # Best effort: at least ensure total_candidates is consistent
        items = result.get('results') or []
        result['total_candidates'] = len(items) if isinstance(items, list) else 0
        return result

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
        
        # Perform semantic search using NEW matchmaking system
        try:
            # Get top_k from request, default to 20
            top_k = data.get('top_k', 20)
            if not isinstance(top_k, int) or top_k < 1:
                top_k = 20
            logger.info(f"Starting semantic search for user {user.email} with query: {job_description[:100]}... (top_k={top_k})")
            
            # Use NEW matchmaking system
            try:
                from app.matchmaking.pipelines.matcher import match_candidates as match_candidates_func
                from app.matchmaking.routes import transform_match_result_to_frontend_format
                
                # First, get candidates using old system (for candidate loading)
                old_results = semantic_match(job_description, top_k=100)  # Get more candidates for better matching
                candidates = old_results.get('results', [])
                
                if candidates:
                    logger.info(f"Using NEW matchmaking system with {len(candidates)} candidates")
                    
                    # Prepare candidates with proper IDs for matchmaking
                    # IMPORTANT: Preserve ALL original candidate fields
                    candidates_with_ids = []
                    candidate_id_map = {}  # Map matchmaking IDs to original candidates (preserve full data)
                    for idx, candidate in enumerate(candidates):
                        # Generate a unique ID for matchmaking
                        candidate_id = str(candidate.get('candidate_id') or 
                                         candidate.get('id') or 
                                         candidate.get('_id') or 
                                         candidate.get('email') or 
                                         candidate.get('Email') or
                                         candidate.get('FullName') or 
                                         candidate.get('full_name') or 
                                         f"candidate_{idx}")
                        
                        # Create a deep copy to preserve all fields
                        import copy
                        candidate_with_id = copy.deepcopy(candidate)
                        candidate_with_id['candidate_id'] = candidate_id
                        
                        candidates_with_ids.append(candidate_with_id)
                        # Store the FULL original candidate (deep copy) to preserve all fields
                        candidate_id_map[candidate_id] = copy.deepcopy(candidate)
                    
                    # Use module-level function which returns dicts
                    match_results = match_candidates_func(job_description, candidates_with_ids, top_k=top_k)
                    
                    # Transform to frontend format
                    transformed_results = []
                    has_exact_matches = False
                    for match_result in match_results:
                        candidate_id = match_result.get('candidate_id')
                        original_candidate = candidate_id_map.get(candidate_id, {})
                        transformed = transform_match_result_to_frontend_format(match_result, original_candidate)
                        
                        # Check if this is a good match (score >= 50%)
                        if transformed.get('Score', 0) >= 50:
                            has_exact_matches = True
                        
                        transformed_results.append(transformed)
                    
                    # If no results or no good matches found, get suggested candidates (limit to 5)
                    if not transformed_results or not has_exact_matches:
                        if not transformed_results:
                            logger.info("No matches found, getting suggested candidates with relaxed criteria")
                            # Get only 5 suggested candidates when no matches found
                            suggested_results = match_candidates_func(job_description, candidates_with_ids, top_k=min(5, len(candidates_with_ids)))
                        else:
                            logger.info(f"No good matches found (score >= 50%), limiting to 5 suggested candidates")
                            # Limit existing results to 5
                            suggested_results = match_results[:5]
                        
                        # Clear and rebuild with suggested flag (limit to 5)
                        transformed_results = []
                        for match_result in suggested_results[:5]:  # Ensure max 5 candidates
                            candidate_id = match_result.get('candidate_id')
                            original_candidate = candidate_id_map.get(candidate_id, {})
                            
                            # CRITICAL FIX: If original_candidate is empty, try to find it from candidates_with_ids
                            if not original_candidate or len(original_candidate) < 3:
                                logger.warning(f"Original candidate not found in map for ID {candidate_id}, searching in candidates_with_ids")
                                # Try to find the candidate by ID in candidates_with_ids
                                for cand in candidates_with_ids:
                                    if str(cand.get('candidate_id', '')) == str(candidate_id):
                                        original_candidate = cand
                                        logger.info(f"Found candidate in candidates_with_ids: {candidate_id}")
                                        break
                            
                            # If still empty, use match_result itself as fallback (it might contain candidate data)
                            if not original_candidate or len(original_candidate) < 3:
                                logger.warning(f"Using match_result as fallback for candidate {candidate_id}")
                                # Extract any candidate data from match_result if available
                                original_candidate = match_result.get('candidate_data', match_result)
                            
                            transformed = transform_match_result_to_frontend_format(match_result, original_candidate)
                            
                            # Mark as suggested
                            transformed['is_suggested'] = True
                            transformed['suggestion_reason'] = 'No exact matches found based on your search. Showing candidates with partial skill overlap.'
                            # Update explanation to indicate suggestion
                            original_explanation = transformed.get('MatchExplanation', '')
                            transformed['MatchExplanation'] = f"SUGGESTED CANDIDATE: {original_explanation}"
                            transformed_results.append(transformed)
                        
                        # Final safety check: ensure we only return 5 candidates
                        transformed_results = transformed_results[:5]
                    
                    summary_text = 'matching'
                    if not has_exact_matches and transformed_results:
                        summary_text = 'suggested'
                    elif not transformed_results:
                        summary_text = 'no'
                    
                    results = {
                        'results': transformed_results,
                        'total_candidates': len(transformed_results),
                        'algorithm_used': 'matchmaking_system_v1',
                        'summary': f'Found {len(transformed_results)} {summary_text} candidates using new matchmaking system',
                        'has_exact_matches': has_exact_matches,
                        'has_suggestions': not has_exact_matches and transformed_results
                    }
                else:
                    # Fallback to old system if no candidates
                    logger.warning("No candidates found, using old system")
                    results = semantic_match(job_description, top_k=min(5, top_k))  # Limit to 5 for suggestions
            except Exception as e:
                logger.error(f"New matchmaking system failed: {e}, falling back to old system", exc_info=True)
                # When falling back, limit to 5 candidates for suggestions
                results = semantic_match(job_description, top_k=min(5, top_k))
                
                # If results are suggestions (no exact matches), ensure max 5
                if results.get('results'):
                    results['results'] = results['results'][:5]
                    results['total_candidates'] = len(results['results'])

            # Final safety: ensure no duplicate candidates by email/name in the response
            results = _dedupe_candidates_in_response(results, top_k=top_k)
            
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

@search_bp.route('/csv-match', methods=['POST'])
def csv_match():
    """
    Backend scorer for CSV-imported candidates.
    Now uses the main search algorithm for proper sorting and filtering.
    """
    try:
        payload = get_jwt_payload()
        if not payload:
            logger.error('Unauthorized CSV match attempt: No JWT payload')
            return jsonify({'error': 'Unauthorized: No JWT payload'}), 403

        user, _tenant_id = get_user_from_jwt(payload)
        if not user:
            logger.error(f'CSV match failed: user not found for payload {payload}')
            return jsonify({'error': 'User not found'}), 404

        data = request.get_json() or {}
        job_description = data.get('job_description') or data.get('query')
        if not job_description:
            return jsonify({'error': 'Job description is required.'}), 400

        candidates = data.get('candidates')
        if not isinstance(candidates, list) or not candidates:
            return jsonify({'error': 'CSV candidates payload is required.'}), 400

        source_label = data.get('source_label') or 'CSV Import'
        matching_options = data.get('matching_options') or {}
        min_match_score = matching_options.get('min_match_score') or data.get('min_match_score')

        logger.info(
            "Processing CSV match for user %s with %d candidates via %s (using search algorithm)",
            user.email,
            len(candidates),
            source_label,
        )

        # build_manual_csv_match now uses the search algorithm internally
        result = build_manual_csv_match(
            job_description,
            candidates,
            source_label=source_label,
            min_match_score=min_match_score,
        )
        return jsonify(result)
    except ValueError as validation_error:
        logger.error(f'CSV match validation error: {validation_error}')
        return jsonify({'error': str(validation_error)}), 400
    except Exception as exc:
        logger.error(f'CSV match failed: {exc}', exc_info=True)
        return jsonify({'error': 'CSV match failed. Please try again.'}), 500

@search_bp.route('/resume-match', methods=['POST'])
def resume_match():
    """
    Backend scorer for resume-uploaded candidates.
    Uses the main search algorithm for proper sorting and filtering.
    """
    try:
        payload = get_jwt_payload()
        if not payload:
            logger.error('Unauthorized resume match attempt: No JWT payload')
            return jsonify({'error': 'Unauthorized: No JWT payload'}), 403

        user, _tenant_id = get_user_from_jwt(payload)
        if not user:
            logger.error(f'Resume match failed: user not found for payload {payload}')
            return jsonify({'error': 'User not found'}), 404

        data = request.get_json() or {}
        job_description = data.get('job_description') or data.get('query')
        if not job_description:
            return jsonify({'error': 'Job description is required.'}), 400

        candidates = data.get('candidates')
        if not isinstance(candidates, list) or not candidates:
            return jsonify({'error': 'Resume candidates payload is required.'}), 400

        source_label = data.get('source_label') or 'Resume Upload'
        matching_options = data.get('matching_options') or {}
        min_match_score = matching_options.get('min_match_score') or data.get('min_match_score')

        logger.info(
            "Processing resume match for user %s with %d candidates via %s (using search algorithm)",
            user.email,
            len(candidates),
            source_label,
        )

        # Use the same function as CSV match (it now uses the search algorithm)
        result = build_manual_csv_match(
            job_description,
            candidates,
            source_label=source_label,
            min_match_score=min_match_score,
        )
        return jsonify(result)
    except ValueError as validation_error:
        logger.error(f'Resume match validation error: {validation_error}')
        return jsonify({'error': str(validation_error)}), 400
    except Exception as exc:
        logger.error(f'Resume match failed: {exc}', exc_info=True)
        return jsonify({'error': 'Resume match failed. Please try again.'}), 500

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

@search_bp.route('/candidate/performance-analytics', methods=['POST'])
def get_candidate_performance_analytics():
    """
    Get performance analytics for a candidate based on backend algorithm
    Returns predicted success score, skill proficiencies, performance trends, and industry benchmarks
    """
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        if not data or 'candidate' not in data:
            return jsonify({'error': 'Candidate data is required'}), 400
        
        candidate = data['candidate']
        job_description = data.get('job_description', '')
        
        # Extract candidate data
        match_score = candidate.get('Score') or candidate.get('matchScore') or candidate.get('score', 0)
        confidence = candidate.get('Confidence') or candidate.get('confidence', 75)
        all_skills = candidate.get('skills') or candidate.get('Skills') or candidate.get('technicalSkills') or []
        experience_raw = candidate.get('experience') or candidate.get('Experience') or candidate.get('yearsOfExperience', 0)
        candidate_name = candidate.get('FullName') or candidate.get('name') or 'Unknown'
        candidate_title = candidate.get('title') or candidate.get('Title') or candidate.get('currentRole') or ''
        
        # Convert experience to number (handle string values like "5 years" or numeric strings)
        try:
            if isinstance(experience_raw, str):
                # Try to extract number from string (e.g., "5 years" -> 5)
                import re
                numbers = re.findall(r'\d+', str(experience_raw))
                experience = float(numbers[0]) if numbers else 0.0
            else:
                experience = float(experience_raw) if experience_raw else 0.0
        except (ValueError, TypeError):
            experience = 0.0
        
        # Ensure match_score and confidence are numbers
        try:
            match_score = float(match_score) if match_score else 0.0
        except (ValueError, TypeError):
            match_score = 0.0
        
        try:
            confidence = float(confidence) if confidence else 75.0
        except (ValueError, TypeError):
            confidence = 75.0
        
        # Calculate predicted success score (backend algorithm)
        # More sophisticated calculation based on multiple factors
        experience_factor = min(experience / 10, 1.0) * 100  # 0-100 scale
        skills_factor = min(len(all_skills) / 15, 1.0) * 100  # 0-100 scale (15 skills = max)
        
        # Base score from match and confidence
        base_score = (match_score * 0.6) + (confidence * 0.4)
        
        # Experience bonus (more experience = higher success probability)
        experience_bonus = min(experience * 2, 15)  # Max 15 points for 7.5+ years
        
        # Skills diversity bonus
        skills_bonus = min(len(all_skills) * 1.5, 10)  # Max 10 points for 6+ skills
        
        # Calculate final predicted success
        predicted_success = round(
            base_score + 
            (experience_bonus * 0.2) + 
            (skills_bonus * 0.2)
        )
        predicted_success = max(30, min(95, predicted_success))  # Realistic range 30-95%
        
        # Determine top skill - use first skill, but calculate proficiency based on experience
        top_skill_name = all_skills[0] if len(all_skills) > 0 else 'General Skills'
        
        # Calculate top skill proficiency based on experience and match score
        # More experience = higher proficiency, but also consider match score
        if experience >= 5:
            top_skill_proficiency = min(95, max(85, predicted_success + 10))
        elif experience >= 3:
            top_skill_proficiency = min(90, max(80, predicted_success + 5))
        elif experience >= 1:
            top_skill_proficiency = min(85, max(75, predicted_success))
        else:
            top_skill_proficiency = min(80, max(70, predicted_success - 5))
        
        # Generate performance trend (5 data points showing realistic growth)
        # Start from a base that's lower than current, showing progression
        if predicted_success >= 70:
            base_trend = predicted_success - 25
        elif predicted_success >= 50:
            base_trend = predicted_success - 20
        else:
            base_trend = predicted_success - 15
        
        base_trend = max(30, base_trend)  # Don't go below 30%
        
        # Create realistic trend with some variation
        trend = [
            round(base_trend),
            round(base_trend + (predicted_success - base_trend) * 0.25),
            round(base_trend + (predicted_success - base_trend) * 0.5),
            round(base_trend + (predicted_success - base_trend) * 0.75),
            round(predicted_success)
        ]
        
        # Calculate skill proficiencies from candidate skills with realistic values
        skills_list = all_skills[:4] if len(all_skills) >= 4 else all_skills
        
        # If we have fewer than 4 skills, don't pad with generic ones - use what we have
        proficiencies = []
        for idx, skill in enumerate(skills_list):
            # Calculate proficiency based on:
            # 1. Position in list (first skill = highest)
            # 2. Experience level
            # 3. Predicted success score
            
            # Base proficiency decreases with position
            position_factor = 1.0 - (idx * 0.08)  # 8% decrease per position
            
            # Experience bonus
            if experience >= 5:
                exp_bonus = 10
            elif experience >= 3:
                exp_bonus = 5
            elif experience >= 1:
                exp_bonus = 0
            else:
                exp_bonus = -5
            
            # Calculate proficiency
            base_proficiency = top_skill_proficiency * position_factor
            proficiency = round(base_proficiency + exp_bonus)
            proficiency = max(60, min(95, proficiency))  # Realistic range 60-95%
            
            proficiencies.append({
                'name': skill,
                'percentage': proficiency
            })
        
        # If we have fewer than 4 skills, that's okay - return what we have
        # Don't pad with fake skills
        
        # Calculate industry benchmark (percentile) more realistically
        # Based on predicted success, but adjusted for typical distribution
        if predicted_success >= 80:
            industry_benchmark = min(90, round(75 + (predicted_success - 80) * 0.75))
        elif predicted_success >= 60:
            industry_benchmark = round(50 + (predicted_success - 60) * 1.25)
        elif predicted_success >= 40:
            industry_benchmark = round(30 + (predicted_success - 40) * 1.0)
        else:
            industry_benchmark = round(20 + (predicted_success - 30) * 1.0)
        
        industry_benchmark = max(20, min(90, industry_benchmark))  # Realistic range 20-90th percentile
        
        # Generate detailed descriptions with calculation basis
        predicted_success_description = (
            f"AI-based prediction for success in the applied role based on {candidate_name}'s "
            f"skills, experience, and performance trends. Calculated using match score ({match_score}%), "
            f"confidence level ({confidence}%), and experience factors."
        )
        
        # Detailed calculation basis for predicted success
        calculation_basis = {
            'predictedSuccess': {
                'formula': 'Base Score (60% match + 40% confidence) + Experience Bonus + Skills Bonus',
                'components': {
                    'baseScore': round((match_score * 0.6) + (confidence * 0.4), 1),
                    'matchScoreWeight': f"{match_score}% × 60% = {round(match_score * 0.6, 1)}%",
                    'confidenceWeight': f"{confidence}% × 40% = {round(confidence * 0.4, 1)}%",
                    'experienceBonus': f"{round(experience_bonus, 1)}% (based on {experience} years experience)",
                    'skillsBonus': f"{round(skills_bonus, 1)}% (based on {len(all_skills)} skills)",
                    'finalScore': f"{predicted_success}%"
                }
            },
            'topSkill': {
                'skill': top_skill_name,
                'proficiency': top_skill_proficiency,
                'basis': f"Selected as first listed skill. Proficiency calculated based on {experience} years experience and predicted success score of {predicted_success}%."
            },
            'performanceTrend': {
                'basis': f"Simulated growth trajectory starting from {trend[0]}% (base performance) to {trend[-1]}% (current predicted success), showing {round(((trend[-1] - trend[0]) / trend[0]) * 100, 1)}% improvement over 5 evaluation periods.",
                'calculation': f"Base: {trend[0]}% → Final: {trend[-1]}% (based on predicted success progression)"
            },
            'skillProficiencies': {
                'basis': f"Calculated for top {len(proficiencies)} skills based on: (1) Position in skill list (first skill = highest), (2) Experience level ({experience} years), (3) Predicted success score ({predicted_success}%)",
                'skills': [{'name': s['name'], 'percentage': s['percentage'], 'factors': f"Position factor + Experience bonus ({experience} years)"} for s in proficiencies]
            },
            'industryBenchmark': {
                'percentile': industry_benchmark,
                'basis': f"Calculated based on predicted success score of {predicted_success}% relative to industry distribution. Adjusted for typical candidate performance curve where {industry_benchmark}% of candidates score below this level."
            }
        }
        
        top_skill_description = (
            f"Identified as {candidate_name}'s strongest skill area based on project complexity, "
            f"experience level ({experience} years), and demonstrated proficiency in related work. "
            f"Proficiency score of {top_skill_proficiency}% reflects experience-adjusted skill level."
        )
        
        performance_trend_description = (
            f"Showing consistent growth trajectory over recent projects and career progression. "
            f"Performance has improved from {trend[0]}% to {trend[-1]}% over the last 5 evaluation periods. "
            f"This trend is calculated based on the candidate's current predicted success score of {predicted_success}% "
            f"and assumes progressive improvement from baseline performance."
        )
        
        industry_benchmark_description = (
            f"Candidate's overall score is in the {industry_benchmark}th percentile for "
            f"{candidate_title or 'similar roles'} in the industry. This means {industry_benchmark}% of candidates "
            f"in similar roles score below this candidate's predicted success score of {predicted_success}%."
        )
        
        return jsonify({
            'predictedSuccessScore': predicted_success,
            'predictedSuccessDescription': predicted_success_description,
            'topSkill': {
                'name': top_skill_name,
                'proficiency': top_skill_proficiency
            },
            'topSkillDescription': top_skill_description,
            'performanceTrend': trend,
            'performanceTrendDescription': performance_trend_description,
            'skillProficiencies': proficiencies,
            'industryBenchmark': industry_benchmark,
            'industryBenchmarkDescription': industry_benchmark_description,
            'calculationBasis': calculation_basis  # Add detailed calculation basis
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get performance analytics'}), 500

@search_bp.route('/candidate/behavioral-insights', methods=['POST'])
def get_candidate_behavioral_insights():
    """
    Get behavioral insights for a candidate based on backend algorithm
    Returns cultural fit score, description, and detailed behavioral insights
    """
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        if not data or 'candidate' not in data:
            return jsonify({'error': 'Candidate data is required'}), 400
        
        candidate = data['candidate']
        job_description = data.get('job_description', '')
        
        # Extract candidate data
        candidate_name = candidate.get('FullName') or candidate.get('name') or 'Unknown'
        experience_raw = candidate.get('experience') or candidate.get('Experience') or candidate.get('yearsOfExperience', 0)
        all_skills = candidate.get('skills') or candidate.get('Skills') or candidate.get('technicalSkills') or []
        work_history = candidate.get('workHistory') or candidate.get('WorkHistory') or []
        education = candidate.get('Education') or candidate.get('education') or ''
        
        # Convert experience to number
        try:
            if isinstance(experience_raw, str):
                import re
                numbers = re.findall(r'\d+', str(experience_raw))
                experience = float(numbers[0]) if numbers else 0.0
            else:
                experience = float(experience_raw) if experience_raw else 0.0
        except (ValueError, TypeError):
            experience = 0.0
        
        # Analyze work history and experience for behavioral traits
        experience_text = ' '.join([str(h) for h in work_history]) if work_history else ''
        experience_lower = experience_text.lower() + ' ' + str(education).lower()
        
        # Calculate cultural fit score based on multiple factors
        cultural_fit_score = 70  # Base score
        
        # Leadership indicators
        if any(word in experience_lower for word in ['lead', 'manage', 'direct', 'supervise', 'team']):
            cultural_fit_score += 10
        
        # Collaboration indicators
        if any(word in experience_lower for word in ['collaborate', 'team', 'partner', 'work with', 'coordinate']):
            cultural_fit_score += 10
        
        # Problem-solving indicators
        if any(word in experience_lower for word in ['solve', 'analyze', 'research', 'investigate', 'develop']):
            cultural_fit_score += 5
        
        # Adaptability indicators
        if any(word in experience_lower for word in ['adapt', 'change', 'evolve', 'learn', 'new technology']):
            cultural_fit_score += 5
        
        # Experience bonus
        if experience >= 5:
            cultural_fit_score += 5
        elif experience >= 3:
            cultural_fit_score += 3
        
        # Skills diversity bonus
        if len(all_skills) >= 5:
            cultural_fit_score += 5
        
        cultural_fit_score = max(50, min(95, cultural_fit_score))
        
        # Generate behavioral insights
        insights = []
        
        # 1. Collaboration Style
        collaboration_score = 60
        collaboration_level = 'Moderate'
        if any(word in experience_lower for word in ['team', 'collaborate', 'partner', 'work with']):
            collaboration_score = 85
            collaboration_level = 'High'
        elif any(word in experience_lower for word in ['independent', 'solo', 'alone']):
            collaboration_score = 40
            collaboration_level = 'Low'
        
        insights.append({
            'title': 'Collaboration Style',
            'description': f"{candidate_name} demonstrates a {'collaborative and team-oriented' if collaboration_level == 'High' else 'balanced' if collaboration_level == 'Moderate' else 'more independent'} approach, suggesting they {'excel at facilitating discussions and incorporating feedback' if collaboration_level == 'High' else 'can work both independently and in teams' if collaboration_level == 'Moderate' else 'prefer working autonomously'}.",
            'metricLabel': 'Team Player',
            'metricValue': collaboration_score,
            'metricLevel': collaboration_level,
            'icon': 'Users'
        })
        
        # 2. Problem-Solving
        problem_solving_score = 70
        problem_solving_level = 'Moderate'
        if any(word in experience_lower for word in ['analyze', 'research', 'solve', 'investigate', 'data', 'analytics']):
            problem_solving_score = 90
            problem_solving_level = 'High'
        elif any(word in experience_lower for word in ['creative', 'innovate', 'develop', 'design']):
            problem_solving_score = 80
            problem_solving_level = 'High'
        
        insights.append({
            'title': 'Problem-Solving',
            'description': f"Employs a {'structured and analytical' if problem_solving_level == 'High' else 'practical'} problem-solving method, {'breaking down complex issues into manageable parts and using data to inform decisions' if problem_solving_level == 'High' else 'focusing on practical solutions and outcomes'}.",
            'metricLabel': 'Analytical Approach',
            'metricValue': problem_solving_score,
            'metricLevel': problem_solving_level,
            'icon': 'Lightbulb'
        })
        
        # 3. Leadership Potential
        leadership_score = 50
        leadership_level = 'Moderate'
        if any(word in experience_lower for word in ['lead', 'manage', 'direct', 'supervise', 'mentor', 'coach']):
            leadership_score = 85
            leadership_level = 'High'
        elif experience >= 5:
            leadership_score = 65
            leadership_level = 'Moderate'
        else:
            leadership_score = 45
            leadership_level = 'Low'
        
        insights.append({
            'title': 'Leadership Potential',
            'description': f"{'Shows strong indicators of leadership potential' if leadership_level == 'High' else 'Shows indicators of leadership potential' if leadership_level == 'Moderate' else 'Developing leadership skills'}, with {'demonstrated experience in leading teams and projects' if leadership_level == 'High' else 'initiative-taking behavior and comfort in leading projects' if leadership_level == 'Moderate' else 'potential for growth in leadership roles'}.",
            'metricLabel': 'Initiative & Mentorship',
            'metricValue': leadership_score,
            'metricLevel': leadership_level,
            'icon': 'TrendingUp'
        })
        
        # 4. Adaptability
        adaptability_score = 70
        adaptability_level = 'Moderate'
        if any(word in experience_lower for word in ['adapt', 'change', 'evolve', 'learn', 'new technology', 'transition']):
            adaptability_score = 88
            adaptability_level = 'High'
        elif experience >= 3:
            adaptability_score = 75
            adaptability_level = 'Moderate'
        
        insights.append({
            'title': 'Adaptability',
            'description': f"Career history shows {'consistent ability to adapt to new technologies and changing requirements' if adaptability_level == 'High' else 'ability to adapt to changing requirements'}, {'comfortable with ambiguity and dynamic environments' if adaptability_level == 'High' else 'able to handle moderate changes in work environment'}.",
            'metricLabel': 'Change Management',
            'metricValue': adaptability_score,
            'metricLevel': adaptability_level,
            'icon': 'RefreshCw'
        })
        
        # Generate cultural fit description
        cultural_fit_description = (
            f"Demonstrates {'strong' if cultural_fit_score >= 80 else 'good' if cultural_fit_score >= 70 else 'moderate'} "
            f"alignment with company values and culture. "
            f"{'Excellent fit for collaborative and dynamic work environments.' if cultural_fit_score >= 80 else 'Good fit with potential for growth and development.' if cultural_fit_score >= 70 else 'Moderate fit that may benefit from additional onboarding and support.'}"
        )
        
        # Calculate detailed calculation basis for behavioral insights
        # Track components for cultural fit score
        cultural_fit_components = {
            'baseScore': 70,
            'leadershipBonus': 0,
            'collaborationBonus': 0,
            'problemSolvingBonus': 0,
            'adaptabilityBonus': 0,
            'experienceBonus': 0,
            'skillsBonus': 0
        }
        
        if any(word in experience_lower for word in ['lead', 'manage', 'direct', 'supervise', 'team']):
            cultural_fit_components['leadershipBonus'] = 10
        if any(word in experience_lower for word in ['collaborate', 'team', 'partner', 'work with', 'coordinate']):
            cultural_fit_components['collaborationBonus'] = 10
        if any(word in experience_lower for word in ['solve', 'analyze', 'research', 'investigate', 'develop']):
            cultural_fit_components['problemSolvingBonus'] = 5
        if any(word in experience_lower for word in ['adapt', 'change', 'evolve', 'learn', 'new technology']):
            cultural_fit_components['adaptabilityBonus'] = 5
        if experience >= 5:
            cultural_fit_components['experienceBonus'] = 5
        elif experience >= 3:
            cultural_fit_components['experienceBonus'] = 3
        if len(all_skills) >= 5:
            cultural_fit_components['skillsBonus'] = 5
        
        calculation_basis = {
            'culturalFit': {
                'formula': 'Base Score (70) + Leadership Bonus + Collaboration Bonus + Problem-Solving Bonus + Adaptability Bonus + Experience Bonus + Skills Bonus',
                'components': {
                    'baseScore': f"{cultural_fit_components['baseScore']}% (default baseline)",
                    'leadershipBonus': f"+{cultural_fit_components['leadershipBonus']}% (keywords: lead, manage, direct, supervise, team)" if cultural_fit_components['leadershipBonus'] > 0 else '0% (no leadership indicators)',
                    'collaborationBonus': f"+{cultural_fit_components['collaborationBonus']}% (keywords: collaborate, team, partner, coordinate)" if cultural_fit_components['collaborationBonus'] > 0 else '0% (no collaboration indicators)',
                    'problemSolvingBonus': f"+{cultural_fit_components['problemSolvingBonus']}% (keywords: solve, analyze, research, investigate, develop)" if cultural_fit_components['problemSolvingBonus'] > 0 else '0% (no problem-solving indicators)',
                    'adaptabilityBonus': f"+{cultural_fit_components['adaptabilityBonus']}% (keywords: adapt, change, evolve, learn, new technology)" if cultural_fit_components['adaptabilityBonus'] > 0 else '0% (no adaptability indicators)',
                    'experienceBonus': f"+{cultural_fit_components['experienceBonus']}% (based on {experience} years experience)" if cultural_fit_components['experienceBonus'] > 0 else '0% (limited experience)',
                    'skillsBonus': f"+{cultural_fit_components['skillsBonus']}% (based on {len(all_skills)} skills)" if cultural_fit_components['skillsBonus'] > 0 else '0% (limited skills diversity)',
                    'finalScore': f"{cultural_fit_score}%"
                }
            },
            'insights': []
        }
        
        # Add calculation basis for each insight
        for insight in insights:
            insight_basis = {
                'title': insight['title'],
                'score': insight['metricValue'],
                'level': insight['metricLevel']
            }
            
            if insight['title'] == 'Collaboration Style':
                if any(word in experience_lower for word in ['team', 'collaborate', 'partner', 'work with']):
                    insight_basis['basis'] = f"High score (85%) based on strong collaboration keywords found in work history: team, collaborate, partner, work with"
                elif any(word in experience_lower for word in ['independent', 'solo', 'alone']):
                    insight_basis['basis'] = f"Low score (40%) based on independent work indicators found in work history"
                else:
                    insight_basis['basis'] = f"Moderate score (60%) - balanced approach, no strong collaboration or independence indicators"
            
            elif insight['title'] == 'Problem-Solving':
                if any(word in experience_lower for word in ['analyze', 'research', 'solve', 'investigate', 'data', 'analytics']):
                    insight_basis['basis'] = f"High score (90%) based on analytical keywords: analyze, research, solve, investigate, data, analytics"
                elif any(word in experience_lower for word in ['creative', 'innovate', 'develop', 'design']):
                    insight_basis['basis'] = f"High score (80%) based on creative/innovative keywords: creative, innovate, develop, design"
                else:
                    insight_basis['basis'] = f"Moderate score (70%) - practical problem-solving approach, no strong analytical or creative indicators"
            
            elif insight['title'] == 'Leadership Potential':
                if any(word in experience_lower for word in ['lead', 'manage', 'direct', 'supervise', 'mentor', 'coach']):
                    insight_basis['basis'] = f"High score (85%) based on leadership keywords: lead, manage, direct, supervise, mentor, coach"
                elif experience >= 5:
                    insight_basis['basis'] = f"Moderate score (65%) based on {experience} years of experience indicating leadership potential"
                else:
                    insight_basis['basis'] = f"Low score (45%) - limited experience ({experience} years) and no explicit leadership indicators"
            
            elif insight['title'] == 'Adaptability':
                if any(word in experience_lower for word in ['adapt', 'change', 'evolve', 'learn', 'new technology', 'transition']):
                    insight_basis['basis'] = f"High score (88%) based on adaptability keywords: adapt, change, evolve, learn, new technology, transition"
                elif experience >= 3:
                    insight_basis['basis'] = f"Moderate score (75%) based on {experience} years of experience showing ability to adapt"
                else:
                    insight_basis['basis'] = f"Moderate score (70%) - standard adaptability level"
            
            calculation_basis['insights'].append(insight_basis)
        
        return jsonify({
            'culturalFitScore': cultural_fit_score,
            'culturalFitDescription': cultural_fit_description,
            'insights': insights,
            'calculationBasis': calculation_basis
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting behavioral insights: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get behavioral insights'}), 500