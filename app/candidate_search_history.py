"""
Candidate Search History API Routes
Handles saving, retrieving, and managing candidate search history
"""
from flask import Blueprint, request, jsonify
from app.db import db
from app.models import CandidateSearchHistory, CandidateSearchResult, CandidateMatchLog, Tenant
from app.utils import get_current_user
from app.simple_logger import get_logger
from datetime import datetime, timedelta
import json

logger = get_logger('candidate_search_history')

candidate_search_bp = Blueprint('candidate_search_history', __name__)

def _normalize_match_score(value):
    if value is None:
        return 0.0
    if isinstance(value, str):
        value = value.replace('%', '').strip()
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score <= 1:
        score *= 100
    elif score > 1000:
        score = score / 100.0
    return round(score, 2)

def _extract_match_score(candidate):
    possible_keys = [
        'match_score', 'Score', 'match_percentage', 'MatchPercentage',
        'matchPercent', 'overall_score', 'overallScore',
        'final_score', 'finalScore', 'matchPercentile'
    ]
    sources = []
    if isinstance(candidate, dict):
        sources.append(candidate)
        data = candidate.get('data')
        if isinstance(data, dict):
            sources.append(data)
        match_details = candidate.get('match_details') or (data.get('match_details') if isinstance(data, dict) else None)
        if isinstance(match_details, dict):
            sources.append(match_details)

    for source in sources:
        for key in possible_keys:
            if key in source:
                score = _normalize_match_score(source.get(key))
                if score > 0:
                    return score

    return 0.0

@candidate_search_bp.route('/api/candidate-search-history', methods=['POST'])
def save_search_history():
    """Save a new candidate search to history"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get tenant ID from user
        tenant_id = user.get('custom:tenant_id')
        if not tenant_id:
            return jsonify({'error': 'Tenant ID not found'}), 400
        
        user_email = (
            user.get('email')
            or user.get('preferred_username')
            or user.get('username')
            or user.get('custom:email')
        )
        
        # Create search history record
        search_history = CandidateSearchHistory(
            tenant_id=int(tenant_id),
            user_id=user.get('sub'),
            user_email=user_email,
            job_description=data.get('job_description', ''),
            search_criteria=json.dumps(data.get('search_criteria', {})),
            candidates_found=data.get('candidates_found', 0),
            search_status=data.get('search_status', 'completed'),
            search_duration_ms=data.get('search_duration_ms'),
            expires_at=datetime.utcnow() + timedelta(days=10)
        )
        
        db.session.add(search_history)
        db.session.flush()  # Get the ID
        
        # Save candidate results if provided
        candidates = data.get('candidates', [])
        search_query = data.get('search_query', '')
        
        for candidate in candidates:
            # Extract match reasons from candidate data (check multiple possible locations)
            candidate_data = candidate.get('data', candidate) if isinstance(candidate.get('data'), dict) else candidate
            
            match_reasons = (
                candidate.get('match_reasons') or 
                candidate_data.get('match_reasons') or 
                candidate_data.get('Match Reasons') or
                []
            )
            match_explanation = (
                candidate.get('match_explanation') or 
                candidate_data.get('match_explanation') or 
                candidate_data.get('matchExplanation') or
                candidate_data.get('Match Explanation') or
                ''
            )
            match_details = (
                candidate.get('match_details') or 
                candidate_data.get('match_details') or 
                candidate_data.get('matchDetails') or
                candidate_data.get('Match Details') or
                {}
            )
            
            # If match_reasons is a list, convert to JSON string
            if isinstance(match_reasons, list):
                match_reasons_json = json.dumps(match_reasons)
            elif isinstance(match_reasons, str):
                # Try to parse if it's a JSON string
                try:
                    parsed = json.loads(match_reasons)
                    match_reasons_json = json.dumps(parsed) if isinstance(parsed, list) else match_reasons
                except:
                    match_reasons_json = json.dumps([match_reasons])
            else:
                match_reasons_json = json.dumps([])
            
            normalized_match_score = _extract_match_score(candidate)

            candidate_result = CandidateSearchResult(
                search_history_id=search_history.id,
                candidate_id=candidate.get('id', ''),
                candidate_name=candidate.get('name', ''),
                candidate_email=candidate.get('email', ''),
                candidate_phone=candidate.get('phone', ''),
                candidate_location=candidate.get('location', ''),
                match_score=normalized_match_score,
                match_reasons=match_reasons_json,
                candidate_data=json.dumps(candidate.get('data', candidate))
            )
            db.session.add(candidate_result)
            db.session.flush()  # Get the ID for the match log
            
            # Create long-term match log for admin analytics
            try:
                match_log_kwargs = {
                    'search_history_id': search_history.id,
                    'candidate_result_id': candidate_result.id,
                    'tenant_id': int(tenant_id),
                    'user_id': user.get('sub'),
                    'user_email': user_email,
                    'candidate_id': candidate.get('id', ''),
                    'candidate_name': candidate.get('name', ''),
                    'candidate_email': candidate.get('email', ''),
                    'job_description': data.get('job_description', ''),
                    'search_query': search_query or data.get('job_description', ''),
                    'search_criteria': json.dumps(data.get('search_criteria', {})),
                    'match_score': normalized_match_score,
                    'match_reasons': match_reasons_json,
                    'match_explanation': match_explanation or (', '.join(match_reasons) if isinstance(match_reasons, list) else str(match_reasons)),
                    'match_details': json.dumps(match_details) if match_details else None,
                    'algorithm_version': candidate.get('algorithm_version', candidate.get('matching_algorithm', 'unknown')),
                    'search_duration_ms': data.get('search_duration_ms')
                }

                # Gracefully handle environments where the column hasn't been migrated yet
                candidate_match_log_table = getattr(CandidateMatchLog, '__table__', None)
                if candidate_match_log_table is not None:
                    table_columns = candidate_match_log_table.columns.keys()
                    if 'user_email' not in table_columns:
                        match_log_kwargs.pop('user_email', None)
                else:
                    # As a fallback, avoid passing unknown kwargs
                    match_log_kwargs.pop('user_email', None)

                match_log = CandidateMatchLog(**match_log_kwargs)

                # If the ORM model exposes user_email, set it after instantiation as well
                if hasattr(match_log, 'user_email') and match_log_kwargs.get('user_email'):
                    match_log.user_email = match_log_kwargs['user_email']

                db.session.add(match_log)
            except Exception as e:
                logger.error(f"Error creating match log for candidate {candidate.get('id', 'unknown')}: {str(e)}")
                # Don't fail the whole operation if logging fails
        
        db.session.commit()
        
        logger.info(f"Saved candidate search history for user {user.get('sub')}, found {len(candidates)} candidates")
        
        return jsonify({
            'success': True,
            'search_id': search_history.id,
            'message': 'Search history saved successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving search history: {str(e)}")
        return jsonify({'error': 'Failed to save search history'}), 500

@candidate_search_bp.route('/api/candidate-search-history', methods=['GET'])
def get_search_history():
    """Get candidate search history for the current user"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        tenant_id = user.get('custom:tenant_id')
        if not tenant_id:
            return jsonify({'error': 'Tenant ID not found'}), 400
        
        # Get query parameters
        limit = request.args.get('limit', 20, type=int)
        include_expired = request.args.get('include_expired', 'false').lower() == 'true'
        
        # Build query
        query = CandidateSearchHistory.query.filter_by(
            tenant_id=int(tenant_id),
            user_id=user.get('sub')
        )
        
        # Filter expired searches if not including them
        if not include_expired:
            query = query.filter(CandidateSearchHistory.expires_at > datetime.utcnow())
        
        # Order by most recent first
        searches = query.order_by(CandidateSearchHistory.created_at.desc()).limit(limit).all()
        
        # Convert to dict and include candidate results
        search_history = []
        for search in searches:
            search_dict = search.to_dict()
            
            # Get candidate results for this search
            candidates = CandidateSearchResult.query.filter_by(
                search_history_id=search.id
            ).order_by(CandidateSearchResult.match_score.desc()).all()
            
            search_dict['candidates'] = [candidate.to_dict() for candidate in candidates]
            search_history.append(search_dict)
        
        return jsonify({
            'success': True,
            'searches': search_history,
            'total': len(search_history)
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving search history: {str(e)}")
        return jsonify({'error': 'Failed to retrieve search history'}), 500

@candidate_search_bp.route('/api/candidate-search-history/<int:search_id>', methods=['GET'])
def get_search_details(search_id):
    """Get detailed information about a specific search"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        tenant_id = user.get('custom:tenant_id')
        if not tenant_id:
            return jsonify({'error': 'Tenant ID not found'}), 400
        
        # Get search history
        search = CandidateSearchHistory.query.filter_by(
            id=search_id,
            tenant_id=int(tenant_id),
            user_id=user.get('sub')
        ).first()
        
        if not search:
            return jsonify({'error': 'Search not found'}), 404
        
        search_dict = search.to_dict()
        
        # Get all candidate results
        candidates = CandidateSearchResult.query.filter_by(
            search_history_id=search_id
        ).order_by(CandidateSearchResult.match_score.desc()).all()
        
        search_dict['candidates'] = [candidate.to_dict() for candidate in candidates]
        
        return jsonify({
            'success': True,
            'search': search_dict
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving search details: {str(e)}")
        return jsonify({'error': 'Failed to retrieve search details'}), 500

@candidate_search_bp.route('/api/candidate-search-history/<int:search_id>/extend', methods=['POST'])
def extend_search_expiry(search_id):
    """Extend the expiry date of a search"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        tenant_id = user.get('custom:tenant_id')
        if not tenant_id:
            return jsonify({'error': 'Tenant ID not found'}), 400
        
        data = request.get_json() or {}
        extend_days = data.get('days', 10)
        
        # Get search history
        search = CandidateSearchHistory.query.filter_by(
            id=search_id,
            tenant_id=int(tenant_id),
            user_id=user.get('sub')
        ).first()
        
        if not search:
            return jsonify({'error': 'Search not found'}), 404
        
        # Extend expiry
        search.extend_expiry(extend_days)
        db.session.commit()
        
        logger.info(f"Extended search {search_id} expiry by {extend_days} days for user {user.get('sub')}")
        
        return jsonify({
            'success': True,
            'message': f'Search expiry extended by {extend_days} days',
            'new_expiry': search.expires_at.isoformat()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error extending search expiry: {str(e)}")
        return jsonify({'error': 'Failed to extend search expiry'}), 500

@candidate_search_bp.route('/api/candidate-search-history/<int:search_id>/candidates/<int:candidate_id>/save', methods=['POST'])
def save_candidate(search_id, candidate_id):
    """Mark a candidate as saved"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        tenant_id = user.get('custom:tenant_id')
        if not tenant_id:
            return jsonify({'error': 'Tenant ID not found'}), 400
        
        # Verify search belongs to user
        search = CandidateSearchHistory.query.filter_by(
            id=search_id,
            tenant_id=int(tenant_id),
            user_id=user.get('sub')
        ).first()
        
        if not search:
            return jsonify({'error': 'Search not found'}), 404
        
        # Update candidate
        candidate = CandidateSearchResult.query.filter_by(
            id=candidate_id,
            search_history_id=search_id
        ).first()
        
        if not candidate:
            return jsonify({'error': 'Candidate not found'}), 404
        
        candidate.is_saved = True
        candidate.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Candidate saved successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving candidate: {str(e)}")
        return jsonify({'error': 'Failed to save candidate'}), 500

@candidate_search_bp.route('/api/candidate-search-history/<int:search_id>/candidates/<int:candidate_id>/contact', methods=['POST'])
def mark_candidate_contacted(search_id, candidate_id):
    """Mark a candidate as contacted"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        tenant_id = user.get('custom:tenant_id')
        if not tenant_id:
            return jsonify({'error': 'Tenant ID not found'}), 400
        
        # Verify search belongs to user
        search = CandidateSearchHistory.query.filter_by(
            id=search_id,
            tenant_id=int(tenant_id),
            user_id=user.get('sub')
        ).first()
        
        if not search:
            return jsonify({'error': 'Search not found'}), 404
        
        # Update candidate
        candidate = CandidateSearchResult.query.filter_by(
            id=candidate_id,
            search_history_id=search_id
        ).first()
        
        if not candidate:
            return jsonify({'error': 'Candidate not found'}), 404
        
        candidate.is_contacted = True
        candidate.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Candidate marked as contacted'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error marking candidate as contacted: {str(e)}")
        return jsonify({'error': 'Failed to mark candidate as contacted'}), 500

@candidate_search_bp.route('/api/candidate-search-history/stats', methods=['GET'])
def get_search_stats():
    """Get search statistics for the current user"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        tenant_id = user.get('custom:tenant_id')
        if not tenant_id:
            return jsonify({'error': 'Tenant ID not found'}), 400
        
        # Get stats
        total_searches = CandidateSearchHistory.query.filter_by(
            tenant_id=int(tenant_id),
            user_id=user.get('sub')
        ).count()
        
        active_searches = CandidateSearchHistory.query.filter(
            CandidateSearchHistory.tenant_id == int(tenant_id),
            CandidateSearchHistory.user_id == user.get('sub'),
            CandidateSearchHistory.expires_at > datetime.utcnow()
        ).count()
        
        total_candidates = CandidateSearchResult.query.join(
            CandidateSearchHistory
        ).filter(
            CandidateSearchHistory.tenant_id == int(tenant_id),
            CandidateSearchHistory.user_id == user.get('sub')
        ).count()
        
        saved_candidates = CandidateSearchResult.query.join(
            CandidateSearchHistory
        ).filter(
            CandidateSearchHistory.tenant_id == int(tenant_id),
            CandidateSearchHistory.user_id == user.get('sub'),
            CandidateSearchResult.is_saved == True
        ).count()
        
        contacted_candidates = CandidateSearchResult.query.join(
            CandidateSearchHistory
        ).filter(
            CandidateSearchHistory.tenant_id == int(tenant_id),
            CandidateSearchHistory.user_id == user.get('sub'),
            CandidateSearchResult.is_contacted == True
        ).count()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_searches': total_searches,
                'active_searches': active_searches,
                'total_candidates': total_candidates,
                'saved_candidates': saved_candidates,
                'contacted_candidates': contacted_candidates
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving search stats: {str(e)}")
        return jsonify({'error': 'Failed to retrieve search stats'}), 500

@candidate_search_bp.route('/api/candidate-search-history', methods=['DELETE'])
def clear_search_history():
    """Clear all candidate search history for the current user"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401

        tenant_id = user.get('custom:tenant_id')
        if not tenant_id:
            return jsonify({'error': 'Tenant ID not found'}), 400

        # Find all search history ids for this user
        history_ids = [
            h.id for h in CandidateSearchHistory.query.filter_by(
                tenant_id=int(tenant_id),
                user_id=user.get('sub')
            ).all()
        ]

        if history_ids:
            # First delete any match logs that reference candidate results for these searches
            candidate_results = CandidateSearchResult.query.filter(
                CandidateSearchResult.search_history_id.in_(history_ids)
            ).all()
            result_ids = [r.id for r in candidate_results]

            if result_ids:
                CandidateMatchLog.query.filter(
                    CandidateMatchLog.candidate_result_id.in_(result_ids)
                ).delete(synchronize_session=False)

            # Also delete any logs directly tied to these search IDs (defensive)
            CandidateMatchLog.query.filter(
                CandidateMatchLog.search_history_id.in_(history_ids)
            ).delete(synchronize_session=False)

            # Delete candidate results for these searches
            CandidateSearchResult.query.filter(
                CandidateSearchResult.search_history_id.in_(history_ids)
            ).delete(synchronize_session=False)

            # Delete the history entries
            CandidateSearchHistory.query.filter(
                CandidateSearchHistory.id.in_(history_ids)
            ).delete(synchronize_session=False)

            db.session.commit()

        logger.info(f"Cleared {len(history_ids)} search history entries for user {user.get('sub')}")

        return jsonify({
            'success': True,
            'message': 'Search history cleared successfully',
            'deleted_count': len(history_ids)
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error clearing search history: {str(e)}")
        return jsonify({'error': 'Failed to clear search history'}), 500


@candidate_search_bp.route('/api/candidate-search-history/<int:search_id>', methods=['DELETE'])
def delete_single_search_history(search_id: int):
    """Delete a single candidate search (and its candidates) for the current user"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401

        tenant_id = user.get('custom:tenant_id')
        if not tenant_id:
            return jsonify({'error': 'Tenant ID not found'}), 400

        # Ensure the search belongs to the current user and tenant
        search = CandidateSearchHistory.query.filter_by(
            id=search_id,
            tenant_id=int(tenant_id),
            user_id=user.get('sub')
        ).first()

        if not search:
            return jsonify({'error': 'Search not found'}), 404

        # First delete any match logs that reference candidate results for this search
        candidate_results = CandidateSearchResult.query.filter_by(
            search_history_id=search.id
        ).all()
        result_ids = [r.id for r in candidate_results]

        if result_ids:
            # Delete logs by candidate_result_id
            CandidateMatchLog.query.filter(
                CandidateMatchLog.candidate_result_id.in_(result_ids)
            ).delete(synchronize_session=False)

        # Also delete any logs directly tied to this search_id (defensive)
        CandidateMatchLog.query.filter(
            CandidateMatchLog.search_history_id == search.id
        ).delete(synchronize_session=False)

        # Now it is safe to delete candidate results
        CandidateSearchResult.query.filter_by(
            search_history_id=search.id
        ).delete(synchronize_session=False)

        # Finally delete the search history entry
        db.session.delete(search)
        db.session.commit()

        logger.info(f"Deleted search history {search_id} for user {user.get('sub')}")

        return jsonify({
            'success': True,
            'message': f'Search history {search_id} deleted successfully'
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting search history {search_id}: {str(e)}")
        return jsonify({'error': 'Failed to delete search history'}), 500


@candidate_search_bp.route('/api/candidate-search-history/bulk-delete', methods=['POST'])
def bulk_delete_search_history():
    """Bulk delete multiple candidate searches for the current user.

    Expects JSON body: { "search_ids": [1, 2, 3] }
    """
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401

        tenant_id = user.get('custom:tenant_id')
        if not tenant_id:
            return jsonify({'error': 'Tenant ID not found'}), 400

        data = request.get_json() or {}
        search_ids = data.get('search_ids') or data.get('ids') or []

        # Normalise IDs to integers and remove invalid entries
        try:
            numeric_ids = [
                int(sid) for sid in search_ids
                if isinstance(sid, (int, str)) and str(sid).strip().isdigit()
            ]
        except Exception:
            numeric_ids = []

        if not numeric_ids:
            return jsonify({'error': 'No valid search IDs provided'}), 400

        # Restrict deletion to searches belonging to this user and tenant
        searches = CandidateSearchHistory.query.filter(
            CandidateSearchHistory.id.in_(numeric_ids),
            CandidateSearchHistory.tenant_id == int(tenant_id),
            CandidateSearchHistory.user_id == user.get('sub')
        ).all()

        if not searches:
            return jsonify({'error': 'No matching searches found for this user'}), 404

        search_ids_to_delete = [s.id for s in searches]

        # Gather all candidate_result IDs for these searches
        candidate_results = CandidateSearchResult.query.filter(
            CandidateSearchResult.search_history_id.in_(search_ids_to_delete)
        ).all()
        result_ids = [r.id for r in candidate_results]

        if result_ids:
            # Delete match logs by candidate_result_id first to satisfy FK constraints
            CandidateMatchLog.query.filter(
                CandidateMatchLog.candidate_result_id.in_(result_ids)
            ).delete(synchronize_session=False)

        # Also delete any logs directly tied to these search IDs (defensive)
        CandidateMatchLog.query.filter(
            CandidateMatchLog.search_history_id.in_(search_ids_to_delete)
        ).delete(synchronize_session=False)

        # Delete candidate results for these searches
        CandidateSearchResult.query.filter(
            CandidateSearchResult.search_history_id.in_(search_ids_to_delete)
        ).delete(synchronize_session=False)

        # Delete the search history entries
        CandidateSearchHistory.query.filter(
            CandidateSearchHistory.id.in_(search_ids_to_delete)
        ).delete(synchronize_session=False)

        db.session.commit()

        logger.info(
            f"Bulk deleted {len(search_ids_to_delete)} search history entries "
            f"for user {user.get('sub')}: {search_ids_to_delete}"
        )

        return jsonify({
            'success': True,
            'message': 'Search history entries deleted successfully',
            'deleted_ids': search_ids_to_delete,
            'requested_ids': numeric_ids
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error bulk deleting search history: {str(e)}")
        return jsonify({'error': 'Failed to bulk delete search history'}), 500