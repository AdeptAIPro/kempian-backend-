"""
Candidate Search History API Routes
Handles saving, retrieving, and managing candidate search history
"""
from flask import Blueprint, request, jsonify
from app.db import db
from app.models import CandidateSearchHistory, CandidateSearchResult, Tenant
from app.utils import get_current_user
from app.simple_logger import get_logger
from datetime import datetime, timedelta
import json

logger = get_logger('candidate_search_history')

candidate_search_bp = Blueprint('candidate_search_history', __name__)

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
        
        # Create search history record
        search_history = CandidateSearchHistory(
            tenant_id=int(tenant_id),
            user_id=user.get('sub'),
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
        for candidate in candidates:
            candidate_result = CandidateSearchResult(
                search_history_id=search_history.id,
                candidate_id=candidate.get('id', ''),
                candidate_name=candidate.get('name', ''),
                candidate_email=candidate.get('email', ''),
                candidate_phone=candidate.get('phone', ''),
                candidate_location=candidate.get('location', ''),
                match_score=candidate.get('match_score', 0.0),
                candidate_data=json.dumps(candidate.get('data', {}))
            )
            db.session.add(candidate_result)
        
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
            # Delete candidate results first
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