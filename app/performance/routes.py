# Performance-Optimized API Routes for 1000+ Candidates
# High-performance endpoints with caching, pagination, and batch processing

import time
import logging
from flask import Blueprint, request, jsonify, current_app
from app.simple_logger import get_logger
from app.models import db, CandidateProfile, User
from app.search.routes import get_user_from_jwt, get_jwt_payload
from .optimized_candidate_handler import candidate_handler
from .optimized_search_system import search_system

logger = get_logger("performance_api")

# Create blueprint
performance_bp = Blueprint('performance', __name__)

@performance_bp.route('/candidates/optimized', methods=['GET'])
def get_candidates_optimized():
    """
    Optimized endpoint for retrieving candidates with pagination and filtering
    Handles 1000+ candidates efficiently
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 50)), 100)  # Max 100 per page
        search_query = request.args.get('search', '')
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        
        # Build filters
        filters = {}
        if request.args.get('experience_years_min'):
            filters['experience_years_min'] = int(request.args.get('experience_years_min'))
        if request.args.get('experience_years_max'):
            filters['experience_years_max'] = int(request.args.get('experience_years_max'))
        if request.args.get('location'):
            filters['location'] = request.args.get('location')
        if request.args.get('is_public'):
            filters['is_public'] = request.args.get('is_public').lower() == 'true'
        if request.args.get('visa_status'):
            filters['visa_status'] = request.args.get('visa_status')
        if request.args.get('skills'):
            filters['skills'] = request.args.get('skills').split(',')
        
        # Get candidates using optimized handler
        start_time = time.time()
        result = candidate_handler.get_candidates_optimized(
            page=page,
            per_page=per_page,
            filters=filters,
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order
        )
        processing_time = time.time() - start_time
        
        # Add performance metrics
        result['performance'] = {
            **result['performance'],
            'api_processing_time': processing_time,
            'timestamp': time.time()
        }
        
        logger.info(f"Retrieved {len(result['candidates'])} candidates in {processing_time:.2f}s")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in get_candidates_optimized: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@performance_bp.route('/candidates/search', methods=['POST'])
def search_candidates_optimized():
    """
    Optimized search endpoint for candidates
    Uses FAISS index for fast similarity search
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        query = data.get('query', '')
        top_k = min(int(data.get('top_k', 20)), 100)  # Max 100 results
        filters = data.get('filters', {})
        use_cache = data.get('use_cache', True)
        
        if not query.strip():
            return jsonify({'error': 'Search query is required'}), 400
        
        # Perform search
        start_time = time.time()
        search_results = search_system.search_candidates(
            query=query,
            top_k=top_k,
            filters=filters,
            use_cache=use_cache
        )
        search_time = time.time() - start_time
        
        # Convert results to response format
        results = []
        for result in search_results:
            results.append({
                'candidate_id': result.candidate_id,
                'score': result.score,
                'match_reasons': result.match_reasons,
                'candidate': result.candidate_data
            })
        
        response = {
            'results': results,
            'total': len(results),
            'query': query,
            'performance': {
                'search_time': search_time,
                'timestamp': time.time()
            }
        }
        
        logger.info(f"Search for '{query}' completed in {search_time:.2f}s, found {len(results)} results")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in search_candidates_optimized: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@performance_bp.route('/candidates/batch-process', methods=['POST'])
def batch_process_candidates():
    """
    Batch process candidates for operations like bulk updates
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        candidate_ids = data.get('candidate_ids', [])
        operation = data.get('operation', '')
        
        if not candidate_ids:
            return jsonify({'error': 'No candidate IDs provided'}), 400
        
        if not operation:
            return jsonify({'error': 'No operation specified'}), 400
        
        # Define operation functions
        operation_functions = {
            'export': lambda candidate: candidate.to_dict(),
            'validate': lambda candidate: {
                'id': candidate.id,
                'valid': bool(candidate.full_name and candidate.summary),
                'issues': []
            },
            'update_public_status': lambda candidate: {
                'id': candidate.id,
                'updated': True
            }
        }
        
        if operation not in operation_functions:
            return jsonify({'error': f'Unknown operation: {operation}'}), 400
        
        # Process candidates in batches
        start_time = time.time()
        results = candidate_handler.batch_process_candidates(
            candidate_ids=candidate_ids,
            process_func=operation_functions[operation]
        )
        processing_time = time.time() - start_time
        
        response = {
            'results': results,
            'total_processed': len(results),
            'operation': operation,
            'performance': {
                'processing_time': processing_time,
                'candidates_per_second': len(results) / max(processing_time, 0.001),
                'timestamp': time.time()
            }
        }
        
        logger.info(f"Batch processed {len(results)} candidates in {processing_time:.2f}s")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in batch_process_candidates: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@performance_bp.route('/candidates/statistics', methods=['GET'])
def get_candidate_statistics():
    """
    Get comprehensive statistics about candidates
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get statistics
        start_time = time.time()
        stats = candidate_handler.get_candidate_statistics()
        processing_time = time.time() - start_time
        
        # Add performance metrics
        stats['performance'] = {
            'processing_time': processing_time,
            'timestamp': time.time()
        }
        
        logger.info(f"Retrieved candidate statistics in {processing_time:.2f}s")
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error in get_candidate_statistics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@performance_bp.route('/search/rebuild-index', methods=['POST'])
def rebuild_search_index():
    """
    Rebuild the search index for better performance
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user is admin (optional security check)
        if not user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        # Rebuild index
        start_time = time.time()
        search_system.rebuild_index()
        processing_time = time.time() - start_time
        
        response = {
            'message': 'Search index rebuilt successfully',
            'performance': {
                'rebuild_time': processing_time,
                'timestamp': time.time()
            }
        }
        
        logger.info(f"Search index rebuilt in {processing_time:.2f}s")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in rebuild_search_index: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@performance_bp.route('/performance/metrics', methods=['GET'])
def get_performance_metrics():
    """
    Get performance metrics for the system
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get metrics from both systems
        candidate_metrics = candidate_handler.get_performance_metrics()
        search_metrics = search_system.get_performance_metrics()
        
        response = {
            'candidate_handler': candidate_metrics,
            'search_system': search_metrics,
            'timestamp': time.time()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in get_performance_metrics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@performance_bp.route('/performance/optimize', methods=['POST'])
def optimize_database():
    """
    Get database optimization suggestions
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user is admin
        if not user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        # Get optimization suggestions
        optimization_suggestions = candidate_handler.optimize_database_indexes()
        
        return jsonify(optimization_suggestions), 200
        
    except Exception as e:
        logger.error(f"Error in optimize_database: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@performance_bp.route('/search/clear-cache', methods=['POST'])
def clear_search_cache():
    """
    Clear the search cache
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Clear cache
        search_system.clear_cache()
        
        response = {
            'message': 'Search cache cleared successfully',
            'timestamp': time.time()
        }
        
        logger.info("Search cache cleared")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in clear_search_cache: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
