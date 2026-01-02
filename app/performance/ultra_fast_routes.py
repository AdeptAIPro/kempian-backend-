# Ultra-Fast API Routes for Maximum Performance
# Optimized endpoints for handling 1000+ candidates with maximum speed

import time
import logging
from flask import Blueprint, request, jsonify, current_app
from app.simple_logger import get_logger
from app.models import db, CandidateProfile, User
from app.search.routes import get_user_from_jwt, get_jwt_payload
from .ultra_fast_optimizer import ultra_fast_processor, ultra_fast_search_engine, UltraFastConfig

logger = get_logger("ultra_fast_api")

# Create blueprint
ultra_fast_bp = Blueprint('ultra_fast', __name__)

@ultra_fast_bp.route('/candidates/ultra-fast', methods=['GET'])
def get_candidates_ultra_fast():
    """
    Ultra-fast candidate retrieval endpoint
    Optimized for maximum speed with 1000+ candidates
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
        per_page = min(int(request.args.get('per_page', 100)), 200)  # Max 200 per page
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
        
        # Get candidates using ultra-fast processor
        result = ultra_fast_processor.get_candidates_ultra_fast(
            page=page,
            per_page=per_page,
            filters=filters,
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        logger.info(f"Ultra-fast retrieval: {len(result['candidates'])} candidates")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in ultra-fast retrieval: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@ultra_fast_bp.route('/candidates/search-ultra-fast', methods=['POST'])
def search_candidates_ultra_fast():
    """
    Ultra-fast search endpoint
    Uses optimized FAISS index for maximum speed
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
        use_cache = data.get('use_cache', True)
        
        if not query.strip():
            return jsonify({'error': 'Search query is required'}), 400
        
        # Perform ultra-fast search
        search_results = ultra_fast_search_engine.search_ultra_fast(
            query=query,
            top_k=top_k,
            use_cache=use_cache
        )
        
        response = {
            'results': search_results,
            'total': len(search_results),
            'query': query,
            'performance': {
                'search_time': ultra_fast_search_engine.performance_stats['avg_search_time'],
                'timestamp': time.time()
            }
        }
        
        logger.info(f"Ultra-fast search: '{query}' - {len(search_results)} results")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in ultra-fast search: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@ultra_fast_bp.route('/candidates/batch-ultra-fast', methods=['POST'])
def batch_process_ultra_fast():
    """
    Ultra-fast batch processing endpoint
    Parallel processing for maximum speed
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
        batch_size = data.get('batch_size', 200)
        
        if not candidate_ids:
            return jsonify({'error': 'No candidate IDs provided'}), 400
        
        if not operation:
            return jsonify({'error': 'No operation specified'}), 400
        
        # Define operation functions
        operation_functions = {
            'export': lambda candidate: candidate,
            'validate': lambda candidate: {
                'id': candidate['id'],
                'valid': bool(candidate.get('full_name') and candidate.get('summary')),
                'issues': []
            },
            'count': lambda candidate: 1,
            'summarize': lambda candidate: {
                'id': candidate['id'],
                'summary': candidate.get('summary', '')[:100] + '...' if candidate.get('summary') else ''
            }
        }
        
        if operation not in operation_functions:
            return jsonify({'error': f'Unknown operation: {operation}'}), 400
        
        # Process candidates with ultra-fast batch processing
        results = ultra_fast_processor.batch_process_ultra_fast(
            candidate_ids=candidate_ids,
            process_func=operation_functions[operation],
            batch_size=batch_size
        )
        
        response = {
            'results': results,
            'total_processed': len(results),
            'operation': operation,
            'performance': ultra_fast_processor.get_performance_stats()
        }
        
        logger.info(f"Ultra-fast batch processing: {len(results)} candidates")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in ultra-fast batch processing: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@ultra_fast_bp.route('/candidates/prefetch', methods=['POST'])
def prefetch_candidates():
    """
    Prefetch candidates for faster subsequent access
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
        
        if not candidate_ids:
            return jsonify({'error': 'No candidate IDs provided'}), 400
        
        # Prefetch candidates
        ultra_fast_processor.prefetch_candidates(candidate_ids)
        
        response = {
            'message': f'Prefetched {len(candidate_ids)} candidates',
            'prefetched_count': len(candidate_ids),
            'timestamp': time.time()
        }
        
        logger.info(f"Prefetched {len(candidate_ids)} candidates")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error prefetching candidates: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@ultra_fast_bp.route('/candidates/single-ultra-fast/<int:candidate_id>', methods=['GET'])
def get_candidate_ultra_fast(candidate_id):
    """
    Ultra-fast single candidate retrieval
    Uses caching for maximum speed
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get candidate using ultra-fast processor
        candidate = ultra_fast_processor.get_candidate_ultra_fast(candidate_id)
        
        if not candidate:
            return jsonify({'error': 'Candidate not found'}), 404
        
        response = {
            'candidate': candidate,
            'performance': {
                'cache_hit': candidate_id in ultra_fast_processor.prefetch_cache or 
                           candidate_id in ultra_fast_processor.cache,
                'timestamp': time.time()
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting candidate {candidate_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@ultra_fast_bp.route('/performance/ultra-fast-metrics', methods=['GET'])
def get_ultra_fast_metrics():
    """
    Get ultra-fast performance metrics
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
        processor_metrics = ultra_fast_processor.get_performance_stats()
        search_metrics = ultra_fast_search_engine.performance_stats
        
        response = {
            'processor': processor_metrics,
            'search_engine': search_metrics,
            'timestamp': time.time()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting ultra-fast metrics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@ultra_fast_bp.route('/performance/initialize-search', methods=['POST'])
def initialize_search_engine():
    """
    Initialize the ultra-fast search engine
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
        
        # Initialize search engine
        start_time = time.time()
        ultra_fast_search_engine.initialize_search_engine()
        initialization_time = time.time() - start_time
        
        response = {
            'message': 'Ultra-fast search engine initialized successfully',
            'initialization_time': initialization_time,
            'timestamp': time.time()
        }
        
        logger.info(f"Search engine initialized in {initialization_time:.2f}s")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error initializing search engine: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@ultra_fast_bp.route('/performance/clear-caches', methods=['POST'])
def clear_ultra_fast_caches():
    """
    Clear all ultra-fast caches
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Clear all caches
        ultra_fast_processor.clear_caches()
        ultra_fast_search_engine.search_cache.clear()
        
        response = {
            'message': 'All ultra-fast caches cleared successfully',
            'timestamp': time.time()
        }
        
        logger.info("All ultra-fast caches cleared")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error clearing caches: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@ultra_fast_bp.route('/performance/optimize-config', methods=['POST'])
def optimize_ultra_fast_config():
    """
    Optimize ultra-fast configuration based on system resources
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
        
        # Get request data
        data = request.get_json() or {}
        
        # Create optimized configuration
        config = UltraFastConfig(
            max_workers=data.get('max_workers', 8),
            batch_size=data.get('batch_size', 200),
            cache_size=data.get('cache_size', 2000),
            prefetch_size=data.get('prefetch_size', 500),
            connection_pool_size=data.get('connection_pool_size', 30),
            enable_async=data.get('enable_async', True),
            enable_compression=data.get('enable_compression', True),
            enable_memory_mapping=data.get('enable_memory_mapping', True)
        )
        
        # Update global instances
        global ultra_fast_processor, ultra_fast_search_engine
        ultra_fast_processor = ultra_fast_processor.__class__(config)
        ultra_fast_search_engine = ultra_fast_search_engine.__class__(config)
        
        response = {
            'message': 'Ultra-fast configuration optimized successfully',
            'config': {
                'max_workers': config.max_workers,
                'batch_size': config.batch_size,
                'cache_size': config.cache_size,
                'prefetch_size': config.prefetch_size,
                'connection_pool_size': config.connection_pool_size,
                'enable_async': config.enable_async,
                'enable_compression': config.enable_compression,
                'enable_memory_mapping': config.enable_memory_mapping
            },
            'timestamp': time.time()
        }
        
        logger.info("Ultra-fast configuration optimized")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error optimizing configuration: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
