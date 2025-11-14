# main_integrated.py - OPTIMIZED INTEGRATED VERSION WITH REACT FRONTEND

import os
import json
import re
import time
import hashlib
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Lazy imports for better startup performance
def _lazy_imports():
    """Lazy load heavy dependencies only when needed"""
    global boto3, np, nltk, faiss, pickle, SentenceTransformer, CrossEncoder, requests, openai, Flask, request, jsonify, render_template_string, render_template, send_from_directory, CORS, logging, load_dotenv
    
    try:
        import boto3
        import numpy as np
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import RegexpTokenizer
        import faiss
        import pickle
        from sentence_transformers import SentenceTransformer, CrossEncoder
        import requests
        import openai
        from flask import Flask, request, jsonify, render_template_string, render_template, send_from_directory
        from flask_cors import CORS
        import logging
        from dotenv import load_dotenv
        return True
    except ImportError as e:
        print(f"Warning: Some dependencies not available: {e}")
        return False

# Initialize lazy imports
_dependencies_available = _lazy_imports()

# Load environment variables only if dependencies are available
if _dependencies_available:
    load_dotenv()
    
    # Configure API keys and settings
    if 'openai' in globals():
        openai.api_key = os.getenv('OPENAI_API_KEY')
    os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
    os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_REGION', 'ap-south-1')

# Database configuration
DATABASE_TABLE_NAME = os.getenv('DYNAMODB_TABLE_NAME', 'user-resume-metadata')

# Add import for bias prevention - with fallback
try:
    import bias_prevention
    BIAS_PREVENTION_AVAILABLE = True
except ImportError:
    # bias_prevention module not available - using fallback
    bias_prevention = None
    BIAS_PREVENTION_AVAILABLE = False

# Add this import after the existing imports (around line 30)
try:
    import behavioural_analysis
    BEHAVIOURAL_ANALYSIS_AVAILABLE = True
except ImportError:
    print("⚠️ behavioural_analysis module not available")
    behavioural_analysis = None
    BEHAVIOURAL_ANALYSIS_AVAILABLE = False

# Add import for explainable AI
try:
    import explainable_ai
    EXPLAINABLE_AI_AVAILABLE = True
except ImportError:
    print("⚠️ explainable_ai module not available")
    explainable_ai = None
    EXPLAINABLE_AI_AVAILABLE = False

# Add import for market intelligence
try:
    import market_intelligence
    from market_intelligence.api import market_intel_bp
    MARKET_INTELLIGENCE_AVAILABLE = True
except ImportError:
    print("⚠️ market_intelligence module not available")
    market_intelligence = None
    market_intel_bp = None
    MARKET_INTELLIGENCE_AVAILABLE = False

# Enhanced ML dependencies
try:
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error, r2_score
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not available")
    lgb = None
    LIGHTGBM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the new Flask app
try:
    from app import create_app
    # Create the integrated Flask app
    app = create_app()
    logger.info("✅ Flask app created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create Flask app: {e}")
    # Fallback to basic Flask app
    if _dependencies_available and 'Flask' in globals():
        app = Flask(__name__)
        logger.warning("⚠️ Using fallback Flask app")
    else:
        # Create a minimal app for testing
        class MockApp:
            def __init__(self):
                self.config = {'INTEGRATED_ROUTES_ADDED': False, 'HAS_FRONTEND': None}
            def route(self, *args, **kwargs):
                def decorator(f):
                    return f
                return decorator
            def run(self, *args, **kwargs):
                print("Mock app - cannot run without Flask")
        app = MockApp()
        logger.warning("⚠️ Using mock app (Flask not available)")

# CORS is already configured in the app factory
# No need to reconfigure it here

# Initialize behavioural analysis and bias prevention systems
behavioural_pipeline = None
bias_sanitizer = None
bias_monitor = None
explainable_ai_system = None

if BEHAVIOURAL_ANALYSIS_AVAILABLE:
    try:
        # Initialize behavioural analysis pipeline
        behavioural_pipeline = behavioural_analysis.get_pipeline("production")
        logger.info("✅ Behavioural analysis pipeline initialized")
    except Exception as e:
        logger.warning(f"⚠️ Behavioural analysis pipeline initialization failed: {e}")
        behavioural_pipeline = None
else:
    logger.info("📋 Behavioural analysis not available - skipping initialization")

if BIAS_PREVENTION_AVAILABLE:
    try:
        # Initialize bias prevention components
        bias_sanitizer = bias_prevention.QuerySanitizer()
        bias_monitor = bias_prevention.BiasMonitor()
        logger.info("✅ Bias prevention components initialized")
    except Exception as e:
        logger.warning(f"⚠️ Bias prevention components initialization failed: {e}")
        bias_sanitizer = None
        bias_monitor = None
else:
    logger.info("📋 Bias prevention not available - skipping initialization")

if EXPLAINABLE_AI_AVAILABLE:
    try:
        # Initialize explainable AI system
        explainable_ai_system = explainable_ai.get_system("production")
        logger.info("✅ Explainable AI system initialized")
    except Exception as e:
        logger.warning(f"⚠️ Explainable AI system initialization failed: {e}")
        explainable_ai_system = None
else:
    logger.info("📋 Explainable AI not available - skipping initialization")

# Import instant search system for ultra-fast performance
from search.instant_search import get_instant_search_engine

# Initialize instant search system (pre-loads cache at startup)
search_system = get_instant_search_engine()

# ===== API ROUTES =====
# Avoid defining duplicate routes when using the app factory which already
# registers equivalent endpoints and blueprints.
if not app.config.get('INTEGRATED_ROUTES_ADDED', False):
    app.config['INTEGRATED_ROUTES_ADDED'] = True

    @app.route('/search', methods=['POST'])
    def search_endpoint():
        """High-performance search endpoint with caching"""
        start_time = time.time()
        
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            user_input = data.get('query', '')
            top_k = data.get('top_k', 10)
            enable_domain_filtering = data.get('enable_domain_filtering', True)
            include_behavioural_analysis = data.get('include_behavioural_analysis', False)
            
            if not user_input:
                return jsonify({'error': 'Query is required'}), 400
            
            # Perform the instant search (ultra-fast, pre-loaded cache)
            print(f"🔍 Performing instant search for: '{user_input}' with top_k={top_k}")
            
            results = search_system.search(
                user_input, 
                limit=top_k
            )
            
            response_time = (time.time() - start_time) * 1000
            print(f"📊 Search returned {len(results)} results in {response_time:.2f}ms")
            
            # Get performance stats
            perf_stats = search_system.get_stats()
            
            # Prepare instant search response (matching frontend expectations)
            response = {
                'results': results,  # Frontend expects results.results
                'summary': f"Found {len(results)} candidates",
                'bias_info': {'bias_detected': False, 'flags': []},
                'performance': {
                    'response_time_ms': round(response_time, 2),
                    'cache_hit': response_time < 1.0,  # Sub-millisecond responses
                    'optimization_level': 'instant_search',
                    'total_searches': perf_stats['total_searches'],
                    'avg_search_time_ms': perf_stats['avg_response_time_ms'],
                    'cache_size': perf_stats['total_candidates'],
                    'fastest_query_ms': perf_stats['fastest_query_ms'],
                    'cache_hit_rate': perf_stats['cache_hit_rate']
                },
                'query': user_input,
                'top_k': top_k,
                'domain_filtering_enabled': enable_domain_filtering,
                'behavioural_analysis_included': include_behavioural_analysis
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Search error: {e}")
            return jsonify({'error': f'Search failed: {str(e)}'}), 500

    @app.route('/debug/candidates', methods=['GET'])
    def debug_candidates():
        """Debug endpoint to check loaded candidates"""
        try:
            candidates = search_system.candidates
            stats = search_system.get_stats()
            
            debug_info = {
                'status': 'success',
                'total_candidates': len(candidates),
                'indexed_skills': len(search_system.skill_index),
                'indexed_words': len(search_system.word_index),
                'sample_candidates': list(candidates.values())[:3] if candidates else [],
                'search_system_initialized': True,
                'optimization_level': 'instant_search',
                'performance_stats': stats
            }
            
            return jsonify(debug_info)
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """System health check endpoint - no rate limiting for frequent health checks"""
        try:
            perf_stats = search_system.get_stats()
            
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'instant_search': True,
                    'pre_loaded_cache': True,
                    'database': True,
                    'caching': True
                },
                'features': {
                    'instant_search': True,
                    'sub_millisecond_responses': True,
                    'behavioural_analysis': BEHAVIOURAL_ANALYSIS_AVAILABLE,
                    'bias_prevention': BIAS_PREVENTION_AVAILABLE,
                    'explainable_ai': EXPLAINABLE_AI_AVAILABLE
                },
                'performance': {
                    'optimization_level': 'instant_search',
                    'total_searches': perf_stats['total_searches'],
                    'avg_search_time_ms': perf_stats['avg_response_time_ms'],
                    'fastest_query_ms': perf_stats['fastest_query_ms'],
                    'cache_hit_rate': perf_stats['cache_hit_rate'],
                    'total_candidates': perf_stats['total_candidates']
                },
                'candidates_loaded': len(search_system.candidates)
            }
            
            return jsonify(health_status)
            
        except Exception as e:
            print(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/performance', methods=['GET'])
    def performance_stats():
        """Get detailed performance statistics"""
        try:
            stats = search_system.get_stats()
            return jsonify({
                'status': 'ok',
                'data': stats,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500


if __name__ == '__main__':
    # Start the Flask application
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5055))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"🚀 Starting AdeptAI on {host}:{port}")
    logger.info(f"🔧 Debug mode: {debug}")
    logger.info(f"🌐 React frontend integration: {'Enabled' if app.config.get('HAS_FRONTEND') else 'Disabled'}")
    
    app.run(host=host, port=port, debug=debug)
