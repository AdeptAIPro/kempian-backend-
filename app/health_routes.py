"""
Health Check Routes for Database and System Monitoring
"""
from flask import Blueprint, jsonify
from app.simple_logger import get_logger
from app.db_health import get_db_health, get_db_stats, handle_db_error
from app.cache import cache_manager
import time
import psutil
import os

logger = get_logger("health")
health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check for database and system"""
    try:
        start_time = time.time()
        
        # Database health
        db_healthy = get_db_health()
        db_stats = get_db_stats()
        
        # System health
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_percent = psutil.disk_usage('/').percent
        
        # Cache health
        cache_healthy = cache_manager.redis_available if hasattr(cache_manager, 'redis_available') else False
        
        response_time = (time.time() - start_time) * 1000  # ms
        
        health_status = {
            'status': 'healthy' if db_healthy and memory_percent < 90 else 'degraded',
            'timestamp': time.time(),
            'response_time_ms': round(response_time, 2),
            'database': {
                'healthy': db_healthy,
                'stats': db_stats
            },
            'system': {
                'memory_percent': memory_percent,
                'cpu_percent': cpu_percent,
                'disk_percent': disk_percent
            },
            'cache': {
                'redis_available': cache_healthy
            }
        }
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@health_bp.route('/health/database', methods=['GET'])
def database_health():
    """Database-specific health check"""
    try:
        db_healthy = get_db_health()
        db_stats = get_db_stats()
        
        return jsonify({
            'healthy': db_healthy,
            'stats': db_stats,
            'timestamp': time.time()
        }), 200 if db_healthy else 503
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return jsonify({
            'healthy': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@health_bp.route('/health/system', methods=['GET'])
def system_health():
    """System resource health check"""
    try:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return jsonify({
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': memory.percent
            },
            'cpu': {
                'usage_percent': cpu
            },
            'disk': {
                'total_gb': round(disk.total / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'used_percent': round((disk.used / disk.total) * 100, 2)
            },
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': time.time()
        }), 500
