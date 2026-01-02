"""
Health Check System for Load Balancer and Monitoring
Provides comprehensive health checks for all system components
"""
import time
import psutil
from datetime import datetime
from flask import Blueprint, jsonify, request
from app.simple_logger import get_logger
from app.monitoring import get_system_health, get_performance_metrics, get_recent_alerts
from app.cache import cache_manager
from app import db

logger = get_logger("health")

# Create health check blueprint
health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint for load balancer"""
    try:
        start_time = time.time()
        
        # Basic system checks
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Database connectivity check
        db_healthy = False
        db_response_time = 0
        try:
            db_start = time.time()
            db.session.execute('SELECT 1')
            db_response_time = time.time() - db_start
            db_healthy = True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
        
        # Cache connectivity check
        cache_healthy = False
        cache_response_time = 0
        try:
            cache_start = time.time()
            cache_manager.set('health_check', 'ok', 10)
            cache_manager.get('health_check')
            cache_response_time = time.time() - cache_start
            cache_healthy = True
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
        
        # Determine overall health
        overall_healthy = db_healthy and cache_healthy and cpu_usage < 90 and memory.percent < 90
        
        response_time = time.time() - start_time
        
        return jsonify({
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'response_time_ms': round(response_time * 1000, 2),
            'components': {
                'database': {
                    'status': 'healthy' if db_healthy else 'unhealthy',
                    'response_time_ms': round(db_response_time * 1000, 2)
                },
                'cache': {
                    'status': 'healthy' if cache_healthy else 'unhealthy',
                    'response_time_ms': round(cache_response_time * 1000, 2)
                },
                'system': {
                    'cpu_usage_percent': round(cpu_usage, 2),
                    'memory_usage_percent': round(memory.percent, 2),
                    'memory_available_mb': round(memory.available / 1024 / 1024, 2)
                }
            }
        }), 200 if overall_healthy else 503
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 503

@health_bp.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """Detailed health check with comprehensive metrics"""
    try:
        start_time = time.time()
        
        # Get system health from monitoring
        system_health = get_system_health()
        
        # Database detailed check
        db_status = {
            'connected': False,
            'response_time_ms': 0,
            'connection_pool_size': 0,
            'active_connections': 0
        }
        
        try:
            db_start = time.time()
            result = db.session.execute('SELECT 1 as test')
            db_status['response_time_ms'] = round((time.time() - db_start) * 1000, 2)
            db_status['connected'] = True
            
            # Get connection pool info if available
            if hasattr(db.engine.pool, 'size'):
                db_status['connection_pool_size'] = db.engine.pool.size()
            if hasattr(db.engine.pool, '_checked_in_connections'):
                db_status['active_connections'] = len(db.engine.pool._checked_in_connections)
                
        except Exception as e:
            db_status['error'] = str(e)
        
        # Cache detailed check
        cache_status = {
            'connected': False,
            'response_time_ms': 0,
            'hit_rate': 0,
            'memory_usage': 'N/A'
        }
        
        try:
            cache_start = time.time()
            test_key = f'health_check_{int(time.time())}'
            cache_manager.set(test_key, 'test_value', 10)
            cache_manager.get(test_key)
            cache_manager.delete(test_key)
            cache_status['response_time_ms'] = round((time.time() - cache_start) * 1000, 2)
            cache_status['connected'] = True
            
            # Get cache stats
            cache_stats = cache_manager.get_stats()
            cache_status['hit_rate'] = cache_stats.get('redis_hit_rate', 0)
            cache_status['memory_usage'] = cache_stats.get('redis_used_memory', 'N/A')
            
        except Exception as e:
            cache_status['error'] = str(e)
        
        # System metrics
        process = psutil.Process()
        system_metrics = {
            'cpu_usage_percent': psutil.cpu_percent(interval=0.1),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'memory_available_mb': round(psutil.virtual_memory().available / 1024 / 1024, 2),
            'process_memory_mb': round(process.memory_info().rss / 1024 / 1024, 2),
            'process_cpu_percent': process.cpu_percent(),
            'process_threads': process.num_threads(),
            'uptime_seconds': time.time() - psutil.boot_time()
        }
        
        # Performance metrics
        performance_metrics = get_performance_metrics(hours=1)
        
        # Recent alerts
        recent_alerts = get_recent_alerts(hours=1)
        
        # Determine overall health
        overall_healthy = (
            db_status['connected'] and 
            cache_status['connected'] and 
            system_metrics['cpu_usage_percent'] < 90 and 
            system_metrics['memory_usage_percent'] < 90
        )
        
        response_time = time.time() - start_time
        
        return jsonify({
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'response_time_ms': round(response_time * 1000, 2),
            'system_health': system_health,
            'database': db_status,
            'cache': cache_status,
            'system_metrics': system_metrics,
            'performance_metrics': performance_metrics,
            'recent_alerts': recent_alerts,
            'uptime_seconds': time.time() - psutil.boot_time()
        }), 200 if overall_healthy else 503
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 503

@health_bp.route('/health/readiness', methods=['GET'])
def readiness_check():
    """Kubernetes readiness probe - checks if service is ready to accept traffic"""
    try:
        # Check critical dependencies
        db_ready = False
        cache_ready = False
        
        # Database readiness
        try:
            db.session.execute('SELECT 1')
            db_ready = True
        except Exception:
            pass
        
        # Cache readiness
        try:
            cache_manager.set('readiness_check', 'ok', 5)
            cache_ready = True
        except Exception:
            pass
        
        # Service is ready if critical dependencies are available
        ready = db_ready and cache_ready
        
        return jsonify({
            'status': 'ready' if ready else 'not_ready',
            'timestamp': datetime.utcnow().isoformat(),
            'database_ready': db_ready,
            'cache_ready': cache_ready
        }), 200 if ready else 503
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            'status': 'not_ready',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 503

@health_bp.route('/health/liveness', methods=['GET'])
def liveness_check():
    """Kubernetes liveness probe - checks if service is alive"""
    try:
        # Basic liveness check - just ensure the process is responsive
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Service is alive if not completely overwhelmed
        alive = cpu_usage < 95 and memory.percent < 95
        
        return jsonify({
            'status': 'alive' if alive else 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_usage_percent': round(cpu_usage, 2),
            'memory_usage_percent': round(memory.percent, 2)
        }), 200 if alive else 503
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 503

@health_bp.route('/health/metrics', methods=['GET'])
def metrics_endpoint():
    """Prometheus-style metrics endpoint"""
    try:
        # Get system metrics
        system_health = get_system_health()
        performance_metrics = get_performance_metrics(hours=1)
        
        # Format as Prometheus metrics
        metrics = []
        
        # System metrics
        metrics.append(f"# HELP system_cpu_usage_percent CPU usage percentage")
        metrics.append(f"# TYPE system_cpu_usage_percent gauge")
        metrics.append(f"system_cpu_usage_percent {system_health.get('cpu_usage', 0)}")
        
        metrics.append(f"# HELP system_memory_usage_percent Memory usage percentage")
        metrics.append(f"# TYPE system_memory_usage_percent gauge")
        metrics.append(f"system_memory_usage_percent {system_health.get('memory_usage', 0)}")
        
        metrics.append(f"# HELP system_db_connections Database connections")
        metrics.append(f"# TYPE system_db_connections gauge")
        metrics.append(f"system_db_connections {system_health.get('db_connections', 0)}")
        
        metrics.append(f"# HELP system_cache_hit_rate Cache hit rate percentage")
        metrics.append(f"# TYPE system_cache_hit_rate gauge")
        metrics.append(f"system_cache_hit_rate {system_health.get('cache_hit_rate', 0)}")
        
        # Uptime
        uptime = system_health.get('uptime_seconds', 0)
        metrics.append(f"# HELP system_uptime_seconds System uptime in seconds")
        metrics.append(f"# TYPE system_uptime_seconds counter")
        metrics.append(f"system_uptime_seconds {uptime}")
        
        return '\n'.join(metrics), 200, {'Content-Type': 'text/plain; charset=utf-8'}
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        return f"# ERROR: {str(e)}", 500, {'Content-Type': 'text/plain; charset=utf-8'}

@health_bp.route('/health/status', methods=['GET'])
def status_endpoint():
    """Simple status endpoint for monitoring dashboards"""
    try:
        system_health = get_system_health()
        
        return jsonify({
            'service': 'kempian-backend',
            'version': '1.0.0',
            'status': system_health.get('status', 'unknown'),
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': system_health.get('uptime_seconds', 0),
            'cpu_usage': system_health.get('cpu_usage', 0),
            'memory_usage': system_health.get('memory_usage', 0),
            'active_alerts': system_health.get('active_alerts', 0)
        }), 200
        
    except Exception as e:
        logger.error(f"Status endpoint failed: {e}")
        return jsonify({
            'service': 'kempian-backend',
            'status': 'error',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 500

@health_bp.route('/health/scaling', methods=['GET'])
def scaling_status():
    """Scaling status and capacity information"""
    try:
        from .connection_pool_monitor import get_connection_pool_stats, get_connection_pool_recommendations
        from .memory_optimizer import get_memory_stats, get_memory_recommendations
        from .async_processor import get_async_stats
        
        # Get scaling statistics
        pool_stats = get_connection_pool_stats()
        memory_stats = get_memory_stats()
        async_stats = get_async_stats()
        
        # Calculate capacity estimates
        max_connections = pool_stats.get('max_connections', 0)
        active_connections = pool_stats.get('active_connections', 0)
        connection_usage = pool_stats.get('connection_usage_percent', 0)
        
        # Estimate concurrent user capacity
        if max_connections > 0:
            # Conservative estimate: 3 users per connection
            estimated_capacity = int(max_connections * 3)
            current_load = int(active_connections * 3)
        else:
            estimated_capacity = 0
            current_load = 0
        
        # Get recommendations
        pool_recommendations = get_connection_pool_recommendations()
        memory_recommendations = get_memory_recommendations()
        
        return jsonify({
            'scaling_status': {
                'estimated_capacity': estimated_capacity,
                'current_load': current_load,
                'capacity_utilization': (current_load / estimated_capacity * 100) if estimated_capacity > 0 else 0,
                'status': 'healthy' if connection_usage < 80 else 'warning' if connection_usage < 90 else 'critical'
            },
            'connection_pool': {
                'max_connections': max_connections,
                'active_connections': active_connections,
                'usage_percent': connection_usage,
                'recommendations': pool_recommendations.get('recommendations', [])
            },
            'memory': {
                'usage_percent': memory_stats.get('memory_percent', 0),
                'pressure': memory_stats.get('memory_pressure', 'unknown'),
                'recommendations': memory_recommendations
            },
            'async_processing': {
                'active_tasks': async_stats.get('active_tasks', 0),
                'queue_size': async_stats.get('queue_size', 0),
                'total_tasks': async_stats.get('total_tasks', 0),
                'avg_processing_time': async_stats.get('avg_processing_time', 0)
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Scaling status endpoint failed: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
