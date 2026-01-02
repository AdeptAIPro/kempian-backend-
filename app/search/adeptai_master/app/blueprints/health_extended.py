"""
Extended health check endpoints for monitoring and observability
"""

from flask import Blueprint, jsonify, request
from typing import Dict, Any
import asyncio
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from ..health_checks import get_health_status, get_readiness_status, get_liveness_status
from ..exceptions import ServiceInitializationError

logger = logging.getLogger(__name__)

health_extended_bp = Blueprint("health_extended", __name__)

# Create a separate limiter for health endpoints with higher limits
health_limiter = Limiter(key_func=get_remote_address, default_limits=["10000/hour"])


@health_extended_bp.route("/health", methods=["GET"])
@health_limiter.limit("10000/hour")  # Very high limit for health checks
def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Run async health checks
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        health_data = loop.run_until_complete(get_health_status())
        loop.close()
        
        # Determine HTTP status code based on health status
        status_code = 200
        if health_data["status"] == "unhealthy":
            status_code = 503
        elif health_data["status"] == "degraded":
            status_code = 200  # Still operational but with warnings
        
        return jsonify(health_data), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "timestamp": "unknown"
        }), 503


@health_extended_bp.route("/health/ready", methods=["GET"])
@health_limiter.limit("10000/hour")  # Very high limit for health checks
def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        readiness_data = loop.run_until_complete(get_readiness_status())
        loop.close()
        
        status_code = 200 if readiness_data["ready"] else 503
        return jsonify(readiness_data), status_code
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            "ready": False,
            "message": f"Readiness check failed: {str(e)}",
            "timestamp": "unknown"
        }), 503


@health_extended_bp.route("/health/live", methods=["GET"])
@health_limiter.limit("10000/hour")  # Very high limit for health checks
def liveness_check():
    """Kubernetes liveness probe endpoint"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        liveness_data = loop.run_until_complete(get_liveness_status())
        loop.close()
        
        return jsonify(liveness_data), 200
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return jsonify({
            "alive": False,
            "message": f"Liveness check failed: {str(e)}",
            "timestamp": "unknown"
        }), 503


@health_extended_bp.route("/health/metrics", methods=["GET"])
def metrics():
    """Prometheus-style metrics endpoint"""
    try:
        import psutil
        import time
        
        # Basic system metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        metrics = {
            "system": {
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "disk_free_gb": disk.free / (1024**3),
                "cpu_usage_percent": cpu_percent,
                "cpu_count": psutil.cpu_count()
            },
            "application": {
                "uptime_seconds": time.time(),  # This would be actual uptime
                "timestamp": time.time()
            }
        }
        
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return jsonify({
            "error": f"Metrics collection failed: {str(e)}"
        }), 500


@health_extended_bp.route("/health/status", methods=["GET"])
def status():
    """Simple status endpoint for load balancers"""
    try:
        # Quick check - just ensure the application is responding
        return jsonify({
            "status": "ok",
            "timestamp": "unknown"  # This would be actual timestamp
        }), 200
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 503
