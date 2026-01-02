import platform
import time
from flask import Blueprint, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


health_bp = Blueprint("health", __name__)

# Create a separate limiter for health endpoints with higher limits
health_limiter = Limiter(key_func=get_remote_address, default_limits=["10000/hour"])


@health_bp.get("/health")
@health_limiter.limit("10000/hour")  # Very high limit for health checks
def health():
    search_ready = current_app.extensions.get("search_system") is not None
    return jsonify({
        "status": "ok",
        "time": time.time(),
        "service": "adeptai",
        "version": 1,
        "search_ready": search_ready,
        "python": platform.python_version(),
    })


