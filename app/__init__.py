import os
from dotenv import load_dotenv
from app.simple_logger import get_logger

# Load environment variables from the backend directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from flask import Flask
from flask_cors import CORS
from .config import Config
from .db import db
# from .models import CeipalIntegration  # Commented out - not in models directory

from flask import jsonify
from datetime import datetime

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Set debug mode based on config
    app.debug = app.config.get('DEBUG', False)
    
    # Initialize simple logging system
    try:
        from .simple_logger import get_logger
        app.logger = get_logger('app')
        app.logger.info("Kempian backend application starting up")
    except ImportError as e:
        print(f"Warning: Could not initialize logging system: {e}")
        # Fallback to basic logging
        import logging
        logging.basicConfig(level=logging.INFO)
        app.logger = logging.getLogger(__name__)
    
    # Only allow your frontend origin and support credentials
    # Do NOT use '*' for origins when supports_credentials=True, per CORS spec.
    CORS(app, supports_credentials=True, origins=[
        "http://localhost:8081", "http://127.0.0.1:8081",
        "http://localhost:5173", "http://127.0.0.1:5173",  # Vite default port
        "https://kempian.ai", "https://new.kempian.ai",
        "https://www.kempian.ai","https://www.new.kempian.ai",
         "https://new1.kempian.com",
        "http://localhost:8082", "http://127.0.0.1:8082",
    ])
    # NOTE: If you change CORS config, restart the backend server to apply changes.

    db.init_app(app)

    # Initialize JWT for AWS Cognito compatibility
    from flask_jwt_extended import JWTManager
    app.config["JWT_SECRET_KEY"] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    app.config["JWT_TOKEN_LOCATION"] = ["headers"]
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = False  # Tokens don't expire
    app.config["JWT_ALGORITHM"] = "RS256"  # AWS Cognito uses RS256
    app.config["JWT_DECODE_ALGORITHMS"] = ["RS256", "HS256"]  # Allow both RS256 and HS256
    JWTManager(app)

    # Register blueprints
    from .auth.routes import auth_bp
    from .tenants.routes import tenants_bp
    from .plans.routes import plans_bp
    from .search.routes import search_bp
    from .stripe.routes import stripe_bp
    from .stripe.webhook import webhook_bp
    from .ceipal.routes import ceipal_bp
    from .stafferlink.routes import stafferlink_bp
    from .subscription.routes import subscription_bp
    from .talent.routes import talent_bp
    from .admin.routes import admin_bp
    from .analytics.routes import analytics_bp
    from .analytics.kpi_routes import kpi_bp
    from .performance.routes import performance_bp
    from .performance.ultra_fast_routes import ultra_fast_bp
    from .performance.batch_signup_routes import batch_signup_bp
    from .jobs import jobs_bp
    from .health import health_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(tenants_bp, url_prefix="/tenant")
    app.register_blueprint(plans_bp, url_prefix="/plans")
    app.register_blueprint(search_bp, url_prefix="/search")
    app.register_blueprint(stripe_bp, url_prefix="/checkout")
    app.register_blueprint(webhook_bp, url_prefix="/webhook")
    app.register_blueprint(ceipal_bp, url_prefix="")
    app.register_blueprint(stafferlink_bp, url_prefix="")
    app.register_blueprint(subscription_bp, url_prefix="/subscription")
    app.register_blueprint(talent_bp, url_prefix="/talent")
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(analytics_bp, url_prefix="/analytics")
    app.register_blueprint(kpi_bp, url_prefix="/kpi")
    app.register_blueprint(performance_bp, url_prefix="/performance")
    app.register_blueprint(ultra_fast_bp, url_prefix="/ultra-fast")
    app.register_blueprint(batch_signup_bp, url_prefix="/batch-signup")
    app.register_blueprint(jobs_bp, url_prefix="/jobs")
    app.register_blueprint(health_bp, url_prefix="/health")
    
    # Register public analytics routes
    from .analytics.public_routes import public_analytics_bp
    from .jobs.public_routes import public_jobs_bp
    app.register_blueprint(public_analytics_bp, url_prefix="/public")
    app.register_blueprint(public_jobs_bp, url_prefix="/public")
    
    # Register logs dashboard (restricted to vinit@adeptaipro.com)
    from .logs_dashboard import logs_dashboard_bp
    app.register_blueprint(logs_dashboard_bp, url_prefix="")
    
    # Initialize Cognito password reset system
    from .auth.cognito_password_reset import init_cognito_password_reset
    init_cognito_password_reset(app)
    
    # Initialize performance monitoring
    from .monitoring import start_performance_monitoring
    start_performance_monitoring(interval=30)  # Monitor every 30 seconds
    
    # Initialize async processing
    from .async_processor import start_async_processing
    start_async_processing()
    
    # Initialize connection pool monitoring - TEMPORARILY DISABLED
    # from .connection_pool_monitor import start_connection_pool_monitoring
    # start_connection_pool_monitoring(interval=15)  # Monitor every 15 seconds
    
    # Initialize memory optimization
    from .memory_optimizer import start_memory_optimization
    start_memory_optimization(interval=30)  # Optimize every 30 seconds

    @app.route("/")
    def index():
        # Security check - don't expose debug info in production
        if app.debug:
            app.logger.info("Health check: DEBUG mode")
            return "Kempian backend is running in DEBUG mode!", 200
        else:
            app.logger.info("Health check: Production mode")
            return "Kempian backend is running!", 200

    @app.route("/health")
    def health_check():
        """Health check endpoint with logging"""
        app.logger.info("Health check endpoint accessed")
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }), 200

    # Log successful startup
    app.logger.info("Kempian backend application started successfully")

    return app 