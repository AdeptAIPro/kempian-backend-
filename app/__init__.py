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
from .module_isolation import safe_import_and_register

from flask import jsonify
from datetime import datetime

def create_app():
    app = Flask(__name__)
    
    # Try to use production config, fallback to development config
    try:
        app.config.from_object(Config)
        app.logger.info("Using production configuration")
    except Exception as e:
        app.logger.warning(f"Production config failed, using development config: {e}")
        from ..dev_config import DevConfig
        app.config.from_object(DevConfig)
    
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
    # Configure CORS with explicit headers and methods to handle preflight requests properly
    allowed_origins = [
        "http://localhost:8081", "http://127.0.0.1:8081",
        "http://localhost:5173", "http://127.0.0.1:5173",  # Vite default port
        "https://kempian.ai", "https://new.kempian.ai",
        "https://www.kempian.ai", "https://www.new.kempian.ai",
        "https://new1.kempian.com",
        "https://kempian.in", "https://kempian.eu",
        "http://localhost:8082", "http://127.0.0.1:8082",
    ]
    
    CORS(app, 
         supports_credentials=True, 
         origins=allowed_origins,
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
         allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
         expose_headers=['Content-Type', 'Authorization'],
         max_age=3600  # Cache preflight requests for 1 hour
    )
    # NOTE: If you change CORS config, restart the backend server to apply changes.

    db.init_app(app)
    
    # Initialize database health monitoring (run with app context)
    from .db_health import db_health_monitor, optimize_db_session
    def init_db_health():
        try:
            with app.app_context():
                optimize_db_session()
            app.logger.info("Database health monitoring initialized")
        except Exception as e:
            app.logger.warning(f"Database health monitoring setup failed: {e}")
    
    init_db_health()

    # Initialize JWT for AWS Cognito compatibility
    from flask_jwt_extended import JWTManager
    app.config["JWT_SECRET_KEY"] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    app.config["JWT_TOKEN_LOCATION"] = ["headers"]
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = False  # Tokens don't expire
    
    # Check if we have a public key for RS256, otherwise use HS256
    jwt_public_key = os.getenv('JWT_PUBLIC_KEY')
    if jwt_public_key:
        app.config["JWT_ALGORITHM"] = "RS256"
        app.config["JWT_PUBLIC_KEY"] = jwt_public_key
    else:
        # Use HS256 for development when no public key is set
        app.config["JWT_ALGORITHM"] = "HS256"
        app.logger.info("Using HS256 for JWT (no public key set)")
    
    app.config["JWT_DECODE_ALGORITHMS"] = ["RS256", "HS256"]  # Allow both RS256 and HS256
    JWTManager(app)

    # Multitenancy middleware: set request-scoped tenant context
    from flask import g, request
    from .utils import get_current_user, get_current_user_flexible
    from .simple_logger import get_logger
    @app.before_request
    def _inject_tenant_context():
        try:
            # Support both Cognito JWTs and custom encoded tokens
            user = get_current_user_flexible() or get_current_user()
            g.user = user
            
            logger = get_logger("tenant_context")
            logger.debug(f"Tenant context - User: {user}")
            
            if user:
                # Try to get tenant_id from JWT token first
                tenant_id = user.get('custom:tenant_id')
                logger.debug(f"Tenant context - JWT tenant_id: {tenant_id}")
                
                if tenant_id:
                    g.tenant_id = int(tenant_id)
                    logger.debug(f"Tenant context - Using JWT tenant_id: {g.tenant_id}")
                else:
                    # Fallback: get tenant_id from database using email
                    from .models import User
                    email = user.get('email')
                    logger.debug(f"Tenant context - Fallback lookup for email: {email}")
                    
                    if email:
                        db_user = User.query.filter_by(email=email).first()
                        if db_user:
                            g.tenant_id = db_user.tenant_id
                            logger.debug(f"Tenant context - Found user in DB, tenant_id: {g.tenant_id}")
                        else:
                            g.tenant_id = None
                            logger.warning(f"Tenant context - User not found in DB for email: {email}")
                    else:
                        g.tenant_id = None
                        logger.warning("Tenant context - No email in JWT token")
            else:
                g.tenant_id = None
                logger.warning("Tenant context - No user from JWT token")
                
            logger.debug(f"Tenant context - Final tenant_id: {g.tenant_id}")
            
        except Exception as e:
            logger = get_logger("tenant_context")
            logger.error(f"Error setting tenant context: {e}")
            g.user = None
            g.tenant_id = None

    # Enforce tenant context on protected routes (global guard)
    @app.before_request
    def _enforce_tenant_on_protected_routes():
        # Skip preflight and public endpoints
        if request.method == 'OPTIONS':
            return None
        path = request.path or ''
        public_prefixes = (
            '/', '/health', '/public', '/auth', '/webhook', '/checkout', '/api/linkedin'
        )
        for prefix in public_prefixes:
            if path == prefix or path.startswith(prefix + '/'):  # treat as public
                return None

        # Only enforce for these service areas
        protected_prefixes = (
            '/jobs', '/analytics', '/kpi', '/subscription', '/talent', '/search', '/performance', '/ultra-fast'
        )
        should_enforce = any(path == p or path.startswith(p + '/') for p in protected_prefixes)
        if not should_enforce:
            return None
        # Allow explicit public subpaths
        if '/public' in path:
            return None
        # Require tenant context
        if not getattr(g, 'tenant_id', None):
            return jsonify({'error': 'Tenant context required'}), 403
        return None

    # Track request start time for activity logging
    @app.before_request
    def _track_request_start():
        """Track request start time for activity logging"""
        if not hasattr(g, 'start_time'):
            import time
            g.start_time = time.time()
    
    # Automatic user activity logging middleware
    @app.after_request
    def _auto_log_user_activity(response):
        """Automatically log all user activities for authenticated users"""
        try:
            from app.utils.user_activity_logger import auto_log_user_activity
            return auto_log_user_activity(response)
        except Exception as e:
            app.logger.error(f"Error in auto-logging middleware: {e}")
            return response

    def register_module(module_name, module_path, blueprint_name, url_prefix="", config_flag=None, default_enabled=True, required=False):
        """Register a module safely with isolation and optional feature flags."""
        enabled = default_enabled if config_flag is None else app.config.get(config_flag, default_enabled)
        if not enabled:
            app.logger.info(f"{module_name} module disabled via config flag {config_flag}")
            return None

        blueprint = safe_import_and_register(
            app,
            module_path,
            blueprint_name,
            url_prefix,
            module_name,
            required=required
        )

        if blueprint is None:
            app.logger.warning(f"{module_name} module not registered (see logs for details)")

        return blueprint

    module_registrations = [
        {"module_name": "module_health", "module_path": "app.module_health", "blueprint_name": "module_health_bp", "url_prefix": ""},
        {"module_name": "auth", "module_path": "app.auth.routes", "blueprint_name": "auth_bp", "url_prefix": "/auth", "config_flag": "ENABLE_SERVICE_AUTH", "default_enabled": True},
        {"module_name": "tenants", "module_path": "app.tenants.routes", "blueprint_name": "tenants_bp", "url_prefix": "/tenant"},
        {"module_name": "plans", "module_path": "app.plans.routes", "blueprint_name": "plans_bp", "url_prefix": "/plans"},
        {"module_name": "search", "module_path": "app.search.routes", "blueprint_name": "search_bp", "url_prefix": "/search", "config_flag": "ENABLE_SERVICE_SEARCH", "default_enabled": True},
        {"module_name": "matchmaking", "module_path": "app.matchmaking.routes", "blueprint_name": "matchmaking_bp", "url_prefix": "/matchmaking", "config_flag": "ENABLE_SERVICE_MATCHMAKING", "default_enabled": True},
        {"module_name": "stripe", "module_path": "app.stripe.routes", "blueprint_name": "stripe_bp", "url_prefix": "/checkout"},
        {"module_name": "stripe_webhook", "module_path": "app.stripe.webhook", "blueprint_name": "webhook_bp", "url_prefix": "/webhook"},
        {"module_name": "ceipal", "module_path": "app.ceipal.routes", "blueprint_name": "ceipal_bp", "url_prefix": ""},
        {"module_name": "stafferlink", "module_path": "app.stafferlink.routes", "blueprint_name": "stafferlink_bp", "url_prefix": ""},
        {"module_name": "jobadder", "module_path": "app.jobadder.routes", "blueprint_name": "jobadder_bp", "url_prefix": ""},
        {"module_name": "jobvite", "module_path": "app.jobvite.routes", "blueprint_name": "jobvite_bp", "url_prefix": ""},
        {"module_name": "jobvite_webhooks", "module_path": "app.jobvite.webhooks", "blueprint_name": "webhook_bp", "url_prefix": ""},
        {"module_name": "jobvite_documents", "module_path": "app.jobvite.routes_documents", "blueprint_name": "documents_bp", "url_prefix": ""},
        {"module_name": "linkedin_recruiter", "module_path": "app.linkedin_recruiter.routes", "blueprint_name": "linkedin_recruiter_bp", "url_prefix": ""},
        {"module_name": "integrations", "module_path": "app.integrations.routes", "blueprint_name": "integrations_bp", "url_prefix": "/api"},
        {"module_name": "subscription", "module_path": "app.subscription.routes", "blueprint_name": "subscription_bp", "url_prefix": "/subscription", "config_flag": "ENABLE_SERVICE_SUBSCRIPTION", "default_enabled": True},
        {"module_name": "talent", "module_path": "app.talent.routes", "blueprint_name": "talent_bp", "url_prefix": "/talent", "config_flag": "ENABLE_SERVICE_TALENT", "default_enabled": True},
        {"module_name": "communications", "module_path": "app.communications.routes", "blueprint_name": "communications_bp", "url_prefix": "/api/communications"},
        {"module_name": "communications_webhooks", "module_path": "app.communications.webhooks", "blueprint_name": "webhooks_bp", "url_prefix": "/webhooks"},
        {"module_name": "admin", "module_path": "app.admin.routes", "blueprint_name": "admin_bp", "url_prefix": "/admin"},
        {"module_name": "analytics", "module_path": "app.analytics.routes", "blueprint_name": "analytics_bp", "url_prefix": "/analytics", "config_flag": "ENABLE_SERVICE_ANALYTICS", "default_enabled": True},
        {"module_name": "analytics_kpi", "module_path": "app.analytics.kpi_routes", "blueprint_name": "kpi_bp", "url_prefix": "/kpi", "config_flag": "ENABLE_SERVICE_ANALYTICS", "default_enabled": True},
        {"module_name": "performance", "module_path": "app.performance.routes", "blueprint_name": "performance_bp", "url_prefix": "/performance"},
        {"module_name": "performance_ultra_fast", "module_path": "app.performance.ultra_fast_routes", "blueprint_name": "ultra_fast_bp", "url_prefix": "/ultra-fast"},
        {"module_name": "performance_batch_signup", "module_path": "app.performance.batch_signup_routes", "blueprint_name": "batch_signup_bp", "url_prefix": "/batch-signup"},
        {"module_name": "onboarding", "module_path": "app.onboarding.routes", "blueprint_name": "onboarding_bp", "url_prefix": "/api"},
        {"module_name": "user", "module_path": "app.user.routes", "blueprint_name": "user_bp", "url_prefix": "/api/user"},
        {"module_name": "jobs", "module_path": "app.jobs", "blueprint_name": "jobs_bp", "url_prefix": "/jobs", "config_flag": "ENABLE_SERVICE_JOBS", "default_enabled": True},
        {"module_name": "ai", "module_path": "app.ai.routes", "blueprint_name": "ai_bp", "url_prefix": "/api/ai"},
        {"module_name": "llm", "module_path": "app.llm.routes", "blueprint_name": "llm_bp", "url_prefix": "/api/llm"},
        {"module_name": "linkedin", "module_path": "app.linkedin.routes", "blueprint_name": "linkedin_bp", "url_prefix": "/api/linkedin"},
        {"module_name": "linkedin_auth", "module_path": "app.auth.linkedin_routes", "blueprint_name": "linkedin_auth_bp", "url_prefix": "/auth/linkedin"},
        {"module_name": "linkedin_api", "module_path": "app.auth.linkedin_api", "blueprint_name": "linkedin_api_bp", "url_prefix": "/api/me/linkedin"},
        {"module_name": "linkedin_revoke", "module_path": "app.auth.linkedin_revoke", "blueprint_name": "linkedin_revoke_bp", "url_prefix": "/api/me/linkedin"},
        {"module_name": "candidate_search_history", "module_path": "app.candidate_search_history", "blueprint_name": "candidate_search_bp", "url_prefix": ""},
        {"module_name": "trial_notifications", "module_path": "app.trial_notifications.routes", "blueprint_name": "trial_notifications_bp", "url_prefix": "/trial-notifications", "default_enabled": True},
        {"module_name": "meetings", "module_path": "app.meetings.routes", "blueprint_name": "meeting_bp", "url_prefix": "/api", "default_enabled": True},
        {"module_name": "analytics_public", "module_path": "app.analytics.public_routes", "blueprint_name": "public_analytics_bp", "url_prefix": "/public"},
        {"module_name": "jobs_public", "module_path": "app.jobs.public_routes", "blueprint_name": "public_jobs_bp", "url_prefix": "/public"},
        {"module_name": "contact", "module_path": "app.contact", "blueprint_name": "contact_bp", "url_prefix": ""},
        {"module_name": "hr_employees", "module_path": "app.hr.employees", "blueprint_name": "hr_employees_bp", "url_prefix": "/api/hr/employees"},
        {"module_name": "hr_organizations", "module_path": "app.hr.organizations", "blueprint_name": "hr_organizations_bp", "url_prefix": "/api/hr/organizations"},
        {"module_name": "hr_timesheets", "module_path": "app.hr.timesheets", "blueprint_name": "hr_timesheets_bp", "url_prefix": "/api/hr/timesheets"},
        {"module_name": "hr_payslips", "module_path": "app.hr.payslips", "blueprint_name": "hr_payslips_bp", "url_prefix": "/api/hr/payslips"},
        {"module_name": "hr_payroll", "module_path": "app.hr.payroll", "blueprint_name": "hr_payroll_bp", "url_prefix": "/api/hr/payroll"},
        {"module_name": "hr_tax", "module_path": "app.hr.tax_management", "blueprint_name": "hr_tax_bp", "url_prefix": "/api/hr/tax"},
        {"module_name": "hr_deductions", "module_path": "app.hr.deductions", "blueprint_name": "hr_deductions_bp", "url_prefix": "/api/hr/deductions"},
        {"module_name": "hr_payruns", "module_path": "app.hr.payruns", "blueprint_name": "hr_payruns_bp", "url_prefix": "/api/hr/payruns"},
        {"module_name": "hr_payroll_settings", "module_path": "app.hr.payroll_settings", "blueprint_name": "hr_payroll_settings_bp", "url_prefix": "/api/hr/payroll-settings"},
        {"module_name": "hr_leave", "module_path": "app.hr.leave_management", "blueprint_name": "hr_leave_bp", "url_prefix": "/api/hr/leave"},
        {"module_name": "hr_compliance", "module_path": "app.hr.compliance_reports", "blueprint_name": "hr_compliance_bp", "url_prefix": "/api/hr/compliance"},
        {"module_name": "india_compliance", "module_path": "app.hr.india_compliance_routes", "blueprint_name": "india_compliance_bp", "url_prefix": "/api/hr/india-compliance"},
        {"module_name": "us_compliance", "module_path": "app.hr.us_compliance_routes", "blueprint_name": "us_compliance_bp", "url_prefix": "/api/hr/us-compliance"},
        {"module_name": "international", "module_path": "app.hr.international_routes", "blueprint_name": "international_bp", "url_prefix": "/api/hr"},
        {"module_name": "enhanced", "module_path": "app.hr.enhanced_routes", "blueprint_name": "enhanced_bp", "url_prefix": "/api/hr"},
        {"module_name": "logs_dashboard", "module_path": "app.logs_dashboard", "blueprint_name": "logs_dashboard_bp", "url_prefix": ""},
        {"module_name": "health_routes", "module_path": "app.health_routes", "blueprint_name": "health_bp", "url_prefix": "/health"},
        {"module_name": "bulk_upload", "module_path": "app.bulk_upload.routes", "blueprint_name": "bulk_upload_bp", "url_prefix": ""},
        {"module_name": "candidate_resume", "module_path": "app.candidate_resume.routes", "blueprint_name": "candidate_resume_bp", "url_prefix": ""},
        {"module_name": "support", "module_path": "app.support.routes", "blueprint_name": "support_bp", "url_prefix": "/api/support"},
        {"module_name": "candidate_upload", "module_path": "app.candidate_upload.routes", "blueprint_name": "upload_bp", "url_prefix": "/api/candidate-upload"},
        {"module_name": "candidates_api", "module_path": "app.candidates.routes", "blueprint_name": "candidates_bp", "url_prefix": "/api"},
        {"module_name": "payment_webhooks", "module_path": "app.hr.payment_webhooks", "blueprint_name": "payment_webhooks_bp", "url_prefix": "/api/hr/payments/webhooks"},
        {"module_name": "fraud_alerts", "module_path": "app.hr.fraud_alerts", "blueprint_name": "fraud_alerts_bp", "url_prefix": "/api/hr/fraud-alerts"},
        {"module_name": "employee_payments", "module_path": "app.hr.employee_payments", "blueprint_name": "employee_payments_bp", "url_prefix": "/api/hr/employee-payments"},
    ]

    for module_config in module_registrations:
        register_module(**module_config)

    
    # Initialize Cognito password reset system
    try:
        from .auth.cognito_password_reset import init_cognito_password_reset
        init_cognito_password_reset(app)
    except Exception as e:
        app.logger.warning(f"Failed to initialize Cognito password reset: {e}")
    
    # Initialize performance monitoring
    try:
        from .monitoring import start_performance_monitoring
        start_performance_monitoring(interval=30)  # Monitor every 30 seconds
    except Exception as e:
        app.logger.warning(f"Failed to start performance monitoring: {e}")
    
    # Initialize async processing
    try:
        from .async_processor import start_async_processing
        start_async_processing()
    except Exception as e:
        app.logger.warning(f"Failed to start async processing: {e}")
    
    # Initialize connection pool monitoring - TEMPORARILY DISABLED
    # from .connection_pool_monitor import start_connection_pool_monitoring
    # start_connection_pool_monitoring(interval=15)  # Monitor every 15 seconds
    
    # Initialize memory optimization
    try:
        from .memory_optimizer import start_memory_optimization
        start_memory_optimization(interval=30)  # Optimize every 30 seconds
    except Exception as e:
        app.logger.warning(f"Failed to start memory optimization: {e}")
    
    # Initialize optimized search system
    try:
        from .search.startup_optimizer import initialize_search_on_startup
        try:
            from .search.adeptai_master.app.services import get_embedding_service  # type: ignore
        except ImportError:
            try:
                from .search.adeptai_components.embedding_service import get_embedding_service  # type: ignore
            except ImportError:
                def get_embedding_service():
                    return None
        
        embedding_service = get_embedding_service()
        if embedding_service:
            initialize_search_on_startup(embedding_service)
            app.logger.info("🚀 Search system optimization started")
        else:
            app.logger.warning("Embedding service not available for search optimization")
    except Exception as e:
        app.logger.warning(f"Could not initialize search optimization: {e}")

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