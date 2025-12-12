from app.simple_logger import get_logger
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'mysql+pymysql://localhost:3307/kempianDB'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Database connection pooling for high concurrency (2000+ users)
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,           # Reduced from 100 - too many connections cause issues
        'max_overflow': 30,        # Reduced from 200 - prevent connection exhaustion
        'pool_timeout': 30,        # Reduced from 60 - faster timeout
        'pool_recycle': 3600,      # Recycle connections every 1 hour (reduced from 2)
        'pool_pre_ping': True,     # Verify connections before use
        'echo': False,             # Disable SQL logging in production
        'connect_args': {
            'charset': 'utf8mb4',
            'autocommit': True,
            'connect_timeout': 5,   # Reduced from 10 - faster connection
            'read_timeout': 15,     # Reduced from 30 - faster reads
            'write_timeout': 15,    # Reduced from 30 - faster writes
            'init_command': "SET SESSION wait_timeout=28800, interactive_timeout=28800"
        }
    }
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
    COGNITO_USER_POOL_ID = os.environ.get('COGNITO_USER_POOL_ID', '')
    COGNITO_CLIENT_ID = os.environ.get('COGNITO_CLIENT_ID', '')
    COGNITO_REGION = os.environ.get('COGNITO_REGION', '')
    SES_REGION = os.environ.get('SES_REGION', '')
    SES_FROM_EMAIL = os.environ.get('SES_FROM_EMAIL', '')
    FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:8081')
    
    # LinkedIn Integration Settings
    LINKEDIN_CLIENT_ID = os.environ.get('LINKEDIN_CLIENT_ID', '')
    LINKEDIN_CLIENT_SECRET = os.environ.get('LINKEDIN_CLIENT_SECRET', '')
    LINKEDIN_REDIRECT_URI = os.environ.get('LINKEDIN_REDIRECT_URI', 'http://localhost:5173/linkedin-callback')
    
    # Service mode flags (monolith vs microservices)
    SERVICE_MODE = os.environ.get('SERVICE_MODE', 'monolith')  # 'monolith' or 'services'
    ENABLE_SERVICE_AUTH = os.environ.get('ENABLE_SERVICE_AUTH', '1') in ('1','true','True')
    ENABLE_SERVICE_JOBS = os.environ.get('ENABLE_SERVICE_JOBS', '1') in ('1','true','True')
    ENABLE_SERVICE_SEARCH = os.environ.get('ENABLE_SERVICE_SEARCH', '1') in ('1','true','True')
    ENABLE_SERVICE_ANALYTICS = os.environ.get('ENABLE_SERVICE_ANALYTICS', '1') in ('1','true','True')
    ENABLE_SERVICE_SUBSCRIPTION = os.environ.get('ENABLE_SERVICE_SUBSCRIPTION', '1') in ('1','true','True')
    ENABLE_SERVICE_TALENT = os.environ.get('ENABLE_SERVICE_TALENT', '1') in ('1','true','True')
    
    # Production settings
    DEBUG = os.environ.get('FLASK_DEBUG', '0').lower() in ('true', '1', 'yes')
    TESTING = os.environ.get('FLASK_TESTING', '0').lower() in ('true', '1', 'yes')
    
    # Security settings for production
    if not DEBUG:
        # Ensure HTTPS in production
        PREFERRED_URL_SCHEME = 'https'
        # Disable debug toolbar
        DEBUG_TB_ENABLED = False 