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
        'pool_size': 100,          # Base number of connections (increased from 50)
        'max_overflow': 200,       # Additional connections when needed (increased from 100)
        'pool_timeout': 60,        # Seconds to wait for connection (increased from 30)
        'pool_recycle': 7200,      # Recycle connections every 2 hours (increased from 1)
        'pool_pre_ping': True,     # Verify connections before use
        'echo': False,             # Disable SQL logging in production
        'connect_args': {
            'charset': 'utf8mb4',
            'autocommit': True,
            'connect_timeout': 10,
            'read_timeout': 30,
            'write_timeout': 30
        }
    }
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
    COGNITO_USER_POOL_ID = os.environ.get('COGNITO_USER_POOL_ID', '')
    COGNITO_CLIENT_ID = os.environ.get('COGNITO_CLIENT_ID', '')
    COGNITO_REGION = os.environ.get('COGNITO_REGION', '')
    SES_REGION = os.environ.get('SES_REGION', '')
    SES_FROM_EMAIL = os.environ.get('SES_FROM_EMAIL', '')
    FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:8081')
    
    # Production settings
    DEBUG = os.environ.get('FLASK_DEBUG', '0').lower() in ('true', '1', 'yes')
    TESTING = os.environ.get('FLASK_TESTING', '0').lower() in ('true', '1', 'yes')
    
    # Security settings for production
    if not DEBUG:
        # Ensure HTTPS in production
        PREFERRED_URL_SCHEME = 'https'
        # Disable debug toolbar
        DEBUG_TB_ENABLED = False 