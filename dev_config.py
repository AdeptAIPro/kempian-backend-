"""
Development Configuration for Kempian Backend
This file provides fallback configuration when services are not available
"""
import os
from app.simple_logger import get_logger

logger = get_logger("dev_config")

class DevConfig:
    """Development configuration with fallbacks for missing services"""
    
    # Flask Configuration
    SECRET_KEY = 'dev-secret-key-for-development'
    FLASK_ENV = 'development'
    FLASK_DEBUG = True
    
    # Database Configuration - Use SQLite for development
    SQLALCHEMY_DATABASE_URI = 'sqlite:///instance/app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Database connection pooling for development
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'max_overflow': 20,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'echo': False,
    }
    
    # AWS Configuration (with fallbacks)
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID', '')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
    AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')
    
    # S3 Configuration
    S3_BUCKET = os.environ.get('S3_BUCKET', 'resume-bucket-adept-ai-pro')
    RESUME_BUCKET = os.environ.get('RESUME_BUCKET', 'resume-bucket-adept-ai-pro')
    RESUME_PREFIX = os.environ.get('RESUME_PREFIX', 'career_resume/')
    
    # DynamoDB Tables
    DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'resume_metadata')
    FEEDBACK_TABLE = os.environ.get('FEEDBACK_TABLE', 'resume_feedback')
    
    # AWS SES Configuration
    SES_REGION = os.environ.get('SES_REGION', 'ap-south-1')
    SES_FROM_EMAIL = os.environ.get('SES_FROM_EMAIL', 'noreply@yourdomain.com')
    
    # AWS Cognito Configuration
    COGNITO_USER_POOL_ID = os.environ.get('COGNITO_USER_POOL_ID', '')
    COGNITO_CLIENT_ID = os.environ.get('COGNITO_CLIENT_ID', '')
    COGNITO_REGION = os.environ.get('COGNITO_REGION', 'ap-south-1')
    
    # CloudFront Configuration
    CLOUDFRONT_DOMAIN = os.environ.get('CLOUDFRONT_DOMAIN', '')
    CF_PRIVATE_KEY_SECRET = os.environ.get('CF_PRIVATE_KEY_SECRET', '')
    
    # Stripe Configuration
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
    
    # Frontend URL
    FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:3000')
    
    # Redis Configuration (with fallback to in-memory)
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
    REDIS_DB = int(os.environ.get('REDIS_DB', '0'))
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', '')
    
    # Service Configuration
    SERVICE_MODE = os.environ.get('SERVICE_MODE', 'monolith')
    ENABLE_SERVICE_AUTH = os.environ.get('ENABLE_SERVICE_AUTH', '1') in ('1','true','True')
    ENABLE_SERVICE_JOBS = os.environ.get('ENABLE_SERVICE_JOBS', '1') in ('1','true','True')
    ENABLE_SERVICE_SEARCH = os.environ.get('ENABLE_SERVICE_SEARCH', '1') in ('1','true','True')
    ENABLE_SERVICE_ANALYTICS = os.environ.get('ENABLE_SERVICE_ANALYTICS', '1') in ('1','true','True')
    ENABLE_SERVICE_SUBSCRIPTION = os.environ.get('ENABLE_SERVICE_SUBSCRIPTION', '1') in ('1','true','True')
    ENABLE_SERVICE_TALENT = os.environ.get('ENABLE_SERVICE_TALENT', '1') in ('1','true','True')
    
    # Search Configuration
    MAX_CANDIDATES_SMALL = int(os.environ.get('MAX_CANDIDATES_SMALL', '10000'))
    MAX_CANDIDATES_MEDIUM = int(os.environ.get('MAX_CANDIDATES_MEDIUM', '50000'))
    MAX_CANDIDATES_LARGE = int(os.environ.get('MAX_CANDIDATES_LARGE', '100000'))
    CACHE_VALIDITY_HOURS = int(os.environ.get('CACHE_VALIDITY_HOURS', '1'))
    FORCE_FULL_LOAD = os.environ.get('FORCE_FULL_LOAD', 'false').lower() in ('true', '1', 'yes')
    ENABLE_PARALLEL_PROCESSING = os.environ.get('ENABLE_PARALLEL_PROCESSING', 'true').lower() in ('true', '1', 'yes')
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '4'))
    VECTOR_CACHE_SIZE = int(os.environ.get('VECTOR_CACHE_SIZE', '1000'))
    ENABLE_SMART_FILTERING = os.environ.get('ENABLE_SMART_FILTERING', 'true').lower() in ('true', '1', 'yes')
    ULTRA_FAST_MODE = os.environ.get('ULTRA_FAST_MODE', 'true').lower() in ('true', '1', 'yes')
    FAST_LOAD_CANDIDATES = int(os.environ.get('FAST_LOAD_CANDIDATES', '200'))
    
    # Development settings
    DEBUG = True
    TESTING = False
    
    def __init__(self):
        logger.info("Development configuration loaded with fallbacks for missing services")
        logger.info(f"Database: {self.SQLALCHEMY_DATABASE_URI}")
        logger.info(f"Redis: {self.REDIS_URL}")
        logger.info(f"Frontend: {self.FRONTEND_URL}")
