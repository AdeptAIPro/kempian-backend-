import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'mysql+pymysql://localhost:3307/kempianDB'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
    COGNITO_USER_POOL_ID = os.environ.get('COGNITO_USER_POOL_ID', '')
    COGNITO_CLIENT_ID = os.environ.get('COGNITO_CLIENT_ID', '')
    COGNITO_REGION = os.environ.get('COGNITO_REGION', '')
    SES_REGION = os.environ.get('SES_REGION', '')
    SES_FROM_EMAIL = os.environ.get('SES_FROM_EMAIL', '')
    FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:3000')
    
    # Production settings
    DEBUG = os.environ.get('FLASK_DEBUG', '0').lower() in ('true', '1', 'yes')
    TESTING = os.environ.get('FLASK_TESTING', '0').lower() in ('true', '1', 'yes')
    
    # Security settings for production
    if not DEBUG:
        # Ensure HTTPS in production
        PREFERRED_URL_SCHEME = 'https'
        # Disable debug toolbar
        DEBUG_TB_ENABLED = False 