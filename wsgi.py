#!/usr/bin/env python3
"""
Production WSGI entry point for Kempian backend
"""
import os

# Set production environment
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_DEBUG'] = '0'

from app import create_app

# Create the Flask application
application = create_app()

if __name__ == "__main__":
    # For local testing only - use gunicorn/uwsgi in production
    application.run(host='0.0.0.0', port=8000, debug=False)
