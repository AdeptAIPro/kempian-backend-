#!/usr/bin/env python3
"""
AdeptAI Main Application Entry Point
===================================

This is the main entry point for the AdeptAI recruitment system.
It initializes all services and starts the Flask application.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app import create_app
from app.config import get_settings

def main():
    """Main application entry point"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting AdeptAI Recruitment System...")
        
        # Get settings
        settings = get_settings()
        
        # Create Flask app
        app = create_app()
        
        # Get configuration
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        debug = os.getenv('DEBUG', 'False').lower() == 'true'
        
        logger.info(f"🌐 Starting server on {host}:{port}")
        logger.info(f"🔧 Debug mode: {debug}")
        
        # Start the application
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Shutting down AdeptAI...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Failed to start AdeptAI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
