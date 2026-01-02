"""
Startup Optimizer
Initializes the optimized search system when the application starts.
"""

import logging
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)

def initialize_search_on_startup(embedding_service, redis_client=None):
    """Initialize the optimized search system on application startup"""
    try:
        logger.info("🚀 Starting search system optimization on startup...")
        
        # Start background initialization
        from app.search.background_initializer import start_background_initialization
        start_background_initialization(embedding_service, redis_client)
        
        # Also try immediate initialization if possible
        def immediate_init():
            try:
                from app.search.background_initializer import force_initialization
                success = force_initialization(embedding_service, redis_client)
                if success:
                    logger.info("✅ Immediate search system initialization successful")
                else:
                    logger.info("⏳ Search system initialization in progress...")
            except Exception as e:
                logger.warning(f"Immediate initialization failed: {e}")
        
        # Run immediate initialization in a separate thread
        init_thread = threading.Thread(target=immediate_init, daemon=True)
        init_thread.start()
        
        logger.info("Search system optimization started successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start search system optimization: {e}")
        return False

def get_startup_status() -> dict:
    """Get the status of startup optimization"""
    try:
        from app.search.background_initializer import get_initialization_status
        from app.search.search_initializer import get_search_status
        
        return {
            'background_initialization': get_initialization_status(),
            'search_systems': get_search_status(),
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Error getting startup status: {e}")
        return {
            'error': str(e),
            'timestamp': time.time()
        }
