"""
Database Health Monitoring and Connection Management
Handles database connection issues and provides health checks
"""
import time
import threading
from sqlalchemy import text
from app.simple_logger import get_logger
from app.db import db

logger = get_logger("db_health")

class DatabaseHealthMonitor:
    """Monitor database health and handle connection issues"""
    
    def __init__(self):
        self.connection_healthy = True
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    def check_database_health(self):
        """Check if database connection is healthy"""
        try:
            # Simple health check query
            result = db.session.execute(text("SELECT 1")).scalar()
            self.connection_healthy = True
            self.last_health_check = time.time()
            logger.debug("Database health check passed")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            self.connection_healthy = False
            return False
    
    def get_connection_stats(self):
        """Get database connection statistics"""
        try:
            # Get connection pool stats
            engine = db.engine
            pool = engine.pool
            
            stats = {
                'pool_size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid(),
                'healthy': self.connection_healthy,
                'last_check': self.last_health_check
            }
            
            logger.info(f"Database connection stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get connection stats: {e}")
            return {'error': str(e)}
    
    def handle_connection_error(self, error):
        """Handle database connection errors with retry logic"""
        logger.error(f"Database connection error: {error}")
        
        # Try to reconnect
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting database reconnection (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(self.retry_delay)
                
                if self.check_database_health():
                    logger.info("Database reconnection successful")
                    return True
                    
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
                
        logger.error("All reconnection attempts failed")
        return False
    
    def optimize_queries(self):
        """Optimize database queries for better performance"""
        try:
            # Set MySQL session variables for better performance
            # Note: Only setting variables that are valid for SET SESSION
            db.session.execute(text("SET SESSION sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO'"))
            db.session.execute(text("SET SESSION tmp_table_size = 134217728"))  # 128MB
            db.session.execute(text("SET SESSION max_heap_table_size = 134217728"))  # 128MB
            db.session.commit()
            logger.info("Database session optimized for performance")
            
        except Exception as e:
            # Don't log as error since these optimizations are optional
            logger.debug(f"Could not set all optimization variables: {e}")

# Global health monitor instance
db_health_monitor = DatabaseHealthMonitor()

def get_db_health():
    """Get current database health status"""
    return db_health_monitor.check_database_health()

def get_db_stats():
    """Get database connection statistics"""
    return db_health_monitor.get_connection_stats()

def handle_db_error(error):
    """Handle database errors with automatic recovery"""
    return db_health_monitor.handle_connection_error(error)

def optimize_db_session():
    """Optimize database session for better performance"""
    db_health_monitor.optimize_queries()
