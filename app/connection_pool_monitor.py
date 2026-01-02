"""
Connection Pool Monitoring for High-Scale Operations
Monitors database connection pool usage and provides optimization
"""
import time
import threading
from typing import Dict, Any
from app.simple_logger import get_logger
from app import db

logger = get_logger("connection_pool_monitor")

class ConnectionPoolMonitor:
    """Monitor and optimize database connection pool usage"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitor_thread = None
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'idle_connections': 0,
            'overflow_connections': 0,
            'max_connections': 0,
            'connection_usage_percent': 0,
            'peak_usage': 0,
            'avg_usage': 0,
            'usage_history': []
        }
        
        logger.info("Connection pool monitor initialized")
    
    def start_monitoring(self, interval: int = 10):
        """Start connection pool monitoring"""
        if self.monitoring_active:
            logger.warning("Connection pool monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Connection pool monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop connection pool monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Connection pool monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_pool_stats()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in connection pool monitoring: {e}")
                time.sleep(interval)
    
    def _collect_pool_stats(self):
        """Collect current connection pool statistics"""
        try:
            from flask import current_app
            
            # Check if we're in an application context
            if not current_app:
                logger.debug("Skipping pool stats collection - no application context")
                return
                
            pool = db.engine.pool
            
            # Get pool statistics
            total_connections = pool.size() + pool.overflow()
            active_connections = len(pool._checked_in_connections) if hasattr(pool, '_checked_in_connections') else 0
            idle_connections = total_connections - active_connections
            overflow_connections = pool.overflow()
            max_connections = pool.size() + pool.overflow()
            
            # Calculate usage percentage
            usage_percent = (active_connections / max_connections * 100) if max_connections > 0 else 0
            
            # Update stats
            self.stats.update({
                'total_connections': total_connections,
                'active_connections': active_connections,
                'idle_connections': idle_connections,
                'overflow_connections': overflow_connections,
                'max_connections': max_connections,
                'connection_usage_percent': usage_percent,
                'timestamp': time.time()
            })
            
            # Update peak usage
            if usage_percent > self.stats['peak_usage']:
                self.stats['peak_usage'] = usage_percent
            
            # Update usage history
            self.stats['usage_history'].append(usage_percent)
            if len(self.stats['usage_history']) > 100:  # Keep last 100 readings
                self.stats['usage_history'] = self.stats['usage_history'][-100:]
            
            # Calculate average usage
            if self.stats['usage_history']:
                self.stats['avg_usage'] = sum(self.stats['usage_history']) / len(self.stats['usage_history'])
            
            # Log warnings if usage is high
            if usage_percent > 80:
                logger.warning(f"High connection pool usage: {usage_percent:.1f}% ({active_connections}/{max_connections})")
            elif usage_percent > 60:
                logger.info(f"Moderate connection pool usage: {usage_percent:.1f}% ({active_connections}/{max_connections})")
            
        except Exception as e:
            # Only log error if it's not the application context issue
            if "Working outside of application context" not in str(e):
                logger.error(f"Error collecting pool stats: {e}")
            else:
                logger.debug("Skipping pool stats collection - no application context")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current connection pool statistics"""
        return self.stats.copy()
    
    def get_pool_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for connection pool optimization"""
        recommendations = []
        
        usage_percent = self.stats['connection_usage_percent']
        avg_usage = self.stats['avg_usage']
        peak_usage = self.stats['peak_usage']
        
        # High usage recommendations
        if usage_percent > 80:
            recommendations.append({
                'type': 'warning',
                'message': f'Connection pool usage is high ({usage_percent:.1f}%). Consider increasing pool size.',
                'action': 'Increase pool_size in SQLALCHEMY_ENGINE_OPTIONS'
            })
        
        if peak_usage > 90:
            recommendations.append({
                'type': 'critical',
                'message': f'Peak connection pool usage was very high ({peak_usage:.1f}%). Pool may be undersized.',
                'action': 'Increase both pool_size and max_overflow'
            })
        
        # Low usage recommendations
        if avg_usage < 30 and usage_percent < 40:
            recommendations.append({
                'type': 'info',
                'message': f'Connection pool usage is low ({usage_percent:.1f}%). Pool may be oversized.',
                'action': 'Consider reducing pool_size to save resources'
            })
        
        # Overflow usage recommendations
        if self.stats['overflow_connections'] > 0:
            recommendations.append({
                'type': 'info',
                'message': f'Using {self.stats["overflow_connections"]} overflow connections. Consider increasing base pool size.',
                'action': 'Increase pool_size to reduce overflow usage'
            })
        
        return {
            'recommendations': recommendations,
            'current_usage': usage_percent,
            'avg_usage': avg_usage,
            'peak_usage': peak_usage,
            'total_connections': self.stats['total_connections'],
            'active_connections': self.stats['active_connections']
        }
    
    def optimize_pool_settings(self) -> Dict[str, Any]:
        """Suggest optimal pool settings based on current usage"""
        avg_usage = self.stats['avg_usage']
        peak_usage = self.stats['peak_usage']
        current_pool_size = self.stats['max_connections']
        
        # Calculate suggested pool size based on usage patterns
        if peak_usage > 80:
            suggested_pool_size = int(current_pool_size * 1.5)  # Increase by 50%
            suggested_overflow = int(suggested_pool_size * 0.5)  # 50% overflow
        elif avg_usage > 60:
            suggested_pool_size = int(current_pool_size * 1.2)  # Increase by 20%
            suggested_overflow = int(suggested_pool_size * 0.4)  # 40% overflow
        elif avg_usage < 30:
            suggested_pool_size = int(current_pool_size * 0.8)  # Decrease by 20%
            suggested_overflow = int(suggested_pool_size * 0.3)  # 30% overflow
        else:
            suggested_pool_size = current_pool_size
            suggested_overflow = int(suggested_pool_size * 0.4)  # 40% overflow
        
        return {
            'current_settings': {
                'pool_size': current_pool_size - int(current_pool_size * 0.4),  # Approximate base size
                'max_overflow': int(current_pool_size * 0.4)  # Approximate overflow
            },
            'suggested_settings': {
                'pool_size': suggested_pool_size - suggested_overflow,
                'max_overflow': suggested_overflow,
                'total_connections': suggested_pool_size
            },
            'reasoning': {
                'avg_usage': avg_usage,
                'peak_usage': peak_usage,
                'current_total': current_pool_size,
                'suggested_total': suggested_pool_size
            }
        }

# Global connection pool monitor instance
pool_monitor = ConnectionPoolMonitor()

def start_connection_pool_monitoring(interval: int = 10):
    """Start the global connection pool monitoring"""
    pool_monitor.start_monitoring(interval)

def stop_connection_pool_monitoring():
    """Stop the global connection pool monitoring"""
    pool_monitor.stop_monitoring()

def get_connection_pool_stats() -> Dict[str, Any]:
    """Get current connection pool statistics"""
    return pool_monitor.get_pool_stats()

def get_connection_pool_recommendations() -> Dict[str, Any]:
    """Get connection pool optimization recommendations"""
    return pool_monitor.get_pool_recommendations()

def get_optimal_pool_settings() -> Dict[str, Any]:
    """Get suggested optimal pool settings"""
    return pool_monitor.optimize_pool_settings()
