"""
Performance Monitoring System for High-Scale Operations
Monitors system performance, resource usage, and provides health checks
"""
import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
from app.simple_logger import get_logger
from app.cache import cache_manager

logger = get_logger("monitoring")

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.start_time = time.time()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance thresholds (optimized for 2000+ users)
        self.thresholds = {
            'cpu_usage': 85.0,      # CPU usage percentage (increased from 80)
            'memory_usage': 90.0,    # Memory usage percentage (increased from 85)
            'response_time': 1.0,    # Response time in seconds (decreased from 2)
            'error_rate': 2.0,       # Error rate percentage (decreased from 5)
            'db_connections': 250,   # Database connections (increased from 100)
            'cache_hit_rate': 80.0   # Cache hit rate percentage (increased from 70)
        }
        
        # Alert system
        self.alerts = []
        self.alert_cooldown = 300  # 5 minutes between same alerts
        
        logger.info("Performance monitor initialized")
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                self._check_thresholds(metrics)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            # Database metrics (if available)
            db_connections = self._get_db_connection_count()
            
            # Cache metrics
            cache_stats = cache_manager.get_stats()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'uptime': time.time() - self.start_time,
                
                # System metrics
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available': memory.available,
                'memory_used': memory.used,
                'disk_usage': disk.percent,
                'disk_free': disk.free,
                
                # Process metrics
                'process_memory_mb': process_memory.rss / 1024 / 1024,
                'process_cpu_percent': process_cpu,
                'process_threads': process.num_threads(),
                'process_open_files': len(process.open_files()),
                
                # Database metrics
                'db_connections': db_connections,
                
                # Cache metrics
                'cache_hit_rate': cache_stats.get('redis_hit_rate', 0),
                'cache_memory_usage': cache_stats.get('redis_used_memory', 'N/A'),
                'cache_connected_clients': cache_stats.get('redis_connected_clients', 0),
                'l1_cache_size': cache_stats.get('l1_cache_size', 0),
                
                # Network metrics
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'network_packets_sent': network.packets_sent,
                'network_packets_recv': network.packets_recv,
                
                # Load metrics
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def _get_db_connection_count(self) -> int:
        """Get current database connection count"""
        try:
            # This would need to be implemented based on your database setup
            # For now, return a placeholder
            return 0
        except Exception:
            return 0
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts"""
        current_time = time.time()
        
        # CPU usage check
        if metrics.get('cpu_usage', 0) > self.thresholds['cpu_usage']:
            self._create_alert('high_cpu', f"High CPU usage: {metrics['cpu_usage']:.1f}%", current_time)
        
        # Memory usage check
        if metrics.get('memory_usage', 0) > self.thresholds['memory_usage']:
            self._create_alert('high_memory', f"High memory usage: {metrics['memory_usage']:.1f}%", current_time)
        
        # Cache hit rate check
        if metrics.get('cache_hit_rate', 0) < self.thresholds['cache_hit_rate']:
            self._create_alert('low_cache_hit_rate', f"Low cache hit rate: {metrics['cache_hit_rate']:.1f}%", current_time)
        
        # Database connections check
        if metrics.get('db_connections', 0) > self.thresholds['db_connections']:
            self._create_alert('high_db_connections', f"High DB connections: {metrics['db_connections']}", current_time)
    
    def _create_alert(self, alert_type: str, message: str, timestamp: float):
        """Create an alert if not in cooldown"""
        # Check if we already have a recent alert of this type
        recent_alerts = [
            alert for alert in self.alerts
            if alert['type'] == alert_type and 
            timestamp - alert['timestamp'] < self.alert_cooldown
        ]
        
        if not recent_alerts:
            alert = {
                'type': alert_type,
                'message': message,
                'timestamp': timestamp,
                'datetime': datetime.utcnow().isoformat()
            }
            self.alerts.append(alert)
            logger.warning(f"ALERT: {message}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        latest_metrics = self.metrics_history[-1]
        
        # Determine overall status
        status = 'healthy'
        if latest_metrics.get('cpu_usage', 0) > self.thresholds['cpu_usage']:
            status = 'warning'
        if latest_metrics.get('memory_usage', 0) > self.thresholds['memory_usage']:
            status = 'critical'
        
        return {
            'status': status,
            'timestamp': latest_metrics.get('timestamp'),
            'uptime_seconds': latest_metrics.get('uptime', 0),
            'cpu_usage': latest_metrics.get('cpu_usage', 0),
            'memory_usage': latest_metrics.get('memory_usage', 0),
            'db_connections': latest_metrics.get('db_connections', 0),
            'cache_hit_rate': latest_metrics.get('cache_hit_rate', 0),
            'active_alerts': len([a for a in self.alerts if time.time() - a['timestamp'] < 3600])
        }
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [
            m for m in self.metrics_history
            if m.get('timestamp') and 
            datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00')).timestamp() > cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No data available'}
        
        # Calculate averages
        cpu_values = [m.get('cpu_usage', 0) for m in recent_metrics if 'cpu_usage' in m]
        memory_values = [m.get('memory_usage', 0) for m in recent_metrics if 'memory_usage' in m]
        cache_values = [m.get('cache_hit_rate', 0) for m in recent_metrics if 'cache_hit_rate' in m]
        
        return {
            'period_hours': hours,
            'data_points': len(recent_metrics),
            'avg_cpu_usage': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            'max_cpu_usage': max(cpu_values) if cpu_values else 0,
            'avg_memory_usage': sum(memory_values) / len(memory_values) if memory_values else 0,
            'max_memory_usage': max(memory_values) if memory_values else 0,
            'avg_cache_hit_rate': sum(cache_values) / len(cache_values) if cache_values else 0,
            'min_cache_hit_rate': min(cache_values) if cache_values else 0,
            'alerts_count': len([a for a in self.alerts if a['timestamp'] > cutoff_time])
        }
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = time.time() - (hours * 3600)
        return [
            alert for alert in self.alerts
            if alert['timestamp'] > cutoff_time
        ]
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
        logger.info("All alerts cleared")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def start_performance_monitoring(interval: int = 30):
    """Start the global performance monitoring"""
    performance_monitor.start_monitoring(interval)

def stop_performance_monitoring():
    """Stop the global performance monitoring"""
    performance_monitor.stop_monitoring()

def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    return performance_monitor.get_current_status()

def get_performance_metrics(hours: int = 1) -> Dict[str, Any]:
    """Get performance metrics for the last N hours"""
    return performance_monitor.get_performance_summary(hours)

def get_recent_alerts(hours: int = 24) -> List[Dict[str, Any]]:
    """Get recent alerts"""
    return performance_monitor.get_alerts(hours)
