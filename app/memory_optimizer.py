"""
Memory Optimization for High-Scale Operations
Optimizes memory usage for 2000+ concurrent users
"""
import gc
import psutil
import threading
import time
from typing import Dict, Any, List
from app.simple_logger import get_logger

logger = get_logger("memory_optimizer")

class MemoryOptimizer:
    """Memory optimization system for high-scale operations"""
    
    def __init__(self):
        self.optimization_active = False
        self.optimizer_thread = None
        self.memory_stats = {
            'total_memory': 0,
            'available_memory': 0,
            'used_memory': 0,
            'memory_percent': 0,
            'process_memory': 0,
            'memory_pressure': 'low',
            'optimization_count': 0,
            'last_optimization': 0
        }
        
        # Memory optimization thresholds
        self.thresholds = {
            'high_memory': 85.0,      # Trigger optimization at 85%
            'critical_memory': 95.0,   # Critical optimization at 95%
            'optimization_interval': 300,  # 5 minutes between optimizations
            'gc_threshold': 80.0       # Trigger GC at 80%
        }
        
        logger.info("Memory optimizer initialized")
    
    def start_optimization(self, interval: int = 30):
        """Start memory optimization monitoring"""
        if self.optimization_active:
            logger.warning("Memory optimization already active")
            return
        
        self.optimization_active = True
        self.optimizer_thread = threading.Thread(
            target=self._optimization_loop,
            args=(interval,),
            daemon=True
        )
        self.optimizer_thread.start()
        logger.info(f"Memory optimization started (interval: {interval}s)")
    
    def stop_optimization(self):
        """Stop memory optimization"""
        self.optimization_active = False
        if self.optimizer_thread:
            self.optimizer_thread.join(timeout=5)
        logger.info("Memory optimization stopped")
    
    def _optimization_loop(self, interval: int):
        """Main optimization loop"""
        while self.optimization_active:
            try:
                self._check_memory_usage()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in memory optimization: {e}")
                time.sleep(interval)
    
    def _check_memory_usage(self):
        """Check memory usage and trigger optimizations"""
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            self.memory_stats.update({
                'total_memory': memory.total,
                'available_memory': memory.available,
                'used_memory': memory.used,
                'memory_percent': memory.percent,
                'process_memory': process.memory_info().rss,
                'timestamp': time.time()
            })
            
            # Determine memory pressure
            if memory.percent >= self.thresholds['critical_memory']:
                self.memory_stats['memory_pressure'] = 'critical'
                self._critical_memory_optimization()
            elif memory.percent >= self.thresholds['high_memory']:
                self.memory_stats['memory_pressure'] = 'high'
                self._high_memory_optimization()
            elif memory.percent >= self.thresholds['gc_threshold']:
                self.memory_stats['memory_pressure'] = 'moderate'
                self._moderate_memory_optimization()
            else:
                self.memory_stats['memory_pressure'] = 'low'
            
            # Log memory status
            if memory.percent > 80:
                logger.warning(f"Memory usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB used)")
            
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
    
    def _moderate_memory_optimization(self):
        """Moderate memory optimization (80-85%)"""
        try:
            # Trigger garbage collection
            collected = gc.collect()
            logger.info(f"Moderate memory optimization: GC collected {collected} objects")
            
            # Update stats
            self.memory_stats['optimization_count'] += 1
            self.memory_stats['last_optimization'] = time.time()
            
        except Exception as e:
            logger.error(f"Error in moderate memory optimization: {e}")
    
    def _high_memory_optimization(self):
        """High memory optimization (85-95%)"""
        try:
            # Check if enough time has passed since last optimization
            if time.time() - self.memory_stats['last_optimization'] < self.thresholds['optimization_interval']:
                return
            
            # Aggressive garbage collection
            collected = 0
            for generation in range(3):
                collected += gc.collect()
            
            # Clear Python caches if available
            try:
                import sys
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
            except:
                pass
            
            logger.warning(f"High memory optimization: GC collected {collected} objects")
            
            # Update stats
            self.memory_stats['optimization_count'] += 1
            self.memory_stats['last_optimization'] = time.time()
            
        except Exception as e:
            logger.error(f"Error in high memory optimization: {e}")
    
    def _critical_memory_optimization(self):
        """Critical memory optimization (>95%)"""
        try:
            # Check if enough time has passed since last optimization
            if time.time() - self.memory_stats['last_optimization'] < 60:  # 1 minute for critical
                return
            
            # Multiple rounds of garbage collection
            total_collected = 0
            for _ in range(3):
                for generation in range(3):
                    total_collected += gc.collect()
            
            # Clear all possible caches
            try:
                import sys
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
                if hasattr(sys, 'intern'):
                    sys.intern.__dict__.clear()
            except:
                pass
            
            # Force garbage collection of cycles
            gc.set_debug(gc.DEBUG_SAVEALL)
            gc.collect()
            gc.set_debug(0)
            
            logger.critical(f"Critical memory optimization: GC collected {total_collected} objects")
            
            # Update stats
            self.memory_stats['optimization_count'] += 1
            self.memory_stats['last_optimization'] = time.time()
            
        except Exception as e:
            logger.error(f"Error in critical memory optimization: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        return self.memory_stats.copy()
    
    def get_memory_recommendations(self) -> List[Dict[str, Any]]:
        """Get memory optimization recommendations"""
        recommendations = []
        memory_percent = self.memory_stats['memory_percent']
        
        if memory_percent >= self.thresholds['critical_memory']:
            recommendations.append({
                'type': 'critical',
                'message': f'Critical memory usage: {memory_percent:.1f}%. Immediate action required.',
                'actions': [
                    'Restart application if possible',
                    'Check for memory leaks',
                    'Consider increasing system memory',
                    'Reduce concurrent user limit'
                ]
            })
        elif memory_percent >= self.thresholds['high_memory']:
            recommendations.append({
                'type': 'warning',
                'message': f'High memory usage: {memory_percent:.1f}%. Optimization recommended.',
                'actions': [
                    'Monitor memory usage closely',
                    'Consider reducing cache sizes',
                    'Check for memory leaks',
                    'Optimize data structures'
                ]
            })
        elif memory_percent >= self.thresholds['gc_threshold']:
            recommendations.append({
                'type': 'info',
                'message': f'Moderate memory usage: {memory_percent:.1f}%. Monitoring recommended.',
                'actions': [
                    'Continue monitoring',
                    'Consider proactive optimization',
                    'Check memory trends'
                ]
            })
        else:
            recommendations.append({
                'type': 'success',
                'message': f'Good memory usage: {memory_percent:.1f}%. System is healthy.',
                'actions': [
                    'Continue current configuration',
                    'Monitor for trends'
                ]
            })
        
        return recommendations
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate memory optimization"""
        try:
            start_time = time.time()
            
            # Multiple rounds of garbage collection
            total_collected = 0
            for _ in range(5):
                for generation in range(3):
                    total_collected += gc.collect()
            
            # Clear caches
            try:
                import sys
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
            except:
                pass
            
            optimization_time = time.time() - start_time
            
            # Update stats
            self.memory_stats['optimization_count'] += 1
            self.memory_stats['last_optimization'] = time.time()
            
            logger.info(f"Forced memory optimization completed in {optimization_time:.2f}s, collected {total_collected} objects")
            
            return {
                'success': True,
                'objects_collected': total_collected,
                'optimization_time': optimization_time,
                'memory_before': self.memory_stats['memory_percent'],
                'memory_after': psutil.virtual_memory().percent
            }
            
        except Exception as e:
            logger.error(f"Error in forced optimization: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()

def start_memory_optimization(interval: int = 30):
    """Start the global memory optimization"""
    memory_optimizer.start_optimization(interval)

def stop_memory_optimization():
    """Stop the global memory optimization"""
    memory_optimizer.stop_optimization()

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics"""
    return memory_optimizer.get_memory_stats()

def get_memory_recommendations() -> List[Dict[str, Any]]:
    """Get memory optimization recommendations"""
    return memory_optimizer.get_memory_recommendations()

def force_memory_optimization() -> Dict[str, Any]:
    """Force immediate memory optimization"""
    return memory_optimizer.force_optimization()
