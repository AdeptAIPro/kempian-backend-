"""
Async Processing System for High-Scale Operations
Handles CPU-intensive tasks asynchronously to prevent blocking
"""
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union
from queue import Queue, Empty
from dataclasses import dataclass
from app.simple_logger import get_logger

logger = get_logger("async_processor")

@dataclass
class AsyncTask:
    """Represents an async task"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    created_at: float = None
    status: str = 'pending'  # pending, running, completed, failed
    result: Any = None
    error: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class AsyncProcessor:
    """High-performance async task processor"""
    
    def __init__(self, max_workers: int = 50, max_processes: int = 16):
        self.max_workers = max_workers
        self.max_processes = max_processes
        
        # Thread pool for I/O bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Process pool for CPU bound tasks
        self.process_pool = ProcessPoolExecutor(max_workers=max_processes)
        
        # Task queue with priority
        self.task_queue = Queue()
        self.running_tasks = {}
        self.completed_tasks = {}
        
        # Background processing
        self.processing_active = False
        self.processing_thread = None
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_processing_time': 0,
            'active_tasks': 0
        }
        
        logger.info(f"AsyncProcessor initialized with {max_workers} threads and {max_processes} processes")
    
    def start_processing(self):
        """Start background task processing"""
        if self.processing_active:
            logger.warning("Processing already active")
            return
        
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Background processing started")
    
    def stop_processing(self):
        """Stop background task processing"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Background processing stopped")
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.processing_active:
            try:
                # Get next task from queue
                try:
                    task = self.task_queue.get(timeout=1)
                except Empty:
                    continue
                
                # Process task
                self._process_task(task)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1)
    
    def _process_task(self, task: AsyncTask):
        """Process a single task"""
        task.status = 'running'
        self.running_tasks[task.task_id] = task
        self.stats['active_tasks'] = len(self.running_tasks)
        
        start_time = time.time()
        
        try:
            # Determine if task is CPU or I/O bound
            if self._is_cpu_bound_task(task.function):
                # Use process pool for CPU bound tasks
                future = self.process_pool.submit(task.function, *task.args, **task.kwargs)
            else:
                # Use thread pool for I/O bound tasks
                future = self.thread_pool.submit(task.function, *task.args, **task.kwargs)
            
            # Wait for completion
            task.result = future.result(timeout=300)  # 5 minute timeout
            task.status = 'completed'
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=True)
            
            logger.debug(f"Task {task.task_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            task.status = 'failed'
            task.error = str(e)
            self._update_stats(time.time() - start_time, success=False)
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Move to completed tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            self.stats['active_tasks'] = len(self.running_tasks)
    
    def _is_cpu_bound_task(self, func: Callable) -> bool:
        """Determine if a task is CPU bound"""
        # Simple heuristic based on function name
        cpu_bound_keywords = [
            'embedding', 'search', 'match', 'calculate', 'process',
            'parse', 'analyze', 'compute', 'transform', 'encode'
        ]
        
        func_name = func.__name__.lower()
        return any(keyword in func_name for keyword in cpu_bound_keywords)
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update processing statistics"""
        self.stats['total_tasks'] += 1
        
        if success:
            self.stats['completed_tasks'] += 1
        else:
            self.stats['failed_tasks'] += 1
        
        # Update average processing time
        total_completed = self.stats['completed_tasks'] + self.stats['failed_tasks']
        if total_completed > 0:
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (total_completed - 1) + processing_time) / 
                total_completed
            )
    
    def submit_task(self, task_id: str, function: Callable, *args, priority: int = 0, **kwargs) -> str:
        """Submit a task for async processing"""
        task = AsyncTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        self.task_queue.put(task)
        logger.debug(f"Task {task_id} submitted for processing")
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check running tasks
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status,
                'created_at': task.created_at,
                'running_time': time.time() - task.created_at
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status,
                'created_at': task.created_at,
                'completed_at': time.time(),
                'processing_time': time.time() - task.created_at,
                'result': task.result if task.status == 'completed' else None,
                'error': task.error if task.status == 'failed' else None
            }
        
        return None
    
    def get_task_result(self, task_id: str, timeout: float = 30) -> Any:
        """Get result of a completed task"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            
            if status is None:
                raise ValueError(f"Task {task_id} not found")
            
            if status['status'] == 'completed':
                return status['result']
            elif status['status'] == 'failed':
                raise RuntimeError(f"Task {task_id} failed: {status.get('error', 'Unknown error')}")
            
            time.sleep(0.1)  # Wait 100ms before checking again
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            'queue_size': self.task_queue.qsize(),
            'processing_active': self.processing_active
        }
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        old_tasks = [
            task_id for task_id, task in self.completed_tasks.items()
            if task.created_at < cutoff_time
        ]
        
        for task_id in old_tasks:
            del self.completed_tasks[task_id]
        
        logger.info(f"Cleaned up {len(old_tasks)} old tasks")
        return len(old_tasks)

# Global async processor instance (optimized for 2000+ users)
async_processor = AsyncProcessor(max_workers=50, max_processes=16)

def start_async_processing():
    """Start the global async processor"""
    async_processor.start_processing()

def stop_async_processing():
    """Stop the global async processor"""
    async_processor.stop_processing()

def submit_async_task(task_id: str, function: Callable, *args, priority: int = 0, **kwargs) -> str:
    """Submit a task for async processing"""
    return async_processor.submit_task(task_id, function, *args, priority=priority, **kwargs)

def get_async_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get status of an async task"""
    return async_processor.get_task_status(task_id)

def get_async_task_result(task_id: str, timeout: float = 30) -> Any:
    """Get result of an async task"""
    return async_processor.get_task_result(task_id, timeout)

def get_async_stats() -> Dict[str, Any]:
    """Get async processing statistics"""
    return async_processor.get_stats()

# Specific async functions for common operations
def async_resume_processing(file_data: bytes, user_id: int) -> str:
    """Submit resume processing task"""
    task_id = f"resume_processing_{user_id}_{int(time.time())}"
    return submit_async_task(
        task_id=task_id,
        function=_process_resume_sync,
        file_data=file_data,
        user_id=user_id,
        priority=1  # High priority
    )

def async_search_processing(query: str, filters: Dict = None) -> str:
    """Submit search processing task"""
    task_id = f"search_processing_{hash(query)}_{int(time.time())}"
    return submit_async_task(
        task_id=task_id,
        function=_process_search_sync,
        query=query,
        filters=filters or {},
        priority=2  # Medium priority
    )

def async_analytics_processing(user_id: int, tenant_id: int) -> str:
    """Submit analytics processing task"""
    task_id = f"analytics_processing_{user_id}_{int(time.time())}"
    return submit_async_task(
        task_id=task_id,
        function=_process_analytics_sync,
        user_id=user_id,
        tenant_id=tenant_id,
        priority=3  # Low priority
    )

# Sync functions that will be called asynchronously
def _process_resume_sync(file_data: bytes, user_id: int) -> Dict[str, Any]:
    """Process resume synchronously (called by async processor)"""
    try:
        # Import here to avoid circular imports
        from app.talent.routes import process_resume_file
        
        result = process_resume_file(file_data, user_id)
        return {
            'success': True,
            'result': result,
            'processed_at': time.time()
        }
    except Exception as e:
        logger.error(f"Resume processing failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'processed_at': time.time()
        }

def _process_search_sync(query: str, filters: Dict) -> Dict[str, Any]:
    """Process search synchronously (called by async processor)"""
    try:
        # Import here to avoid circular imports
        from app.search.service import semantic_match
        
        result = semantic_match(query)
        return {
            'success': True,
            'result': result,
            'processed_at': time.time()
        }
    except Exception as e:
        logger.error(f"Search processing failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'processed_at': time.time()
        }

def _process_analytics_sync(user_id: int, tenant_id: int) -> Dict[str, Any]:
    """Process analytics synchronously (called by async processor)"""
    try:
        # Import here to avoid circular imports
        from app.analytics.routes import _generate_kpi_data
        
        result = _generate_kpi_data(user_id, tenant_id)
        return {
            'success': True,
            'result': result,
            'processed_at': time.time()
        }
    except Exception as e:
        logger.error(f"Analytics processing failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'processed_at': time.time()
        }
