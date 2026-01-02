import numpy as np
from typing import List, Dict, Callable, Any, Generator
from app.simple_logger import get_logger
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = get_logger("search")

class BatchProcessor:
    """
    Efficient batch processing for embeddings and ML operations
    """
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.stats = {
            'total_processed': 0,
            'total_time': 0,
            'avg_throughput': 0
        }
        self._lock = threading.Lock()
    
    def process_embeddings_batch(
        self, 
        texts: List[str], 
        model, 
        batch_size: int = None
    ) -> np.ndarray:
        """
        Process embeddings in batches for better GPU utilization
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        start_time = time.time()
        
        if len(texts) <= batch_size:
            # Single batch
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        else:
            # Multiple batches
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
        
        # Update stats
        end_time = time.time()
        processing_time = end_time - start_time
        
        with self._lock:
            self.stats['total_processed'] += len(texts)
            self.stats['total_time'] += processing_time
            self.stats['avg_throughput'] = self.stats['total_processed'] / self.stats['total_time']
        
        logger.debug(f"Processed {len(texts)} embeddings in {processing_time:.3f}s")
        
        return embeddings
    
    def process_candidates_parallel(
        self,
        candidates: List[Dict],
        processing_func: Callable[[Dict], Any],
        max_workers: int = None
    ) -> List[Any]:
        """
        Process candidates in parallel
        """
        if max_workers is None:
            max_workers = self.max_workers
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_candidate = {
                executor.submit(processing_func, candidate): candidate 
                for candidate in candidates
            }
            
            # Collect results
            for future in as_completed(future_to_candidate):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    candidate = future_to_candidate[future]
                    logger.error(f"Error processing candidate {candidate.get('email', 'unknown')}: {e}")
                    results.append(None)
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        end_time = time.time()
        logger.info(f"Processed {len(candidates)} candidates in {end_time - start_time:.3f}s using {max_workers} workers")
        
        return results
    
    def batch_similarity_search(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        batch_size: int = None
    ) -> np.ndarray:
        """
        Compute similarities in batches to manage memory
        """
        if batch_size is None:
            batch_size = self.batch_size * 10  # Larger batches for similarity computation
        
        n_candidates = candidate_embeddings.shape[0]
        
        if n_candidates <= batch_size:
            # Single batch
            from sentence_transformers.util import cos_sim
            similarities = cos_sim(query_embedding, candidate_embeddings)[0]
            return similarities.cpu().numpy()
        
        # Multiple batches
        all_similarities = []
        
        for i in range(0, n_candidates, batch_size):
            batch_embeddings = candidate_embeddings[i:i + batch_size]
            
            from sentence_transformers.util import cos_sim
            batch_similarities = cos_sim(query_embedding, batch_embeddings)[0]
            all_similarities.append(batch_similarities.cpu().numpy())
        
        return np.concatenate(all_similarities)
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        with self._lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics"""
        with self._lock:
            self.stats = {
                'total_processed': 0,
                'total_time': 0,
                'avg_throughput': 0
            }


class StreamProcessor:
    """
    Process data in streaming fashion for real-time applications
    """
    
    def __init__(self, buffer_size: int = 100, flush_interval: float = 5.0):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.last_flush = time.time()
        self._lock = threading.Lock()
    
    def add_item(self, item: Any) -> bool:
        """
        Add item to buffer. Returns True if buffer was flushed.
        """
        with self._lock:
            self.buffer.append(item)
            
            # Check if we need to flush
            should_flush = (
                len(self.buffer) >= self.buffer_size or
                time.time() - self.last_flush >= self.flush_interval
            )
            
            if should_flush:
                self._flush_buffer()
                return True
            
            return False
    
    def _flush_buffer(self):
        """Flush the current buffer"""
        if self.buffer:
            logger.debug(f"Flushing buffer with {len(self.buffer)} items")
            self.buffer.clear()
            self.last_flush = time.time()
    
    def force_flush(self):
        """Force flush the buffer"""
        with self._lock:
            self._flush_buffer()
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        with self._lock:
            return len(self.buffer)


def chunk_list(lst: List[Any], chunk_size: int) -> Generator[List[Any], None, None]:
    """
    Split list into chunks of specified size
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def optimize_batch_size(
    processing_func: Callable,
    test_data: List[Any],
    max_batch_size: int = 128,
    target_time: float = 1.0
) -> int:
    """
    Automatically determine optimal batch size based on processing time
    """
    if not test_data:
        return 32  # Default
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16, 32, 64]
    if max_batch_size > 64:
        batch_sizes.extend([128, 256, 512])
    
    batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
    
    best_batch_size = 32
    best_throughput = 0
    
    for batch_size in batch_sizes:
        # Use subset of test data
        test_subset = test_data[:min(batch_size * 2, len(test_data))]
        
        try:
            start_time = time.time()
            
            # Process in batches
            for chunk in chunk_list(test_subset, batch_size):
                processing_func(chunk)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate throughput
            throughput = len(test_subset) / processing_time
            
            # Check if this batch size meets our target time per batch
            time_per_batch = (processing_time * batch_size) / len(test_subset)
            
            if time_per_batch <= target_time and throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
            
            logger.debug(f"Batch size {batch_size}: {throughput:.2f} items/sec, {time_per_batch:.3f}s per batch")
            
        except Exception as e:
            logger.warning(f"Error testing batch size {batch_size}: {e}")
            continue
    
    logger.info(f"Optimal batch size determined: {best_batch_size} (throughput: {best_throughput:.2f} items/sec)")
    return best_batch_size