"""
Benchmark runner for evaluating matchmaking system performance.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..pipelines.matcher import CandidateJobMatcher, match_candidates
from .metrics import calculate_metrics, calculate_map

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run benchmarks on the matchmaking system."""
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.matcher = CandidateJobMatcher()
    
    def run_benchmark(
        self,
        test_cases: List[Dict[str, Any]],
        ground_truth: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Run benchmark on test cases.
        
        Args:
            test_cases: List of test cases, each containing:
                - job_description: Job description
                - candidates: List of candidate dictionaries
                - expected_top_candidates: List of expected top candidate IDs (optional)
            ground_truth: Dictionary mapping job_id to list of relevant candidate IDs
        
        Returns:
            Benchmark results dictionary
        """
        results = {
            'total_cases': len(test_cases),
            'metrics': [],
            'average_metrics': {},
            'execution_times': []
        }
        
        all_predictions = []
        all_ground_truth = []
        
        for i, test_case in enumerate(test_cases):
            try:
                job_description = test_case.get('job_description', '')
                candidates = test_case.get('candidates', [])
                expected_top = test_case.get('expected_top_candidates', [])
                
                # Run matching
                start_time = time.time()
                match_results = self.matcher.match_candidates(
                    job_description,
                    candidates
                )
                execution_time = time.time() - start_time
                
                results['execution_times'].append(execution_time)
                
                # Extract predicted candidate IDs
                predicted_ids = [r['candidate_id'] for r in match_results]
                all_predictions.append(predicted_ids)
                
                # Calculate metrics if ground truth available
                if expected_top:
                    all_ground_truth.append(expected_top)
                    metrics = calculate_metrics(predicted_ids, expected_top, k=10)
                    metrics['test_case'] = i
                    results['metrics'].append(metrics)
                
            except Exception as e:
                logger.error(f"Error in benchmark test case {i}: {e}")
                continue
        
        # Calculate average metrics
        if results['metrics']:
            avg_metrics = {}
            for metric_name in ['precision', 'recall', 'f1', 'ndcg']:
                values = [m[metric_name] for m in results['metrics'] if metric_name in m]
                if values:
                    avg_metrics[metric_name] = sum(values) / len(values)
            results['average_metrics'] = avg_metrics
        
        # Calculate MAP if ground truth available
        if all_ground_truth and len(all_ground_truth) == len(all_predictions):
            results['map'] = calculate_map(all_predictions, all_ground_truth)
        
        # Calculate average execution time
        if results['execution_times']:
            results['avg_execution_time'] = sum(results['execution_times']) / len(results['execution_times'])
            results['min_execution_time'] = min(results['execution_times'])
            results['max_execution_time'] = max(results['execution_times'])
        
        return results
    
    def run_single_test(
        self,
        job_description: str,
        candidates: List[Dict[str, Any]],
        expected_top: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Args:
            job_description: Job description text
            candidates: List of candidate dictionaries
            expected_top: Expected top candidate IDs (optional)
        
        Returns:
            Test results dictionary
        """
        start_time = time.time()
        match_results = self.matcher.match_candidates(job_description, candidates)
        execution_time = time.time() - start_time
        
        predicted_ids = [r['candidate_id'] for r in match_results]
        
        result = {
            'match_results': match_results,
            'predicted_ids': predicted_ids,
            'execution_time': execution_time
        }
        
        if expected_top:
            metrics = calculate_metrics(predicted_ids, expected_top, k=10)
            result['metrics'] = metrics
        
        return result

