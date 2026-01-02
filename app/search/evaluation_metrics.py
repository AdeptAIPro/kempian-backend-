"""
Evaluation Metrics and A/B Testing Framework
Production-grade metrics tracking and A/B testing for ranking models.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from sklearn.metrics import ndcg_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Evaluation metrics"""
    precision_at_5: float
    precision_at_10: float
    ndcg_at_10: float
    mrr: float
    recall_at_100: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    timestamp: datetime


class MetricsEvaluator:
    """Evaluate ranking model performance"""
    
    def __init__(self):
        self.metrics_history: List[EvaluationMetrics] = []
    
    def evaluate(
        self,
        y_true: List[int],
        y_pred: List[float],
        latencies: Optional[List[float]] = None
    ) -> EvaluationMetrics:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels (1 = relevant, 0 = not relevant)
            y_pred: Predicted scores
            latencies: Query latencies in seconds
        """
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        # Sort by predicted score
        sorted_indices = np.argsort(y_pred_array)[::-1]
        sorted_labels = y_true_array[sorted_indices]
        
        # Precision@5 and Precision@10
        precision_5 = np.mean(sorted_labels[:5]) if len(sorted_labels) >= 5 else np.mean(sorted_labels)
        precision_10 = np.mean(sorted_labels[:10]) if len(sorted_labels) >= 10 else np.mean(sorted_labels)
        
        # nDCG@10
        if SKLEARN_AVAILABLE:
            ndcg_10 = ndcg_score(
                y_true_array.reshape(-1, 1),
                y_pred_array.reshape(-1, 1),
                k=10
            )
        else:
            ndcg_10 = self._calculate_ndcg(sorted_labels, k=10)
        
        # MRR
        mrr = self._calculate_mrr(sorted_labels)
        
        # Recall@100
        recall_100 = np.sum(sorted_labels[:100]) / np.sum(y_true_array) if np.sum(y_true_array) > 0 else 0.0
        
        # Latency metrics
        if latencies:
            latency_p50 = np.percentile(latencies, 50)
            latency_p95 = np.percentile(latencies, 95)
            latency_p99 = np.percentile(latencies, 99)
        else:
            latency_p50 = latency_p95 = latency_p99 = 0.0
        
        metrics = EvaluationMetrics(
            precision_at_5=float(precision_5),
            precision_at_10=float(precision_10),
            ndcg_at_10=float(ndcg_10),
            mrr=float(mrr),
            recall_at_100=float(recall_100),
            latency_p50=float(latency_p50),
            latency_p95=float(latency_p95),
            latency_p99=float(latency_p99),
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_ndcg(self, sorted_labels: np.ndarray, k: int = 10) -> float:
        """Calculate nDCG manually"""
        # DCG
        dcg = 0.0
        for i in range(min(k, len(sorted_labels))):
            relevance = sorted_labels[i]
            dcg += relevance / np.log2(i + 2)
        
        # IDCG (ideal DCG)
        ideal_labels = np.sort(sorted_labels)[::-1]
        idcg = 0.0
        for i in range(min(k, len(ideal_labels))):
            relevance = ideal_labels[i]
            idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, sorted_labels: np.ndarray) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, label in enumerate(sorted_labels):
            if label == 1:
                return 1.0 / (i + 1)
        return 0.0
    
    def compare_models(
        self,
        baseline_metrics: EvaluationMetrics,
        new_metrics: EvaluationMetrics
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        comparison = {
            'precision@5_change': new_metrics.precision_at_5 - baseline_metrics.precision_at_5,
            'precision@10_change': new_metrics.precision_at_10 - baseline_metrics.precision_at_10,
            'ndcg@10_change': new_metrics.ndcg_at_10 - baseline_metrics.ndcg_at_10,
            'mrr_change': new_metrics.mrr - baseline_metrics.mrr,
            'latency_p95_change': new_metrics.latency_p95 - baseline_metrics.latency_p95,
            'improvement': {
                'precision@5': new_metrics.precision_at_5 > baseline_metrics.precision_at_5,
                'ndcg@10': new_metrics.ndcg_at_10 > baseline_metrics.ndcg_at_10,
                'latency': new_metrics.latency_p95 <= baseline_metrics.latency_p95 * 1.2  # Allow 20% increase
            }
        }
        
        return comparison


class ABTestingFramework:
    """A/B testing framework for ranking models"""
    
    def __init__(self):
        self.tests: Dict[str, Dict] = {}
        self.results: Dict[str, List] = defaultdict(list)
    
    def create_test(
        self,
        test_id: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.5,
        duration_days: int = 14
    ):
        """Create A/B test"""
        self.tests[test_id] = {
            'test_id': test_id,
            'control_model': control_model,
            'treatment_model': treatment_model,
            'traffic_split': traffic_split,
            'duration_days': duration_days,
            'start_date': datetime.now(),
            'end_date': datetime.now() + timedelta(days=duration_days),
            'status': 'active'
        }
    
    def record_result(
        self,
        test_id: str,
        variant: str,  # 'control' or 'treatment'
        metrics: EvaluationMetrics,
        user_id: str
    ):
        """Record A/B test result"""
        if test_id not in self.tests:
            logger.warning(f"Test {test_id} not found")
            return
        
        self.results[test_id].append({
            'variant': variant,
            'metrics': metrics,
            'user_id': user_id,
            'timestamp': datetime.now()
        })
    
    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        results = self.results[test_id]
        
        if len(results) < 100:  # Minimum sample size
            return {'status': 'insufficient_data', 'sample_size': len(results)}
        
        # Separate control and treatment
        control_results = [r for r in results if r['variant'] == 'control']
        treatment_results = [r for r in results if r['variant'] == 'treatment']
        
        if not control_results or not treatment_results:
            return {'status': 'insufficient_data'}
        
        # Calculate average metrics
        control_avg = self._average_metrics([r['metrics'] for r in control_results])
        treatment_avg = self._average_metrics([r['metrics'] for r in treatment_results])
        
        # Statistical significance (simplified)
        significance = self._calculate_significance(control_results, treatment_results)
        
        # Decision
        decision = self._make_decision(control_avg, treatment_avg, significance)
        
        return {
            'test_id': test_id,
            'control_metrics': control_avg,
            'treatment_metrics': treatment_avg,
            'improvement': {
                'precision@5': treatment_avg['precision_at_5'] - control_avg['precision_at_5'],
                'ndcg@10': treatment_avg['ndcg_at_10'] - control_avg['ndcg_at_10']
            },
            'significance': significance,
            'decision': decision,
            'sample_size': {
                'control': len(control_results),
                'treatment': len(treatment_results)
            }
        }
    
    def _average_metrics(self, metrics_list: List[EvaluationMetrics]) -> Dict[str, float]:
        """Calculate average metrics"""
        return {
            'precision_at_5': np.mean([m.precision_at_5 for m in metrics_list]),
            'precision_at_10': np.mean([m.precision_at_10 for m in metrics_list]),
            'ndcg_at_10': np.mean([m.ndcg_at_10 for m in metrics_list]),
            'mrr': np.mean([m.mrr for m in metrics_list]),
            'latency_p95': np.mean([m.latency_p95 for m in metrics_list])
        }
    
    def _calculate_significance(
        self,
        control_results: List[Dict],
        treatment_results: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate statistical significance (simplified)"""
        # Extract precision@5 values
        control_precisions = [r['metrics'].precision_at_5 for r in control_results]
        treatment_precisions = [r['metrics'].precision_at_5 for r in treatment_results]
        
        # Simple t-test (would use scipy.stats in production)
        control_mean = np.mean(control_precisions)
        treatment_mean = np.mean(treatment_precisions)
        
        control_std = np.std(control_precisions)
        treatment_std = np.std(treatment_precisions)
        
        # Simplified significance check
        diff = treatment_mean - control_mean
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        se = pooled_std / np.sqrt(len(control_results) + len(treatment_results))
        
        z_score = diff / se if se > 0 else 0
        is_significant = abs(z_score) > 1.96  # 95% confidence
        
        return {
            'is_significant': is_significant,
            'z_score': float(z_score),
            'p_value': 0.05 if is_significant else 0.5  # Simplified
        }
    
    def _make_decision(
        self,
        control_avg: Dict[str, float],
        treatment_avg: Dict[str, float],
        significance: Dict[str, Any]
    ) -> str:
        """Make decision on A/B test"""
        improvement = treatment_avg['precision_at_5'] - control_avg['precision_at_5']
        
        if not significance['is_significant']:
            return 'no_significant_difference'
        
        if improvement > 0.05:  # 5% improvement
            return 'deploy_treatment'
        elif improvement < -0.05:  # 5% degradation
            return 'keep_control'
        else:
            return 'marginal_improvement'


# Global instances
_metrics_evaluator = None
_ab_testing = None

def get_metrics_evaluator() -> MetricsEvaluator:
    """Get or create global metrics evaluator"""
    global _metrics_evaluator
    if _metrics_evaluator is None:
        _metrics_evaluator = MetricsEvaluator()
    return _metrics_evaluator

def get_ab_testing() -> ABTestingFramework:
    """Get or create global A/B testing framework"""
    global _ab_testing
    if _ab_testing is None:
        _ab_testing = ABTestingFramework()
    return _ab_testing

