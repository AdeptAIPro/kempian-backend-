"""
Evaluation metrics for matchmaking system.
Provides precision, recall, NDCG, and other ranking metrics.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def calculate_precision(
    predicted: List[str],
    relevant: List[str],
    k: Optional[int] = None
) -> float:
    """
    Calculate precision@k.
    
    Args:
        predicted: List of predicted candidate IDs (ordered by score)
        relevant: List of relevant candidate IDs
        k: Top K to consider (None for all)
    
    Returns:
        Precision score (0-1)
    """
    if not predicted:
        return 0.0
    
    if k is not None:
        predicted = predicted[:k]
    
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    
    # Count relevant items in top k
    relevant_count = sum(1 for item in predicted if item in relevant_set)
    
    return relevant_count / len(predicted)


def calculate_recall(
    predicted: List[str],
    relevant: List[str],
    k: Optional[int] = None
) -> float:
    """
    Calculate recall@k.
    
    Args:
        predicted: List of predicted candidate IDs (ordered by score)
        relevant: List of relevant candidate IDs
        k: Top K to consider (None for all)
    
    Returns:
        Recall score (0-1)
    """
    if not relevant:
        return 1.0 if not predicted else 0.0
    
    if k is not None:
        predicted = predicted[:k]
    
    relevant_set = set(relevant)
    predicted_set = set(predicted)
    
    # Count relevant items found
    found_count = len(relevant_set & predicted_set)
    
    return found_count / len(relevant_set)


def calculate_f1_score(
    predicted: List[str],
    relevant: List[str],
    k: Optional[int] = None
) -> float:
    """
    Calculate F1 score@k.
    
    Args:
        predicted: List of predicted candidate IDs
        relevant: List of relevant candidate IDs
        k: Top K to consider
    
    Returns:
        F1 score (0-1)
    """
    precision = calculate_precision(predicted, relevant, k)
    recall = calculate_recall(predicted, relevant, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def calculate_ndcg(
    predicted: List[str],
    relevant: List[str],
    k: Optional[int] = None,
    gains: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)@k.
    
    Args:
        predicted: List of predicted candidate IDs (ordered by score)
        relevant: List of relevant candidate IDs
        k: Top K to consider (None for all)
        gains: Optional dictionary mapping candidate ID to gain value
    
    Returns:
        NDCG score (0-1)
    """
    if not predicted:
        return 0.0
    
    if k is not None:
        predicted = predicted[:k]
    
    relevant_set = set(relevant)
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(predicted):
        if item in relevant_set:
            if gains and item in gains:
                gain = gains[item]
            else:
                gain = 1.0  # Binary relevance
            
            # Discounted gain: gain / log2(i+2) (i+2 because i is 0-indexed)
            dcg += gain / np.log2(i + 2)
    
    # Calculate ideal DCG (IDCG)
    ideal_gains = []
    for item in relevant:
        if gains and item in gains:
            ideal_gains.append(gains[item])
        else:
            ideal_gains.append(1.0)
    
    # Sort gains in descending order
    ideal_gains.sort(reverse=True)
    
    idcg = 0.0
    for i, gain in enumerate(ideal_gains):
        if k is not None and i >= k:
            break
        idcg += gain / np.log2(i + 2)
    
    # Calculate NDCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_map(
    predictions: List[List[str]],
    ground_truth: List[List[str]]
) -> float:
    """
    Calculate Mean Average Precision (MAP).
    
    Args:
        predictions: List of predicted candidate ID lists (one per query)
        ground_truth: List of relevant candidate ID lists (one per query)
    
    Returns:
        MAP score (0-1)
    """
    if len(predictions) != len(ground_truth):
        logger.warning("Predictions and ground truth have different lengths")
        return 0.0
    
    if not predictions:
        return 0.0
    
    ap_scores = []
    for pred, relevant in zip(predictions, ground_truth):
        if not relevant:
            continue
        
        relevant_set = set(relevant)
        precisions = []
        relevant_count = 0
        
        for i, item in enumerate(pred):
            if item in relevant_set:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precisions.append(precision)
        
        if precisions:
            ap = sum(precisions) / len(relevant_set)
            ap_scores.append(ap)
    
    if not ap_scores:
        return 0.0
    
    return sum(ap_scores) / len(ap_scores)


def calculate_metrics(
    predicted: List[str],
    relevant: List[str],
    k: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate all metrics at once.
    
    Args:
        predicted: List of predicted candidate IDs
        relevant: List of relevant candidate IDs
        k: Top K to consider
    
    Returns:
        Dictionary of metric names to scores
    """
    return {
        'precision': calculate_precision(predicted, relevant, k),
        'recall': calculate_recall(predicted, relevant, k),
        'f1': calculate_f1_score(predicted, relevant, k),
        'ndcg': calculate_ndcg(predicted, relevant, k)
    }

