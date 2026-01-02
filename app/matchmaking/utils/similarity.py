"""
Similarity calculation utilities.
Provides cosine similarity and batch processing functions.
"""

import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    try:
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Clamp to [0, 1] range
        return float(np.clip(similarity, 0.0, 1.0))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


def batch_cosine_similarity(
    query_vec: np.ndarray,
    candidate_vecs: Union[List[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Calculate cosine similarity between a query vector and multiple candidate vectors.
    
    Args:
        query_vec: Query vector (1D array)
        candidate_vecs: List of candidate vectors or 2D array
        
    Returns:
        Array of similarity scores
    """
    try:
        # Convert to numpy array if needed
        if isinstance(candidate_vecs, list):
            candidate_vecs = np.array(candidate_vecs)
        
        # Normalize query vector
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        # Normalize candidate vectors
        candidate_norms = candidate_vecs / (np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarities
        similarities = np.dot(candidate_norms, query_norm)
        
        # Clamp to [0, 1] range
        return np.clip(similarities, 0.0, 1.0)
    except Exception as e:
        logger.error(f"Error calculating batch cosine similarity: {e}")
        return np.zeros(len(candidate_vecs) if hasattr(candidate_vecs, '__len__') else 1)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vec: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

