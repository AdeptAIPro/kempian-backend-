"""
Embedding generation using SentenceTransformers.
Provides efficient embedding generation with caching and batch processing.
"""

import logging
import numpy as np
from typing import List, Optional, Union
import threading

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)

# Global model instance (singleton pattern)
_model_instance: Optional['Embedder'] = None
_model_lock = threading.Lock()


class Embedder:
    """SentenceTransformer wrapper for embedding generation."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedder with specified model.
        
        Args:
            model_name: Name of SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._initialized = False
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
            return
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the SentenceTransformer model."""
        if self._initialized:
            return
        
        try:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            self.model = None
            self._initialized = False
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Encode text(s) into embedding vectors.
        
        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings (1D for single text, 2D for multiple texts)
        """
        if not self.model:
            logger.warning("Model not available, cannot encode")
            return None
        
        if not texts:
            return None
        
        # Convert single string to list
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        try:
            # Encode texts
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Return single vector for single input
            if is_single:
                return embeddings[0]
            
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return None
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Encode a batch of texts into embedding vectors.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            2D numpy array of embeddings (num_texts x embedding_dim)
        """
        return self.encode(texts, batch_size=batch_size)
    
    def is_available(self) -> bool:
        """
        Check if embedding model is available.
        
        Returns:
            True if model is loaded and available
        """
        return self.model is not None and self._initialized


def get_embedder(model_name: str = 'all-MiniLM-L6-v2') -> Embedder:
    """
    Get or create global embedder instance (singleton pattern).
    
    Args:
        model_name: Name of SentenceTransformer model to use
        
    Returns:
        Embedder instance
    """
    global _model_instance
    
    with _model_lock:
        if _model_instance is None or _model_instance.model_name != model_name:
            _model_instance = Embedder(model_name=model_name)
    
    return _model_instance

