"""
Enhanced embedding utilities extracted from main system
"""
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MultiModelEmbeddingService:
    """Centralized embedding service with multiple models"""
    
    def __init__(self):
        self.models = {}
        self.cache = {}
        self.load_models()
    
    def load_models(self):
        """Load different models for different purposes"""
        try:
            # General purpose model
            self.models['general'] = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
            # Fast model for real-time queries
            self.models['fast'] = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("✅ Multi-model embedding service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding models: {e}")
            # Fallback to single model
            self.models['general'] = SentenceTransformer('all-MiniLM-L6-v2')
            self.models['fast'] = self.models['general']
    
    def get_embedding(self, text: str, model_type: str = 'general') -> np.ndarray:
        """Get embedding with caching"""
        cache_key = hashlib.md5(f"{model_type}:{text}".encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        model = self.models.get(model_type, self.models['general'])
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        
        # Cache with size limit
        if len(self.cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.cache.keys())[:100]
            for key in oldest_keys:
                del self.cache[key]
        
        self.cache[cache_key] = embedding
        return embedding
    
    def get_batch_embeddings(self, texts: List[str], model_type: str = 'general') -> np.ndarray:
        """Get batch embeddings efficiently"""
        model = self.models.get(model_type, self.models['general'])
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

# Global instance
embedding_service = MultiModelEmbeddingService()