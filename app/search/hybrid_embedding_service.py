"""
Hybrid Embedding Service
Two-tier system: Fast bi-encoder for FAISS retrieval + Accurate cross-encoder for re-ranking.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available")

logger = logging.getLogger(__name__)


class HybridEmbeddingService:
    """Two-tier embedding system for fast retrieval and accurate re-ranking"""
    
    def __init__(self):
        self.bi_encoder = None
        self.cross_encoder = None
        self.domain_models: Dict[str, SentenceTransformer] = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize bi-encoder and cross-encoder models"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence transformers not available, using fallback")
            return
        
        try:
            # Fast bi-encoder for FAISS retrieval
            logger.info("Loading bi-encoder model...")
            self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Bi-encoder loaded successfully")
            
            # Accurate cross-encoder for re-ranking
            logger.info("Loading cross-encoder model...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Cross-encoder loaded successfully")
            
            # Domain-specific models (lazy load)
            self.domain_models = {}
            
        except Exception as e:
            logger.error(f"Error initializing embedding models: {e}")
            self.bi_encoder = None
            self.cross_encoder = None
    
    def get_domain_model(self, domain: str) -> Optional[SentenceTransformer]:
        """Get domain-specific bi-encoder model (lazy load)"""
        if domain not in self.domain_models:
            try:
                # Use general model for now, can be replaced with fine-tuned models
                if self.bi_encoder:
                    self.domain_models[domain] = self.bi_encoder
            except Exception as e:
                logger.warning(f"Could not load domain model for {domain}: {e}")
                return self.bi_encoder
        
        return self.domain_models.get(domain, self.bi_encoder)
    
    def encode_query(self, query: str, domain: str = 'general') -> Optional[np.ndarray]:
        """
        Encode query using bi-encoder (fast, for FAISS retrieval)
        
        Args:
            query: Job description or search query
            domain: Domain type ('healthcare', 'it/tech', 'general')
        
        Returns:
            Query embedding vector
        """
        if not self.bi_encoder:
            logger.warning("Bi-encoder not available")
            return None
        
        try:
            model = self.get_domain_model(domain)
            if not model:
                return None
            
            embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
            return embedding[0]  # Return single vector
            
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            return None
    
    def encode_candidates_batch(self, candidates: List[str], domain: str = 'general') -> Optional[np.ndarray]:
        """
        Encode candidate texts in batch (fast, for FAISS indexing)
        
        Args:
            candidates: List of candidate text strings
            domain: Domain type
        
        Returns:
            Matrix of candidate embeddings
        """
        if not self.bi_encoder or not candidates:
            return None
        
        try:
            model = self.get_domain_model(domain)
            if not model:
                return None
            
            embeddings = model.encode(
                candidates,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding candidates batch: {e}")
            return None
    
    def rerank_candidates(
        self,
        query: str,
        candidate_texts: List[str],
        top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Re-rank candidates using cross-encoder (accurate but slower)
        
        Args:
            query: Job description
            candidate_texts: List of candidate resume texts
            top_k: Number of top candidates to return
        
        Returns:
            List of (candidate_index, score) tuples, sorted by score descending
        """
        if not self.cross_encoder or not candidate_texts:
            return []
        
        try:
            # Create query-candidate pairs
            pairs = [[query, candidate_text] for candidate_text in candidate_texts]
            
            # Score pairs using cross-encoder
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            
            # Convert to list of (index, score) tuples
            scored_candidates = [(i, float(score)) for i, score in enumerate(scores)]
            
            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k
            return scored_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Error re-ranking candidates: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        candidate_texts: List[str],
        candidate_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        top_k_retrieval: int = 200,
        top_k_final: int = 20,
        domain: str = 'general'
    ) -> List[Tuple[int, float, float]]:
        """
        Hybrid search: FAISS retrieval + cross-encoder re-ranking
        
        Args:
            query: Job description
            candidate_texts: List of candidate texts
            candidate_embeddings: Pre-computed candidate embeddings (from FAISS)
            query_embedding: Pre-computed query embedding
            top_k_retrieval: Number of candidates to retrieve from FAISS
            top_k_final: Final number of candidates to return
            domain: Domain type
        
        Returns:
            List of (candidate_index, bi_encoder_score, cross_encoder_score) tuples
        """
        start_time = time.time()
        
        try:
            # Step 1: FAISS retrieval (fast)
            if candidate_embeddings is None or query_embedding is None:
                logger.warning("Embeddings not available, skipping FAISS retrieval")
                return []
            
            # Calculate cosine similarities
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            candidate_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
            
            similarities = np.dot(candidate_norms, query_norm).flatten()
            
            # Get top_k_retrieval candidates
            top_indices = np.argsort(similarities)[::-1][:top_k_retrieval]
            
            # Step 2: Cross-encoder re-ranking (accurate)
            top_candidate_texts = [candidate_texts[i] for i in top_indices]
            
            if not self.cross_encoder:
                # Fallback: return bi-encoder results
                results = [
                    (int(idx), float(similarities[idx]), 0.0)
                    for idx in top_indices[:top_k_final]
                ]
                return results
            
            # Re-rank with cross-encoder
            reranked = self.rerank_candidates(query, top_candidate_texts, top_k=top_k_retrieval)
            
            # Map back to original indices
            results = []
            for rerank_idx, cross_score in reranked:
                original_idx = top_indices[rerank_idx]
                bi_score = float(similarities[original_idx])
                results.append((int(original_idx), bi_score, float(cross_score)))
            
            elapsed = time.time() - start_time
            logger.info(f"Hybrid search completed: {len(results)} results in {elapsed:.2f}s")
            
            return results[:top_k_final]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for bi-encoder"""
        if self.bi_encoder:
            # all-MiniLM-L6-v2 has 384 dimensions
            return 384
        return 0


# Global instance
_hybrid_embedding_service = None

def get_hybrid_embedding_service() -> HybridEmbeddingService:
    """Get or create global hybrid embedding service instance"""
    global _hybrid_embedding_service
    if _hybrid_embedding_service is None:
        _hybrid_embedding_service = HybridEmbeddingService()
    return _hybrid_embedding_service

