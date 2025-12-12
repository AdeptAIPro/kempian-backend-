"""
Enhanced embedding utilities extracted from main system
"""
import hashlib
import logging
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)

class HashFallbackEncoder:
    """Deterministic hashing-based encoder used when SentenceTransformer models are unavailable."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _encode_single(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeats = (self.dimension * 4 // len(digest)) + 1
        buffer = (digest * repeats)[: self.dimension * 4]
        vector = np.frombuffer(buffer, dtype=np.uint32).astype(np.float32)[: self.dimension]
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def encode(self, texts, convert_to_numpy: bool = True, normalize_embeddings: bool = True, **_) -> np.ndarray:
        if isinstance(texts, str):
            embedding = self._encode_single(texts)
            return embedding if convert_to_numpy else embedding.tolist()

        embeddings = [self._encode_single(text) for text in texts]
        matrix = np.vstack(embeddings)
        if not convert_to_numpy:
            return matrix.tolist()
        return matrix


class MultiModelEmbeddingService:
    """Centralized embedding service with multiple models"""

    def __init__(self):
        self.models = {}
        self.cache = {}
        self.load_models()

    def _setup_fallback_models(self, reason: str) -> None:
        logger.warning("⚠️ Falling back to hash-based embeddings (%s)", reason)
        fallback = HashFallbackEncoder()
        self.models["general"] = fallback
        self.models["fast"] = fallback

    def load_models(self):
        """Load different models for different purposes"""
        if SentenceTransformer is None:
            self._setup_fallback_models("sentence_transformers not available")
            return

        try:
            self.models["general"] = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            self.models["fast"] = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Multi-model embedding service initialized")
        except Exception as exc:
            logger.error("❌ Failed to load embedding models: %s", exc)
            self._setup_fallback_models("model loading failure")

    def get_embedding(self, text: str, model_type: str = "general") -> np.ndarray:
        """Get embedding with caching"""
        cache_key = hashlib.md5(f"{model_type}:{text}".encode()).hexdigest()

        if cache_key in self.cache:
            return self.cache[cache_key]

        model = self.models.get(model_type, self.models["general"])
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        if len(self.cache) > 1000:
            oldest_keys = list(self.cache.keys())[:100]
            for key in oldest_keys:
                del self.cache[key]

        self.cache[cache_key] = embedding
        return embedding

    def get_batch_embeddings(self, texts: List[str], model_type: str = "general") -> np.ndarray:
        """Get batch embeddings efficiently"""
        model = self.models.get(model_type, self.models["general"])
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


# Global instance
embedding_service = MultiModelEmbeddingService()