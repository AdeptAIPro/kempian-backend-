"""Utility modules for text processing and similarity calculations."""

from .text_cleaner import TextCleaner, normalize_text
from .similarity import cosine_similarity, batch_cosine_similarity

__all__ = [
    'TextCleaner',
    'normalize_text',
    'cosine_similarity',
    'batch_cosine_similarity',
]

