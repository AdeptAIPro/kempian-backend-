"""
Text normalization and cleaning utilities.
Handles text preprocessing for resume and job description parsing.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TextCleaner:
    """Text cleaning and normalization utilities."""
    
    @staticmethod
    def clean(text: Optional[str]) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if text is None:
            return ""
        
        if not isinstance(text, str):
            text = str(text)
        
        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep spaces, letters, numbers, and common punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def normalize(text: Optional[str]) -> str:
        """
        Normalize text to lowercase and clean.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized lowercase text
        """
        cleaned = TextCleaner.clean(text)
        return cleaned.lower()
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """
        Remove extra whitespace from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        """
        Extract email address from text.
        
        Args:
            text: Input text
            
        Returns:
            Email address or None
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None
    
    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        """
        Extract phone number from text.
        
        Args:
            text: Input text
            
        Returns:
            Phone number or None
        """
        # Common phone patterns
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',  # (123) 456-7890
            r'\b\d{10}\b',  # 10 digits
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return None


def normalize_text(text: Optional[str]) -> str:
    """
    Convenience function to normalize text.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    return TextCleaner.normalize(text)

