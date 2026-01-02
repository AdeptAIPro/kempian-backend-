"""
NER-Based Skill Extraction

Uses Named Entity Recognition (NER) to extract skills from candidate text,
replacing pattern-based skill extraction with ML-based approach.
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Install with: pip install transformers")

# Try to import spacy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spacy not available. Install with: pip install spacy")


class SkillExtractorNER:
    """
    NER-based skill extractor
    
    Uses fine-tuned BERT models or spaCy for skill extraction
    """
    
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        """
        Initialize NER Skill Extractor
        
        Args:
            model_name: HuggingFace model name for NER
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None
        
        # Initialize model
        if TRANSFORMERS_AVAILABLE:
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model=model_name,
                    aggregation_strategy="simple"
                )
                logger.info(f"Initialized NER model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize transformer NER: {e}")
                self.ner_pipeline = None
        
        # Fallback to spaCy
        if not self.ner_pipeline and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Initialized spaCy NER model")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp = None
        
        # Skill patterns as fallback
        self._init_skill_patterns()
    
    def _init_skill_patterns(self):
        """Initialize skill patterns as fallback"""
        self.skill_patterns = {
            'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'ruby', 'php', 'swift', 'kotlin'],
            'web': ['react', 'vue', 'angular', 'node', 'django', 'flask', 'fastapi', 'spring', 'express'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch'],
            'ml_ai': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'machine learning', 'deep learning'],
            'medical': ['nursing', 'patient care', 'medical diagnosis', 'surgery', 'pharmacy', 'radiology'],
            'finance': ['accounting', 'financial analysis', 'cfa', 'cpa', 'risk management', 'trading']
        }
        
        # Canonicalization map (aliases → canonical skill names)
        self.canonical_aliases = {
            'js': 'javascript',
            'node.js': 'node',
            'nodejs': 'node',
            'tf': 'tensorflow',
            'pyTorch': 'pytorch',
            'sklearn': 'scikit-learn',
            'postgre': 'postgresql',
            'postgres': 'postgresql',
            'gcloud': 'gcp',
            'ms azure': 'azure'
        }
    
    def extract_skills(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract skills from text using NER
        
        Args:
            text: Input text (resume, job description, etc.)
            
        Returns:
            List of extracted skills with confidence scores
        """
        if not text:
            return []
        
        skills = []
        
        # Use transformer NER if available
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text)
                for entity in entities:
                    # Filter for relevant entity types
                    if entity['entity_group'] in ['ORG', 'MISC', 'PER']:
                        # Check if it looks like a skill
                        skill_text = entity['word'].strip()
                        if self._is_skill(skill_text):
                            skills.append({
                                'skill': skill_text,
                                'confidence': entity['score'],
                                'method': 'ner_transformer'
                            })
            except Exception as e:
                logger.warning(f"Transformer NER failed: {e}")
        
        # Use spaCy NER if available
        if not skills and hasattr(self, 'nlp') and self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'MISC']:
                        skill_text = ent.text.strip()
                        if self._is_skill(skill_text):
                            skills.append({
                                'skill': skill_text,
                                'confidence': 0.8,  # spaCy doesn't provide confidence
                                'method': 'ner_spacy'
                            })
            except Exception as e:
                logger.warning(f"spaCy NER failed: {e}")
        
        # Fallback to pattern matching
        if not skills:
            skills = self._extract_skills_pattern(text)
        
        # Remove duplicates
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill['skill'].lower()
            # Canonicalize aliases
            canonical = self.canonical_aliases.get(skill_lower, skill_lower)
            if canonical != skill_lower:
                skill['skill'] = canonical
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        return unique_skills
    
    def _is_skill(self, text: str) -> bool:
        """Check if text looks like a skill"""
        text_lower = text.lower()
        
        # Check against skill patterns
        for category, patterns in self.skill_patterns.items():
            for pattern in patterns:
                if pattern in text_lower or text_lower in pattern:
                    return True
        
        # Check if it's a known technology/term
        if len(text) > 2 and len(text) < 50:
            # Check for common skill indicators
            if any(indicator in text_lower for indicator in ['developer', 'engineer', 'specialist', 'expert']):
                return True
        
        return False
    
    def _extract_skills_pattern(self, text: str) -> List[Dict[str, Any]]:
        """Fallback pattern-based skill extraction"""
        skills = []
        text_lower = text.lower()
        
        for category, patterns in self.skill_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Find the full context
                    pattern_regex = re.compile(rf'\b{re.escape(pattern)}\b', re.IGNORECASE)
                    matches = pattern_regex.findall(text)
                    for match in matches:
                        skills.append({
                            'skill': match,
                            'confidence': 0.7,
                            'method': 'pattern_match',
                            'category': category
                        })
        
        return skills
    
    def extract_skills_simple(self, text: str) -> List[str]:
        """
        Extract skills as simple list (backwards compatibility)
        
        Args:
            text: Input text
            
        Returns:
            List of skill strings
        """
        skills = self.extract_skills(text)
        return [skill['skill'] for skill in skills]


# Global instance
_skill_extractor = None


def get_skill_extractor() -> SkillExtractorNER:
    """Get or create global skill extractor instance"""
    global _skill_extractor
    if _skill_extractor is None:
        _skill_extractor = SkillExtractorNER()
    return _skill_extractor

