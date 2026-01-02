"""
Skill extraction using ontology-based matching.
Handles skill canonicalization, alias matching, and fuzzy matching.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

from ..utils.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSkill:
    """Extracted skill with confidence score."""
    skill_id: str
    canonical_name: str
    confidence: float
    matched_text: str


class SkillExtractor:
    """Extract and canonicalize skills using ontology."""
    
    def __init__(self, ontology_path: Optional[str] = None):
        """
        Initialize skill extractor.
        
        Args:
            ontology_path: Path to skill ontology JSON file
        """
        self.ontology_path = ontology_path or self._get_default_ontology_path()
        self.skill_ontology: Dict[str, Dict] = {}
        self.alias_to_skill_id: Dict[str, str] = {}
        self.skill_hierarchy: Dict[str, List[str]] = defaultdict(list)
        self.category_map: Dict[str, List[str]] = defaultdict(list)
        
        self._load_ontology()
    
    def _get_default_ontology_path(self) -> str:
        """Get default path to skill ontology."""
        current_dir = Path(__file__).parent.parent
        return str(current_dir / 'data' / 'skill_ontology.json')
    
    def _load_ontology(self):
        """Load skill ontology from JSON file."""
        try:
            if not Path(self.ontology_path).exists():
                logger.warning(f"Skill ontology not found at {self.ontology_path}")
                return
            
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Build ontology structures
            for skill_data in data.get('skills', []):
                skill_id = skill_data['skill_id']
                self.skill_ontology[skill_id] = skill_data
                
                # Map canonical name
                canonical = skill_data['canonical_name'].lower()
                self.alias_to_skill_id[canonical] = skill_id
                
                # Map aliases
                for alias in skill_data.get('aliases', []):
                    normalized_alias = TextCleaner.normalize(alias)
                    self.alias_to_skill_id[normalized_alias] = skill_id
                
                # Map synonyms
                for synonym in skill_data.get('synonyms', []):
                    normalized_synonym = TextCleaner.normalize(synonym)
                    self.alias_to_skill_id[normalized_synonym] = skill_id
                
                # Build hierarchy
                parent_id = skill_data.get('parent_skill_id')
                if parent_id:
                    self.skill_hierarchy[parent_id].append(skill_id)
                
                # Build category map
                category = skill_data.get('category', 'general')
                self.category_map[category].append(skill_id)
            
            logger.info(f"Loaded {len(self.skill_ontology)} skills from ontology")
        except Exception as e:
            logger.error(f"Error loading skill ontology: {e}")
    
    def extract_skills(self, text: str, min_confidence: float = 0.7) -> List[ExtractedSkill]:
        """
        Extract skills from text using ontology matching.
        
        Args:
            text: Input text to extract skills from
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of extracted skills with confidence scores
        """
        if not text:
            return []
        
        normalized_text = TextCleaner.normalize(text)
        extracted: List[ExtractedSkill] = []
        seen_skill_ids: Set[str] = set()
        
        # Direct matching against aliases and canonical names
        for alias, skill_id in self.alias_to_skill_id.items():
            if alias in normalized_text:
                if skill_id not in seen_skill_ids:
                    skill_data = self.skill_ontology[skill_id]
                    extracted.append(ExtractedSkill(
                        skill_id=skill_id,
                        canonical_name=skill_data['canonical_name'],
                        confidence=1.0,
                        matched_text=alias
                    ))
                    seen_skill_ids.add(skill_id)
        
        # Word boundary matching for better accuracy
        word_boundary_matches = self._extract_with_word_boundaries(normalized_text)
        for skill_id, matched_text in word_boundary_matches:
            if skill_id not in seen_skill_ids:
                skill_data = self.skill_ontology[skill_id]
                extracted.append(ExtractedSkill(
                    skill_id=skill_id,
                    canonical_name=skill_data['canonical_name'],
                    confidence=0.9,
                    matched_text=matched_text
                ))
                seen_skill_ids.add(skill_id)
        
        # Fuzzy matching for misspellings
        fuzzy_matches = self._fuzzy_match(normalized_text, seen_skill_ids)
        for skill_id, matched_text, confidence in fuzzy_matches:
            if confidence >= min_confidence:
                skill_data = self.skill_ontology[skill_id]
                extracted.append(ExtractedSkill(
                    skill_id=skill_id,
                    canonical_name=skill_data['canonical_name'],
                    confidence=confidence,
                    matched_text=matched_text
                ))
                seen_skill_ids.add(skill_id)
        
        return extracted
    
    def _extract_with_word_boundaries(self, text: str) -> List[Tuple[str, str]]:
        """Extract skills using word boundary matching."""
        matches = []
        words = set(text.split())
        
        for alias, skill_id in self.alias_to_skill_id.items():
            # Check if alias appears as a whole word
            alias_words = alias.split()
            if len(alias_words) == 1:
                # Single word alias
                if alias in words:
                    matches.append((skill_id, alias))
            else:
                # Multi-word alias - check if all words appear together
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text):
                    matches.append((skill_id, alias))
        
        return matches
    
    def _fuzzy_match(self, text: str, seen_skill_ids: Set[str], threshold: float = 0.75) -> List[Tuple[str, str, float]]:
        """
        Enhanced fuzzy match skills using advanced string similarity.
        
        Args:
            text: Normalized text
            seen_skill_ids: Already matched skill IDs to skip
            threshold: Similarity threshold (lowered to 0.75 for better recall)
            
        Returns:
            List of (skill_id, matched_text, confidence) tuples
        """
        matches = []
        text_words = text.split()
        text_phrases = self._extract_phrases(text)  # Extract 2-3 word phrases
        
        for skill_id, skill_data in self.skill_ontology.items():
            if skill_id in seen_skill_ids:
                continue
            
            canonical = skill_data['canonical_name'].lower()
            best_match = None
            best_score = 0.0
            match_type = 'word'
            
            # Check against canonical name (full string match first)
            full_text_sim = self._string_similarity(text, canonical)
            if full_text_sim > best_score and full_text_sim >= threshold:
                best_score = full_text_sim
                best_match = canonical
                match_type = 'full'
            
            # Check canonical name against individual words
            for word in text_words:
                if len(word) < 2:  # Lowered from 3 to catch abbreviations like "js", "ts"
                    continue
                similarity = self._string_similarity(word, canonical)
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = word
                    match_type = 'word'
            
            # Check canonical name against phrases (for multi-word skills)
            for phrase in text_phrases:
                similarity = self._string_similarity(phrase, canonical)
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = phrase
                    match_type = 'phrase'
            
            # Check against aliases (with higher priority)
            for alias in skill_data.get('aliases', []):
                normalized_alias = TextCleaner.normalize(alias)
                
                # Full text match
                full_alias_sim = self._string_similarity(text, normalized_alias)
                if full_alias_sim > best_score and full_alias_sim >= threshold:
                    best_score = full_alias_sim
                    best_match = normalized_alias
                    match_type = 'alias_full'
                
                # Word matches
                for word in text_words:
                    if len(word) < 2:
                        continue
                    similarity = self._string_similarity(word, normalized_alias)
                    if similarity > best_score and similarity >= threshold:
                        best_score = similarity
                        best_match = word
                        match_type = 'alias_word'
                
                # Phrase matches
                for phrase in text_phrases:
                    similarity = self._string_similarity(phrase, normalized_alias)
                    if similarity > best_score and similarity >= threshold:
                        best_score = similarity
                        best_match = phrase
                        match_type = 'alias_phrase'
            
            # Check synonyms
            for synonym in skill_data.get('synonyms', []):
                normalized_synonym = TextCleaner.normalize(synonym)
                for phrase in text_phrases:
                    similarity = self._string_similarity(phrase, normalized_synonym)
                    if similarity > best_score and similarity >= threshold:
                        best_score = similarity
                        best_match = phrase
                        match_type = 'synonym'
            
            if best_match:
                # Boost confidence for exact/close matches
                if match_type in ['full', 'alias_full']:
                    best_score = min(1.0, best_score * 1.05)
                elif match_type in ['phrase', 'alias_phrase']:
                    best_score = min(1.0, best_score * 1.02)
                
                matches.append((skill_id, best_match, best_score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches
    
    def _extract_phrases(self, text: str, max_phrase_length: int = 3) -> List[str]:
        """Extract 2-3 word phrases from text for better multi-word skill matching."""
        words = text.split()
        phrases = []
        
        for i in range(len(words)):
            for length in range(2, min(max_phrase_length + 1, len(words) - i + 1)):
                phrase = ' '.join(words[i:i+length])
                phrases.append(phrase)
        
        return phrases
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate advanced string similarity using multiple algorithms.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        if not s1 or not s2:
            return 0.0
        
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        
        # Exact match
        if s1 == s2:
            return 1.0
        
        # Remove common suffixes/prefixes for better matching
        s1_clean = s1.replace('.js', '').replace('.net', '').replace('js', '').replace('net', '')
        s2_clean = s2.replace('.js', '').replace('.net', '').replace('js', '').replace('net', '')
        if s1_clean == s2_clean:
            return 0.95
        
        # Substring match (one contains the other)
        if s1 in s2 or s2 in s1:
            # Longer string contains shorter - high confidence
            min_len = min(len(s1), len(s2))
            max_len = max(len(s1), len(s2))
            if min_len >= 3:  # At least 3 characters
                return 0.85 + (min_len / max_len) * 0.1
        
        # Levenshtein distance-based similarity
        levenshtein_sim = self._levenshtein_similarity(s1, s2)
        
        # Jaro-Winkler similarity (better for short strings)
        jaro_winkler_sim = self._jaro_winkler_similarity(s1, s2)
        
        # Character set similarity
        set_sim = self._character_set_similarity(s1, s2)
        
        # Word-based similarity (for multi-word skills)
        word_sim = self._word_based_similarity(s1, s2)
        
        # Weighted combination
        # Levenshtein and Jaro-Winkler are most important for skill matching
        final_score = (
            levenshtein_sim * 0.35 +
            jaro_winkler_sim * 0.35 +
            word_sim * 0.20 +
            set_sim * 0.10
        )
        
        return min(1.0, final_score)
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein distance-based similarity."""
        if not s1 or not s2:
            return 0.0
        
        # Calculate Levenshtein distance
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        distance = dp[m][n]
        max_len = max(len(s1), len(s2))
        
        if max_len == 0:
            return 1.0
        
        return 1.0 - (distance / max_len)
    
    def _jaro_winkler_similarity(self, s1: str, s2: str, p: float = 0.1) -> float:
        """Calculate Jaro-Winkler similarity (better for short strings and typos)."""
        if s1 == s2:
            return 1.0
        
        # Jaro similarity
        jaro_sim = self._jaro_similarity(s1, s2)
        
        if jaro_sim < 0.7:
            return jaro_sim
        
        # Winkler modification: boost for common prefix
        prefix_len = 0
        min_len = min(len(s1), len(s2))
        for i in range(min(4, min_len)):  # Max 4 character prefix
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break
        
        return jaro_sim + (p * prefix_len * (1 - jaro_sim))
    
    def _jaro_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro similarity."""
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        match_window = max(len1, len2) // 2 - 1
        if match_window < 0:
            match_window = 0
        
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        # Find matches
        for i in range(len1):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len2)
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        jaro = (
            matches / len1 +
            matches / len2 +
            (matches - transpositions / 2) / matches
        ) / 3.0
        
        return jaro
    
    def _character_set_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity based on character sets."""
        set1 = set(s1)
        set2 = set(s2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _word_based_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity based on word overlap (for multi-word skills)."""
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        # Also check for partial word matches
        partial_matches = 0
        for w1 in words1:
            for w2 in words2:
                if w1 in w2 or w2 in w1:
                    partial_matches += 0.5
        
        base_score = intersection / union
        partial_bonus = min(0.2, partial_matches / max(len(words1), len(words2)) * 0.2)
        
        return min(1.0, base_score + partial_bonus)
    
    def canonicalize_skill(self, skill_name: str) -> Optional[str]:
        """
        Canonicalize a skill name to its skill_id.
        
        Args:
            skill_name: Skill name to canonicalize
            
        Returns:
            Canonical skill_id or None
        """
        normalized = TextCleaner.normalize(skill_name)
        return self.alias_to_skill_id.get(normalized)
    
    def get_skill_info(self, skill_id: str) -> Optional[Dict]:
        """
        Get skill information by skill_id.
        
        Args:
            skill_id: Skill ID
            
        Returns:
            Skill data dictionary or None
        """
        return self.skill_ontology.get(skill_id)
    
    def get_child_skills(self, skill_id: str) -> List[str]:
        """
        Get child skills in the hierarchy.
        
        Args:
            skill_id: Parent skill ID
            
        Returns:
            List of child skill IDs
        """
        return self.skill_hierarchy.get(skill_id, [])
    
    def get_skills_by_category(self, category: str) -> List[str]:
        """
        Get all skills in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of skill IDs
        """
        return self.category_map.get(category, [])

