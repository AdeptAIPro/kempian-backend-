"""
Production-Grade Skill Canonicalization System
Complete ontology with parent/child relationships, seniority detection, and weighted matching.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import difflib
from collections import defaultdict
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SkillNode:
    """Skill node in ontology hierarchy"""
    skill_id: str
    canonical_name: str
    category: str  # e.g., 'frontend', 'backend', 'database', 'cloud', 'healthcare'
    subcategory: Optional[str] = None
    level: str  # 'language', 'framework', 'tool', 'platform', 'methodology', 'certification'
    parent_skill_id: Optional[str] = None
    aliases: List[str] = None
    synonyms: List[str] = None
    abbreviations: List[str] = None
    plurals: List[str] = None
    misspellings: List[str] = None
    tool_variants: List[str] = None
    required_weight: float = 1.0  # Weight for required skills
    preferred_weight: float = 0.5  # Weight for preferred skills
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.synonyms is None:
            self.synonyms = []
        if self.abbreviations is None:
            self.abbreviations = []
        if self.plurals is None:
            self.plurals = []
        if self.misspellings is None:
            self.misspellings = []
        if self.tool_variants is None:
            self.tool_variants = []


@dataclass
class SkillSeniority:
    """Skill seniority level detection"""
    skill_id: str
    seniority: str  # 'junior', 'mid', 'senior', 'expert'
    confidence: float
    evidence: List[str]  # Text evidence for detection


class ProductionSkillOntology:
    """Production-grade skill canonicalization with full ontology"""
    
    def __init__(self, ontology_path: Optional[str] = None):
        self.ontology_path = ontology_path or self._get_default_ontology_path()
        self.skill_ontology: Dict[str, SkillNode] = {}
        self.alias_to_skill_id: Dict[str, str] = {}
        self.skill_hierarchy: Dict[str, List[str]] = defaultdict(list)  # parent -> children
        self.category_hierarchy: Dict[str, List[str]] = defaultdict(list)  # category -> skills
        self.embedding_model = None
        self.skill_embeddings: Dict[str, np.ndarray] = {}
        
        # Seniority detection patterns
        self.seniority_patterns = {
            'junior': [
                r'junior\s+\w+',
                r'entry\s+level',
                r'beginner',
                r'0-2\s+years',
                r'less\s+than\s+2\s+years'
            ],
            'mid': [
                r'mid\s+level',
                r'intermediate',
                r'2-5\s+years',
                r'3-5\s+years'
            ],
            'senior': [
                r'senior\s+\w+',
                r'sr\.?\s+\w+',
                r'5\+\s+years',
                r'5-10\s+years',
                r'experienced'
            ],
            'expert': [
                r'expert\s+\w+',
                r'principal\s+\w+',
                r'lead\s+\w+',
                r'10\+\s+years',
                r'architect',
                r'thought\s+leader'
            ]
        }
        
        # Load ontology
        self._load_ontology()
        
        # Initialize embedding model
        self._initialize_embeddings()
    
    def _get_default_ontology_path(self) -> str:
        """Get default path to skill ontology"""
        current_dir = Path(__file__).parent
        return str(current_dir / 'data' / 'production_skill_ontology.json')
    
    def _load_ontology(self):
        """Load complete skill ontology"""
        try:
            if not Path(self.ontology_path).exists():
                logger.warning(f"Ontology not found, creating comprehensive default")
                self._create_comprehensive_ontology()
                return
            
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Build ontology
            for skill_data in data.get('skills', []):
                skill = SkillNode(
                    skill_id=skill_data['skill_id'],
                    canonical_name=skill_data['canonical_name'],
                    category=skill_data.get('category', 'general'),
                    subcategory=skill_data.get('subcategory'),
                    level=skill_data.get('level', 'skill'),
                    parent_skill_id=skill_data.get('parent_skill_id'),
                    aliases=skill_data.get('aliases', []),
                    synonyms=skill_data.get('synonyms', []),
                    abbreviations=skill_data.get('abbreviations', []),
                    plurals=skill_data.get('plurals', []),
                    misspellings=skill_data.get('misspellings', []),
                    tool_variants=skill_data.get('tool_variants', []),
                    required_weight=skill_data.get('required_weight', 1.0),
                    preferred_weight=skill_data.get('preferred_weight', 0.5)
                )
                
                self.skill_ontology[skill.skill_id] = skill
                
                # Map all variations to skill_id
                all_variations = (
                    [skill.canonical_name] +
                    skill.aliases + skill.synonyms + skill.abbreviations +
                    skill.plurals + skill.misspellings + skill.tool_variants
                )
                
                for variation in all_variations:
                    normalized = self._normalize_skill_name(variation)
                    self.alias_to_skill_id[normalized] = skill.skill_id
                
                # Build hierarchy
                if skill.parent_skill_id:
                    self.skill_hierarchy[skill.parent_skill_id].append(skill.skill_id)
                
                # Build category hierarchy
                self.category_hierarchy[skill.category].append(skill.skill_id)
            
            logger.info(f"Loaded {len(self.skill_ontology)} skills with full hierarchy")
            
        except Exception as e:
            logger.error(f"Error loading ontology: {e}")
            self._create_comprehensive_ontology()
    
    def _initialize_embeddings(self):
        """Initialize embedding model for fuzzy matching"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Pre-compute embeddings for all canonical skills
                canonical_names = [skill.canonical_name for skill in self.skill_ontology.values()]
                if canonical_names:
                    embeddings = self.embedding_model.encode(canonical_names, show_progress_bar=False)
                    for i, skill_id in enumerate(self.skill_ontology.keys()):
                        self.skill_embeddings[skill_id] = embeddings[i]
                logger.info("Skill embeddings initialized")
            except Exception as e:
                logger.warning(f"Could not initialize embeddings: {e}")
    
    def _normalize_skill_name(self, skill_name: str) -> str:
        """Normalize skill name for matching"""
        if not skill_name:
            return ''
        
        # Lowercase, strip, remove special chars
        normalized = skill_name.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        normalized = re.sub(r'\.js$', 'js', normalized)  # Normalize .js
        normalized = re.sub(r'\.net$', 'net', normalized)  # Normalize .net
        
        return normalized
    
    def canonicalize_skill(
        self,
        raw_skill: str,
        threshold_exact: float = 0.95,
        threshold_fuzzy: float = 0.80,
        threshold_embedding: float = 0.75
    ) -> Optional[Tuple[str, str, float, str]]:
        """
        Canonicalize skill with confidence levels
        
        Returns: (skill_id, canonical_name, confidence, match_type) or None
        match_type: 'exact', 'alias', 'fuzzy', 'embedding', 'hierarchy'
        """
        if not raw_skill:
            return None
        
        normalized = self._normalize_skill_name(raw_skill)
        
        # 1. Exact match
        if normalized in self.alias_to_skill_id:
            skill_id = self.alias_to_skill_id[normalized]
            skill = self.skill_ontology[skill_id]
            return (skill_id, skill.canonical_name, 1.0, 'exact')
        
        # 2. Fuzzy string matching
        best_fuzzy_match = None
        best_fuzzy_score = 0.0
        
        for alias, skill_id in self.alias_to_skill_id.items():
            similarity = difflib.SequenceMatcher(None, normalized, alias).ratio()
            
            # Boost exact substring matches
            if normalized in alias or alias in normalized:
                similarity = max(similarity, 0.92)
            
            if similarity > best_fuzzy_score:
                best_fuzzy_score = similarity
                best_fuzzy_match = (skill_id, alias)
        
        if best_fuzzy_match and best_fuzzy_score >= threshold_fuzzy:
            skill_id, matched_alias = best_fuzzy_match
            skill = self.skill_ontology[skill_id]
            return (skill_id, skill.canonical_name, best_fuzzy_score, 'fuzzy')
        
        # 3. Embedding-based matching
        if self.embedding_model and best_fuzzy_score < threshold_embedding:
            embedding_match = self._embedding_match(normalized, threshold_embedding)
            if embedding_match:
                return embedding_match
        
        # 4. Hierarchy matching (check if raw_skill is a parent/child)
        hierarchy_match = self._hierarchy_match(normalized)
        if hierarchy_match:
            return hierarchy_match
        
        return None
    
    def _embedding_match(self, normalized_skill: str, threshold: float) -> Optional[Tuple[str, str, float, str]]:
        """Match skill using embeddings"""
        if not self.embedding_model or not self.skill_embeddings:
            return None
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            query_embedding = self.embedding_model.encode([normalized_skill])
            
            best_score = 0.0
            best_skill_id = None
            
            for skill_id, skill_embedding in self.skill_embeddings.items():
                similarity = cosine_similarity(query_embedding, [skill_embedding])[0][0]
                if similarity > best_score:
                    best_score = float(similarity)
                    best_skill_id = skill_id
            
            if best_skill_id and best_score >= threshold:
                skill = self.skill_ontology[best_skill_id]
                return (best_skill_id, skill.canonical_name, best_score, 'embedding')
        
        except Exception as e:
            logger.error(f"Error in embedding match: {e}")
        
        return None
    
    def _hierarchy_match(self, normalized_skill: str) -> Optional[Tuple[str, str, float, str]]:
        """Match skill through hierarchy relationships"""
        # Check if normalized skill matches a parent or child
        for skill_id, skill in self.skill_ontology.items():
            # Check parent
            if skill.parent_skill_id:
                parent_skill = self.skill_ontology.get(skill.parent_skill_id)
                if parent_skill:
                    parent_normalized = self._normalize_skill_name(parent_skill.canonical_name)
                    if normalized_skill in parent_normalized or parent_normalized in normalized_skill:
                        return (skill_id, skill.canonical_name, 0.7, 'hierarchy')
            
            # Check children
            children = self.skill_hierarchy.get(skill_id, [])
            for child_id in children:
                child_skill = self.skill_ontology.get(child_id)
                if child_skill:
                    child_normalized = self._normalize_skill_name(child_skill.canonical_name)
                    if normalized_skill in child_normalized or child_normalized in normalized_skill:
                        return (skill_id, skill.canonical_name, 0.7, 'hierarchy')
        
        return None
    
    def detect_skill_seniority(self, skill_id: str, experience_text: str) -> SkillSeniority:
        """Detect seniority level for a skill from experience text"""
        experience_lower = experience_text.lower()
        
        best_seniority = 'mid'  # Default
        best_confidence = 0.5
        evidence = []
        
        # Check patterns
        for seniority, patterns in self.seniority_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, experience_lower, re.IGNORECASE)
                if matches:
                    confidence = min(0.9, 0.5 + len(matches) * 0.1)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_seniority = seniority
                        evidence.extend(matches)
        
        # Check years of experience
        years_match = re.search(r'(\d+)\+?\s*years?', experience_lower)
        if years_match:
            years = int(years_match.group(1))
            if years >= 10:
                best_seniority = 'expert'
                best_confidence = 0.9
                evidence.append(f"{years} years")
            elif years >= 5:
                best_seniority = 'senior'
                best_confidence = 0.8
                evidence.append(f"{years} years")
            elif years >= 2:
                best_seniority = 'mid'
                best_confidence = 0.7
                evidence.append(f"{years} years")
            else:
                best_seniority = 'junior'
                best_confidence = 0.7
                evidence.append(f"{years} years")
        
        return SkillSeniority(
            skill_id=skill_id,
            seniority=best_seniority,
            confidence=best_confidence,
            evidence=evidence
        )
    
    def calculate_weighted_skill_match(
        self,
        job_required_skills: List[str],
        job_preferred_skills: List[str],
        candidate_skill_ids: List[str],
        strict_required: bool = True
    ) -> Tuple[float, Dict]:
        """
        Calculate weighted skill match with required/preferred distinction
        
        Returns: (score, details)
        """
        if not job_required_skills and not job_preferred_skills:
            return (0.0, {'required_matches': 0, 'preferred_matches': 0, 'total_required': 0})
        
        # Canonicalize job skills
        required_skill_ids = []
        for skill in job_required_skills:
            result = self.canonicalize_skill(skill)
            if result and result[0]:
                required_skill_ids.append(result[0])
        
        preferred_skill_ids = []
        for skill in job_preferred_skills or []:
            result = self.canonicalize_skill(skill)
            if result and result[0]:
                preferred_skill_ids.append(result[0])
        
        candidate_skill_set = set(candidate_skill_ids)
        
        # Strict exact match on required skills
        required_matches = len(set(required_skill_ids) & candidate_skill_set)
        required_total = len(required_skill_ids)
        
        # If strict mode and not all required skills match, return low score
        if strict_required and required_matches < required_total:
            return (0.0, {
                'required_matches': required_matches,
                'required_total': required_total,
                'strict_fail': True
            })
        
        # Weighted partial match from ontology
        preferred_matches = 0
        hierarchy_matches = 0
        
        # Check preferred skills
        for pref_skill_id in preferred_skill_ids:
            if pref_skill_id in candidate_skill_set:
                preferred_matches += 1
            else:
                # Check hierarchy
                skill = self.skill_ontology.get(pref_skill_id)
                if skill and skill.parent_skill_id:
                    if skill.parent_skill_id in candidate_skill_set:
                        hierarchy_matches += 0.5
                
                # Check children
                children = self.skill_hierarchy.get(pref_skill_id, [])
                for child_id in children:
                    if child_id in candidate_skill_set:
                        hierarchy_matches += 0.7
                        break
        
        # Calculate weighted score
        required_score = (required_matches / required_total) if required_total > 0 else 0.0
        preferred_score = (preferred_matches / len(preferred_skill_ids)) if preferred_skill_ids else 0.0
        hierarchy_score = hierarchy_matches / max(1, len(preferred_skill_ids) + required_total)
        
        # Weighted combination
        total_score = (
            required_score * 0.6 +  # Required skills are 60% weight
            preferred_score * 0.3 +  # Preferred skills are 30% weight
            hierarchy_score * 0.1    # Hierarchy matches are 10% weight
        )
        
        details = {
            'required_matches': required_matches,
            'required_total': required_total,
            'preferred_matches': preferred_matches,
            'preferred_total': len(preferred_skill_ids),
            'hierarchy_matches': hierarchy_matches,
            'required_score': required_score,
            'preferred_score': preferred_score,
            'hierarchy_score': hierarchy_score,
            'total_score': total_score
        }
        
        return (total_score, details)
    
    def get_skill_hierarchy_path(self, skill_id: str) -> List[str]:
        """Get full hierarchy path from root to skill"""
        path = []
        current_id = skill_id
        
        while current_id:
            skill = self.skill_ontology.get(current_id)
            if not skill:
                break
            path.insert(0, skill.canonical_name)
            current_id = skill.parent_skill_id
        
        return path
    
    def _create_comprehensive_ontology(self):
        """Create comprehensive skill ontology"""
        # This would be loaded from a comprehensive JSON file
        # For now, create directory structure
        ontology_dir = Path(self.ontology_path).parent
        ontology_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating comprehensive ontology - see data/production_skill_ontology.json")


# Global instance
_production_ontology = None

def get_production_ontology(ontology_path: Optional[str] = None) -> ProductionSkillOntology:
    """Get or create global production ontology instance"""
    global _production_ontology
    if _production_ontology is None:
        _production_ontology = ProductionSkillOntology(ontology_path)
    return _production_ontology

