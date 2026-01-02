"""
Skill Canonicalization System
Maps skill synonyms, abbreviations, and variations to canonical skill IDs for accurate matching.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import difflib
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CanonicalSkill:
    """Canonical skill representation"""
    skill_id: str
    canonical_name: str
    category: str  # e.g., 'frontend', 'backend', 'database', 'cloud'
    level: str  # 'skill', 'framework', 'tool', 'language'
    parent_skill_id: Optional[str] = None
    aliases: List[str] = None
    synonyms: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.synonyms is None:
            self.synonyms = []


class SkillCanonicalizer:
    """Canonicalizes skills to normalized skill IDs"""
    
    def __init__(self, ontology_path: Optional[str] = None):
        self.ontology_path = ontology_path or self._get_default_ontology_path()
        self.skill_ontology: Dict[str, CanonicalSkill] = {}
        self.alias_to_skill_id: Dict[str, str] = {}
        self.skill_hierarchy: Dict[str, List[str]] = defaultdict(list)  # parent -> children
        self.embedding_model = None
        
        # Load ontology
        self._load_ontology()
        
        # Initialize embedding model for fuzzy matching
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded for skill matching")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
    
    def _get_default_ontology_path(self) -> str:
        """Get default path to skill ontology"""
        current_dir = Path(__file__).parent
        return str(current_dir / 'data' / 'skill_ontology.json')
    
    def _load_ontology(self):
        """Load skill ontology from JSON file"""
        try:
            if not Path(self.ontology_path).exists():
                logger.warning(f"Skill ontology not found at {self.ontology_path}, creating default")
                self._create_default_ontology()
                return
            
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Build ontology
            for skill_data in data.get('skills', []):
                skill = CanonicalSkill(
                    skill_id=skill_data['skill_id'],
                    canonical_name=skill_data['canonical_name'],
                    category=skill_data.get('category', 'general'),
                    level=skill_data.get('level', 'skill'),
                    parent_skill_id=skill_data.get('parent_skill_id'),
                    aliases=skill_data.get('aliases', []),
                    synonyms=skill_data.get('synonyms', [])
                )
                
                self.skill_ontology[skill.skill_id] = skill
                
                # Map aliases to skill_id
                for alias in skill.aliases + skill.synonyms + [skill.canonical_name]:
                    normalized_alias = self._normalize_skill_name(alias)
                    self.alias_to_skill_id[normalized_alias] = skill.skill_id
                
                # Build hierarchy
                if skill.parent_skill_id:
                    self.skill_hierarchy[skill.parent_skill_id].append(skill.skill_id)
            
            logger.info(f"Loaded {len(self.skill_ontology)} canonical skills")
            
        except Exception as e:
            logger.error(f"Error loading skill ontology: {e}")
            self._create_default_ontology()
    
    def _create_default_ontology(self):
        """Create default skill ontology with common tech skills"""
        default_skills = [
            {
                'skill_id': 'react',
                'canonical_name': 'React',
                'category': 'frontend',
                'level': 'framework',
                'parent_skill_id': 'javascript',
                'aliases': ['reactjs', 'react.js', 'reactjs', 'react native'],
                'synonyms': ['react framework', 'react library']
            },
            {
                'skill_id': 'javascript',
                'canonical_name': 'JavaScript',
                'category': 'frontend',
                'level': 'language',
                'aliases': ['js', 'ecmascript', 'node.js', 'nodejs'],
                'synonyms': ['javascript programming', 'js development']
            },
            {
                'skill_id': 'python',
                'canonical_name': 'Python',
                'category': 'backend',
                'level': 'language',
                'aliases': ['py', 'python3', 'python 3'],
                'synonyms': ['python programming', 'python development']
            },
            {
                'skill_id': 'aws',
                'canonical_name': 'AWS',
                'category': 'cloud',
                'level': 'platform',
                'aliases': ['amazon web services', 'amazon aws'],
                'synonyms': ['aws cloud', 'amazon cloud']
            },
            {
                'skill_id': 'docker',
                'canonical_name': 'Docker',
                'category': 'devops',
                'level': 'tool',
                'aliases': ['docker container', 'dockerization'],
                'synonyms': ['containerization', 'containers']
            },
            {
                'skill_id': 'kubernetes',
                'canonical_name': 'Kubernetes',
                'category': 'devops',
                'level': 'tool',
                'aliases': ['k8s', 'kube'],
                'synonyms': ['container orchestration', 'k8s orchestration']
            },
            {
                'skill_id': 'sql',
                'canonical_name': 'SQL',
                'category': 'database',
                'level': 'language',
                'aliases': ['structured query language'],
                'synonyms': ['sql database', 'sql queries']
            },
            {
                'skill_id': 'postgresql',
                'canonical_name': 'PostgreSQL',
                'category': 'database',
                'level': 'database',
                'parent_skill_id': 'sql',
                'aliases': ['postgres', 'postgresql'],
                'synonyms': ['postgres database']
            },
            {
                'skill_id': 'mongodb',
                'canonical_name': 'MongoDB',
                'category': 'database',
                'level': 'database',
                'aliases': ['mongo', 'mongodb'],
                'synonyms': ['nosql database', 'document database']
            },
            {
                'skill_id': 'java',
                'canonical_name': 'Java',
                'category': 'backend',
                'level': 'language',
                'aliases': ['java programming', 'java development'],
                'synonyms': ['java language', 'java platform']
            },
            {
                'skill_id': 'spring',
                'canonical_name': 'Spring',
                'category': 'backend',
                'level': 'framework',
                'parent_skill_id': 'java',
                'aliases': ['spring framework', 'spring boot'],
                'synonyms': ['spring java', 'spring mvc']
            },
        ]
        
        # Save default ontology
        ontology_dir = Path(self.ontology_path).parent
        ontology_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.ontology_path, 'w', encoding='utf-8') as f:
            json.dump({'skills': default_skills}, f, indent=2)
        
        # Load it
        self._load_ontology()
    
    def _normalize_skill_name(self, skill_name: str) -> str:
        """Normalize skill name for matching"""
        if not skill_name:
            return ''
        
        # Lowercase, strip, remove special chars
        normalized = skill_name.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        
        return normalized
    
    def canonicalize_skill(self, raw_skill: str, threshold: float = 0.85) -> Optional[Tuple[str, str, float]]:
        """
        Canonicalize a raw skill to skill_id
        
        Returns: (skill_id, canonical_name, confidence) or None if no match
        """
        if not raw_skill:
            return None
        
        normalized = self._normalize_skill_name(raw_skill)
        
        # Exact match first
        if normalized in self.alias_to_skill_id:
            skill_id = self.alias_to_skill_id[normalized]
            skill = self.skill_ontology[skill_id]
            return (skill_id, skill.canonical_name, 1.0)
        
        # Fuzzy string matching
        best_match = None
        best_score = 0.0
        
        for alias, skill_id in self.alias_to_skill_id.items():
            # Use SequenceMatcher for fuzzy matching
            similarity = difflib.SequenceMatcher(None, normalized, alias).ratio()
            
            # Boost exact substring matches
            if normalized in alias or alias in normalized:
                similarity = max(similarity, 0.9)
            
            if similarity > best_score:
                best_score = similarity
                best_match = (skill_id, alias)
        
        # Embedding-based matching if available
        if self.embedding_model and best_score < threshold:
            embedding_match = self._embedding_match(normalized, threshold)
            if embedding_match and embedding_match[2] > best_score:
                return embedding_match
        
        if best_match and best_score >= threshold:
            skill_id, matched_alias = best_match
            skill = self.skill_ontology[skill_id]
            return (skill_id, skill.canonical_name, best_score)
        
        return None
    
    def _embedding_match(self, normalized_skill: str, threshold: float) -> Optional[Tuple[str, str, float]]:
        """Match skill using embeddings"""
        if not self.embedding_model:
            return None
        
        try:
            # Get embeddings for all canonical skills
            canonical_names = [skill.canonical_name for skill in self.skill_ontology.values()]
            canonical_embeddings = self.embedding_model.encode(canonical_names)
            query_embedding = self.embedding_model.encode([normalized_skill])
            
            # Calculate cosine similarity
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarities = cosine_similarity(query_embedding, canonical_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = float(similarities[best_idx])
            
            if best_score >= threshold:
                skill_id = list(self.skill_ontology.keys())[best_idx]
                skill = self.skill_ontology[skill_id]
                return (skill_id, skill.canonical_name, best_score)
        
        except Exception as e:
            logger.error(f"Error in embedding match: {e}")
        
        return None
    
    def canonicalize_skill_list(self, raw_skills: List[str], threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        """Canonicalize a list of raw skills"""
        canonicalized = []
        
        for raw_skill in raw_skills:
            if not raw_skill:
                continue
            
            result = self.canonicalize_skill(raw_skill, threshold)
            if result:
                canonicalized.append(result)
            else:
                # Keep original if no match found
                normalized = self._normalize_skill_name(raw_skill)
                canonicalized.append((None, raw_skill, 0.0))
        
        return canonicalized
    
    def get_skill_hierarchy(self, skill_id: str) -> List[str]:
        """Get child skills in hierarchy"""
        return self.skill_hierarchy.get(skill_id, [])
    
    def get_parent_skill(self, skill_id: str) -> Optional[str]:
        """Get parent skill ID"""
        skill = self.skill_ontology.get(skill_id)
        return skill.parent_skill_id if skill else None
    
    def calculate_skill_match_score(self, job_skill_ids: List[str], candidate_skill_ids: List[str]) -> Tuple[float, Dict]:
        """
        Calculate weighted skill match score with hierarchy awareness
        
        Returns: (score, details)
        """
        if not job_skill_ids:
            return (0.0, {'exact_matches': 0, 'hierarchy_matches': 0, 'total_required': 0})
        
        job_skill_set = set(job_skill_ids)
        candidate_skill_set = set(candidate_skill_ids)
        
        exact_matches = len(job_skill_set & candidate_skill_set)
        
        # Check hierarchy matches (parent-child relationships)
        hierarchy_matches = 0
        for job_skill_id in job_skill_set:
            if job_skill_id in candidate_skill_set:
                continue  # Already counted as exact match
            
            # Check if candidate has child skills
            child_skills = self.get_skill_hierarchy(job_skill_id)
            if any(child in candidate_skill_set for child in child_skills):
                hierarchy_matches += 0.7  # Partial credit for hierarchy match
            
            # Check if candidate has parent skill
            parent_skill = self.get_parent_skill(job_skill_id)
            if parent_skill and parent_skill in candidate_skill_set:
                hierarchy_matches += 0.5  # Less credit for parent match
        
        total_matches = exact_matches + hierarchy_matches
        score = min(1.0, total_matches / len(job_skill_ids))
        
        details = {
            'exact_matches': exact_matches,
            'hierarchy_matches': round(hierarchy_matches, 2),
            'total_required': len(job_skill_ids),
            'match_ratio': score
        }
        
        return (score, details)


# Global instance
_skill_canonicalizer = None

def get_skill_canonicalizer(ontology_path: Optional[str] = None) -> SkillCanonicalizer:
    """Get or create global skill canonicalizer instance"""
    global _skill_canonicalizer
    if _skill_canonicalizer is None:
        _skill_canonicalizer = SkillCanonicalizer(ontology_path)
    return _skill_canonicalizer

def canonicalize_skills(raw_skills: List[str], threshold: float = 0.85) -> List[Tuple[str, str, float]]:
    """Convenience function to canonicalize skills"""
    canonicalizer = get_skill_canonicalizer()
    return canonicalizer.canonicalize_skill_list(raw_skills, threshold)

