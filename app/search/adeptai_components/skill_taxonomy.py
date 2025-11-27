"""
Skill Taxonomy and Proficiency Level System
===========================================

Provides hierarchical skill relationships, proficiency levels, and context-aware weighting
for improved candidate matching.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ProficiencyLevel(Enum):
    """Skill proficiency levels"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class SkillNode:
    """Represents a skill in the taxonomy"""
    name: str
    canonical_name: str
    aliases: List[str]
    category: str
    parent_skills: List[str]  # Broader skills
    child_skills: List[str]  # More specific skills
    related_skills: List[str]  # Similar/transferable skills
    proficiency_indicators: Dict[ProficiencyLevel, List[str]]  # Keywords indicating level
    industry_contexts: List[str]  # Industries where this skill is relevant


class SkillTaxonomy:
    """
    Hierarchical skill taxonomy with proficiency levels and context awareness.
    
    Features:
    - Skill hierarchy (parent-child relationships)
    - Skill aliases and canonicalization
    - Related/transferable skills
    - Proficiency level detection
    - Industry context awareness
    """
    
    def __init__(self, taxonomy_file: Optional[str] = None):
        """
        Initialize skill taxonomy.
        
        Args:
            taxonomy_file: Path to JSON file with skill taxonomy (optional)
        """
        self.skills: Dict[str, SkillNode] = {}
        self.alias_map: Dict[str, str] = {}  # alias -> canonical_name
        self.category_map: Dict[str, List[str]] = {}  # category -> [skill_names]
        
        # Load default taxonomy
        self._load_default_taxonomy()
        
        # Load from file if provided
        if taxonomy_file and os.path.exists(taxonomy_file):
            self.load_taxonomy(taxonomy_file)
    
    def _load_default_taxonomy(self):
        """Load default skill taxonomy with common skills"""
        
        # Programming Languages
        self._add_skill(
            name="Python",
            category="programming_language",
            aliases=["py", "python3", "python2"],
            parent_skills=["programming"],
            related_skills=["Django", "Flask", "FastAPI", "NumPy", "Pandas"],
            proficiency_indicators={
                ProficiencyLevel.BEGINNER: ["basic", "familiar", "learning"],
                ProficiencyLevel.INTERMEDIATE: ["experience", "working knowledge", "proficient"],
                ProficiencyLevel.ADVANCED: ["expertise", "deep knowledge", "advanced", "senior"],
                ProficiencyLevel.EXPERT: ["expert", "master", "architect", "lead", "specialist"]
            },
            industry_contexts=["technology", "data_science", "web_development", "automation"]
        )
        
        self._add_skill(
            name="JavaScript",
            category="programming_language",
            aliases=["js", "ecmascript", "nodejs", "node.js"],
            parent_skills=["programming"],
            related_skills=["TypeScript", "React", "Vue", "Angular", "Node.js"],
            proficiency_indicators={
                ProficiencyLevel.BEGINNER: ["basic", "familiar", "learning"],
                ProficiencyLevel.INTERMEDIATE: ["experience", "working knowledge", "proficient"],
                ProficiencyLevel.ADVANCED: ["expertise", "deep knowledge", "advanced", "senior"],
                ProficiencyLevel.EXPERT: ["expert", "master", "architect", "lead", "specialist"]
            },
            industry_contexts=["technology", "web_development", "frontend"]
        )
        
        self._add_skill(
            name="Java",
            category="programming_language",
            aliases=["java8", "java11", "jdk"],
            parent_skills=["programming"],
            related_skills=["Spring", "Spring Boot", "Hibernate", "Maven", "Gradle"],
            proficiency_indicators={
                ProficiencyLevel.BEGINNER: ["basic", "familiar", "learning"],
                ProficiencyLevel.INTERMEDIATE: ["experience", "working knowledge", "proficient"],
                ProficiencyLevel.ADVANCED: ["expertise", "deep knowledge", "advanced", "senior"],
                ProficiencyLevel.EXPERT: ["expert", "master", "architect", "lead", "specialist"]
            },
            industry_contexts=["technology", "enterprise", "backend"]
        )
        
        # Web Frameworks
        self._add_skill(
            name="React",
            category="web_framework",
            aliases=["reactjs", "react.js"],
            parent_skills=["JavaScript", "frontend"],
            related_skills=["Redux", "Next.js", "TypeScript", "JSX"],
            proficiency_indicators={
                ProficiencyLevel.BEGINNER: ["basic", "familiar", "learning"],
                ProficiencyLevel.INTERMEDIATE: ["experience", "working knowledge", "proficient"],
                ProficiencyLevel.ADVANCED: ["expertise", "hooks", "advanced patterns", "senior"],
                ProficiencyLevel.EXPERT: ["expert", "master", "architect", "lead", "specialist"]
            },
            industry_contexts=["technology", "web_development", "frontend"]
        )
        
        self._add_skill(
            name="Django",
            category="web_framework",
            aliases=["djangoframework"],
            parent_skills=["Python", "backend"],
            related_skills=["Flask", "FastAPI", "REST", "ORM"],
            proficiency_indicators={
                ProficiencyLevel.BEGINNER: ["basic", "familiar", "learning"],
                ProficiencyLevel.INTERMEDIATE: ["experience", "working knowledge", "proficient"],
                ProficiencyLevel.ADVANCED: ["expertise", "advanced patterns", "senior"],
                ProficiencyLevel.EXPERT: ["expert", "master", "architect", "lead", "specialist"]
            },
            industry_contexts=["technology", "web_development", "backend"]
        )
        
        # Cloud & DevOps
        self._add_skill(
            name="AWS",
            category="cloud_platform",
            aliases=["amazon web services", "amazon aws"],
            parent_skills=["cloud", "devops"],
            related_skills=["EC2", "S3", "Lambda", "Docker", "Kubernetes"],
            proficiency_indicators={
                ProficiencyLevel.BEGINNER: ["basic", "familiar", "learning"],
                ProficiencyLevel.INTERMEDIATE: ["experience", "working knowledge", "proficient"],
                ProficiencyLevel.ADVANCED: ["expertise", "certified", "advanced", "senior"],
                ProficiencyLevel.EXPERT: ["expert", "master", "architect", "solutions architect", "specialist"]
            },
            industry_contexts=["technology", "cloud", "devops"]
        )
        
        self._add_skill(
            name="Docker",
            category="devops_tool",
            aliases=["docker container", "dockerfile"],
            parent_skills=["devops", "containerization"],
            related_skills=["Kubernetes", "Docker Compose", "CI/CD"],
            proficiency_indicators={
                ProficiencyLevel.BEGINNER: ["basic", "familiar", "learning"],
                ProficiencyLevel.INTERMEDIATE: ["experience", "working knowledge", "proficient"],
                ProficiencyLevel.ADVANCED: ["expertise", "advanced", "senior"],
                ProficiencyLevel.EXPERT: ["expert", "master", "architect", "lead", "specialist"]
            },
            industry_contexts=["technology", "devops", "cloud"]
        )
        
        # Databases
        self._add_skill(
            name="PostgreSQL",
            category="database",
            aliases=["postgres", "postgre", "postgresql"],
            parent_skills=["database", "sql"],
            related_skills=["MySQL", "MongoDB", "Redis", "SQL"],
            proficiency_indicators={
                ProficiencyLevel.BEGINNER: ["basic", "familiar", "learning"],
                ProficiencyLevel.INTERMEDIATE: ["experience", "working knowledge", "proficient"],
                ProficiencyLevel.ADVANCED: ["expertise", "advanced", "senior", "optimization"],
                ProficiencyLevel.EXPERT: ["expert", "master", "DBA", "database architect", "specialist"]
            },
            industry_contexts=["technology", "backend", "data"]
        )
        
        # Machine Learning
        self._add_skill(
            name="TensorFlow",
            category="ml_framework",
            aliases=["tf", "tensorflow"],
            parent_skills=["machine_learning", "deep_learning"],
            related_skills=["PyTorch", "Keras", "scikit-learn", "NumPy"],
            proficiency_indicators={
                ProficiencyLevel.BEGINNER: ["basic", "familiar", "learning"],
                ProficiencyLevel.INTERMEDIATE: ["experience", "working knowledge", "proficient"],
                ProficiencyLevel.ADVANCED: ["expertise", "advanced", "senior", "research"],
                ProficiencyLevel.EXPERT: ["expert", "master", "ML engineer", "research scientist", "specialist"]
            },
            industry_contexts=["technology", "data_science", "ai"]
        )
        
        # Add more common skills
        self._add_skill(
            name="Git",
            category="version_control",
            aliases=["git version control", "github", "gitlab"],
            parent_skills=["version_control"],
            related_skills=["GitHub", "GitLab", "Bitbucket", "CI/CD"],
            proficiency_indicators={
                ProficiencyLevel.BEGINNER: ["basic", "familiar", "learning"],
                ProficiencyLevel.INTERMEDIATE: ["experience", "working knowledge", "proficient"],
                ProficiencyLevel.ADVANCED: ["expertise", "advanced", "senior"],
                ProficiencyLevel.EXPERT: ["expert", "master", "lead", "specialist"]
            },
            industry_contexts=["technology", "software_development"]
        )
        
        logger.info(f"Loaded default skill taxonomy with {len(self.skills)} skills")
    
    def _add_skill(
        self,
        name: str,
        category: str,
        aliases: List[str],
        parent_skills: List[str],
        related_skills: List[str],
        proficiency_indicators: Dict[ProficiencyLevel, List[str]],
        industry_contexts: List[str]
    ):
        """Add a skill to the taxonomy"""
        canonical_name = name.lower().strip()
        
        skill_node = SkillNode(
            name=name,
            canonical_name=canonical_name,
            aliases=aliases,
            category=category,
            parent_skills=parent_skills,
            child_skills=[],
            related_skills=related_skills,
            proficiency_indicators=proficiency_indicators,
            industry_contexts=industry_contexts
        )
        
        self.skills[canonical_name] = skill_node
        
        # Map aliases to canonical name
        for alias in aliases:
            self.alias_map[alias.lower()] = canonical_name
        
        # Add to category map
        if category not in self.category_map:
            self.category_map[category] = []
        self.category_map[category].append(canonical_name)
    
    def canonicalize_skill(self, skill: str) -> Optional[str]:
        """
        Convert skill name to canonical form.
        
        Args:
            skill: Skill name (may be alias or variant)
            
        Returns:
            Canonical skill name or None if not found
        """
        skill_lower = skill.lower().strip()
        
        # Direct match
        if skill_lower in self.skills:
            return skill_lower
        
        # Alias match
        if skill_lower in self.alias_map:
            return self.alias_map[skill_lower]
        
        # Fuzzy match (check if skill contains canonical name or vice versa)
        for canonical_name, skill_node in self.skills.items():
            if canonical_name in skill_lower or skill_lower in canonical_name:
                return canonical_name
        
        return None
    
    def get_related_skills(self, skill: str, include_parents: bool = True, include_children: bool = True) -> List[str]:
        """
        Get related skills (transferable, parent, child).
        
        Args:
            skill: Skill name
            include_parents: Include parent skills
            include_children: Include child skills
            
        Returns:
            List of related skill names
        """
        canonical = self.canonicalize_skill(skill)
        if not canonical:
            return []
        
        skill_node = self.skills[canonical]
        related = set(skill_node.related_skills)
        
        if include_parents:
            for parent in skill_node.parent_skills:
                parent_canonical = self.canonicalize_skill(parent)
                if parent_canonical:
                    related.add(parent_canonical)
        
        if include_children:
            for child in skill_node.child_skills:
                child_canonical = self.canonicalize_skill(child)
                if child_canonical:
                    related.add(child_canonical)
        
        return list(related)
    
    def detect_proficiency_level(self, skill: str, context: str) -> Tuple[ProficiencyLevel, float]:
        """
        Detect proficiency level from context text.
        
        Args:
            skill: Skill name
            context: Text context (resume, job description, etc.)
            
        Returns:
            Tuple of (proficiency_level, confidence)
        """
        canonical = self.canonicalize_skill(skill)
        if not canonical:
            return ProficiencyLevel.INTERMEDIATE, 0.5
        
        skill_node = self.skills[canonical]
        context_lower = context.lower()
        
        # Check for proficiency indicators
        scores = {}
        for level, indicators in skill_node.proficiency_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in context_lower)
            if matches > 0:
                scores[level] = matches / len(indicators)
        
        # Check for experience years
        years_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)', context_lower)
        if years_match:
            years = int(years_match.group(1))
            if years >= 7:
                scores[ProficiencyLevel.EXPERT] = scores.get(ProficiencyLevel.EXPERT, 0) + 0.5
            elif years >= 5:
                scores[ProficiencyLevel.ADVANCED] = scores.get(ProficiencyLevel.ADVANCED, 0) + 0.5
            elif years >= 3:
                scores[ProficiencyLevel.INTERMEDIATE] = scores.get(ProficiencyLevel.INTERMEDIATE, 0) + 0.5
        
        # Check for role titles
        expert_titles = ['expert', 'master', 'architect', 'lead', 'senior', 'principal', 'specialist']
        advanced_titles = ['senior', 'lead', 'advanced']
        if any(title in context_lower for title in expert_titles):
            scores[ProficiencyLevel.EXPERT] = scores.get(ProficiencyLevel.EXPERT, 0) + 0.3
        
        # Determine level
        if scores:
            best_level = max(scores.items(), key=lambda x: x[1])
            confidence = min(best_level[1], 1.0)
            return best_level[0], confidence
        
        # Default to intermediate if no indicators found
        return ProficiencyLevel.INTERMEDIATE, 0.5
    
    def calculate_skill_match_score(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        skill_weights: Optional[Dict[str, float]] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive skill match score with taxonomy and proficiency.
        
        Args:
            candidate_skills: List of candidate's skills
            required_skills: List of required skills
            skill_weights: Optional weights for each required skill (default: equal weights)
            context: Optional context text for proficiency detection
            
        Returns:
            Dictionary with match scores and details
        """
        if not required_skills:
            return {
                'overall_score': 0.0,
                'exact_matches': 0,
                'related_matches': 0,
                'missing_skills': required_skills.copy(),
                'skill_details': []
            }
        
        # Normalize weights
        if skill_weights is None:
            skill_weights = {skill: 1.0 for skill in required_skills}
        
        total_weight = sum(skill_weights.values())
        if total_weight > 0:
            skill_weights = {k: v / total_weight for k, v in skill_weights.items()}
        
        # Canonicalize all skills
        candidate_canonical = [self.canonicalize_skill(s) for s in candidate_skills if self.canonicalize_skill(s)]
        required_canonical = [self.canonicalize_skill(s) for s in required_skills if self.canonicalize_skill(s)]
        
        exact_matches = set()
        related_matches = {}
        missing_skills = []
        skill_details = []
        
        # Check each required skill
        for req_skill in required_canonical:
            if not req_skill:
                continue
            
            weight = skill_weights.get(req_skill, 1.0 / len(required_canonical))
            
            # Check for exact match
            if req_skill in candidate_canonical:
                exact_matches.add(req_skill)
                proficiency, confidence = self.detect_proficiency_level(req_skill, context or "")
                
                skill_details.append({
                    'skill': req_skill,
                    'match_type': 'exact',
                    'proficiency': proficiency.name,
                    'confidence': confidence,
                    'weight': weight,
                    'score': weight * (proficiency.value / 4.0)  # Normalize to 0-1
                })
            else:
                # Check for related/transferable skills
                related = self.get_related_skills(req_skill)
                found_related = False
                
                for rel_skill in related:
                    if rel_skill in candidate_canonical:
                        related_matches[req_skill] = rel_skill
                        proficiency, confidence = self.detect_proficiency_level(rel_skill, context or "")
                        
                        # Transferable skills get lower score (0.7x)
                        skill_details.append({
                            'skill': req_skill,
                            'match_type': 'related',
                            'matched_skill': rel_skill,
                            'proficiency': proficiency.name,
                            'confidence': confidence,
                            'weight': weight,
                            'score': weight * 0.7 * (proficiency.value / 4.0)
                        })
                        found_related = True
                        break
                
                if not found_related:
                    missing_skills.append(req_skill)
                    skill_details.append({
                        'skill': req_skill,
                        'match_type': 'missing',
                        'weight': weight,
                        'score': 0.0
                    })
        
        # Calculate overall score
        overall_score = sum(detail['score'] for detail in skill_details)
        
        return {
            'overall_score': overall_score,
            'exact_matches': len(exact_matches),
            'related_matches': len(related_matches),
            'missing_skills': missing_skills,
            'skill_details': skill_details,
            'match_percentage': (len(exact_matches) + len(related_matches)) / len(required_canonical) if required_canonical else 0.0
        }
    
    def get_skill_weight_for_context(
        self,
        skill: str,
        job_context: Dict[str, Any]
    ) -> float:
        """
        Get context-aware weight for a skill based on job context.
        
        Args:
            skill: Skill name
            job_context: Dictionary with job context (role, industry, company_size, etc.)
            
        Returns:
            Weight for the skill (0.0 to 1.0)
        """
        canonical = self.canonicalize_skill(skill)
        if not canonical:
            return 0.5  # Default weight
        
        skill_node = self.skills[canonical]
        
        # Base weight
        weight = 0.5
        
        # Industry context boost
        industry = job_context.get('industry', '').lower()
        if industry in [ctx.lower() for ctx in skill_node.industry_contexts]:
            weight += 0.2
        
        # Role type boost
        role = job_context.get('role_type', '').lower()
        if role in ['backend', 'backend developer'] and 'backend' in skill_node.parent_skills:
            weight += 0.2
        elif role in ['frontend', 'frontend developer'] and 'frontend' in skill_node.parent_skills:
            weight += 0.2
        elif role in ['fullstack', 'full stack']:
            weight += 0.1
        
        # Required vs nice-to-have
        if job_context.get('required', True):
            weight += 0.3
        
        # Company size context
        company_size = job_context.get('company_size', '').lower()
        if company_size == 'startup' and 'cloud' in skill_node.category:
            weight += 0.1  # Startups value cloud skills more
        
        return min(weight, 1.0)
    
    def save_taxonomy(self, filepath: str):
        """Save taxonomy to JSON file"""
        data = {
            'skills': {
                name: {
                    'name': node.name,
                    'canonical_name': node.canonical_name,
                    'aliases': node.aliases,
                    'category': node.category,
                    'parent_skills': node.parent_skills,
                    'child_skills': node.child_skills,
                    'related_skills': node.related_skills,
                    'proficiency_indicators': {
                        level.name: indicators
                        for level, indicators in node.proficiency_indicators.items()
                    },
                    'industry_contexts': node.industry_contexts
                }
                for name, node in self.skills.items()
            },
            'alias_map': self.alias_map,
            'category_map': self.category_map
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved taxonomy to {filepath}")
    
    def load_taxonomy(self, filepath: str):
        """Load taxonomy from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct skills
        for name, skill_data in data['skills'].items():
            proficiency_indicators = {
                ProficiencyLevel[level_name]: indicators
                for level_name, indicators in skill_data['proficiency_indicators'].items()
            }
            
            skill_node = SkillNode(
                name=skill_data['name'],
                canonical_name=skill_data['canonical_name'],
                aliases=skill_data['aliases'],
                category=skill_data['category'],
                parent_skills=skill_data['parent_skills'],
                child_skills=skill_data['child_skills'],
                related_skills=skill_data['related_skills'],
                proficiency_indicators=proficiency_indicators,
                industry_contexts=skill_data['industry_contexts']
            )
            
            self.skills[name] = skill_node
        
        self.alias_map = data.get('alias_map', {})
        self.category_map = data.get('category_map', {})
        
        logger.info(f"Loaded taxonomy from {filepath}")


# Global instance
_skill_taxonomy = None


def get_skill_taxonomy(taxonomy_file: Optional[str] = None) -> SkillTaxonomy:
    """Get or create global skill taxonomy instance"""
    global _skill_taxonomy
    if _skill_taxonomy is None:
        _skill_taxonomy = SkillTaxonomy(taxonomy_file=taxonomy_file)
    return _skill_taxonomy

