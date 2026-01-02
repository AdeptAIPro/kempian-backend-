"""
Context-Aware Skill Weighting System
=====================================

Provides intelligent skill weighting based on job context, role type, industry, and other factors.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


class RoleType(Enum):
    """Role type categories"""
    BACKEND = "backend"
    FRONTEND = "frontend"
    FULLSTACK = "fullstack"
    DEVOPS = "devops"
    DATA_SCIENCE = "data_science"
    ML_ENGINEER = "ml_engineer"
    MOBILE = "mobile"
    QA = "qa"
    SECURITY = "security"
    ARCHITECT = "architect"
    MANAGER = "manager"
    UNKNOWN = "unknown"


class CompanySize(Enum):
    """Company size categories"""
    STARTUP = "startup"  # < 50
    SMALL = "small"  # 50-200
    MEDIUM = "medium"  # 200-1000
    LARGE = "large"  # 1000-5000
    ENTERPRISE = "enterprise"  # > 5000


@dataclass
class JobContext:
    """Job context information"""
    role_type: RoleType
    industry: str
    company_size: CompanySize
    required_skills: List[str]
    nice_to_have_skills: List[str]
    experience_level: str  # junior, mid, senior, lead
    project_type: Optional[str] = None  # greenfield, legacy, migration
    team_size: Optional[int] = None
    remote: bool = False
    urgency: str = "normal"  # urgent, normal, exploratory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'role_type': self.role_type.value,
            'industry': self.industry,
            'company_size': self.company_size.value,
            'required_skills': self.required_skills,
            'nice_to_have_skills': self.nice_to_have_skills,
            'experience_level': self.experience_level,
            'project_type': self.project_type,
            'team_size': self.team_size,
            'remote': self.remote,
            'urgency': self.urgency
        }


class ContextAwareWeighting:
    """
    Context-aware skill weighting system.
    
    Features:
    - Role-specific skill weights
    - Industry context awareness
    - Company size considerations
    - Project type adjustments
    - Experience level matching
    - Required vs nice-to-have distinction
    """
    
    def __init__(self):
        """Initialize context-aware weighting system"""
        self._init_role_skill_weights()
        self._init_industry_weights()
        self._init_company_size_weights()
        self._init_project_type_weights()
    
    def _init_role_skill_weights(self):
        """Initialize role-specific skill weights"""
        self.role_weights = {
            RoleType.BACKEND: {
                'programming_language': 0.25,
                'web_framework': 0.20,
                'database': 0.20,
                'api': 0.15,
                'cloud_platform': 0.10,
                'devops_tool': 0.10
            },
            RoleType.FRONTEND: {
                'programming_language': 0.20,
                'web_framework': 0.30,
                'ui_library': 0.20,
                'css_framework': 0.15,
                'build_tool': 0.10,
                'testing': 0.05
            },
            RoleType.FULLSTACK: {
                'programming_language': 0.20,
                'web_framework': 0.25,
                'database': 0.15,
                'api': 0.15,
                'cloud_platform': 0.10,
                'devops_tool': 0.10,
                'ui_library': 0.05
            },
            RoleType.DEVOPS: {
                'cloud_platform': 0.30,
                'devops_tool': 0.30,
                'containerization': 0.20,
                'ci_cd': 0.15,
                'monitoring': 0.05
            },
            RoleType.DATA_SCIENCE: {
                'ml_framework': 0.25,
                'programming_language': 0.20,
                'data_analysis': 0.20,
                'database': 0.15,
                'visualization': 0.10,
                'cloud_platform': 0.10
            },
            RoleType.ML_ENGINEER: {
                'ml_framework': 0.30,
                'programming_language': 0.20,
                'deep_learning': 0.20,
                'data_processing': 0.15,
                'cloud_platform': 0.10,
                'mlops': 0.05
            }
        }
    
    def _init_industry_weights(self):
        """Initialize industry-specific skill weights"""
        self.industry_weights = {
            'fintech': {
                'security': 0.20,
                'compliance': 0.15,
                'database': 0.15,
                'api': 0.15,
                'cloud_platform': 0.10
            },
            'healthcare': {
                'security': 0.25,
                'compliance': 0.20,
                'data_privacy': 0.15,
                'database': 0.15,
                'api': 0.10
            },
            'ecommerce': {
                'web_framework': 0.25,
                'database': 0.20,
                'payment': 0.15,
                'cloud_platform': 0.15,
                'performance': 0.10
            },
            'startup': {
                'programming_language': 0.20,
                'web_framework': 0.20,
                'cloud_platform': 0.20,
                'fullstack': 0.15,
                'rapid_prototyping': 0.10
            }
        }
    
    def _init_company_size_weights(self):
        """Initialize company size-specific weights"""
        self.company_size_weights = {
            CompanySize.STARTUP: {
                'fullstack': 0.20,
                'cloud_platform': 0.20,
                'rapid_prototyping': 0.15,
                'generalist': 0.15
            },
            CompanySize.ENTERPRISE: {
                'security': 0.20,
                'compliance': 0.15,
                'scalability': 0.15,
                'enterprise_tools': 0.15
            }
        }
    
    def _init_project_type_weights(self):
        """Initialize project type-specific weights"""
        self.project_type_weights = {
            'greenfield': {
                'modern_frameworks': 0.20,
                'best_practices': 0.15,
                'cloud_native': 0.15
            },
            'legacy': {
                'legacy_systems': 0.25,
                'migration': 0.20,
                'maintenance': 0.15
            },
            'migration': {
                'migration': 0.30,
                'cloud_platform': 0.20,
                'modernization': 0.15
            }
        }
    
    def detect_role_type(self, query: str, job_description: Optional[str] = None) -> RoleType:
        """
        Detect role type from query or job description.
        
        Args:
            query: Search query
            job_description: Optional job description text
            
        Returns:
            Detected role type
        """
        text = (query + " " + (job_description or "")).lower()
        
        # Check for role type keywords
        if any(kw in text for kw in ['backend', 'back-end', 'server', 'api', 'server-side']):
            if any(kw in text for kw in ['frontend', 'front-end', 'fullstack', 'full stack']):
                return RoleType.FULLSTACK
            return RoleType.BACKEND
        
        if any(kw in text for kw in ['frontend', 'front-end', 'ui', 'ux', 'react', 'vue', 'angular']):
            return RoleType.FRONTEND
        
        if any(kw in text for kw in ['devops', 'dev ops', 'sre', 'infrastructure', 'deployment']):
            return RoleType.DEVOPS
        
        if any(kw in text for kw in ['data scientist', 'data science', 'analyst', 'analytics']):
            return RoleType.DATA_SCIENCE
        
        if any(kw in text for kw in ['ml engineer', 'machine learning', 'ai engineer', 'deep learning']):
            return RoleType.ML_ENGINEER
        
        if any(kw in text for kw in ['mobile', 'ios', 'android', 'react native', 'flutter']):
            return RoleType.MOBILE
        
        if any(kw in text for kw in ['qa', 'quality assurance', 'testing', 'test engineer']):
            return RoleType.QA
        
        if any(kw in text for kw in ['security', 'cybersecurity', 'infosec']):
            return RoleType.SECURITY
        
        if any(kw in text for kw in ['architect', 'architecture', 'solution architect']):
            return RoleType.ARCHITECT
        
        if any(kw in text for kw in ['manager', 'lead', 'director', 'head of']):
            return RoleType.MANAGER
        
        return RoleType.UNKNOWN
    
    def detect_company_size(self, query: str, job_description: Optional[str] = None) -> CompanySize:
        """
        Detect company size from query or job description.
        
        Args:
            query: Search query
            job_description: Optional job description text
            
        Returns:
            Detected company size
        """
        text = (query + " " + (job_description or "")).lower()
        
        # Check for company size indicators
        if any(kw in text for kw in ['startup', 'early stage', 'seed', 'series a']):
            return CompanySize.STARTUP
        
        if any(kw in text for kw in ['enterprise', 'fortune 500', 'large corporation', 'multinational']):
            return CompanySize.ENTERPRISE
        
        if any(kw in text for kw in ['small company', 'small team', 'boutique']):
            return CompanySize.SMALL
        
        if any(kw in text for kw in ['medium', 'mid-size', 'growing company']):
            return CompanySize.MEDIUM
        
        # Default based on other indicators
        if any(kw in text for kw in ['scalable', 'high volume', 'millions of users']):
            return CompanySize.LARGE
        
        return CompanySize.MEDIUM  # Default
    
    def detect_experience_level(self, query: str) -> str:
        """
        Detect experience level from query.
        
        Args:
            query: Search query
            
        Returns:
            Experience level (junior, mid, senior, lead)
        """
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['lead', 'principal', 'architect', 'director', 'head']):
            return 'lead'
        
        if any(kw in query_lower for kw in ['senior', 'sr', 'experienced', '5+', '7+']):
            return 'senior'
        
        if any(kw in query_lower for kw in ['junior', 'jr', 'entry', 'graduate', 'intern']):
            return 'junior'
        
        return 'mid'  # Default
    
    def extract_job_context(self, query: str, job_description: Optional[str] = None) -> JobContext:
        """
        Extract job context from query and job description.
        
        Args:
            query: Search query
            job_description: Optional job description text
            
        Returns:
            JobContext object
        """
        role_type = self.detect_role_type(query, job_description)
        company_size = self.detect_company_size(query, job_description)
        experience_level = self.detect_experience_level(query)
        
        # Extract skills (basic extraction - can be enhanced)
        text = (query + " " + (job_description or "")).lower()
        
        # Detect project type
        project_type = None
        if any(kw in text for kw in ['greenfield', 'new project', 'from scratch']):
            project_type = 'greenfield'
        elif any(kw in text for kw in ['legacy', 'maintenance', 'existing system']):
            project_type = 'legacy'
        elif any(kw in text for kw in ['migration', 'migrate', 'modernization']):
            project_type = 'migration'
        
        # Detect remote
        remote = any(kw in text for kw in ['remote', 'work from home', 'wfh', 'distributed'])
        
        # Detect urgency
        urgency = 'normal'
        if any(kw in text for kw in ['urgent', 'asap', 'immediate', 'quick']):
            urgency = 'urgent'
        elif any(kw in text for kw in ['exploratory', 'research', 'explore']):
            urgency = 'exploratory'
        
        # Extract required vs nice-to-have skills (simplified)
        required_skills = []
        nice_to_have_skills = []
        
        # This is a simplified extraction - in production, use NLP/LLM
        # For now, assume all mentioned skills are required unless marked as "nice to have"
        if 'nice to have' in text or 'preferred' in text:
            # Split skills into required and nice-to-have
            # This would need more sophisticated parsing
            pass
        
        return JobContext(
            role_type=role_type,
            industry='technology',  # Default - can be enhanced with industry detection
            company_size=company_size,
            required_skills=required_skills,
            nice_to_have_skills=nice_to_have_skills,
            experience_level=experience_level,
            project_type=project_type,
            remote=remote,
            urgency=urgency
        )
    
    def calculate_skill_weights(
        self,
        skills: List[str],
        job_context: JobContext
    ) -> Dict[str, float]:
        """
        Calculate context-aware weights for skills.
        
        Args:
            skills: List of skills
            job_context: Job context information
            
        Returns:
            Dictionary mapping skill -> weight
        """
        weights = {}
        
        # Base weights
        base_weight = 1.0 / len(skills) if skills else 0.0
        
        for skill in skills:
            weight = base_weight
            
            # Role type adjustment
            if job_context.role_type in self.role_weights:
                role_weights = self.role_weights[job_context.role_type]
                # Check if skill category matches role preferences
                # This is simplified - in production, map skills to categories
                skill_lower = skill.lower()
                
                # Adjust based on role preferences
                if job_context.role_type == RoleType.BACKEND:
                    if any(kw in skill_lower for kw in ['python', 'java', 'node', 'api', 'database', 'sql']):
                        weight *= 1.3
                elif job_context.role_type == RoleType.FRONTEND:
                    if any(kw in skill_lower for kw in ['react', 'vue', 'angular', 'javascript', 'css', 'html']):
                        weight *= 1.3
                elif job_context.role_type == RoleType.DEVOPS:
                    if any(kw in skill_lower for kw in ['aws', 'docker', 'kubernetes', 'ci/cd', 'terraform']):
                        weight *= 1.3
            
            # Industry adjustment
            if job_context.industry in self.industry_weights:
                industry_weights = self.industry_weights[job_context.industry]
                skill_lower = skill.lower()
                
                if job_context.industry == 'fintech' and any(kw in skill_lower for kw in ['security', 'compliance', 'encryption']):
                    weight *= 1.2
                elif job_context.industry == 'healthcare' and any(kw in skill_lower for kw in ['security', 'compliance', 'hipaa']):
                    weight *= 1.2
            
            # Company size adjustment
            if job_context.company_size in self.company_size_weights:
                company_weights = self.company_size_weights[job_context.company_size]
                skill_lower = skill.lower()
                
                if job_context.company_size == CompanySize.STARTUP:
                    if any(kw in skill_lower for kw in ['fullstack', 'cloud', 'aws', 'docker']):
                        weight *= 1.2
                elif job_context.company_size == CompanySize.ENTERPRISE:
                    if any(kw in skill_lower for kw in ['security', 'compliance', 'scalability']):
                        weight *= 1.2
            
            # Project type adjustment
            if job_context.project_type and job_context.project_type in self.project_type_weights:
                project_weights = self.project_type_weights[job_context.project_type]
                skill_lower = skill.lower()
                
                if job_context.project_type == 'greenfield':
                    if any(kw in skill_lower for kw in ['react', 'vue', 'modern', 'cloud']):
                        weight *= 1.1
                elif job_context.project_type == 'legacy':
                    if any(kw in skill_lower for kw in ['java', 'cobol', 'mainframe', 'legacy']):
                        weight *= 1.2
            
            # Required vs nice-to-have
            if skill in job_context.required_skills:
                weight *= 1.5  # Required skills get higher weight
            elif skill in job_context.nice_to_have_skills:
                weight *= 0.7  # Nice-to-have skills get lower weight
            
            # Experience level adjustment
            if job_context.experience_level == 'senior' or job_context.experience_level == 'lead':
                # Senior roles value advanced skills more
                if any(kw in skill_lower for kw in ['architecture', 'design', 'leadership', 'mentoring']):
                    weight *= 1.2
            
            weights[skill] = weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_weighted_skill_score(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        job_context: JobContext
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate weighted skill match score with context awareness.
        
        Args:
            candidate_skills: List of candidate's skills
            required_skills: List of required skills
            job_context: Job context information
            
        Returns:
            Tuple of (weighted_score, details_dict)
        """
        # Calculate context-aware weights
        skill_weights = self.calculate_skill_weights(required_skills, job_context)
        
        # Calculate match scores (this would use skill taxonomy)
        # For now, simplified matching
        matches = {}
        for req_skill in required_skills:
            weight = skill_weights.get(req_skill, 0.0)
            
            # Check for exact match
            if req_skill.lower() in [s.lower() for s in candidate_skills]:
                matches[req_skill] = {
                    'matched': True,
                    'weight': weight,
                    'score': weight * 1.0
                }
            else:
                # Check for partial/fuzzy match
                req_lower = req_skill.lower()
                for cand_skill in candidate_skills:
                    cand_lower = cand_skill.lower()
                    if req_lower in cand_lower or cand_lower in req_lower:
                        matches[req_skill] = {
                            'matched': True,
                            'matched_skill': cand_skill,
                            'weight': weight,
                            'score': weight * 0.7  # Partial match gets 0.7x
                        }
                        break
                else:
                    matches[req_skill] = {
                        'matched': False,
                        'weight': weight,
                        'score': 0.0
                    }
        
        # Calculate overall weighted score
        overall_score = sum(match['score'] for match in matches.values())
        
        return overall_score, {
            'overall_score': overall_score,
            'matches': matches,
            'skill_weights': skill_weights,
            'context': job_context.to_dict()
        }


# Global instance
_context_weighting = None


def get_context_weighting() -> ContextAwareWeighting:
    """Get or create global context-aware weighting instance"""
    global _context_weighting
    if _context_weighting is None:
        _context_weighting = ContextAwareWeighting()
    return _context_weighting

