"""
Job description parsing and normalization.
Extracts structured requirements from job descriptions.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from ..utils.text_cleaner import TextCleaner
from .skill_extractor import SkillExtractor

logger = logging.getLogger(__name__)


@dataclass
class JobData:
    """Structured job data."""
    job_id: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_years_required: float
    education_required: Optional[str]
    job_description: str


class JobParser:
    """Parse and extract structured requirements from job descriptions."""
    
    def __init__(self, skill_extractor: Optional[SkillExtractor] = None):
        """
        Initialize job parser.
        
        Args:
            skill_extractor: Skill extractor instance (creates new if None)
        """
        self.skill_extractor = skill_extractor or SkillExtractor()
        self.text_cleaner = TextCleaner()
        
        # Patterns for required skills
        self.required_patterns = [
            r'(?:required|must have|must|essential|mandatory|necessary)\s*:?\s*([^\.]+)',
            r'(?:must|should)\s+(?:have|know|be)\s+(?:experience|knowledge|skills?)\s+(?:in|with|of)\s+([^\.]+)',
            r'(?:looking for|seeking)\s+([^\.]+)',
        ]
        
        # Patterns for preferred skills
        self.preferred_patterns = [
            r'(?:preferred|nice to have|bonus|plus|advantage|desirable)\s*:?\s*([^\.]+)',
            r'(?:would be|is)\s+(?:a\s+)?(?:plus|advantage|bonus)\s+([^\.]+)',
        ]
        
        # Experience requirement patterns
        self.experience_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+(?:\.\d+)?)\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
            r'minimum\s+(?:of\s*)?(\d+(?:\.\d+)?)\s*years?',
            r'at\s+least\s+(\d+(?:\.\d+)?)\s*years?',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        ]
        
        # Education requirement patterns
        self.education_patterns = [
            r'(?:bachelor|master|phd|doctorate|degree|bs|ms|mba|ba|ma|bsc|msc)',
            r'(?:bachelor\'?s|master\'?s|doctorate)',
        ]
    
    def parse(self, job_data: Dict[str, Any]) -> JobData:
        """
        Parse job data into structured JobData.
        
        Args:
            job_data: Dictionary containing job information
                Expected keys: job_id, job_description, description, etc.
        
        Returns:
            Structured JobData object
        """
        try:
            # Extract job ID
            job_id = str(job_data.get('job_id') or 
                        job_data.get('id') or 
                        job_data.get('_id') or 
                        'unknown')
            
            # Extract job description
            job_description = self._extract_job_description(job_data)
            
            # Extract required and preferred skills
            required_skills, preferred_skills = self._extract_skills(job_description)
            
            # Extract experience requirement
            experience_years_required = self._extract_experience_requirement(job_description)
            
            # Extract education requirement
            education_required = self._extract_education_requirement(job_description)
            
            return JobData(
                job_id=job_id,
                required_skills=required_skills,
                preferred_skills=preferred_skills,
                experience_years_required=experience_years_required,
                education_required=education_required,
                job_description=job_description
            )
        except Exception as e:
            logger.error(f"Error parsing job description: {e}")
            # Return minimal valid JobData
            return JobData(
                job_id=str(job_data.get('job_id', 'unknown')),
                required_skills=[],
                preferred_skills=[],
                experience_years_required=0.0,
                education_required=None,
                job_description=''
            )
    
    def _extract_job_description(self, job_data: Dict[str, Any]) -> str:
        """Extract job description text."""
        desc_fields = [
            'job_description', 'jobDescription', 'JobDescription',
            'description', 'Description', 'job_desc', 'jobDesc',
            'requirements', 'Requirements', 'qualifications', 'Qualifications'
        ]
        
        text_parts = []
        for field in desc_fields:
            value = job_data.get(field)
            if value:
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, list):
                    text_parts.append(' '.join(str(v) for v in value))
        
        # Combine all text parts
        combined_text = ' '.join(text_parts)
        return self.text_cleaner.clean(combined_text)
    
    def _extract_skills(self, job_description: str) -> Tuple[List[str], List[str]]:
        """
        Extract required and preferred skills from job description.
        
        Args:
            job_description: Job description text
            
        Returns:
            Tuple of (required_skills, preferred_skills)
        """
        required_skills = []
        preferred_skills = []
        
        # Split job description into sections
        sections = re.split(r'[.!?]\s+', job_description)
        
        for section in sections:
            section_lower = section.lower()
            
            # Check if section contains required keywords
            is_required = any(keyword in section_lower for keyword in 
                            ['required', 'must have', 'must', 'essential', 'mandatory'])
            
            # Check if section contains preferred keywords
            is_preferred = any(keyword in section_lower for keyword in 
                             ['preferred', 'nice to have', 'bonus', 'plus', 'advantage'])
            
            # Extract skills from section using skill extractor
            extracted = self.skill_extractor.extract_skills(section)
            skill_names = [skill.canonical_name for skill in extracted]
            
            if is_required:
                required_skills.extend(skill_names)
            elif is_preferred:
                preferred_skills.extend(skill_names)
            else:
                # Default to required if no indicator
                required_skills.extend(skill_names)
        
        # Also extract all skills from full description
        all_extracted = self.skill_extractor.extract_skills(job_description)
        all_skill_names = [skill.canonical_name for skill in all_extracted]
        
        # Merge and deduplicate
        required_set = set(required_skills)
        preferred_set = set(preferred_skills)
        
        # Add skills not already categorized
        for skill in all_skill_names:
            if skill not in required_set and skill not in preferred_set:
                required_set.add(skill)  # Default to required
        
        return list(required_set), list(preferred_set)
    
    def _extract_experience_requirement(self, job_description: str) -> float:
        """Extract experience years requirement from job description."""
        if not job_description:
            return 0.0
        
        years_found = []
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE)
            for match in matches:
                try:
                    years = float(match)
                    years_found.append(years)
                except (ValueError, TypeError):
                    continue
        
        if years_found:
            # Return maximum requirement found
            return max(years_found)
        
        return 0.0
    
    def _extract_education_requirement(self, job_description: str) -> Optional[str]:
        """Extract education requirement from job description."""
        if not job_description:
            return None
        
        normalized_text = job_description.lower()
        
        # Check for education keywords
        for pattern in self.education_patterns:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                # Extract sentence containing education requirement
                sentences = re.split(r'[.!?]\s+', job_description)
                for sentence in sentences:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        return sentence.strip()
        
        return None

