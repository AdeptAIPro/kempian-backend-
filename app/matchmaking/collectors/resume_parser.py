"""
Resume parsing and normalization.
Extracts structured data from resume text.
"""

import re
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..utils.text_cleaner import TextCleaner
from .skill_extractor import SkillExtractor

logger = logging.getLogger(__name__)


@dataclass
class ResumeData:
    """Structured resume data."""
    candidate_id: str
    full_name: str
    email: Optional[str]
    skills: List[str]
    experience_years: float
    education: Optional[str]
    certifications: List[str]
    resume_text: str


class ResumeParser:
    """Parse and extract structured data from resumes."""
    
    def __init__(self, skill_extractor: Optional[SkillExtractor] = None):
        """
        Initialize resume parser.
        
        Args:
            skill_extractor: Skill extractor instance (creates new if None)
        """
        self.skill_extractor = skill_extractor or SkillExtractor()
        self.text_cleaner = TextCleaner()
        
        # Experience patterns
        self.experience_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+(?:\.\d+)?)\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
            r'experience[:\s]+(\d+(?:\.\d+)?)\s*years?',
            r'(\d+)\+?\s*years?\s*(?:in|with)',
        ]
        
        # Education patterns
        self.education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'degree',
            'bs', 'ms', 'mba', 'ba', 'ma', 'bsc', 'msc',
            'university', 'college', 'education'
        ]
        
        # Certification patterns
        self.certification_keywords = [
            'certified', 'certification', 'certificate', 'cert',
            'aws certified', 'azure certified', 'google cloud',
            'pmp', 'scrum', 'agile'
        ]
    
    def parse(self, candidate_data: Dict[str, Any]) -> ResumeData:
        """
        Parse candidate data into structured ResumeData.
        
        Args:
            candidate_data: Dictionary containing candidate information
                Expected keys: candidate_id, full_name/name, email, resume_text,
                skills, experience, education, etc.
        
        Returns:
            Structured ResumeData object
        """
        try:
            # Extract candidate ID
            candidate_id = str(candidate_data.get('candidate_id') or 
                             candidate_data.get('id') or 
                             candidate_data.get('_id') or 
                             'unknown')
            
            # Extract full name
            full_name = self._extract_name(candidate_data)
            
            # Extract email
            email = self._extract_email(candidate_data)
            
            # Extract resume text
            resume_text = self._extract_resume_text(candidate_data)
            
            # Extract skills
            skills = self._extract_skills(candidate_data, resume_text)
            
            # Extract experience years
            experience_years = self._extract_experience_years(candidate_data, resume_text)
            
            # Extract education
            education = self._extract_education(candidate_data, resume_text)
            
            # Extract certifications
            certifications = self._extract_certifications(candidate_data, resume_text)
            
            return ResumeData(
                candidate_id=candidate_id,
                full_name=full_name,
                email=email,
                skills=skills,
                experience_years=experience_years,
                education=education,
                certifications=certifications,
                resume_text=resume_text
            )
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            # Return minimal valid ResumeData
            return ResumeData(
                candidate_id=str(candidate_data.get('candidate_id', 'unknown')),
                full_name='',
                email=None,
                skills=[],
                experience_years=0.0,
                education=None,
                certifications=[],
                resume_text=''
            )
    
    def _extract_name(self, candidate_data: Dict[str, Any]) -> str:
        """Extract full name from candidate data."""
        name_fields = ['full_name', 'name', 'FullName', 'fullName', 'candidate_name']
        for field in name_fields:
            name = candidate_data.get(field)
            if name and isinstance(name, str) and name.strip():
                return name.strip()
        return ''
    
    def _extract_email(self, candidate_data: Dict[str, Any]) -> Optional[str]:
        """Extract email from candidate data."""
        email_fields = ['email', 'Email', 'email_address', 'e_mail']
        for field in email_fields:
            email = candidate_data.get(field)
            if email and isinstance(email, str):
                extracted = self.text_cleaner.extract_email(email)
                if extracted:
                    return extracted
        
        # Try extracting from resume text
        resume_text = self._extract_resume_text(candidate_data)
        if resume_text:
            extracted = self.text_cleaner.extract_email(resume_text)
            if extracted:
                return extracted
        
        return None
    
    def _extract_resume_text(self, candidate_data: Dict[str, Any]) -> str:
        """Extract resume text from candidate data."""
        text_fields = [
            'resume_text', 'resumeText', 'ResumeText', 'resume',
            'Resume', 'summary', 'Summary', 'profile_summary',
            'profileSummary', 'bio', 'Bio', 'description', 'Description',
            'experience', 'Experience', 'work_experience'
        ]
        
        text_parts = []
        for field in text_fields:
            value = candidate_data.get(field)
            if value:
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, list):
                    text_parts.append(' '.join(str(v) for v in value))
        
        # Combine all text parts
        combined_text = ' '.join(text_parts)
        return self.text_cleaner.clean(combined_text)
    
    def _extract_skills(self, candidate_data: Dict[str, Any], resume_text: str) -> List[str]:
        """Extract skills from candidate data and resume text."""
        skills = []
        
        # Extract from skills field
        skills_field = candidate_data.get('skills') or candidate_data.get('Skills')
        if skills_field:
            if isinstance(skills_field, list):
                skills.extend([str(s).strip() for s in skills_field if s])
            elif isinstance(skills_field, str):
                # Split by comma, semicolon, or newline
                skills_list = re.split(r'[,;\n]', skills_field)
                skills.extend([s.strip() for s in skills_list if s.strip()])
        
        # Extract from resume text using skill extractor
        if resume_text:
            extracted_skills = self.skill_extractor.extract_skills(resume_text)
            skill_names = [skill.canonical_name for skill in extracted_skills]
            skills.extend(skill_names)
        
        # Canonicalize and deduplicate
        canonicalized = []
        seen = set()
        for skill in skills:
            if not skill:
                continue
            canonical_id = self.skill_extractor.canonicalize_skill(skill)
            if canonical_id:
                skill_info = self.skill_extractor.get_skill_info(canonical_id)
                if skill_info:
                    canonical_name = skill_info['canonical_name']
                    if canonical_name not in seen:
                        canonicalized.append(canonical_name)
                        seen.add(canonical_name)
            else:
                # Keep original if not in ontology
                normalized = skill.lower().strip()
                if normalized not in seen:
                    canonicalized.append(skill.strip())
                    seen.add(normalized)
        
        return canonicalized
    
    def _extract_experience_years(self, candidate_data: Dict[str, Any], resume_text: str) -> float:
        """Extract experience years from candidate data and resume text."""
        # Try direct field first
        exp_fields = ['experience_years', 'total_experience', 'years_of_experience', 
                     'totalExperience', 'experienceYears']
        for field in exp_fields:
            value = candidate_data.get(field)
            if value is not None:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    pass
        
        # Extract from text using patterns
        text_to_search = resume_text or ''
        if not text_to_search:
            text_to_search = str(candidate_data.get('experience', ''))
        
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text_to_search, re.IGNORECASE)
            if matches:
                try:
                    # Take the maximum value found
                    years = max(float(m) for m in matches)
                    return years
                except (ValueError, TypeError):
                    continue
        
        return 0.0
    
    def _extract_education(self, candidate_data: Dict[str, Any], resume_text: str) -> Optional[str]:
        """Extract education information."""
        # Try direct field
        edu_fields = ['education', 'Education', 'educational_background', 'degree']
        for field in edu_fields:
            value = candidate_data.get(field)
            if value and isinstance(value, str) and value.strip():
                return value.strip()
        
        # Extract from resume text
        if resume_text:
            normalized_text = resume_text.lower()
            for keyword in self.education_keywords:
                if keyword in normalized_text:
                    # Extract sentence or paragraph containing education info
                    sentences = re.split(r'[.!?]\s+', resume_text)
                    for sentence in sentences:
                        if keyword.lower() in sentence.lower():
                            return sentence.strip()
        
        return None
    
    def _extract_certifications(self, candidate_data: Dict[str, Any], resume_text: str) -> List[str]:
        """Extract certifications from candidate data and resume text."""
        certifications = []
        
        # Try direct field
        cert_field = candidate_data.get('certifications') or candidate_data.get('Certifications')
        if cert_field:
            if isinstance(cert_field, list):
                certifications.extend([str(c).strip() for c in cert_field if c])
            elif isinstance(cert_field, str):
                cert_list = re.split(r'[,;\n]', cert_field)
                certifications.extend([c.strip() for c in cert_list if c.strip()])
        
        # Extract from resume text
        if resume_text:
            normalized_text = resume_text.lower()
            for keyword in self.certification_keywords:
                if keyword in normalized_text:
                    # Find certification mentions
                    pattern = rf'\b{re.escape(keyword)}\s+([A-Z][A-Za-z\s]+)'
                    matches = re.findall(pattern, resume_text, re.IGNORECASE)
                    certifications.extend([m.strip() for m in matches])
        
        # Deduplicate
        return list(dict.fromkeys(certifications))  # Preserves order

