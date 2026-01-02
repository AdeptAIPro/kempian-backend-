"""
Natural language query parsing utilities
"""
import re
from typing import Dict, List, Optional
from app.simple_logger import get_logger
from dataclasses import dataclass

@dataclass
class ParsedJobQuery:
    """Structured job query representation"""
    title: str
    required_skills: List[str]
    preferred_skills: List[str]
    min_experience: int
    max_experience: int
    location: str
    industry: str
    seniority_level: str
    education_level: str

class NaturalLanguageQueryParser:
    """Parse natural language job queries into structured format"""
    
    def __init__(self):
        self.job_title_patterns = {
            'developer': ['developer', 'engineer', 'programmer', 'coder'],
            'manager': ['manager', 'lead', 'director', 'supervisor'],
            'analyst': ['analyst', 'specialist', 'consultant'],
            'nurse': ['nurse', 'rn', 'cna', 'nursing']
        }
        
        self.seniority_patterns = {
            'entry': ['entry', 'junior', 'graduate', 'intern', 'trainee'],
            'mid': ['mid', 'intermediate', 'regular'],
            'senior': ['senior', 'sr', 'lead', 'principal', 'staff'],
            'executive': ['director', 'vp', 'chief', 'head']
        }
        
        self.skill_extraction_patterns = [
            r'(?:experience\s+(?:with|in)|skilled\s+in|proficient\s+in)\s+([^.]+)',
            r'(?:knowledge\s+of|familiar\s+with)\s+([^.]+)',
            r'(?:using|working\s+with)\s+([^.]+)'
        ]
    
    def parse_query(self, query: str) -> ParsedJobQuery:
        """Parse natural language query into structured format"""
        query_lower = query.lower()
        
        return ParsedJobQuery(
            title=self._extract_job_title(query_lower),
            required_skills=self._extract_skills(query_lower, required=True),
            preferred_skills=self._extract_skills(query_lower, required=False),
            min_experience=self._extract_min_experience(query_lower),
            max_experience=self._extract_max_experience(query_lower),
            location=self._extract_location(query),
            industry=self._extract_industry(query_lower),
            seniority_level=self._extract_seniority(query_lower),
            education_level=self._extract_education(query_lower)
        )
    
    def _extract_job_title(self, query: str) -> str:
        """Extract job title from query"""
        for canonical_title, variations in self.job_title_patterns.items():
            for variation in variations:
                if variation in query:
                    # Look for seniority modifiers
                    for seniority in ['senior', 'junior', 'lead', 'principal']:
                        if f'{seniority} {variation}' in query:
                            return f'{seniority} {canonical_title}'
                    return canonical_title
        
        # Fallback: extract first few words
        words = query.split()[:3]
        return ' '.join(words)
    
    def _extract_skills(self, query: str, required: bool = True) -> List[str]:
        """Extract skills from query"""
        skills = []
        
        for pattern in self.skill_extraction_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Split by common separators
                skill_parts = re.split(r'[,\s+and\s+&]', match)
                skills.extend([skill.strip() for skill in skill_parts if skill.strip()])
        
        # Common skills that might appear directly
        common_skills = [
            'python', 'java', 'javascript', 'react', 'node', 'aws', 'sql',
            'machine learning', 'data science', 'healthcare', 'nursing'
        ]
        
        for skill in common_skills:
            if skill in query and skill not in skills:
                skills.append(skill)
        
        return skills[:5]  # Limit to top 5
    
    def _extract_min_experience(self, query: str) -> int:
        """Extract minimum experience requirement"""
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'minimum\s+(\d+)\s*years?',
            r'at\s+least\s+(\d+)\s*years?',
            r'(\d+)-\d+\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return 0
    
    def _extract_max_experience(self, query: str) -> int:
        """Extract maximum experience limit"""
        patterns = [
            r'\d+-(\d+)\s*years?',
            r'maximum\s+(\d+)\s*years?',
            r'up\s+to\s+(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return 50  # Default max
    
    def _extract_location(self, query: str) -> str:
        """Extract location from query"""
        # Look for city, state patterns
        location_pattern = r'(?:in|at|from|located)\s+([A-Z][a-z]+(?:\s*,\s*[A-Z]{2,})?)'
        match = re.search(location_pattern, query)
        
        if match:
            return match.group(1)
        
        return ""
    
    def _extract_industry(self, query: str) -> str:
        """Extract industry from query"""
        industry_keywords = {
            'healthcare': ['healthcare', 'medical', 'hospital', 'clinical', 'nursing'],
            'technology': ['tech', 'software', 'it', 'programming', 'development'],
            'finance': ['finance', 'banking', 'fintech', 'investment'],
            'education': ['education', 'university', 'school', 'academic']
        }
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in query for keyword in keywords):
                return industry
        
        return 'general'
    
    def _extract_seniority(self, query: str) -> str:
        """Extract seniority level from query"""
        for seniority, patterns in self.seniority_patterns.items():
            if any(pattern in query for pattern in patterns):
                return seniority
        
        return 'mid'  # Default
    
    def _extract_education(self, query: str) -> str:
        """Extract education requirements"""
        education_patterns = {
            'phd': ['phd', 'doctorate', 'doctoral'],
            'masters': ['masters', 'master', 'ms', 'ma', 'mba'],
            'bachelors': ['bachelors', 'bachelor', 'bs', 'ba', 'degree'],
            'associates': ['associates', 'associate']
        }
        
        for education, patterns in education_patterns.items():
            if any(pattern in query for pattern in patterns):
                return education
        
        return ""

# Global instance
query_parser = NaturalLanguageQueryParser()