# advanced_query_parser.py - Enhanced Job Query Parser for Complex Requirements

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from app.simple_logger import get_logger
from dataclasses import dataclass
from collections import defaultdict

logger = get_logger("search")

@dataclass
class JobRequirement:
    """Structured representation of a job requirement"""
    category: str  # 'skill', 'experience', 'education', 'responsibility', 'benefit'
    requirement: str
    priority: str  # 'required', 'preferred', 'nice_to_have'
    weight: float  # 0.0 to 1.0
    keywords: List[str]

@dataclass
class ParsedJobQuery:
    """Complete parsed job query with structured requirements"""
    job_title: str
    company_info: str
    requirements: List[JobRequirement]
    responsibilities: List[str]
    benefits: List[str]
    experience_years: Optional[int]
    seniority_level: str
    technical_skills: List[str]
    soft_skills: List[str]
    education_requirements: List[str]
    location_info: str
    work_type: str  # 'remote', 'hybrid', 'onsite'

class AdvancedJobQueryParser:
    """Advanced parser for complex job descriptions and requirements"""
    
    def __init__(self):
        self.skill_categories = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 
                'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab'
            ],
            'web_frameworks': [
                'react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt.js', 'express.js',
                'django', 'flask', 'fastapi', 'spring', 'laravel', 'rails'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'dynamodb', 'oracle', 'sql server', 'sqlite'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 'vercel'
            ],
            'devops_tools': [
                'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions', 
                'terraform', 'ansible', 'chef', 'puppet'
            ],
            'ml_ai_tools': [
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'opencv',
                'hugging face', 'transformers', 'langchain'
            ],
            'testing_tools': [
                'jest', 'pytest', 'junit', 'selenium', 'cypress', 'mocha', 'chai'
            ]
        }
        
        self.soft_skills = [
            'communication', 'leadership', 'teamwork', 'problem solving', 'analytical',
            'creative', 'adaptable', 'collaborative', 'mentoring', 'project management'
        ]
        
        self.seniority_indicators = {
            'senior': ['senior', 'sr.', 'lead', 'principal', 'staff', 'architect'],
            'mid': ['mid-level', 'intermediate', 'experienced'],
            'junior': ['junior', 'jr.', 'entry-level', 'associate', 'graduate']
        }
        
        self.experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:in|with)',
            r'minimum\s*(?:of\s*)?(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?'
        ]
        
        logger.info("AdvancedJobQueryParser initialized with comprehensive skill categories")

    def parse_job_query(self, query: str) -> ParsedJobQuery:
        """Parse a complex job description into structured requirements"""
        try:
            # Clean and normalize the query
            cleaned_query = self._clean_query(query)
            
            # Extract basic job information
            job_title = self._extract_job_title(cleaned_query)
            company_info = self._extract_company_info(cleaned_query)
            
            # Parse different sections
            requirements = self._parse_requirements_section(cleaned_query)
            responsibilities = self._parse_responsibilities_section(cleaned_query)
            benefits = self._parse_benefits_section(cleaned_query)
            
            # Extract specific information
            experience_years = self._extract_experience_years(cleaned_query)
            seniority_level = self._determine_seniority_level(cleaned_query, job_title)
            technical_skills = self._extract_technical_skills(cleaned_query)
            soft_skills = self._extract_soft_skills(cleaned_query)
            education_requirements = self._extract_education_requirements(cleaned_query)
            location_info = self._extract_location_info(cleaned_query)
            work_type = self._extract_work_type(cleaned_query)
            
            parsed_query = ParsedJobQuery(
                job_title=job_title,
                company_info=company_info,
                requirements=requirements,
                responsibilities=responsibilities,
                benefits=benefits,
                experience_years=experience_years,
                seniority_level=seniority_level,
                technical_skills=technical_skills,
                soft_skills=soft_skills,
                education_requirements=education_requirements,
                location_info=location_info,
                work_type=work_type
            )
            
            logger.info(f"Successfully parsed job query: {job_title} with {len(requirements)} requirements")
            return parsed_query
            
        except Exception as e:
            logger.error(f"Error parsing job query: {e}")
            # Return a basic parsed query as fallback
            return self._create_fallback_parsed_query(query)

    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query text"""
        # Remove extra whitespace and normalize line breaks
        cleaned = re.sub(r'\s+', ' ', query.strip())
        # Remove bullet points and special characters
        cleaned = re.sub(r'[•\-\*]\s*', '', cleaned)
        return cleaned

    def _extract_job_title(self, query: str) -> str:
        """Extract job title from the query"""
        # Look for common job title patterns
        title_patterns = [
            r'^([A-Z][^.!?]*(?:Engineer|Developer|Manager|Analyst|Scientist|Architect|Lead|Director))',
            r'(?:seeking|hiring|looking for)\s+(?:an?\s+)?([^.!?]*(?:Engineer|Developer|Manager|Analyst|Scientist))',
            r'Position:\s*([^.!?\n]+)',
            r'Role:\s*([^.!?\n]+)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: look for the first line that might be a title
        lines = query.split('\n')
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 100 and any(word in first_line.lower() for word in ['engineer', 'developer', 'manager', 'analyst']):
                return first_line
        
        return "Software Position"

    def _extract_company_info(self, query: str) -> str:
        """Extract company information"""
        # Look for company mentions
        company_patterns = [
            r'(?:at|with|for)\s+([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Company|Technologies))',
            r'Company:\s*([^.!?\n]+)',
            r'Organization:\s*([^.!?\n]+)'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        
        return "Technology Company"

    def _parse_requirements_section(self, query: str) -> List[JobRequirement]:
        """Parse the requirements section into structured requirements"""
        requirements = []
        
        # Find requirements section
        req_section = self._extract_section(query, ['requirements', 'qualifications', 'must have', 'skills needed'])
        
        if req_section:
            # Split into individual requirements
            req_items = self._split_requirements(req_section)
            
            for item in req_items:
                requirement = self._parse_single_requirement(item)
                if requirement:
                    requirements.append(requirement)
        
        # Also parse skills from the entire query
        tech_skills = self._extract_technical_skills(query)
        for skill in tech_skills:
            req = JobRequirement(
                category='skill',
                requirement=f"Experience with {skill}",
                priority='required',
                weight=0.8,
                keywords=[skill.lower()]
            )
            requirements.append(req)
        
        return requirements

    def _extract_section(self, query: str, section_keywords: List[str]) -> str:
        """Extract a specific section from the query"""
        query_lower = query.lower()
        
        for keyword in section_keywords:
            pattern = rf'{keyword}:?\s*\n?(.*?)(?:\n\n|\n[A-Z][a-z]+:|$)'
            match = re.search(pattern, query_lower, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""

    def _split_requirements(self, req_section: str) -> List[str]:
        """Split requirements section into individual items"""
        # Split by common delimiters
        items = re.split(r'[•\-\*]\s*|(?:\n\s*)+(?=\w)', req_section)
        
        # Clean and filter items
        cleaned_items = []
        for item in items:
            item = item.strip()
            if len(item) > 10 and not item.startswith(('and', 'or', 'the', 'a ')):
                cleaned_items.append(item)
        
        return cleaned_items

    def _parse_single_requirement(self, req_text: str) -> Optional[JobRequirement]:
        """Parse a single requirement into structured format"""
        req_lower = req_text.lower()
        
        # Determine category
        category = 'general'
        if any(skill_cat in req_lower for skill_list in self.skill_categories.values() for skill_cat in skill_list):
            category = 'skill'
        elif any(word in req_lower for word in ['year', 'experience']):
            category = 'experience'
        elif any(word in req_lower for word in ['degree', 'bachelor', 'master', 'education']):
            category = 'education'
        
        # Determine priority
        priority = 'required'
        if any(word in req_lower for word in ['preferred', 'nice to have', 'bonus', 'plus']):
            priority = 'preferred'
        elif any(word in req_lower for word in ['required', 'must', 'essential']):
            priority = 'required'
        
        # Determine weight based on priority and category
        weight = 0.9 if priority == 'required' else 0.6
        if category == 'skill':
            weight += 0.1
        
        # Extract keywords
        keywords = self._extract_keywords_from_requirement(req_text)
        
        return JobRequirement(
            category=category,
            requirement=req_text,
            priority=priority,
            weight=min(1.0, weight),
            keywords=keywords
        )

    def _extract_keywords_from_requirement(self, req_text: str) -> List[str]:
        """Extract relevant keywords from a requirement"""
        keywords = []
        req_lower = req_text.lower()
        
        # Check for technical skills
        for category, skills in self.skill_categories.items():
            for skill in skills:
                if skill in req_lower:
                    keywords.append(skill)
        
        # Check for soft skills
        for skill in self.soft_skills:
            if skill in req_lower:
                keywords.append(skill)
        
        # Extract years of experience
        exp_match = re.search(r'(\d+)\+?\s*years?', req_lower)
        if exp_match:
            keywords.append(f"{exp_match.group(1)}_years")
        
        return keywords

    def _parse_responsibilities_section(self, query: str) -> List[str]:
        """Parse responsibilities section"""
        resp_section = self._extract_section(query, ['responsibilities', 'duties', 'you will', 'role involves'])
        
        if resp_section:
            return self._split_requirements(resp_section)
        
        return []

    def _parse_benefits_section(self, query: str) -> List[str]:
        """Parse benefits section"""
        benefits_section = self._extract_section(query, ['benefits', 'we offer', 'perks', 'compensation'])
        
        if benefits_section:
            return self._split_requirements(benefits_section)
        
        return []

    def _extract_experience_years(self, query: str) -> Optional[int]:
        """Extract required years of experience"""
        for pattern in self.experience_patterns:
            match = re.search(pattern, query.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None

    def _determine_seniority_level(self, query: str, job_title: str) -> str:
        """Determine seniority level from query and job title"""
        combined_text = f"{query} {job_title}".lower()
        
        for level, indicators in self.seniority_indicators.items():
            if any(indicator in combined_text for indicator in indicators):
                return level.title()
        
        # Default based on experience years if mentioned
        exp_years = self._extract_experience_years(query)
        if exp_years:
            if exp_years >= 7:
                return 'Senior'
            elif exp_years >= 3:
                return 'Mid'
            else:
                return 'Junior'
        
        return 'Mid'  # Default

    def _extract_technical_skills(self, query: str) -> List[str]:
        """Extract technical skills from the query"""
        skills = []
        query_lower = query.lower()
        
        for category, skill_list in self.skill_categories.items():
            for skill in skill_list:
                if skill in query_lower:
                    skills.append(skill)
        
        return list(set(skills))  # Remove duplicates

    def _extract_soft_skills(self, query: str) -> List[str]:
        """Extract soft skills from the query"""
        skills = []
        query_lower = query.lower()
        
        for skill in self.soft_skills:
            if skill in query_lower:
                skills.append(skill)
        
        return skills

    def _extract_education_requirements(self, query: str) -> List[str]:
        """Extract education requirements"""
        education = []
        query_lower = query.lower()
        
        education_patterns = [
            r'(bachelor[\'s]*\s*degree)',
            r'(master[\'s]*\s*degree)',
            r'(phd|doctorate)',
            r'(computer science|engineering|mathematics)',
            r'(degree\s*in\s*[\w\s]+)'
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, query_lower)
            education.extend(matches)
        
        return education

    def _extract_location_info(self, query: str) -> str:
        """Extract location information"""
        location_patterns = [
            r'(?:location|based in|located in):\s*([^.!?\n]+)',
            r'(?:remote|hybrid|on-site|onsite)',
            r'([A-Z][a-z]+,\s*[A-Z]{2})',  # City, State
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1 if match.lastindex else 0).strip()
        
        return ""

    def _extract_work_type(self, query: str) -> str:
        """Extract work type (remote, hybrid, onsite)"""
        query_lower = query.lower()
        
        if 'remote' in query_lower:
            return 'remote'
        elif 'hybrid' in query_lower:
            return 'hybrid'
        elif any(word in query_lower for word in ['on-site', 'onsite', 'office']):
            return 'onsite'
        
        return 'flexible'

    def _create_fallback_parsed_query(self, query: str) -> ParsedJobQuery:
        """Create a basic parsed query as fallback"""
        return ParsedJobQuery(
            job_title="Software Position",
            company_info="Technology Company",
            requirements=[],
            responsibilities=[],
            benefits=[],
            experience_years=None,
            seniority_level="Mid",
            technical_skills=self._extract_technical_skills(query),
            soft_skills=self._extract_soft_skills(query),
            education_requirements=[],
            location_info="",
            work_type="flexible"
        )

    def get_search_weights(self, parsed_query: ParsedJobQuery) -> Dict[str, float]:
        """Generate search weights based on parsed query"""
        weights = {
            'technical_skills': 0.35,
            'experience_match': 0.25,
            'seniority_match': 0.15,
            'education_match': 0.10,
            'soft_skills': 0.10,
            'location_match': 0.05
        }
        
        # Adjust weights based on query characteristics
        if len(parsed_query.technical_skills) > 5:
            weights['technical_skills'] += 0.1
            weights['experience_match'] -= 0.05
            weights['soft_skills'] -= 0.05
        
        if parsed_query.experience_years and parsed_query.experience_years > 5:
            weights['experience_match'] += 0.1
            weights['technical_skills'] -= 0.05
            weights['seniority_match'] -= 0.05
        
        return weights