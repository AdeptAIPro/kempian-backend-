"""
Production-Grade Experience Parser
Converts raw resumes into structured experience records with validation and seniority detection.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date
from dateutil.parser import parse as date_parse
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

@dataclass
class StructuredExperience:
    """Structured experience record"""
    company: str
    title_normalized: str
    title_original: str
    start_date: Optional[date]
    end_date: Optional[date]
    is_current: bool
    duration_months: int
    skills: List[str]
    achievements: List[str]
    location: Optional[str]
    impact_metrics: List[Dict[str, float]]  # e.g., [{"type": "cost_reduction", "value": 20.0, "unit": "percent"}]
    seniority_level: str  # 'junior', 'mid', 'senior', 'expert'
    confidence: float
    needs_review: bool
    review_reason: Optional[str] = None


class ExperienceParser:
    """Parse and structure experience from resumes"""
    
    def __init__(self):
        # Title normalization patterns
        self.title_patterns = {
            'software_engineer': [
                r'software\s+engineer',
                r'se\s+',
                r'software\s+developer',
                r'programmer',
                r'coder'
            ],
            'senior_software_engineer': [
                r'senior\s+software\s+engineer',
                r'sr\.?\s+software\s+engineer',
                r'senior\s+developer',
                r'lead\s+developer'
            ],
            'principal_engineer': [
                r'principal\s+engineer',
                r'staff\s+engineer',
                r'architect'
            ],
            'registered_nurse': [
                r'registered\s+nurse',
                r'rn\s+',
                r'r\.?n\.?'
            ],
            'nurse_practitioner': [
                r'nurse\s+practitioner',
                r'np\s+',
                r'n\.?p\.?'
            ]
        }
        
        # Seniority detection patterns
        self.seniority_keywords = {
            'junior': ['junior', 'entry', 'associate', 'trainee', 'intern'],
            'mid': ['mid', 'intermediate', 'regular', 'standard'],
            'senior': ['senior', 'sr.', 'sr ', 'lead', 'experienced'],
            'expert': ['expert', 'principal', 'staff', 'architect', 'director', 'manager']
        }
        
        # Impact metric patterns
        self.impact_patterns = {
            'cost_reduction': [
                r'reduced\s+cost\s+by\s+(\d+(?:\.\d+)?)\s*%',
                r'decreased\s+expenses\s+by\s+(\d+(?:\.\d+)?)\s*%',
                r'saved\s+\$?(\d+(?:,\d+)?(?:\.\d+)?)',
                r'cost\s+reduction\s+of\s+(\d+(?:\.\d+)?)\s*%'
            ],
            'revenue_increase': [
                r'increased\s+revenue\s+by\s+(\d+(?:\.\d+)?)\s*%',
                r'generated\s+\$?(\d+(?:,\d+)?(?:\.\d+)?)',
                r'revenue\s+growth\s+of\s+(\d+(?:\.\d+)?)\s*%'
            ],
            'efficiency_improvement': [
                r'improved\s+efficiency\s+by\s+(\d+(?:\.\d+)?)\s*%',
                r'reduced\s+time\s+by\s+(\d+(?:\.\d+)?)\s*%',
                r'increased\s+productivity\s+by\s+(\d+(?:\.\d+)?)\s*%'
            ],
            'team_size': [
                r'managed\s+(\d+)\s+(?:team\s+members?|people|staff)',
                r'led\s+a\s+team\s+of\s+(\d+)',
                r'supervised\s+(\d+)\s+employees?'
            ],
            'scale_impact': [
                r'scaled\s+to\s+(\d+(?:,\d+)?)\s+users?',
                r'handled\s+(\d+(?:,\d+)?)\s+transactions?',
                r'processed\s+(\d+(?:,\d+)?)\s+records?'
            ]
        }
        
        # Date patterns
        self.date_patterns = [
            r'(\w+\s+\d{4})\s*[-–—]\s*(\w+\s+\d{4}|present|current)',
            r'(\d{1,2}[/-]\d{4})\s*[-–—]\s*(\d{1,2}[/-]\d{4}|present|current)',
            r'(\d{4})\s*[-–—]\s*(\d{4}|present|current)',
            r'from\s+(\w+\s+\d{4})\s+to\s+(\w+\s+\d{4}|present|current)',
            r'(\w+\s+\d{4})\s+to\s+(\w+\s+\d{4}|present|current)'
        ]
    
    def parse_experience(self, experience_text: str, resume_context: Optional[Dict] = None) -> List[StructuredExperience]:
        """
        Parse experience text into structured records
        
        Args:
            experience_text: Raw experience text from resume
            resume_context: Additional context (skills, location, etc.)
        
        Returns:
            List of StructuredExperience records
        """
        if not experience_text:
            return []
        
        # Split into individual experiences (by company/role)
        experience_sections = self._split_experience_sections(experience_text)
        
        structured_experiences = []
        
        for section in experience_sections:
            try:
                structured = self._parse_single_experience(section, resume_context)
                if structured:
                    structured_experiences.append(structured)
            except Exception as e:
                logger.error(f"Error parsing experience section: {e}")
                # Create fallback record
                structured_experiences.append(self._create_fallback_experience(section))
        
        return structured_experiences
    
    def _split_experience_sections(self, text: str) -> List[str]:
        """Split experience text into individual job sections"""
        # Split by common delimiters
        sections = re.split(r'\n\s*\n|\n(?=[A-Z][a-z]+\s+[A-Z])', text)
        
        # Filter out very short sections
        sections = [s.strip() for s in sections if len(s.strip()) > 20]
        
        return sections
    
    def _parse_single_experience(self, section: str, context: Optional[Dict] = None) -> Optional[StructuredExperience]:
        """Parse a single experience section"""
        # Extract company
        company = self._extract_company(section)
        
        # Extract title
        title_original, title_normalized = self._extract_and_normalize_title(section)
        
        # Extract dates
        start_date, end_date, is_current = self._extract_dates(section)
        
        # Validate dates
        date_validation = self._validate_dates(start_date, end_date)
        needs_review = not date_validation['valid']
        
        # Calculate duration
        duration_months = self._calculate_duration(start_date, end_date, is_current)
        
        # Extract skills
        skills = self._extract_skills_from_experience(section, context)
        
        # Extract achievements
        achievements = self._extract_achievements(section)
        
        # Extract impact metrics
        impact_metrics = self._extract_impact_metrics(section)
        
        # Extract location
        location = self._extract_location(section, context)
        
        # Detect seniority
        seniority_level, seniority_confidence = self._detect_seniority(section, title_normalized)
        
        # Overall confidence
        confidence = self._calculate_confidence(
            date_validation, seniority_confidence, len(achievements), len(impact_metrics)
        )
        
        return StructuredExperience(
            company=company,
            title_normalized=title_normalized,
            title_original=title_original,
            start_date=start_date,
            end_date=end_date,
            is_current=is_current,
            duration_months=duration_months,
            skills=skills,
            achievements=achievements,
            location=location,
            impact_metrics=impact_metrics,
            seniority_level=seniority_level,
            confidence=confidence,
            needs_review=needs_review,
            review_reason=date_validation.get('reason')
        )
    
    def _extract_company(self, text: str) -> str:
        """Extract company name"""
        # Look for company patterns (usually first line or after title)
        lines = text.split('\n')
        if lines:
            # First non-empty line is often company
            for line in lines[:3]:
                line = line.strip()
                if line and len(line) > 2:
                    # Remove common prefixes
                    line = re.sub(r'^(at|worked at|employed at)\s+', '', line, flags=re.IGNORECASE)
                    return line[:100]  # Limit length
        
        return "Unknown Company"
    
    def _extract_and_normalize_title(self, text: str) -> Tuple[str, str]:
        """Extract and normalize job title"""
        # Look for title patterns
        title_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Engineer|Developer|Manager|Nurse|Analyst|Designer|Architect))'
        match = re.search(title_pattern, text)
        
        if match:
            title_original = match.group(1)
        else:
            # Fallback: first line or look for common title keywords
            lines = text.split('\n')
            title_original = lines[0].strip() if lines else "Unknown Title"
        
        # Normalize title
        title_normalized = self._normalize_title(title_original)
        
        return title_original, title_normalized
    
    def _normalize_title(self, title: str) -> str:
        """Normalize job title to standard form"""
        title_lower = title.lower()
        
        # Check against patterns
        for normalized, patterns in self.title_patterns.items():
            for pattern in patterns:
                if re.search(pattern, title_lower):
                    return normalized
        
        # Default normalization
        return title_lower.replace(' ', '_')
    
    def _extract_dates(self, text: str) -> Tuple[Optional[date], Optional[date], bool]:
        """Extract start and end dates"""
        is_current = False
        
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_str = match.group(1)
                end_str = match.group(2).lower()
                
                # Parse start date
                try:
                    start_date = self._parse_date(start_str)
                except:
                    start_date = None
                
                # Parse end date
                if end_str in ['present', 'current', 'now']:
                    end_date = None
                    is_current = True
                else:
                    try:
                        end_date = self._parse_date(end_str)
                    except:
                        end_date = None
                
                if start_date:
                    return start_date, end_date, is_current
        
        return None, None, False
    
    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object"""
        try:
            # Try dateutil parser first
            return date_parse(date_str, default=datetime(2000, 1, 1)).date()
        except:
            # Try manual parsing
            # Format: "January 2020" or "Jan 2020"
            month_year = re.search(r'(\w+)\s+(\d{4})', date_str)
            if month_year:
                month_name = month_year.group(1)
                year = int(month_year.group(2))
                
                month_map = {
                    'january': 1, 'jan': 1,
                    'february': 2, 'feb': 2,
                    'march': 3, 'mar': 3,
                    'april': 4, 'apr': 4,
                    'may': 5,
                    'june': 6, 'jun': 6,
                    'july': 7, 'jul': 7,
                    'august': 8, 'aug': 8,
                    'september': 9, 'sep': 9, 'sept': 9,
                    'october': 10, 'oct': 10,
                    'november': 11, 'nov': 11,
                    'december': 12, 'dec': 12
                }
                
                month = month_map.get(month_name.lower(), 1)
                return date(year, month, 1)
            
            raise ValueError(f"Could not parse date: {date_str}")
    
    def _validate_dates(self, start_date: Optional[date], end_date: Optional[date]) -> Dict:
        """Validate date ranges"""
        validation = {'valid': True, 'reason': None}
        
        if not start_date:
            validation['valid'] = False
            validation['reason'] = 'Missing start date'
            return validation
        
        if end_date and end_date < start_date:
            validation['valid'] = False
            validation['reason'] = 'End date before start date'
            return validation
        
        # Check for unrealistic dates (too far in future or past)
        today = date.today()
        if start_date > today:
            validation['valid'] = False
            validation['reason'] = 'Start date in future'
            return validation
        
        if end_date and end_date > today:
            validation['valid'] = False
            validation['reason'] = 'End date in future (should be present/current)'
            return validation
        
        # Check for very old dates (likely errors)
        if start_date.year < 1950:
            validation['valid'] = False
            validation['reason'] = 'Start date too old (likely error)'
            return validation
        
        return validation
    
    def _calculate_duration(self, start_date: Optional[date], end_date: Optional[date], is_current: bool) -> int:
        """Calculate duration in months"""
        if not start_date:
            return 0
        
        if is_current or not end_date:
            end = date.today()
        else:
            end = end_date
        
        delta = relativedelta(end, start_date)
        return delta.years * 12 + delta.months
    
    def _extract_skills_from_experience(self, text: str, context: Optional[Dict] = None) -> List[str]:
        """Extract skills mentioned in experience"""
        skills = []
        
        # Common skill keywords
        skill_keywords = [
            'python', 'javascript', 'java', 'react', 'angular', 'vue',
            'aws', 'docker', 'kubernetes', 'sql', 'mongodb', 'postgresql',
            'nursing', 'patient care', 'medication administration'
        ]
        
        text_lower = text.lower()
        for keyword in skill_keywords:
            if keyword in text_lower:
                skills.append(keyword)
        
        # Add skills from context if available
        if context and 'skills' in context:
            context_skills = context['skills']
            if isinstance(context_skills, list):
                skills.extend(context_skills)
        
        return list(set(skills))  # Remove duplicates
    
    def _extract_achievements(self, text: str) -> List[str]:
        """Extract achievement bullets"""
        achievements = []
        
        # Look for bullet points
        bullets = re.findall(r'[•\-\*]\s*(.+?)(?=\n|$)', text, re.MULTILINE)
        achievements.extend(bullets)
        
        # Look for numbered lists
        numbered = re.findall(r'\d+\.\s*(.+?)(?=\n|$)', text, re.MULTILINE)
        achievements.extend(numbered)
        
        # Clean and filter
        achievements = [a.strip() for a in achievements if len(a.strip()) > 10]
        
        return achievements[:10]  # Limit to top 10
    
    def _extract_impact_metrics(self, text: str) -> List[Dict[str, float]]:
        """Extract quantifiable impact metrics"""
        metrics = []
        
        for metric_type, patterns in self.impact_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value_str = match.group(1).replace(',', '')
                    try:
                        value = float(value_str)
                        metrics.append({
                            'type': metric_type,
                            'value': value,
                            'unit': 'percent' if '%' in match.group(0) else 'absolute',
                            'text': match.group(0)
                        })
                    except ValueError:
                        continue
        
        return metrics
    
    def _extract_location(self, text: str, context: Optional[Dict] = None) -> Optional[str]:
        """Extract location from experience"""
        # Look for location patterns
        location_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2}|[A-Z][a-z]+)'
        match = re.search(location_pattern, text)
        
        if match:
            return f"{match.group(1)}, {match.group(2)}"
        
        # Fallback to context
        if context and 'location' in context:
            return context['location']
        
        return None
    
    def _detect_seniority(self, text: str, title_normalized: str) -> Tuple[str, float]:
        """Detect seniority level from text and title"""
        text_lower = text.lower()
        title_lower = title_normalized.lower()
        
        # Check title
        for level, keywords in self.seniority_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return (level, 0.9)
        
        # Check text
        for level, keywords in self.seniority_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                confidence = min(0.8, 0.5 + matches * 0.1)
                return (level, confidence)
        
        # Default to mid
        return ('mid', 0.5)
    
    def _calculate_confidence(
        self,
        date_validation: Dict,
        seniority_confidence: float,
        achievement_count: int,
        metric_count: int
    ) -> float:
        """Calculate overall parsing confidence"""
        confidence = 0.5  # Base
        
        # Date validation
        if date_validation['valid']:
            confidence += 0.2
        
        # Seniority confidence
        confidence += seniority_confidence * 0.1
        
        # Achievement count
        confidence += min(0.1, achievement_count * 0.02)
        
        # Metric count
        confidence += min(0.1, metric_count * 0.05)
        
        return min(1.0, confidence)
    
    def _create_fallback_experience(self, section: str) -> StructuredExperience:
        """Create fallback experience record when parsing fails"""
        return StructuredExperience(
            company="Unknown",
            title_normalized="unknown",
            title_original=section[:50],
            start_date=None,
            end_date=None,
            is_current=False,
            duration_months=0,
            skills=[],
            achievements=[],
            location=None,
            impact_metrics=[],
            seniority_level='mid',
            confidence=0.3,
            needs_review=True,
            review_reason='Parsing failed'
        )


# Global instance
_experience_parser = None

def get_experience_parser() -> ExperienceParser:
    """Get or create global experience parser instance"""
    global _experience_parser
    if _experience_parser is None:
        _experience_parser = ExperienceParser()
    return _experience_parser

def parse_experience(experience_text: str, resume_context: Optional[Dict] = None) -> List[StructuredExperience]:
    """Convenience function to parse experience"""
    parser = get_experience_parser()
    return parser.parse_experience(experience_text, resume_context)

