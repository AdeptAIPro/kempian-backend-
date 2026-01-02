"""
Simplified input validation for AdeptAI application
Provides basic validation for security and reliability
"""

import re
import html
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, EmailStr
import bleach

from app.exceptions import ValidationError


class BaseValidationModel(BaseModel):
    """Base model with common validation methods"""
    
    class Config:
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment
    
    def sanitize_html(self, value: str) -> str:
        """Sanitize HTML content"""
        if not isinstance(value, str):
            return value
        return bleach.clean(value, tags=[], strip=True)
    
    def validate_no_sql_injection(self, value: str) -> str:
        """Basic SQL injection prevention"""
        if not isinstance(value, str):
            return value
        
        # Common SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|\/\*|\*\/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+'.*'\s*=\s*'.*')",
            r"(\bUNION\s+SELECT\b)",
            r"(\bDROP\s+TABLE\b)",
            r"(\bINSERT\s+INTO\b)",
            r"(\bDELETE\s+FROM\b)"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError("Invalid input detected")
        
        return value


class SearchRequest(BaseValidationModel):
    """Validation for search requests"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    include_explanation: bool = Field(default=False, description="Include explanation for results")
    include_behavioural_analysis: bool = Field(default=False, description="Run behavioral analysis and include in results")
    enable_domain_filtering: bool = Field(default=True, description="Enable domain filtering when supported")
    
    def model_post_init(self, __context):
        """Post-initialization validation"""
        # Sanitize query
        if self.query:
            self.query = html.escape(self.query.strip())
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'<script.*?>.*?</script>',
                r'javascript:',
                r'data:',
                r'vbscript:',
                r'on\w+\s*='
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, self.query, re.IGNORECASE):
                    raise ValueError("Query contains invalid content")
        
        # Validate filters
        if self.filters:
            allowed_keys = {
                'location', 'experience_level', 'skills', 'industry',
                'salary_min', 'salary_max', 'company_size', 'remote'
            }
            
            for key in self.filters.keys():
                if key not in allowed_keys:
                    raise ValueError(f"Invalid filter key: {key}")


class CandidateData(BaseValidationModel):
    """Validation for candidate data"""
    email: EmailStr = Field(..., description="Candidate email")
    full_name: str = Field(..., min_length=1, max_length=100, description="Full name")
    phone: Optional[str] = Field(None, description="Phone number")
    skills: List[str] = Field(..., min_items=1, max_items=50, description="List of skills")
    experience_years: int = Field(..., ge=0, le=50, description="Years of experience")
    location: str = Field(..., min_length=1, max_length=100, description="Location")
    resume_text: str = Field(..., min_length=10, max_length=50000, description="Resume text content")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    github_url: Optional[str] = Field(None, description="GitHub profile URL")
    
    def model_post_init(self, __context):
        """Post-initialization validation"""
        # Sanitize full name
        if self.full_name:
            self.full_name = html.escape(self.full_name.strip())
            
            # Check for valid name characters
            if not re.match(r'^[a-zA-Z\s\.\-\']+$', self.full_name):
                raise ValueError("Name contains invalid characters")
        
        # Sanitize skills
        if self.skills:
            sanitized_skills = []
            for skill in self.skills:
                if skill and skill.strip():
                    skill = html.escape(skill.strip())
                    if len(skill) > 50:
                        raise ValueError(f"Skill too long: {skill}")
                    sanitized_skills.append(skill)
            
            if not sanitized_skills:
                raise ValueError("At least one valid skill is required")
            
            self.skills = sanitized_skills
        
        # Sanitize resume text
        if self.resume_text:
            self.resume_text = bleach.clean(self.resume_text, tags=[], strip=True)
            
            # Check for suspicious content
            suspicious_patterns = [
                r'<script.*?>.*?</script>',
                r'javascript:',
                r'data:',
                r'vbscript:',
                r'on\w+\s*='
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, self.resume_text, re.IGNORECASE):
                    raise ValueError("Resume contains invalid content")
        
        # Validate phone number
        if self.phone and not re.match(r'^\+?[\d\s\-\(\)]+$', self.phone):
            raise ValueError("Invalid phone number format")
        
        # Validate URLs
        if self.linkedin_url and not re.match(r'^https://linkedin\.com/in/[\w\-]+/?$', self.linkedin_url):
            raise ValueError("Invalid LinkedIn URL format")
        
        if self.github_url and not re.match(r'^https://github\.com/[\w\-]+/?$', self.github_url):
            raise ValueError("Invalid GitHub URL format")


class JobDescription(BaseValidationModel):
    """Validation for job descriptions"""
    title: str = Field(..., min_length=1, max_length=200, description="Job title")
    company: str = Field(..., min_length=1, max_length=100, description="Company name")
    description: str = Field(..., min_length=10, max_length=10000, description="Job description")
    requirements: List[str] = Field(..., min_items=1, max_items=20, description="Job requirements")
    location: str = Field(..., min_length=1, max_length=100, description="Job location")
    salary_min: Optional[int] = Field(None, ge=0, description="Minimum salary")
    salary_max: Optional[int] = Field(None, ge=0, description="Maximum salary")
    remote: bool = Field(default=False, description="Remote work available")
    experience_level: str = Field(..., description="Experience level required")
    
    def model_post_init(self, __context):
        """Post-initialization validation"""
        # Validate experience level
        valid_levels = ['entry', 'mid', 'senior', 'lead', 'executive']
        if self.experience_level not in valid_levels:
            raise ValueError(f"Invalid experience level. Must be one of: {valid_levels}")
        
        # Sanitize description
        if self.description:
            self.description = bleach.clean(self.description, tags=[], strip=True)
            
            # Check for suspicious content
            suspicious_patterns = [
                r'<script.*?>.*?</script>',
                r'javascript:',
                r'data:',
                r'vbscript:',
                r'on\w+\s*='
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, self.description, re.IGNORECASE):
                    raise ValueError("Job description contains invalid content")
        
        # Sanitize requirements
        if self.requirements:
            sanitized_requirements = []
            for req in self.requirements:
                if req and req.strip():
                    req = html.escape(req.strip())
                    if len(req) > 500:
                        raise ValueError(f"Requirement too long: {req}")
                    sanitized_requirements.append(req)
            
            if not sanitized_requirements:
                raise ValueError("At least one valid requirement is required")
            
            self.requirements = sanitized_requirements
        
        # Validate salary range
        if self.salary_min and self.salary_max and self.salary_min > self.salary_max:
            raise ValueError("Minimum salary cannot be greater than maximum salary")


def validate_input(data: Dict[str, Any], model_class: type) -> BaseValidationModel:
    """Validate input data using specified model"""
    try:
        return model_class(**data)
    except Exception as e:
        raise ValidationError(
            f"Input validation failed: {str(e)}",
            error_code="VALIDATION_ERROR",
            details={"validation_errors": str(e)}
        )


def sanitize_string(value: str) -> str:
    """Sanitize string input"""
    if not isinstance(value, str):
        return value
    
    # Remove HTML tags
    value = bleach.clean(value, tags=[], strip=True)
    
    # Escape special characters once
    value = html.escape(value, quote=True)
    
    return value


def validate_file_upload(filename: str, content_type: str, max_size: int = 10 * 1024 * 1024) -> bool:
    """Validate file upload"""
    # Check file extension
    allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
    file_ext = filename.lower().split('.')[-1]
    if f'.{file_ext}' not in allowed_extensions:
        raise ValidationError(
            f"File type not allowed: {file_ext}",
            error_code="INVALID_FILE_TYPE"
        )
    
    # Check content type
    allowed_content_types = {
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain'
    }
    if content_type not in allowed_content_types:
        raise ValidationError(
            f"Content type not allowed: {content_type}",
            error_code="INVALID_CONTENT_TYPE"
        )
    
    return True
