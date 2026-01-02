import re
from typing import Dict, Any, Optional

def extract_required_experience_from_query(job_query: str) -> Optional[int]:
    if not job_query:
        return None
    
    try:
        patterns = [
            r'(\d+)\+?\s*years?\s*experience',
            r'experience:\s*(\d+)\+?',
            r'(\d+)\+?\s*years?\s*in',
            r'minimum\s*(\d+)\s*years'
        ]
        for pattern in patterns:
            try:
                match = re.search(pattern, job_query.lower())
                if match:
                    return int(match.group(1))
            except (re.error, ValueError, AttributeError) as e:
                # Log regex or parsing errors but continue with next pattern
                continue
    except Exception as e:
        # Fallback for any unexpected errors
        return None
    
    return None

def extract_required_seniority_from_query(job_query: str) -> Optional[int]:
    if not job_query:
        return None
    
    try:
        seniority_mapping = {'junior': 1, 'mid': 2, 'senior': 3, 'lead': 4, 'principal': 5}
        query_lower = job_query.lower()
        
        for level, value in seniority_mapping.items():
            if level in query_lower:
                return value
    except (AttributeError, TypeError) as e:
        # Handle cases where job_query is not a string or has no lower() method
        return None
    
    return None

def seniority_to_numeric(seniority: str) -> int:
    try:
        if not seniority or not isinstance(seniority, str):
            return 2  # Default to mid-level
        
        seniority_mapping = {'junior': 1, 'mid': 2, 'senior': 3, 'lead': 4, 'principal': 5}
        return seniority_mapping.get(seniority.lower(), 2)
    except (AttributeError, TypeError) as e:
        # Handle cases where seniority is not a string or has no lower() method
        return 2  # Default to mid-level

def calculate_domain_relevance(candidate_profile: Dict[str, Any], job_query: str) -> float:
    try:
        if not candidate_profile or not job_query:
            return 0.5
        
        candidate_industries = candidate_profile.get('industries', [])
        candidate_roles = candidate_profile.get('previous_roles', [])
        
        if not candidate_industries and not candidate_roles:
            return 0.5
        
        query_lower = job_query.lower()
        relevance_score = 0.0
        
        for industry in candidate_industries:
            try:
                if industry and isinstance(industry, str) and industry.lower() in query_lower:
                    relevance_score += 0.3
            except (AttributeError, TypeError):
                continue
                
        for role in candidate_roles:
            try:
                if role and isinstance(role, str) and role.lower() in query_lower:
                    relevance_score += 0.2
            except (AttributeError, TypeError):
                continue
                
        return min(relevance_score, 1.0)
        
    except Exception as e:
        # Fallback for any unexpected errors
        return 0.5
