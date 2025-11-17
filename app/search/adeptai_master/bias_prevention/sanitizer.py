import re
from datetime import datetime
from typing import Dict, Any, List
from .patterns import RACE_PATTERNS, CULTURE_PATTERNS, RELIGION_PATTERNS, PROTECTED_CHARACTERISTIC_PATTERNS

class ResumeSanitizer:
    """Removes race, culture, and religion bias-related info from resumes while preserving professional information"""

    def __init__(self):
        self.protected_patterns = PROTECTED_CHARACTERISTIC_PATTERNS
        # Fields that should NOT be sanitized (professional information)
        self.preserved_fields = {
            'phone', 'email', 'location', 'education', 'skills', 'experience',
            'certifications', 'languages', 'projects', 'achievements'
        }

    def sanitize_resume(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize resume data focusing only on race, culture, and religion identifiers"""
        data = self._sanitize_protected_fields(data)
        data = self._sanitize_text_content(data)
        data['bias_processed'] = True
        data['timestamp'] = datetime.now().isoformat()
        data['sanitization_scope'] = 'race_culture_religion_only'
        return data

    def _sanitize_protected_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize only fields that contain race, culture, or religion identifiers"""
        # Only sanitize name field if it contains protected characteristics
        if 'full_name' in data and self._contains_protected_characteristics(data['full_name']):
            data['full_name'] = '[CANDIDATE_NAME]'
            data['name_sanitized'] = True
        
        # Only sanitize organization field if it contains religious/cultural institutions
        if 'organization' in data and self._contains_protected_characteristics(data['organization']):
            data['organization'] = '[ORGANIZATION]'
            data['org_sanitized'] = True
        
        # Only sanitize school field if it contains religious/cultural institutions
        if 'school' in data and self._contains_protected_characteristics(data['school']):
            data['school'] = '[EDUCATION_INSTITUTION]'
            data['school_sanitized'] = True
        
        return data

    def _sanitize_text_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize text content focusing only on race, culture, and religion patterns"""
        if 'resume_text' not in data:
            return data
        
        text = data['resume_text']
        original_text = text
        
        # Apply sanitization for each protected characteristic category
        for category, patterns in self.protected_patterns.items():
            for pattern in patterns:
                # Use category-specific replacement tokens
                replacement = f'[{category.upper()}_IDENTIFIER]'
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Only update if changes were made
        if text != original_text:
            data['resume_text'] = text
            data['text_sanitized'] = True
            data['sanitization_details'] = self._get_sanitization_summary(original_text, text)
        
        return data

    def _contains_protected_characteristics(self, text: str) -> bool:
        """Check if text contains any race, culture, or religion identifiers"""
        if not isinstance(text, str):
            return False
        
        for patterns in self.protected_patterns.values():
            for pattern in patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return True
        return False

    def _get_sanitization_summary(self, original_text: str, sanitized_text: str) -> Dict[str, Any]:
        """Generate summary of what was sanitized"""
        summary = {
            'characters_removed': len(original_text) - len(sanitized_text),
            'protected_characteristics_found': [],
            'sanitization_timestamp': datetime.now().isoformat()
        }
        
        # Identify which categories were found and sanitized
        for category, patterns in self.protected_patterns.items():
            for pattern in patterns:
                if re.search(pattern, original_text, flags=re.IGNORECASE):
                    if category not in summary['protected_characteristics_found']:
                        summary['protected_characteristics_found'].append(category)
        
        return summary

class QuerySanitizer:
    """Cleans search queries from race, culture, and religion bias terms only"""

    def __init__(self):
        self.protected_patterns = PROTECTED_CHARACTERISTIC_PATTERNS

    def sanitize_query(self, query: str) -> str:
        """Remove only race, culture, and religion bias terms from queries"""
        if not isinstance(query, str):
            return query
        
        original_query = query
        sanitized_query = query
        
        # Apply sanitization for each protected characteristic category
        for category, patterns in self.protected_patterns.items():
            for pattern in patterns:
                # Remove the bias term completely
                sanitized_query = re.sub(pattern, '', sanitized_query, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        sanitized_query = re.sub(r'\s+', ' ', sanitized_query).strip()
        
        # Only return sanitized version if changes were made
        if sanitized_query != original_query:
            return sanitized_query
        
        return original_query

    def get_sanitization_report(self, query: str) -> Dict[str, Any]:
        """Generate report of what was sanitized from the query"""
        if not isinstance(query, str):
            return {'error': 'Query must be a string'}
        
        report = {
            'original_query': query,
            'protected_characteristics_found': [],
            'sanitization_applied': False
        }
        
        # Check which protected characteristics are present
        for category, patterns in self.protected_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, flags=re.IGNORECASE):
                    if category not in report['protected_characteristics_found']:
                        report['protected_characteristics_found'].append(category)
        
        report['sanitization_applied'] = len(report['protected_characteristics_found']) > 0
        
        return report
