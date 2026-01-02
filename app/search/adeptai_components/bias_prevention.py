# bias_prevention.py - FIXED VERSION without circular imports

import re
import json
import numpy as np
import hashlib
from datetime import datetime
from app.simple_logger import get_logger
from typing import Dict, List, Any, Optional
import logging
from collections import Counter 

logger = get_logger("search")

class BiasPreventionProcessor:
    """
    Core bias prevention processor for fair recruitment search
    """
    
    def __init__(self):
        # Sensitive patterns that could reveal protected characteristics
        self.sensitive_patterns = {
            'names': self._build_name_patterns(),
            'locations': self._build_location_patterns(),
            'demographic_indicators': self._build_demographic_patterns(),
            'protected_associations': self._build_protected_association_patterns()
        }
        
        # Replacement tokens for anonymization
        self.replacement_tokens = {
            'CANDIDATE_NAME': '[CANDIDATE]',
            'LOCATION': '[LOCATION]',
            'ORGANIZATION': '[ORG]',
            'SCHOOL': '[EDU_INST]',
            'PHONE': '[PHONE]',
            'EMAIL': 'candidate@domain.com'
        }
    
    def _build_name_patterns(self):
        """Build patterns to identify names that could reveal protected characteristics"""
        return [
            # Ethnic/cultural name patterns
            r'\b(von|van|de la|del|ibn|bin|singh|kumar|patel|o\'|mc|mac)\w+\b',
            # Religious titles
            r'\b(rabbi|imam|pastor|father|sister|brother)\s+\w+\b',
            # Cultural suffixes
            r'\b\w+(sr|jr|iii|iv|phd|md|esq)\b'
        ]
    
    def _build_location_patterns(self):
        """Build patterns for location-based bias indicators"""
        return [
            # Full addresses
            r'\d+\s+[\w\s]+(street|st\.?|avenue|ave\.?|road|rd\.?|drive|dr\.?|lane|ln\.?)',
            # ZIP codes
            r'\b\d{5}(-\d{4})?\b',
            # Cities that might indicate demographics
            r'\b(detroit|compton|beverly\s+hills|manhattan|bronx|brooklyn|chicago|atlanta)\b',
            # Countries/regions
            r'\b(india|pakistan|nigeria|mexico|china|bangladesh|philippines|vietnam)\b'
        ]
    
    def _build_demographic_patterns(self):
        """Patterns that might reveal demographic information"""
        return [
            # Age indicators
            r'\b(born\s+(in\s+)?\d{4}|age\s*:?\s*\d+|\d+\s+years?\s+old)\b',
            # Marital/family status
            r'\b(married|single|divorced|widowed|spouse|husband|wife|mother|father|parent)\b',
            # Military service (protected in some contexts)
            r'\b(veteran|military|army|navy|air\s+force|marines|coast\s+guard)\b',
            # Photo references
            r'\.(jpg|jpeg|png|gif|bmp|tiff)(\s|$)',
            # Personal pronouns that might indicate gender
            r'\b(he|she|his|her|him)\b(?!\s+(company|organization|team|department))'
        ]
    
    def _build_protected_association_patterns(self):
        """Patterns indicating protected group associations"""
        return [
            # Religious organizations
            r'\b(church|mosque|synagogue|temple|cathedral|parish|congregation)\b',
            # Ethnic/cultural organizations
            r'\b(hispanic|latino|african|asian|native|indigenous)\s+(american|society|association|council|organization)\b',
            # Gender-specific organizations
            r'\b(women\'?s|ladies|sorority|fraternity|brotherhood|sisterhood)\b',
            # LGBTQ+ organizations
            r'\b(lgbt|lgbtq|gay|lesbian|pride|rainbow|equality)\b',
            # Disability-related organizations
            r'\b(disability|disabled|handicapped|accessibility|special\s+needs)\b'
        ]
    
    def sanitize_resume_data(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove or anonymize sensitive information from resume data
        """
        sanitized_data = resume_data.copy()
        
        # Anonymize personal identifiers
        sanitized_data = self._anonymize_personal_info(sanitized_data)
        
        # Sanitize resume text content
        sanitized_data = self._sanitize_text_content(sanitized_data)
        
        # Remove photo/image references
        sanitized_data = self._remove_visual_identifiers(sanitized_data)
        
        # Add bias processing metadata
        sanitized_data['bias_processed'] = True
        sanitized_data['sanitization_timestamp'] = datetime.now().isoformat()
        
        return sanitized_data
    
    def _anonymize_personal_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize personal identifying information"""
        
        # Handle both field name variations
        name_fields = ['full_name', 'FullName', 'name']
        for field in name_fields:
            if field in data:
                data[field] = self.replacement_tokens['CANDIDATE_NAME']
        
        # Anonymize location fields
        location_fields = ['address', 'current_location', 'location', 'city', 'state', 'country']
        for field in location_fields:
            if field in data:
                data[field] = self.replacement_tokens['LOCATION']
        
        # Anonymize source URL
        url_fields = ['sourceURL', 'SourceURL', 'source_url']
        for field in url_fields:
            if field in data:
                data[field] = '[SOURCE_URL]'
        
        # Handle phone numbers
        phone_fields = ['phone', 'phone_number', 'contact_number']
        for field in phone_fields:
            if field in data:
                data[field] = self.replacement_tokens['PHONE']
        
        # Partially anonymize email
        email_fields = ['email', 'email_address', 'contact_email']
        for field in email_fields:
            if field in data and data[field]:
                email = data[field]
                if '@' in email:
                    domain = email.split('@')[1]
                    data[field] = f"candidate@{domain}"
                else:
                    data[field] = self.replacement_tokens['EMAIL']
        
        return data
    
    def _sanitize_text_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize text content in resume and related fields"""
        text_fields = ['resume_text', 'ResumeText', 'description', 'summary']
        for field in text_fields:
            if field in data and data[field]:
                sanitized_text = self._apply_sanitization_patterns(data[field])
                data[field] = sanitized_text
        return data
    
    def _apply_sanitization_patterns(self, text: str) -> str:
        """Apply all sanitization patterns to text"""
        sanitized_text = text
        
        # Remove name patterns
        for pattern in self.sensitive_patterns['names']:
            sanitized_text = re.sub(pattern, '[NAME]', sanitized_text, flags=re.IGNORECASE)
        
        # Remove location patterns
        for pattern in self.sensitive_patterns['locations']:
            sanitized_text = re.sub(pattern, '[LOCATION]', sanitized_text, flags=re.IGNORECASE)
        
        # Remove demographic indicators
        for pattern in self.sensitive_patterns['demographic_indicators']:
            sanitized_text = re.sub(pattern, '[DEMOGRAPHIC_INDICATOR]', sanitized_text, flags=re.IGNORECASE)
            
        # Remove protected associations
        for pattern in self.sensitive_patterns['protected_associations']:
            sanitized_text = re.sub(pattern, '[PROTECTED_ASSOCIATION]', sanitized_text, flags=re.IGNORECASE)
            
        # Generic phone number pattern
        sanitized_text = re.sub(r'\b(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4})\b', '[PHONE_NUMBER]', sanitized_text)
        
        # Generic email pattern
        sanitized_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email@domain.com', sanitized_text)
        
        return sanitized_text
    
    def _remove_visual_identifiers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove references to visual identifiers like photos."""
        image_fields = ['photo_url', 'profile_picture', 'image_link']
        for field in image_fields:
            if field in data:
                data[field] = '[IMAGE_REMOVED]'
        
        if 'resume_text' in data and data['resume_text']:
            data['resume_text'] = re.sub(r'\b(photo|image|picture|headshot|avatar)\b', '[VISUAL_REFERENCE_REMOVED]', data['resume_text'], flags=re.IGNORECASE)
            
        return data

    def sanitize_query(self, query: str) -> str:
        """Sanitize search query to prevent bias from explicit terms."""
        sanitized_query = query
        
        # Remove direct mentions of protected characteristics
        patterns_to_remove = [
            r'\b(male|female|man|woman|boy|girl|transgender|non-binary)\b',
            r'\b(asian|african|hispanic|caucasian|white|black|indian)\b',
            r'\b(christian|muslim|jewish|hindu|buddhist)\b',
            r'\b(young|old|senior|junior|recent\s+grad)\b',
            r'\b(married|single|divorced)\b',
            r'\b(veteran|military)\b',
            r'\b(disabled|handicapped)\b'
        ]
        
        for pattern in patterns_to_remove:
            sanitized_query = re.sub(pattern, '', sanitized_query, flags=re.IGNORECASE).strip()
            
        # Clean up extra spaces
        sanitized_query = re.sub(r'\s+', ' ', sanitized_query).strip()
        
        logger.info(f"Query sanitized: '{query}' -> '{sanitized_query}'")
        return sanitized_query

class BiasMonitor:
    """
    Monitors search results for potential bias indicators
    """
    def __init__(self):
        self.bias_metrics = {
            'query_count': 0,
            'flagged_queries': 0,
            'bias_score_history': []
        }
        logger.info("BiasMonitor initialized")

    def monitor_search_results(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Monitor search results for potential biases
        """
        self.bias_metrics['query_count'] += 1
        
        assessment = {
            "query": query,
            "result_count": len(results),
            "potential_bias_flags": [],
            "diversity_metrics": {
                "seniority_levels": Counter(),
                "grades": Counter(),
                "confidence_levels": Counter()
            },
            "bias_detected": False
        }

        # Check for bias indicators in query
        bias_flags = self._check_query_bias(query)
        assessment['potential_bias_flags'].extend(bias_flags)
        
        if bias_flags:
            assessment['bias_detected'] = True
            self.bias_metrics['flagged_queries'] += 1

        # Analyze result diversity
        for result in results:
            # Monitor diversity of returned results
            if 'SeniorityLevel' in result:
                assessment['diversity_metrics']['seniority_levels'][result['SeniorityLevel']] += 1
            if 'Grade' in result:
                assessment['diversity_metrics']['grades'][result['Grade']] += 1
            if 'Confidence' in result:
                confidence_band = 'High' if result['Confidence'] > 80 else ('Medium' if result['Confidence'] > 60 else 'Low')
                assessment['diversity_metrics']['confidence_levels'][confidence_band] += 1
        
        # Calculate bias score based on diversity
        bias_score = self._calculate_bias_score(assessment)
        assessment['bias_score'] = bias_score
        self.bias_metrics['bias_score_history'].append(bias_score)

        if bias_score > 50:
            assessment['bias_detected'] = True
            assessment['potential_bias_flags'].append({
                'type': 'low_diversity',
                'description': f'Low diversity in results (bias score: {bias_score})'
            })

        logger.info(f"Bias monitoring complete for query '{query}'. Bias detected: {assessment['bias_detected']}")
        return assessment

    def _check_query_bias(self, query: str) -> List[Dict[str, str]]:
        """Check query for bias indicators"""
        flags = []
        query_lower = query.lower()
        
        # Check for protected characteristic mentions
        protected_terms = [
            (['male', 'female', 'man', 'woman'], 'gender'),
            (['young', 'old', 'age'], 'age'),
            (['asian', 'african', 'hispanic', 'white', 'black'], 'ethnicity'),
            (['christian', 'muslim', 'jewish'], 'religion')
        ]
        
        for terms, category in protected_terms:
            if any(term in query_lower for term in terms):
                flags.append({
                    'type': f'{category}_bias',
                    'description': f'Query contains {category}-related terms'
                })
        
        return flags

    def _calculate_bias_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate a bias score based on result diversity"""
        score = 0
        result_count = assessment['result_count']
        
        if result_count == 0:
            return 0
        
        # Check seniority level diversity
        seniority_levels = assessment['diversity_metrics']['seniority_levels']
        if len(seniority_levels) > 1:
            most_common_count = seniority_levels.most_common(1)[0][1]
            if most_common_count / result_count > 0.8:  # If one level dominates
                score += 30
        
        # Check grade diversity
        grades = assessment['diversity_metrics']['grades']
        if len(grades) > 1:
            most_common_count = grades.most_common(1)[0][1]
            if most_common_count / result_count > 0.9:  # If one grade dominates
                score += 20
        
        return min(score, 100)  # Cap at 100

    def get_bias_metrics(self) -> Dict[str, Any]:
        """Get current bias monitoring metrics"""
        avg_bias_score = np.mean(self.bias_metrics['bias_score_history']) if self.bias_metrics['bias_score_history'] else 0
        
        return {
            'total_queries': self.bias_metrics['query_count'],
            'flagged_queries': self.bias_metrics['flagged_queries'],
            'flag_rate': (self.bias_metrics['flagged_queries'] / max(self.bias_metrics['query_count'], 1)) * 100,
            'average_bias_score': round(avg_bias_score, 2),
            'recent_scores': self.bias_metrics['bias_score_history'][-10:]  # Last 10 scores
        }

def bias_config() -> Dict[str, Any]:
    """Return bias prevention configuration"""
    return {
        "anonymization_enabled": True,
        "query_sanitization_enabled": True,
        "monitoring_enabled": True,
        "fair_embedding_enabled": False
    }

def sanitize_dynamodb_data_batch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sanitizes a batch of DynamoDB items using BiasPreventionProcessor
    """
    processor = BiasPreventionProcessor()
    sanitized_items = []
    
    for item in items:
        try:
            sanitized_item = processor.sanitize_resume_data(item)
            sanitized_items.append(sanitized_item)
        except Exception as e:
            logger.error(f"Error sanitizing DynamoDB item {item.get('candidate_id', 'N/A')}: {e}")
            sanitized_items.append(item)  # Keep original if sanitization fails
    
    logger.info(f"Sanitized {len(sanitized_items)} out of {len(items)} DynamoDB items.")
    return sanitized_items

# Factory functions for creating bias prevention components
def create_bias_processor() -> BiasPreventionProcessor:
    """Create a new bias prevention processor"""
    return BiasPreventionProcessor()

def create_bias_monitor() -> BiasMonitor:
    """Create a new bias monitor"""
    return BiasMonitor()

# Integration helper function
def integrate_bias_prevention_with_search(search_system_instance):
    """
    Integrate bias prevention with search system
    """
    if search_system_instance is None:
        logger.warning("⚠️ Search system instance is None, bias prevention not integrated.")
        return False

    try:
        bias_processor = create_bias_processor()
        bias_monitor = create_bias_monitor()

        # Store original search method
        original_search_method = getattr(search_system_instance, 'search', None)
        if not original_search_method:
            logger.error("❌ Search system has no 'search' method")
            return False

        def bias_aware_search(query: str, top_k: int = 10, **kwargs):
            # Sanitize query
            sanitized_query = bias_processor.sanitize_query(query)
            
            # Perform search with sanitized query
            search_results = original_search_method(sanitized_query, top_k, **kwargs)
            
            # Handle different return formats
            if isinstance(search_results, tuple) and len(search_results) >= 3:
                # Format: (faiss_results, semantic_scores, candidate_data_map)
                faiss_results, semantic_scores, candidate_data_map = search_results[:3]
                
                # Convert to list format for monitoring
                results_for_monitoring = []
                for score, candidate_id in faiss_results:
                    candidate_details = candidate_data_map.get(candidate_id, {})
                    results_for_monitoring.append({
                        "candidate_id": candidate_id,
                        "Score": score,
                        "Grade": candidate_details.get('Grade', 'N/A'),
                        "SeniorityLevel": candidate_details.get('SeniorityLevel', 'N/A'),
                        "Confidence": candidate_details.get('Confidence', 75)
                    })
                
                # Monitor for bias
                bias_assessment = bias_monitor.monitor_search_results(query, results_for_monitoring)
                
                # Store assessment in search system
                search_system_instance.latest_bias_assessment = bias_assessment
                
                return search_results
            else:
                # Handle list format results
                if isinstance(search_results, list):
                    bias_assessment = bias_monitor.monitor_search_results(query, search_results)
                    search_system_instance.latest_bias_assessment = bias_assessment
                
                return search_results

        # Replace the search method
        search_system_instance.search = bias_aware_search
        search_system_instance.bias_aware_search_active = True
        search_system_instance.bias_processor = bias_processor
        search_system_instance.bias_monitor = bias_monitor
        
        logger.info("✅ Bias prevention integrated into search system")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to integrate bias prevention: {e}")
        return False