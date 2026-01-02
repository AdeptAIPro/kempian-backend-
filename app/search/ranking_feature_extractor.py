"""
Ranking Feature Extractor for XGBoost/LightGBM Training
Extracts comprehensive features for learning-to-rank models.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

from app.search.skill_canonicalizer import get_skill_canonicalizer
from app.search.geocoding_service import get_geocoding_service
from app.search.hybrid_embedding_service import get_hybrid_embedding_service

logger = logging.getLogger(__name__)


class RankingFeatureExtractor:
    """Extract features for ranking model training"""
    
    def __init__(self):
        self.skill_canonicalizer = get_skill_canonicalizer()
        self.geocoding_service = get_geocoding_service()
        self.embedding_service = get_hybrid_embedding_service()
    
    def extract_features(
        self,
        job_description: str,
        candidate: Dict[str, Any],
        job_location: Optional[str] = None,
        job_required_skills: Optional[List[str]] = None,
        job_preferred_skills: Optional[List[str]] = None,
        dense_similarity: Optional[float] = None,
        cross_encoder_score: Optional[float] = None,
        tfidf_score: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Extract comprehensive features for ranking
        
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        try:
            # 1. Dense similarity score (bi-encoder)
            features['dense_similarity'] = dense_similarity if dense_similarity is not None else 0.0
            
            # 2. Cross-encoder score
            features['cross_encoder_score'] = cross_encoder_score if cross_encoder_score is not None else 0.0
            
            # 3. TF-IDF cosine score
            features['tfidf_score'] = tfidf_score if tfidf_score is not None else 0.0
            
            # 4. Exact skill count match
            skill_match_features = self._extract_skill_match_features(
                job_description, candidate, job_required_skills, job_preferred_skills
            )
            features.update(skill_match_features)
            
            # 5. Experience match
            features.update(self._extract_experience_features(job_description, candidate))
            
            # 6. Location distance
            features.update(self._extract_location_features(candidate, job_location))
            
            # 7. Certification match
            features.update(self._extract_certification_features(job_description, candidate))
            
            # 8. Education match
            features.update(self._extract_education_features(job_description, candidate))
            
            # 9. Recency features
            features.update(self._extract_recency_features(candidate))
            
            # 10. Data completeness
            features.update(self._extract_completeness_features(candidate))
            
            # 11. Domain alignment
            features.update(self._extract_domain_features(job_description, candidate))
            
            # 12. Seniority match
            features.update(self._extract_seniority_features(job_description, candidate))
            
            # 13. Skill diversity
            features.update(self._extract_skill_diversity_features(candidate))
            
            # 14. Historical interaction (if available)
            features.update(self._extract_interaction_features(candidate))
            
            # 15. Source reliability
            features.update(self._extract_source_features(candidate))
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
        
        return features
    
    def _extract_skill_match_features(
        self,
        job_description: str,
        candidate: Dict[str, Any],
        job_required_skills: Optional[List[str]],
        job_preferred_skills: Optional[List[str]]
    ) -> Dict[str, float]:
        """Extract skill matching features"""
        features = {}
        
        try:
            # Get candidate skills
            candidate_skills_raw = candidate.get('skills', [])
            if isinstance(candidate_skills_raw, str):
                candidate_skills_raw = [s.strip() for s in candidate_skills_raw.split(',')]
            
            # Canonicalize candidate skills
            candidate_skill_ids = []
            for skill in candidate_skills_raw:
                result = self.skill_canonicalizer.canonicalize_skill(skill)
                if result and result[0]:  # skill_id exists
                    candidate_skill_ids.append(result[0])
            
            # Extract job skills if not provided
            if not job_required_skills:
                job_required_skills = self._extract_skills_from_text(job_description)
            
            # Canonicalize job skills
            job_required_skill_ids = []
            for skill in job_required_skills or []:
                result = self.skill_canonicalizer.canonicalize_skill(skill)
                if result and result[0]:
                    job_required_skill_ids.append(result[0])
            
            # Calculate skill match
            if job_required_skill_ids:
                match_score, details = self.skill_canonicalizer.calculate_skill_match_score(
                    job_required_skill_ids, candidate_skill_ids
                )
                
                features['exact_skill_count'] = details['exact_matches']
                features['weighted_skill_match'] = match_score
                features['skill_match_ratio'] = details['match_ratio']
            else:
                features['exact_skill_count'] = 0.0
                features['weighted_skill_match'] = 0.0
                features['skill_match_ratio'] = 0.0
            
            # Preferred skills match
            if job_preferred_skills:
                preferred_skill_ids = []
                for skill in job_preferred_skills:
                    result = self.skill_canonicalizer.canonicalize_skill(skill)
                    if result and result[0]:
                        preferred_skill_ids.append(result[0])
                
                if preferred_skill_ids:
                    preferred_match, _ = self.skill_canonicalizer.calculate_skill_match_score(
                        preferred_skill_ids, candidate_skill_ids
                    )
                    features['preferred_skill_match'] = preferred_match
                else:
                    features['preferred_skill_match'] = 0.0
            else:
                features['preferred_skill_match'] = 0.0
            
            # Total skills count
            features['candidate_skill_count'] = float(len(candidate_skill_ids))
            features['job_skill_count'] = float(len(job_required_skill_ids))
            
        except Exception as e:
            logger.error(f"Error extracting skill features: {e}")
            features.update({
                'exact_skill_count': 0.0,
                'weighted_skill_match': 0.0,
                'skill_match_ratio': 0.0,
                'preferred_skill_match': 0.0,
                'candidate_skill_count': 0.0,
                'job_skill_count': 0.0
            })
        
        return features
    
    def _extract_experience_features(self, job_description: str, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract experience matching features"""
        features = {}
        
        try:
            # Extract experience requirement from job
            job_exp_years = self._extract_experience_years(job_description)
            
            # Get candidate experience
            candidate_exp_years = self._parse_candidate_experience(candidate)
            
            features['candidate_experience_years'] = float(candidate_exp_years)
            features['job_experience_required'] = float(job_exp_years) if job_exp_years else 0.0
            
            # Experience match score
            if job_exp_years:
                if candidate_exp_years >= job_exp_years:
                    features['experience_match'] = 1.0  # Meets or exceeds
                else:
                    gap = job_exp_years - candidate_exp_years
                    if gap <= 1:
                        features['experience_match'] = 0.9
                    elif gap <= 2:
                        features['experience_match'] = 0.7
                    elif gap <= 3:
                        features['experience_match'] = 0.5
                    else:
                        features['experience_match'] = max(0.0, 1.0 - (gap * 0.1))
            else:
                features['experience_match'] = 0.8  # Default if no requirement
            
            # Experience gap
            if job_exp_years:
                features['experience_gap'] = float(max(0, job_exp_years - candidate_exp_years))
            else:
                features['experience_gap'] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting experience features: {e}")
            features.update({
                'candidate_experience_years': 0.0,
                'job_experience_required': 0.0,
                'experience_match': 0.0,
                'experience_gap': 0.0
            })
        
        return features
    
    def _extract_location_features(self, candidate: Dict[str, Any], job_location: Optional[str]) -> Dict[str, float]:
        """Extract location matching features"""
        features = {}
        
        try:
            candidate_location_str = candidate.get('location', '')
            candidate_is_remote = candidate.get('is_remote', False) or candidate.get('remote', False)
            candidate_willing_to_relocate = candidate.get('willing_to_relocate', False)
            
            # Geocode locations
            candidate_loc = self.geocoding_service.geocode_location(candidate_location_str, candidate_is_remote)
            
            if job_location:
                job_loc = self.geocoding_service.geocode_location(job_location, False)
                
                if candidate_loc and job_loc:
                    # Calculate distance
                    distance_km = self.geocoding_service.calculate_distance_km(candidate_loc, job_loc)
                    features['location_distance_km'] = distance_km
                    
                    # Distance score
                    features['location_distance_score'] = self.geocoding_service.calculate_distance_score(
                        distance_km, sigma=50.0
                    )
                    
                    # Same location
                    features['same_location'] = 1.0 if distance_km < 10.0 else 0.0
                else:
                    features['location_distance_km'] = float('inf')
                    features['location_distance_score'] = 0.0
                    features['same_location'] = 0.0
            else:
                features['location_distance_km'] = 0.0
                features['location_distance_score'] = 1.0
                features['same_location'] = 0.0
            
            # Remote flags
            features['candidate_is_remote'] = 1.0 if candidate_is_remote else 0.0
            features['candidate_willing_to_relocate'] = 1.0 if candidate_willing_to_relocate else 0.0
            
            # Remote-eligible alignment
            if candidate_is_remote or candidate_willing_to_relocate:
                features['remote_eligible_alignment'] = 1.0
            else:
                features['remote_eligible_alignment'] = 0.5  # Neutral
            
        except Exception as e:
            logger.error(f"Error extracting location features: {e}")
            features.update({
                'location_distance_km': float('inf'),
                'location_distance_score': 0.0,
                'same_location': 0.0,
                'candidate_is_remote': 0.0,
                'candidate_willing_to_relocate': 0.0,
                'remote_eligible_alignment': 0.0
            })
        
        return features
    
    def _extract_certification_features(self, job_description: str, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract certification matching features"""
        features = {}
        
        try:
            # Extract certifications from job
            job_certs = self._extract_certifications(job_description)
            
            # Get candidate certifications
            candidate_certs = candidate.get('certifications', [])
            if isinstance(candidate_certs, str):
                candidate_certs = [c.strip() for c in candidate_certs.split(',')]
            
            # Match certifications
            if job_certs:
                candidate_certs_lower = [c.lower() for c in candidate_certs]
                matches = sum(1 for cert in job_certs if cert.lower() in ' '.join(candidate_certs_lower))
                features['certification_match'] = float(matches) / len(job_certs) if job_certs else 0.0
                features['certification_match_count'] = float(matches)
            else:
                features['certification_match'] = 0.5  # Neutral if no requirement
                features['certification_match_count'] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting certification features: {e}")
            features.update({
                'certification_match': 0.0,
                'certification_match_count': 0.0
            })
        
        return features
    
    def _extract_education_features(self, job_description: str, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract education matching features"""
        features = {}
        
        try:
            # Extract education requirement from job
            job_education_levels = self._extract_education_levels(job_description)
            
            # Get candidate education
            candidate_education = candidate.get('education', '')
            
            # Match education
            if job_education_levels:
                candidate_edu_lower = candidate_education.lower()
                matches = sum(1 for level in job_education_levels if level.lower() in candidate_edu_lower)
                features['education_match'] = 1.0 if matches > 0 else 0.0
            else:
                features['education_match'] = 0.5  # Neutral if no requirement
            
        except Exception as e:
            logger.error(f"Error extracting education features: {e}")
            features['education_match'] = 0.0
        
        return features
    
    def _extract_recency_features(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract recency features"""
        features = {}
        
        try:
            # Resume update date
            updated_at = candidate.get('updated_at') or candidate.get('last_updated')
            if updated_at:
                if isinstance(updated_at, str):
                    from dateutil.parser import parse
                    updated_at = parse(updated_at)
                
                days_since_update = (datetime.now() - updated_at).days
                features['days_since_resume_update'] = float(days_since_update)
                features['resume_recency_score'] = max(0.0, 1.0 - (days_since_update / 365.0))  # Decay over 1 year
            else:
                features['days_since_resume_update'] = 365.0  # Default to old
                features['resume_recency_score'] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting recency features: {e}")
            features.update({
                'days_since_resume_update': 365.0,
                'resume_recency_score': 0.0
            })
        
        return features
    
    def _extract_completeness_features(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract data completeness features"""
        features = {}
        
        try:
            # Count filled fields
            fields = ['name', 'email', 'skills', 'experience', 'education', 'location', 'resume_text']
            filled_count = sum(1 for field in fields if candidate.get(field))
            
            features['data_completeness'] = float(filled_count) / len(fields)
            
            # Specific completeness
            features['has_email'] = 1.0 if candidate.get('email') else 0.0
            features['has_phone'] = 1.0 if candidate.get('phone') else 0.0
            features['has_resume_text'] = 1.0 if candidate.get('resume_text') else 0.0
            
        except Exception as e:
            logger.error(f"Error extracting completeness features: {e}")
            features.update({
                'data_completeness': 0.0,
                'has_email': 0.0,
                'has_phone': 0.0,
                'has_resume_text': 0.0
            })
        
        return features
    
    def _extract_domain_features(self, job_description: str, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract domain alignment features"""
        features = {}
        
        try:
            # Detect domains
            job_domain = self._detect_domain(job_description)
            candidate_domain = candidate.get('domain') or candidate.get('domain_tag') or candidate.get('category', 'general')
            
            # Domain match
            features['domain_match'] = 1.0 if job_domain.lower() == candidate_domain.lower() else 0.0
            
        except Exception as e:
            logger.error(f"Error extracting domain features: {e}")
            features['domain_match'] = 0.0
        
        return features
    
    def _extract_seniority_features(self, job_description: str, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract seniority matching features"""
        features = {}
        
        try:
            # Detect seniority levels
            job_seniority = self._detect_seniority(job_description)
            candidate_seniority = self._detect_candidate_seniority(candidate)
            
            # Seniority match
            seniority_levels = {'junior': 1, 'mid': 2, 'senior': 3, 'lead': 4, 'principal': 5}
            job_level = seniority_levels.get(job_seniority, 2)
            candidate_level = seniority_levels.get(candidate_seniority, 2)
            
            features['seniority_match_distance'] = float(abs(job_level - candidate_level))
            features['seniority_match'] = 1.0 if job_level == candidate_level else max(0.0, 1.0 - abs(job_level - candidate_level) * 0.3)
            
        except Exception as e:
            logger.error(f"Error extracting seniority features: {e}")
            features.update({
                'seniority_match_distance': 0.0,
                'seniority_match': 0.5
            })
        
        return features
    
    def _extract_skill_diversity_features(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract skill diversity features"""
        features = {}
        
        try:
            skills = candidate.get('skills', [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(',')]
            
            # Skill count
            features['skill_count'] = float(len(skills))
            
            # Skill diversity (entropy) - simplified
            if skills:
                # Count unique skill categories
                categories = set()
                for skill in skills:
                    result = self.skill_canonicalizer.canonicalize_skill(skill)
                    if result and result[0]:
                        skill_obj = self.skill_canonicalizer.skill_ontology.get(result[0])
                        if skill_obj:
                            categories.add(skill_obj.category)
                
                features['skill_diversity'] = float(len(categories)) / max(1, len(skills))
            else:
                features['skill_diversity'] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting skill diversity features: {e}")
            features.update({
                'skill_count': 0.0,
                'skill_diversity': 0.0
            })
        
        return features
    
    def _extract_interaction_features(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract historical interaction features"""
        features = {}
        
        try:
            # These would come from your database/analytics
            features['candidate_response_rate'] = float(candidate.get('response_rate', 0.5))
            features['recruiter_interaction_score'] = float(candidate.get('interaction_score', 0.5))
            
        except Exception as e:
            logger.error(f"Error extracting interaction features: {e}")
            features.update({
                'candidate_response_rate': 0.5,
                'recruiter_interaction_score': 0.5
            })
        
        return features
    
    def _extract_source_features(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract source reliability features"""
        features = {}
        
        try:
            source = candidate.get('source', 'unknown')
            
            # Source reliability scores (customize based on your sources)
            source_scores = {
                'linkedin': 0.9,
                'indeed': 0.8,
                'github': 0.85,
                'stackoverflow': 0.85,
                'internal': 1.0,
                'referral': 0.95,
                'unknown': 0.5
            }
            
            features['source_reliability'] = source_scores.get(source.lower(), 0.5)
            
        except Exception as e:
            logger.error(f"Error extracting source features: {e}")
            features['source_reliability'] = 0.5
        
        return features
    
    # Helper methods
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text"""
        # Simple extraction - can be enhanced with NER
        common_skills = [
            'python', 'javascript', 'java', 'react', 'angular', 'vue', 'node.js',
            'aws', 'docker', 'kubernetes', 'sql', 'mongodb', 'postgresql',
            'spring', 'django', 'flask', 'express', 'typescript', 'c#', 'c++'
        ]
        
        text_lower = text.lower()
        found_skills = [skill for skill in common_skills if skill in text_lower]
        
        return found_skills
    
    def _extract_experience_years(self, text: str) -> Optional[int]:
        """Extract experience requirement from text"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:in|with)',
            r'minimum\s*(?:of\s*)?(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _parse_candidate_experience(self, candidate: Dict[str, Any]) -> int:
        """Parse candidate experience in years"""
        exp = candidate.get('experience_years') or candidate.get('total_experience_years')
        if isinstance(exp, (int, float)):
            return int(exp)
        if isinstance(exp, str):
            match = re.search(r'(\d+)', exp)
            if match:
                return int(match.group(1))
        return 0
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certification requirements"""
        cert_keywords = ['certified', 'certification', 'certificate', 'license', 'licensed']
        # Simple extraction - can be enhanced
        return [kw for kw in cert_keywords if kw in text.lower()]
    
    def _extract_education_levels(self, text: str) -> List[str]:
        """Extract education level requirements"""
        levels = []
        if 'bachelor' in text.lower() or "bachelor's" in text.lower():
            levels.append('bachelor')
        if 'master' in text.lower() or "master's" in text.lower():
            levels.append('master')
        if 'phd' in text.lower() or 'ph.d' in text.lower():
            levels.append('phd')
        return levels
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain from text"""
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['healthcare', 'medical', 'nursing', 'hospital']):
            return 'healthcare'
        if any(kw in text_lower for kw in ['software', 'developer', 'programming', 'tech', 'it']):
            return 'it/tech'
        return 'general'
    
    def _detect_seniority(self, text: str) -> str:
        """Detect seniority level from job description"""
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['principal', 'architect', 'lead']):
            return 'principal'
        if any(kw in text_lower for kw in ['senior', 'sr.', 'sr ']):
            return 'senior'
        if any(kw in text_lower for kw in ['junior', 'jr.', 'entry']):
            return 'junior'
        return 'mid'
    
    def _detect_candidate_seniority(self, candidate: Dict[str, Any]) -> str:
        """Detect candidate seniority from title/experience"""
        title = (candidate.get('title') or candidate.get('current_position') or '').lower()
        exp_years = self._parse_candidate_experience(candidate)
        
        if any(kw in title for kw in ['principal', 'architect', 'lead']):
            return 'principal'
        if exp_years >= 5 or any(kw in title for kw in ['senior', 'sr.', 'sr ']):
            return 'senior'
        if exp_years < 2 or any(kw in title for kw in ['junior', 'jr.', 'entry']):
            return 'junior'
        return 'mid'


# Global instance
_feature_extractor = None

def get_feature_extractor() -> RankingFeatureExtractor:
    """Get or create global feature extractor instance"""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = RankingFeatureExtractor()
    return _feature_extractor

def extract_features(job_description: str, candidate: Dict[str, Any], **kwargs) -> Dict[str, float]:
    """Convenience function to extract features"""
    extractor = get_feature_extractor()
    return extractor.extract_features(job_description, candidate, **kwargs)

