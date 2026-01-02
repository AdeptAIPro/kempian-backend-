# enhanced_candidate_matcher.py - Advanced Candidate Matching System

import re
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from app.simple_logger import get_logger
from dataclasses import dataclass
from collections import defaultdict

from .advanced_query_parser import AdvancedJobQueryParser, ParsedJobQuery, JobRequirement

logger = get_logger("search")

@dataclass
class MatchScore:
    """Detailed matching score breakdown"""
    overall_score: float
    technical_skills_score: float
    experience_score: float
    seniority_score: float
    education_score: float
    soft_skills_score: float
    location_score: float
    confidence: float
    match_explanation: str
    missing_requirements: List[str]
    strength_areas: List[str]

class EnhancedCandidateMatchingSystem:
    """Advanced candidate matching system with intelligent scoring"""
    
    def __init__(self):
        self.query_parser = AdvancedJobQueryParser()
        self.skill_synonyms = self._load_skill_synonyms()
        self.experience_weights = {
            'exact_match': 1.0,
            'over_qualified': 0.9,
            'slightly_under': 0.7,
            'significantly_under': 0.4
        }
        logger.info("EnhancedCandidateMatchingSystem initialized")

    def _load_skill_synonyms(self) -> Dict[str, List[str]]:
        """Load skill synonyms for better matching"""
        return {
            'javascript': ['js', 'ecmascript', 'node.js', 'nodejs'],
            'python': ['py', 'python3', 'django', 'flask'],
            'react': ['reactjs', 'react.js'],
            'angular': ['angularjs', 'angular.js'],
            'machine learning': ['ml', 'artificial intelligence', 'ai', 'deep learning'],
            'aws': ['amazon web services', 'ec2', 's3', 'lambda'],
            'docker': ['containerization', 'containers'],
            'kubernetes': ['k8s', 'container orchestration'],
            'sql': ['mysql', 'postgresql', 'database'],
            'git': ['version control', 'github', 'gitlab'],
            'agile': ['scrum', 'kanban', 'sprint'],
            'api': ['rest api', 'restful', 'graphql'],
            'frontend': ['front-end', 'ui', 'user interface'],
            'backend': ['back-end', 'server-side'],
            'fullstack': ['full-stack', 'full stack'],
            'devops': ['ci/cd', 'deployment', 'infrastructure']
        }

    def match_candidate_to_job(self, candidate_profile: Dict[str, Any], job_query: str) -> MatchScore:
        """Match a candidate to a job with detailed scoring"""
        try:
            # Parse the job query
            parsed_query = self.query_parser.parse_job_query(job_query)
            
            # Get search weights based on query
            weights = self.query_parser.get_search_weights(parsed_query)
            
            # Calculate individual scores
            tech_score = self._calculate_technical_skills_score(candidate_profile, parsed_query)
            exp_score = self._calculate_experience_score(candidate_profile, parsed_query)
            seniority_score = self._calculate_seniority_score(candidate_profile, parsed_query)
            education_score = self._calculate_education_score(candidate_profile, parsed_query)
            soft_skills_score = self._calculate_soft_skills_score(candidate_profile, parsed_query)
            location_score = self._calculate_location_score(candidate_profile, parsed_query)
            
            # Calculate weighted overall score
            overall_score = (
                tech_score * weights['technical_skills'] +
                exp_score * weights['experience_match'] +
                seniority_score * weights['seniority_match'] +
                education_score * weights['education_match'] +
                soft_skills_score * weights['soft_skills'] +
                location_score * weights['location_match']
            )
            
            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(candidate_profile, parsed_query)
            
            # Generate match explanation
            match_explanation = self._generate_detailed_explanation(
                candidate_profile, parsed_query, tech_score, exp_score, seniority_score
            )
            
            # Identify missing requirements and strengths
            missing_requirements = self._identify_missing_requirements(candidate_profile, parsed_query)
            strength_areas = self._identify_strength_areas(candidate_profile, parsed_query)
            
            return MatchScore(
                overall_score=min(100, max(0, overall_score)),
                technical_skills_score=tech_score,
                experience_score=exp_score,
                seniority_score=seniority_score,
                education_score=education_score,
                soft_skills_score=soft_skills_score,
                location_score=location_score,
                confidence=confidence,
                match_explanation=match_explanation,
                missing_requirements=missing_requirements,
                strength_areas=strength_areas
            )
            
        except Exception as e:
            logger.error(f"Error matching candidate: {e}")
            return self._create_fallback_match_score()

    def _calculate_technical_skills_score(self, candidate: Dict[str, Any], parsed_query: ParsedJobQuery) -> float:
        """Calculate technical skills matching score with advanced logic"""
        if not parsed_query.technical_skills:
            return 75.0  # Default if no specific skills mentioned
        
        candidate_skills = self._extract_candidate_skills(candidate)
        if not candidate_skills:
            return 20.0  # Low score if candidate has no skills listed
        
        # Normalize skills for comparison
        candidate_skills_normalized = self._normalize_skills(candidate_skills)
        required_skills_normalized = self._normalize_skills(parsed_query.technical_skills)
        
        # Calculate matches with synonym support
        matches = 0
        total_required = len(required_skills_normalized)
        skill_match_details = []
        
        for required_skill in required_skills_normalized:
            match_found = False
            match_strength = 0
            
            # Direct match
            if required_skill in candidate_skills_normalized:
                matches += 1
                match_found = True
                match_strength = 1.0
            else:
                # Check synonyms
                for candidate_skill in candidate_skills_normalized:
                    similarity = self._calculate_skill_similarity(required_skill, candidate_skill)
                    if similarity > 0.7:
                        matches += similarity
                        match_found = True
                        match_strength = similarity
                        break
            
            skill_match_details.append({
                'skill': required_skill,
                'matched': match_found,
                'strength': match_strength
            })
        
        # Base score calculation
        base_score = (matches / total_required) * 100 if total_required > 0 else 0
        
        # Bonus for having more skills than required
        skill_breadth_bonus = min(10, len(candidate_skills_normalized) - total_required) if len(candidate_skills_normalized) > total_required else 0
        
        # Penalty for missing critical skills (skills mentioned multiple times or with high priority)
        critical_skills_penalty = self._calculate_critical_skills_penalty(parsed_query, skill_match_details)
        
        final_score = base_score + skill_breadth_bonus - critical_skills_penalty
        return min(100, max(0, final_score))

    def _calculate_experience_score(self, candidate: Dict[str, Any], parsed_query: ParsedJobQuery) -> float:
        """Calculate experience matching score"""
        candidate_exp = self._extract_candidate_experience(candidate)
        required_exp = parsed_query.experience_years
        
        if required_exp is None:
            return 80.0  # Default if no specific experience mentioned
        
        if candidate_exp == 0:
            return 30.0  # Low score for no experience when required
        
        # Calculate experience match
        if candidate_exp >= required_exp:
            if candidate_exp <= required_exp + 2:
                return 95.0  # Perfect match
            elif candidate_exp <= required_exp + 5:
                return 90.0  # Slightly over-qualified
            else:
                return 85.0  # Over-qualified
        else:
            # Under-qualified
            gap = required_exp - candidate_exp
            if gap == 1:
                return 75.0  # Close enough
            elif gap == 2:
                return 60.0  # Somewhat under
            elif gap <= 3:
                return 45.0  # Significantly under
            else:
                return 25.0  # Very under-qualified

    def _calculate_seniority_score(self, candidate: Dict[str, Any], parsed_query: ParsedJobQuery) -> float:
        """Calculate seniority level matching score"""
        candidate_seniority = self._extract_candidate_seniority(candidate)
        required_seniority = parsed_query.seniority_level.lower()
        
        seniority_hierarchy = {'junior': 1, 'mid': 2, 'senior': 3}
        
        candidate_level = seniority_hierarchy.get(candidate_seniority.lower(), 2)
        required_level = seniority_hierarchy.get(required_seniority, 2)
        
        if candidate_level == required_level:
            return 95.0
        elif abs(candidate_level - required_level) == 1:
            return 75.0
        else:
            return 50.0

    def _calculate_education_score(self, candidate: Dict[str, Any], parsed_query: ParsedJobQuery) -> float:
        """Calculate education matching score"""
        if not parsed_query.education_requirements:
            return 80.0  # Default if no education requirements
        
        candidate_education = self._extract_candidate_education(candidate)
        if not candidate_education:
            return 60.0  # Default for missing education info
        
        education_lower = candidate_education.lower()
        
        # Check for degree matches
        degree_score = 0
        for req in parsed_query.education_requirements:
            req_lower = req.lower()
            if any(degree in education_lower for degree in ['bachelor', 'master', 'phd', 'doctorate']):
                if any(degree in req_lower for degree in ['bachelor', 'master', 'phd']):
                    degree_score = 90
                    break
            elif 'degree' in req_lower and 'degree' in education_lower:
                degree_score = 75
        
        # Check for field matches (computer science, engineering, etc.)
        field_score = 0
        for req in parsed_query.education_requirements:
            if any(field in education_lower for field in ['computer science', 'engineering', 'mathematics', 'technology']):
                field_score = 85
                break
        
        return max(degree_score, field_score, 60)

    def _calculate_soft_skills_score(self, candidate: Dict[str, Any], parsed_query: ParsedJobQuery) -> float:
        """Calculate soft skills matching score"""
        if not parsed_query.soft_skills:
            return 75.0  # Default if no soft skills mentioned
        
        candidate_text = self._get_candidate_full_text(candidate)
        candidate_text_lower = candidate_text.lower()
        
        matches = 0
        for skill in parsed_query.soft_skills:
            if skill.lower() in candidate_text_lower:
                matches += 1
        
        if len(parsed_query.soft_skills) == 0:
            return 75.0
        
        score = (matches / len(parsed_query.soft_skills)) * 100
        return min(95, max(50, score))

    def _calculate_location_score(self, candidate: Dict[str, Any], parsed_query: ParsedJobQuery) -> float:
        """Calculate location matching score"""
        if not parsed_query.location_info or parsed_query.work_type == 'remote':
            return 90.0  # High score for remote or no location requirement
        
        candidate_location = candidate.get('location', '').lower()
        required_location = parsed_query.location_info.lower()
        
        if not candidate_location:
            return 70.0  # Default for missing location
        
        if required_location in candidate_location or candidate_location in required_location:
            return 95.0
        
        return 60.0  # Different location

    def _calculate_confidence(self, candidate: Dict[str, Any], parsed_query: ParsedJobQuery) -> float:
        """Calculate confidence score based on data completeness"""
        completeness_factors = []
        
        # Check candidate data completeness
        if candidate.get('skills') and len(candidate['skills']) > 0:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.3)
        
        if candidate.get('experience_years', 0) > 0:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.5)
        
        if candidate.get('resume_text') and len(candidate['resume_text']) > 100:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.6)
        
        if candidate.get('education'):
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.7)
        
        # Check query specificity
        query_specificity = 0.8
        if len(parsed_query.technical_skills) > 3:
            query_specificity += 0.1
        if parsed_query.experience_years:
            query_specificity += 0.1
        
        base_confidence = sum(completeness_factors) / len(completeness_factors)
        final_confidence = (base_confidence * 0.7) + (query_specificity * 0.3)
        
        return min(95, max(60, final_confidence * 100))

    def _generate_detailed_explanation(self, candidate: Dict[str, Any], parsed_query: ParsedJobQuery, 
                                     tech_score: float, exp_score: float, seniority_score: float) -> str:
        """Generate detailed match explanation"""
        explanations = []
        
        # Technical skills explanation
        if tech_score >= 80:
            explanations.append(f"Excellent technical skill alignment ({tech_score:.0f}%)")
        elif tech_score >= 60:
            explanations.append(f"Good technical skill match ({tech_score:.0f}%)")
        else:
            explanations.append(f"Limited technical skill overlap ({tech_score:.0f}%)")
        
        # Experience explanation
        candidate_exp = self._extract_candidate_experience(candidate)
        if parsed_query.experience_years:
            if candidate_exp >= parsed_query.experience_years:
                explanations.append(f"Meets experience requirement ({candidate_exp} vs {parsed_query.experience_years} years)")
            else:
                gap = parsed_query.experience_years - candidate_exp
                explanations.append(f"Experience gap of {gap} years ({candidate_exp} vs {parsed_query.experience_years} required)")
        else:
            explanations.append(f"Has {candidate_exp} years of experience")
        
        # Seniority explanation
        candidate_seniority = self._extract_candidate_seniority(candidate)
        if seniority_score >= 90:
            explanations.append(f"Perfect seniority match ({candidate_seniority} level)")
        else:
            explanations.append(f"Seniority level: {candidate_seniority} (seeking {parsed_query.seniority_level})")
        
        return ". ".join(explanations) + "."

    def _identify_missing_requirements(self, candidate: Dict[str, Any], parsed_query: ParsedJobQuery) -> List[str]:
        """Identify missing requirements"""
        missing = []
        
        candidate_skills = self._normalize_skills(self._extract_candidate_skills(candidate))
        
        for required_skill in parsed_query.technical_skills:
            skill_found = False
            normalized_required = required_skill.lower().strip()
            
            for candidate_skill in candidate_skills:
                if self._calculate_skill_similarity(normalized_required, candidate_skill) > 0.7:
                    skill_found = True
                    break
            
            if not skill_found:
                missing.append(required_skill)
        
        # Check experience gap
        candidate_exp = self._extract_candidate_experience(candidate)
        if parsed_query.experience_years and candidate_exp < parsed_query.experience_years:
            gap = parsed_query.experience_years - candidate_exp
            missing.append(f"{gap} more years of experience")
        
        return missing[:5]  # Limit to top 5 missing items

    def _identify_strength_areas(self, candidate: Dict[str, Any], parsed_query: ParsedJobQuery) -> List[str]:
        """Identify candidate's strength areas"""
        strengths = []
        
        candidate_skills = self._extract_candidate_skills(candidate)
        candidate_exp = self._extract_candidate_experience(candidate)
        
        # Technical strengths
        matching_skills = []
        for skill in candidate_skills:
            for required_skill in parsed_query.technical_skills:
                if self._calculate_skill_similarity(skill.lower(), required_skill.lower()) > 0.7:
                    matching_skills.append(skill)
        
        if matching_skills:
            strengths.append(f"Strong in {', '.join(matching_skills[:3])}")
        
        # Experience strength
        if parsed_query.experience_years and candidate_exp > parsed_query.experience_years:
            extra_exp = candidate_exp - parsed_query.experience_years
            strengths.append(f"{extra_exp} years additional experience")
        
        # Additional skills beyond requirements
        extra_skills = [skill for skill in candidate_skills if skill.lower() not in [rs.lower() for rs in parsed_query.technical_skills]]
        if extra_skills:
            strengths.append(f"Additional skills: {', '.join(extra_skills[:2])}")
        
        return strengths[:3]  # Limit to top 3 strengths

    # Helper methods
    def _extract_candidate_skills(self, candidate: Dict[str, Any]) -> List[str]:
        """Extract skills from candidate profile"""
        skills = candidate.get('skills', [])
        if isinstance(skills, str):
            return [s.strip() for s in skills.split(',') if s.strip()]
        elif isinstance(skills, list):
            return [str(s).strip() for s in skills if s]
        return []

    def _extract_candidate_experience(self, candidate: Dict[str, Any]) -> int:
        """Extract experience years from candidate"""
        exp = candidate.get('experience_years', 0) or candidate.get('total_experience_years', 0)
        try:
            return int(float(str(exp))) if exp else 0
        except (ValueError, TypeError):
            return 0

    def _extract_candidate_seniority(self, candidate: Dict[str, Any]) -> str:
        """Extract seniority level from candidate"""
        seniority = candidate.get('seniority_level', '')
        if seniority:
            return seniority
        
        # Infer from experience
        exp = self._extract_candidate_experience(candidate)
        if exp >= 7:
            return 'Senior'
        elif exp >= 3:
            return 'Mid'
        else:
            return 'Junior'

    def _extract_candidate_education(self, candidate: Dict[str, Any]) -> str:
        """Extract education from candidate"""
        return candidate.get('education', '') or ''

    def _get_candidate_full_text(self, candidate: Dict[str, Any]) -> str:
        """Get full text representation of candidate"""
        resume_text = candidate.get('resume_text', '')
        skills_text = ', '.join(self._extract_candidate_skills(candidate))
        education_text = self._extract_candidate_education(candidate)
        
        return f"{resume_text} {skills_text} {education_text}"

    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize skills for better matching"""
        normalized = []
        for skill in skills:
            skill_lower = skill.lower().strip()
            # Remove common prefixes/suffixes
            skill_lower = re.sub(r'\b(experience with|knowledge of|proficient in)\b', '', skill_lower).strip()
            if skill_lower:
                normalized.append(skill_lower)
        return normalized

    def _calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity between two skills"""
        skill1_lower = skill1.lower()
        skill2_lower = skill2.lower()
        
        # Exact match
        if skill1_lower == skill2_lower:
            return 1.0
        
        # Check if one contains the other
        if skill1_lower in skill2_lower or skill2_lower in skill1_lower:
            return 0.9
        
        # Check synonyms
        for main_skill, synonyms in self.skill_synonyms.items():
            if (skill1_lower == main_skill or skill1_lower in synonyms) and \
               (skill2_lower == main_skill or skill2_lower in synonyms):
                return 0.8
        
        # Basic string similarity (Jaccard similarity)
        words1 = set(skill1_lower.split())
        words2 = set(skill2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _calculate_critical_skills_penalty(self, parsed_query: ParsedJobQuery, skill_match_details: List[Dict]) -> float:
        """Calculate penalty for missing critical skills"""
        penalty = 0
        
        # Skills mentioned multiple times or in requirements are critical
        critical_skills = []
        for req in parsed_query.requirements:
            if req.category == 'skill' and req.priority == 'required':
                critical_skills.extend(req.keywords)
        
        for detail in skill_match_details:
            if detail['skill'] in critical_skills and not detail['matched']:
                penalty += 15  # Heavy penalty for missing critical skills
            elif not detail['matched']:
                penalty += 5   # Light penalty for missing non-critical skills
        
        return min(penalty, 40)  # Cap penalty at 40 points

    def _create_fallback_match_score(self) -> MatchScore:
        """Create fallback match score for error cases"""
        return MatchScore(
            overall_score=50.0,
            technical_skills_score=50.0,
            experience_score=50.0,
            seniority_score=50.0,
            education_score=50.0,
            soft_skills_score=50.0,
            location_score=50.0,
            confidence=60.0,
            match_explanation="Unable to perform detailed matching due to processing error",
            missing_requirements=["Unable to determine"],
            strength_areas=["Unable to determine"]
        )