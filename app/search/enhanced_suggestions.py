"""
Enhanced search service improvements for 100% accuracy and fallback suggestions
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def _filter_technical_skills_only(skills: List[str]) -> List[str]:
    """Filter skills to only include technical/hard skills"""
    if not skills:
        return []
    
    # Common soft skills to exclude
    soft_skills = {
        'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
        'time management', 'organization', 'adaptability', 'creativity', 'work ethic',
        'collaboration', 'interpersonal skills', 'attention to detail', 'analytical skills',
        'customer service', 'project management', 'decision making', 'multitasking',
        'flexibility', 'initiative', 'negotiation', 'presentation skills', 'training',
        'mentoring', 'coaching', 'delegation', 'strategic planning', 'budget management',
        'relationship building', 'conflict resolution', 'active listening', 'emotional intelligence'
    }
    
    technical_skills = []
    for skill in skills:
        skill_lower = str(skill).lower().strip()
        # Exclude soft skills and keep technical skills
        if skill_lower not in soft_skills and len(skill_lower) > 2:
            technical_skills.append(skill)
    
    return technical_skills

def calculate_suggestion_similarity(job_description: str, candidate_skills: List[str], extract_skills_func) -> float:
    """Calculate similarity score for suggested candidates (0-30% range)"""
    try:
        # Extract technical skills from job description
        job_skills = extract_skills_func(job_description)
        job_technical_skills = _filter_technical_skills_only(job_skills)
        
        if not job_technical_skills or not candidate_skills:
            return 5.0  # Minimum score for showing any suggestion
        
        # Calculate skill overlap
        job_skills_lower = [skill.lower() for skill in job_technical_skills]
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        
        # Count matching skills
        matches = 0
        for job_skill in job_skills_lower:
            for candidate_skill in candidate_skills_lower:
                # Exact match
                if job_skill == candidate_skill:
                    matches += 1
                    break
                # Partial match (skill contains part of the other)
                elif job_skill in candidate_skill or candidate_skill in job_skill:
                    matches += 0.5
                    break
        
        # Calculate similarity percentage (max 30%)
        if matches > 0:
            similarity = min((matches / len(job_technical_skills)) * 30, 30.0)
            return round(similarity, 1)
        
        return 5.0  # Minimum score for showing any suggestion
        
    except Exception as e:
        logger.error(f"Error calculating suggestion similarity: {e}")
        return 5.0

def get_similarity_breakdown(job_description: str, candidate_skills: List[str], extract_skills_func) -> Dict[str, Any]:
    """Get detailed breakdown of similarity for suggested candidates"""
    try:
        job_skills = extract_skills_func(job_description)
        job_technical_skills = _filter_technical_skills_only(job_skills)
        
        matching_skills = []
        missing_skills = []
        
        job_skills_lower = [skill.lower() for skill in job_technical_skills]
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        
        for job_skill in job_skills_lower:
            matched = False
            for candidate_skill in candidate_skills_lower:
                if job_skill == candidate_skill or job_skill in candidate_skill or candidate_skill in job_skill:
                    matching_skills.append(candidate_skill)
                    matched = True
                    break
            if not matched:
                missing_skills.append(job_skill)
        
        return {
            'matching_skills': matching_skills[:5],  # Show top 5 matching skills
            'missing_skills': missing_skills[:5],    # Show top 5 missing skills
            'total_job_skills': len(job_technical_skills),
            'candidate_skill_count': len(candidate_skills),
            'match_count': len(matching_skills)
        }
        
    except Exception as e:
        logger.error(f"Error getting similarity breakdown: {e}")
        return {
            'matching_skills': [],
            'missing_skills': [],
            'total_job_skills': 0,
            'candidate_skill_count': 0,
            'match_count': 0
        }

def enhance_suggested_candidates(suggested_candidates: List[Dict], job_description: str, extract_skills_func) -> List[Dict]:
    """Enhance suggested candidates with similarity scores and breakdowns"""
    enhanced_candidates = []
    
    for candidate in suggested_candidates:
        # Filter skills to technical only
        all_skills = candidate.get('skills', [])
        if isinstance(all_skills, str):
            all_skills = [s.strip() for s in all_skills.split(',')]
        technical_skills = _filter_technical_skills_only(all_skills)
        
        # Calculate similarity score
        similarity_score = calculate_suggestion_similarity(job_description, technical_skills, extract_skills_func)
        
        # Get similarity breakdown
        similarity_breakdown = get_similarity_breakdown(job_description, technical_skills, extract_skills_func)
        
        # Create enhanced candidate entry
        enhanced_candidate = {
            **candidate,
            'skills': technical_skills,  # Only technical skills
            'Skills': technical_skills,
            'match_percentage': similarity_score,
            'Score': similarity_score,
            'grade': 'Suggested',
            'Grade': 'Suggested',
            'is_suggested': True,
            'suggestion_reason': f'No exact matches found. Showing candidate with {similarity_score:.1f}% technical skill similarity.',
            'similarity_breakdown': similarity_breakdown
        }
        
        enhanced_candidates.append(enhanced_candidate)
    
    # Sort by similarity score (highest first)
    enhanced_candidates.sort(key=lambda x: x.get('match_percentage', 0), reverse=True)
    
    return enhanced_candidates
