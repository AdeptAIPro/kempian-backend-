"""
Patch to integrate enhanced suggestions into the main service.py
Apply this patch to backend/app/search/service.py after line 921
"""

# Add these methods to the FallbackAlgorithm class in service.py

def _calculate_suggestion_similarity(self, job_description: str, candidate_skills: List[str]) -> float:
    """Calculate similarity score for suggested candidates (0-30% range)"""
    try:
        # Extract technical skills from job description
        job_skills = self._extract_skills_from_job(job_description)
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

def _get_similarity_breakdown(self, job_description: str, candidate_skills: List[str]) -> Dict[str, Any]:
    """Get detailed breakdown of similarity for suggested candidates"""
    try:
        job_skills = self._extract_skills_from_job(job_description)
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

# Update the keyword_search method around line 1404 to use these new methods:
# Replace the similarity_score calculation with:
# similarity_score = self._calculate_suggestion_similarity(job_description, technical_skills)
#
# Add similarity_breakdown:
# 'similarity_breakdown': self._get_similarity_breakdown(job_description, technical_skills)
