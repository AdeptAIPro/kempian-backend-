"""
Main pipeline orchestrator for candidate-job matching.
Provides the primary entry point for the matchmaking system.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict

from ..collectors.resume_parser import ResumeParser, ResumeData
from ..collectors.job_parser import JobParser, JobData
from ..collectors.skill_extractor import SkillExtractor
from ..scoring.scorer import Scorer, MatchScore
from ..embeddings.embedder import get_embedder

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Match result with detailed scoring and explanation."""
    candidate_id: str
    score: float
    matched_skills: List[str]
    missing_skills: List[str]
    details: MatchScore
    explanation: str
    candidate_data: Optional[Dict[str, Any]] = None  # Store original candidate data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'candidate_id': self.candidate_id,
            'score': round(self.score, 4),
            'matched_skills': self.matched_skills,
            'missing_skills': self.missing_skills,
            'details': {
                'skill_score': round(self.details.skill_score, 4),
                'experience_score': round(self.details.experience_score, 4),
                'semantic_score': round(self.details.semantic_score, 4),
                'additional_score': round(self.details.additional_score, 4)
            },
            'explanation': self.explanation
        }
        # CRITICAL: Include candidate_data if available (for suggested candidates)
        if self.candidate_data:
            result['candidate_data'] = self.candidate_data
        return result


class CandidateJobMatcher:
    """Main orchestrator for candidate-job matching."""
    
    def __init__(self):
        """Initialize matcher with all required components."""
        self.skill_extractor = SkillExtractor()
        self.resume_parser = ResumeParser(skill_extractor=self.skill_extractor)
        self.job_parser = JobParser(skill_extractor=self.skill_extractor)
        self.scorer = Scorer()
        self.embedder = get_embedder()
    
    def match_candidates(
        self,
        job_description: Union[str, Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[MatchResult]:
        """
        Match candidates to a job description.
        
        Args:
            job_description: Job description text or dictionary with job data
            candidates: List of candidate dictionaries
            top_k: Return only top K results (None for all)
        
        Returns:
            Sorted list of MatchResult objects (highest score first)
        """
        try:
            # Parse job description
            if isinstance(job_description, str):
                job_data = self.job_parser.parse({'job_description': job_description})
            else:
                job_data = self.job_parser.parse(job_description)
            
            # Generate job embedding once
            job_embedding = None
            if self.embedder.is_available():
                job_embedding = self.embedder.encode(job_data.job_description)
            
            # Process each candidate
            results = []
            for candidate in candidates:
                try:
                    result = self._match_single_candidate(
                        candidate,
                        job_data,
                        job_embedding
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error matching candidate {candidate.get('candidate_id', 'unknown')}: {e}")
                    continue
            
            # Sort by score (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Return top K if specified
            if top_k is not None and top_k > 0:
                results = results[:top_k]
            
            return results
        except Exception as e:
            logger.error(f"Error in match_candidates: {e}")
            return []
    
    def _match_single_candidate(
        self,
        candidate: Dict[str, Any],
        job_data: JobData,
        job_embedding: Optional[Any] = None
    ) -> Optional[MatchResult]:
        """
        Match a single candidate to job data.
        
        Args:
            candidate: Candidate dictionary
            job_data: Parsed job data
            job_embedding: Pre-computed job embedding
        
        Returns:
            MatchResult or None if error
        """
        try:
            # Parse candidate resume
            resume_data = self.resume_parser.parse(candidate)
            
            # Generate resume embedding
            resume_embedding = None
            if self.embedder.is_available():
                resume_embedding = self.embedder.encode(resume_data.resume_text)
            
            # Calculate match score
            match_score = self.scorer.calculate_score(
                resume_data,
                job_data,
                resume_embedding,
                job_embedding
            )
            
            # Identify matched and missing skills
            matched_skills, missing_skills = self._analyze_skills(
                resume_data.skills,
                job_data.required_skills,
                job_data.preferred_skills
            )
            
            # Generate explanation
            explanation = self._generate_explanation(
                resume_data,
                job_data,
                match_score,
                matched_skills,
                missing_skills
            )
            
            return MatchResult(
                candidate_id=resume_data.candidate_id,
                score=match_score.overall_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                details=match_score,
                explanation=explanation,
                candidate_data=candidate  # CRITICAL: Store original candidate data
            )
        except Exception as e:
            logger.error(f"Error matching single candidate: {e}")
            return None
    
    def _analyze_skills(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        preferred_skills: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Analyze which skills match and which are missing.
        
        Args:
            candidate_skills: Candidate's skills
            required_skills: Required job skills
            preferred_skills: Preferred job skills
        
        Returns:
            Tuple of (matched_skills, missing_skills)
        """
        candidate_skills_lower = [s.lower() for s in candidate_skills]
        all_required = required_skills + preferred_skills
        
        matched = []
        missing = []
        
        for skill in all_required:
            skill_lower = skill.lower()
            if skill_lower in candidate_skills_lower:
                matched.append(skill)
            else:
                # Check for partial matches
                found = False
                for cand_skill in candidate_skills_lower:
                    if skill_lower in cand_skill or cand_skill in skill_lower:
                        matched.append(skill)
                        found = True
                        break
                if not found:
                    missing.append(skill)
        
        return matched, missing
    
    def _generate_explanation(
        self,
        resume_data: ResumeData,
        job_data: JobData,
        match_score: MatchScore,
        matched_skills: List[str],
        missing_skills: List[str]
    ) -> str:
        """
        Generate human-readable explanation for the match.
        
        Args:
            resume_data: Parsed resume data
            job_data: Parsed job data
            match_score: Calculated match score
            matched_skills: List of matched skills
            missing_skills: List of missing skills
        
        Returns:
            Explanation string
        """
        parts = []
        
        # Overall score
        score_percent = int(match_score.overall_score * 100)
        parts.append(f"Overall match score: {score_percent}%")
        
        # Skill match
        if matched_skills:
            parts.append(f"Matched {len(matched_skills)} required/preferred skills: {', '.join(matched_skills[:5])}")
        if missing_skills:
            parts.append(f"Missing {len(missing_skills)} skills: {', '.join(missing_skills[:5])}")
        
        # Experience
        if resume_data.experience_years > 0:
            if resume_data.experience_years >= job_data.experience_years_required:
                parts.append(f"Meets experience requirement ({resume_data.experience_years:.1f} years)")
            else:
                parts.append(f"Below experience requirement ({resume_data.experience_years:.1f} vs {job_data.experience_years_required:.1f} years)")
        
        # Education
        if resume_data.education and job_data.education_required:
            parts.append("Education requirement met")
        
        # Certifications
        if resume_data.certifications:
            parts.append(f"Has {len(resume_data.certifications)} certification(s)")
        
        # Score breakdown
        parts.append(f"Score breakdown: Skills {int(match_score.skill_score*100)}%, "
                    f"Experience {int(match_score.experience_score*100)}%, "
                    f"Semantic {int(match_score.semantic_score*100)}%")
        
        return ". ".join(parts) + "."


def match_candidates(
    job_description: Union[str, Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to match candidates to a job.
    
    Args:
        job_description: Job description text or dictionary
        candidates: List of candidate dictionaries
        top_k: Return only top K results
    
    Returns:
        List of match result dictionaries
    """
    matcher = CandidateJobMatcher()
    results = matcher.match_candidates(job_description, candidates, top_k)
    return [result.to_dict() for result in results]

