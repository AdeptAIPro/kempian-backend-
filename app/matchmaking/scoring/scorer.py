"""
Multi-factor weighted scoring engine.
Calculates comprehensive match scores with explainability.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..utils.similarity import cosine_similarity
from ..embeddings.embedder import get_embedder
from ..collectors.resume_parser import ResumeData
from ..collectors.job_parser import JobData

logger = logging.getLogger(__name__)


@dataclass
class MatchScore:
    """Detailed match score breakdown."""
    skill_score: float
    experience_score: float
    semantic_score: float
    additional_score: float
    overall_score: float


class Scorer:
    """Multi-factor weighted scoring engine."""
    
    # Weight configuration
    SKILL_WEIGHT = 0.40
    EXPERIENCE_WEIGHT = 0.20
    SEMANTIC_WEIGHT = 0.30
    ADDITIONAL_WEIGHT = 0.10
    
    def __init__(self):
        """Initialize scorer."""
        self.embedder = get_embedder()
    
    def calculate_score(
        self,
        resume_data: ResumeData,
        job_data: JobData,
        resume_embedding: Optional[Any] = None,
        job_embedding: Optional[Any] = None
    ) -> MatchScore:
        """
        Calculate comprehensive match score.
        
        Args:
            resume_data: Parsed resume data
            job_data: Parsed job data
            resume_embedding: Pre-computed resume embedding (optional)
            job_embedding: Pre-computed job embedding (optional)
        
        Returns:
            MatchScore with detailed breakdown
        """
        try:
            # Calculate skill match score
            skill_score = self._calculate_skill_score(
                resume_data.skills,
                job_data.required_skills,
                job_data.preferred_skills
            )
            
            # Calculate experience fit score
            experience_score = self._calculate_experience_score(
                resume_data.experience_years,
                job_data.experience_years_required
            )
            
            # Calculate semantic embedding score
            semantic_score = self._calculate_semantic_score(
                resume_data.resume_text,
                job_data.job_description,
                resume_embedding,
                job_embedding
            )
            
            # Calculate additional factors score
            additional_score = self._calculate_additional_score(
                resume_data,
                job_data
            )
            
            # Calculate weighted overall score
            overall_score = (
                skill_score * self.SKILL_WEIGHT +
                experience_score * self.EXPERIENCE_WEIGHT +
                semantic_score * self.SEMANTIC_WEIGHT +
                additional_score * self.ADDITIONAL_WEIGHT
            )
            
            # Ensure score is in [0, 1] range
            overall_score = max(0.0, min(1.0, overall_score))
            
            return MatchScore(
                skill_score=skill_score,
                experience_score=experience_score,
                semantic_score=semantic_score,
                additional_score=additional_score,
                overall_score=overall_score
            )
        except Exception as e:
            logger.error(f"Error calculating score: {e}")
            return MatchScore(
                skill_score=0.0,
                experience_score=0.0,
                semantic_score=0.0,
                additional_score=0.0,
                overall_score=0.0
            )
    
    def _calculate_skill_score(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        preferred_skills: List[str]
    ) -> float:
        """
        Calculate skill match score using canonicalization and fuzzy matching.
        
        Formula: (matched_required_skills / total_required_skills) * 0.8 + 
                 (matched_preferred_skills / total_preferred_skills) * 0.2
        
        Args:
            candidate_skills: List of candidate skills
            required_skills: List of required job skills
            preferred_skills: List of preferred job skills
        
        Returns:
            Skill match score (0-1)
        """
        if not required_skills and not preferred_skills:
            # No skills specified, return neutral score
            return 0.5
        
        # Canonicalize all skills for better matching
        from ..collectors.skill_extractor import SkillExtractor
        skill_extractor = SkillExtractor()
        
        # Canonicalize candidate skills
        candidate_canonical = set()
        for skill in candidate_skills:
            canonical = skill_extractor.canonicalize_skill(skill)
            if canonical:
                skill_info = skill_extractor.get_skill_info(canonical)
                if skill_info:
                    candidate_canonical.add(skill_info['canonical_name'].lower())
            else:
                # Keep original if not found in ontology
                candidate_canonical.add(skill.lower())
        
        # Canonicalize required skills
        required_canonical = []
        for skill in required_skills:
            canonical = skill_extractor.canonicalize_skill(skill)
            if canonical:
                skill_info = skill_extractor.get_skill_info(canonical)
                if skill_info:
                    required_canonical.append(skill_info['canonical_name'].lower())
                else:
                    required_canonical.append(skill.lower())
            else:
                required_canonical.append(skill.lower())
        
        # Canonicalize preferred skills
        preferred_canonical = []
        for skill in preferred_skills:
            canonical = skill_extractor.canonicalize_skill(skill)
            if canonical:
                skill_info = skill_extractor.get_skill_info(canonical)
                if skill_info:
                    preferred_canonical.append(skill_info['canonical_name'].lower())
                else:
                    preferred_canonical.append(skill.lower())
            else:
                preferred_canonical.append(skill.lower())
        
        # Match required skills with fuzzy matching
        matched_required = 0
        for req_skill in required_canonical:
            req_skill_lower = req_skill.lower()
            if req_skill_lower in candidate_canonical:
                matched_required += 1.0
            else:
                # Fuzzy match with threshold
                best_match_score = 0.0
                for cand_skill in candidate_canonical:
                    # Use string similarity for fuzzy matching
                    similarity = self._fuzzy_skill_match(req_skill_lower, cand_skill)
                    if similarity > best_match_score:
                        best_match_score = similarity
                
                if best_match_score >= 0.8:  # High confidence match
                    matched_required += 1.0
                elif best_match_score >= 0.6:  # Medium confidence match
                    matched_required += 0.7
                elif best_match_score >= 0.4:  # Low confidence match
                    matched_required += 0.3
        
        # Match preferred skills with fuzzy matching
        matched_preferred = 0
        for pref_skill in preferred_canonical:
            pref_skill_lower = pref_skill.lower()
            if pref_skill_lower in candidate_canonical:
                matched_preferred += 1.0
            else:
                # Fuzzy match
                best_match_score = 0.0
                for cand_skill in candidate_canonical:
                    similarity = self._fuzzy_skill_match(pref_skill_lower, cand_skill)
                    if similarity > best_match_score:
                        best_match_score = similarity
                
                if best_match_score >= 0.8:
                    matched_preferred += 1.0
                elif best_match_score >= 0.6:
                    matched_preferred += 0.7
                elif best_match_score >= 0.4:
                    matched_preferred += 0.3
        
        # Calculate scores
        required_score = 0.0
        if required_skills:
            required_score = min(1.0, matched_required / len(required_skills))
        
        preferred_score = 0.0
        if preferred_skills:
            preferred_score = min(1.0, matched_preferred / len(preferred_skills))
        
        # Weighted combination: 80% required, 20% preferred
        if required_skills and preferred_skills:
            skill_score = required_score * 0.8 + preferred_score * 0.2
        elif required_skills:
            skill_score = required_score
        else:
            skill_score = preferred_score
        
        return skill_score
    
    def _fuzzy_skill_match(self, skill1: str, skill2: str) -> float:
        """Calculate fuzzy similarity between two skills."""
        if not skill1 or not skill2:
            return 0.0
        
        skill1 = skill1.lower().strip()
        skill2 = skill2.lower().strip()
        
        # Exact match
        if skill1 == skill2:
            return 1.0
        
        # Substring match
        if skill1 in skill2 or skill2 in skill1:
            min_len = min(len(skill1), len(skill2))
            max_len = max(len(skill1), len(skill2))
            if min_len >= 3:
                return 0.85 + (min_len / max_len) * 0.1
        
        # Character-based similarity
        set1 = set(skill1)
        set2 = set(skill2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        char_sim = intersection / union if union > 0 else 0.0
        
        # Word-based similarity (for multi-word skills)
        words1 = set(skill1.split())
        words2 = set(skill2.split())
        if words1 and words2:
            word_intersection = len(words1 & words2)
            word_union = len(words1 | words2)
            word_sim = word_intersection / word_union if word_union > 0 else 0.0
        else:
            word_sim = 0.0
        
        # Combined score
        return max(char_sim, word_sim * 0.8)
    
    def _calculate_experience_score(
        self,
        candidate_years: float,
        required_years: float
    ) -> float:
        """
        Calculate experience fit score.
        
        Formula: min(1.0, candidate_years / required_years)
        
        Args:
            candidate_years: Candidate's experience in years
            required_years: Required experience in years
        
        Returns:
            Experience fit score (0-1)
        """
        if required_years <= 0:
            # No requirement specified, return neutral score
            return 0.5
        
        if candidate_years <= 0:
            # No experience, return low score
            return 0.1
        
        # Calculate ratio
        ratio = candidate_years / required_years
        
        # Cap at 1.0 (over-qualified is still good)
        score = min(1.0, ratio)
        
        # Apply slight bonus for over-qualification
        if ratio > 1.0:
            score = min(1.0, 0.9 + (ratio - 1.0) * 0.1)
        
        return score
    
    def _calculate_semantic_score(
        self,
        resume_text: str,
        job_description: str,
        resume_embedding: Optional[Any] = None,
        job_embedding: Optional[Any] = None
    ) -> float:
        """
        Calculate semantic similarity score using embeddings.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            resume_embedding: Pre-computed resume embedding
            job_embedding: Pre-computed job embedding
        
        Returns:
            Semantic similarity score (0-1)
        """
        if not resume_text or not job_description:
            return 0.0
        
        try:
            # Use pre-computed embeddings if available
            if resume_embedding is not None and job_embedding is not None:
                return cosine_similarity(resume_embedding, job_embedding)
            
            # Generate embeddings if not provided
            if not self.embedder.is_available():
                logger.warning("Embedder not available, using fallback")
                return self._fallback_semantic_score(resume_text, job_description)
            
            # Generate embeddings
            resume_emb = self.embedder.encode(resume_text)
            job_emb = self.embedder.encode(job_description)
            
            if resume_emb is None or job_emb is None:
                return self._fallback_semantic_score(resume_text, job_description)
            
            return cosine_similarity(resume_emb, job_emb)
        except Exception as e:
            logger.error(f"Error calculating semantic score: {e}")
            return self._fallback_semantic_score(resume_text, job_description)
    
    def _fallback_semantic_score(self, text1: str, text2: str) -> float:
        """
        Fallback semantic score using simple word overlap.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Fallback similarity score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_additional_score(
        self,
        resume_data: ResumeData,
        job_data: JobData
    ) -> float:
        """
        Calculate additional factors score (education, certifications, etc.).
        
        Args:
            resume_data: Parsed resume data
            job_data: Parsed job data
        
        Returns:
            Additional factors score (0-1)
        """
        score = 0.0
        factors = 0
        
        # Education match
        if resume_data.education and job_data.education_required:
            education_lower = resume_data.education.lower()
            required_lower = job_data.education_required.lower()
            
            # Check for education keywords match
            education_keywords = ['bachelor', 'master', 'phd', 'degree', 'bs', 'ms', 'mba']
            for keyword in education_keywords:
                if keyword in education_lower and keyword in required_lower:
                    score += 0.3
                    factors += 1
                    break
        
        # Certifications bonus
        if resume_data.certifications:
            # Certifications are always a plus
            score += min(0.2, len(resume_data.certifications) * 0.1)
            factors += 1
        
        # Normalize score
        if factors > 0:
            return min(1.0, score)
        
        # Return neutral score if no factors
        return 0.5

