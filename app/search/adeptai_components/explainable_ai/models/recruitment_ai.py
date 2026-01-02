from typing import Dict, Any, List
from .dataclasses import DecisionExplanation
from ..utils.features_utils import (
    extract_required_experience_from_query,
    extract_required_seniority_from_query,
    seniority_to_numeric,
    calculate_domain_relevance
)
from ..utils.scoring_utils import calculate_feature_contributions
from ..utils.explanation_utils import (
    generate_decision_summary,
    determine_confidence_level,
    generate_recommendation,
    identify_risk_factors,
    identify_strength_areas
)
from ..config import CONTRIBUTION_THRESHOLDS


class ExplainableRecruitmentAI:
    def explain_candidate_selection(self, candidate_profile: Dict[str, Any], job_query: str, match_scores: Dict[str, float], ranking_position: int) -> DecisionExplanation:
        # Validate inputs
        if not isinstance(candidate_profile, dict):
            raise ValueError("candidate_profile must be a dict")
        if not isinstance(match_scores, dict):
            raise ValueError("match_scores must be a dict")
        if not isinstance(job_query, str) or not job_query.strip():
            raise ValueError("job_query must be a non-empty string")

        # Normalize candidate profile
        safe_profile: Dict[str, Any] = dict(candidate_profile)
        experience_years = safe_profile.get('experience_years')
        try:
            safe_profile['experience_years'] = int(experience_years) if experience_years is not None else 0
        except (TypeError, ValueError):
            safe_profile['experience_years'] = 0
        seniority_level = safe_profile.get('seniority_level')
        safe_profile['seniority_level'] = str(seniority_level).lower() if isinstance(seniority_level, str) else 'mid'
        skills = safe_profile.get('skills')
        safe_profile['skills'] = skills if isinstance(skills, list) else []

        # Normalize and validate scores (ensure required keys exist and are floats)
        expected_score_keys: List[str] = [
            'overall_score',
            'technical_skills_score',
            'experience_score',
            'seniority_score',
            'education_score',
            'soft_skills_score',
            'location_score'
        ]
        normalized_scores: Dict[str, float] = {}
        for key in expected_score_keys:
            value = match_scores.get(key, 0.0)
            try:
                normalized_scores[key] = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                normalized_scores[key] = 0.0

        feature_values = self._extract_feature_values(safe_profile, normalized_scores, job_query)
        contributions = calculate_feature_contributions(feature_values, normalized_scores)
        # Calculate overall score from the original normalized scores, not from contributions
        overall_score = normalized_scores.get('overall_score', 0.0)
        top_positive = [fc.feature_name for fc in contributions if fc.direction=="positive" and fc.contribution>CONTRIBUTION_THRESHOLDS['positive']][:3]
        top_negative = [fc.feature_name for fc in contributions if fc.direction=="negative" and fc.contribution<CONTRIBUTION_THRESHOLDS['negative']][:3]
        return DecisionExplanation(
            overall_score=overall_score,
            feature_contributions=contributions,
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
            decision_summary=generate_decision_summary(overall_score, top_positive, top_negative, ranking_position),
            confidence_level=determine_confidence_level(contributions),
            recommendation=generate_recommendation(overall_score, top_negative),
            risk_factors=identify_risk_factors(contributions, safe_profile),
            strength_areas=identify_strength_areas(contributions)
        )

    def _extract_feature_values(self, profile: Dict[str, Any], scores: Dict[str, float], job_query: str) -> Dict[str, float]:
        experience_years = profile.get('experience_years', 0)
        required_experience = extract_required_experience_from_query(job_query)
        experience_gap = abs(experience_years - required_experience) if required_experience else 0
        seniority_level = profile.get('seniority_level', 'mid')
        seniority_numeric = seniority_to_numeric(seniority_level)
        required_seniority = extract_required_seniority_from_query(job_query)
        seniority_gap = abs(seniority_numeric - required_seniority) if required_seniority else 0
        skills = profile.get('skills', [])
        skill_breadth = len(skills) if skills else 0
        domain_relevance = calculate_domain_relevance(profile, job_query)
        return {
            'technical_skills_match': scores.get('technical_skills_score', 0) / 100.0,
            'experience_years_match': scores.get('experience_score', 0) / 100.0,
            'seniority_level_match': scores.get('seniority_score', 0) / 100.0,
            'education_match': scores.get('education_score', 0) / 100.0,
            'soft_skills_match': scores.get('soft_skills_score', 0) / 100.0,
            'location_match': scores.get('location_score', 0) / 100.0,
            'domain_relevance': domain_relevance,
            'skill_breadth': min(skill_breadth / 20.0, 1.0),
            'experience_gap': max(0, 1.0 - (experience_gap / 10.0)),
            'seniority_gap': max(0, 1.0 - (seniority_gap / 3.0))
        }


