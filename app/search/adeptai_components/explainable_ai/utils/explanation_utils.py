from typing import List, Dict, Any
from ..models.dataclasses import FeatureContribution, DecisionExplanation
from ..config.settings import SCORE_THRESHOLDS, FEATURE_DESCRIPTIONS
from ..config import CONFIDENCE_THRESHOLDS

def generate_decision_summary(overall_score: float, positive_factors: List[str], negative_factors: List[str], ranking_position: int) -> str:
    if overall_score >= SCORE_THRESHOLDS['excellent']:
        base_summary = f"Candidate #{ranking_position} has an excellent score of {overall_score:.1f}."
    elif overall_score >= SCORE_THRESHOLDS['good']:
        base_summary = f"Candidate #{ranking_position} has a good score of {overall_score:.1f}."
    elif overall_score >= SCORE_THRESHOLDS['average']:
        base_summary = f"Candidate #{ranking_position} has an average score of {overall_score:.1f}."
    else:
        base_summary = f"Candidate #{ranking_position} has a below-average score of {overall_score:.1f}."
    if positive_factors:
        factor_desc = ", ".join([FEATURE_DESCRIPTIONS.get(f, f) for f in positive_factors])
        base_summary += f" Strengths: {factor_desc}."
    if negative_factors:
        factor_desc = ", ".join([FEATURE_DESCRIPTIONS.get(f, f) for f in negative_factors])
        base_summary += f" Concerns: {factor_desc}."
    return base_summary

def determine_confidence_level(contributions: List[FeatureContribution]) -> str:
    high = [fc for fc in contributions if fc.contribution > CONFIDENCE_THRESHOLDS['high_contribution']]
    low = [fc for fc in contributions if fc.contribution < CONFIDENCE_THRESHOLDS['low_contribution']]
    if len(high) >= CONFIDENCE_THRESHOLDS['very_high']['high_count'] and len(low) <= CONFIDENCE_THRESHOLDS['very_high']['low_count']: return "Very High"
    elif len(high) >= CONFIDENCE_THRESHOLDS['high']['high_count'] and len(low) <= CONFIDENCE_THRESHOLDS['high']['low_count']: return "High"
    elif len(high) >= CONFIDENCE_THRESHOLDS['moderate']['high_count']: return "Moderate"
    else: return "Low"

def generate_recommendation(overall_score: float, negative_factors: List[str]) -> str:
    if overall_score >= SCORE_THRESHOLDS['excellent']:
        return "Strongly recommend for immediate consideration."
    elif overall_score >= SCORE_THRESHOLDS['good']:
        return "Recommend for consideration."
    elif overall_score >= SCORE_THRESHOLDS['average']:
        return "Consider with reservations." if negative_factors else "Consider for the role."
    else:
        return "Not recommended for this role."

def identify_risk_factors(contributions: List[FeatureContribution], profile: Dict[str, Any]) -> List[str]:
    risks = []
    for fc in contributions:
        if fc.direction == 'negative' and fc.contribution < 0.02:
            risks.append(f"Concern with {fc.feature_name}")
    if profile.get('experience_years', 0) < 2:
        risks.append("Limited professional experience")
    if not profile.get('skills'):
        risks.append("No technical skills listed")
    return risks[:5]

def identify_strength_areas(contributions: List[FeatureContribution]) -> List[str]:
    strengths = []
    for fc in contributions:
        if fc.direction == 'positive' and fc.contribution > 0.05:
            strengths.append(f"Strong {fc.feature_name}")
    return strengths[:5]
