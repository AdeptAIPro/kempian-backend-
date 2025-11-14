# Explainable AI Package for Recruitment System
from .models.recruitment_ai import ExplainableRecruitmentAI
from .models.dataclasses import DecisionExplanation, FeatureContribution
from .utils.explanation_utils import (
    generate_decision_summary,
    determine_confidence_level,
    generate_recommendation,
    identify_risk_factors,
    identify_strength_areas
)
from .utils.scoring_utils import calculate_feature_contributions
from .utils.features_utils import (
    extract_required_experience_from_query,
    extract_required_seniority_from_query,
    seniority_to_numeric,
    calculate_domain_relevance
)


__all__ = [
    'ExplainableRecruitmentAI',
    'DecisionExplanation',
    'FeatureContribution',
    'generate_decision_summary',
    'determine_confidence_level',
    'generate_recommendation',
    'identify_risk_factors',
    'identify_strength_areas',
    'calculate_feature_contributions',
    'extract_required_experience_from_query',
    'extract_required_seniority_from_query',
    'seniority_to_numeric',
    'calculate_domain_relevance'
]
