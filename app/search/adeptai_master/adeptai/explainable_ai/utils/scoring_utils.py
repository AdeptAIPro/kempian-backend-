from typing import Dict, List
from ..models.dataclasses import FeatureContribution
from ..config.settings import WEIGHTS, FEATURE_DESCRIPTIONS
from ..config import CONTRIBUTION_THRESHOLDS

def calculate_feature_contributions(feature_values: Dict[str, float], match_scores: Dict[str, float]) -> List[FeatureContribution]:
    overall_score = match_scores.get('overall_score', 0) / 100.0
    contributions = []
    for feature_name, value in feature_values.items():
        weight = WEIGHTS.get(feature_name, 0.05)
        contribution = value * weight
        try:
            percentage = (contribution / overall_score * 100) if overall_score > 0 else 0
        except ZeroDivisionError:
            percentage = 0.0
        if value >= CONTRIBUTION_THRESHOLDS['positive']:
            direction = 'positive'
        elif value <= CONTRIBUTION_THRESHOLDS['negative']:
            direction = 'negative'
        else:
            direction = 'neutral'
        explanation = generate_feature_explanation(feature_name, value)
        contributions.append(FeatureContribution(
            feature_name=feature_name,
            contribution=contribution,
            percentage=percentage,
            direction=direction,
            explanation=explanation
        ))
    contributions.sort(key=lambda x: x.contribution, reverse=True)
    return contributions

def generate_feature_explanation(feature_name: str, value: float) -> str:
    feature_desc = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
    if value >= 0.9: strength = "excellent"
    elif value >= 0.8: strength = "very strong"
    elif value >= 0.7: strength = "strong"
    elif value >= 0.6: strength = "good"
    elif value >= 0.5: strength = "moderate"
    elif value >= 0.4: strength = "below average"
    else: strength = "weak"
    return f"{strength.capitalize()} {feature_desc} (score: {value:.2f})"
