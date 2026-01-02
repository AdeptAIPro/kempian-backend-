from dataclasses import dataclass
from typing import List

@dataclass
class FeatureContribution:
    feature_name: str
    contribution: float
    percentage: float
    direction: str  # 'positive', 'negative', 'neutral'
    explanation: str

@dataclass
class DecisionExplanation:
    overall_score: float
    feature_contributions: List[FeatureContribution]
    top_positive_factors: List[str]
    top_negative_factors: List[str]
    decision_summary: str
    confidence_level: str
    recommendation: str
    risk_factors: List[str]
    strength_areas: List[str]
