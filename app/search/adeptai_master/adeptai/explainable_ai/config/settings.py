FEATURE_NAMES = [
    'technical_skills_match',
    'experience_years_match', 
    'seniority_level_match',
    'education_match',
    'soft_skills_match',
    'location_match',
    'domain_relevance',
    'skill_breadth',
    'experience_gap',
    'seniority_gap'
]

FEATURE_DESCRIPTIONS = {
    'technical_skills_match': 'Technical skills alignment with job requirements',
    'experience_years_match': 'Years of experience compared to requirements',
    'seniority_level_match': 'Seniority level alignment (Junior/Mid/Senior)',
    'education_match': 'Educational background relevance',
    'soft_skills_match': 'Soft skills and personality traits match',
    'location_match': 'Geographic location compatibility',
    'domain_relevance': 'Industry and domain knowledge relevance',
    'skill_breadth': 'Diversity and breadth of technical skills',
    'experience_gap': 'Experience level difference from requirements',
    'seniority_gap': 'Seniority level difference from requirements'
}

SCORE_THRESHOLDS = {
    'excellent': 85.0,
    'good': 70.0,
    'average': 55.0,
    'below_average': 40.0
}

WEIGHTS = {
    'technical_skills_match': 0.25,
    'experience_years_match': 0.20,
    'seniority_level_match': 0.15,
    'education_match': 0.10,
    'soft_skills_match': 0.10,
    'location_match': 0.05,
    'domain_relevance': 0.05,
    'skill_breadth': 0.05,
    'experience_gap': 0.03,
    'seniority_gap': 0.02
}

# Contribution thresholds for determining positive/negative factors
CONTRIBUTION_THRESHOLDS = {
    'positive': 0.05,  # Features with contribution > 0.05 are considered positive
    'negative': 0.02,  # Features with contribution < 0.02 are considered negative
    'high_contribution': 0.05,  # For confidence level calculations
    'low_contribution': 0.01    # For confidence level calculations
}

# Confidence level thresholds
CONFIDENCE_THRESHOLDS = {
    'very_high': {'high_count': 6, 'low_count': 2},
    'high': {'high_count': 4, 'low_count': 3},
    'moderate': {'high_count': 3, 'low_count': None},
    'low': {'high_count': None, 'low_count': None}
}

# Gap calculation constants
EXPERIENCE_GAP_MAX = 10.0  # Maximum experience gap for normalization
SENIORITY_GAP_MAX = 3.0   # Maximum seniority gap for normalization
SKILL_BREADTH_MAX = 20.0  # Maximum skill count for normalization
