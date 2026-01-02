"""
Talent Supply & Competition Analysis

Provides:
- analyze_talent_availability()
- competitive_intelligence()

Data sources (simulated): LinkedIn updates, Glassdoor reviews, public layoff trackers,
job boards, internal search metrics. Replace stubs with real connectors as needed.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Dict, List, Any


# ---------- Talent Pool Mapping ----------

def count_job_seekers_by_skill() -> Dict[str, int]:
    # Simulated counts by skill; in production aggregate from job boards and internal data
    return {
        "Python": 14523,
        "JavaScript": 17890,
        "Java": 13210,
        "Kubernetes": 5210,
        "SQL": 19340,
        "Machine Learning": 6840,
    }


def estimate_passive_pool() -> Dict[str, int]:
    # Estimation from professional network activity signals and enrichment vendors
    active = count_job_seekers_by_skill()
    return {skill: int(count * 2.5) for skill, count in active.items()}


def map_talent_clusters() -> Dict[str, Dict[str, int]]:
    # Geographic clusters by city/region and skill (simulated heatmap counts)
    clusters = {
        "San Francisco": {"Python": 5200, "JavaScript": 6100, "Kubernetes": 2300},
        "New York": {"Python": 4800, "Java": 5100, "SQL": 7200},
        "Seattle": {"Python": 3100, "Java": 2700, "AWS": 2600},
        "Bengaluru": {"Python": 8400, "Java": 9100, "Machine Learning": 3300},
        "London": {"Python": 4500, "JavaScript": 4200, "Data": 3900},
    }
    return clusters


def predict_relocation_willingness() -> Dict[str, float]:
    # Probability (0-1) of relocation per region based on cost, demand, remote policies
    return {
        "San Francisco": 0.32,
        "New York": 0.28,
        "Seattle": 0.35,
        "Remote": 0.62,
        "Austin": 0.44,
        "London": 0.30,
        "Bengaluru": 0.38,
    }


def analyze_talent_availability() -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "active_candidates": count_job_seekers_by_skill(),
        "passive_candidates": estimate_passive_pool(),
        "geographic_distribution": map_talent_clusters(),
        "mobility_likelihood": predict_relocation_willingness(),
    }


# ---------- Competitive Landscape ----------

def track_linkedin_updates() -> Dict[str, Any]:
    # Hiring velocity score per competitor (0-100)
    companies = ["AlphaTech", "BetaBank", "Cloudy", "DataForge", "NextRetail"]
    return {c: {"hiring_velocity": random.randint(20, 95), "open_roles": random.randint(30, 800)} for c in companies}


def identify_vulnerable_companies() -> List[Dict[str, Any]]:
    # Combine layoff signals + attrition chatter to estimate vulnerability
    candidates = [
        {"company": "AlphaTech", "recent_layoffs": True, "attrition_signal": 0.62},
        {"company": "FinServe", "recent_layoffs": False, "attrition_signal": 0.31},
        {"company": "Cloudy", "recent_layoffs": True, "attrition_signal": 0.55},
        {"company": "RetailX", "recent_layoffs": True, "attrition_signal": 0.49},
    ]
    # Rank opportunities
    for c in candidates:
        c["poaching_score"] = round((0.5 if c["recent_layoffs"] else 0.2) + c["attrition_signal"] * 0.6, 2)
    return sorted(candidates, key=lambda x: x["poaching_score"], reverse=True)


def analyze_glassdoor_reviews() -> Dict[str, Any]:
    # Employer brand sentiment (1-5) and topics (mocked)
    return {
        "AlphaTech": {"rating": 3.6, "culture": 3.8, "compensation": 3.4, "topics": ["work-life", "growth", "pay"]},
        "Cloudy": {"rating": 4.1, "culture": 4.2, "compensation": 4.0, "topics": ["innovation", "benefits", "leadership"]},
        "DataForge": {"rating": 3.9, "culture": 3.7, "compensation": 3.8, "topics": ["learning", "pace", "stability"]},
    }


def calculate_market_competition() -> Dict[str, Any]:
    # Hiring difficulty score by role family (0-100)
    return {
        "Software Engineering": 78,
        "Data Science": 74,
        "DevOps": 69,
        "Product Management": 72,
        "Design": 58,
        "Sales": 52,
    }


def competitive_intelligence() -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "competitor_hiring_velocity": track_linkedin_updates(),
        "talent_poaching_opportunities": identify_vulnerable_companies(),
        "employer_brand_sentiment": analyze_glassdoor_reviews(),
        "hiring_difficulty_score": calculate_market_competition(),
    }


