"""
Fast Data Generation for Market Intelligence

Pre-generated mock data for instant responses during development
"""

from __future__ import annotations

import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio


class FastDataGenerator:
    """Generate realistic mock data instantly"""
    
    def __init__(self):
        self.cache = {}
        self.last_generated = {}
    
    def get_salary_trends(self) -> Dict[str, Any]:
        """Get instant salary trends data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_inflation": round(random.uniform(3, 8), 1),
            "salary_trends": {
                "software_engineer": {
                    "average_salary": random.randint(80000, 150000),
                    "trend_direction": random.choice(["rising", "stable"]),
                    "growth_rate": round(random.uniform(0, 15), 1)
                },
                "data_scientist": {
                    "average_salary": random.randint(90000, 160000),
                    "trend_direction": random.choice(["rising", "stable"]),
                    "growth_rate": round(random.uniform(0, 15), 1)
                },
                "product_manager": {
                    "average_salary": random.randint(100000, 180000),
                    "trend_direction": random.choice(["rising", "stable"]),
                    "growth_rate": round(random.uniform(0, 15), 1)
                }
            },
            "market_competitiveness": round(random.uniform(0.6, 0.9), 2),
            "confidence_score": round(random.uniform(0.8, 0.95), 2)
        }
    
    def get_skill_demands(self) -> Dict[str, Any]:
        """Get instant skill demand data"""
        skills = ["Python", "JavaScript", "React", "Node.js", "AWS", "Docker", "Kubernetes", "Machine Learning"]
        
        skill_evolution = {}
        for skill in skills:
            skill_evolution[skill] = {
                "current_demand": random.randint(100, 1000),
                "growth_rate": round(random.uniform(-10, 50), 1),
                "trend_direction": random.choice(["rising", "stable", "declining"]),
                "adoption_score": random.randint(50, 100)
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "skill_evolution": skill_evolution,
            "emerging_skills": ["Rust", "WebAssembly", "Edge Computing"],
            "declining_skills": ["jQuery", "Flash", "Internet Explorer"],
            "confidence_score": round(random.uniform(0.7, 0.9), 2)
        }
    
    def get_talent_availability(self) -> Dict[str, Any]:
        """Get instant talent availability data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_candidates": random.randint(5000, 20000),
            "passive_candidates": random.randint(15000, 50000),
            "geographic_distribution": {
                "san_francisco": random.randint(200, 800),
                "new_york": random.randint(300, 900),
                "seattle": random.randint(150, 600),
                "remote": random.randint(500, 1500)
            },
            "mobility_likelihood": round(random.uniform(0.3, 0.7), 2),
            "confidence_score": round(random.uniform(0.7, 0.95), 2)
        }
    
    def get_competitive_landscape(self) -> Dict[str, Any]:
        """Get instant competitive landscape data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "industry_growth": {
                "technology": round(random.uniform(5, 15), 1),
                "finance": round(random.uniform(2, 8), 1),
                "healthcare": round(random.uniform(3, 10), 1)
            },
            "competitor_activity": {
                "hiring_velocity": round(random.uniform(20, 90), 1),
                "layoff_risk": round(random.uniform(5, 25), 1),
                "funding_environment": random.choice(["favorable", "neutral", "challenging"])
            },
            "market_sentiment": round(random.uniform(0.4, 0.8), 2),
            "confidence_score": round(random.uniform(0.6, 0.85), 2)
        }
    
    def get_economic_indicators(self) -> Dict[str, Any]:
        """Get instant economic indicators"""
        return {
            "timestamp": datetime.now().isoformat(),
            "unemployment_rate": round(random.uniform(3.5, 6.5), 1),
            "gdp_growth": round(random.uniform(1.5, 4.0), 1),
            "inflation_rate": round(random.uniform(2.0, 5.0), 1),
            "labor_force_participation": round(random.uniform(60, 65), 1),
            "confidence_score": round(random.uniform(0.8, 0.95), 2)
        }
    
    def get_behavior_insights(self) -> Dict[str, Any]:
        """Get instant behavior insights"""
        return {
            "timestamp": datetime.now().isoformat(),
            "job_switch_probability": round(random.uniform(0.2, 0.6), 2),
            "remote_work_preference": round(random.uniform(0.4, 0.8), 2),
            "salary_satisfaction": round(random.uniform(0.5, 0.9), 2),
            "interview_acceptance_rate": round(random.uniform(0.6, 0.9), 2),
            "confidence_score": round(random.uniform(0.7, 0.9), 2)
        }
    
    def get_market_alerts(self) -> List[Dict[str, Any]]:
        """Get instant market alerts"""
        alert_types = [
            {
                "alert_id": f"alert_{random.randint(1000, 9999)}",
                "type": "salary_inflation",
                "severity": "medium",
                "title": f"Salary Inflation Detected: {random.uniform(8, 15):.1f}%",
                "description": f"Average salary increases across target roles exceed {random.uniform(8, 15):.1f}%",
                "confidence": round(random.uniform(0.7, 0.9), 2),
                "timestamp": datetime.now().isoformat()
            },
            {
                "alert_id": f"alert_{random.randint(1000, 9999)}",
                "type": "emerging_skills",
                "severity": "low",
                "title": f"Emerging Skill: {random.choice(['Rust', 'WebAssembly', 'Edge AI'])}",
                "description": f"{random.choice(['Rust', 'WebAssembly', 'Edge AI'])} shows {random.uniform(50, 200):.0f}% growth in job postings",
                "confidence": round(random.uniform(0.6, 0.8), 2),
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        return random.sample(alert_types, random.randint(0, 2))
    
    def get_candidate_profiles(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get instant candidate profiles"""
        profiles = []
        skills_pool = ["Python", "JavaScript", "React", "Node.js", "AWS", "Docker", "SQL", "Machine Learning"]
        
        for i in range(count):
            profile = {
                "candidate_id": f"candidate_{i+1}",
                "current_tenure_months": random.randint(6, 60),
                "avg_tenure_months": round(random.uniform(12, 36), 1),
                "remote_preference": round(random.uniform(0, 1), 2),
                "salary_satisfaction": round(random.uniform(0.2, 1.0), 2),
                "growth_satisfaction": round(random.uniform(0.2, 1.0), 2),
                "active_job_search": random.choice([True, False]),
                "skills": random.sample(skills_pool, random.randint(3, 6)),
                "experience_years": random.randint(1, 10),
                "location": random.choice(["San Francisco", "New York", "Seattle", "Remote"]),
                "last_updated": datetime.now().isoformat()
            }
            profiles.append(profile)
        
        return profiles
    
    def get_complete_market_data(self) -> Dict[str, Any]:
        """Get complete market intelligence data instantly"""
        return {
            "timestamp": datetime.now().isoformat(),
            "talent_availability": self.get_talent_availability(),
            "compensation_trends": self.get_salary_trends(),
            "skill_evolution": self.get_skill_demands(),
            "competitive_landscape": self.get_competitive_landscape(),
            "economic_indicators": self.get_economic_indicators(),
            "behavioral_insights": self.get_behavior_insights(),
            "market_alerts": self.get_market_alerts(),
            "data_source": "fast_mock_data",
            "generation_time_ms": 1  # Instant generation
        }
    
    async def get_cached_data(self, data_type: str, ttl_seconds: int = 300) -> Dict[str, Any]:
        """Get cached data or generate new"""
        cache_key = f"fast_data:{data_type}"
        
        # Check if we have recent data
        if data_type in self.last_generated:
            last_time = self.last_generated[data_type]
            if (datetime.now() - last_time).seconds < ttl_seconds:
                return self.cache.get(data_type, {})
        
        # Generate new data
        data_generators = {
            "salary_trends": self.get_salary_trends,
            "skill_demands": self.get_skill_demands,
            "talent_availability": self.get_talent_availability,
            "competitive_landscape": self.get_competitive_landscape,
            "economic_indicators": self.get_economic_indicators,
            "behavior_insights": self.get_behavior_insights,
            "market_alerts": self.get_market_alerts,
            "complete": self.get_complete_market_data
        }
        
        if data_type in data_generators:
            data = data_generators[data_type]()
            self.cache[data_type] = data
            self.last_generated[data_type] = datetime.now()
            return data
        
        return {}


# Global fast data generator
fast_data_generator = FastDataGenerator()
