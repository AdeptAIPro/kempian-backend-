"""
Candidate Behavior Prediction Module

Provides behavioral market intelligence including:
- Job switch probability analysis
- Salary increase expectations modeling
- Remote work preferences tracking
- Interview acceptance rate prediction
- Offer acceptance probability modeling
"""

from __future__ import annotations

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .hybrid_llm_service import market_intelligence_llm, TaskComplexity


class BehaviorSignal(Enum):
    """Behavioral signals for prediction"""
    HIGH_TENURE = "high_tenure"
    RECENT_JOB_SWITCH = "recent_job_switch"
    ACTIVE_JOB_SEARCH = "active_job_search"
    PASSIVE_OPEN = "passive_open"
    REMOTE_PREFERENCE = "remote_preference"
    SALARY_FOCUSED = "salary_focused"
    GROWTH_FOCUSED = "growth_focused"


@dataclass
class CandidateBehaviorProfile:
    """Individual candidate behavior profile"""
    candidate_id: str
    current_tenure_months: int
    total_jobs: int
    avg_tenure_months: float
    last_switch_months_ago: int
    active_job_search: bool
    passive_open: bool
    remote_preference: float  # 0-1
    salary_satisfaction: float  # 0-1
    growth_satisfaction: float  # 0-1
    industry_switches: int
    location_switches: int
    last_updated: datetime


@dataclass
class BehaviorPrediction:
    """Behavior prediction results"""
    candidate_id: str
    job_switch_probability: float
    salary_increase_expectation: float
    remote_work_preference: float
    interview_acceptance_rate: float
    offer_acceptance_probability: float
    confidence_score: float
    key_factors: List[str]
    recommendations: List[str]
    last_updated: datetime


class BehaviorAnalyzer:
    """Analyzes candidate behavior patterns and predicts future actions"""
    
    def __init__(self):
        self.tenure_weights = {
            "current_tenure": 0.3,
            "avg_tenure": 0.25,
            "last_switch": 0.2,
            "total_jobs": 0.15,
            "industry_switches": 0.1
        }
        
        self.salary_weights = {
            "current_satisfaction": 0.4,
            "market_trends": 0.3,
            "tenure_factor": 0.2,
            "industry_factor": 0.1
        }
    
    def analyze_tenure_patterns(self, profiles: List[CandidateBehaviorProfile]) -> Dict[str, Any]:
        """Analyze tenure patterns across candidate pool"""
        if not profiles:
            return {"error": "No profiles provided"}
        
        # Calculate tenure statistics
        current_tenures = [p.current_tenure_months for p in profiles]
        avg_tenures = [p.avg_tenure_months for p in profiles]
        total_jobs = [p.total_jobs for p in profiles]
        
        # Identify patterns
        high_tenure_candidates = [p for p in profiles if p.current_tenure_months > 36]
        job_hoppers = [p for p in profiles if p.avg_tenure_months < 18]
        recent_switchers = [p for p in profiles if p.last_switch_months_ago < 12]
        
        return {
            "total_candidates": len(profiles),
            "avg_current_tenure": float(np.mean(current_tenures)),
            "avg_historical_tenure": float(np.mean(avg_tenures)),
            "high_tenure_count": len(high_tenure_candidates),
            "job_hoppers_count": len(job_hoppers),
            "recent_switchers_count": len(recent_switchers),
            "tenure_distribution": {
                "0-12_months": len([p for p in profiles if p.current_tenure_months <= 12]),
                "13-24_months": len([p for p in profiles if 12 < p.current_tenure_months <= 24]),
                "25-36_months": len([p for p in profiles if 24 < p.current_tenure_months <= 36]),
                "37+_months": len([p for p in profiles if p.current_tenure_months > 36])
            }
        }
    
    def model_compensation_trends(self, profiles: List[CandidateBehaviorProfile], 
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Model compensation trends and expectations"""
        if not profiles:
            return {"error": "No profiles provided"}
        
        # Calculate salary satisfaction trends
        salary_satisfaction = [p.salary_satisfaction for p in profiles]
        avg_satisfaction = float(np.mean(salary_satisfaction))
        
        # Model expectations based on satisfaction and market trends
        market_inflation = market_data.get("salary_inflation_rate", 0.05)
        base_expectation = avg_satisfaction * (1 + market_inflation)
        
        # Adjust for tenure and industry factors
        tenure_factor = self._calculate_tenure_factor(profiles)
        industry_factor = market_data.get("industry_growth_rate", 0.03)
        
        expected_increase = base_expectation * (1 + tenure_factor + industry_factor)
        
        return {
            "avg_salary_satisfaction": avg_satisfaction,
            "expected_increase_rate": float(expected_increase),
            "market_inflation_rate": market_inflation,
            "tenure_factor": tenure_factor,
            "industry_factor": industry_factor,
            "satisfaction_distribution": {
                "very_satisfied": len([p for p in profiles if p.salary_satisfaction > 0.8]),
                "satisfied": len([p for p in profiles if 0.6 < p.salary_satisfaction <= 0.8]),
                "neutral": len([p for p in profiles if 0.4 < p.salary_satisfaction <= 0.6]),
                "dissatisfied": len([p for p in profiles if p.salary_satisfaction <= 0.4])
            }
        }
    
    def track_workplace_trends(self, profiles: List[CandidateBehaviorProfile]) -> Dict[str, Any]:
        """Track remote work preferences and workplace trends"""
        if not profiles:
            return {"error": "No profiles provided"}
        
        remote_preferences = [p.remote_preference for p in profiles]
        avg_remote_pref = float(np.mean(remote_preferences))
        
        # Categorize preferences
        remote_only = [p for p in profiles if p.remote_preference > 0.8]
        hybrid_preferred = [p for p in profiles if 0.4 < p.remote_preference <= 0.8]
        office_preferred = [p for p in profiles if p.remote_preference <= 0.4]
        
        return {
            "avg_remote_preference": avg_remote_pref,
            "remote_only_count": len(remote_only),
            "hybrid_preferred_count": len(hybrid_preferred),
            "office_preferred_count": len(office_preferred),
            "preference_distribution": {
                "0-20%": len([p for p in profiles if p.remote_preference <= 0.2]),
                "21-40%": len([p for p in profiles if 0.2 < p.remote_preference <= 0.4]),
                "41-60%": len([p for p in profiles if 0.4 < p.remote_preference <= 0.6]),
                "61-80%": len([p for p in profiles if 0.6 < p.remote_preference <= 0.8]),
                "81-100%": len([p for p in profiles if p.remote_preference > 0.8])
            }
        }
    
    def predict_response_likelihood(self, profiles: List[CandidateBehaviorProfile], 
                                  job_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Predict interview acceptance rates based on job attributes"""
        if not profiles:
            return {"error": "No profiles provided"}
        
        # Extract job attributes
        job_remote_ok = job_attributes.get("remote_friendly", True)
        job_salary_range = job_attributes.get("salary_range", [80000, 120000])
        job_growth_potential = job_attributes.get("growth_potential", 0.5)
        job_company_stage = job_attributes.get("company_stage", "startup")
        
        # Calculate match scores for each candidate
        match_scores = []
        for profile in profiles:
            score = self._calculate_job_match_score(
                profile, job_remote_ok, job_salary_range, 
                job_growth_potential, job_company_stage
            )
            match_scores.append(score)
        
        # Predict acceptance rates
        avg_match_score = float(np.mean(match_scores))
        predicted_acceptance_rate = min(0.95, max(0.05, avg_match_score))
        
        return {
            "predicted_acceptance_rate": predicted_acceptance_rate,
            "avg_match_score": avg_match_score,
            "high_match_candidates": len([s for s in match_scores if s > 0.8]),
            "medium_match_candidates": len([s for s in match_scores if 0.5 < s <= 0.8]),
            "low_match_candidates": len([s for s in match_scores if s <= 0.5])
        }
    
    def model_decision_factors(self, profiles: List[CandidateBehaviorProfile], 
                             offer_details: Dict[str, Any]) -> Dict[str, Any]:
        """Model offer acceptance probability and decision factors"""
        if not profiles:
            return {"error": "No profiles provided"}
        
        # Extract offer details
        offer_salary = offer_details.get("salary", 100000)
        offer_equity = offer_details.get("equity", 0)
        offer_remote = offer_details.get("remote_ok", True)
        offer_benefits = offer_details.get("benefits_score", 0.5)
        
        # Calculate acceptance probabilities
        acceptance_probs = []
        key_factors = []
        
        for profile in profiles:
            prob, factors = self._calculate_acceptance_probability(
                profile, offer_salary, offer_equity, offer_remote, offer_benefits
            )
            acceptance_probs.append(prob)
            key_factors.extend(factors)
        
        avg_acceptance_prob = float(np.mean(acceptance_probs))
        
        # Identify most common decision factors
        factor_counts = {}
        for factor in key_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "avg_acceptance_probability": avg_acceptance_prob,
            "high_probability_candidates": len([p for p in acceptance_probs if p > 0.8]),
            "medium_probability_candidates": len([p for p in acceptance_probs if 0.5 < p <= 0.8]),
            "low_probability_candidates": len([p for p in acceptance_probs if p <= 0.5]),
            "top_decision_factors": [{"factor": f, "count": c} for f, c in top_factors]
        }
    
    def _calculate_tenure_factor(self, profiles: List[CandidateBehaviorProfile]) -> float:
        """Calculate tenure-based adjustment factor"""
        avg_tenure = np.mean([p.avg_tenure_months for p in profiles])
        if avg_tenure < 12:
            return 0.1  # Job hoppers expect higher increases
        elif avg_tenure > 36:
            return -0.05  # Long-tenured employees more stable
        else:
            return 0.0
    
    def _calculate_job_match_score(self, profile: CandidateBehaviorProfile, 
                                 remote_ok: bool, salary_range: List[int], 
                                 growth_potential: float, company_stage: str) -> float:
        """Calculate job-candidate match score"""
        score = 0.0
        
        # Remote work match
        if remote_ok and profile.remote_preference > 0.5:
            score += 0.3
        elif not remote_ok and profile.remote_preference < 0.5:
            score += 0.3
        
        # Salary expectations (simplified)
        if profile.salary_satisfaction > 0.7:
            score += 0.2
        
        # Growth potential match
        if profile.growth_satisfaction > 0.6 and growth_potential > 0.5:
            score += 0.3
        
        # Company stage preference (simplified)
        if company_stage in ["startup", "scale-up"] and profile.growth_satisfaction > 0.7:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_acceptance_probability(self, profile: CandidateBehaviorProfile, 
                                        salary: float, equity: float, 
                                        remote: bool, benefits: float) -> Tuple[float, List[str]]:
        """Calculate offer acceptance probability and key factors"""
        prob = 0.5
        factors = []
        
        # Salary factor
        if salary > 100000:  # Simplified threshold
            prob += 0.2
            factors.append("competitive_salary")
        
        # Equity factor
        if equity > 0:
            prob += 0.1
            factors.append("equity_offering")
        
        # Remote work factor
        if remote and profile.remote_preference > 0.7:
            prob += 0.15
            factors.append("remote_work")
        
        # Benefits factor
        if benefits > 0.7:
            prob += 0.1
            factors.append("strong_benefits")
        
        # Tenure stability factor
        if profile.current_tenure_months > 24:
            prob += 0.05
            factors.append("stability_seeker")
        
        return min(0.95, max(0.05, prob)), factors


class BehaviorPredictor:
    """Main class for candidate behavior prediction with hybrid LLM integration"""
    
    def __init__(self):
        self.analyzer = BehaviorAnalyzer()
        self.llm_service = market_intelligence_llm
    
    async def predict_candidate_behavior(self, profiles: List[CandidateBehaviorProfile], 
                                       market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main prediction function with hybrid LLM enhancement"""
        if market_data is None:
            market_data = {}
        
        # Generate sample profiles if none provided
        if not profiles:
            profiles = self._generate_sample_profiles(50)
        
        # Run traditional analyses
        tenure_analysis = self.analyzer.analyze_tenure_patterns(profiles)
        compensation_trends = self.analyzer.model_compensation_trends(profiles, market_data)
        workplace_trends = self.analyzer.track_workplace_trends(profiles)
        
        # Sample job attributes for response prediction
        job_attributes = {
            "remote_friendly": True,
            "salary_range": [90000, 130000],
            "growth_potential": 0.7,
            "company_stage": "scale-up"
        }
        
        response_prediction = self.analyzer.predict_response_likelihood(profiles, job_attributes)
        
        # Sample offer details for acceptance prediction
        offer_details = {
            "salary": 110000,
            "equity": 25000,
            "remote_ok": True,
            "benefits_score": 0.8
        }
        
        decision_factors = self.analyzer.model_decision_factors(profiles, offer_details)
        
        # Enhance with hybrid LLM analysis
        llm_enhanced_analysis = await self._enhance_with_llm_analysis(profiles, market_data)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "job_switch_probability": tenure_analysis,
            "salary_increase_expectations": compensation_trends,
            "remote_work_preferences": workplace_trends,
            "interview_acceptance_rate": response_prediction,
            "offer_acceptance_probability": decision_factors,
            "llm_enhanced_insights": llm_enhanced_analysis,
            "sample_size": len(profiles)
        }
    
    async def _enhance_with_llm_analysis(self, profiles: List[CandidateBehaviorProfile], 
                                       market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance analysis with hybrid LLM models"""
        try:
            # Prepare data for LLM analysis
            candidate_data = {
                "profiles": [
                    {
                        "candidate_id": p.candidate_id,
                        "current_tenure_months": p.current_tenure_months,
                        "avg_tenure_months": p.avg_tenure_months,
                        "remote_preference": p.remote_preference,
                        "salary_satisfaction": p.salary_satisfaction,
                        "growth_satisfaction": p.growth_satisfaction,
                        "active_job_search": p.active_job_search
                    }
                    for p in profiles[:10]  # Limit for cost efficiency
                ],
                "market_context": market_data
            }
            
            # Get hybrid LLM analysis
            llm_analysis = await self.llm_service.analyze_behavior_patterns(candidate_data)
            
            return {
                "ai_insights": llm_analysis,
                "enhancement_applied": True,
                "models_used": ["gpt-4o-mini", "claude-3-5-sonnet-20241022"]
            }
        except Exception as e:
            logger.error(f"Error in LLM enhancement: {e}")
            return {
                "ai_insights": None,
                "enhancement_applied": False,
                "error": str(e)
            }
    
    def _generate_sample_profiles(self, count: int) -> List[CandidateBehaviorProfile]:
        """Generate sample candidate profiles for testing"""
        profiles = []
        
        for i in range(count):
            profile = CandidateBehaviorProfile(
                candidate_id=f"candidate_{i+1}",
                current_tenure_months=random.randint(6, 60),
                total_jobs=random.randint(1, 5),
                avg_tenure_months=random.uniform(12, 36),
                last_switch_months_ago=random.randint(1, 24),
                active_job_search=random.choice([True, False]),
                passive_open=random.choice([True, False]),
                remote_preference=random.uniform(0, 1),
                salary_satisfaction=random.uniform(0.2, 1.0),
                growth_satisfaction=random.uniform(0.2, 1.0),
                industry_switches=random.randint(0, 3),
                location_switches=random.randint(0, 2),
                last_updated=datetime.now()
            )
            profiles.append(profile)
        
        return profiles


# Convenience function for API access
async def predict_candidate_behavior(profiles: List[CandidateBehaviorProfile] = None, 
                                   market_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Predict candidate behavior for API endpoint with hybrid LLM enhancement"""
    predictor = BehaviorPredictor()
    return await predictor.predict_candidate_behavior(profiles, market_data)
