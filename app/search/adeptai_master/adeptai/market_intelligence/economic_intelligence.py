"""
Economic & Industry Intelligence Module

Provides:
- Macro-economic indicators (unemployment, VC funding, industry growth, remote work)
- Market timing intelligence with AI integration
- Hiring confidence and sourcing optimization

Data sources (simulated): BLS, FRED, Crunchbase, industry reports, remote work surveys
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EconomicIndicator:
    """Economic indicator data point"""
    name: str
    value: float
    unit: str
    trend: str  # "rising", "falling", "stable"
    confidence: float
    last_updated: datetime
    source: str


@dataclass
class IndustryHealth:
    """Industry health metrics"""
    industry: str
    growth_rate: float
    hiring_velocity: float
    layoff_risk: float
    funding_environment: str  # "favorable", "neutral", "challenging"
    remote_adoption: float
    last_updated: datetime


class EconomicDataCollector:
    """Collects macro-economic indicators from various sources"""
    
    def __init__(self):
        self.rate_delay = 0.1
    
    async def fetch_economic_data(self) -> Dict[str, EconomicIndicator]:
        """Fetch current economic indicators"""
        indicators = {}
        
        # Unemployment rates by sector (simulated BLS data)
        sectors = ["Technology", "Finance", "Healthcare", "Manufacturing", "Retail", "Education"]
        for sector in sectors:
            base_rate = random.uniform(2.5, 6.5)
            trend = random.choice(["rising", "falling", "stable"])
            indicators[f"unemployment_{sector.lower()}"] = EconomicIndicator(
                name=f"Unemployment Rate - {sector}",
                value=round(base_rate, 1),
                unit="percentage",
                trend=trend,
                confidence=random.uniform(0.8, 0.95),
                last_updated=datetime.now() - timedelta(days=random.randint(1, 7)),
                source="BLS"
            )
        
        # VC funding trends (simulated Crunchbase data)
        funding_indicators = [
            ("total_vc_funding", "Total VC Funding", "billion", 45.2),
            ("early_stage_funding", "Early Stage Funding", "billion", 12.8),
            ("late_stage_funding", "Late Stage Funding", "billion", 28.4),
            ("ai_funding", "AI/ML Funding", "billion", 8.7),
            ("fintech_funding", "FinTech Funding", "billion", 5.3)
        ]
        
        for key, name, unit, base_value in funding_indicators:
            trend = random.choice(["rising", "falling", "stable"])
            change_factor = random.uniform(0.85, 1.15)
            indicators[key] = EconomicIndicator(
                name=name,
                value=round(base_value * change_factor, 1),
                unit=unit,
                trend=trend,
                confidence=random.uniform(0.75, 0.90),
                last_updated=datetime.now() - timedelta(days=random.randint(1, 14)),
                source="Crunchbase"
            )
        
        # Remote work adoption rates
        remote_indicators = [
            ("tech_remote_adoption", "Tech Remote Work Adoption", "percentage", 78.5),
            ("finance_remote_adoption", "Finance Remote Work Adoption", "percentage", 45.2),
            ("healthcare_remote_adoption", "Healthcare Remote Work Adoption", "percentage", 23.8),
            ("overall_remote_adoption", "Overall Remote Work Adoption", "percentage", 42.1)
        ]
        
        for key, name, unit, base_value in remote_indicators:
            trend = random.choice(["rising", "falling", "stable"])
            change_factor = random.uniform(0.95, 1.05)
            indicators[key] = EconomicIndicator(
                name=name,
                value=round(base_value * change_factor, 1),
                unit=unit,
                trend=trend,
                confidence=random.uniform(0.70, 0.85),
                last_updated=datetime.now() - timedelta(days=random.randint(1, 30)),
                source="Remote Work Survey"
            )
        
        return indicators
    
    async def analyze_sector_performance(self) -> List[IndustryHealth]:
        """Analyze industry health and performance"""
        industries = [
            "Technology", "Finance", "Healthcare", "Manufacturing", 
            "Retail", "Education", "Consulting", "Real Estate"
        ]
        
        health_metrics = []
        for industry in industries:
            # Generate realistic industry health metrics
            growth_rate = random.uniform(-2.0, 8.0)
            hiring_velocity = random.uniform(20, 90)
            layoff_risk = random.uniform(5, 35)
            
            # Determine funding environment based on industry
            if industry in ["Technology", "Healthcare"]:
                funding_env = random.choice(["favorable", "favorable", "neutral"])
            elif industry in ["Manufacturing", "Retail"]:
                funding_env = random.choice(["challenging", "neutral", "neutral"])
            else:
                funding_env = random.choice(["favorable", "neutral", "challenging"])
            
            # Remote adoption varies by industry
            if industry == "Technology":
                remote_adoption = random.uniform(70, 90)
            elif industry == "Finance":
                remote_adoption = random.uniform(40, 60)
            elif industry == "Healthcare":
                remote_adoption = random.uniform(15, 35)
            else:
                remote_adoption = random.uniform(25, 55)
            
            health_metrics.append(IndustryHealth(
                industry=industry,
                growth_rate=round(growth_rate, 1),
                hiring_velocity=round(hiring_velocity, 1),
                layoff_risk=round(layoff_risk, 1),
                funding_environment=funding_env,
                remote_adoption=round(remote_adoption, 1),
                last_updated=datetime.now() - timedelta(days=random.randint(1, 7))
            ))
        
        return health_metrics


class MarketTimingIntelligence:
    """AI-powered market timing and optimization"""
    
    def __init__(self):
        self.economic_collector = EconomicDataCollector()
    
    async def market_timing_intelligence(self) -> Dict[str, Any]:
        """Generate market timing intelligence with AI integration"""
        try:
            # Fetch economic data
            economic_indicators = await self.economic_collector.fetch_economic_data()
            industry_health = await self.economic_collector.analyze_sector_performance()
            
            # Calculate hiring confidence index
            hiring_confidence = self._calculate_hiring_confidence_index(
                economic_indicators, industry_health
            )
            
            # Optimize sourcing timing
            best_timing = self._optimize_outreach_timing(
                economic_indicators, industry_health
            )
            
            # Suggest compensation strategy
            budget_recommendations = self._suggest_compensation_strategy(
                economic_indicators, industry_health
            )
            
            # Modify search parameters based on urgency
            urgency_adjustments = self._modify_search_parameters(
                economic_indicators, industry_health
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "hiring_confidence_index": hiring_confidence,
                "best_sourcing_timing": best_timing,
                "budget_recommendations": budget_recommendations,
                "urgency_adjustments": urgency_adjustments,
                "economic_indicators": {
                    key: {
                        "name": ind.name,
                        "value": ind.value,
                        "unit": ind.unit,
                        "trend": ind.trend,
                        "confidence": ind.confidence,
                        "last_updated": ind.last_updated.isoformat(),
                        "source": ind.source
                    }
                    for key, ind in economic_indicators.items()
                },
                "industry_health": [
                    {
                        "industry": ih.industry,
                        "growth_rate": ih.growth_rate,
                        "hiring_velocity": ih.hiring_velocity,
                        "layoff_risk": ih.layoff_risk,
                        "funding_environment": ih.funding_environment,
                        "remote_adoption": ih.remote_adoption,
                        "last_updated": ih.last_updated.isoformat()
                    }
                    for ih in industry_health
                ]
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_hiring_confidence_index(self, 
                                         economic_indicators: Dict[str, EconomicIndicator],
                                         industry_health: List[IndustryHealth]) -> Dict[str, Any]:
        """Calculate hiring confidence index (0-100)"""
        # Factors: unemployment rates, VC funding, industry growth
        unemployment_scores = []
        for key, indicator in economic_indicators.items():
            if "unemployment" in key:
                # Lower unemployment = higher confidence
                score = max(0, 100 - indicator.value * 10)
                unemployment_scores.append(score)
        
        avg_unemployment_score = np.mean(unemployment_scores) if unemployment_scores else 50
        
        # VC funding factor
        vc_funding = economic_indicators.get("total_vc_funding", EconomicIndicator("", 0, "", "stable", 0, datetime.now(), ""))
        funding_score = min(100, vc_funding.value * 2)  # Scale funding to 0-100
        
        # Industry growth factor
        avg_growth = np.mean([ih.growth_rate for ih in industry_health])
        growth_score = max(0, min(100, 50 + avg_growth * 5))
        
        # Weighted combination
        confidence = (avg_unemployment_score * 0.4 + funding_score * 0.3 + growth_score * 0.3)
        
        return {
            "overall_confidence": round(confidence, 1),
            "unemployment_factor": round(avg_unemployment_score, 1),
            "funding_factor": round(funding_score, 1),
            "growth_factor": round(growth_score, 1),
            "recommendation": self._get_confidence_recommendation(confidence)
        }
    
    def _optimize_outreach_timing(self, 
                                economic_indicators: Dict[str, EconomicIndicator],
                                industry_health: List[IndustryHealth]) -> Dict[str, Any]:
        """Optimize best times for candidate outreach"""
        # Analyze patterns to suggest optimal timing
        current_month = datetime.now().month
        
        # Seasonal adjustments
        seasonal_factors = {
            1: 0.8,   # January - post-holiday slowdown
            2: 0.9,   # February - recovery
            3: 1.1,   # March - Q1 push
            4: 1.0,   # April - steady
            5: 1.0,   # May - steady
            6: 0.9,   # June - summer slowdown
            7: 0.8,   # July - vacation season
            8: 0.9,   # August - recovery
            9: 1.2,   # September - Q4 push
            10: 1.1,  # October - strong
            11: 1.0,  # November - steady
            12: 0.7   # December - holiday slowdown
        }
        
        current_factor = seasonal_factors.get(current_month, 1.0)
        
        # Remote work adoption factor
        remote_adoption = economic_indicators.get("overall_remote_adoption", EconomicIndicator("", 50, "", "stable", 0, datetime.now(), ""))
        remote_factor = remote_adoption.value / 100
        
        # Calculate optimal timing score
        timing_score = current_factor * (0.7 + 0.3 * remote_factor) * 100
        
        return {
            "optimal_timing_score": round(timing_score, 1),
            "current_month_factor": round(current_factor, 2),
            "remote_work_advantage": round(remote_factor, 2),
            "best_months": ["September", "October", "March", "April"],
            "avoid_months": ["December", "July", "January"],
            "recommendation": self._get_timing_recommendation(timing_score)
        }
    
    def _suggest_compensation_strategy(self, 
                                     economic_indicators: Dict[str, EconomicIndicator],
                                     industry_health: List[IndustryHealth]) -> Dict[str, Any]:
        """Suggest compensation strategy based on market conditions"""
        # Analyze market conditions
        avg_growth = np.mean([ih.growth_rate for ih in industry_health])
        avg_hiring_velocity = np.mean([ih.hiring_velocity for ih in industry_health])
        vc_funding = economic_indicators.get("total_vc_funding", EconomicIndicator("", 0, "", "stable", 0, datetime.now(), ""))
        
        # Determine strategy
        if avg_growth > 5 and vc_funding.value > 40:
            strategy = "aggressive"
            base_multiplier = 1.15
            equity_multiplier = 1.25
        elif avg_growth > 2 and avg_hiring_velocity > 60:
            strategy = "competitive"
            base_multiplier = 1.05
            equity_multiplier = 1.10
        elif avg_growth < 0 or avg_hiring_velocity < 40:
            strategy = "conservative"
            base_multiplier = 0.95
            equity_multiplier = 0.90
        else:
            strategy = "balanced"
            base_multiplier = 1.0
            equity_multiplier = 1.0
        
        return {
            "strategy": strategy,
            "base_salary_multiplier": base_multiplier,
            "equity_multiplier": equity_multiplier,
            "market_conditions": {
                "avg_industry_growth": round(avg_growth, 1),
                "avg_hiring_velocity": round(avg_hiring_velocity, 1),
                "vc_funding_level": round(vc_funding.value, 1)
            },
            "recommendations": self._get_compensation_recommendations(strategy)
        }
    
    def _modify_search_parameters(self, 
                                economic_indicators: Dict[str, EconomicIndicator],
                                industry_health: List[IndustryHealth]) -> Dict[str, Any]:
        """Modify search parameters based on market urgency"""
        # Calculate urgency score
        avg_layoff_risk = np.mean([ih.layoff_risk for ih in industry_health])
        avg_hiring_velocity = np.mean([ih.hiring_velocity for ih in industry_health])
        
        urgency_score = (avg_hiring_velocity - avg_layoff_risk) / 100
        
        # Adjust search parameters
        if urgency_score > 0.3:
            # High urgency - expand search
            location_radius = "global"
            experience_flexibility = "high"
            salary_range_expansion = 1.2
        elif urgency_score > 0.1:
            # Medium urgency - moderate expansion
            location_radius = "regional"
            experience_flexibility = "medium"
            salary_range_expansion = 1.1
        else:
            # Low urgency - be selective
            location_radius = "local"
            experience_flexibility = "low"
            salary_range_expansion = 1.0
        
        return {
            "urgency_score": round(urgency_score, 2),
            "location_radius": location_radius,
            "experience_flexibility": experience_flexibility,
            "salary_range_expansion": salary_range_expansion,
            "search_priority": "high" if urgency_score > 0.3 else "medium" if urgency_score > 0.1 else "low",
            "recommendations": self._get_urgency_recommendations(urgency_score)
        }
    
    def _get_confidence_recommendation(self, confidence: float) -> str:
        """Get hiring confidence recommendation"""
        if confidence > 80:
            return "Excellent time to hire - market conditions are very favorable"
        elif confidence > 60:
            return "Good time to hire - market conditions are favorable"
        elif confidence > 40:
            return "Moderate hiring conditions - proceed with caution"
        else:
            return "Challenging hiring conditions - consider delaying or adjusting strategy"
    
    def _get_timing_recommendation(self, timing_score: float) -> str:
        """Get timing recommendation"""
        if timing_score > 90:
            return "Optimal timing for outreach - high response rates expected"
        elif timing_score > 70:
            return "Good timing for outreach - moderate response rates expected"
        elif timing_score > 50:
            return "Average timing - consider waiting for better conditions"
        else:
            return "Poor timing - delay outreach until conditions improve"
    
    def _get_compensation_recommendations(self, strategy: str) -> List[str]:
        """Get compensation strategy recommendations"""
        recommendations = {
            "aggressive": [
                "Offer 15% above market rate for top talent",
                "Increase equity grants by 25%",
                "Consider signing bonuses for critical roles",
                "Fast-track offers to avoid losing candidates"
            ],
            "competitive": [
                "Match market rates with slight premium for key skills",
                "Standard equity packages with room for negotiation",
                "Focus on total compensation package value",
                "Emphasize growth opportunities and company culture"
            ],
            "conservative": [
                "Stay within budget constraints",
                "Focus on non-monetary benefits and growth",
                "Consider contract-to-hire arrangements",
                "Target candidates open to lower compensation"
            ],
            "balanced": [
                "Offer market-competitive packages",
                "Balance base salary and equity appropriately",
                "Focus on long-term value proposition",
                "Be flexible but within reasonable bounds"
            ]
        }
        return recommendations.get(strategy, [])
    
    def _get_urgency_recommendations(self, urgency_score: float) -> List[str]:
        """Get urgency-based recommendations"""
        if urgency_score > 0.3:
            return [
                "Expand search to global talent pool",
                "Consider remote-first positions",
                "Increase salary ranges by 20%",
                "Accelerate interview process",
                "Consider multiple offers simultaneously"
            ]
        elif urgency_score > 0.1:
            return [
                "Expand to regional talent pool",
                "Be flexible on experience requirements",
                "Increase salary ranges by 10%",
                "Streamline interview process",
                "Prepare backup candidates"
            ]
        else:
            return [
                "Focus on local talent pool",
                "Be selective with experience requirements",
                "Maintain current salary ranges",
                "Take time for thorough evaluation",
                "Build strong candidate pipeline"
            ]


# Convenience functions for direct API access
async def fetch_economic_indicators() -> Dict[str, Any]:
    """Fetch economic indicators for API endpoint"""
    collector = EconomicDataCollector()
    indicators = await collector.fetch_economic_data()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "indicators": {
            key: {
                "name": ind.name,
                "value": ind.value,
                "unit": ind.unit,
                "trend": ind.trend,
                "confidence": ind.confidence,
                "last_updated": ind.last_updated.isoformat(),
                "source": ind.source
            }
            for key, ind in indicators.items()
        }
    }


async def market_timing_intelligence() -> Dict[str, Any]:
    """Generate market timing intelligence for API endpoint"""
    intelligence = MarketTimingIntelligence()
    return await intelligence.market_timing_intelligence()
