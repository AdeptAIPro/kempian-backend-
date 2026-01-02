"""
Data models for Market Intelligence module
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, date
from enum import Enum
import json


class TrendDirection(Enum):
    """Enum for trend directions"""
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    VOLATILE = "volatile"


class SkillCategory(Enum):
    """Enum for skill categories"""
    TECHNICAL = "technical"
    SOFT = "soft"
    DOMAIN_SPECIFIC = "domain_specific"
    EMERGING = "emerging"
    LEGACY = "legacy"


class IndustryType(Enum):
    """Enum for industry types"""
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    MANUFACTURING = "manufacturing"
    CONSULTING = "consulting"
    RETAIL = "retail"
    OTHER = "other"


@dataclass
class SalaryDataPoint:
    """Individual salary data point"""
    position: str
    company: Optional[str] = None
    location: str = "Unknown"
    salary_min: float = 0.0
    salary_max: float = 0.0
    salary_median: float = 0.0
    currency: str = "USD"
    experience_level: str = "Mid-level"
    industry: IndustryType = IndustryType.OTHER
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    
    @property
    def salary_range(self) -> float:
        """Calculate salary range"""
        return self.salary_max - self.salary_min
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "position": self.position,
            "company": self.company,
            "location": self.location,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "salary_median": self.salary_median,
            "currency": self.currency,
            "experience_level": self.experience_level,
            "industry": self.industry.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }


@dataclass
class SkillDemandDataPoint:
    """Individual skill demand data point"""
    skill: str
    category: SkillCategory
    demand_score: float  # 0-100
    job_count: int
    growth_rate: float  # percentage
    industry: IndustryType
    location: str = "Global"
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "skill": self.skill,
            "category": self.category.value,
            "demand_score": self.demand_score,
            "job_count": self.job_count,
            "growth_rate": self.growth_rate,
            "industry": self.industry.value,
            "location": self.location,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }


@dataclass
class SalaryTrend:
    """Salary trend analysis result"""
    position: str
    industry: IndustryType
    location: str
    current_median: float
    trend_direction: TrendDirection
    trend_strength: float  # 0-100
    period_months: int
    data_points: List[SalaryDataPoint] = field(default_factory=list)
    forecast: Optional[Dict[str, float]] = None
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "position": self.position,
            "industry": self.industry.value,
            "location": self.location,
            "current_median": self.current_median,
            "trend_direction": self.trend_direction.value,
            "trend_strength": self.trend_strength,
            "period_months": self.period_months,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "forecast": self.forecast,
            "confidence_score": self.confidence_score
        }


@dataclass
class SkillDemand:
    """Skill demand analysis result"""
    skill: str
    category: SkillCategory
    current_demand: float
    trend_direction: TrendDirection
    trend_strength: float
    growth_rate: float
    job_count: int
    industry_breakdown: Dict[str, float] = field(default_factory=dict)
    location_breakdown: Dict[str, float] = field(default_factory=dict)
    related_skills: List[str] = field(default_factory=list)
    forecast: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "skill": self.skill,
            "category": self.category.value,
            "current_demand": self.current_demand,
            "trend_direction": self.trend_direction.value,
            "trend_strength": self.trend_strength,
            "growth_rate": self.growth_rate,
            "job_count": self.job_count,
            "industry_breakdown": self.industry_breakdown,
            "location_breakdown": self.location_breakdown,
            "related_skills": self.related_skills,
            "forecast": self.forecast
        }


@dataclass
class JobMarketTrend:
    """Job market trend analysis"""
    industry: IndustryType
    location: str
    total_jobs: int
    growth_rate: float
    average_salary: float
    top_skills: List[str] = field(default_factory=list)
    emerging_roles: List[str] = field(default_factory=list)
    declining_roles: List[str] = field(default_factory=list)
    market_competitiveness: float = 0.0  # 0-100
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "industry": self.industry.value,
            "location": self.location,
            "total_jobs": self.total_jobs,
            "growth_rate": self.growth_rate,
            "average_salary": self.average_salary,
            "top_skills": self.top_skills,
            "emerging_roles": self.emerging_roles,
            "declining_roles": self.declining_roles,
            "market_competitiveness": self.market_competitiveness,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class IndustryInsight:
    """Industry-specific market insight"""
    industry: IndustryType
    market_size: float
    growth_rate: float
    key_trends: List[str] = field(default_factory=list)
    top_skills: List[str] = field(default_factory=list)
    salary_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    job_availability: float = 0.0
    competition_level: float = 0.0
    future_outlook: str = "neutral"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "industry": self.industry.value,
            "market_size": self.market_size,
            "growth_rate": self.growth_rate,
            "key_trends": self.key_trends,
            "top_skills": self.top_skills,
            "salary_ranges": self.salary_ranges,
            "job_availability": self.job_availability,
            "competition_level": self.competition_level,
            "future_outlook": self.future_outlook,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MarketForecast:
    """Market forecast data"""
    forecast_type: str  # "salary", "skill_demand", "job_market"
    target: str  # position, skill, or industry
    timeframe_months: int
    predictions: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    factors: List[str] = field(default_factory=list)
    accuracy_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "forecast_type": self.forecast_type,
            "target": self.target,
            "timeframe_months": self.timeframe_months,
            "predictions": self.predictions,
            "confidence_intervals": self.confidence_intervals,
            "factors": self.factors,
            "accuracy_score": self.accuracy_score,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class MarketIntelligenceData:
    """Comprehensive market intelligence data container"""
    salary_trends: List[SalaryTrend] = field(default_factory=list)
    skill_demands: List[SkillDemand] = field(default_factory=list)
    job_market_trends: List[JobMarketTrend] = field(default_factory=list)
    industry_insights: List[IndustryInsight] = field(default_factory=list)
    forecasts: List[MarketForecast] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    data_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "salary_trends": [st.to_dict() for st in self.salary_trends],
            "skill_demands": [sd.to_dict() for sd in self.skill_demands],
            "job_market_trends": [jmt.to_dict() for jmt in self.job_market_trends],
            "industry_insights": [ii.to_dict() for ii in self.industry_insights],
            "forecasts": [f.to_dict() for f in self.forecasts],
            "last_updated": self.last_updated.isoformat(),
            "data_sources": self.data_sources
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_salary_trends": len(self.salary_trends),
            "total_skill_demands": len(self.skill_demands),
            "total_job_market_trends": len(self.job_market_trends),
            "total_industry_insights": len(self.industry_insights),
            "total_forecasts": len(self.forecasts),
            "last_updated": self.last_updated.isoformat(),
            "data_sources_count": len(self.data_sources)
        }
