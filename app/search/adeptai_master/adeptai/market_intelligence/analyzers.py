"""
Analyzers for Market Intelligence module
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter, defaultdict
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import math

from .models import (
    SalaryDataPoint, SkillDemandDataPoint, SalaryTrend, SkillDemand,
    JobMarketTrend, IndustryInsight, MarketForecast, TrendDirection,
    SkillCategory, IndustryType
)

logger = logging.getLogger(__name__)


class BaseAnalyzer:
    """Base class for all analyzers"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _calculate_trend_direction(self, values: List[float], threshold: float = 0.05) -> TrendDirection:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return TrendDirection.STABLE
        
        # Calculate linear regression slope
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        
        # Normalize slope by the mean value
        mean_value = np.mean(values)
        normalized_slope = slope / mean_value if mean_value != 0 else 0
        
        # Determine trend direction
        if abs(normalized_slope) < threshold:
            return TrendDirection.STABLE
        elif normalized_slope > threshold:
            return TrendDirection.RISING
        elif normalized_slope < -threshold:
            return TrendDirection.FALLING
        else:
            return TrendDirection.VOLATILE
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate trend strength (0-100) from a list of values"""
        if len(values) < 2:
            return 0.0
        
        # Calculate coefficient of determination (R²)
        x = np.arange(len(values))
        slope, intercept, r_value, _, _ = stats.linregress(x, values)
        r_squared = r_value ** 2
        
        # Convert to 0-100 scale
        return min(100.0, max(0.0, r_squared * 100))
    
    def _calculate_confidence_score(self, data_points: List[Any], min_points: int = 5) -> float:
        """Calculate confidence score based on data quality and quantity"""
        if len(data_points) < min_points:
            return max(0.0, (len(data_points) / min_points) * 50)
        
        # Base confidence on number of data points
        base_confidence = min(90.0, 50 + (len(data_points) - min_points) * 5)
        
        # Adjust based on data recency
        now = datetime.now()
        recent_points = sum(1 for dp in data_points 
                           if hasattr(dp, 'timestamp') and 
                           (now - dp.timestamp).days <= 30)
        
        recency_factor = min(1.0, recent_points / len(data_points))
        return min(100.0, base_confidence * recency_factor)


class SalaryTrendAnalyzer(BaseAnalyzer):
    """Analyzes salary trends from collected data"""
    
    def analyze_salary_trends(self, 
                            salary_data: List[SalaryDataPoint], 
                            period_months: int = 12) -> List[SalaryTrend]:
        """Analyze salary trends from collected data"""
        trends = []
        
        # Group data by position, industry, and location
        grouped_data = self._group_salary_data(salary_data)
        
        for key, data_points in grouped_data.items():
            try:
                trend = self._analyze_single_salary_trend(data_points, period_months)
                if trend:
                    trends.append(trend)
            except Exception as e:
                self.logger.error(f"Failed to analyze salary trend for {key}: {e}")
                continue
        
        return trends
    
    def _group_salary_data(self, salary_data: List[SalaryDataPoint]) -> Dict[Tuple[str, str, str], List[SalaryDataPoint]]:
        """Group salary data by position, industry, and location"""
        grouped = defaultdict(list)
        
        for data_point in salary_data:
            key = (data_point.position, data_point.industry.value, data_point.location)
            grouped[key].append(data_point)
        
        return dict(grouped)
    
    def _analyze_single_salary_trend(self, 
                                   data_points: List[SalaryDataPoint], 
                                   period_months: int) -> Optional[SalaryTrend]:
        """Analyze trend for a single position/industry/location combination"""
        if len(data_points) < 2:
            return None
        
        # Sort by timestamp
        data_points.sort(key=lambda x: x.timestamp)
        
        # Extract median salaries over time
        salaries = [dp.salary_median for dp in data_points]
        timestamps = [dp.timestamp for dp in data_points]
        
        # Calculate trend direction and strength
        trend_direction = self._calculate_trend_direction(salaries)
        trend_strength = self._calculate_trend_strength(salaries)
        
        # Calculate current median
        current_median = np.median(salaries)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(data_points)
        
        # Generate forecast
        forecast = self._generate_salary_forecast(salaries, timestamps, period_months)
        
        # Get position, industry, and location from first data point
        first_dp = data_points[0]
        
        return SalaryTrend(
            position=first_dp.position,
            industry=first_dp.industry,
            location=first_dp.location,
            current_median=current_median,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            period_months=period_months,
            data_points=data_points,
            forecast=forecast,
            confidence_score=confidence_score
        )
    
    def _generate_salary_forecast(self, 
                                salaries: List[float], 
                                timestamps: List[datetime], 
                                period_months: int) -> Optional[Dict[str, float]]:
        """Generate salary forecast using linear regression"""
        if len(salaries) < 3:
            return None
        
        try:
            # Convert timestamps to numeric values (days since first timestamp)
            first_timestamp = timestamps[0]
            x = [(ts - first_timestamp).days for ts in timestamps]
            
            # Fit linear regression
            x = np.array(x).reshape(-1, 1)
            y = np.array(salaries)
            
            model = LinearRegression()
            model.fit(x, y)
            
            # Predict future values
            future_days = [30, 60, 90, 180, 365]  # 1, 2, 3, 6, 12 months
            future_salaries = model.predict(np.array(future_days).reshape(-1, 1))
            
            return {
                "1_month": float(future_salaries[0]),
                "2_months": float(future_salaries[1]),
                "3_months": float(future_salaries[2]),
                "6_months": float(future_salaries[3]),
                "12_months": float(future_salaries[4])
            }
        except Exception as e:
            self.logger.warning(f"Failed to generate salary forecast: {e}")
            return None


class SkillDemandAnalyzer(BaseAnalyzer):
    """Analyzes skill demand trends from collected data"""
    
    def analyze_skill_demands(self, 
                            skill_data: List[SkillDemandDataPoint]) -> List[SkillDemand]:
        """Analyze skill demand trends from collected data"""
        demands = []
        
        # Group data by skill
        grouped_data = self._group_skill_data(skill_data)
        
        for skill, data_points in grouped_data.items():
            try:
                demand = self._analyze_single_skill_demand(data_points)
                if demand:
                    demands.append(demand)
            except Exception as e:
                self.logger.error(f"Failed to analyze skill demand for {skill}: {e}")
                continue
        
        return demands
    
    def _group_skill_data(self, skill_data: List[SkillDemandDataPoint]) -> Dict[str, List[SkillDemandDataPoint]]:
        """Group skill data by skill name"""
        grouped = defaultdict(list)
        
        for data_point in skill_data:
            grouped[data_point.skill].append(data_point)
        
        return dict(grouped)
    
    def _analyze_single_skill_demand(self, data_points: List[SkillDemandDataPoint]) -> Optional[SkillDemand]:
        """Analyze demand for a single skill"""
        if len(data_points) < 1:
            return None
        
        # Sort by timestamp
        data_points.sort(key=lambda x: x.timestamp)
        
        # Extract demand scores and job counts over time
        demand_scores = [dp.demand_score for dp in data_points]
        job_counts = [dp.job_count for dp in data_points]
        growth_rates = [dp.growth_rate for dp in data_points]
        
        # Calculate trend direction and strength
        trend_direction = self._calculate_trend_direction(demand_scores)
        trend_strength = self._calculate_trend_strength(demand_scores)
        
        # Calculate current metrics
        current_demand = np.mean(demand_scores)
        current_job_count = np.sum(job_counts)
        avg_growth_rate = np.mean(growth_rates)
        
        # Calculate industry and location breakdowns
        industry_breakdown = self._calculate_industry_breakdown(data_points)
        location_breakdown = self._calculate_location_breakdown(data_points)
        
        # Find related skills
        related_skills = self._find_related_skills(data_points)
        
        # Generate forecast
        forecast = self._generate_skill_demand_forecast(demand_scores, data_points)
        
        # Get skill info from first data point
        first_dp = data_points[0]
        
        return SkillDemand(
            skill=first_dp.skill,
            category=first_dp.category,
            current_demand=current_demand,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            growth_rate=avg_growth_rate,
            job_count=current_job_count,
            industry_breakdown=industry_breakdown,
            location_breakdown=location_breakdown,
            related_skills=related_skills,
            forecast=forecast
        )
    
    def _calculate_industry_breakdown(self, data_points: List[SkillDemandDataPoint]) -> Dict[str, float]:
        """Calculate demand breakdown by industry"""
        industry_demands = defaultdict(list)
        
        for dp in data_points:
            industry_demands[dp.industry.value].append(dp.demand_score)
        
        breakdown = {}
        for industry, demands in industry_demands.items():
            breakdown[industry] = float(np.mean(demands))
        
        return breakdown
    
    def _calculate_location_breakdown(self, data_points: List[SkillDemandDataPoint]) -> Dict[str, float]:
        """Calculate demand breakdown by location"""
        location_demands = defaultdict(list)
        
        for dp in data_points:
            location_demands[dp.location].append(dp.demand_score)
        
        breakdown = {}
        for location, demands in location_demands.items():
            breakdown[location] = float(np.mean(demands))
        
        return breakdown
    
    def _find_related_skills(self, data_points: List[SkillDemandDataPoint]) -> List[str]:
        """Find related skills based on co-occurrence patterns"""
        # This is a simplified implementation
        # In a real system, you would use more sophisticated techniques
        
        skill_categories = {
            SkillCategory.TECHNICAL: ["Python", "JavaScript", "Machine Learning", "AWS", "Docker"],
            SkillCategory.SOFT: ["Leadership", "Communication", "Teamwork", "Problem Solving"],
            SkillCategory.EMERGING: ["Machine Learning", "TensorFlow", "PyTorch", "Kubernetes", "Docker"]
        }
        
        if not data_points:
            return []
        
        category = data_points[0].category
        related = skill_categories.get(category, [])
        
        # Remove the current skill from related skills
        current_skill = data_points[0].skill
        return [skill for skill in related if skill != current_skill][:5]
    
    def _generate_skill_demand_forecast(self, 
                                      demand_scores: List[float], 
                                      data_points: List[SkillDemandDataPoint]) -> Optional[Dict[str, float]]:
        """Generate skill demand forecast"""
        if len(demand_scores) < 3:
            return None
        
        try:
            # Use simple linear regression for forecasting
            x = np.arange(len(demand_scores)).reshape(-1, 1)
            y = np.array(demand_scores)
            
            model = LinearRegression()
            model.fit(x, y)
            
            # Predict future values
            future_periods = [1, 2, 3, 6, 12]  # months
            future_demands = model.predict(np.array(future_periods).reshape(-1, 1))
            
            return {
                "1_month": float(future_demands[0]),
                "2_months": float(future_demands[1]),
                "3_months": float(future_demands[2]),
                "6_months": float(future_demands[3]),
                "12_months": float(future_demands[4])
            }
        except Exception as e:
            self.logger.warning(f"Failed to generate skill demand forecast: {e}")
            return None


class MarketTrendAnalyzer(BaseAnalyzer):
    """Analyzes overall market trends"""
    
    def analyze_market_trends(self, 
                            salary_trends: List[SalaryTrend],
                            skill_demands: List[SkillDemand],
                            job_market_trends: List[JobMarketTrend]) -> Dict[str, Any]:
        """Analyze overall market trends"""
        analysis = {
            "market_health_score": self._calculate_market_health_score(salary_trends, skill_demands, job_market_trends),
            "hot_skills": self._identify_hot_skills(skill_demands),
            "declining_skills": self._identify_declining_skills(skill_demands),
            "salary_growth_industries": self._identify_salary_growth_industries(salary_trends),
            "job_growth_industries": self._identify_job_growth_industries(job_market_trends),
            "market_volatility": self._calculate_market_volatility(salary_trends, skill_demands),
            "emerging_trends": self._identify_emerging_trends(skill_demands, job_market_trends),
            "risk_factors": self._identify_risk_factors(salary_trends, skill_demands, job_market_trends)
        }
        
        return analysis
    
    def _calculate_market_health_score(self, 
                                     salary_trends: List[SalaryTrend],
                                     skill_demands: List[SkillDemand],
                                     job_market_trends: List[JobMarketTrend]) -> float:
        """Calculate overall market health score (0-100)"""
        if not salary_trends and not skill_demands and not job_market_trends:
            return 50.0
        
        scores = []
        
        # Salary trend score
        if salary_trends:
            rising_trends = sum(1 for st in salary_trends if st.trend_direction == TrendDirection.RISING)
            total_trends = len(salary_trends)
            salary_score = (rising_trends / total_trends) * 100 if total_trends > 0 else 50
            scores.append(salary_score)
        
        # Skill demand score
        if skill_demands:
            high_demand_skills = sum(1 for sd in skill_demands if sd.current_demand > 70)
            total_skills = len(skill_demands)
            skill_score = (high_demand_skills / total_skills) * 100 if total_skills > 0 else 50
            scores.append(skill_score)
        
        # Job market score
        if job_market_trends:
            positive_growth = sum(1 for jmt in job_market_trends if jmt.growth_rate > 0)
            total_markets = len(job_market_trends)
            job_score = (positive_growth / total_markets) * 100 if total_markets > 0 else 50
            scores.append(job_score)
        
        return float(np.mean(scores)) if scores else 50.0
    
    def _identify_hot_skills(self, skill_demands: List[SkillDemand]) -> List[Dict[str, Any]]:
        """Identify hot skills based on demand and growth"""
        hot_skills = []
        
        for skill_demand in skill_demands:
            if (skill_demand.current_demand > 75 and 
                skill_demand.trend_direction == TrendDirection.RISING and
                skill_demand.growth_rate > 5):
                hot_skills.append({
                    "skill": skill_demand.skill,
                    "demand_score": skill_demand.current_demand,
                    "growth_rate": skill_demand.growth_rate,
                    "trend_strength": skill_demand.trend_strength
                })
        
        # Sort by demand score
        hot_skills.sort(key=lambda x: x["demand_score"], reverse=True)
        return hot_skills[:10]
    
    def _identify_declining_skills(self, skill_demands: List[SkillDemand]) -> List[Dict[str, Any]]:
        """Identify declining skills"""
        declining_skills = []
        
        for skill_demand in skill_demands:
            if (skill_demand.trend_direction == TrendDirection.FALLING and
                skill_demand.growth_rate < -2):
                declining_skills.append({
                    "skill": skill_demand.skill,
                    "demand_score": skill_demand.current_demand,
                    "growth_rate": skill_demand.growth_rate,
                    "trend_strength": skill_demand.trend_strength
                })
        
        # Sort by growth rate (most negative first)
        declining_skills.sort(key=lambda x: x["growth_rate"])
        return declining_skills[:10]
    
    def _identify_salary_growth_industries(self, salary_trends: List[SalaryTrend]) -> List[Dict[str, Any]]:
        """Identify industries with strong salary growth"""
        industry_growth = defaultdict(list)
        
        for trend in salary_trends:
            if trend.trend_direction == TrendDirection.RISING:
                industry_growth[trend.industry.value].append(trend.trend_strength)
        
        growth_industries = []
        for industry, strengths in industry_growth.items():
            avg_strength = np.mean(strengths)
            growth_industries.append({
                "industry": industry,
                "average_trend_strength": avg_strength,
                "trend_count": len(strengths)
            })
        
        # Sort by average trend strength
        growth_industries.sort(key=lambda x: x["average_trend_strength"], reverse=True)
        return growth_industries[:5]
    
    def _identify_job_growth_industries(self, job_market_trends: List[JobMarketTrend]) -> List[Dict[str, Any]]:
        """Identify industries with strong job growth"""
        growth_industries = []
        
        for trend in job_market_trends:
            if trend.growth_rate > 5:  # 5% growth threshold
                growth_industries.append({
                    "industry": trend.industry.value,
                    "growth_rate": trend.growth_rate,
                    "total_jobs": trend.total_jobs,
                    "location": trend.location
                })
        
        # Sort by growth rate
        growth_industries.sort(key=lambda x: x["growth_rate"], reverse=True)
        return growth_industries[:5]
    
    def _calculate_market_volatility(self, 
                                   salary_trends: List[SalaryTrend],
                                   skill_demands: List[SkillDemand]) -> float:
        """Calculate market volatility score (0-100)"""
        volatility_scores = []
        
        # Salary volatility
        for trend in salary_trends:
            if len(trend.data_points) > 1:
                salaries = [dp.salary_median for dp in trend.data_points]
                cv = np.std(salaries) / np.mean(salaries) if np.mean(salaries) > 0 else 0
                volatility_scores.append(cv * 100)
        
        # Skill demand volatility
        for skill_demand in skill_demands:
            if skill_demand.trend_direction == TrendDirection.VOLATILE:
                volatility_scores.append(skill_demand.trend_strength)
        
        return float(np.mean(volatility_scores)) if volatility_scores else 0.0
    
    def _identify_emerging_trends(self, 
                                skill_demands: List[SkillDemand],
                                job_market_trends: List[JobMarketTrend]) -> List[str]:
        """Identify emerging market trends"""
        trends = []
        
        # Emerging skills
        emerging_skills = [sd.skill for sd in skill_demands 
                          if sd.category == SkillCategory.EMERGING and 
                          sd.trend_direction == TrendDirection.RISING]
        
        if emerging_skills:
            trends.append(f"Emerging technical skills: {', '.join(emerging_skills[:5])}")
        
        # High growth industries
        high_growth_industries = [jmt.industry.value for jmt in job_market_trends 
                                if jmt.growth_rate > 10]
        
        if high_growth_industries:
            trends.append(f"High growth industries: {', '.join(high_growth_industries)}")
        
        # Remote work trends
        remote_skills = [sd.skill for sd in skill_demands 
                        if "remote" in sd.skill.lower() or "virtual" in sd.skill.lower()]
        
        if remote_skills:
            trends.append(f"Remote work skills in demand: {', '.join(remote_skills)}")
        
        return trends
    
    def _identify_risk_factors(self, 
                             salary_trends: List[SalaryTrend],
                             skill_demands: List[SkillDemand],
                             job_market_trends: List[JobMarketTrend]) -> List[str]:
        """Identify potential risk factors in the market"""
        risks = []
        
        # Declining salary trends
        declining_salaries = [st for st in salary_trends if st.trend_direction == TrendDirection.FALLING]
        if len(declining_salaries) > len(salary_trends) * 0.3:  # More than 30% declining
            risks.append("High percentage of declining salary trends")
        
        # Low demand skills
        low_demand_skills = [sd for sd in skill_demands if sd.current_demand < 30]
        if len(low_demand_skills) > len(skill_demands) * 0.4:  # More than 40% low demand
            risks.append("High percentage of low-demand skills")
        
        # Declining job markets
        declining_markets = [jmt for jmt in job_market_trends if jmt.growth_rate < 0]
        if len(declining_markets) > len(job_market_trends) * 0.2:  # More than 20% declining
            risks.append("Multiple declining job markets")
        
        # High volatility
        volatility = self._calculate_market_volatility(salary_trends, skill_demands)
        if volatility > 70:
            risks.append("High market volatility")
        
        return risks


class IndustryAnalyzer(BaseAnalyzer):
    """Analyzes industry-specific trends and insights"""
    
    def analyze_industry_trends(self, 
                              industry_insights: List[IndustryInsight],
                              salary_trends: List[SalaryTrend],
                              skill_demands: List[SkillDemand]) -> Dict[str, Any]:
        """Analyze industry-specific trends"""
        analysis = {}
        
        for insight in industry_insights:
            industry = insight.industry.value
            
            # Get related salary trends
            industry_salary_trends = [st for st in salary_trends if st.industry.value == industry]
            
            # Get related skill demands
            industry_skill_demands = [sd for sd in skill_demands 
                                    if any(industry in sd.industry_breakdown for industry in [industry])]
            
            analysis[industry] = {
                "market_size": insight.market_size,
                "growth_rate": insight.growth_rate,
                "future_outlook": insight.future_outlook,
                "salary_analysis": self._analyze_industry_salaries(industry_salary_trends),
                "skill_analysis": self._analyze_industry_skills(industry_skill_demands),
                "competitiveness": insight.competition_level,
                "job_availability": insight.job_availability,
                "key_trends": insight.key_trends,
                "top_skills": insight.top_skills
            }
        
        return analysis
    
    def _analyze_industry_salaries(self, salary_trends: List[SalaryTrend]) -> Dict[str, Any]:
        """Analyze salary trends for a specific industry"""
        if not salary_trends:
            return {"average_salary": 0, "trend_direction": "unknown", "salary_growth": 0}
        
        salaries = [st.current_median for st in salary_trends]
        trend_directions = [st.trend_direction for st in salary_trends]
        
        # Calculate average salary
        avg_salary = np.mean(salaries)
        
        # Determine overall trend direction
        rising_count = sum(1 for td in trend_directions if td == TrendDirection.RISING)
        falling_count = sum(1 for td in trend_directions if td == TrendDirection.FALLING)
        
        if rising_count > falling_count:
            trend_direction = "rising"
        elif falling_count > rising_count:
            trend_direction = "falling"
        else:
            trend_direction = "stable"
        
        # Calculate salary growth rate
        salary_growth = 0
        if len(salary_trends) > 0:
            # Use the first trend's forecast if available
            first_trend = salary_trends[0]
            if first_trend.forecast:
                current = first_trend.current_median
                future = first_trend.forecast.get("12_months", current)
                salary_growth = ((future - current) / current) * 100 if current > 0 else 0
        
        return {
            "average_salary": float(avg_salary),
            "trend_direction": trend_direction,
            "salary_growth": float(salary_growth)
        }
    
    def _analyze_industry_skills(self, skill_demands: List[SkillDemand]) -> Dict[str, Any]:
        """Analyze skill demands for a specific industry"""
        if not skill_demands:
            return {"top_skills": [], "skill_diversity": 0, "emerging_skills": []}
        
        # Sort skills by demand
        sorted_skills = sorted(skill_demands, key=lambda x: x.current_demand, reverse=True)
        top_skills = [sd.skill for sd in sorted_skills[:10]]
        
        # Calculate skill diversity (number of different skill categories)
        categories = set(sd.category for sd in skill_demands)
        skill_diversity = len(categories)
        
        # Identify emerging skills
        emerging_skills = [sd.skill for sd in skill_demands 
                          if sd.category == SkillCategory.EMERGING and 
                          sd.trend_direction == TrendDirection.RISING]
        
        return {
            "top_skills": top_skills,
            "skill_diversity": skill_diversity,
            "emerging_skills": emerging_skills
        }
