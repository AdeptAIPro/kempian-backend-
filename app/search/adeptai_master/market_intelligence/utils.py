"""
Utility functions for Market Intelligence module
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import asdict
import hashlib

from .models import (
    MarketIntelligenceData, SalaryTrend, SkillDemand, 
    JobMarketTrend, IndustryInsight, MarketForecast
)

logger = logging.getLogger(__name__)


class MarketIntelligenceCache:
    """Simple in-memory cache for market intelligence data"""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.cache = {}
        self.max_size = max_size
        self.ttl_hours = ttl_hours
    
    def _generate_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data"""
        # Create a hash of the sorted data for consistent keys
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if datetime.now() > entry['expires_at']:
            del self.cache[key]
            return None
        
        return entry['data']
    
    def set(self, key: str, data: Any) -> None:
        """Set data in cache"""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['created_at'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'data': data,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=self.ttl_hours)
        }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)


class DataAggregator:
    """Utility class for aggregating market intelligence data"""
    
    @staticmethod
    def aggregate_salary_data(salary_trends: List[SalaryTrend]) -> Dict[str, Any]:
        """Aggregate salary trends into summary statistics"""
        if not salary_trends:
            return {"total_trends": 0, "average_salary": 0}
        
        salaries = [st.current_median for st in salary_trends]
        trend_directions = [st.trend_direction.value for st in salary_trends]
        
        # Calculate statistics
        avg_salary = np.mean(salaries)
        median_salary = np.median(salaries)
        std_salary = np.std(salaries)
        
        # Count trend directions
        trend_counts = {}
        for direction in trend_directions:
            trend_counts[direction] = trend_counts.get(direction, 0) + 1
        
        # Calculate confidence scores
        confidence_scores = [st.confidence_score for st in salary_trends]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            "total_trends": len(salary_trends),
            "average_salary": float(avg_salary),
            "median_salary": float(median_salary),
            "salary_std": float(std_salary),
            "trend_distribution": trend_counts,
            "average_confidence": float(avg_confidence),
            "salary_range": {
                "min": float(np.min(salaries)),
                "max": float(np.max(salaries))
            }
        }
    
    @staticmethod
    def aggregate_skill_data(skill_demands: List[SkillDemand]) -> Dict[str, Any]:
        """Aggregate skill demands into summary statistics"""
        if not skill_demands:
            return {"total_skills": 0, "average_demand": 0}
        
        demands = [sd.current_demand for sd in skill_demands]
        growth_rates = [sd.growth_rate for sd in skill_demands]
        job_counts = [sd.job_count for sd in skill_demands]
        
        # Calculate statistics
        avg_demand = np.mean(demands)
        avg_growth = np.mean(growth_rates)
        total_jobs = sum(job_counts)
        
        # Categorize skills
        categories = {}
        for sd in skill_demands:
            category = sd.category.value
            if category not in categories:
                categories[category] = []
            categories[category].append(sd.skill)
        
        # Find hot and declining skills
        hot_skills = [sd.skill for sd in skill_demands 
                     if sd.current_demand > 75 and sd.growth_rate > 5]
        declining_skills = [sd.skill for sd in skill_demands 
                           if sd.growth_rate < -2]
        
        return {
            "total_skills": len(skill_demands),
            "average_demand": float(avg_demand),
            "average_growth": float(avg_growth),
            "total_jobs": total_jobs,
            "skill_categories": {k: len(v) for k, v in categories.items()},
            "hot_skills": hot_skills[:10],
            "declining_skills": declining_skills[:10],
            "demand_range": {
                "min": float(np.min(demands)),
                "max": float(np.max(demands))
            }
        }
    
    @staticmethod
    def aggregate_market_data(job_market_trends: List[JobMarketTrend]) -> Dict[str, Any]:
        """Aggregate job market trends into summary statistics"""
        if not job_market_trends:
            return {"total_markets": 0, "average_growth": 0}
        
        growth_rates = [jmt.growth_rate for jmt in job_market_trends]
        total_jobs = [jmt.total_jobs for jmt in job_market_trends]
        avg_salaries = [jmt.average_salary for jmt in job_market_trends]
        
        # Calculate statistics
        avg_growth = np.mean(growth_rates)
        total_job_count = sum(total_jobs)
        avg_salary = np.mean(avg_salaries)
        
        # Group by industry
        industries = {}
        for jmt in job_market_trends:
            industry = jmt.industry.value
            if industry not in industries:
                industries[industry] = {
                    "total_jobs": 0,
                    "growth_rate": 0,
                    "average_salary": 0,
                    "count": 0
                }
            industries[industry]["total_jobs"] += jmt.total_jobs
            industries[industry]["growth_rate"] += jmt.growth_rate
            industries[industry]["average_salary"] += jmt.average_salary
            industries[industry]["count"] += 1
        
        # Calculate averages for each industry
        for industry_data in industries.values():
            count = industry_data["count"]
            industry_data["growth_rate"] /= count
            industry_data["average_salary"] /= count
        
        return {
            "total_markets": len(job_market_trends),
            "average_growth": float(avg_growth),
            "total_jobs": total_job_count,
            "average_salary": float(avg_salary),
            "industry_breakdown": industries,
            "growth_range": {
                "min": float(np.min(growth_rates)),
                "max": float(np.max(growth_rates))
            }
        }


class DataValidator:
    """Utility class for validating market intelligence data"""
    
    @staticmethod
    def validate_salary_data(salary_data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate salary data quality"""
        errors = []
        
        if not salary_data:
            errors.append("No salary data provided")
            return False, errors
        
        required_fields = ["position", "salary_median", "industry", "location"]
        
        for i, data_point in enumerate(salary_data):
            for field in required_fields:
                if field not in data_point:
                    errors.append(f"Missing required field '{field}' in data point {i}")
            
            # Validate salary values
            if "salary_median" in data_point:
                salary = data_point["salary_median"]
                if not isinstance(salary, (int, float)) or salary <= 0:
                    errors.append(f"Invalid salary value in data point {i}: {salary}")
            
            # Validate industry
            if "industry" in data_point:
                industry = data_point["industry"]
                valid_industries = ["technology", "finance", "healthcare", "education", 
                                  "manufacturing", "consulting", "retail", "other"]
                if industry.lower() not in valid_industries:
                    errors.append(f"Invalid industry in data point {i}: {industry}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_skill_data(skill_data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate skill demand data quality"""
        errors = []
        
        if not skill_data:
            errors.append("No skill data provided")
            return False, errors
        
        required_fields = ["skill", "demand_score", "job_count", "industry"]
        
        for i, data_point in enumerate(skill_data):
            for field in required_fields:
                if field not in data_point:
                    errors.append(f"Missing required field '{field}' in data point {i}")
            
            # Validate demand score
            if "demand_score" in data_point:
                demand = data_point["demand_score"]
                if not isinstance(demand, (int, float)) or not (0 <= demand <= 100):
                    errors.append(f"Invalid demand score in data point {i}: {demand}")
            
            # Validate job count
            if "job_count" in data_point:
                count = data_point["job_count"]
                if not isinstance(count, int) or count < 0:
                    errors.append(f"Invalid job count in data point {i}: {count}")
        
        return len(errors) == 0, errors


class ReportGenerator:
    """Utility class for generating market intelligence reports"""
    
    @staticmethod
    def generate_executive_summary(market_data: MarketIntelligenceData) -> Dict[str, Any]:
        """Generate executive summary of market intelligence data"""
        summary = {
            "report_date": market_data.last_updated.isoformat(),
            "data_sources": market_data.data_sources,
            "key_metrics": {},
            "insights": [],
            "recommendations": []
        }
        
        # Calculate key metrics
        if market_data.salary_trends:
            salary_aggregate = DataAggregator.aggregate_salary_data(market_data.salary_trends)
            summary["key_metrics"]["salary"] = {
                "average_salary": salary_aggregate["average_salary"],
                "total_positions_analyzed": salary_aggregate["total_trends"],
                "confidence_score": salary_aggregate["average_confidence"]
            }
        
        if market_data.skill_demands:
            skill_aggregate = DataAggregator.aggregate_skill_data(market_data.skill_demands)
            summary["key_metrics"]["skills"] = {
                "average_demand": skill_aggregate["average_demand"],
                "total_skills_analyzed": skill_aggregate["total_skills"],
                "hot_skills_count": len(skill_aggregate["hot_skills"])
            }
        
        if market_data.job_market_trends:
            market_aggregate = DataAggregator.aggregate_market_data(market_data.job_market_trends)
            summary["key_metrics"]["job_market"] = {
                "average_growth": market_aggregate["average_growth"],
                "total_jobs": market_aggregate["total_jobs"],
                "markets_analyzed": market_aggregate["total_markets"]
            }
        
        # Generate insights
        summary["insights"] = ReportGenerator._generate_insights(market_data)
        
        # Generate recommendations
        summary["recommendations"] = ReportGenerator._generate_recommendations(market_data)
        
        return summary
    
    @staticmethod
    def _generate_insights(market_data: MarketIntelligenceData) -> List[str]:
        """Generate key insights from market data"""
        insights = []
        
        # Salary insights
        if market_data.salary_trends:
            rising_trends = [st for st in market_data.salary_trends 
                           if st.trend_direction.value == "rising"]
            if len(rising_trends) > len(market_data.salary_trends) * 0.6:
                insights.append("Strong upward salary trends observed across most positions")
            elif len(rising_trends) < len(market_data.salary_trends) * 0.3:
                insights.append("Salary growth appears to be slowing in many positions")
        
        # Skill insights
        if market_data.skill_demands:
            high_demand_skills = [sd for sd in market_data.skill_demands 
                                if sd.current_demand > 80]
            if high_demand_skills:
                skill_names = [sd.skill for sd in high_demand_skills[:5]]
                insights.append(f"High demand skills identified: {', '.join(skill_names)}")
        
        # Market insights
        if market_data.job_market_trends:
            positive_growth = [jmt for jmt in market_data.job_market_trends 
                             if jmt.growth_rate > 0]
            if len(positive_growth) > len(market_data.job_market_trends) * 0.8:
                insights.append("Job market showing strong growth across most industries")
        
        return insights
    
    @staticmethod
    def _generate_recommendations(market_data: MarketIntelligenceData) -> List[str]:
        """Generate recommendations based on market data"""
        recommendations = []
        
        # Skill recommendations
        if market_data.skill_demands:
            hot_skills = [sd for sd in market_data.skill_demands 
                         if sd.current_demand > 75 and sd.growth_rate > 5]
            if hot_skills:
                skill_names = [sd.skill for sd in hot_skills[:3]]
                recommendations.append(f"Consider developing expertise in: {', '.join(skill_names)}")
        
        # Salary recommendations
        if market_data.salary_trends:
            high_growth_positions = [st for st in market_data.salary_trends 
                                   if st.trend_direction.value == "rising" and st.trend_strength > 70]
            if high_growth_positions:
                position_names = [st.position for st in high_growth_positions[:3]]
                recommendations.append(f"High salary growth positions: {', '.join(position_names)}")
        
        # Market recommendations
        if market_data.job_market_trends:
            high_growth_industries = [jmt for jmt in market_data.job_market_trends 
                                    if jmt.growth_rate > 10]
            if high_growth_industries:
                industry_names = [jmt.industry.value for jmt in high_growth_industries[:3]]
                recommendations.append(f"Consider opportunities in growing industries: {', '.join(industry_names)}")
        
        return recommendations


class DataExporter:
    """Utility class for exporting market intelligence data"""
    
    @staticmethod
    def export_to_json(market_data: MarketIntelligenceData, filepath: str) -> bool:
        """Export market intelligence data to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(market_data.to_dict(), f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to export data to JSON: {e}")
            return False
    
    @staticmethod
    def export_to_csv(market_data: MarketIntelligenceData, filepath: str) -> bool:
        """Export market intelligence data to CSV files"""
        try:
            # Export salary trends
            if market_data.salary_trends:
                salary_df = pd.DataFrame([st.to_dict() for st in market_data.salary_trends])
                salary_df.to_csv(f"{filepath}_salary_trends.csv", index=False)
            
            # Export skill demands
            if market_data.skill_demands:
                skill_df = pd.DataFrame([sd.to_dict() for sd in market_data.skill_demands])
                skill_df.to_csv(f"{filepath}_skill_demands.csv", index=False)
            
            # Export job market trends
            if market_data.job_market_trends:
                job_df = pd.DataFrame([jmt.to_dict() for jmt in market_data.job_market_trends])
                job_df.to_csv(f"{filepath}_job_market_trends.csv", index=False)
            
            return True
        except Exception as e:
            logger.error(f"Failed to export data to CSV: {e}")
            return False
