"""
Data collectors for Market Intelligence module
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import time
import random

from .models import (
    SalaryDataPoint, SkillDemandDataPoint, IndustryType, 
    SkillCategory, SalaryTrend, SkillDemand, JobMarketTrend, IndustryInsight
)

logger = logging.getLogger(__name__)


class BaseDataCollector:
    """Base class for data collectors"""
    
    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = None
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()


class SalaryDataCollector(BaseDataCollector):
    """Collects salary data from various sources"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        super().__init__(rate_limit=0.5)
        self.api_keys = api_keys or {}
        self.sources = {
            'glassdoor': self._collect_glassdoor_data,
            'indeed': self._collect_indeed_data,
            'linkedin': self._collect_linkedin_data,
            'payscale': self._collect_payscale_data,
            'salary_com': self._collect_salary_com_data
        }
    
    async def collect_salary_data(self, 
                                positions: List[str], 
                                locations: List[str] = None,
                                industries: List[IndustryType] = None) -> List[SalaryDataPoint]:
        """Collect salary data for given positions and locations"""
        if locations is None:
            locations = ["Global"]
        if industries is None:
            industries = list(IndustryType)
        
        all_data = []
        
        for position in positions:
            for location in locations:
                for industry in industries:
                    try:
                        # Collect from multiple sources
                        for source_name, collector_func in self.sources.items():
                            try:
                                data_points = await collector_func(position, location, industry)
                                all_data.extend(data_points)
                                await self._rate_limit()
                            except Exception as e:
                                logger.warning(f"Failed to collect from {source_name}: {e}")
                                continue
                    except Exception as e:
                        logger.error(f"Failed to collect salary data for {position} in {location}: {e}")
                        continue
        
        return all_data
    
    async def _collect_glassdoor_data(self, position: str, location: str, industry: IndustryType) -> List[SalaryDataPoint]:
        """Collect data from Glassdoor (simulated)"""
        # In a real implementation, you would use Glassdoor's API
        # For now, we'll simulate realistic data
        await self._rate_limit()
        
        # Simulate API call delay
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Generate realistic salary data
        base_salaries = {
            "Software Engineer": (80000, 150000),
            "Data Scientist": (90000, 160000),
            "Product Manager": (100000, 180000),
            "DevOps Engineer": (85000, 155000),
            "UX Designer": (70000, 130000),
            "Marketing Manager": (60000, 120000),
            "Sales Manager": (65000, 140000),
            "Financial Analyst": (55000, 100000)
        }
        
        salary_range = base_salaries.get(position, (50000, 100000))
        
        # Add location and industry modifiers
        location_multiplier = self._get_location_multiplier(location)
        industry_multiplier = self._get_industry_multiplier(industry)
        
        min_salary = int(salary_range[0] * location_multiplier * industry_multiplier)
        max_salary = int(salary_range[1] * location_multiplier * industry_multiplier)
        median_salary = (min_salary + max_salary) / 2
        
        return [SalaryDataPoint(
            position=position,
            location=location,
            salary_min=min_salary,
            salary_max=max_salary,
            salary_median=median_salary,
            industry=industry,
            source="glassdoor",
            timestamp=datetime.now()
        )]
    
    async def _collect_indeed_data(self, position: str, location: str, industry: IndustryType) -> List[SalaryDataPoint]:
        """Collect data from Indeed (simulated)"""
        await self._rate_limit()
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Similar to Glassdoor but with slight variations
        base_salaries = {
            "Software Engineer": (75000, 145000),
            "Data Scientist": (85000, 155000),
            "Product Manager": (95000, 175000),
            "DevOps Engineer": (80000, 150000),
            "UX Designer": (65000, 125000),
            "Marketing Manager": (55000, 115000),
            "Sales Manager": (60000, 135000),
            "Financial Analyst": (50000, 95000)
        }
        
        salary_range = base_salaries.get(position, (45000, 95000))
        location_multiplier = self._get_location_multiplier(location)
        industry_multiplier = self._get_industry_multiplier(industry)
        
        min_salary = int(salary_range[0] * location_multiplier * industry_multiplier)
        max_salary = int(salary_range[1] * location_multiplier * industry_multiplier)
        median_salary = (min_salary + max_salary) / 2
        
        return [SalaryDataPoint(
            position=position,
            location=location,
            salary_min=min_salary,
            salary_max=max_salary,
            salary_median=median_salary,
            industry=industry,
            source="indeed",
            timestamp=datetime.now()
        )]
    
    async def _collect_linkedin_data(self, position: str, location: str, industry: IndustryType) -> List[SalaryDataPoint]:
        """Collect data from LinkedIn (simulated)"""
        await self._rate_limit()
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # LinkedIn typically has higher salaries
        base_salaries = {
            "Software Engineer": (85000, 160000),
            "Data Scientist": (95000, 170000),
            "Product Manager": (105000, 190000),
            "DevOps Engineer": (90000, 165000),
            "UX Designer": (75000, 140000),
            "Marketing Manager": (65000, 130000),
            "Sales Manager": (70000, 150000),
            "Financial Analyst": (60000, 110000)
        }
        
        salary_range = base_salaries.get(position, (55000, 110000))
        location_multiplier = self._get_location_multiplier(location)
        industry_multiplier = self._get_industry_multiplier(industry)
        
        min_salary = int(salary_range[0] * location_multiplier * industry_multiplier)
        max_salary = int(salary_range[1] * location_multiplier * industry_multiplier)
        median_salary = (min_salary + max_salary) / 2
        
        return [SalaryDataPoint(
            position=position,
            location=location,
            salary_min=min_salary,
            salary_max=max_salary,
            salary_median=median_salary,
            industry=industry,
            source="linkedin",
            timestamp=datetime.now()
        )]
    
    async def _collect_payscale_data(self, position: str, location: str, industry: IndustryType) -> List[SalaryDataPoint]:
        """Collect data from PayScale (simulated)"""
        await self._rate_limit()
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # PayScale typically has more conservative estimates
        base_salaries = {
            "Software Engineer": (70000, 130000),
            "Data Scientist": (80000, 140000),
            "Product Manager": (90000, 160000),
            "DevOps Engineer": (75000, 135000),
            "UX Designer": (60000, 115000),
            "Marketing Manager": (50000, 105000),
            "Sales Manager": (55000, 125000),
            "Financial Analyst": (45000, 90000)
        }
        
        salary_range = base_salaries.get(position, (40000, 90000))
        location_multiplier = self._get_location_multiplier(location)
        industry_multiplier = self._get_industry_multiplier(industry)
        
        min_salary = int(salary_range[0] * location_multiplier * industry_multiplier)
        max_salary = int(salary_range[1] * location_multiplier * industry_multiplier)
        median_salary = (min_salary + max_salary) / 2
        
        return [SalaryDataPoint(
            position=position,
            location=location,
            salary_min=min_salary,
            salary_max=max_salary,
            salary_median=median_salary,
            industry=industry,
            source="payscale",
            timestamp=datetime.now()
        )]
    
    async def _collect_salary_com_data(self, position: str, location: str, industry: IndustryType) -> List[SalaryDataPoint]:
        """Collect data from Salary.com (simulated)"""
        await self._rate_limit()
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Salary.com provides comprehensive data
        base_salaries = {
            "Software Engineer": (78000, 142000),
            "Data Scientist": (88000, 152000),
            "Product Manager": (98000, 172000),
            "DevOps Engineer": (83000, 147000),
            "UX Designer": (68000, 127000),
            "Marketing Manager": (58000, 117000),
            "Sales Manager": (63000, 137000),
            "Financial Analyst": (53000, 97000)
        }
        
        salary_range = base_salaries.get(position, (48000, 97000))
        location_multiplier = self._get_location_multiplier(location)
        industry_multiplier = self._get_industry_multiplier(industry)
        
        min_salary = int(salary_range[0] * location_multiplier * industry_multiplier)
        max_salary = int(salary_range[1] * location_multiplier * industry_multiplier)
        median_salary = (min_salary + max_salary) / 2
        
        return [SalaryDataPoint(
            position=position,
            location=location,
            salary_min=min_salary,
            salary_max=max_salary,
            salary_median=median_salary,
            industry=industry,
            source="salary_com",
            timestamp=datetime.now()
        )]
    
    def _get_location_multiplier(self, location: str) -> float:
        """Get salary multiplier based on location"""
        location_multipliers = {
            "San Francisco": 1.4,
            "New York": 1.3,
            "Seattle": 1.2,
            "Boston": 1.15,
            "Los Angeles": 1.1,
            "Chicago": 1.05,
            "Austin": 1.0,
            "Denver": 0.95,
            "Atlanta": 0.9,
            "Dallas": 0.85,
            "Global": 1.0
        }
        return location_multipliers.get(location, 1.0)
    
    def _get_industry_multiplier(self, industry: IndustryType) -> float:
        """Get salary multiplier based on industry"""
        industry_multipliers = {
            IndustryType.TECHNOLOGY: 1.2,
            IndustryType.FINANCE: 1.15,
            IndustryType.CONSULTING: 1.1,
            IndustryType.HEALTHCARE: 1.05,
            IndustryType.EDUCATION: 0.9,
            IndustryType.MANUFACTURING: 0.95,
            IndustryType.RETAIL: 0.85,
            IndustryType.OTHER: 1.0
        }
        return industry_multipliers.get(industry, 1.0)


class SkillDemandCollector(BaseDataCollector):
    """Collects skill demand data from various sources"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        super().__init__(rate_limit=0.3)
        self.api_keys = api_keys or {}
        self.sources = {
            'linkedin_jobs': self._collect_linkedin_jobs_data,
            'indeed_jobs': self._collect_indeed_jobs_data,
            'github_trends': self._collect_github_trends_data,
            'stack_overflow': self._collect_stack_overflow_data
        }
    
    async def collect_skill_demand_data(self, 
                                      skills: List[str], 
                                      industries: List[IndustryType] = None) -> List[SkillDemandDataPoint]:
        """Collect skill demand data for given skills"""
        if industries is None:
            industries = list(IndustryType)
        
        all_data = []
        
        for skill in skills:
            for industry in industries:
                try:
                    # Collect from multiple sources
                    for source_name, collector_func in self.sources.items():
                        try:
                            data_points = await collector_func(skill, industry)
                            all_data.extend(data_points)
                            await self._rate_limit()
                        except Exception as e:
                            logger.warning(f"Failed to collect from {source_name}: {e}")
                            continue
                except Exception as e:
                    logger.error(f"Failed to collect skill demand data for {skill}: {e}")
                    continue
        
        return all_data
    
    async def _collect_linkedin_jobs_data(self, skill: str, industry: IndustryType) -> List[SkillDemandDataPoint]:
        """Collect skill demand from LinkedIn jobs (simulated)"""
        await self._rate_limit()
        await asyncio.sleep(random.uniform(0.1, 0.2))
        
        # Simulate job count and demand based on skill
        skill_demand_data = {
            "Python": (85, 1200),
            "JavaScript": (90, 1500),
            "Machine Learning": (75, 800),
            "React": (88, 1000),
            "AWS": (82, 900),
            "Docker": (78, 700),
            "Kubernetes": (70, 600),
            "TensorFlow": (65, 500),
            "PyTorch": (60, 400),
            "SQL": (95, 2000),
            "Git": (92, 1800),
            "Agile": (80, 1000),
            "Leadership": (75, 1200),
            "Communication": (85, 1500)
        }
        
        demand_score, job_count = skill_demand_data.get(skill, (50, 300))
        
        # Add industry modifier
        industry_multiplier = self._get_industry_skill_multiplier(industry, skill)
        demand_score = min(100, demand_score * industry_multiplier)
        job_count = int(job_count * industry_multiplier)
        
        # Calculate growth rate (simulated)
        growth_rate = random.uniform(-5, 15)
        
        return [SkillDemandDataPoint(
            skill=skill,
            category=self._categorize_skill(skill),
            demand_score=demand_score,
            job_count=job_count,
            growth_rate=growth_rate,
            industry=industry,
            source="linkedin_jobs",
            timestamp=datetime.now()
        )]
    
    async def _collect_indeed_jobs_data(self, skill: str, industry: IndustryType) -> List[SkillDemandDataPoint]:
        """Collect skill demand from Indeed jobs (simulated)"""
        await self._rate_limit()
        await asyncio.sleep(random.uniform(0.1, 0.2))
        
        # Similar to LinkedIn but with variations
        skill_demand_data = {
            "Python": (80, 1000),
            "JavaScript": (85, 1300),
            "Machine Learning": (70, 700),
            "React": (82, 900),
            "AWS": (75, 800),
            "Docker": (72, 650),
            "Kubernetes": (65, 550),
            "TensorFlow": (60, 450),
            "PyTorch": (55, 350),
            "SQL": (90, 1800),
            "Git": (88, 1600),
            "Agile": (75, 900),
            "Leadership": (70, 1100),
            "Communication": (80, 1300)
        }
        
        demand_score, job_count = skill_demand_data.get(skill, (45, 250))
        industry_multiplier = self._get_industry_skill_multiplier(industry, skill)
        demand_score = min(100, demand_score * industry_multiplier)
        job_count = int(job_count * industry_multiplier)
        growth_rate = random.uniform(-3, 12)
        
        return [SkillDemandDataPoint(
            skill=skill,
            category=self._categorize_skill(skill),
            demand_score=demand_score,
            job_count=job_count,
            growth_rate=growth_rate,
            industry=industry,
            source="indeed_jobs",
            timestamp=datetime.now()
        )]
    
    async def _collect_github_trends_data(self, skill: str, industry: IndustryType) -> List[SkillDemandDataPoint]:
        """Collect skill demand from GitHub trends (simulated)"""
        await self._rate_limit()
        await asyncio.sleep(random.uniform(0.1, 0.2))
        
        # GitHub trends for technical skills
        if skill in ["Python", "JavaScript", "Machine Learning", "React", "AWS", "Docker", "Kubernetes", "TensorFlow", "PyTorch"]:
            skill_demand_data = {
                "Python": (90, 500),
                "JavaScript": (85, 600),
                "Machine Learning": (80, 300),
                "React": (82, 400),
                "AWS": (78, 350),
                "Docker": (75, 300),
                "Kubernetes": (70, 250),
                "TensorFlow": (65, 200),
                "PyTorch": (60, 150)
            }
            
            demand_score, job_count = skill_demand_data.get(skill, (50, 100))
            industry_multiplier = self._get_industry_skill_multiplier(industry, skill)
            demand_score = min(100, demand_score * industry_multiplier)
            job_count = int(job_count * industry_multiplier)
            growth_rate = random.uniform(5, 25)  # GitHub trends tend to be more volatile
            
            return [SkillDemandDataPoint(
                skill=skill,
                category=self._categorize_skill(skill),
                demand_score=demand_score,
                job_count=job_count,
                growth_rate=growth_rate,
                industry=industry,
                source="github_trends",
                timestamp=datetime.now()
            )]
        
        return []
    
    async def _collect_stack_overflow_data(self, skill: str, industry: IndustryType) -> List[SkillDemandDataPoint]:
        """Collect skill demand from Stack Overflow (simulated)"""
        await self._rate_limit()
        await asyncio.sleep(random.uniform(0.1, 0.2))
        
        # Stack Overflow data for technical skills
        if skill in ["Python", "JavaScript", "Machine Learning", "React", "AWS", "Docker", "Kubernetes", "TensorFlow", "PyTorch", "SQL", "Git"]:
            skill_demand_data = {
                "Python": (88, 800),
                "JavaScript": (85, 900),
                "Machine Learning": (75, 400),
                "React": (80, 500),
                "AWS": (78, 450),
                "Docker": (75, 400),
                "Kubernetes": (70, 350),
                "TensorFlow": (65, 300),
                "PyTorch": (60, 250),
                "SQL": (92, 1000),
                "Git": (90, 900)
            }
            
            demand_score, job_count = skill_demand_data.get(skill, (50, 200))
            industry_multiplier = self._get_industry_skill_multiplier(industry, skill)
            demand_score = min(100, demand_score * industry_multiplier)
            job_count = int(job_count * industry_multiplier)
            growth_rate = random.uniform(0, 20)
            
            return [SkillDemandDataPoint(
                skill=skill,
                category=self._categorize_skill(skill),
                demand_score=demand_score,
                job_count=job_count,
                growth_rate=growth_rate,
                industry=industry,
                source="stack_overflow",
                timestamp=datetime.now()
            )]
        
        return []
    
    def _categorize_skill(self, skill: str) -> SkillCategory:
        """Categorize a skill based on its name"""
        technical_skills = ["Python", "JavaScript", "Machine Learning", "React", "AWS", "Docker", "Kubernetes", "TensorFlow", "PyTorch", "SQL", "Git"]
        soft_skills = ["Leadership", "Communication", "Agile", "Teamwork", "Problem Solving"]
        emerging_skills = ["Machine Learning", "TensorFlow", "PyTorch", "Kubernetes", "Docker", "AWS"]
        
        if skill in technical_skills:
            if skill in emerging_skills:
                return SkillCategory.EMERGING
            return SkillCategory.TECHNICAL
        elif skill in soft_skills:
            return SkillCategory.SOFT
        else:
            return SkillCategory.DOMAIN_SPECIFIC
    
    def _get_industry_skill_multiplier(self, industry: IndustryType, skill: str) -> float:
        """Get skill demand multiplier based on industry and skill"""
        multipliers = {
            IndustryType.TECHNOLOGY: {
                "Python": 1.3, "JavaScript": 1.3, "Machine Learning": 1.4,
                "React": 1.3, "AWS": 1.2, "Docker": 1.2, "Kubernetes": 1.2,
                "TensorFlow": 1.4, "PyTorch": 1.4, "SQL": 1.1, "Git": 1.1
            },
            IndustryType.FINANCE: {
                "Python": 1.2, "Machine Learning": 1.3, "SQL": 1.4,
                "Leadership": 1.2, "Communication": 1.1
            },
            IndustryType.HEALTHCARE: {
                "Machine Learning": 1.3, "Python": 1.1, "Leadership": 1.1
            },
            IndustryType.EDUCATION: {
                "Communication": 1.3, "Leadership": 1.2, "Python": 1.0
            }
        }
        
        industry_multipliers = multipliers.get(industry, {})
        return industry_multipliers.get(skill, 1.0)


class JobMarketCollector(BaseDataCollector):
    """Collects job market data"""
    
    async def collect_job_market_data(self, 
                                    industries: List[IndustryType], 
                                    locations: List[str] = None) -> List[JobMarketTrend]:
        """Collect job market trends for given industries and locations"""
        if locations is None:
            locations = ["Global"]
        
        all_trends = []
        
        for industry in industries:
            for location in locations:
                try:
                    trend = await self._collect_industry_job_trends(industry, location)
                    all_trends.append(trend)
                    await self._rate_limit()
                except Exception as e:
                    logger.error(f"Failed to collect job market data for {industry} in {location}: {e}")
                    continue
        
        return all_trends
    
    async def _collect_industry_job_trends(self, industry: IndustryType, location: str) -> JobMarketTrend:
        """Collect job market trends for a specific industry and location"""
        await self._rate_limit()
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Simulate job market data
        industry_data = {
            IndustryType.TECHNOLOGY: {
                "total_jobs": 50000,
                "growth_rate": 8.5,
                "average_salary": 120000,
                "top_skills": ["Python", "JavaScript", "Machine Learning", "AWS", "React"],
                "emerging_roles": ["AI Engineer", "DevOps Engineer", "Data Engineer", "Cloud Architect"],
                "declining_roles": ["Legacy Developer", "Mainframe Developer"],
                "competitiveness": 75
            },
            IndustryType.FINANCE: {
                "total_jobs": 30000,
                "growth_rate": 5.2,
                "average_salary": 110000,
                "top_skills": ["SQL", "Python", "Leadership", "Communication", "Analytics"],
                "emerging_roles": ["FinTech Developer", "Quantitative Analyst", "Risk Analyst"],
                "declining_roles": ["Traditional Banker", "Manual Processor"],
                "competitiveness": 70
            },
            IndustryType.HEALTHCARE: {
                "total_jobs": 40000,
                "growth_rate": 12.3,
                "average_salary": 95000,
                "top_skills": ["Leadership", "Communication", "Machine Learning", "Python", "Analytics"],
                "emerging_roles": ["Health Data Analyst", "Telemedicine Specialist", "AI Healthcare Specialist"],
                "declining_roles": ["Manual Record Keeper", "Traditional Administrator"],
                "competitiveness": 60
            }
        }
        
        data = industry_data.get(industry, {
            "total_jobs": 20000,
            "growth_rate": 3.0,
            "average_salary": 80000,
            "top_skills": ["Communication", "Leadership", "Problem Solving"],
            "emerging_roles": [],
            "declining_roles": [],
            "competitiveness": 50
        })
        
        # Apply location modifier
        location_multiplier = self._get_location_job_multiplier(location)
        data["total_jobs"] = int(data["total_jobs"] * location_multiplier)
        data["average_salary"] = int(data["average_salary"] * location_multiplier)
        
        return JobMarketTrend(
            industry=industry,
            location=location,
            total_jobs=data["total_jobs"],
            growth_rate=data["growth_rate"],
            average_salary=data["average_salary"],
            top_skills=data["top_skills"],
            emerging_roles=data["emerging_roles"],
            declining_roles=data["declining_roles"],
            market_competitiveness=data["competitiveness"],
            timestamp=datetime.now()
        )
    
    def _get_location_job_multiplier(self, location: str) -> float:
        """Get job count multiplier based on location"""
        location_multipliers = {
            "San Francisco": 1.5,
            "New York": 1.4,
            "Seattle": 1.2,
            "Boston": 1.1,
            "Los Angeles": 1.0,
            "Chicago": 0.9,
            "Austin": 0.8,
            "Denver": 0.7,
            "Atlanta": 0.6,
            "Dallas": 0.5,
            "Global": 1.0
        }
        return location_multipliers.get(location, 1.0)


class IndustryInsightCollector(BaseDataCollector):
    """Collects industry-specific insights"""
    
    async def collect_industry_insights(self, industries: List[IndustryType]) -> List[IndustryInsight]:
        """Collect industry insights for given industries"""
        all_insights = []
        
        for industry in industries:
            try:
                insight = await self._collect_industry_insight(industry)
                all_insights.append(insight)
                await self._rate_limit()
            except Exception as e:
                logger.error(f"Failed to collect industry insight for {industry}: {e}")
                continue
        
        return all_insights
    
    async def _collect_industry_insight(self, industry: IndustryType) -> IndustryInsight:
        """Collect insight for a specific industry"""
        await self._rate_limit()
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Simulate industry insight data
        industry_insights = {
            IndustryType.TECHNOLOGY: {
                "market_size": 5000000000000,  # 5 trillion
                "growth_rate": 8.5,
                "key_trends": [
                    "AI and Machine Learning adoption",
                    "Cloud migration acceleration",
                    "Remote work technology",
                    "Cybersecurity focus",
                    "Edge computing growth"
                ],
                "top_skills": ["Python", "Machine Learning", "AWS", "Docker", "React"],
                "salary_ranges": {
                    "entry": {"min": 60000, "max": 90000},
                    "mid": {"min": 90000, "max": 150000},
                    "senior": {"min": 150000, "max": 250000}
                },
                "job_availability": 85,
                "competition_level": 75,
                "future_outlook": "very_positive"
            },
            IndustryType.FINANCE: {
                "market_size": 3000000000000,  # 3 trillion
                "growth_rate": 5.2,
                "key_trends": [
                    "FinTech disruption",
                    "Digital banking transformation",
                    "Regulatory technology (RegTech)",
                    "Cryptocurrency integration",
                    "AI in trading and risk management"
                ],
                "top_skills": ["SQL", "Python", "Leadership", "Communication", "Analytics"],
                "salary_ranges": {
                    "entry": {"min": 50000, "max": 80000},
                    "mid": {"min": 80000, "max": 130000},
                    "senior": {"min": 130000, "max": 200000}
                },
                "job_availability": 70,
                "competition_level": 70,
                "future_outlook": "positive"
            },
            IndustryType.HEALTHCARE: {
                "market_size": 4000000000000,  # 4 trillion
                "growth_rate": 12.3,
                "key_trends": [
                    "Telemedicine expansion",
                    "AI in diagnostics",
                    "Personalized medicine",
                    "Digital health records",
                    "Preventive care focus"
                ],
                "top_skills": ["Leadership", "Communication", "Machine Learning", "Python", "Analytics"],
                "salary_ranges": {
                    "entry": {"min": 45000, "max": 70000},
                    "mid": {"min": 70000, "max": 120000},
                    "senior": {"min": 120000, "max": 180000}
                },
                "job_availability": 90,
                "competition_level": 60,
                "future_outlook": "very_positive"
            }
        }
        
        data = industry_insights.get(industry, {
            "market_size": 1000000000000,  # 1 trillion
            "growth_rate": 3.0,
            "key_trends": ["Digital transformation", "Automation", "Sustainability"],
            "top_skills": ["Communication", "Leadership", "Problem Solving"],
            "salary_ranges": {
                "entry": {"min": 40000, "max": 60000},
                "mid": {"min": 60000, "max": 100000},
                "senior": {"min": 100000, "max": 150000}
            },
            "job_availability": 60,
            "competition_level": 50,
            "future_outlook": "neutral"
        })
        
        return IndustryInsight(
            industry=industry,
            market_size=data["market_size"],
            growth_rate=data["growth_rate"],
            key_trends=data["key_trends"],
            top_skills=data["top_skills"],
            salary_ranges=data["salary_ranges"],
            job_availability=data["job_availability"],
            competition_level=data["competition_level"],
            future_outlook=data["future_outlook"],
            timestamp=datetime.now()
        )
