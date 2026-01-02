"""
API endpoints for Market Intelligence module
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from flask import Blueprint, request, jsonify, current_app
from pydantic import BaseModel, Field, validator
import json
import numpy as np
import threading
from functools import lru_cache
import weakref
import time
import hashlib

from .data_collectors import (
    SalaryDataCollector, SkillDemandCollector, 
    JobMarketCollector, IndustryInsightCollector
)
from .analyzers import (
    SalaryTrendAnalyzer, SkillDemandAnalyzer, 
    MarketTrendAnalyzer, IndustryAnalyzer
)
from .models import (
    MarketIntelligenceData, IndustryType, SkillCategory,
    TrendDirection, MarketForecast
)
from .skills_forecasting import SkillsForecaster
from .talent_competition import (
    analyze_talent_availability,
    competitive_intelligence,
)
from .economic_intelligence import (
    fetch_economic_indicators,
    market_timing_intelligence,
)
from .behavior_prediction import predict_candidate_behavior
from .market_alerts import (
    process_market_signals,
    get_engine_status,
    get_alerts_summary,
)
from .integration_pipeline import (
    run_market_intelligence_pipeline,
    enhance_candidate_recommendations,
)
from .hybrid_llm_service import hybrid_llm_service, market_intelligence_llm
from .api_config import APIConfig

logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Create Blueprint
market_intel_bp = Blueprint("market_intelligence", __name__, url_prefix="/api/market-intelligence")

# Pydantic models for request validation
class MarketIntelligenceRequest(BaseModel):
    """Request model for market intelligence data collection"""
    positions: List[str] = Field(..., description="List of job positions to analyze")
    skills: List[str] = Field(..., description="List of skills to analyze")
    industries: List[str] = Field(default=["technology", "finance", "healthcare"], 
                                description="List of industries to analyze")
    locations: List[str] = Field(default=["Global"], 
                               description="List of locations to analyze")
    include_forecasts: bool = Field(default=True, description="Include market forecasts")
    analysis_depth: str = Field(default="standard", 
                               description="Analysis depth: basic, standard, comprehensive")

    @validator('industries')
    def validate_industries(cls, v):
        valid_industries = [industry.value for industry in IndustryType]
        for industry in v:
            if industry.lower() not in [i.lower() for i in valid_industries]:
                raise ValueError(f"Invalid industry: {industry}")
        return v

    @validator('analysis_depth')
    def validate_analysis_depth(cls, v):
        valid_depths = ["basic", "standard", "comprehensive"]
        if v not in valid_depths:
            raise ValueError(f"Analysis depth must be one of: {valid_depths}")
        return v


class SalaryAnalysisRequest(BaseModel):
    """Request model for salary analysis"""
    positions: List[str] = Field(..., description="List of job positions to analyze")
    locations: List[str] = Field(default=["Global"], description="List of locations to analyze")
    industries: List[str] = Field(default=["technology"], description="List of industries to analyze")
    period_months: int = Field(default=12, ge=1, le=24, description="Analysis period in months")


class SkillAnalysisRequest(BaseModel):
    """Request model for skill analysis"""
    skills: List[str] = Field(..., description="List of skills to analyze")
    industries: List[str] = Field(default=["technology"], description="List of industries to analyze")
    include_related: bool = Field(default=True, description="Include related skills analysis")


class MarketTrendsRequest(BaseModel):
    """Request model for market trends analysis"""
    industries: List[str] = Field(default=["technology", "finance", "healthcare"], 
                                description="List of industries to analyze")
    locations: List[str] = Field(default=["Global"], description="List of locations to analyze")
    timeframe_months: int = Field(default=12, ge=1, le=24, description="Analysis timeframe in months")


class MarketIntelligenceAPI:
    """Main API class for Market Intelligence - OPTIMIZED VERSION"""
    
    def __init__(self):
        # Lazy initialization for better performance
        self._initialized = False
        self._lock = threading.RLock()
        
        # Core components (lazy loaded)
        self._salary_collector = None
        self._skill_collector = None
        self._job_market_collector = None
        self._industry_collector = None
        self._salary_analyzer = None
        self._skill_analyzer = None
        self._market_analyzer = None
        self._industry_analyzer = None
        self._skills_forecaster = None
        
        # Optimized caching with TTL and memory management
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.cache_timestamps = {}
        self.max_cache_size = 1000
        
        # Performance tracking
        self.request_count = 0
        self.avg_response_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Background task management
        self._background_tasks = set()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        
        logger.info("🚀 MarketIntelligenceAPI initialized with optimizations")
    
    def _ensure_initialized(self):
        """Lazy initialization of components"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    try:
                        self._salary_collector = SalaryDataCollector()
                        self._skill_collector = SkillDemandCollector()
                        self._job_market_collector = JobMarketCollector()
                        self._industry_collector = IndustryInsightCollector()
                        
                        self._salary_analyzer = SalaryTrendAnalyzer()
                        self._skill_analyzer = SkillDemandAnalyzer()
                        self._market_analyzer = MarketTrendAnalyzer()
                        self._industry_analyzer = IndustryAnalyzer()
                        self._skills_forecaster = SkillsForecaster()
                        
                        self._initialized = True
                        logger.info("✅ MarketIntelligenceAPI components initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize components: {e}")
                        raise
    
    @property
    def salary_collector(self):
        self._ensure_initialized()
        return self._salary_collector
    
    @property
    def skill_collector(self):
        self._ensure_initialized()
        return self._skill_collector
    
    @property
    def job_market_collector(self):
        self._ensure_initialized()
        return self._job_market_collector
    
    @property
    def industry_collector(self):
        self._ensure_initialized()
        return self._industry_collector
    
    @property
    def salary_analyzer(self):
        self._ensure_initialized()
        return self._salary_analyzer
    
    @property
    def skill_analyzer(self):
        self._ensure_initialized()
        return self._skill_analyzer
    
    @property
    def market_analyzer(self):
        self._ensure_initialized()
        return self._market_analyzer
    
    @property
    def industry_analyzer(self):
        self._ensure_initialized()
        return self._industry_analyzer
    
    @property
    def skills_forecaster(self):
        self._ensure_initialized()
        return self._skills_forecaster
    
    def _get_cache_key(self, request_data: MarketIntelligenceRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'positions': sorted(request_data.positions),
            'skills': sorted(request_data.skills),
            'industries': sorted(request_data.industries),
            'locations': sorted(request_data.locations),
            'include_forecasts': request_data.include_forecasts,
            'analysis_depth': request_data.analysis_depth
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        expired_keys = []
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        
        # If cache is still too large, remove oldest entries
        if len(self.cache) > self.max_cache_size:
            sorted_items = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_items[:len(self.cache) - self.max_cache_size]]
            for key in keys_to_remove:
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
        
        self._last_cleanup = current_time
        logger.debug(f"Cache cleanup completed. Removed {len(expired_keys)} expired entries")

    async def get_comprehensive_analysis(self, request_data: MarketIntelligenceRequest) -> Dict[str, Any]:
        """Get comprehensive market intelligence analysis - OPTIMIZED VERSION"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(request_data)
            if cache_key in self.cache:
                self.cache_hits += 1
                logger.info("🎯 Cache hit for comprehensive analysis")
                return self.cache[cache_key]
            
            self.cache_misses += 1
            
            # Convert string industries to IndustryType enums
            industries = [IndustryType(industry.lower()) for industry in request_data.industries]
            
            # Clean up cache periodically
            self._cleanup_cache()
            
            # Collect data from all sources in parallel for better performance
            logger.info("Starting comprehensive market intelligence data collection...")
            
            # Use asyncio.gather for parallel execution
            salary_task = self.salary_collector.collect_salary_data(
                positions=request_data.positions,
                locations=request_data.locations,
                industries=industries
            )
            
            skill_task = self.skill_collector.collect_skill_demand_data(
                skills=request_data.skills,
                industries=industries
            )
            
            job_market_task = self.job_market_collector.collect_job_market_data(
                industries=industries,
                locations=request_data.locations
            )
            
            industry_task = self.industry_collector.collect_industry_insights(
                industries=industries
            )
            
            # Execute all data collection tasks in parallel
            salary_data, skill_data, job_market_data, industry_insights = await asyncio.gather(
                salary_task, skill_task, job_market_task, industry_task,
                return_exceptions=True
            )
            
            # Handle any exceptions from parallel tasks
            if isinstance(salary_data, Exception):
                logger.error(f"Salary data collection failed: {salary_data}")
                salary_data = []
            if isinstance(skill_data, Exception):
                logger.error(f"Skill data collection failed: {skill_data}")
                skill_data = []
            if isinstance(job_market_data, Exception):
                logger.error(f"Job market data collection failed: {job_market_data}")
                job_market_data = []
            if isinstance(industry_insights, Exception):
                logger.error(f"Industry insights collection failed: {industry_insights}")
                industry_insights = []
            
            # Analyze collected data in parallel
            logger.info("Analyzing collected data...")
            
            # Run analysis tasks in parallel
            analysis_tasks = [
                asyncio.create_task(self._analyze_salary_trends(salary_data)),
                asyncio.create_task(self._analyze_skill_demands(skill_data)),
                asyncio.create_task(self._analyze_job_market(job_market_data)),
                asyncio.create_task(self._analyze_industry_insights(industry_insights))
            ]
            
            salary_trends, skill_demands, job_market_analysis, industry_analysis = await asyncio.gather(
                *analysis_tasks, return_exceptions=True
            )
            
            # Handle analysis exceptions
            if isinstance(salary_trends, Exception):
                logger.error(f"Salary analysis failed: {salary_trends}")
                salary_trends = []
            if isinstance(skill_demands, Exception):
                logger.error(f"Skill analysis failed: {skill_demands}")
                skill_demands = []
            if isinstance(job_market_analysis, Exception):
                logger.error(f"Job market analysis failed: {job_market_analysis}")
                job_market_analysis = {}
            if isinstance(industry_analysis, Exception):
                logger.error(f"Industry analysis failed: {industry_analysis}")
                industry_analysis = {}
            
            # Generate market trends
            market_trends = self.market_analyzer.analyze_market_trends(
                salary_trends, skill_demands, job_market_data
            )
            
            # Generate forecasts if requested
            forecasts = []
            if request_data.include_forecasts:
                forecasts = self._generate_forecasts(salary_trends, skill_demands, job_market_data)
            
            # Create comprehensive response
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "analysis_depth": request_data.analysis_depth,
                "data": {
                    "salary_trends": [st.to_dict() for st in salary_trends],
                    "skill_demands": [sd.to_dict() for sd in skill_demands],
                    "job_market_trends": [jmt.to_dict() for jmt in job_market_data],
                    "industry_insights": [ii.to_dict() for ii in industry_insights],
                    "market_analysis": market_trends,
                    "industry_analysis": industry_analysis,
                    "forecasts": [f.to_dict() for f in forecasts]
                },
                "summary": {
                    "total_positions_analyzed": len(request_data.positions),
                    "total_skills_analyzed": len(request_data.skills),
                    "total_industries_analyzed": len(industries),
                    "total_locations_analyzed": len(request_data.locations),
                    "data_points_collected": len(salary_data) + len(skill_data),
                    "analysis_confidence": self._calculate_analysis_confidence(salary_trends, skill_demands)
                },
                "performance": {
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "cache_hit": False,
                    "parallel_processing": True
                }
            }
            
            # Cache the response
            self.cache[cache_key] = response
            self.cache_timestamps[cache_key] = time.time()
            
            # Update performance metrics
            response_time = time.time() - start_time
            self.avg_response_time = (
                (self.avg_response_time * (self.request_count - 1) + response_time) / self.request_count
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_salary_analysis(self, request_data: SalaryAnalysisRequest) -> Dict[str, Any]:
        """Get salary trend analysis"""
        try:
            industries = [IndustryType(industry.lower()) for industry in request_data.industries]
            
            # Collect salary data
            salary_data = await self.salary_collector.collect_salary_data(
                positions=request_data.positions,
                locations=request_data.locations,
                industries=industries
            )
            
            # Analyze salary trends
            salary_trends = self.salary_analyzer.analyze_salary_trends(
                salary_data, request_data.period_months
            )
            
            # Calculate summary statistics
            summary = self._calculate_salary_summary(salary_trends)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "salary_trends": [st.to_dict() for st in salary_trends],
                    "summary": summary
                }
            }
            
        except Exception as e:
            logger.error(f"Error in salary analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_skill_analysis(self, request_data: SkillAnalysisRequest) -> Dict[str, Any]:
        """Get skill demand analysis"""
        try:
            industries = [IndustryType(industry.lower()) for industry in request_data.industries]
            
            # Collect skill data
            skill_data = await self.skill_collector.collect_skill_demand_data(
                skills=request_data.skills,
                industries=industries
            )
            
            # Analyze skill demands
            skill_demands = self.skill_analyzer.analyze_skill_demands(skill_data)
            
            # Calculate summary statistics
            summary = self._calculate_skill_summary(skill_demands)
            
            # Add related skills analysis if requested
            related_analysis = {}
            if request_data.include_related:
                related_analysis = self._analyze_related_skills(skill_demands)

            # NEW: Forecast skills demand using composite multi-source history
            forecasts = await self.skills_forecaster.forecast_skills(
                skills=request_data.skills,
                industries=industries if industries else None,
                months_history=12
            )
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "skill_demands": [sd.to_dict() for sd in skill_demands],
                    "summary": summary,
                    "related_analysis": related_analysis,
                    "forecasts": [f.to_dict() for f in forecasts]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in skill analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_market_trends(self, request_data: MarketTrendsRequest) -> Dict[str, Any]:
        """Get overall market trends analysis"""
        try:
            industries = [IndustryType(industry.lower()) for industry in request_data.industries]
            
            # Collect all relevant data
            salary_data = await self.salary_collector.collect_salary_data(
                positions=["Software Engineer", "Data Scientist", "Product Manager"],  # Common positions
                locations=request_data.locations,
                industries=industries
            )
            
            skill_data = await self.skill_collector.collect_skill_demand_data(
                skills=["Python", "JavaScript", "Machine Learning", "AWS", "Leadership"],
                industries=industries
            )
            
            job_market_data = await self.job_market_collector.collect_job_market_data(
                industries=industries,
                locations=request_data.locations
            )
            
            # Analyze trends
            salary_trends = self.salary_analyzer.analyze_salary_trends(salary_data)
            skill_demands = self.skill_analyzer.analyze_skill_demands(skill_data)
            market_trends = self.market_analyzer.analyze_market_trends(
                salary_trends, skill_demands, job_market_data
            )
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "market_trends": market_trends,
                    "salary_trends": [st.to_dict() for st in salary_trends],
                    "skill_demands": [sd.to_dict() for sd in skill_demands],
                    "job_market_trends": [jmt.to_dict() for jmt in job_market_data]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in market trends analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_forecasts(self, salary_trends, skill_demands, job_market_data) -> List[MarketForecast]:
        """Generate market forecasts"""
        forecasts = []
        
        # Salary forecasts
        for trend in salary_trends:
            if trend.forecast:
                forecast = MarketForecast(
                    forecast_type="salary",
                    target=trend.position,
                    timeframe_months=12,
                    predictions=trend.forecast,
                    factors=["market_demand", "industry_growth", "skill_shortage"],
                    accuracy_score=trend.confidence_score
                )
                forecasts.append(forecast)
        
        # Skill demand forecasts
        for skill_demand in skill_demands:
            if skill_demand.forecast:
                forecast = MarketForecast(
                    forecast_type="skill_demand",
                    target=skill_demand.skill,
                    timeframe_months=12,
                    predictions=skill_demand.forecast,
                    factors=["technology_trends", "industry_adoption", "job_market_demand"],
                    accuracy_score=skill_demand.trend_strength
                )
                forecasts.append(forecast)
        
        return forecasts
    
    async def _analyze_salary_trends(self, salary_data):
        """Async wrapper for salary trend analysis"""
        return self.salary_analyzer.analyze_salary_trends(salary_data)
    
    async def _analyze_skill_demands(self, skill_data):
        """Async wrapper for skill demand analysis"""
        return self.skill_analyzer.analyze_skill_demands(skill_data)
    
    async def _analyze_job_market(self, job_market_data):
        """Async wrapper for job market analysis"""
        return {"job_market_data": job_market_data}
    
    async def _analyze_industry_insights(self, industry_insights):
        """Async wrapper for industry analysis"""
        return self.industry_analyzer.analyze_industry_trends(industry_insights, [], [])
    
    def _calculate_analysis_confidence(self, salary_trends, skill_demands) -> float:
        """Calculate overall analysis confidence score"""
        if not salary_trends and not skill_demands:
            return 0.0
        
        confidence_scores = []
        
        for trend in salary_trends:
            confidence_scores.append(trend.confidence_score)
        
        for skill_demand in skill_demands:
            confidence_scores.append(skill_demand.trend_strength)
        
        return float(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.0
    
    def _calculate_salary_summary(self, salary_trends) -> Dict[str, Any]:
        """Calculate salary analysis summary"""
        if not salary_trends:
            return {"average_salary": 0, "trending_positions": [], "salary_growth": 0}
        
        salaries = [st.current_median for st in salary_trends]
        trending_positions = [st.position for st in salary_trends 
                            if st.trend_direction == TrendDirection.RISING]
        
        # Calculate average salary growth
        salary_growth = 0
        growth_count = 0
        for trend in salary_trends:
            if trend.forecast and "12_months" in trend.forecast:
                current = trend.current_median
                future = trend.forecast["12_months"]
                if current > 0:
                    growth = ((future - current) / current) * 100
                    salary_growth += growth
                    growth_count += 1
        
        avg_salary_growth = salary_growth / growth_count if growth_count > 0 else 0
        
        return {
            "average_salary": float(sum(salaries) / len(salaries)),
            "trending_positions": trending_positions[:5],
            "salary_growth": float(avg_salary_growth),
            "total_positions_analyzed": len(salary_trends)
        }
    
    def _calculate_skill_summary(self, skill_demands) -> Dict[str, Any]:
        """Calculate skill analysis summary"""
        if not skill_demands:
            return {"hot_skills": [], "declining_skills": [], "skill_diversity": 0}
        
        hot_skills = [sd.skill for sd in skill_demands 
                     if sd.current_demand > 75 and sd.trend_direction == TrendDirection.RISING]
        
        declining_skills = [sd.skill for sd in skill_demands 
                           if sd.trend_direction == TrendDirection.FALLING]
        
        categories = set(sd.category for sd in skill_demands)
        
        return {
            "hot_skills": hot_skills[:10],
            "declining_skills": declining_skills[:10],
            "skill_diversity": len(categories),
            "total_skills_analyzed": len(skill_demands)
        }
    
    def _analyze_related_skills(self, skill_demands) -> Dict[str, Any]:
        """Analyze related skills patterns"""
        skill_categories = {}
        skill_industries = {}
        
        for skill_demand in skill_demands:
            # Group by category
            category = skill_demand.category.value
            if category not in skill_categories:
                skill_categories[category] = []
            skill_categories[category].append({
                "skill": skill_demand.skill,
                "demand": skill_demand.current_demand
            })
            
            # Group by industry
            for industry, demand in skill_demand.industry_breakdown.items():
                if industry not in skill_industries:
                    skill_industries[industry] = []
                skill_industries[industry].append({
                    "skill": skill_demand.skill,
                    "demand": demand
                })
        
        return {
            "by_category": skill_categories,
            "by_industry": skill_industries
        }


# Initialize API instance
market_intel_api = MarketIntelligenceAPI()

# Global event loop for better performance
_global_loop = None
_loop_lock = threading.Lock()

def get_or_create_loop():
    """Get or create a global event loop for better performance"""
    global _global_loop
    with _loop_lock:
        if _global_loop is None or _global_loop.is_closed():
            _global_loop = asyncio.new_event_loop()
            # Start the loop in a background thread
            loop_thread = threading.Thread(target=_global_loop.run_forever, daemon=True)
            loop_thread.start()
        return _global_loop

def run_async_in_loop(coro):
    """Run async coroutine in the global loop"""
    loop = get_or_create_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)  # 30 second timeout

# API Endpoints
@market_intel_bp.route("/comprehensive", methods=["POST"])
def comprehensive_analysis():
    """Get comprehensive market intelligence analysis - OPTIMIZED"""
    try:
        request_data = MarketIntelligenceRequest(**request.get_json())
        
        # Use optimized async execution
        result = run_async_in_loop(
            market_intel_api.get_comprehensive_analysis(request_data)
        )
        
        return jsonify(result, cls=NumpyEncoder)
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/salary-analysis", methods=["POST"])
def salary_analysis():
    """Get salary trend analysis - OPTIMIZED"""
    try:
        request_data = SalaryAnalysisRequest(**request.get_json())
        
        # Use optimized async execution
        result = run_async_in_loop(
            market_intel_api.get_salary_analysis(request_data)
        )
        
        return jsonify(result, cls=NumpyEncoder)
        
    except Exception as e:
        logger.error(f"Error in salary analysis endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/skill-analysis", methods=["POST"])
def skill_analysis():
    """Get skill demand analysis - OPTIMIZED"""
    try:
        request_data = SkillAnalysisRequest(**request.get_json())
        
        # Use optimized async execution
        result = run_async_in_loop(
            market_intel_api.get_skill_analysis(request_data)
        )
        
        return jsonify(result, cls=NumpyEncoder)
        
    except Exception as e:
        logger.error(f"Error in skill analysis endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


class SkillForecastRequest(BaseModel):
    """Request model for standalone skill forecasting"""
    skills: List[str]
    industries: List[str] = ["technology"]
    months_history: int = 12


@market_intel_bp.route("/skill-forecast", methods=["POST"])
def skill_forecast():
    """Get standalone skill demand forecasts"""
    try:
        req = SkillForecastRequest(**request.get_json())
        industries = [IndustryType(ind.lower()) for ind in req.industries]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        forecasts = loop.run_until_complete(
            market_intel_api.skills_forecaster.forecast_skills(
                skills=req.skills,
                industries=industries,
                months_history=req.months_history
            )
        )
        loop.close()
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "forecasts": [f.to_dict() for f in forecasts]
            }
        })
    except Exception as e:
        logger.error(f"Error in skill forecast endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/market-trends", methods=["POST"])
def market_trends():
    """Get overall market trends analysis"""
    try:
        request_data = MarketTrendsRequest(**request.get_json())
        
        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            market_intel_api.get_market_trends(request_data)
        )
        loop.close()
        
        return jsonify(result, cls=NumpyEncoder)
        
    except Exception as e:
        logger.error(f"Error in market trends endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for market intelligence service"""
    return jsonify({
        "status": "healthy",
        "service": "market_intelligence",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


@market_intel_bp.route("/available-industries", methods=["GET"])
def available_industries():
    """Get list of available industries"""
    return jsonify({
        "industries": [industry.value for industry in IndustryType],
        "descriptions": {
            industry.value: f"{industry.value.title()} industry"
            for industry in IndustryType
        }
    })


@market_intel_bp.route("/available-skills", methods=["GET"])
def available_skills():
    """Get list of available skill categories"""
    return jsonify({
        "categories": [category.value for category in SkillCategory],
        "descriptions": {
            "technical": "Technical and programming skills",
            "soft": "Soft skills and interpersonal abilities",
            "domain_specific": "Industry-specific knowledge and skills",
            "emerging": "New and trending skills",
            "legacy": "Traditional skills that may be declining"
        }
    })


@market_intel_bp.route("/dashboard", methods=["GET"])
def dashboard():
    """Serve the market intelligence dashboard"""
    from flask import render_template
    return render_template("market_intelligence_dashboard.html")


@market_intel_bp.route("/talent-supply", methods=["GET"])
def talent_supply():
    """Talent pool mapping summary"""
    try:
        data = analyze_talent_availability()
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in talent supply endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/competitive-intel", methods=["GET"])
def competitive_intel():
    """Competitive landscape intelligence"""
    try:
        data = competitive_intelligence()
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in competitive intelligence endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/economic-indicators", methods=["GET"])
def economic_indicators():
    """Get macro-economic indicators"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(fetch_economic_indicators())
        loop.close()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in economic indicators endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/market-timing", methods=["GET"])
def market_timing():
    """Get AI-powered market timing intelligence"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(market_timing_intelligence())
        loop.close()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in market timing endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/behavior-prediction", methods=["POST"])
def behavior_prediction():
    """Get candidate behavior predictions with hybrid LLM enhancement"""
    try:
        request_data = request.get_json() or {}
        profiles = request_data.get("profiles", [])
        market_data = request_data.get("market_data", {})
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(predict_candidate_behavior(profiles, market_data))
        loop.close()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in behavior prediction endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/market-alerts", methods=["GET"])
def market_alerts():
    """Process market signals and get alerts"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(process_market_signals())
        loop.close()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in market alerts endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/alerts-summary", methods=["GET"])
def alerts_summary():
    """Get alerts summary"""
    try:
        data = get_alerts_summary()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in alerts summary endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/engine-status", methods=["GET"])
def engine_status():
    """Get market intelligence engine status"""
    try:
        data = get_engine_status()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in engine status endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/pipeline", methods=["GET"])
def market_pipeline():
    """Run complete market intelligence pipeline"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(run_market_intelligence_pipeline())
        loop.close()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in market pipeline endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/enhance-candidates", methods=["POST"])
def enhance_candidates():
    """Enhance candidate recommendations with market intelligence"""
    try:
        request_data = request.get_json() or {}
        candidate_pool = request_data.get("candidate_pool", [])
        job_specifications = request_data.get("job_specifications", {})
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(enhance_candidate_recommendations(candidate_pool, job_specifications))
        loop.close()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in enhance candidates endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/llm-usage-stats", methods=["GET"])
def llm_usage_stats():
    """Get hybrid LLM usage statistics and cost optimization report"""
    try:
        stats = hybrid_llm_service.get_usage_stats()
        cost_report = market_intelligence_llm.get_cost_optimization_report()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "usage_stats": stats,
                "cost_optimization": cost_report
            }
        })
    except Exception as e:
        logger.error(f"Error in LLM usage stats endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/screen-candidates", methods=["POST"])
def screen_candidates():
    """Screen candidates using GPT-4o-mini for cost efficiency"""
    try:
        request_data = request.get_json() or {}
        candidates = request_data.get("candidates", [])
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        screened_candidates = loop.run_until_complete(market_intelligence_llm.screen_candidates(candidates))
        loop.close()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "screened_candidates": screened_candidates,
                "total_candidates": len(candidates),
                "screening_completed": True
            }
        })
    except Exception as e:
        logger.error(f"Error in screen candidates endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@market_intel_bp.route("/api-status", methods=["GET"])
def api_status():
    """Get API configuration status and recommendations"""
    try:
        llm_config = APIConfig.get_llm_config()
        data_config = APIConfig.get_data_apis_config()
        validation = APIConfig.validate_required_apis()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "llm_apis": llm_config,
                "data_apis": data_config,
                "validation": validation,
                "setup_instructions": {
                    "llm_apis": "Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables",
                    "data_apis": "Configure external data APIs for real-world market intelligence",
                    "env_template": "See api_config.py for complete environment variable template"
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in API status endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400
