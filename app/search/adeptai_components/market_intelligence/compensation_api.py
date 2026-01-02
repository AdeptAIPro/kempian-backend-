"""
API endpoints for comprehensive Salary & Compensation Intelligence
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from flask import Blueprint, request, jsonify, current_app
from pydantic import BaseModel, Field, validator

from .salary_intelligence import (
    RealTimeSalaryCollector, CompensationBenchmarker,
    CompensationDataPoint, CompensationBenchmark, CompensationType, DataSource
)
from .models import IndustryType, SkillCategory

logger = logging.getLogger(__name__)

# Create Blueprint
compensation_bp = Blueprint("compensation_intelligence", __name__, url_prefix="/api/compensation")

# Pydantic models for request validation
class CompensationAnalysisRequest(BaseModel):
    """Request model for comprehensive compensation analysis"""
    positions: List[str] = Field(..., description="List of job positions to analyze")
    locations: List[str] = Field(default=["Global"], description="List of locations to analyze")
    industries: List[str] = Field(default=["technology"], description="List of industries to analyze")
    include_equity: bool = Field(default=True, description="Include equity and stock option data")
    include_benefits: bool = Field(default=True, description="Include benefits analysis")
    include_benchmarks: bool = Field(default=True, description="Include compensation benchmarks")
    data_sources: List[str] = Field(default=["glassdoor", "payscale", "levels_fyi"], 
                                  description="Data sources to use")

    @validator('industries')
    def validate_industries(cls, v):
        valid_industries = [industry.value for industry in IndustryType]
        for industry in v:
            if industry.lower() not in [i.lower() for i in valid_industries]:
                raise ValueError(f"Invalid industry: {industry}")
        return v

    @validator('data_sources')
    def validate_data_sources(cls, v):
        valid_sources = [source.value for source in DataSource]
        for source in v:
            if source.lower() not in [s.lower() for s in valid_sources]:
                raise ValueError(f"Invalid data source: {source}")
        return v


class SalaryBenchmarkRequest(BaseModel):
    """Request model for salary benchmarking"""
    position: str = Field(..., description="Job position to benchmark")
    location: str = Field(default="Global", description="Location for benchmarking")
    industry: str = Field(default="technology", description="Industry for benchmarking")
    experience_level: str = Field(default="Mid-level", description="Experience level")
    company_size: str = Field(default="Any", description="Company size filter")


class EquityAnalysisRequest(BaseModel):
    """Request model for equity analysis"""
    positions: List[str] = Field(..., description="List of positions to analyze")
    industries: List[str] = Field(default=["technology"], description="List of industries")
    include_stock_options: bool = Field(default=True, description="Include stock options analysis")
    include_rsu: bool = Field(default=True, description="Include RSU analysis")


class CompensationIntelligenceAPI:
    """Main API class for Compensation Intelligence"""
    
    def __init__(self):
        self.salary_collector = RealTimeSalaryCollector()
        self.benchmarker = CompensationBenchmarker()
    
    async def get_comprehensive_compensation_analysis(self, request_data: CompensationAnalysisRequest) -> Dict[str, Any]:
        """Get comprehensive compensation analysis"""
        try:
            # Convert string industries to IndustryType enums
            industries = [IndustryType(industry.lower()) for industry in request_data.industries]
            
            # Collect comprehensive compensation data
            logger.info("Starting comprehensive compensation data collection...")
            
            compensation_data = await self.salary_collector.collect_comprehensive_compensation_data(
                positions=request_data.positions,
                locations=request_data.locations,
                industries=industries
            )
            
            # Create benchmarks if requested
            benchmarks = []
            if request_data.include_benchmarks:
                benchmarks = self.benchmarker.create_benchmarks(compensation_data)
            
            # Analyze compensation data
            analysis = self._analyze_compensation_data(compensation_data, benchmarks)
            
            # Generate insights and recommendations
            insights = self._generate_compensation_insights(compensation_data, benchmarks)
            recommendations = self._generate_compensation_recommendations(compensation_data, benchmarks)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "compensation_data": [dp.to_dict() for dp in compensation_data],
                    "benchmarks": [bm.to_dict() for bm in benchmarks],
                    "analysis": analysis,
                    "insights": insights,
                    "recommendations": recommendations
                },
                "summary": {
                    "total_positions_analyzed": len(request_data.positions),
                    "total_data_points": len(compensation_data),
                    "total_benchmarks": len(benchmarks),
                    "data_sources_used": list(set(dp.source for dp in compensation_data)),
                    "average_confidence": sum(dp.confidence_score for dp in compensation_data) / len(compensation_data) if compensation_data else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive compensation analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_salary_benchmarks(self, request_data: SalaryBenchmarkRequest) -> Dict[str, Any]:
        """Get detailed salary benchmarks for a specific position"""
        try:
            industry = IndustryType(request_data.industry.lower())
            
            # Collect data for the specific position
            compensation_data = await self.salary_collector.collect_comprehensive_compensation_data(
                positions=[request_data.position],
                locations=[request_data.location],
                industries=[industry]
            )
            
            # Create benchmarks
            benchmarks = self.benchmarker.create_benchmarks(compensation_data)
            
            # Filter by experience level and company size if specified
            filtered_data = self._filter_by_criteria(compensation_data, request_data)
            
            # Calculate detailed statistics
            statistics = self._calculate_detailed_statistics(filtered_data)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "position": request_data.position,
                    "location": request_data.location,
                    "industry": request_data.industry,
                    "benchmarks": [bm.to_dict() for bm in benchmarks],
                    "statistics": statistics,
                    "sample_size": len(filtered_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in salary benchmarking: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_equity_analysis(self, request_data: EquityAnalysisRequest) -> Dict[str, Any]:
        """Get detailed equity and stock option analysis"""
        try:
            industries = [IndustryType(industry.lower()) for industry in request_data.industries]
            
            # Collect compensation data with focus on equity
            compensation_data = await self.salary_collector.collect_comprehensive_compensation_data(
                positions=request_data.positions,
                locations=["Global"],  # Equity data is often location-agnostic
                industries=industries
            )
            
            # Filter for equity-heavy roles
            equity_data = [dp for dp in compensation_data if dp.equity_value > 0]
            
            # Analyze equity patterns
            equity_analysis = self._analyze_equity_patterns(equity_data)
            
            # Calculate equity benchmarks
            equity_benchmarks = self._calculate_equity_benchmarks(equity_data)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "equity_analysis": equity_analysis,
                    "equity_benchmarks": equity_benchmarks,
                    "sample_size": len(equity_data),
                    "positions_analyzed": request_data.positions,
                    "industries_analyzed": request_data.industries
                }
            }
            
        except Exception as e:
            logger.error(f"Error in equity analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_compensation_data(self, data: List[CompensationDataPoint], benchmarks: List[CompensationBenchmark]) -> Dict[str, Any]:
        """Analyze compensation data and generate insights"""
        if not data:
            return {"error": "No data available for analysis"}
        
        # Calculate overall statistics
        base_salaries = [dp.base_salary for dp in data]
        total_comps = [dp.total_compensation for dp in data]
        equity_values = [dp.equity_value for dp in data]
        bonuses = [dp.bonus for dp in data]
        
        analysis = {
            "base_salary": {
                "mean": float(sum(base_salaries) / len(base_salaries)),
                "median": float(sorted(base_salaries)[len(base_salaries) // 2]),
                "min": float(min(base_salaries)),
                "max": float(max(base_salaries)),
                "std": float(np.std(base_salaries))
            },
            "total_compensation": {
                "mean": float(sum(total_comps) / len(total_comps)),
                "median": float(sorted(total_comps)[len(total_comps) // 2]),
                "min": float(min(total_comps)),
                "max": float(max(total_comps)),
                "std": float(np.std(total_comps))
            },
            "equity": {
                "mean": float(sum(equity_values) / len(equity_values)),
                "median": float(sorted(equity_values)[len(equity_values) // 2]),
                "min": float(min(equity_values)),
                "max": float(max(equity_values)),
                "std": float(np.std(equity_values))
            },
            "bonus": {
                "mean": float(sum(bonuses) / len(bonuses)),
                "median": float(sorted(bonuses)[len(bonuses) // 2]),
                "min": float(min(bonuses)),
                "max": float(max(bonuses)),
                "std": float(np.std(bonuses))
            }
        }
        
        # Add trend analysis
        analysis["trends"] = self._analyze_compensation_trends(data)
        
        # Add industry comparison
        analysis["industry_comparison"] = self._compare_by_industry(data)
        
        # Add location comparison
        analysis["location_comparison"] = self._compare_by_location(data)
        
        return analysis
    
    def _analyze_compensation_trends(self, data: List[CompensationDataPoint]) -> Dict[str, Any]:
        """Analyze compensation trends over time"""
        # Group by source and analyze trends
        trends = {}
        
        for source in set(dp.source for dp in data):
            source_data = [dp for dp in data if dp.source == source]
            if len(source_data) > 1:
                # Simple trend analysis (in a real implementation, you'd use time series analysis)
                base_salaries = [dp.base_salary for dp in source_data]
                trends[source] = {
                    "trend_direction": "stable",  # Simplified
                    "volatility": float(np.std(base_salaries) / np.mean(base_salaries)) if np.mean(base_salaries) > 0 else 0
                }
        
        return trends
    
    def _compare_by_industry(self, data: List[CompensationDataPoint]) -> Dict[str, Any]:
        """Compare compensation across industries"""
        industry_stats = {}
        
        for industry in set(dp.industry for dp in data):
            industry_data = [dp for dp in data if dp.industry == industry]
            if industry_data:
                base_salaries = [dp.base_salary for dp in industry_data]
                total_comps = [dp.total_compensation for dp in industry_data]
                
                industry_stats[industry.value] = {
                    "avg_base_salary": float(sum(base_salaries) / len(base_salaries)),
                    "avg_total_comp": float(sum(total_comps) / len(total_comps)),
                    "sample_size": len(industry_data)
                }
        
        return industry_stats
    
    def _compare_by_location(self, data: List[CompensationDataPoint]) -> Dict[str, Any]:
        """Compare compensation across locations"""
        location_stats = {}
        
        for location in set(dp.location for dp in data):
            location_data = [dp for dp in data if dp.location == location]
            if location_data:
                base_salaries = [dp.base_salary for dp in location_data]
                total_comps = [dp.total_compensation for dp in location_data]
                
                location_stats[location] = {
                    "avg_base_salary": float(sum(base_salaries) / len(base_salaries)),
                    "avg_total_comp": float(sum(total_comps) / len(total_comps)),
                    "sample_size": len(location_data)
                }
        
        return location_stats
    
    def _filter_by_criteria(self, data: List[CompensationDataPoint], criteria: SalaryBenchmarkRequest) -> List[CompensationDataPoint]:
        """Filter data by experience level and company size"""
        filtered = data
        
        if criteria.experience_level != "Any":
            filtered = [dp for dp in filtered if criteria.experience_level.lower() in dp.experience_level.lower()]
        
        if criteria.company_size != "Any":
            filtered = [dp for dp in filtered if criteria.company_size.lower() in dp.company_size.lower()]
        
        return filtered
    
    def _calculate_detailed_statistics(self, data: List[CompensationDataPoint]) -> Dict[str, Any]:
        """Calculate detailed statistics for filtered data"""
        if not data:
            return {"error": "No data available"}
        
        base_salaries = [dp.base_salary for dp in data]
        total_comps = [dp.total_compensation for dp in data]
        equity_values = [dp.equity_value for dp in data]
        bonuses = [dp.bonus for dp in data]
        
        return {
            "base_salary": self._calculate_percentiles(base_salaries),
            "total_compensation": self._calculate_percentiles(total_comps),
            "equity": self._calculate_percentiles(equity_values),
            "bonus": self._calculate_percentiles(bonuses),
            "compensation_ratio": {
                "equity_to_base": float(sum(equity_values) / sum(base_salaries)) if sum(base_salaries) > 0 else 0,
                "bonus_to_base": float(sum(bonuses) / sum(base_salaries)) if sum(base_salaries) > 0 else 0,
                "total_to_base": float(sum(total_comps) / sum(base_salaries)) if sum(base_salaries) > 0 else 0
            }
        }
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentiles for a list of values"""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        percentiles = {}
        percentile_ranges = {
            "10th": 0.10,
            "25th": 0.25,
            "50th": 0.50,
            "75th": 0.75,
            "90th": 0.90,
            "95th": 0.95,
            "99th": 0.99
        }
        
        for name, p in percentile_ranges.items():
            index = int((n - 1) * p)
            percentiles[name] = float(sorted_values[index])
        
        return percentiles
    
    def _analyze_equity_patterns(self, data: List[CompensationDataPoint]) -> Dict[str, Any]:
        """Analyze equity patterns in the data"""
        if not data:
            return {"error": "No equity data available"}
        
        # Analyze by position
        position_equity = {}
        for dp in data:
            if dp.position not in position_equity:
                position_equity[dp.position] = []
            position_equity[dp.position].append(dp.equity_value)
        
        # Calculate statistics for each position
        position_stats = {}
        for position, equity_values in position_equity.items():
            position_stats[position] = {
                "avg_equity": float(sum(equity_values) / len(equity_values)),
                "median_equity": float(sorted(equity_values)[len(equity_values) // 2]),
                "max_equity": float(max(equity_values)),
                "sample_size": len(equity_values)
            }
        
        # Analyze by industry
        industry_equity = {}
        for dp in data:
            if dp.industry.value not in industry_equity:
                industry_equity[dp.industry.value] = []
            industry_equity[dp.industry.value].append(dp.equity_value)
        
        industry_stats = {}
        for industry, equity_values in industry_equity.items():
            industry_stats[industry] = {
                "avg_equity": float(sum(equity_values) / len(equity_values)),
                "median_equity": float(sorted(equity_values)[len(equity_values) // 2]),
                "sample_size": len(equity_values)
            }
        
        return {
            "by_position": position_stats,
            "by_industry": industry_stats,
            "overall": {
                "avg_equity": float(sum(dp.equity_value for dp in data) / len(data)),
                "median_equity": float(sorted([dp.equity_value for dp in data])[len(data) // 2]),
                "total_sample_size": len(data)
            }
        }
    
    def _calculate_equity_benchmarks(self, data: List[CompensationDataPoint]) -> Dict[str, Any]:
        """Calculate equity benchmarks"""
        if not data:
            return {}
        
        equity_values = [dp.equity_value for dp in data]
        stock_options = [dp.stock_options for dp in data]
        rsu_values = [dp.rsu_value for dp in data]
        
        return {
            "equity": self._calculate_percentiles(equity_values),
            "stock_options": self._calculate_percentiles(stock_options),
            "rsu": self._calculate_percentiles(rsu_values)
        }
    
    def _generate_compensation_insights(self, data: List[CompensationDataPoint], benchmarks: List[CompensationBenchmark]) -> List[str]:
        """Generate compensation insights"""
        insights = []
        
        if not data:
            return ["No compensation data available for analysis"]
        
        # Calculate overall statistics
        base_salaries = [dp.base_salary for dp in data]
        total_comps = [dp.total_compensation for dp in data]
        equity_values = [dp.equity_value for dp in data]
        
        avg_base = sum(base_salaries) / len(base_salaries)
        avg_total = sum(total_comps) / len(total_comps)
        avg_equity = sum(equity_values) / len(equity_values)
        
        # Generate insights
        if avg_equity > avg_base * 0.5:
            insights.append("High equity component observed - typical for tech and startup roles")
        
        if avg_total > avg_base * 1.3:
            insights.append("Significant total compensation above base salary - strong bonus/equity culture")
        
        # Industry insights
        industry_analysis = self._compare_by_industry(data)
        if len(industry_analysis) > 1:
            highest_paying = max(industry_analysis.items(), key=lambda x: x[1]["avg_base_salary"])
            insights.append(f"Highest paying industry: {highest_paying[0]} (${highest_paying[1]['avg_base_salary']:,.0f} avg base)")
        
        # Location insights
        location_analysis = self._compare_by_location(data)
        if len(location_analysis) > 1:
            highest_paying_loc = max(location_analysis.items(), key=lambda x: x[1]["avg_base_salary"])
            insights.append(f"Highest paying location: {highest_paying_loc[0]} (${highest_paying_loc[1]['avg_base_salary']:,.0f} avg base)")
        
        return insights
    
    def _generate_compensation_recommendations(self, data: List[CompensationDataPoint], benchmarks: List[CompensationBenchmark]) -> List[str]:
        """Generate compensation recommendations"""
        recommendations = []
        
        if not data:
            return ["Gather more compensation data for meaningful recommendations"]
        
        # Analyze compensation structure
        base_salaries = [dp.base_salary for dp in data]
        total_comps = [dp.total_compensation for dp in data]
        equity_values = [dp.equity_value for dp in data]
        
        avg_base = sum(base_salaries) / len(base_salaries)
        avg_total = sum(total_comps) / len(total_comps)
        avg_equity = sum(equity_values) / len(equity_values)
        
        # Equity recommendations
        if avg_equity < avg_base * 0.1:
            recommendations.append("Consider increasing equity component for competitive positioning")
        elif avg_equity > avg_base * 0.8:
            recommendations.append("High equity component - ensure proper vesting schedules and valuation")
        
        # Total compensation recommendations
        comp_ratio = avg_total / avg_base if avg_base > 0 else 1
        if comp_ratio < 1.2:
            recommendations.append("Consider increasing bonus/equity to improve total compensation package")
        elif comp_ratio > 2.0:
            recommendations.append("Very high total compensation - ensure sustainable compensation structure")
        
        # Benchmark recommendations
        if benchmarks:
            recommendations.append("Use provided benchmarks for competitive positioning in negotiations")
        
        return recommendations


# Initialize API instance
compensation_api = CompensationIntelligenceAPI()


# API Endpoints
@compensation_bp.route("/comprehensive", methods=["POST"])
def comprehensive_compensation_analysis():
    """Get comprehensive compensation analysis"""
    try:
        request_data = CompensationAnalysisRequest(**request.get_json())
        
        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            compensation_api.get_comprehensive_compensation_analysis(request_data)
        )
        loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in comprehensive compensation analysis endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@compensation_bp.route("/benchmarks", methods=["POST"])
def salary_benchmarks():
    """Get detailed salary benchmarks"""
    try:
        request_data = SalaryBenchmarkRequest(**request.get_json())
        
        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            compensation_api.get_salary_benchmarks(request_data)
        )
        loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in salary benchmarks endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@compensation_bp.route("/equity", methods=["POST"])
def equity_analysis():
    """Get equity and stock option analysis"""
    try:
        request_data = EquityAnalysisRequest(**request.get_json())
        
        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            compensation_api.get_equity_analysis(request_data)
        )
        loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in equity analysis endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@compensation_bp.route("/sources", methods=["GET"])
def available_sources():
    """Get list of available data sources"""
    return jsonify({
        "data_sources": [source.value for source in DataSource],
        "descriptions": {
            "glassdoor": "Glassdoor salary surveys and company reviews",
            "payscale": "PayScale compensation data and salary reports",
            "levels_fyi": "Levels.fyi tech compensation data",
            "bls": "Bureau of Labor Statistics official data",
            "census": "US Census Bureau demographic and economic data",
            "compensation_reports": "Industry compensation reports",
            "equity_data": "Stock option and equity compensation data"
        }
    })


@compensation_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for compensation intelligence service"""
    return jsonify({
        "status": "healthy",
        "service": "compensation_intelligence",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


@compensation_bp.route("/dashboard", methods=["GET"])
def dashboard():
    """Serve the compensation intelligence dashboard"""
    from flask import render_template
    return render_template("compensation_intelligence_dashboard.html")
