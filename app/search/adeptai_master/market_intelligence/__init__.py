"""
Market Intelligence Module for AdeptAI

This module provides comprehensive market intelligence capabilities including:
- Salary trend analysis
- Skill demand tracking
- Job market analytics
- Industry insights
- Market forecasting

Version: 1.0.0
"""

from .data_collectors import (
    SalaryDataCollector,
    SkillDemandCollector,
    JobMarketCollector,
    IndustryInsightCollector
)

from .analyzers import (
    SalaryTrendAnalyzer,
    SkillDemandAnalyzer,
    MarketTrendAnalyzer,
    IndustryAnalyzer
)

from .models import (
    MarketIntelligenceData,
    SalaryTrend,
    SkillDemand,
    JobMarketTrend,
    IndustryInsight,
    MarketForecast
)

from .api import MarketIntelligenceAPI
from .salary_intelligence import (
    RealTimeSalaryCollector, CompensationBenchmarker,
    CompensationDataPoint, CompensationBenchmark, CompensationType, DataSource
)
from .compensation_api import CompensationIntelligenceAPI

__version__ = "1.0.0"

__all__ = [
    # Data Collectors
    "SalaryDataCollector",
    "SkillDemandCollector", 
    "JobMarketCollector",
    "IndustryInsightCollector",
    "RealTimeSalaryCollector",
    "CompensationBenchmarker",
    
    # Analyzers
    "SalaryTrendAnalyzer",
    "SkillDemandAnalyzer",
    "MarketTrendAnalyzer",
    "IndustryAnalyzer",
    
    # Models
    "MarketIntelligenceData",
    "SalaryTrend",
    "SkillDemand",
    "JobMarketTrend",
    "IndustryInsight",
    "MarketForecast",
    "CompensationDataPoint",
    "CompensationBenchmark",
    "CompensationType",
    "DataSource",
    
    # APIs
    "MarketIntelligenceAPI",
    "CompensationIntelligenceAPI"
]
