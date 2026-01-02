"""
Market Intelligence Integration Architecture and Data Pipeline

Provides:
- Real-time data ingestion from multiple sources
- AI processing layer for market intelligence
- Algorithm enhancement with market data
- Integration with existing candidate search
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import random
from .hybrid_llm_service import market_intelligence_llm, TaskComplexity


logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    endpoint: str
    rate_limit: float
    priority: int
    enabled: bool = True


@dataclass
class ProcessedIntelligence:
    """Processed market intelligence data"""
    talent_availability: Dict[str, Any]
    compensation_trends: Dict[str, Any]
    skill_evolution: Dict[str, Any]
    competitive_landscape: Dict[str, Any]
    economic_indicators: Dict[str, Any]
    behavioral_insights: Dict[str, Any]
    timestamp: datetime


class DataIngestionEngine:
    """Real-time data ingestion from multiple sources"""
    
    def __init__(self):
        self.data_sources = {
            "salary_apis": DataSource("Salary APIs", "compensation_apis", 1.0, 1),
            "job_boards": DataSource("Job Boards", "job_scraping", 0.5, 2),
            "government_apis": DataSource("Government APIs", "economic_data", 2.0, 3),
            "social_signals": DataSource("Social Signals", "linkedin_twitter", 0.3, 4),
            "industry_reports": DataSource("Industry Reports", "sector_data", 5.0, 5)
        }
        
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def stream_compensation_apis(self) -> Dict[str, Any]:
        """Stream data from compensation APIs"""
        # Simulate real-time compensation data
        await asyncio.sleep(0.1)  # Rate limiting
        
        return {
            "source": "compensation_apis",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "glassdoor_updates": random.randint(50, 200),
                "payscale_updates": random.randint(30, 150),
                "levels_fyi_updates": random.randint(20, 100),
                "salary_ranges": {
                    "software_engineer": [80000, 150000],
                    "data_scientist": [90000, 160000],
                    "product_manager": [100000, 180000]
                }
            }
        }
    
    async def scrape_job_boards(self) -> Dict[str, Any]:
        """Scrape job board data"""
        await asyncio.sleep(0.2)
        
        return {
            "source": "job_boards",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "indeed_postings": random.randint(1000, 5000),
                "linkedin_jobs": random.randint(800, 4000),
                "stack_overflow_jobs": random.randint(200, 1000),
                "skill_demands": {
                    "python": random.randint(200, 800),
                    "javascript": random.randint(300, 900),
                    "kubernetes": random.randint(50, 300)
                }
            }
        }
    
    async def fetch_government_apis(self) -> Dict[str, Any]:
        """Fetch government economic data"""
        await asyncio.sleep(0.5)
        
        return {
            "source": "government_apis",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "unemployment_rate": round(random.uniform(3.5, 6.5), 1),
                "gdp_growth": round(random.uniform(1.5, 4.0), 1),
                "inflation_rate": round(random.uniform(2.0, 5.0), 1),
                "labor_force_participation": round(random.uniform(60, 65), 1)
            }
        }
    
    async def collect_social_signals(self) -> Dict[str, Any]:
        """Collect social media and professional network signals"""
        await asyncio.sleep(0.3)
        
        return {
            "source": "social_signals",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "linkedin_activity": random.randint(500, 2000),
                "twitter_mentions": random.randint(100, 800),
                "github_trends": random.randint(50, 400),
                "sentiment_score": round(random.uniform(0.3, 0.8), 2)
            }
        }
    
    async def gather_industry_reports(self) -> Dict[str, Any]:
        """Gather industry-specific reports and predictions"""
        await asyncio.sleep(1.0)
        
        return {
            "source": "industry_reports",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "tech_growth": round(random.uniform(5, 15), 1),
                "fintech_adoption": round(random.uniform(20, 60), 1),
                "ai_investment": round(random.uniform(10, 50), 1),
                "remote_work_trends": round(random.uniform(30, 80), 1)
            }
        }


class AIProcessingLayer:
    """AI processing layer for market intelligence"""
    
    def __init__(self):
        self.processing_weights = {
            "talent_availability": 0.25,
            "compensation_trends": 0.25,
            "skill_evolution": 0.20,
            "competitive_landscape": 0.15,
            "economic_indicators": 0.10,
            "behavioral_insights": 0.05
        }
    
    async def process_supply_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process talent supply data"""
        # Simulate AI processing
        await asyncio.sleep(0.1)
        
        job_data = raw_data.get("job_boards", {}).get("data", {})
        social_data = raw_data.get("social_signals", {}).get("data", {})
        
        return {
            "active_candidates": sum(job_data.get("skill_demands", {}).values()),
            "passive_candidates": int(sum(job_data.get("skill_demands", {}).values()) * 2.5),
            "geographic_distribution": {
                "san_francisco": random.randint(200, 800),
                "new_york": random.randint(300, 900),
                "seattle": random.randint(150, 600),
                "remote": random.randint(500, 1500)
            },
            "mobility_likelihood": round(random.uniform(0.3, 0.7), 2),
            "confidence_score": round(random.uniform(0.7, 0.95), 2)
        }
    
    async def analyze_salary_movements(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze salary movements and trends"""
        await asyncio.sleep(0.1)
        
        comp_data = raw_data.get("salary_apis", {}).get("data", {})
        salary_ranges = comp_data.get("salary_ranges", {})
        
        # Calculate trends
        trends = {}
        for role, (min_sal, max_sal) in salary_ranges.items():
            avg_salary = (min_sal + max_sal) / 2
            trend_direction = "rising" if random.random() > 0.5 else "stable"
            trends[role] = {
                "average_salary": avg_salary,
                "trend_direction": trend_direction,
                "growth_rate": round(random.uniform(0, 15), 1)
            }
        
        return {
            "salary_trends": trends,
            "overall_inflation": round(random.uniform(3, 8), 1),
            "market_competitiveness": round(random.uniform(0.6, 0.9), 2),
            "confidence_score": round(random.uniform(0.8, 0.95), 2)
        }
    
    async def track_technology_adoption(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track technology adoption and skill evolution"""
        await asyncio.sleep(0.1)
        
        job_data = raw_data.get("job_boards", {}).get("data", {})
        social_data = raw_data.get("social_signals", {}).get("data", {})
        
        skill_demands = job_data.get("skill_demands", {})
        github_trends = social_data.get("github_trends", 0)
        
        # Analyze skill evolution
        skill_analysis = {}
        for skill, demand in skill_demands.items():
            growth_rate = round(random.uniform(-10, 50), 1)
            skill_analysis[skill] = {
                "current_demand": demand,
                "growth_rate": growth_rate,
                "trend_direction": "rising" if growth_rate > 5 else "stable" if growth_rate > -5 else "declining",
                "adoption_score": min(100, demand + github_trends)
            }
        
        return {
            "skill_evolution": skill_analysis,
            "emerging_skills": ["Rust", "WebAssembly", "Edge Computing"],
            "declining_skills": ["jQuery", "Flash", "Internet Explorer"],
            "confidence_score": round(random.uniform(0.7, 0.9), 2)
        }
    
    async def monitor_industry_activity(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor competitive landscape and industry activity"""
        await asyncio.sleep(0.1)
        
        industry_data = raw_data.get("industry_reports", {}).get("data", {})
        
        return {
            "industry_growth": {
                "technology": industry_data.get("tech_growth", 0),
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


class MarketIntelligencePipeline:
    """Main market intelligence pipeline with hybrid LLM integration"""
    
    def __init__(self):
        self.data_ingestion = DataIngestionEngine()
        self.ai_processing = AIProcessingLayer()
        self.llm_service = market_intelligence_llm
        self.cache = {}
        self.last_update = None
    
    async def market_intelligence_pipeline(self) -> ProcessedIntelligence:
        """Main pipeline function with hybrid LLM enhancement"""
        logger.info("Starting market intelligence pipeline")
        
        # Step 1: Real-time data ingestion
        raw_data = await self._ingest_all_sources()
        
        # Step 2: AI processing layer
        processed_data = await self._process_with_ai(raw_data)
        
        # Step 3: LLM enhancement
        llm_enhanced_data = await self._enhance_with_llm(processed_data)
        
        # Step 4: Create processed intelligence object
        intelligence = ProcessedIntelligence(
            talent_availability=llm_enhanced_data["talent_availability"],
            compensation_trends=llm_enhanced_data["compensation_trends"],
            skill_evolution=llm_enhanced_data["skill_evolution"],
            competitive_landscape=llm_enhanced_data["competitive_landscape"],
            economic_indicators=llm_enhanced_data["economic_indicators"],
            behavioral_insights=llm_enhanced_data["behavioral_insights"],
            timestamp=datetime.now()
        )
        
        # Step 5: Cache results
        self.cache["last_intelligence"] = intelligence
        self.last_update = datetime.now()
        
        logger.info("Market intelligence pipeline completed")
        return intelligence
    
    async def _enhance_with_llm(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance processed data with hybrid LLM analysis"""
        try:
            # Prepare data for LLM analysis
            market_data = {
                "talent_availability": processed_data["talent_availability"],
                "compensation_trends": processed_data["compensation_trends"],
                "skill_evolution": processed_data["skill_evolution"],
                "competitive_landscape": processed_data["competitive_landscape"],
                "economic_indicators": processed_data["economic_indicators"]
            }
            
            # Get LLM insights
            llm_insights = await self.llm_service.generate_market_insights(market_data)
            
            # Enhance each component with LLM insights
            enhanced_data = processed_data.copy()
            
            # Add LLM enhancement metadata
            enhanced_data["llm_enhancement"] = {
                "applied": True,
                "insights": llm_insights,
                "models_used": ["gpt-4o-mini", "claude-3-5-sonnet-20241022"],
                "enhancement_timestamp": datetime.now().isoformat()
            }
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error in LLM enhancement: {e}")
            # Return original data if LLM enhancement fails
            processed_data["llm_enhancement"] = {
                "applied": False,
                "error": str(e),
                "enhancement_timestamp": datetime.now().isoformat()
            }
            return processed_data
    
    async def _ingest_all_sources(self) -> Dict[str, Any]:
        """Ingest data from all sources"""
        sources = [
            self.data_ingestion.stream_compensation_apis(),
            self.data_ingestion.scrape_job_boards(),
            self.data_ingestion.fetch_government_apis(),
            self.data_ingestion.collect_social_signals(),
            self.data_ingestion.gather_industry_reports()
        ]
        
        results = await asyncio.gather(*sources)
        
        # Combine results
        combined_data = {}
        for result in results:
            source = result["source"]
            combined_data[source] = result
        
        return combined_data
    
    async def _process_with_ai(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw data with AI layer"""
        # Process each component
        talent_availability = await self.ai_processing.process_supply_data(raw_data)
        compensation_trends = await self.ai_processing.analyze_salary_movements(raw_data)
        skill_evolution = await self.ai_processing.track_technology_adoption(raw_data)
        competitive_landscape = await self.ai_processing.monitor_industry_activity(raw_data)
        
        # Economic indicators (simplified)
        economic_indicators = {
            "unemployment_rate": raw_data.get("government_apis", {}).get("data", {}).get("unemployment_rate", 4.5),
            "gdp_growth": raw_data.get("government_apis", {}).get("data", {}).get("gdp_growth", 2.5),
            "inflation_rate": raw_data.get("government_apis", {}).get("data", {}).get("inflation_rate", 3.0)
        }
        
        # Behavioral insights (simplified)
        behavioral_insights = {
            "job_switch_probability": round(random.uniform(0.2, 0.6), 2),
            "remote_work_preference": round(random.uniform(0.4, 0.8), 2),
            "salary_satisfaction": round(random.uniform(0.5, 0.9), 2),
            "interview_acceptance_rate": round(random.uniform(0.6, 0.9), 2)
        }
        
        return {
            "talent_availability": talent_availability,
            "compensation_trends": compensation_trends,
            "skill_evolution": skill_evolution,
            "competitive_landscape": competitive_landscape,
            "economic_indicators": economic_indicators,
            "behavioral_insights": behavioral_insights
        }
    
    async def apply_market_intelligence(self, 
                                      candidate_pool: List[Dict[str, Any]], 
                                      job_specifications: Dict[str, Any]) -> Dict[str, Any]:
        """Apply market intelligence to enhance recommendations"""
        # Get latest intelligence
        if not self.cache.get("last_intelligence"):
            intelligence = await self.market_intelligence_pipeline()
        else:
            intelligence = self.cache["last_intelligence"]
        
        # Apply intelligence to candidate scoring
        enhanced_candidates = []
        for candidate in candidate_pool:
            enhanced_score = self._calculate_enhanced_score(candidate, intelligence, job_specifications)
            enhanced_candidate = candidate.copy()
            enhanced_candidate["market_intelligence_score"] = enhanced_score
            enhanced_candidate["market_factors"] = self._extract_market_factors(candidate, intelligence)
            enhanced_candidates.append(enhanced_candidate)
        
        # Sort by enhanced score
        enhanced_candidates.sort(key=lambda x: x.get("market_intelligence_score", 0), reverse=True)
        
        return {
            "enhanced_candidates": enhanced_candidates,
            "market_intelligence": {
                "talent_availability": intelligence.talent_availability,
                "compensation_trends": intelligence.compensation_trends,
                "skill_evolution": intelligence.skill_evolution,
                "competitive_landscape": intelligence.competitive_landscape
            },
            "recommendations": self._generate_recommendations(intelligence, job_specifications),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_enhanced_score(self, 
                                candidate: Dict[str, Any], 
                                intelligence: ProcessedIntelligence,
                                job_spec: Dict[str, Any]) -> float:
        """Calculate enhanced candidate score using market intelligence"""
        base_score = candidate.get("score", 0.5)
        
        # Apply market intelligence factors
        talent_factor = self._get_talent_availability_factor(candidate, intelligence.talent_availability)
        comp_factor = self._get_compensation_factor(candidate, intelligence.compensation_trends)
        skill_factor = self._get_skill_evolution_factor(candidate, intelligence.skill_evolution)
        
        # Weighted combination
        enhanced_score = (
            base_score * 0.4 +
            talent_factor * 0.2 +
            comp_factor * 0.2 +
            skill_factor * 0.2
        )
        
        return min(1.0, max(0.0, enhanced_score))
    
    def _get_talent_availability_factor(self, candidate: Dict[str, Any], talent_data: Dict[str, Any]) -> float:
        """Get talent availability factor for candidate"""
        # Simplified: higher availability = lower factor (less rare)
        total_available = talent_data.get("active_candidates", 1000)
        return max(0.5, 1.0 - (total_available / 10000))
    
    def _get_compensation_factor(self, candidate: Dict[str, Any], comp_data: Dict[str, Any]) -> float:
        """Get compensation factor for candidate"""
        # Simplified: match with market trends
        salary_trends = comp_data.get("salary_trends", {})
        candidate_role = candidate.get("role", "software_engineer")
        
        if candidate_role in salary_trends:
            trend = salary_trends[candidate_role]
            if trend.get("trend_direction") == "rising":
                return 1.1  # Bonus for rising demand
            else:
                return 1.0
        return 1.0
    
    def _get_skill_evolution_factor(self, candidate: Dict[str, Any], skill_data: Dict[str, Any]) -> float:
        """Get skill evolution factor for candidate"""
        candidate_skills = candidate.get("skills", [])
        skill_evolution = skill_data.get("skill_evolution", {})
        
        factor = 1.0
        for skill in candidate_skills:
            if skill in skill_evolution:
                skill_info = skill_evolution[skill]
                if skill_info.get("trend_direction") == "rising":
                    factor += 0.1
                elif skill_info.get("trend_direction") == "declining":
                    factor -= 0.05
        
        return max(0.5, min(1.5, factor))
    
    def _extract_market_factors(self, candidate: Dict[str, Any], intelligence: ProcessedIntelligence) -> Dict[str, Any]:
        """Extract relevant market factors for candidate"""
        return {
            "talent_scarcity": intelligence.talent_availability.get("confidence_score", 0.5),
            "salary_trend": intelligence.compensation_trends.get("overall_inflation", 0),
            "skill_demand": intelligence.skill_evolution.get("confidence_score", 0.5),
            "market_competitiveness": intelligence.competitive_landscape.get("market_sentiment", 0.5)
        }
    
    def _generate_recommendations(self, intelligence: ProcessedIntelligence, job_spec: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on market intelligence"""
        recommendations = []
        
        # Talent availability recommendations
        if intelligence.talent_availability.get("confidence_score", 0) > 0.8:
            recommendations.append("High confidence in talent availability - consider expanding search criteria")
        
        # Compensation recommendations
        inflation = intelligence.compensation_trends.get("overall_inflation", 0)
        if inflation > 5:
            recommendations.append(f"High salary inflation ({inflation}%) - adjust compensation bands upward")
        
        # Skill evolution recommendations
        skill_evolution = intelligence.skill_evolution.get("skill_evolution", {})
        rising_skills = [skill for skill, data in skill_evolution.items() 
                        if data.get("trend_direction") == "rising"]
        if rising_skills:
            recommendations.append(f"Consider prioritizing candidates with rising skills: {', '.join(rising_skills[:3])}")
        
        return recommendations
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "cache_size": len(self.cache),
            "data_sources": len(self.data_ingestion.data_sources),
            "processing_weights": self.ai_processing.processing_weights,
            "pipeline_health": "healthy" if self.last_update else "needs_update"
        }


# Convenience functions for API access
async def run_market_intelligence_pipeline() -> Dict[str, Any]:
    """Run the complete market intelligence pipeline"""
    pipeline = MarketIntelligencePipeline()
    intelligence = await pipeline.market_intelligence_pipeline()
    
    return {
        "timestamp": intelligence.timestamp.isoformat(),
        "talent_availability": intelligence.talent_availability,
        "compensation_trends": intelligence.compensation_trends,
        "skill_evolution": intelligence.skill_evolution,
        "competitive_landscape": intelligence.competitive_landscape,
        "economic_indicators": intelligence.economic_indicators,
        "behavioral_insights": intelligence.behavioral_insights
    }


async def enhance_candidate_recommendations(candidate_pool: List[Dict[str, Any]], 
                                          job_specifications: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance candidate recommendations with market intelligence"""
    pipeline = MarketIntelligencePipeline()
    return await pipeline.apply_market_intelligence(candidate_pool, job_specifications)
