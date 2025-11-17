"""
Real-Time Market Alerts and Dynamic Algorithm Adjustments

Provides:
- Real-time market signal detection
- Automatic algorithm adjustments
- Alert management system
- Market intelligence engine
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from .hybrid_llm_service import market_intelligence_llm, TaskComplexity


logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of market alerts"""
    LAYOFF_ANNOUNCEMENT = "layoff_announcement"
    SALARY_INFLATION = "salary_inflation"
    EMERGING_SKILLS = "emerging_skills"
    COMPETITOR_HIRING = "competitor_hiring"
    ECONOMIC_DOWNTURN = "economic_downturn"
    TALENT_SHORTAGE = "talent_shortage"
    REMOTE_WORK_SHIFT = "remote_work_shift"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MarketAlert:
    """Market alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    affected_industries: List[str]
    affected_locations: List[str]
    confidence_score: float
    timestamp: datetime
    expires_at: Optional[datetime] = None
    actions_taken: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmAdjustment:
    """Algorithm adjustment configuration"""
    adjustment_id: str
    alert_trigger: AlertType
    parameter_name: str
    old_value: Any
    new_value: Any
    adjustment_reason: str
    timestamp: datetime
    active: bool = True


class MarketSignalDetector:
    """Detects market signals and triggers alerts"""
    
    def __init__(self):
        self.detection_thresholds = {
            AlertType.LAYOFF_ANNOUNCEMENT: 0.7,
            AlertType.SALARY_INFLATION: 0.8,
            AlertType.EMERGING_SKILLS: 0.6,
            AlertType.COMPETITOR_HIRING: 0.5,
            AlertType.ECONOMIC_DOWNTURN: 0.9,
            AlertType.TALENT_SHORTAGE: 0.7,
            AlertType.REMOTE_WORK_SHIFT: 0.6
        }
    
    async def detect_layoffs_in_target_companies(self) -> Optional[MarketAlert]:
        """Detect layoff announcements in target companies"""
        # Simulate layoff detection
        layoff_probability = random.uniform(0, 1)
        
        if layoff_probability > self.detection_thresholds[AlertType.LAYOFF_ANNOUNCEMENT]:
            companies = ["TechCorp", "StartupXYZ", "ScaleUpInc"]
            affected_company = random.choice(companies)
            
            return MarketAlert(
                alert_id=f"layoff_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type=AlertType.LAYOFF_ANNOUNCEMENT,
                severity=AlertSeverity.HIGH,
                title=f"Layoff Announcement: {affected_company}",
                description=f"{affected_company} announced layoffs affecting 15% of workforce",
                affected_industries=["Technology"],
                affected_locations=["San Francisco", "New York"],
                confidence_score=layoff_probability,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=30),
                metadata={"company": affected_company, "layoff_percentage": 15}
            )
        
        return None
    
    async def detect_salary_inflation(self) -> Optional[MarketAlert]:
        """Detect salary inflation trends"""
        # Simulate salary inflation detection
        inflation_probability = random.uniform(0, 1)
        
        if inflation_probability > self.detection_thresholds[AlertType.SALARY_INFLATION]:
            inflation_rate = random.uniform(8, 15)
            
            return MarketAlert(
                alert_id=f"salary_inflation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type=AlertType.SALARY_INFLATION,
                severity=AlertSeverity.MEDIUM,
                title=f"Salary Inflation Detected: {inflation_rate:.1f}%",
                description=f"Average salary increases across target roles exceed {inflation_rate:.1f}%",
                affected_industries=["Technology", "Finance"],
                affected_locations=["Global"],
                confidence_score=inflation_probability,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=90),
                metadata={"inflation_rate": inflation_rate, "affected_roles": ["Software Engineer", "Data Scientist"]}
            )
        
        return None
    
    async def detect_emerging_skills(self) -> Optional[MarketAlert]:
        """Detect emerging skills trends"""
        # Simulate emerging skills detection
        emergence_probability = random.uniform(0, 1)
        
        if emergence_probability > self.detection_thresholds[AlertType.EMERGING_SKILLS]:
            emerging_skills = ["Rust", "WebAssembly", "Quantum Computing", "Edge AI"]
            skill = random.choice(emerging_skills)
            growth_rate = random.uniform(50, 200)
            
            return MarketAlert(
                alert_id=f"emerging_skill_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type=AlertType.EMERGING_SKILLS,
                severity=AlertSeverity.MEDIUM,
                title=f"Emerging Skill Trend: {skill}",
                description=f"{skill} shows {growth_rate:.0f}% growth in job postings",
                affected_industries=["Technology"],
                affected_locations=["Global"],
                confidence_score=emergence_probability,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=180),
                metadata={"skill": skill, "growth_rate": growth_rate}
            )
        
        return None
    
    async def detect_competitor_hiring(self) -> Optional[MarketAlert]:
        """Detect competitor hiring activity"""
        # Simulate competitor hiring detection
        hiring_probability = random.uniform(0, 1)
        
        if hiring_probability > self.detection_thresholds[AlertType.COMPETITOR_HIRING]:
            competitors = ["AlphaTech", "BetaCorp", "GammaInc"]
            competitor = random.choice(competitors)
            open_roles = random.randint(50, 200)
            
            return MarketAlert(
                alert_id=f"competitor_hiring_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type=AlertType.COMPETITOR_HIRING,
                severity=AlertSeverity.LOW,
                title=f"Competitor Hiring Spree: {competitor}",
                description=f"{competitor} has {open_roles} open positions in target roles",
                affected_industries=["Technology"],
                affected_locations=["San Francisco", "Seattle"],
                confidence_score=hiring_probability,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=14),
                metadata={"competitor": competitor, "open_roles": open_roles}
            )
        
        return None


class MarketIntelligenceEngine:
    """Main market intelligence engine with auto-adjustments and hybrid LLM integration"""
    
    def __init__(self):
        self.alerts: List[MarketAlert] = []
        self.auto_adjustments = True
        self.adjustments: List[AlgorithmAdjustment] = []
        self.signal_detector = MarketSignalDetector()
        self.llm_service = market_intelligence_llm
        
        # Current algorithm parameters
        self.current_params = {
            "search_radius": "local",
            "salary_multiplier": 1.0,
            "experience_flexibility": "medium",
            "skill_priorities": ["Python", "JavaScript", "SQL"],
            "sourcing_activity_level": "normal",
            "compensation_aggressiveness": "balanced"
        }
    
    async def process_market_signals(self) -> List[MarketAlert]:
        """Process all market signals and generate alerts with LLM enhancement"""
        new_alerts = []
        
        # Detect various market signals
        detectors = [
            self.signal_detector.detect_layoffs_in_target_companies(),
            self.signal_detector.detect_salary_inflation(),
            self.signal_detector.detect_emerging_skills(),
            self.signal_detector.detect_competitor_hiring()
        ]
        
        for detector in detectors:
            try:
                alert = await detector
                if alert:
                    # Enhance alert with LLM analysis
                    enhanced_alert = await self._enhance_alert_with_llm(alert)
                    new_alerts.append(enhanced_alert)
                    self.alerts.append(enhanced_alert)
                    
                    # Apply automatic adjustments if enabled
                    if self.auto_adjustments:
                        await self._apply_automatic_adjustments(enhanced_alert)
            except Exception as e:
                logger.error(f"Error in market signal detection: {e}")
        
        return new_alerts
    
    async def _enhance_alert_with_llm(self, alert: MarketAlert) -> MarketAlert:
        """Enhance alert with LLM analysis"""
        try:
            # Prepare market data for LLM analysis
            market_data = {
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "affected_industries": alert.affected_industries,
                "confidence_score": alert.confidence_score,
                "metadata": alert.metadata
            }
            
            # Get LLM insights for alert
            llm_insights = await self.llm_service.generate_market_insights(market_data)
            
            # Enhance alert with LLM insights
            enhanced_metadata = alert.metadata.copy()
            enhanced_metadata["llm_analysis"] = llm_insights
            enhanced_metadata["ai_enhanced"] = True
            
            # Create enhanced alert
            enhanced_alert = MarketAlert(
                alert_id=alert.alert_id,
                alert_type=alert.alert_type,
                severity=alert.severity,
                title=alert.title,
                description=alert.description,
                affected_industries=alert.affected_industries,
                affected_locations=alert.affected_locations,
                confidence_score=alert.confidence_score,
                timestamp=alert.timestamp,
                expires_at=alert.expires_at,
                actions_taken=alert.actions_taken,
                metadata=enhanced_metadata
            )
            
            return enhanced_alert
            
        except Exception as e:
            logger.error(f"Error enhancing alert with LLM: {e}")
            return alert  # Return original alert if enhancement fails
    
    async def _apply_automatic_adjustments(self, alert: MarketAlert):
        """Apply automatic algorithm adjustments based on alerts"""
        adjustments = []
        
        if alert.alert_type == AlertType.LAYOFF_ANNOUNCEMENT:
            # Expand search radius and increase sourcing activity
            adjustments.extend([
                AlgorithmAdjustment(
                    adjustment_id=f"layoff_radius_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_trigger=AlertType.LAYOFF_ANNOUNCEMENT,
                    parameter_name="search_radius",
                    old_value=self.current_params["search_radius"],
                    new_value="global",
                    adjustment_reason="Layoffs detected - expand search to capture displaced talent",
                    timestamp=datetime.now()
                ),
                AlgorithmAdjustment(
                    adjustment_id=f"layoff_sourcing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_trigger=AlertType.LAYOFF_ANNOUNCEMENT,
                    parameter_name="sourcing_activity_level",
                    old_value=self.current_params["sourcing_activity_level"],
                    new_value="high",
                    adjustment_reason="Increase sourcing activity to capitalize on layoffs",
                    timestamp=datetime.now()
                )
            ])
        
        elif alert.alert_type == AlertType.SALARY_INFLATION:
            # Adjust compensation bands
            inflation_rate = alert.metadata.get("inflation_rate", 10)
            new_multiplier = 1 + (inflation_rate / 100)
            
            adjustments.append(AlgorithmAdjustment(
                adjustment_id=f"salary_inflation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_trigger=AlertType.SALARY_INFLATION,
                parameter_name="salary_multiplier",
                old_value=self.current_params["salary_multiplier"],
                new_value=new_multiplier,
                adjustment_reason=f"Salary inflation {inflation_rate:.1f}% - adjust compensation bands",
                timestamp=datetime.now()
            ))
        
        elif alert.alert_type == AlertType.EMERGING_SKILLS:
            # Update skill priorities
            new_skill = alert.metadata.get("skill", "New Skill")
            current_skills = self.current_params["skill_priorities"].copy()
            if new_skill not in current_skills:
                current_skills.insert(0, new_skill)  # Add to front
                if len(current_skills) > 5:  # Keep only top 5
                    current_skills = current_skills[:5]
            
            adjustments.append(AlgorithmAdjustment(
                adjustment_id=f"emerging_skill_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_trigger=AlertType.EMERGING_SKILLS,
                parameter_name="skill_priorities",
                old_value=self.current_params["skill_priorities"],
                new_value=current_skills,
                adjustment_reason=f"Emerging skill {new_skill} - update priority list",
                timestamp=datetime.now()
            ))
        
        # Apply adjustments
        for adjustment in adjustments:
            self._apply_adjustment(adjustment)
            self.adjustments.append(adjustment)
    
    def _apply_adjustment(self, adjustment: AlgorithmAdjustment):
        """Apply a single algorithm adjustment"""
        self.current_params[adjustment.parameter_name] = adjustment.new_value
        logger.info(f"Applied adjustment: {adjustment.adjustment_reason}")
    
    def increase_sourcing_activity(self):
        """Manually increase sourcing activity"""
        adjustment = AlgorithmAdjustment(
            adjustment_id=f"manual_sourcing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_trigger=AlertType.LAYOFF_ANNOUNCEMENT,
            parameter_name="sourcing_activity_level",
            old_value=self.current_params["sourcing_activity_level"],
            new_value="high",
            adjustment_reason="Manual increase in sourcing activity",
            timestamp=datetime.now()
        )
        self._apply_adjustment(adjustment)
        self.adjustments.append(adjustment)
    
    def update_budget_recommendations(self, inflation_rate: float = 0.1):
        """Update budget recommendations based on salary inflation"""
        new_multiplier = 1 + inflation_rate
        adjustment = AlgorithmAdjustment(
            adjustment_id=f"budget_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_trigger=AlertType.SALARY_INFLATION,
            parameter_name="salary_multiplier",
            old_value=self.current_params["salary_multiplier"],
            new_value=new_multiplier,
            adjustment_reason=f"Budget update for {inflation_rate*100:.1f}% salary inflation",
            timestamp=datetime.now()
        )
        self._apply_adjustment(adjustment)
        self.adjustments.append(adjustment)
    
    def adjust_candidate_scoring(self, new_skills: List[str]):
        """Adjust candidate scoring based on emerging skills"""
        current_skills = self.current_params["skill_priorities"].copy()
        for skill in new_skills:
            if skill not in current_skills:
                current_skills.insert(0, skill)
        
        if len(current_skills) > 5:
            current_skills = current_skills[:5]
        
        adjustment = AlgorithmAdjustment(
            adjustment_id=f"skill_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_trigger=AlertType.EMERGING_SKILLS,
            parameter_name="skill_priorities",
            old_value=self.current_params["skill_priorities"],
            new_value=current_skills,
            adjustment_reason=f"Updated skill priorities: {', '.join(new_skills)}",
            timestamp=datetime.now()
        )
        self._apply_adjustment(adjustment)
        self.adjustments.append(adjustment)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current engine status and parameters"""
        return {
            "timestamp": datetime.now().isoformat(),
            "auto_adjustments_enabled": self.auto_adjustments,
            "active_alerts": len([a for a in self.alerts if not a.expires_at or a.expires_at > datetime.now()]),
            "total_adjustments": len(self.adjustments),
            "current_parameters": self.current_params.copy(),
            "recent_alerts": [
                {
                    "type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alerts[-5:]  # Last 5 alerts
            ]
        }
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get summary of all alerts"""
        active_alerts = [a for a in self.alerts if not a.expires_at or a.expires_at > datetime.now()]
        
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "alerts_by_type": {
                alert_type.value: len([a for a in active_alerts if a.alert_type == alert_type])
                for alert_type in AlertType
            },
            "alerts_by_severity": {
                severity.value: len([a for a in active_alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            "recent_alerts": [
                {
                    "id": alert.alert_id,
                    "type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat(),
                    "confidence": alert.confidence_score
                }
                for alert in active_alerts[-10:]  # Last 10 active alerts
            ]
        }


# Convenience functions for API access
async def process_market_signals() -> Dict[str, Any]:
    """Process market signals and return results"""
    engine = MarketIntelligenceEngine()
    new_alerts = await engine.process_market_signals()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "new_alerts_count": len(new_alerts),
        "alerts": [
            {
                "id": alert.alert_id,
                "type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "confidence": alert.confidence_score,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in new_alerts
        ]
    }


def get_engine_status() -> Dict[str, Any]:
    """Get current engine status"""
    engine = MarketIntelligenceEngine()
    return engine.get_current_status()


def get_alerts_summary() -> Dict[str, Any]:
    """Get alerts summary"""
    engine = MarketIntelligenceEngine()
    return engine.get_alerts_summary()
