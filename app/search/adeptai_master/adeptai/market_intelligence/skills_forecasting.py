"""
Skills Demand Forecasting module

Data sources (simulated connectors):
- Job postings: Indeed, LinkedIn, Stack Overflow Jobs
- Technology adoption: GitHub trends, Stack Overflow surveys
- Industry reports: sector predictions
- Course enrollments: Coursera, Udemy

Forecasting: simple linear regression on composite demand index
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from .models import IndustryType, SkillCategory


logger = logging.getLogger(__name__)


@dataclass
class SkillTimePoint:
    timestamp: datetime
    job_postings: int
    github_trend: float
    so_trend: float
    course_enrollments: int
    industry_signal: float

    def composite_index(self) -> float:
        # Weighted composite demand index
        # jobs 40%, github 20%, stackoverflow 15%, courses 15%, industry 10%
        jobs_norm = self.job_postings
        gh_norm = self.github_trend
        so_norm = self.so_trend
        course_norm = self.course_enrollments
        ind_norm = self.industry_signal
        return (
            0.4 * jobs_norm +
            0.2 * gh_norm +
            0.15 * so_norm +
            0.15 * course_norm +
            0.1 * ind_norm
        )


@dataclass
class SkillForecast:
    skill: str
    category: SkillCategory
    industry: Optional[IndustryType]
    history: List[SkillTimePoint] = field(default_factory=list)
    demand_index_current: float = 0.0
    forecast_1m: float = 0.0
    forecast_3m: float = 0.0
    forecast_6m: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "skill": self.skill,
            "category": self.category.value,
            "industry": self.industry.value if self.industry else None,
            "demand_index_current": self.demand_index_current,
            "forecast": {
                "1_month": self.forecast_1m,
                "3_months": self.forecast_3m,
                "6_months": self.forecast_6m
            },
            "confidence": self.confidence
        }


class SkillDataCollectors:
    """Simulated async collectors for multiple sources."""

    def __init__(self):
        self.rate_delay = 0.1

    async def collect_history(self, skill: str, months: int = 12) -> List[SkillTimePoint]:
        now = datetime.now()
        points: List[SkillTimePoint] = []
        base_jobs = self._base_jobs(skill)
        base_gh = self._base_github(skill)
        base_so = self._base_so(skill)
        base_course = self._base_course(skill)
        base_ind = self._base_industry_signal(skill)

        for i in range(months, 0, -1):
            ts = now - timedelta(days=30 * i)
            noise = lambda s: s * random.uniform(-0.08, 0.08)
            trend = 1.0 + (0.02 if self._is_emerging(skill) else -0.005) * (months - i)
            points.append(
                SkillTimePoint(
                    timestamp=ts,
                    job_postings=max(0, int((base_jobs * trend) + noise(base_jobs))),
                    github_trend=max(0.0, (base_gh * trend) + noise(base_gh)),
                    so_trend=max(0.0, (base_so * trend) + noise(base_so)),
                    course_enrollments=max(0, int((base_course * trend) + noise(base_course))),
                    industry_signal=max(0.0, (base_ind * trend) + noise(base_ind)),
                )
            )
            await asyncio.sleep(self.rate_delay)

        return points

    def _is_emerging(self, skill: str) -> bool:
        emerging = {"kubernetes", "rust", "gpt", "pytorch", "terraform", "snowflake"}
        return skill.strip().lower() in emerging

    def _base_jobs(self, skill: str) -> int:
        defaults = {"python": 1800, "javascript": 2200, "java": 2000, "sql": 2500}
        return defaults.get(skill.lower(), 800)

    def _base_github(self, skill: str) -> float:
        defaults = {"python": 95.0, "javascript": 90.0, "java": 70.0, "rust": 55.0}
        return defaults.get(skill.lower(), 40.0)

    def _base_so(self, skill: str) -> float:
        defaults = {"python": 85.0, "javascript": 88.0, "java": 72.0, "kubernetes": 60.0}
        return defaults.get(skill.lower(), 45.0)

    def _base_course(self, skill: str) -> int:
        defaults = {"python": 12000, "machine learning": 8000, "aws": 6000, "sql": 10000}
        return defaults.get(skill.lower(), 3000)

    def _base_industry_signal(self, skill: str) -> float:
        defaults = {"ai": 80.0, "machine learning": 78.0, "cloud": 70.0}
        return defaults.get(skill.lower(), 50.0)


class SkillsForecaster:
    def __init__(self):
        self.collectors = SkillDataCollectors()

    async def forecast_skills(
        self,
        skills: List[str],
        industries: Optional[List[IndustryType]] = None,
        months_history: int = 12,
    ) -> List[SkillForecast]:
        forecasts: List[SkillForecast] = []
        industries = industries or [None]

        for skill in skills:
            try:
                history = await self.collectors.collect_history(skill, months_history)
                demand_series = [tp.composite_index() for tp in history]
                demand_current = demand_series[-1] if demand_series else 0.0
                f1, f3, f6, conf = self._make_forecast(history, demand_series)
                forecasts.append(
                    SkillForecast(
                        skill=skill,
                        category=self._categorize_skill(skill),
                        industry=industries[0] if industries and industries[0] else None,
                        history=history,
                        demand_index_current=float(demand_current),
                        forecast_1m=float(f1),
                        forecast_3m=float(f3),
                        forecast_6m=float(f6),
                        confidence=float(conf),
                    )
                )
            except Exception as e:
                logger.warning("Forecast failed for %s: %s", skill, e)
                continue

        return forecasts

    def _make_forecast(self, history: List[SkillTimePoint], series: List[float]) -> Tuple[float, float, float, float]:
        if len(series) < 4:
            v = series[-1] if series else 0.0
            return v, v, v, 0.3

        x = np.arange(len(series)).reshape(-1, 1)
        y = np.array(series)
        model = LinearRegression()
        model.fit(x, y)

        future = np.array([len(series) + k for k in [1, 3, 6]]).reshape(-1, 1)
        preds = model.predict(future)

        # Confidence: normalized R^2 and signal-to-noise
        y_hat = model.predict(x)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
        r2 = max(0.0, min(1.0, 1 - ss_res / ss_tot))
        snr = float(np.var(y) / (np.var(y - y_hat) + 1e-6))
        conf = max(0.1, min(0.99, 0.5 * r2 + 0.5 * (np.tanh(snr / 10))))

        return preds[0], preds[1], preds[2], conf

    def _categorize_skill(self, skill: str) -> SkillCategory:
        tech = {"python", "javascript", "java", "sql", "kubernetes", "docker", "aws", "pytorch", "tensorflow", "rust"}
        soft = {"leadership", "communication", "agile"}
        emerging = {"kubernetes", "pytorch", "rust", "terraform", "gpt"}
        s = skill.strip().lower()
        if s in emerging:
            return SkillCategory.EMERGING
        if s in tech:
            return SkillCategory.TECHNICAL
        if s in soft:
            return SkillCategory.SOFT
        return SkillCategory.DOMAIN_SPECIFIC


