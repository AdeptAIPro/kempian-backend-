"""
Production-Ready Candidate-Job Matchmaking System
Complete AI-based job-candidate matching engine with modular architecture.
"""

__version__ = "1.0.0"

from .pipelines.matcher import match_candidates, MatchResult
from .collectors.resume_parser import ResumeParser, ResumeData
from .collectors.job_parser import JobParser, JobData
from .scoring.scorer import MatchScore

__all__ = [
    'match_candidates',
    'MatchResult',
    'ResumeParser',
    'ResumeData',
    'JobParser',
    'JobData',
    'MatchScore',
]

