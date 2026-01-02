"""Main pipeline orchestrator for candidate-job matching."""

from .matcher import match_candidates, MatchResult, CandidateJobMatcher

__all__ = [
    'match_candidates',
    'MatchResult',
    'CandidateJobMatcher',
]

