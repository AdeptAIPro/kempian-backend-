from __future__ import annotations
from typing import Dict
from transformers import pipeline

class EmotionAnalyzer:
    """
    GoEmotions-based emotion distribution.
    We map emotions into behavioral signals (confidence, stress, positivity, empathy).
    Model: joeddav/distilbert-base-uncased-go-emotions-student
    """

    MODEL = "joeddav/distilbert-base-uncased-go-emotions-student"

    # Heuristic mapping from emotions → behavioral factors
    MAP = {
        "confidence": ["pride", "admiration", "approval"],
        "stress": ["nervousness", "fear", "disappointment", "remorse"],
        "positivity": ["joy", "fun", "desire", "optimism", "relief", "gratitude"],
        "empathy": ["caring", "love", "amusement", "curiosity", "realization"]
    }

    def __init__(self, device: int | str | None = None):
        self.clf = pipeline(
            task="text-classification",
            model=self.MODEL,
            top_k=None,
            device=device
        )

    def analyze(self, text: str) -> Dict[str, float]:
        res = self.clf(text)
        # res: List[List[{'label': 'joy', 'score': 0.9}, ...]]
        dist = {r["label"]: float(r["score"]) for r in res[0]}
        # Aggregate into behavioral factors
        factors = {}
        for k, labels in self.MAP.items():
            factors[k] = float(sum(dist.get(lbl, 0.0) for lbl in labels))
        # Normalize to [0,1] by clipping (heuristic; scores are already 0..1-ish)
        for k in factors:
            factors[k] = min(1.0, factors[k])
        return {
            "emotions": dist,
            "factors": factors
        }
