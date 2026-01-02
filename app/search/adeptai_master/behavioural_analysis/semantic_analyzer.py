from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util

class SemanticAnalyzer:
    """
    Semantic signals using Sentence-BERT:
      • Resume ↔ JD similarity
      • Achievement ↔ Requirement alignment
      • Temporal progression: semantic shift of role titles over time
      • Collaboration/Leadership/Innovation proxies via semantic match to exemplars (no keywords)
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    EXEMPLARS = {
        "leadership": [
            "led cross-functional teams to deliver outcomes",
            "mentored engineers and owned delivery",
            "influenced stakeholders and drove strategy"
        ],
        "collaboration": [
            "worked closely with design, product and engineering",
            "facilitated cross-team communication and coordination",
            "paired programming and knowledge sharing"
        ],
        "innovation": [
            "prototyped and iterated novel solutions",
            "introduced new techniques and improved systems",
            "designed scalable architectures for new problems"
        ],
        "adaptability": [
            "learned new technologies quickly to meet goals",
            "adapted to changing requirements and constraints",
            "thrived in ambiguous and dynamic environments"
        ],
    }

    def __init__(self, model_name: str = None, device: str | None = None):
        self.model = SentenceTransformer(model_name or self.DEFAULT_MODEL, device=device)

        # Cache exemplar embeddings
        self._exemplar_embs = {
            k: self.model.encode(v, convert_to_tensor=True, normalize_embeddings=True)
            for k, v in self.EXEMPLARS.items()
        }

    def encode(self, texts: List[str]):
        return self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def cosine(self, a, b) -> float:
        return float(util.cos_sim(a, b).mean().item())

    def resume_jd_similarity(self, resume_text: str, job_text: str) -> float:
        e1 = self.encode([resume_text])[0]
        e2 = self.encode([job_text])[0]
        return float(util.cos_sim(e1, e2).item())

    def analyze_progression(self, roles: List[str]) -> float:
        """Higher score when later roles differ meaningfully from earlier roles (growth/trajectory)."""
        roles = [r for r in roles if r.strip()]
        if len(roles) < 2:
            return 0.5
        embs = self.encode(roles)
        sims = []
        for i in range(len(embs) - 1):
            sims.append(float(util.cos_sim(embs[i], embs[i + 1]).item()))
        mean_sim = float(np.mean(sims))
        return float(1.0 - mean_sim)  # larger shift → more progression

    def exemplar_alignment(self, text: str) -> Dict[str, float]:
        """
        Alignment to behavioral exemplars (semantic, no keyword checks).
        Returns normalized cosine score per behavioral dimension.
        """
        t = self.encode([text])
        scores: Dict[str, float] = {}
        for dim, bank in self._exemplar_embs.items():
            # Many-to-one similarity: average similarity against exemplar set
            sim = util.cos_sim(t, bank).mean().item()
            scores[dim] = float(sim)
        return scores

    def segment_alignment(self, segments: List[str], job_text: str) -> List[Tuple[str, float]]:
        """Rank resume bullet points/segments by semantic alignment to the job description."""
        if not segments:
            return []
        seg_emb = self.encode(segments)
        job_emb = self.encode([job_text])
        sims = util.cos_sim(seg_emb, job_emb).cpu().numpy().reshape(-1)
        return sorted(zip(segments, sims.tolist()), key=lambda x: x[1], reverse=True)
