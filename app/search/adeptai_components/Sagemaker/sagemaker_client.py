"""
Unified SageMaker client wrappers used across the AdeptAI stack.

These lightweight adapters keep the public API that the rest of the codebase
expects (domain classifier, LTR, embedding, etc.) while degrading gracefully
when boto3 or the configured endpoints are unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # pragma: no cover - boto3 not installed in some environments
    boto3 = None  # type: ignore
    BotoCoreError = ClientError = Exception  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_REGION = os.getenv("SAGEMAKER_REGION", os.getenv("AWS_REGION", "us-east-1"))


class SageMakerClientError(RuntimeError):
    """Raised when a SageMaker invocation fails."""


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class BaseSageMakerClient:
    """Shared utilities for specific SageMaker service adapters."""

    def __init__(self, endpoint_name: Optional[str], region: Optional[str] = None, timeout: int = 30, operation_type: Optional[str] = None):
        # Use unified endpoint if available, otherwise fall back to specific endpoint
        unified_endpoint = os.getenv("SAGEMAKER_UNIFIED_ENDPOINT", "unified-ml-pipeline-endpoint")
        self.endpoint_name = unified_endpoint if unified_endpoint else endpoint_name
        self.region = region or DEFAULT_REGION
        self.timeout = timeout
        self.operation_type = operation_type
        self._runtime = None

        if boto3 and self.endpoint_name:
            try:
                self._runtime = boto3.client("sagemaker-runtime", region_name=self.region)
            except Exception as exc:  # pragma: no cover - boto3 misconfiguration
                logger.warning("Failed to create SageMaker runtime client: %s", exc)
                self._runtime = None

    @property
    def available(self) -> bool:
        return bool(self._runtime and self.endpoint_name)

    def _invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the configured endpoint and return parsed JSON."""
        if not self.endpoint_name:
            raise SageMakerClientError("SageMaker endpoint name is not configured.")
        if not self._runtime:
            raise SageMakerClientError("SageMaker runtime client is not available (boto3 missing?).")

        # Add operation type to payload for unified endpoint routing
        if self.operation_type:
            payload["operation"] = self.operation_type

        try:
            response = self._runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload).encode("utf-8"),
            )
            body = response.get("Body")
            if body is None:
                return {}
            response_str = body.read().decode("utf-8") if hasattr(body, "read") else str(body)
            if not response_str:
                return {}
            try:
                return json.loads(response_str)
            except json.JSONDecodeError:
                logger.debug("Received non-JSON response from %s: %s", self.endpoint_name, response_str)
                return {"raw": response_str}
        except (BotoCoreError, ClientError, ValueError) as exc:
            raise SageMakerClientError(f"SageMaker invocation failed: {exc}") from exc


class SageMakerDomainClassifierClient(BaseSageMakerClient):
    """Wrapper around a SageMaker endpoint that classifies domains."""

    def __init__(self, endpoint_name: Optional[str] = None, region: Optional[str] = None, timeout: int = 30):
        super().__init__(endpoint_name, region, timeout, operation_type="classify_domain")

    def classify_domain(self, text: str) -> Tuple[str, float]:
        if not text:
            return "unknown", 0.0

        payload = {"text": text}
        try:
            result = self._invoke(payload)
        except SageMakerClientError as exc:
            logger.warning("Domain classifier invocation failed: %s", exc)
            return "unknown", 0.0

        # Handle multiple possible response formats
        if isinstance(result, dict):
            if "domain" in result:
                return str(result.get("domain") or "unknown"), _safe_float(result.get("confidence"))
            predictions = result.get("predictions") or result.get("labels")
            if isinstance(predictions, list) and predictions:
                first = predictions[0]
                if isinstance(first, dict):
                    domain = first.get("domain") or first.get("label") or first.get("class")
                    score = _safe_float(first.get("confidence") or first.get("score"))
                    return str(domain or "unknown"), score
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    return str(first[0]), _safe_float(first[1])
        return "unknown", 0.0

    def should_filter_candidate(
        self,
        candidate_domain: str,
        query_domain: str,
        candidate_confidence: float,
        query_confidence: float,
    ) -> bool:
        """
        Mirror the filtering logic expected by the search system when using SageMaker.

        We keep the behaviour consistent with the local ML classifier so the rest of
        the pipeline can call this method regardless of which implementation backs it.
        """
        candidate_domain = (candidate_domain or "unknown").lower()
        query_domain = (query_domain or "unknown").lower()

        if candidate_domain == "unknown" or query_domain == "unknown":
            return False

        if candidate_domain == query_domain:
            return False

        if candidate_confidence > 0.8 and query_confidence > 0.8 and candidate_domain != query_domain:
            return True

        if candidate_confidence > 0.9 and candidate_domain != query_domain:
            return True

        if candidate_confidence < 0.5 and query_confidence < 0.5:
            return False

        return False


class SageMakerLTRClient(BaseSageMakerClient):
    """Learning-to-Rank adapter."""

    def __init__(self, endpoint_name: Optional[str] = None, region: Optional[str] = None, timeout: int = 30):
        super().__init__(endpoint_name, region, timeout, operation_type="ltr_rank")

    def rank_candidates(
        self,
        query: str,
        candidates: Iterable[Dict[str, Any]],
        feature_scores: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        payload = {
            "query": query,
            "candidates": list(candidates),
            "features": feature_scores or {},
        }
        try:
            result = self._invoke(payload)
        except SageMakerClientError as exc:
            logger.warning("LTR invocation failed: %s", exc)
            return []

        if isinstance(result, dict):
            if "ranked_candidates" in result:
                return result["ranked_candidates"]
            if "scores" in result:
                return result["scores"]
            if "predictions" in result:
                return result["predictions"]
        if isinstance(result, list):
            return result
        return []


class SageMakerLLMQueryEnhancerClient(BaseSageMakerClient):
    """Query enhancement adapter used for LLM-powered expansion."""

    def __init__(self, endpoint_name: Optional[str] = None, region: Optional[str] = None, timeout: int = 30):
        super().__init__(endpoint_name, region, timeout, operation_type="enhance_query")

    def enhance_query(self, query: str, use_llm: bool = True) -> Dict[str, Any]:
        payload = {"query": query, "use_llm": use_llm}
        try:
            result = self._invoke(payload)
        except SageMakerClientError as exc:
            logger.warning("Query enhancer invocation failed: %s", exc)
            return {"expanded_terms": [], "variations": [], "metadata": {"error": str(exc)}}

        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"expanded_terms": result, "variations": result}
        if isinstance(result, str):
            return {"expanded_terms": [result], "variations": [result]}
        return {"expanded_terms": [], "variations": []}


class SageMakerJobFitPredictorClient(BaseSageMakerClient):
    """Adapter for job fit predictions."""

    def __init__(self, endpoint_name: Optional[str] = None, region: Optional[str] = None, timeout: int = 30):
        super().__init__(endpoint_name, region, timeout, operation_type="predict_job_fit")

    def predict_fit(self, candidate: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"candidate": candidate, "job": job}
        try:
            result = self._invoke(payload)
        except SageMakerClientError as exc:
            logger.warning("Job fit predictor invocation failed: %s", exc)
            return {"fit_score": 0.0, "reasons": [], "error": str(exc)}

        if isinstance(result, dict):
            return result
        return {"fit_score": 0.0, "reasons": [], "raw": result}


class SageMakerDenseRetrievalClient(BaseSageMakerClient):
    """Adapter for dense retrieval search."""

    def __init__(self, endpoint_name: Optional[str] = None, region: Optional[str] = None, timeout: int = 30):
        super().__init__(endpoint_name, region, timeout, operation_type="dense_retrieval")

    def search(self, query: str, top_k: int = 20) -> List[Any]:
        payload = {"query": query, "top_k": top_k}
        try:
            result = self._invoke(payload)
        except SageMakerClientError as exc:
            logger.warning("Dense retrieval invocation failed: %s", exc)
            return []

        if isinstance(result, dict):
            matches = result.get("matches") or result.get("results")
            if isinstance(matches, list):
                return matches[:top_k]
        if isinstance(result, list):
            return result[:top_k]
        return []


class SageMakerEmbeddingClient(BaseSageMakerClient):
    """Adapter for embedding generation."""

    def __init__(self, endpoint_name: Optional[str] = None, region: Optional[str] = None, timeout: int = 30):
        super().__init__(endpoint_name, region, timeout, operation_type="generate_embedding")

    def encode(self, text: str, model_type: str = "general") -> List[float]:
        payload = {"text": text, "model_type": model_type}
        try:
            result = self._invoke(payload)
        except SageMakerClientError as exc:
            logger.warning("Embedding invocation failed: %s", exc)
            return []

        vector = None
        if isinstance(result, dict):
            vector = result.get("embedding") or result.get("vector") or result.get("embeddings")
        elif isinstance(result, list):
            vector = result

        if isinstance(vector, list):
            return [float(v) for v in vector if isinstance(v, (int, float))]
        return []


@dataclass
class SageMakerClientConfig:
    """Simple container for client configuration."""

    unified_endpoint: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SageMakerClientConfig":
        return cls(
            unified_endpoint=data.get("unified_endpoint"),
        )


class SageMakerClientManager:
    """Factory/manager for SageMaker client adapters."""

    def __init__(self, region: Optional[str] = None, config: Optional[SageMakerClientConfig] = None):
        self.region = region or DEFAULT_REGION
        self.config = config or SageMakerClientConfig.from_dict({})

    @classmethod
    def from_config(cls, config: Dict[str, Any], region: Optional[str] = None) -> "SageMakerClientManager":
        return cls(region=region, config=SageMakerClientConfig.from_dict(config or {}))

    def create_domain_classifier(self) -> SageMakerDomainClassifierClient:
        # All clients now use unified endpoint
        return SageMakerDomainClassifierClient(None, region=self.region)

    def create_ltr_client(self) -> SageMakerLTRClient:
        return SageMakerLTRClient(None, region=self.region)

    def create_llm_enhancer(self) -> SageMakerLLMQueryEnhancerClient:
        return SageMakerLLMQueryEnhancerClient(None, region=self.region)

    def create_job_fit_predictor(self) -> SageMakerJobFitPredictorClient:
        return SageMakerJobFitPredictorClient(None, region=self.region)

    def create_dense_retrieval_client(self) -> SageMakerDenseRetrievalClient:
        return SageMakerDenseRetrievalClient(None, region=self.region)

    def create_embedding_client(self) -> SageMakerEmbeddingClient:
        return SageMakerEmbeddingClient(None, region=self.region)


__all__ = [
    "SageMakerClientError",
    "SageMakerClientManager",
    "SageMakerDomainClassifierClient",
    "SageMakerLTRClient",
    "SageMakerLLMQueryEnhancerClient",
    "SageMakerJobFitPredictorClient",
    "SageMakerDenseRetrievalClient",
    "SageMakerEmbeddingClient",
    "SageMakerClientConfig",
]


