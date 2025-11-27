"""Thin JobAdder API client used by routes and background jobs."""

from __future__ import annotations

from datetime import datetime, timedelta
import base64
import json
import os
from typing import Any, Dict, Optional

import requests
from sqlalchemy.exc import SQLAlchemyError

from app.models import JobAdderIntegration, db
from app.simple_logger import get_logger

from .auth import JobAdderAuth

logger = get_logger("jobadder_client")


class JobAdderAPIError(Exception):
    """Raised when JobAdder API returns an error response."""


class JobAdderClient:
    """Authenticated API client for JobAdder."""

    def __init__(self, integration: JobAdderIntegration):
        if not integration:
            raise ValueError("JobAdder integration is required")

        self.integration = integration
        self.api_base_url = os.getenv("JOBADDER_API_BASE_URL", "https://api.jobadder.com/v2")
        self.scope = os.getenv("JOBADDER_DEFAULT_SCOPE", "jobadder.api offline_access")

        client_secret = self._decode_secret(integration.client_secret)
        self.auth = JobAdderAuth(integration.client_id, client_secret, scope=self.scope)

        # Hydrate auth state with any existing tokens
        self.auth.access_token = integration.access_token
        self.auth.refresh_token = integration.refresh_token
        self.auth.token_expires_at = integration.token_expires_at

    @staticmethod
    def _decode_secret(encoded_secret: str) -> str:
        if not encoded_secret:
            raise JobAdderAPIError("Missing JobAdder client secret")
        try:
            return base64.b64decode(encoded_secret.encode()).decode()
        except Exception as exc:  # pragma: no cover - defensive
            raise JobAdderAPIError("Unable to decode JobAdder client secret") from exc

    def _ensure_token(self, force_refresh: bool = False) -> str:
        should_refresh = force_refresh
        expires_at: Optional[datetime] = self.integration.token_expires_at
        token = self.integration.access_token

        if not token:
            should_refresh = True
        elif not expires_at:
            should_refresh = True
        else:
            buffer = timedelta(minutes=5)
            if datetime.utcnow() + buffer >= expires_at:
                should_refresh = True

        if should_refresh:
            refresh_token = self.integration.refresh_token
            if not refresh_token:
                raise JobAdderAPIError("Missing JobAdder refresh token. Please reconnect the integration.")

            success, result = self.auth.refresh_access_token(refresh_token)
            if not success:
                raise JobAdderAPIError(f"Failed to refresh JobAdder token: {result}")

            token = self.auth.access_token
            self.integration.access_token = token
            self.integration.refresh_token = self.auth.refresh_token
            self.integration.token_expires_at = self.auth.token_expires_at

            try:
                db.session.add(self.integration)
                db.session.commit()
            except SQLAlchemyError as exc:
                db.session.rollback()
                logger.error("Failed to persist JobAdder token refresh: %s", exc)
                raise JobAdderAPIError("Unable to persist JobAdder token refresh") from exc

        return token

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        token = self._ensure_token()
        url = f"{self.api_base_url}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload)
        else:
            data = None

        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            data=data,
            timeout=30,
        )

        if response.status_code == 401:
            # Try once more with a forced refresh
            token = self._ensure_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                data=data,
                timeout=30,
            )

        if response.status_code >= 400:
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            raise JobAdderAPIError(
                f"JobAdder API error ({response.status_code}): {error_body}"
            )

        if not response.content:
            return {}

        return response.json()

    # High-level resource helpers -------------------------------------------------

    def get_jobs(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/jobs", params=params)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}")

    def get_candidates(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/candidates", params=params)

    def get_candidate(self, candidate_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/candidates/{candidate_id}")

    def get_applications(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/applications", params=params)

    def get_application(self, application_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/applications/{application_id}")

