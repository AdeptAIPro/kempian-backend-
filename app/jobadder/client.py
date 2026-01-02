"""Thin JobAdder API client used by routes and background jobs."""

from __future__ import annotations

from datetime import datetime, timedelta
import base64
import json
import os
from typing import Any, Dict, Optional

import requests
import time
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
        self.scope = os.getenv("JOBADDER_DEFAULT_SCOPE", "read write offline_access")

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

        # Handle rate limiting (429 Too Many Requests)
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    wait_seconds = int(retry_after)
                    logger.warning(
                        "JobAdder API rate limit exceeded. Waiting %d seconds as per Retry-After header",
                        wait_seconds
                    )
                    time.sleep(wait_seconds)
                    # Retry the request once after waiting
                    response = requests.request(
                        method,
                        url,
                        headers=headers,
                        params=params,
                        data=data,
                        timeout=30,
                    )
                except (ValueError, TypeError):
                    logger.error("Invalid Retry-After header value: %s", retry_after)
            else:
                # No Retry-After header, use exponential backoff (default 60 seconds)
                logger.warning(
                    "JobAdder API rate limit exceeded but no Retry-After header. Waiting 60 seconds."
                )
                time.sleep(60)
                # Retry the request once after waiting
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

    # Companies
    def get_companies(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/companies", params=params)

    def get_company(self, company_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/companies/{company_id}")

    # Contacts
    def get_contacts(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/contacts", params=params)

    def get_contact(self, contact_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/contacts/{contact_id}")

    # Placements
    def get_placements(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/placements", params=params)

    def get_placement(self, placement_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/placements/{placement_id}")

    # Notes
    def get_notes(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/notes", params=params)

    def get_note(self, note_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/notes/{note_id}")

    # Activities
    def get_activities(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/activities", params=params)

    def get_activity(self, activity_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/activities/{activity_id}")

    # Tasks
    def get_tasks(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/tasks", params=params)

    def get_task(self, task_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/tasks/{task_id}")

    # Users
    def get_users(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/users", params=params)

    def get_user(self, user_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/users/{user_id}")

    # Workflows
    def get_workflows(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/workflows", params=params)

    def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/workflows/{workflow_id}")

    # Custom Fields
    def get_custom_fields(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/customfields", params=params)

    def get_custom_field(self, custom_field_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/customfields/{custom_field_id}")

    # Requisitions
    def get_requisitions(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/requisitions", params=params)

    def get_requisition(self, requisition_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/requisitions/{requisition_id}")

    # Job Boards
    def get_job_boards(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/jobboards", params=params)

    def get_job_board(self, job_board_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/jobboards/{job_board_id}")

    # Write Operations - Jobs
    def create_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/jobs", payload=payload)

    def update_job(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/jobs/{job_id}", payload=payload)

    def delete_job(self, job_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/jobs/{job_id}")

    # Write Operations - Candidates
    def create_candidate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/candidates", payload=payload)

    def update_candidate(self, candidate_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/candidates/{candidate_id}", payload=payload)

    def delete_candidate(self, candidate_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/candidates/{candidate_id}")

    # Write Operations - Applications
    def create_application(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/applications", payload=payload)

    def update_application(self, application_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/applications/{application_id}", payload=payload)

    def delete_application(self, application_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/applications/{application_id}")

    # Write Operations - Companies
    def create_company(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/companies", payload=payload)

    def update_company(self, company_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/companies/{company_id}", payload=payload)

    def delete_company(self, company_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/companies/{company_id}")

    # Write Operations - Contacts
    def create_contact(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/contacts", payload=payload)

    def update_contact(self, contact_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/contacts/{contact_id}", payload=payload)

    def delete_contact(self, contact_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/contacts/{contact_id}")

    # Write Operations - Placements
    def create_placement(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/placements", payload=payload)

    def update_placement(self, placement_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/placements/{placement_id}", payload=payload)

    def delete_placement(self, placement_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/placements/{placement_id}")

    # Write Operations - Notes
    def create_note(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/notes", payload=payload)

    def update_note(self, note_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/notes/{note_id}", payload=payload)

    def delete_note(self, note_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/notes/{note_id}")

    # Write Operations - Activities
    def create_activity(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/activities", payload=payload)

    def update_activity(self, activity_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/activities/{activity_id}", payload=payload)

    def delete_activity(self, activity_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/activities/{activity_id}")

    # Write Operations - Tasks
    def create_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/tasks", payload=payload)

    def update_task(self, task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/tasks/{task_id}", payload=payload)

    def delete_task(self, task_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/tasks/{task_id}")

    # Webhooks
    def get_webhooks(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/webhooks", params=params)

    def get_webhook(self, webhook_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/webhooks/{webhook_id}")

    def create_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/webhooks", payload=payload)

    def update_webhook(self, webhook_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/webhooks/{webhook_id}", payload=payload)

    def delete_webhook(self, webhook_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/webhooks/{webhook_id}")

    # Partner Action Buttons
    def get_partner_action_buttons(self, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/partneractionbuttons", params=params)

    def get_partner_action_button(self, button_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/partneractionbuttons/{button_id}")

    def create_partner_action_button(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/partneractionbuttons", payload=payload)

    def update_partner_action_button(self, button_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", f"/partneractionbuttons/{button_id}", payload=payload)

    def delete_partner_action_button(self, button_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/partneractionbuttons/{button_id}")

    # File Operations
    def upload_file(self, resource_type: str, resource_id: str, file_path: str, file_data: bytes, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """Upload a file attachment to a resource."""
        token = self._ensure_token()
        url = f"{self.api_base_url}/{resource_type}/{resource_id}/attachments"
        headers = {
            "Authorization": f"Bearer {token}",
        }
        
        files = {
            'file': (file_path, file_data, content_type)
        }
        
        response = requests.post(url, headers=headers, files=files, timeout=60)
        
        if response.status_code >= 400:
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            raise JobAdderAPIError(
                f"JobAdder API error ({response.status_code}): {error_body}"
            )
        
        return response.json() if response.content else {}

    def get_file(self, resource_type: str, resource_id: str, attachment_id: str) -> bytes:
        """Download a file attachment from a resource."""
        token = self._ensure_token()
        url = f"{self.api_base_url}/{resource_type}/{resource_id}/attachments/{attachment_id}"
        headers = {
            "Authorization": f"Bearer {token}",
        }
        
        response = requests.get(url, headers=headers, timeout=60)
        
        if response.status_code >= 400:
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            raise JobAdderAPIError(
                f"JobAdder API error ({response.status_code}): {error_body}"
            )
        
        return response.content

    def delete_file(self, resource_type: str, resource_id: str, attachment_id: str) -> Dict[str, Any]:
        """Delete a file attachment from a resource."""
        return self._request("DELETE", f"/{resource_type}/{resource_id}/attachments/{attachment_id}")

