"""JobAdder OAuth2 authentication."""

from datetime import datetime, timedelta
import os
import requests
from app.simple_logger import get_logger

logger = get_logger("jobadder_auth")


class JobAdderAuth:
    """OAuth2 authentication for JobAdder API."""

    def __init__(self, client_id, client_secret, scope=None):
        self.client_id = client_id
        self.client_secret = client_secret
        
        # Determine environment (sandbox vs production)
        # If JOBADDER_ENVIRONMENT is set, use it; otherwise default to production
        environment = os.getenv("JOBADDER_ENVIRONMENT", "production").lower()
        
        if environment == "sandbox":
            base_url = "https://id-sandbox.jobadder.com"
            api_base_url = os.getenv("JOBADDER_API_BASE_URL", "https://api-sandbox.jobadder.com/v2")
        else:
            base_url = "https://id.jobadder.com"
            api_base_url = os.getenv("JOBADDER_API_BASE_URL", "https://api.jobadder.com/v2")
        
        # JobAdder uses /oauth2/authorize and /oauth2/token endpoints
        # Allow override via environment variables for custom setups
        self.authorize_url = os.getenv("JOBADDER_AUTHORIZE_URL", f"{base_url}/oauth2/authorize")
        self.token_url = os.getenv("JOBADDER_TOKEN_URL", f"{base_url}/oauth2/token")
        self.api_base_url = api_base_url
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        # Always use 'read write offline_access' - never use 'jobadder.api'
        # Ensure scope is space-separated and will be URL-encoded (%20)
        default_scope = "read write offline_access"
        self.scope = scope or os.getenv("JOBADDER_DEFAULT_SCOPE", default_scope)
        # Validate scope - reject if it contains jobadder.api
        if "jobadder.api" in self.scope.lower():
            logger.warning("Invalid scope detected: %s. Using default 'read write offline_access'", self.scope)
            self.scope = default_scope

    def exchange_authorization_code(self, code, redirect_uri):
        """Exchange authorization code for access & refresh tokens."""
        try:
            response = requests.post(
                self.token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                expires_in = data.get("expires_in", 3600)
                self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                return True, data

            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error_description", f"HTTP {response.status_code}")
            logger.error("Authorization code exchange failed: %s", error_msg)
            return False, error_msg

        except requests.exceptions.RequestException as exc:
            logger.error("Token exchange request failed: %s", exc)
            return False, f"Connection failed: {str(exc)}"
        except Exception as exc:
            logger.error("Authorization code exchange error: %s", exc)
            return False, str(exc)

    def refresh_access_token(self, refresh_token):
        """Refresh access token using refresh_token grant."""
        try:
            response = requests.post(
                self.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token", refresh_token)
                expires_in = data.get("expires_in", 3600)
                self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                return True, data

            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error_description", f"HTTP {response.status_code}")
            logger.error("Refresh token request failed: %s", error_msg)
            return False, error_msg

        except requests.exceptions.RequestException as exc:
            logger.error("Refresh token request failed: %s", exc)
            return False, f"Connection failed: {str(exc)}"
        except Exception as exc:
            logger.error("Refresh token error: %s", exc)
            return False, str(exc)

    def get_account_info(self, access_token):
        """Fetch account information using access token."""
        try:
            # Try /users/me endpoint first (standard JobAdder API endpoint)
            url = f"{self.api_base_url}/users/me"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return True, data
            
            # Fallback to minimal account info
            logger.warning("Could not fetch account info from /users/me")
            return True, {
                "name": "JobAdder Account",
                "email": None,
                "userId": None,
                "companyId": None,
            }

        except Exception as e:
            logger.error("Error fetching account info: %s", e)
            return False, str(e)

    def validate_credentials(self):
        """
        Legacy helper retained for backwards compatibility.
        Attempts a client credentials grant and account lookup.
        """
        try:
            response = requests.post(
                self.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": self.scope,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                access_token = data.get("access_token")
                success, account_info = self.get_account_info(access_token)
                if not success:
                    return False, account_info
                return True, {
                    "access_token": access_token,
                    "account_info": account_info,
                }

            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error_description", f"HTTP {response.status_code}")
            logger.error("Validation token request failed: %s", error_msg)
            return False, error_msg

        except Exception as exc:
            logger.error("Validation error: %s", exc)
            return False, str(exc)
