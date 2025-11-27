"""
Jobvite Onboarding API Client
Handles encrypted requests/responses (RSA + AES).
"""

import requests
from typing import Dict, List, Optional, Any
from app.jobvite.crypto import (
    encrypt_onboarding_payload,
    decrypt_onboarding_response,
    load_rsa_private_key
)
from app.simple_logger import get_logger
import time

logger = get_logger("jobvite_onboarding_client")

class JobviteOnboardingClient:
    """Client for Jobvite Onboarding API (encrypted)"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str,
                 our_private_key_pem: str, jobvite_public_key_pem: str,
                 service_account_username: Optional[str] = None,
                 service_account_password: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')
        self.our_private_key_pem = our_private_key_pem
        self.jobvite_public_key_pem = jobvite_public_key_pem
        # Service account takes precedence if provided (for Onboarding API)
        self.service_account_username = service_account_username
        self.service_account_password = service_account_password
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Generate basic auth headers (service account or API key/secret)"""
        import base64
        # Use service account if provided, otherwise fall back to API key/secret
        if self.service_account_username and self.service_account_password:
            credentials = f"{self.service_account_username}:{self.service_account_password}"
        else:
            credentials = f"{self.api_key}:{self.api_secret}"
        
        encoded = base64.b64encode(credentials.encode()).decode()
        return {
            'Authorization': f'Basic {encoded}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def _make_encrypted_request(self, method: str, endpoint: str, 
                                filter_json: Dict, retry_count: int = 0) -> Dict[str, Any]:
        """
        Make encrypted request to Onboarding API.
        
        Steps:
        1. Encrypt filter_json using RSA + AES
        2. POST encrypted payload
        3. Decrypt response
        
        Includes retry logic for rate limits.
        """
        # Step 1: Encrypt payload
        encrypted_payload = encrypt_onboarding_payload(
            filter_json,
            self.jobvite_public_key_pem
        )
        
        # Step 2: Make request
        url = f"{self.base_url}{endpoint}"
        headers = self._get_auth_headers()
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=encrypted_payload,
                timeout=30
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                if retry_count < 3:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited, retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                    return self._make_encrypted_request(method, endpoint, filter_json, retry_count + 1)
                else:
                    logger.error("Rate limit exceeded after 3 retries")
                    response.raise_for_status()
            
            response.raise_for_status()
            
            # Step 3: Decrypt response
            response_json = response.json()
            decrypted = decrypt_onboarding_response(
                response_json,
                self.our_private_key_pem
            )
            
            return decrypted
        except requests.exceptions.RequestException as e:
            logger.error(f"Onboarding API request failed: {e}")
            raise
    
    def get_processes(self, filters: Optional[Dict] = None,
                     start: int = 0,
                     count: int = 50,
                     default_sort_type: str = "ASC") -> Dict[str, Any]:
        """
        Get onboarding processes.
        
        Args:
            filters: Filter criteria (hireDate, kickoffDate, status, etc.)
            start: Pagination start
            count: Page size
            default_sort_type: "ASC" or "DESC"
        
        Returns:
            Decrypted process data
        """
        filter_json = {
            "start": start,
            "count": count,
            "defaultSortType": default_sort_type
        }
        
        if filters:
            filter_json.update(filters)
        
        return self._make_encrypted_request('POST', '/processes', filter_json)
    
    def get_tasks(self, filters: Optional[Dict] = None,
                  start: int = 0,
                  count: int = 50,
                  return_file_info: bool = False) -> Dict[str, Any]:
        """
        Get onboarding tasks.
        
        Args:
            filters: Filter criteria (processId, taskType, status, etc.)
            start: Pagination start
            count: Page size
            return_file_info: If True, include file data (base64)
        
        Returns:
            Decrypted task data
        """
        filter_json = {
            "start": start,
            "count": count,
            "returnFileInfo": return_file_info
        }
        
        if filters:
            filter_json.update(filters)
        
        return self._make_encrypted_request('POST', '/tasks', filter_json)
    
    def set_milestone(self, process_ids: List[str], milestone: str, 
                     operation: str = "set") -> Dict[str, Any]:
        """
        Mark milestone for process(es).
        
        Args:
            process_ids: List of process IDs
            milestone: Milestone name (e.g., "api_retrieved")
            operation: "set" (replace) or "add" (append)
        
        Returns:
            Response from milestone API
        """
        filter_json = {
            "processIds": process_ids,
            "milestone": milestone,
            "operation": operation
        }
        
        return self._make_encrypted_request('POST', '/milestones', filter_json)

