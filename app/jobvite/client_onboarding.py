"""
Jobvite Onboarding API Client
Handles encrypted requests/responses (RSA + AES).
"""

import requests
import os
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
                 service_account_password: Optional[str] = None,
                 use_singular_paths: Optional[bool] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')
        self.our_private_key_pem = our_private_key_pem
        self.jobvite_public_key_pem = jobvite_public_key_pem
        # Service account takes precedence if provided (for Onboarding API)
        self.service_account_username = service_account_username
        self.service_account_password = service_account_password
        
        # Support both singular (/process) and plural (/processes) paths per PDF spec
        # Default to singular if env var is set, otherwise use plural for backward compatibility
        if use_singular_paths is None:
            use_singular_paths = os.getenv('JOBVITE_USE_SINGULAR_ONBOARDING_PATHS', 'false').lower() == 'true'
        self.use_singular_paths = use_singular_paths
    
    def _get_endpoint_path(self, name: str) -> str:
        """
        Get endpoint path, supporting both singular and plural forms.
        
        Args:
            name: Endpoint name ('process', 'task', or 'milestone')
        
        Returns:
            Endpoint path ('/process' or '/processes', etc.)
        """
        if self.use_singular_paths:
            return f"/{name}"
        else:
            return f"/{name}s"
    
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
        except requests.exceptions.ConnectionError as e:
            error_msg = str(e)
            logger.error(f"Onboarding API connection error: {error_msg}")
            # Check if it's a DNS resolution error
            if 'Failed to resolve' in error_msg or 'getaddrinfo failed' in error_msg or 'NameResolutionError' in error_msg:
                raise ValueError(
                    f"Cannot reach Jobvite Onboarding API server. DNS resolution failed for {self.base_url}. "
                    f"This could be a network issue or the server may not be accessible from your network."
                )
            else:
                raise ValueError(f"Network connection failed to Jobvite Onboarding API: {error_msg}")
        except requests.exceptions.Timeout as e:
            logger.error(f"Onboarding API timeout: {e}")
            raise ValueError(f"Request timeout to Jobvite Onboarding API at {self.base_url}. The server may be slow or unreachable.")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 0
            error_detail = ""
            try:
                if e.response and e.response.text:
                    error_detail = e.response.text[:200]
            except:
                pass
            
            if status_code == 401:
                raise ValueError(
                    f"Jobvite Onboarding API authentication failed (401 Unauthorized). "
                    f"Invalid service account credentials or API key/secret. "
                    f"Error: {error_detail}"
                )
            elif status_code == 403:
                raise ValueError(
                    f"Jobvite Onboarding API access denied (403 Forbidden). "
                    f"Insufficient permissions or invalid credentials. "
                    f"Error: {error_detail}"
                )
            elif status_code == 404:
                raise ValueError(
                    f"Jobvite Onboarding API endpoint not found (404). "
                    f"URL: {url}, Error: {error_detail}"
                )
            else:
                raise ValueError(f"Jobvite Onboarding API error ({status_code}): {error_detail}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Onboarding API request failed: {e}")
            raise ValueError(f"Jobvite Onboarding API request failed: {str(e)}")
    
    def get_processes(self, filters: Optional[Dict] = None,
                     start: int = 0,
                     count: int = 50,
                     default_sort_type: str = "ASC") -> Dict[str, Any]:
        """
        Get onboarding processes.
        
        Args:
            filters: Filter criteria using operators (eq, in, nin, lt, lte, gt, gte)
                    Example: {
                        "filter": {
                            "process": {
                                "hireDate": {"gte": "2024-01-01"},
                                "status": {"in": ["active", "completed"]}
                            }
                        }
                    }
            start: Pagination start index (default: 0, must be >= 0)
            count: Page size (default: 50, typically max: 100)
            default_sort_type: "ASC" or "DESC" (default: "ASC")
        
        Returns:
            Decrypted process data with structure:
            {
                "processes": [...],
                "total": <number>,
                "start": <number>,
                "count": <number>
            }
        
        Supported Filter Operators (per Jobvite Onboarding API docs):
        - eq: Equals
        - in: In (array of values)
        - nin: Not in
        - lt: Less than
        - lte: Less than or equal
        - gt: Greater than
        - gte: Greater than or equal
        """
        # Validate parameters
        if start < 0:
            raise ValueError("start parameter must be >= 0")
        if count < 1 or count > 1000:
            raise ValueError("count parameter must be between 1 and 1000")
        if default_sort_type not in ("ASC", "DESC"):
            raise ValueError("default_sort_type must be 'ASC' or 'DESC'")
        
        filter_json = {
            "start": start,
            "count": count,
            "defaultSortType": default_sort_type
        }
        
        # Merge filters if provided
        if filters:
            # If filters already have "filter" key, use it directly
            if "filter" in filters:
                filter_json["filter"] = filters["filter"]
            else:
                # Otherwise, wrap in "filter" structure
                filter_json["filter"] = filters
        
        endpoint = self._get_endpoint_path('process')
        return self._make_encrypted_request('POST', endpoint, filter_json)
    
    def get_tasks(self, filters: Optional[Dict] = None,
                  start: int = 0,
                  count: int = 50,
                  return_file_info: bool = False) -> Dict[str, Any]:
        """
        Get onboarding tasks.
        
        Args:
            filters: Filter criteria using operators (eq, in, nin, lt, lte, gt, gte)
                    Example: {
                        "filter": {
                            "task": {
                                "processId": {"in": ["process_id_1", "process_id_2"]},
                                "status": {"eq": "active"},
                                "type": {"in": ["W4", "I9", "DOC"]}
                            }
                        }
                    }
            start: Pagination start index (default: 0, must be >= 0)
            count: Page size (default: 50, typically max: 100)
            return_file_info: If True, include file data (base64 encoded)
        
        Returns:
            Decrypted task data with structure:
            {
                "tasks": [...],
                "total": <number>,
                "start": <number>,
                "count": <number>
            }
        
        Supported Filter Operators (per Jobvite Onboarding API docs):
        - eq: Equals
        - in: In (array of values)
        - nin: Not in
        - lt: Less than
        - lte: Less than or equal
        - gt: Greater than
        - gte: Greater than or equal
        """
        # Validate parameters
        if start < 0:
            raise ValueError("start parameter must be >= 0")
        if count < 1 or count > 1000:
            raise ValueError("count parameter must be between 1 and 1000")
        
        filter_json = {
            "start": start,
            "count": count,
            "returnFileInfo": return_file_info
        }
        
        # Merge filters if provided
        if filters:
            # If filters already have "filter" key, use it directly
            if "filter" in filters:
                filter_json["filter"] = filters["filter"]
            else:
                # Otherwise, wrap in "filter" structure
                filter_json["filter"] = filters
        
        endpoint = self._get_endpoint_path('task')
        return self._make_encrypted_request('POST', endpoint, filter_json)
    
    def set_milestone(self, process_ids: List[str], milestones: List[str], 
                     operation: str = "add", milestone_type: str = "api_retrieved") -> Dict[str, Any]:
        """
        Mark milestone(s) for process(es) as API retrieved.
        
        Per Jobvite Onboarding API documentation:
        - filter.process.id.in: List of process IDs
        - milestone.milestones: List of milestone names (e.g., ["payrollmileStone", "desksetupMileStone"])
        - milestone.operation: "set" (replace all) or "add" (append)
        - milestone.type: "api_retrieved" (required)
        
        Args:
            process_ids: List of Jobvite process IDs
            milestones: List of milestone names to mark (e.g., ["payrollmileStone"])
            operation: "set" (replace all milestones) or "add" (append to existing)
            milestone_type: Type of milestone operation (default: "api_retrieved")
        
        Returns:
            Response from milestone API with status and updated milestones
        
        Example:
            set_milestone(
                process_ids=["57b3598d00b00c3f988997b5"],
                milestones=["payrollmileStone", "desksetupMileStone"],
                operation="add"
            )
        """
        # Per Jobvite Onboarding API spec: filter structure
        filter_json = {
            "filter": {
                "process": {
                    "id": {
                        "in": process_ids
                    }
                }
            },
            "milestone": {
                "milestones": milestones,  # Note: plural "milestones" array
                "operation": operation,  # "set" or "add"
                "type": milestone_type  # Required: "api_retrieved"
            }
        }
        
        endpoint = self._get_endpoint_path('milestone')
        return self._make_encrypted_request('POST', endpoint, filter_json)

