"""
Jobvite API v2 Client
Handles authentication and API calls.

As of April 2024, Jobvite uses simple header-based authentication:
- x-jvi-api: API key
- x-jvi-sc: Secret key

Supported Query Parameters:
- start: Pagination start index (default: 0)
- count: Number of results per page (default: 50, max: typically 100)
- requisitionId: Filter jobs by requisition ID
- applicationId: Filter candidates by application ID
- status: Filter by status (jobs/candidates)
- workflowState: Filter candidates by workflow state
- location: Filter by location
- department: Filter by department

Supported Filter Operators (for Onboarding API):
- eq: Equals
- in: In (array of values)
- nin: Not in
- lt: Less than
- lte: Less than or equal
- gt: Greater than
- gte: Greater than or equal
"""

import requests
import time
import os
from typing import Dict, List, Optional, Any
from app.simple_logger import get_logger
from app.jobvite.logging_utils import log_jobvite_request
from app.jobvite.crypto import build_jobvite_v2_hmac_headers

logger = get_logger("jobvite_v2_client")

class JobviteV2Client:
    """Client for Jobvite API v2"""
    
    def __init__(self, api_key: str, api_secret: str, company_id: str, base_url: str):
        # Validate credentials are not empty
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        if not api_secret or not api_secret.strip():
            raise ValueError("API secret cannot be empty")
        if not company_id or not company_id.strip():
            raise ValueError("Company ID cannot be empty")
        if not base_url or not base_url.strip():
            raise ValueError("Base URL cannot be empty")
        
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.company_id = company_id.strip()
        self.base_url = base_url.rstrip('/')
    
        # Log base URL for debugging (to catch wrong URLs)
        if '/api/v2' not in self.base_url and '/v2' in self.base_url:
            logger.warning(
                f"[WARNING] Base URL appears incorrect: {self.base_url}. "
                f"Expected format: https://api.jobvite.com/api/v2 (with /api/v2). "
                f"Current format missing /api. This will cause authentication failures."
            )
        logger.debug(f"JobviteV2Client initialized with base_url: {self.base_url}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Generate authentication headers per Jobvite API v2 specification.
        
        Uses HMAC authentication (X-JVI-API, X-JVI-SIGN, X-JVI-EPOCH) per official PDF spec.
        Can fall back to legacy simple headers if JOBVITE_V2_USE_HMAC=false.
        
        HMAC Formula: Base64(HMAC_SHA256(apiSecret, apiKey + "|" + epoch))
        """
        # Check if HMAC is enabled (default: true per PDF spec)
        use_hmac = os.getenv('JOBVITE_V2_USE_HMAC', 'true').lower() == 'true'
        
        if use_hmac:
            # Use HMAC authentication per official PDF specification
            return build_jobvite_v2_hmac_headers(self.api_key, self.api_secret)
        else:
            # Legacy simple header auth (for backward compatibility during transition)
            logger.warning("Using legacy simple header auth. HMAC is recommended per Jobvite PDF spec.")
            return {
                'x-jvi-api': self.api_key,
                'x-jvi-sc': self.api_secret,
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'Kempian/3.0'
            }
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     json_data: Optional[Dict] = None, retry_count: int = 0) -> requests.Response:
        """
        Make authenticated request to Jobvite API with retry logic for rate limits.
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_auth_headers()
        
        # Log request details for debugging (without sensitive data)
        logger.debug(f"Jobvite API request: {method} {url} (params: {params})")
        # Log header names (not values) for debugging
        header_names = list(headers.keys())
        logger.debug(f"Jobvite API headers: {header_names}")
        # Log masked API key (first 4 chars only)
        masked_key = self.api_key[:4] + "..." if len(self.api_key) > 4 else "****"
        logger.debug(f"Jobvite API key (masked): {masked_key}, Company ID: {self.company_id}")
        
        start_time = time.time()
        
        try:
            # Per Jobvite spec: Do NOT automatically follow redirects
            # If redirect points to non-API URL (HTML), treat as AUTH/CONFIG failure
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=30,
                allow_redirects=False  # Disable auto-follow redirects per spec
            )
            
            # Handle redirects explicitly per spec
            if response.status_code in (301, 302, 303, 307, 308):
                redirect_url = response.headers.get('Location', '')
                
                # Check if redirect is just adding query parameters (normal behavior)
                # e.g., /jobs -> /jobs?start=0&count=1
                if redirect_url and redirect_url.startswith(url.split('?')[0]):
                    # Just query params being added - this is fine, use the redirect URL
                    logger.debug(f"Jobvite API redirect adding query params: {url} -> {redirect_url}")
                    url = redirect_url
                    # Make new request with the redirect URL (which includes params)
                    response = requests.request(
                        method=method,
                        url=redirect_url if redirect_url.startswith('http') else f"{self.base_url}{redirect_url}",
                        headers=headers,
                        params=None,  # Params already in URL
                        json=json_data,
                        timeout=30,
                        allow_redirects=False
                    )
                # Check if redirect is to non-API URL (HTML page) - treat as auth failure
                elif redirect_url and ('app.jobvite.com' in redirect_url.lower() or 
                                     'admin' in redirect_url.lower() or 
                                     '.html' in redirect_url.lower()):
                    error_msg = (
                        f"Jobvite API authentication failed. "
                        f"Received {response.status_code} redirect to non-API URL: {redirect_url}. "
                        f"This indicates invalid credentials or wrong endpoint configuration. "
                        f"\n\nTroubleshooting:"
                        f"\n1. Verify API Key is correct (starts with: {masked_key})"
                        f"\n2. Verify API Secret is correct"
                        f"\n3. Verify Company ID is correct: {self.company_id}"
                        f"\n4. Ensure you're using correct environment (Production vs Stage)"
                        f"\n5. Check API credentials have required permissions"
                        f"\n\nRequest URL: {url}"
                        f"\nBase URL: {self.base_url}"
                        f"\nEndpoint: {endpoint}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    # Redirect to different API endpoint - follow it (but log warning)
                    logger.warning(f"Following redirect to different API endpoint: {redirect_url}")
                    response = requests.request(
                        method=method,
                        url=redirect_url if redirect_url.startswith('http') else f"{self.base_url}{redirect_url}",
                        headers=headers,
                        params=params,
                        json=json_data,
                        timeout=30,
                        allow_redirects=False  # Still don't auto-follow further redirects
                    )
            
            # Log response details for debugging
            logger.debug(f"Jobvite API response: Status {response.status_code}, URL: {response.url}, Headers: {dict(response.headers)}")
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Check for redirects to non-API URLs (indicates authentication failure or wrong endpoint)
            if response.url != url and response.url != url.rstrip('/'):
                # Check if redirect is to a non-API domain (e.g., app.jobvite.com instead of api.jobvite.com)
                original_domain = url.split('/')[2] if '//' in url else ''
                final_domain = response.url.split('/')[2] if '//' in response.url else ''
                
                if 'api' not in final_domain.lower() and 'api' in original_domain.lower():
                    # Redirected away from API - this is a critical error
                    # Check response for more details
                    response_preview = response.text[:500] if response.text else ""
                    error_msg = (
                        f"Jobvite API authentication failed. "
                        f"The API redirected from {url} to {response.url}, which indicates invalid credentials or wrong endpoint. "
                        f"\n\nTroubleshooting steps:"
                        f"\n1. Verify your API Key is correct (starts with: {masked_key})"
                        f"\n2. Verify your API Secret is correct"
                        f"\n3. Verify your Company ID is correct: {self.company_id}"
                        f"\n4. Ensure you're using the correct environment (Production vs Stage)"
                        f"\n5. Check that your API credentials have the necessary permissions"
                        f"\n\nBase URL: {self.base_url}"
                        f"\nEndpoint: {endpoint}"
                        f"\nResponse preview: {response_preview}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    logger.warning(f"Jobvite API redirect detected: {url} -> {response.url}")
            
            # Check if response is HTML (common when authentication fails or wrong endpoint)
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type and response.status_code == 200:
                # This usually means authentication failed or endpoint doesn't exist
                response_text_preview = response.text[:500] if response.text else ""
                error_msg = (
                    f"Jobvite API returned HTML instead of JSON. "
                    f"Request URL: {url}, Final URL: {response.url}, Status: {response.status_code}, "
                    f"Content-Type: {content_type}. "
                    f"This usually indicates: 1) Invalid API credentials (key/secret/companyId), "
                    f"2) Wrong base URL ({self.base_url}), or 3) Invalid endpoint ({endpoint}). "
                    f"Response preview: {response_text_preview}"
                )
                logger.error(error_msg)
                # Raise immediately for HTML responses - they're never valid API responses
                raise ValueError(error_msg)
            
            # Handle authentication errors per spec
            if response.status_code == 401:
                # Try to get more details from response - handle multiple error formats
                error_detail = ""
                error_code = None
                try:
                    if response.text:
                        import json
                        error_data = response.json()
                        if isinstance(error_data, dict):
                            # Format 1: {"status": {"code": 401, "messages": [...]}}
                            if 'status' in error_data:
                                status_obj = error_data['status']
                                if isinstance(status_obj, dict):
                                    messages = status_obj.get('messages', [])
                                    error_detail = '; '.join(messages) if messages else ""
                                    error_code = status_obj.get('code')
                            # Format 2: {"message": "..."}
                            elif 'message' in error_data:
                                error_detail = error_data['message']
                            # Format 3: {"error": "..."}
                            elif 'error' in error_data:
                                error_detail = str(error_data['error'])
                            # Format 4: {"error": {"code": "...", "message": "..."}}
                            elif isinstance(error_data.get('error'), dict):
                                error_obj = error_data['error']
                                error_detail = error_obj.get('message', str(error_obj))
                                error_code = error_obj.get('code')
                            # Format 5: {"errors": [...]}
                            elif 'errors' in error_data:
                                errors = error_data['errors']
                                if isinstance(errors, list):
                                    error_detail = '; '.join([str(e) for e in errors])
                                else:
                                    error_detail = str(errors)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse error response: {parse_error}")
                    error_detail = response.text[:200] if response.text else ""
                
                # Check if this is a candidate endpoint - might be a permissions issue
                is_candidate_endpoint = '/candidate' in endpoint.lower()
                permission_note = ""
                if is_candidate_endpoint:
                    permission_note = (
                        f"\n\n[WARNING] IMPORTANT: This is a CANDIDATE endpoint. Your API key may not have permissions "
                        f"to access candidates. Some JobVite API keys only have access to jobs, not candidates.\n"
                        f"Contact JobVite support to enable candidate access for your API key if needed."
                    )
                
                error_msg = (
                    f"Jobvite API authentication failed (401 Unauthorized). "
                    f"Invalid API key or secret. "
                    f"\n\nAPI Key (masked): {masked_key}"
                    f"\nCompany ID: {self.company_id}"
                    f"\nBase URL: {self.base_url}"
                    f"\nEndpoint: {endpoint}"
                    f"\nRequest URL: {url}"
                    f"\n\nError details: {error_detail if error_detail else 'No additional details'}"
                    f"\n\nTroubleshooting:"
                    f"\n1. Verify the API Key is correct in Jobvite admin panel"
                    f"\n2. Verify the API Secret is correct (check for typos, extra spaces)"
                    f"\n3. Ensure you're using Production credentials with Production URL (or Stage with Stage URL)"
                    f"\n4. Check that the API key hasn't been revoked or expired"
                    f"\n5. Verify the API key has permissions to access the {endpoint} endpoint{permission_note}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if response.status_code == 403:
                error_msg = (
                    f"Jobvite API access denied (403 Forbidden). "
                    f"API key lacks required permissions or Company ID is incorrect. "
                    f"API Key (masked): {masked_key}, Company ID: {self.company_id}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if response.status_code == 404:
                # Try to parse error message from response - handle multiple error formats
                error_detail = "Unknown error"
                error_code = None
                try:
                    if response.text:
                        import json
                        error_data = response.json()
                        if isinstance(error_data, dict):
                            # Format 1: {"status": {"code": 404, "messages": [...]}}
                            if 'status' in error_data:
                                status_obj = error_data['status']
                                if isinstance(status_obj, dict):
                                    messages = status_obj.get('messages', [])
                                    error_detail = '; '.join(messages) if messages else str(error_data)
                                    error_code = status_obj.get('code')
                            # Format 2: {"message": "..."}
                            elif 'message' in error_data:
                                error_detail = error_data['message']
                            # Format 3: {"error": "..."}
                            elif 'error' in error_data:
                                error_detail = str(error_data['error'])
                            # Format 4: {"error": {"code": "...", "message": "..."}}
                            elif isinstance(error_data.get('error'), dict):
                                error_obj = error_data['error']
                                error_detail = error_obj.get('message', str(error_obj))
                                error_code = error_obj.get('code')
                            # Format 5: {"errors": [...]}
                            elif 'errors' in error_data:
                                errors = error_data['errors']
                                if isinstance(errors, list):
                                    error_detail = '; '.join([str(e) for e in errors])
                                else:
                                    error_detail = str(errors)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse 404 error response: {parse_error}")
                    error_detail = response.text[:200] if response.text else "No error details"
                
                error_msg = (
                    f"Jobvite API endpoint not found (404). "
                    f"URL: {url}, Error: {error_detail}. "
                    f"\n\nPossible causes:"
                    f"\n1. Endpoint path is incorrect (trying: {endpoint})"
                    f"\n2. API version mismatch (using: {self.base_url})"
                    f"\n3. Company ID or credentials don't have access to this endpoint"
                    f"\n4. Wrong environment (Production vs Stage)"
                    f"\n\nAPI Key (masked): {masked_key}, Company ID: {self.company_id}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                if retry_count < 3:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited, retrying after {retry_after} seconds")
                    log_jobvite_request(
                        tenant_id=0,  # Will be set by caller
                        company_id=self.company_id,
                        endpoint=endpoint,
                        method=method,
                        status_code=429,
                        duration_ms=duration_ms,
                        rate_limit_retries=retry_count + 1
                    )
                    time.sleep(retry_after)
                    return self._make_request(method, endpoint, params, json_data, retry_count + 1)
                else:
                    logger.error("Rate limit exceeded after 3 retries")
                    log_jobvite_request(
                        tenant_id=0,
                        company_id=self.company_id,
                        endpoint=endpoint,
                        method=method,
                        status_code=429,
                        duration_ms=duration_ms,
                        rate_limit_retries=retry_count + 1,
                        error="Rate limit exceeded after 3 retries"
                    )
                    response.raise_for_status()
            
            # Handle 5xx server errors per spec (before raise_for_status)
            if 500 <= response.status_code < 600:
                error_msg = (
                    f"Jobvite API server error ({response.status_code}). "
                    f"This is a transient server-side issue. Consider retrying. "
                    f"URL: {url}"
                )
                logger.error(error_msg)
            
            response.raise_for_status()
            
            # Log successful request
            items_count = None
            if response.headers.get('Content-Type', '').startswith('application/json'):
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        items_count = len(data.get('jobs', data.get('candidates', data.get('items', []))))
                except:
                    pass
            
            log_jobvite_request(
                tenant_id=0,
                company_id=self.company_id,
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                duration_ms=duration_ms,
                items_count=items_count,
                rate_limit_retries=retry_count
            )
            
            return response
        except requests.exceptions.HTTPError as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"  # Truncate response
            log_jobvite_request(
                tenant_id=0,
                company_id=self.company_id,
                endpoint=endpoint,
                method=method,
                status_code=e.response.status_code,
                duration_ms=duration_ms,
                error=error_msg
            )
            logger.error(f"Jobvite API error: {e.response.status_code} - {error_msg}")
            raise
        except requests.exceptions.RequestException as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            log_jobvite_request(
                tenant_id=0,
                company_id=self.company_id,
                endpoint=endpoint,
                method=method,
                duration_ms=duration_ms,
                error=error_msg
            )
            logger.error(f"Jobvite API request failed: {error_msg}")
            raise
    
    def get_candidate(self, candidate_id: Optional[str] = None, 
                     application_id: Optional[str] = None,
                     filters: Optional[Dict] = None,
                     start: int = 0,
                     count: int = 50) -> Dict[str, Any]:
        """
        Get candidate(s) from Jobvite.
        
        Args:
            candidate_id: Jobvite candidate ID (if provided, returns single candidate, ignores other params)
            application_id: Jobvite application ID (alternative to candidate_id)
            filters: Additional filters (workflowState, email, jobId, etc.)
                    Common filter keys: workflowState, email, jobId, status
            start: Pagination start index (default: 0, must be >= 0)
            count: Number of results per page (default: 50, typically max: 100)
        
        Returns:
            Candidate data or paginated list of candidates with structure:
            {
                "candidates": [...],
                "total": <number>,
                "start": <number>,
                "count": <number>
            }
            OR single candidate object if candidate_id provided
        """
        # Validate parameters
        if start < 0:
            raise ValueError("start parameter must be >= 0")
        if count < 1 or count > 1000:
            raise ValueError("count parameter must be between 1 and 1000")
        if candidate_id and application_id:
            logger.warning("Both candidate_id and application_id provided, using candidate_id")
        
        # Per Jobvite API spec: endpoint is /candidate (singular), matching /job pattern
        endpoint = "/candidate"
        params = {
            'start': start,
            'count': count
        }
        
        if candidate_id:
            endpoint = f"/candidate/{candidate_id}"
            # Even for single candidate, include start/count to avoid redirect issues
            params = {
                'start': 0,
                'count': 1
            }
        elif application_id:
            params['applicationId'] = application_id
        
        # Add additional filters if provided
        if filters:
            # Validate filter keys (common Jobvite API filter keys)
            allowed_filter_keys = {
                'workflowState', 'email', 'jobId', 'status', 
                'firstName', 'lastName', 'phone', 'personalDataProcessingStatus'
            }
            for key in filters.keys():
                if key not in allowed_filter_keys and not key.startswith('custom'):
                    logger.warning(f"Unknown filter key: {key}. This may be a custom field or unsupported filter.")
            params.update(filters)
        
        response = self._make_request('GET', endpoint, params=params)
        
        # Log basic response info (safe encoding for Windows console)
        final_url = getattr(response, 'url', endpoint)
        try:
            response_length = len(response.text) if response.text else 0
            logger.debug(
                f"JobVite get_candidate() response: endpoint={endpoint}, status={response.status_code}, "
                f"length={response_length} bytes, final_url={final_url}, params={params}"
            )
        except Exception as e:
            logger.debug(f"JobVite get_candidate() response: endpoint={endpoint}, status={response.status_code}")
        
        # Check if response has content before parsing JSON
        if not response.text or not response.text.strip():
            logger.error(
                f"[ERROR] Empty response from Jobvite API candidate endpoint\n"
                f"   Endpoint: {endpoint}\n"
                f"   Status: {response.status_code}\n"
                f"   This may indicate the API key lacks candidate permissions"
            )
            raise ValueError(f"Empty response from Jobvite API. Status: {response.status_code}")
        
        # Check Content-Type header
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            # HTML response usually means authentication failure or wrong endpoint
            response_preview = response.text[:500] if response.text else ""
            logger.error(
                f"Jobvite API returned HTML instead of JSON. "
                f"Endpoint: {endpoint}, Status: {response.status_code}, "
                f"Final URL: {response.url if hasattr(response, 'url') else 'unknown'}. "
                f"This usually indicates: 1) Invalid API credentials, 2) Wrong base URL, or 3) Invalid endpoint. "
                f"Response preview: {response_preview}"
            )
            raise ValueError(
                f"Jobvite API returned HTML instead of JSON. This usually indicates invalid credentials or wrong endpoint. "
                f"Status: {response.status_code}, Content-Type: {content_type}. "
                f"Please verify your API key, secret, company ID, and base URL are correct."
            )
        elif content_type and not content_type.startswith('application/json'):
            logger.error(f"Non-JSON response from Jobvite API: {endpoint} (Content-Type: {content_type}, status: {response.status_code})")
            raise ValueError(f"Expected JSON response but got {content_type}. Status: {response.status_code}, Response: {response.text[:200]}")
        
        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"Failed to parse JSON from Jobvite API: {endpoint} (status: {response.status_code}, response: {response.text[:200]})")
            raise ValueError(f"Invalid JSON response from Jobvite API. Status: {response.status_code}, Error: {str(e)}, Response: {response.text[:200]}")
        
        # Validate response structure
        if not isinstance(data, dict):
            logger.warning(f"Jobvite API returned non-dict response: {type(data)}")
            # Some endpoints might return arrays directly, wrap it
            if isinstance(data, list):
                return {'candidates': data, 'total': len(data), 'start': start, 'count': count}
            return data
        
        # Ensure consistent response format
        if candidate_id:
            # Single candidate response - return as-is or wrap if needed
            if 'candidate' in data:
                return data['candidate']
            return data
        else:
            # List response - ensure 'candidates' key exists
            if 'candidates' not in data:
                # Check for alternative keys
                if 'items' in data:
                    data['candidates'] = data['items']
                    logger.debug(f"Using 'items' key for candidates (found {len(data['items'])} items)")
                elif 'data' in data:
                    data['candidates'] = data['data']
                    logger.debug(f"Using 'data' key for candidates (found {len(data['data'])} items)")
                else:
                    # No candidates key found, create empty list
                    available_keys = list(data.keys())
                    logger.warning(
                        f"[WARNING] Candidate response missing 'candidates' key. "
                        f"Available keys: {available_keys}. "
                        f"This may indicate:\n"
                        f"  1. API key lacks candidate permissions\n"
                        f"  2. No candidates match the filters\n"
                        f"  3. API response format changed"
                    )
                    data['candidates'] = []
            
            # Debug: Log if candidates list is empty
            candidates_list = data.get('candidates', [])
            total = data.get('total', 0)
            if not candidates_list and total == 0:
                logger.info(
                    f"[INFO] No candidates returned from JobVite API. "
                    f"Total: {total}, Start: {start}, Count: {count}, "
                    f"Filters: {filters if filters else 'None'}"
                )
            elif not candidates_list and total > 0:
                logger.warning(
                    f"[WARNING] Candidate response shows total={total} but empty candidates list. "
                    f"This may indicate a pagination issue or API response format problem."
                )
        
        return data
    
    def get_candidate_with_artifacts(self, candidate_id: str) -> Dict[str, Any]:
        """
        Get candidate with encoded artifacts (documents).
        
        This endpoint returns base64-encoded documents.
        """
        endpoint = f"/candidate/{candidate_id}/artifacts"
        response = self._make_request('GET', endpoint)
        
        # Check if response has content before parsing JSON
        if not response.text or not response.text.strip():
            logger.error(f"Empty response from Jobvite API: {endpoint} (status: {response.status_code})")
            raise ValueError(f"Empty response from Jobvite API. Status: {response.status_code}")
        
        # Check Content-Type header
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            # HTML response usually means authentication failure or wrong endpoint
            response_preview = response.text[:500] if response.text else ""
            logger.error(
                f"Jobvite API returned HTML instead of JSON. "
                f"Endpoint: {endpoint}, Status: {response.status_code}, "
                f"Final URL: {response.url if hasattr(response, 'url') else 'unknown'}. "
                f"This usually indicates: 1) Invalid API credentials, 2) Wrong base URL, or 3) Invalid endpoint. "
                f"Response preview: {response_preview}"
            )
            raise ValueError(
                f"Jobvite API returned HTML instead of JSON. This usually indicates invalid credentials or wrong endpoint. "
                f"Status: {response.status_code}, Content-Type: {content_type}. "
                f"Please verify your API key, secret, company ID, and base URL are correct."
            )
        elif content_type and not content_type.startswith('application/json'):
            logger.error(f"Non-JSON response from Jobvite API: {endpoint} (Content-Type: {content_type}, status: {response.status_code})")
            raise ValueError(f"Expected JSON response but got {content_type}. Status: {response.status_code}, Response: {response.text[:200]}")
        
        try:
            return response.json()
        except ValueError as e:
            logger.error(f"Failed to parse JSON from Jobvite API: {endpoint} (status: {response.status_code}, response: {response.text[:200]})")
            raise ValueError(f"Invalid JSON response from Jobvite API. Status: {response.status_code}, Error: {str(e)}, Response: {response.text[:200]}")
    
    def get_job(self, job_id: Optional[str] = None, 
               requisition_id: Optional[str] = None,
               filters: Optional[Dict] = None,
               start: int = 0,
               count: int = 50) -> Dict[str, Any]:
        """
        Get job(s) from Jobvite.
        
        Args:
            job_id: Jobvite job ID (if provided, returns single job, ignores other params)
            requisition_id: Requisition ID (alternative to job_id)
            filters: Additional filters (status, location, department, etc.)
                    Common filter keys: status, location, department, category
            start: Pagination start index (default: 0, must be >= 0)
            count: Number of results per page (default: 50, typically max: 100)
        
        Returns:
            Job data or paginated list of jobs with structure:
            {
                "jobs": [...],
                "total": <number>,
                "start": <number>,
                "count": <number>
            }
            OR single job object if job_id provided
        
        Note: Handle 'total=0 on last page' quirk per Jobvite docs.
        """
        # Validate parameters
        if start < 0:
            raise ValueError("start parameter must be >= 0")
        if count < 1 or count > 1000:
            raise ValueError("count parameter must be between 1 and 1000")
        if job_id and requisition_id:
            logger.warning("Both job_id and requisition_id provided, using job_id")
        
        # Per Jobvite API spec: endpoint is /job (singular), not /jobs
        # Jobvite redirects /job to /job?start=0&count=1, so ALWAYS include start and count
        endpoint = "/job"
        params = {
            'start': start,
            'count': count
        }
        
        if job_id:
            endpoint = f"/job/{job_id}"
            # Even for single job, include start/count to avoid redirect issues
            params = {
                'start': 0,
                'count': 1
            }
        elif requisition_id:
            params['requisitionId'] = requisition_id
        
        # Add additional filters if provided
        if filters:
            # Validate filter keys (common Jobvite API filter keys)
            allowed_filter_keys = {
                'status', 'location', 'department', 'category', 
                'region', 'subsidiary', 'remoteType', 'salaryCurrency',
                'salaryMin', 'salaryMax', 'salaryFrequency'
            }
            for key in filters.keys():
                if key not in allowed_filter_keys and not key.startswith('custom'):
                    logger.warning(f"Unknown filter key: {key}. This may be a custom field or unsupported filter.")
            params.update(filters)
        
        response = self._make_request('GET', endpoint, params=params)
        
        # Log basic response info (safe encoding for Windows console)
        final_url = getattr(response, 'url', endpoint)
        try:
            # Safe encoding for logging - replace problematic Unicode chars
            response_length = len(response.text) if response.text else 0
            logger.debug(
                f"JobVite get_job() response: endpoint={endpoint}, status={response.status_code}, "
                f"length={response_length} bytes, final_url={final_url}"
            )
        except Exception as e:
            # If logging fails, just log minimal info
            logger.debug(f"JobVite get_job() response: endpoint={endpoint}, status={response.status_code}")
        
        # Check if response has content before parsing JSON
        if not response.text or not response.text.strip():
            logger.error(f"Empty response from Jobvite API: {endpoint} (status: {response.status_code})")
            raise ValueError(f"Empty response from Jobvite API. Status: {response.status_code}")
        
        # Check Content-Type header
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            # HTML response usually means authentication failure or wrong endpoint
            response_preview = response.text[:500] if response.text else ""
            logger.error(
                f"Jobvite API returned HTML instead of JSON. "
                f"Endpoint: {endpoint}, Status: {response.status_code}, "
                f"Final URL: {response.url if hasattr(response, 'url') else 'unknown'}. "
                f"This usually indicates: 1) Invalid API credentials, 2) Wrong base URL, or 3) Invalid endpoint. "
                f"Response preview: {response_preview}"
            )
            raise ValueError(
                f"Jobvite API returned HTML instead of JSON. This usually indicates invalid credentials or wrong endpoint. "
                f"Status: {response.status_code}, Content-Type: {content_type}. "
                f"Please verify your API key, secret, company ID, and base URL are correct."
            )
        elif content_type and not content_type.startswith('application/json'):
            logger.error(f"Non-JSON response from Jobvite API: {endpoint} (Content-Type: {content_type}, status: {response.status_code})")
            raise ValueError(f"Expected JSON response but got {content_type}. Status: {response.status_code}, Response: {response.text[:200]}")
        
        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"Failed to parse JSON from Jobvite API: {endpoint} (status: {response.status_code}, response: {response.text[:200]})")
            raise ValueError(f"Invalid JSON response from Jobvite API. Status: {response.status_code}, Error: {str(e)}, Response: {response.text[:200]}")
        
        # Validate response structure
        if not isinstance(data, dict):
            logger.warning(f"Jobvite API returned non-dict response: {type(data)}")
            # Some endpoints might return arrays directly, wrap it
            if isinstance(data, list):
                return {'jobs': data, 'total': len(data), 'start': start, 'count': count}
            return data
        
        # Handle single job response
        if job_id:
            # Single job response - return as-is or wrap if needed
            if 'job' in data:
                return data['job']
            return data
        
        # List response - use fallback parsing: jobs OR requisitions OR []
        # This is the core fix: Jobvite returns 'requisitions' not 'jobs' for many tenants
        # Check for jobs OR requisitions first (primary keys)
        if 'jobs' in data:
            jobs = data['jobs']
        elif 'requisitions' in data:
            jobs = data['requisitions']
            logger.debug(f"Found jobs under 'requisitions' key: {len(jobs) if isinstance(jobs, list) else 'non-list'}")
        else:
            # Fallback to other possible keys
            if 'items' in data:
                jobs = data['items']
            elif 'data' in data:
                jobs = data['data']
            elif 'job' in data:
                # Singular job key
                jobs = [data['job']] if isinstance(data['job'], dict) else data.get('job', [])
            else:
                # No recognized key found - raise clear error
                available_keys = list(data.keys())
                error_msg = (
                    f"Jobvite API response missing 'jobs' or 'requisitions' key. "
                    f"Available keys: {available_keys}. "
                    f"This may indicate an API change or unsupported response format."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Ensure jobs is a list
        if not isinstance(jobs, list):
            if isinstance(jobs, dict):
                # If it's a dict with 'items', extract items
                jobs = jobs.get('items', [jobs])
            else:
                jobs = [jobs] if jobs else []
        
        # Get total - use provided total or calculate from jobs length
        total = data.get("total", len(jobs))
        
        # Handle pagination quirk: if total=0 on last page, we've reached the end
        if total == 0 and start > 0:
            # This is the last page (Jobvite quirk)
            return {'jobs': [], 'total': 0, 'start': start, 'count': count, 'returned': 0}
        
        # Return consistent format
        return {
            'jobs': jobs,
            'total': total,
            'start': start,
            'count': count,
            'returned': len(jobs)
        }
    
    def get_application(self, application_id: Optional[str] = None,
                       filters: Optional[Dict] = None,
                       start: int = 0,
                       count: int = 50) -> Dict[str, Any]:
        """
        Get application(s) from Jobvite.
        
        Args:
            application_id: Jobvite application ID (if provided, returns single application)
            filters: Additional filters
            start: Pagination start index (default: 0)
            count: Number of results per page (default: 50)
        
        Returns:
            Application data or paginated list
        """
        endpoint = "/application"
        params = {'start': start, 'count': count}
        
        if application_id:
            endpoint = f"/application/{application_id}"
            params = {}
        
        if filters:
            params.update(filters)
        
        response = self._make_request('GET', endpoint, params=params)
        return response.json()
    
    def get_application_history(self, application_id: str,
                               start: int = 0,
                               count: int = 50) -> Dict[str, Any]:
        """
        Get application history.
        
        Args:
            application_id: Jobvite application ID
            start: Pagination start index (default: 0)
            count: Number of results per page (default: 50)
        
        Returns:
            Application history data
        """
        endpoint = f"/application/{application_id}/history"
        params = {'start': start, 'count': count}
        
        response = self._make_request('GET', endpoint, params=params)
        return response.json()
    
    def get_contact(self, contact_id: Optional[str] = None,
                   filters: Optional[Dict] = None,
                   start: int = 0,
                   count: int = 50) -> Dict[str, Any]:
        """
        Get contact(s) from Jobvite.
        
        Args:
            contact_id: Jobvite contact ID (if provided, returns single contact)
            filters: Additional filters
            start: Pagination start index (default: 0)
            count: Number of results per page (default: 50)
        
        Returns:
            Contact data or paginated list
        """
        endpoint = "/contact"
        params = {'start': start, 'count': count}
        
        if contact_id:
            endpoint = f"/contact/{contact_id}"
            params = {}
        
        if filters:
            params.update(filters)
        
        response = self._make_request('GET', endpoint, params=params)
        return response.json()
    
    def get_employee(self, employee_id: Optional[str] = None,
                    filters: Optional[Dict] = None,
                    start: int = 0,
                    count: int = 50) -> Dict[str, Any]:
        """
        Get employee(s) from Jobvite.
        
        Args:
            employee_id: Jobvite employee ID (if provided, returns single employee)
            filters: Additional filters
            start: Pagination start index (default: 0)
            count: Number of results per page (default: 50)
        
        Returns:
            Employee data or paginated list
        """
        endpoint = "/employee"
        params = {'start': start, 'count': count}
        
        if employee_id:
            endpoint = f"/employee/{employee_id}"
            params = {}
        
        if filters:
            params.update(filters)
        
        response = self._make_request('GET', endpoint, params=params)
        return response.json()
    
    def get_offer(self, offer_id: Optional[str] = None,
                 filters: Optional[Dict] = None,
                 start: int = 0,
                 count: int = 50) -> Dict[str, Any]:
        """
        Get offer(s) from Jobvite.
        
        Args:
            offer_id: Jobvite offer ID (if provided, returns single offer)
            filters: Additional filters
            start: Pagination start index (default: 0)
            count: Number of results per page (default: 50)
        
        Returns:
            Offer data or paginated list
        """
        endpoint = "/offer"
        params = {'start': start, 'count': count}
        
        if offer_id:
            endpoint = f"/offer/{offer_id}"
            params = {}
        
        if filters:
            params.update(filters)
        
        response = self._make_request('GET', endpoint, params=params)
        return response.json()
    
    def get_interview(self, interview_id: Optional[str] = None,
                     filters: Optional[Dict] = None,
                     start: int = 0,
                     count: int = 50) -> Dict[str, Any]:
        """
        Get interview(s) from Jobvite.
        
        Args:
            interview_id: Jobvite interview ID (if provided, returns single interview)
            filters: Additional filters
            start: Pagination start index (default: 0)
            count: Number of results per page (default: 50)
        
        Returns:
            Interview data or paginated list
        """
        endpoint = "/interview"
        params = {'start': start, 'count': count}
        
        if interview_id:
            endpoint = f"/interview/{interview_id}"
            params = {}
        
        if filters:
            params.update(filters)
        
        response = self._make_request('GET', endpoint, params=params)
        return response.json()
    
    def get_customfield(self, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get custom fields from Jobvite.
        
        Args:
            filters: Additional filters
        
        Returns:
            Custom field data
        """
        endpoint = "/customfield"
        params = filters or {}
        
        response = self._make_request('GET', endpoint, params=params)
        return response.json()
    
    def list_webhooks(self, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        List webhooks configured in Jobvite.
        
        Args:
            filters: Additional filters
        
        Returns:
            Webhook list
        """
        endpoint = "/webhook"
        params = filters or {}
        
        response = self._make_request('GET', endpoint, params=params)
        return response.json()
    
    def create_webhook(self, webhook_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new webhook in Jobvite.
        
        Args:
            webhook_config: Webhook configuration dict
        
        Returns:
            Created webhook data
        """
        endpoint = "/webhook"
        response = self._make_request('POST', endpoint, json_data=webhook_config)
        return response.json()
    
    def delete_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """
        Delete a webhook from Jobvite.
        
        Args:
            webhook_id: Webhook ID to delete
        
        Returns:
            Deletion confirmation
        """
        endpoint = f"/webhook/{webhook_id}"
        response = self._make_request('DELETE', endpoint)
        return response.json() if response.text else {}
    
    def paginate_all(self, fetch_fn, *args, limit: Optional[int] = None, start_offset: int = 0, **kwargs) -> List[Dict]:
        """
        Helper to paginate through all results with comprehensive edge case handling.
        
        Args:
            fetch_fn: Function that accepts start, count and returns paginated response
            *args, **kwargs: Additional arguments for fetch_fn
            limit: Optional limit on total number of items to fetch (stops when reached)
            start_offset: Starting position for pagination (default: 0). Used for incremental sync.
        
        Returns:
            List of all items across all pages (up to limit if specified)
        
        Handles Jobvite API pagination quirks:
        - total=0 on last page (known quirk)
        - Empty results on last page
        - Response format variations (jobs, candidates, items keys)
        - Maximum page limit protection (prevents infinite loops)
        """
        all_items = []
        start = start_offset  # Start from the offset position for incremental sync
        count = 50  # Default page size
        max_pages = 1000  # Safety limit to prevent infinite loops
        page_count = 0
        
        # HARD SAFETY LIMIT: Never fetch more than 1000 items even if limit is higher
        MAX_SAFETY_LIMIT = 1000
        
        # DEBUG: Log pagination parameters
        logger.info(f"[PAGINATION] ========== PAGINATE_ALL CALLED ==========")
        logger.info(f"[PAGINATION] Function: {fetch_fn.__name__ if hasattr(fetch_fn, '__name__') else str(fetch_fn)}")
        logger.info(f"[PAGINATION] Start offset: {start_offset}")
        logger.info(f"[PAGINATION] Limit: {limit} (type: {type(limit)})")
        logger.info(f"[PAGINATION] Initial start position: {start}")
        logger.info(f"[PAGINATION] Page size: {count}")
        
        if limit is None or limit <= 0:
            logger.info(f"[PAGINATION] No limit specified, will fetch all items (max {MAX_SAFETY_LIMIT} for safety)")
            # Check if this is a candidate fetch - if so, warn about potential large dataset
            if 'candidate' in str(fetch_fn).lower() or 'get_candidate' in str(fetch_fn):
                logger.warning(f"[PAGINATION WARNING] Fetching candidates without limit! This could fetch millions of records.")
                logger.warning(f"[PAGINATION WARNING] Consider setting candidateSyncLimit in sync config to limit the fetch.")
        else:
            if limit > MAX_SAFETY_LIMIT:
                logger.warning(f"[PAGINATION] Limit {limit} exceeds safety limit {MAX_SAFETY_LIMIT}. Capping to {MAX_SAFETY_LIMIT}")
                limit = MAX_SAFETY_LIMIT
            logger.info(f"[PAGINATION] HARD LIMITED to {limit} items - will stop immediately when reached")
            logger.info(f"[PAGINATION] Will fetch from position {start_offset} to {start_offset + limit - 1}")
        
        logger.info(f"[PAGINATION] Starting pagination: start={start}, count={count}, limit={limit}")
        
        while page_count < max_pages:
            # Log state at start of each iteration
            logger.info(f"[PAGINATION] ========== PAGE {page_count + 1} ==========")
            logger.info(f"[PAGINATION] Current state: start={start}, count={count}, collected={len(all_items)}, limit={limit}")
            
            # HARD LIMIT CHECK #1: Stop before making API call if limit already reached
            if limit is not None and limit > 0 and len(all_items) >= limit:
                logger.info(f"[PAGINATION] Limit of {limit} already reached. Stopping pagination. Current count: {len(all_items)}")
                break
            
            # HARD LIMIT CHECK #2: Don't make API call if we would exceed limit
            # Note: For incremental sync, we check if start exceeds (start_offset + limit)
            if limit is not None and limit > 0:
                max_start = start_offset + limit
                if start >= max_start:
                    logger.info(f"[PAGINATION] Start position {start} would exceed max position {max_start}. Stopping pagination.")
                    break
            
            # HARD LIMIT CHECK: Adjust count to not exceed limit in this batch
            adjusted_count = count
            if limit is not None and limit > 0:
                remaining = limit - len(all_items)
                if remaining <= 0:
                    logger.info(f"[PAGINATION] No remaining items needed. Limit {limit} already reached with {len(all_items)} items.")
                    break
                # Don't request more than we need
                adjusted_count = min(count, remaining)
                if adjusted_count < count:
                    logger.info(f"[PAGINATION] Adjusted batch size from {count} to {adjusted_count} to respect limit of {limit}")
            
            try:
                # Log before making API call
                logger.info(f"[PAGINATION] ========== API CALL #{page_count + 1} ==========")
                logger.info(f"[PAGINATION] Request parameters:")
                logger.info(f"[PAGINATION]   - start: {start} (absolute position in Jobvite)")
                logger.info(f"[PAGINATION]   - count: {adjusted_count} (items to fetch in this batch)")
                logger.info(f"[PAGINATION]   - collected so far: {len(all_items)}")
                if limit:
                    logger.info(f"[PAGINATION]   - remaining to collect: {limit - len(all_items)}")
                    logger.info(f"[PAGINATION]   - target limit: {limit}")
                    logger.info(f"[PAGINATION]   - start_offset was: {start_offset}")
                    logger.info(f"[PAGINATION]   - max position: {start_offset + limit}")
                
                result = fetch_fn(*args, start=start, count=adjusted_count, **kwargs)
                page_count += 1
                
                logger.info(f"[PAGINATION] API call #{page_count} completed: result_type={type(result)}")
            except ValueError as e:
                # Authentication/permissions errors
                error_str = str(e)
                if '401' in error_str or 'Unauthorized' in error_str or 'permissions' in error_str.lower():
                    logger.error(
                        f"[ERROR] CANDIDATE PAGINATION FAILED: API PERMISSIONS ISSUE\n"
                        f"{'=' * 70}\n"
                        f"Failed to fetch candidates at start={start}, count={count}\n"
                        f"Your JobVite API key does NOT have permissions to access the /candidate endpoint.\n"
                        f"The API key works for jobs but NOT for candidates.\n\n"
                        f"SOLUTION: Contact JobVite support to enable candidate endpoint access.\n"
                        f"Error: {error_str[:400]}"
                    )
                else:
                    logger.error(f"Error during pagination at start={start}: {error_str}")
                raise  # Re-raise to be caught by sync function
            except Exception as e:
                logger.error(
                    f"[ERROR] Unexpected error during candidate pagination at start={start}, count={count}\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Error message: {str(e)[:400]}"
                )
                raise  # Re-raise to be caught by sync function
            
            if isinstance(result, dict):
                # Handle different response key formats - check jobs OR requisitions first
                items = (
                    result.get('jobs', []) or 
                    result.get('requisitions', []) or
                    result.get('candidates', []) or 
                    result.get('items', []) or 
                    result.get('data', []) or
                    []
                )
                
                # Debug: Log if candidates key is missing
                if not result.get('candidates') and not result.get('jobs') and not result.get('requisitions'):
                    available_keys = list(result.keys())
                    logger.debug(
                        f"[WARNING] Candidate pagination: Response missing 'candidates' key. "
                        f"Available keys: {available_keys}. "
                        f"Page: {page_count}, Start: {start}, Count: {count}"
                    )
                
                # Ensure items is a list
                if not isinstance(items, list):
                    items = [items] if items else []
                
                total = result.get('total', None)
                
                # Debug: Log pagination progress for candidates
                if 'candidate' in str(fetch_fn).lower() or 'get_candidate' in str(fetch_fn):
                    logger.debug(
                        f"[DEBUG] Candidate pagination progress: "
                        f"Page {page_count}, Start {start}, "
                        f"Found {len(items)} candidates, Total: {total}, "
                        f"All items so far: {len(all_items)}"
                    )
                
                # Add items to collection (respecting limit if set)
                if items:
                    logger.info(f"[PAGINATION] API returned {len(items)} items in this batch")
                    logger.info(f"[PAGINATION] Current collection status: {len(all_items)} items collected")
                    
                    if limit is not None and limit > 0:
                        # Only add items up to the limit
                        remaining = limit - len(all_items)
                        logger.info(f"[PAGINATION] Limit enforcement:")
                        logger.info(f"[PAGINATION]   - Items in batch: {len(items)}")
                        logger.info(f"[PAGINATION]   - Already collected: {len(all_items)}")
                        logger.info(f"[PAGINATION]   - Limit: {limit}")
                        logger.info(f"[PAGINATION]   - Remaining slots: {remaining}")
                        
                        if remaining > 0:
                            items_to_add = items[:remaining]
                            all_items.extend(items_to_add)
                            logger.info(f"[PAGINATION] Added {len(items_to_add)} items to collection")
                            logger.info(f"[PAGINATION] Total collected: {len(all_items)}/{limit}")
                            
                            if len(items) > remaining:
                                logger.info(f"[PAGINATION] NOTE: Batch had {len(items)} items but only {remaining} slots available. Trimmed to fit limit.")
                        else:
                            logger.info(f"[PAGINATION] SKIP: No remaining slots (remaining={remaining}), batch not added")
                        
                        # HARD STOP #3: If we've reached or exceeded the limit, stop immediately
                        if len(all_items) >= limit:
                            logger.info(f"[PAGINATION] ========== LIMIT REACHED ==========")
                            logger.info(f"[PAGINATION] Collected {len(all_items)} items, limit was {limit}")
                            logger.info(f"[PAGINATION] Stopping pagination immediately")
                            break
                    else:
                        all_items.extend(items)
                        logger.info(f"[PAGINATION] Added all {len(items)} items (no limit set, total: {len(all_items)})")
                else:
                    logger.info(f"[PAGINATION] API returned 0 items in this batch")
                
                # HARD LIMIT CHECK #4: Stop immediately if limit reached (before any other checks)
                if limit is not None and limit > 0 and len(all_items) >= limit:
                    logger.info(f"[PAGINATION] Limit of {limit} reached. Final count: {len(all_items)}")
                    break
                
                # Check if we've reached the end
                # Edge case 1: No items returned
                if not items:
                    logger.info(f"[PAGINATION] No items returned, reached end of data")
                    break
                
                # Edge case 2: Jobvite quirk - total=0 on last page
                if total == 0 and start > start_offset:
                    logger.info(f"[PAGINATION] Total=0 on last page, reached end of data")
                    break
                
                # Edge case 3: Fewer items than requested (last page)
                if len(items) < adjusted_count:
                    logger.info(f"[PAGINATION] Fewer items than requested ({len(items)} < {adjusted_count}), reached last page")
                    break
                
                # Edge case 4: Continue until start >= total (proper pagination check)
                # BUT: Skip this check if we have a limit and haven't reached it yet
                if total is not None:
                    if start >= total:
                        logger.info(f"[PAGINATION] Start position {start} >= total {total}, reached end of data")
                        break
                    # Only check total if we don't have a limit, or if we haven't reached limit yet
                    if limit is None or limit <= 0:
                        if len(all_items) >= total:
                            logger.info(f"[PAGINATION] Collected items {len(all_items)} >= total {total}, reached end of data")
                            break
                
                # Edge case 5: Same number of items as previous page (potential loop)
                if page_count > 1 and len(items) == 0:
                    logger.info(f"[PAGINATION] No items on page {page_count}, potential loop detected, stopping")
                    break
                
                # HARD LIMIT CHECK #5: Don't move to next page if we've reached limit
                if limit is not None and limit > 0 and len(all_items) >= limit:
                    logger.info(f"[PAGINATION] Not moving to next page - limit {limit} reached. Current count: {len(all_items)}")
                    break
                
                # Move to next page ONLY if limit not reached
                start += adjusted_count
                logger.info(f"[PAGINATION] Incremented start to {start}, current items: {len(all_items)}, limit: {limit}")
                
                # HARD LIMIT CHECK #6: Don't continue if next start would exceed limit
                if limit is not None and limit > 0:
                    max_start = start_offset + limit
                    if start >= max_start:
                        logger.info(f"[PAGINATION] Next start position {start} would exceed max position {max_start}. Stopping.")
                        break
            elif isinstance(result, list):
                # Single page response as array
                if limit is not None and limit > 0:
                    remaining = limit - len(all_items)
                    if remaining > 0:
                        all_items.extend(result[:remaining])
                    if len(all_items) >= limit:
                        logger.info(f"[PAGINATION] Reached limit of {limit} items from list response. Final count: {len(all_items)}")
                else:
                    all_items.extend(result)
                break
            else:
                # Single item response
                if limit is not None and limit > 0:
                    if len(all_items) < limit:
                        all_items.append(result)
                    if len(all_items) >= limit:
                        logger.info(f"[PAGINATION] Reached limit of {limit} items from single item response. Final count: {len(all_items)}")
                else:
                    all_items.append(result)
                break
        
        if page_count >= max_pages:
            logger.warning(f"[PAGINATION] Reached maximum page limit ({max_pages}). Stopped at {len(all_items)} items.")
        
        # Final hard limit enforcement - trim if somehow we exceeded
        if limit is not None and limit > 0 and len(all_items) > limit:
            logger.warning(f"[PAGINATION] Pagination exceeded limit! Had {len(all_items)} items but limit was {limit}. Trimming to {limit}.")
            all_items = all_items[:limit]
        
        logger.info(f"[PAGINATION] ========== PAGINATION COMPLETE ==========")
        logger.info(f"[PAGINATION] Summary:")
        logger.info(f"[PAGINATION]   - Final count: {len(all_items)} items collected")
        logger.info(f"[PAGINATION]   - Pages fetched: {page_count}")
        logger.info(f"[PAGINATION]   - Start offset: {start_offset}")
        logger.info(f"[PAGINATION]   - Final start position: {start}")
        logger.info(f"[PAGINATION]   - Limit requested: {limit}")
        if limit:
            logger.info(f"[PAGINATION]   - Limit reached: {len(all_items) >= limit if limit else 'N/A'}")
            logger.info(f"[PAGINATION]   - Position range: {start_offset} to {start_offset + len(all_items) - 1}")
        logger.info(f"[PAGINATION] ========================================")
        
        if limit is not None and limit > 0:
            logger.info(f"[FINAL] Pagination complete. Limit: {limit}, Final count: {len(all_items)}")
        else:
            logger.info(f"[FINAL] Pagination complete. No limit, Final count: {len(all_items)}")
        
        return all_items

