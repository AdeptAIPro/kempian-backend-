"""
Jobvite API v2 Client
Handles authentication (HMAC-SHA256) and API calls.
"""

import requests
import time
import base64
import hmac
import hashlib
from typing import Dict, List, Optional, Any
from app.simple_logger import get_logger
from app.jobvite.logging_utils import log_jobvite_request

logger = get_logger("jobvite_v2_client")

class JobviteV2Client:
    """Client for Jobvite API v2"""
    
    def __init__(self, api_key: str, api_secret: str, company_id: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.company_id = company_id
        self.base_url = base_url.rstrip('/')
    
    def _generate_hmac_signature(self, epoch: int) -> str:
        """
        Generate HMAC-SHA256 signature for authentication.
        
        Format: <apiKey>|<epoch>
        Signature: Base64(HMAC-SHA256(string, secret))
        """
        message = f"{self.api_key}|{epoch}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Generate authentication headers"""
        epoch = int(time.time())
        signature = self._generate_hmac_signature(epoch)
        
        return {
            'X-JVI-API': self.api_key,
            'X-JVI-SIGN': signature,
            'X-JVI-EPOCH': str(epoch),
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     json_data: Optional[Dict] = None, retry_count: int = 0) -> requests.Response:
        """
        Make authenticated request to Jobvite API with retry logic for rate limits.
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_auth_headers()
        
        start_time = time.time()
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=30
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
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
            candidate_id: Jobvite candidate ID
            application_id: Jobvite application ID
            filters: Additional filters (workflowState, etc.)
            start: Pagination start index
            count: Number of results per page
        
        Returns:
            Candidate data or list of candidates
        """
        endpoint = "/candidates"
        params = {
            'start': start,
            'count': count
        }
        
        if candidate_id:
            endpoint = f"/candidates/{candidate_id}"
            params = {}
        elif application_id:
            params['applicationId'] = application_id
        
        if filters:
            params.update(filters)
        
        response = self._make_request('GET', endpoint, params=params)
        return response.json()
    
    def get_candidate_with_artifacts(self, candidate_id: str) -> Dict[str, Any]:
        """
        Get candidate with encoded artifacts (documents).
        
        This endpoint returns base64-encoded documents.
        """
        endpoint = f"/candidates/{candidate_id}/artifacts"
        response = self._make_request('GET', endpoint)
        return response.json()
    
    def get_job(self, job_id: Optional[str] = None, 
               requisition_id: Optional[str] = None,
               filters: Optional[Dict] = None,
               start: int = 0,
               count: int = 50) -> Dict[str, Any]:
        """
        Get job(s) from Jobvite.
        
        Args:
            job_id: Jobvite job ID
            requisition_id: Requisition ID
            filters: Additional filters (status, location, etc.)
            start: Pagination start index
            count: Number of results per page
        
        Returns:
            Job data or paginated list of jobs
        
        Note: Handle 'total=0 on last page' quirk per Jobvite docs.
        """
        endpoint = "/jobs"
        params = {
            'start': start,
            'count': count
        }
        
        if job_id:
            endpoint = f"/jobs/{job_id}"
            params = {}
        elif requisition_id:
            params['requisitionId'] = requisition_id
        
        if filters:
            params.update(filters)
        
        response = self._make_request('GET', endpoint, params=params)
        data = response.json()
        
        # Handle pagination quirk: if total=0 on last page, we've reached the end
        if isinstance(data, dict) and 'total' in data:
            if data.get('total') == 0 and start > 0:
                # This is the last page
                return {'jobs': [], 'total': 0, 'start': start, 'count': count}
        
        return data
    
    def paginate_all(self, fetch_fn, *args, **kwargs) -> List[Dict]:
        """
        Helper to paginate through all results.
        
        Args:
            fetch_fn: Function that accepts start, count and returns {'items': [...], 'total': N}
            *args, **kwargs: Additional arguments for fetch_fn
        
        Returns:
            List of all items across all pages
        """
        all_items = []
        start = 0
        count = 50  # Default page size
        
        while True:
            result = fetch_fn(*args, start=start, count=count, **kwargs)
            
            if isinstance(result, dict):
                items = result.get('jobs', result.get('candidates', result.get('items', [])))
                total = result.get('total', 0)
                
                all_items.extend(items)
                
                # Check if we've reached the end
                # Jobvite quirk: total=0 on last page
                if len(items) < count or (total == 0 and start > 0):
                    break
                
                start += count
            else:
                # Single item response
                all_items.append(result)
                break
        
        return all_items

