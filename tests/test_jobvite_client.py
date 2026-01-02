"""
Unit tests for Jobvite API clients.
Tests authentication, request building, and error handling.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from app.jobvite.client_v2 import JobviteV2Client

class TestJobviteV2Client(unittest.TestCase):
    """Test JobviteV2Client"""
    
    def setUp(self):
        """Set up test client"""
        self.client = JobviteV2Client(
            api_key="test_api_key",
            api_secret="test_api_secret",
            company_id="test_company",
            base_url="https://api.jobvite.com/v2"
        )
    
    def test_auth_headers_generation(self):
        """Test HMAC signature generation"""
        headers = self.client._get_auth_headers()
        
        self.assertIn("X-JVI-API", headers)
        self.assertIn("X-JVI-SIGN", headers)
        self.assertIn("X-JVI-EPOCH", headers)
        self.assertEqual(headers["X-JVI-API"], "test_api_key")
        self.assertIsNotNone(headers["X-JVI-SIGN"])
        self.assertIsNotNone(headers["X-JVI-EPOCH"])
    
    @patch('app.jobvite.client_v2.requests.request')
    def test_rate_limit_retry(self, mock_request):
        """Test rate limit retry logic"""
        # Mock 429 response, then 200
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}
        
        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"jobs": []}
        mock_response_200.raise_for_status = Mock()
        
        mock_request.side_effect = [mock_response_429, mock_response_200]
        
        with patch('time.sleep'):  # Skip actual sleep
            response = self.client._make_request("GET", "/jobs")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(mock_request.call_count, 2)
    
    @patch('app.jobvite.client_v2.requests.request')
    def test_max_retries_exceeded(self, mock_request):
        """Test that max retries are enforced"""
        # Mock 429 responses
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "1"}
        mock_response.raise_for_status = Mock(side_effect=Exception("Rate limited"))
        
        mock_request.return_value = mock_response
        
        with patch('time.sleep'):  # Skip actual sleep
            with self.assertRaises(Exception):
                self.client._make_request("GET", "/jobs", retry_count=3)

if __name__ == '__main__':
    unittest.main()

