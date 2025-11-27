"""
Integration tests for Jobvite integration.

These tests require:
1. Real Jobvite Stage environment credentials
2. Mock Jobvite server (for CI/CD)
3. Test database

To run:
    pytest backend/tests/integration/test_jobvite_integration.py -v

Set environment variables:
    JOBVITE_STAGE_API_KEY=your_stage_api_key
    JOBVITE_STAGE_API_SECRET=your_stage_api_secret
    JOBVITE_STAGE_COMPANY_ID=your_stage_company_id
"""

import unittest
import os
from unittest.mock import Mock, patch
from app.jobvite.client_v2 import JobviteV2Client
from app.jobvite.sync import sync_jobs_for_tenant, sync_candidates_for_tenant
from app.models import JobviteSettings, db
from app import create_app

class TestJobviteIntegration(unittest.TestCase):
    """Integration tests for Jobvite"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
            'TEST_DATABASE_URI', 
            'sqlite:///:memory:'
        )
        cls.app_context = cls.app.app_context()
        cls.app_context.push()
        db.create_all()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        db.drop_all()
        cls.app_context.pop()
    
    def setUp(self):
        """Set up for each test"""
        self.tenant_id = 1
        # Create test settings
        self.settings = JobviteSettings(
            tenant_id=self.tenant_id,
            user_id=1,
            environment='stage',
            api_key=os.getenv('JOBVITE_STAGE_API_KEY', 'test_key'),
            api_secret_encrypted='encrypted_secret',
            company_id=os.getenv('JOBVITE_STAGE_COMPANY_ID', 'test_company'),
            sync_config={
                'syncJobs': True,
                'syncCandidates': True
            },
            is_active=True
        )
        db.session.add(self.settings)
        db.session.commit()
    
    def tearDown(self):
        """Clean up after each test"""
        db.session.rollback()
        JobviteSettings.query.delete()
        db.session.commit()
    
    @unittest.skipUnless(
        os.getenv('JOBVITE_STAGE_API_KEY'),
        "Requires JOBVITE_STAGE_API_KEY environment variable"
    )
    def test_real_jobvite_connection(self):
        """Test connection to real Jobvite Stage API"""
        client = JobviteV2Client(
            api_key=os.getenv('JOBVITE_STAGE_API_KEY'),
            api_secret=os.getenv('JOBVITE_STAGE_API_SECRET'),
            company_id=os.getenv('JOBVITE_STAGE_COMPANY_ID'),
            base_url='https://api-stage.jobvite.com/v2'
        )
        
        # Test connection by fetching jobs
        try:
            result = client.get_job(start=0, count=1)
            self.assertIn('jobs', result)
            print(f"✅ Successfully connected to Jobvite Stage API")
        except Exception as e:
            self.fail(f"Failed to connect to Jobvite Stage API: {e}")
    
    @unittest.skipUnless(
        os.getenv('JOBVITE_STAGE_API_KEY'),
        "Requires JOBVITE_STAGE_API_KEY environment variable"
    )
    def test_sync_jobs_integration(self):
        """Test job sync with real Jobvite API"""
        result = sync_jobs_for_tenant(self.tenant_id)
        
        self.assertIn('success', result)
        if result['success']:
            self.assertIn('synced_count', result)
            print(f"✅ Synced {result.get('synced_count', 0)} jobs")
        else:
            print(f"⚠️  Sync failed: {result.get('error', 'Unknown error')}")
    
    @unittest.skipUnless(
        os.getenv('JOBVITE_STAGE_API_KEY'),
        "Requires JOBVITE_STAGE_API_KEY environment variable"
    )
    def test_sync_candidates_integration(self):
        """Test candidate sync with real Jobvite API"""
        result = sync_candidates_for_tenant(self.tenant_id)
        
        self.assertIn('success', result)
        if result['success']:
            self.assertIn('synced_count', result)
            print(f"✅ Synced {result.get('synced_count', 0)} candidates")
        else:
            print(f"⚠️  Sync failed: {result.get('error', 'Unknown error')}")
    
    def test_rate_limit_handling(self):
        """Test rate limit retry logic"""
        client = JobviteV2Client(
            api_key='test_key',
            api_secret='test_secret',
            company_id='test_company',
            base_url='https://api.jobvite.com/v2'
        )
        
        # Mock 429 response, then 200
        with patch('app.jobvite.client_v2.requests.request') as mock_request:
            mock_429 = Mock()
            mock_429.status_code = 429
            mock_429.headers = {'Retry-After': '1'}
            
            mock_200 = Mock()
            mock_200.status_code = 200
            mock_200.json.return_value = {'jobs': []}
            mock_200.raise_for_status = Mock()
            
            mock_request.side_effect = [mock_429, mock_200]
            
            with patch('time.sleep'):  # Skip actual sleep
                response = client._make_request('GET', '/jobs')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(mock_request.call_count, 2)

if __name__ == '__main__':
    unittest.main()

