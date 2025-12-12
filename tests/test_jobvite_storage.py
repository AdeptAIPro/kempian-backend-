"""
Unit tests for Jobvite document storage.
Tests S3 upload, retrieval, and URL generation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import base64
from app.jobvite.storage import (
    upload_document_from_base64,
    get_document_from_s3,
    get_document_url,
    delete_document_from_s3
)

class TestJobviteStorage(unittest.TestCase):
    """Test document storage functions"""
    
    def setUp(self):
        """Set up test data"""
        self.test_content = "This is a test document"
        self.test_base64 = base64.b64encode(self.test_content.encode()).decode()
        self.test_filename = "test_resume.pdf"
    
    @patch('app.jobvite.storage.s3_client')
    def test_upload_document_success(self, mock_s3_client):
        """Test successful document upload"""
        mock_s3_client.upload_fileobj = Mock()
        
        s3_key, s3_url, public_url = upload_document_from_base64(
            content_base64=self.test_base64,
            filename=self.test_filename,
            tenant_id=1,
            candidate_id=123,
            doc_type="resume"
        )
        
        self.assertIsNotNone(s3_key)
        self.assertIsNotNone(s3_url)
        self.assertIsNotNone(public_url)
        self.assertIn("jobvite/documents/1/123/", s3_key)
        mock_s3_client.upload_fileobj.assert_called_once()
    
    @patch('app.jobvite.storage.s3_client')
    def test_upload_document_invalid_base64(self, mock_s3_client):
        """Test upload with invalid base64"""
        s3_key, s3_url, public_url = upload_document_from_base64(
            content_base64="invalid_base64!!!",
            filename=self.test_filename,
            tenant_id=1,
            candidate_id=123
        )
        
        self.assertIsNone(s3_key)
        self.assertIsNone(s3_url)
        self.assertIsNone(public_url)
    
    @patch('app.jobvite.storage.s3_client')
    def test_get_document_from_s3(self, mock_s3_client):
        """Test document retrieval from S3"""
        mock_response = Mock()
        mock_response['Body'].read.return_value = self.test_content.encode()
        mock_s3_client.get_object.return_value = mock_response
        
        content = get_document_from_s3("jobvite/documents/1/123/test.pdf")
        
        self.assertEqual(content, self.test_content.encode())
        mock_s3_client.get_object.assert_called_once()
    
    @patch('app.jobvite.storage.s3_client')
    def test_get_document_url(self, mock_s3_client):
        """Test presigned URL generation"""
        mock_s3_client.generate_presigned_url.return_value = "https://presigned-url.com/test"
        
        url = get_document_url("jobvite/documents/1/123/test.pdf", expires_in=3600)
        
        self.assertEqual(url, "https://presigned-url.com/test")
        mock_s3_client.generate_presigned_url.assert_called_once()
    
    @patch('app.jobvite.storage.s3_client')
    def test_delete_document(self, mock_s3_client):
        """Test document deletion"""
        mock_s3_client.delete_object = Mock()
        
        result = delete_document_from_s3("jobvite/documents/1/123/test.pdf")
        
        self.assertTrue(result)
        mock_s3_client.delete_object.assert_called_once()

if __name__ == '__main__':
    unittest.main()

