"""
Document storage utilities for Jobvite integration.
Handles S3 upload and retrieval of candidate documents.
"""

import os
import base64
import uuid
from datetime import datetime
from typing import Optional, Tuple
import boto3
from io import BytesIO
from app.simple_logger import get_logger

logger = get_logger("jobvite_storage")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-south-1')
)

# S3 bucket for Jobvite documents
JOBVITE_DOCUMENTS_BUCKET = os.getenv('JOBVITE_DOCUMENTS_BUCKET', 'jobvite-documents')
JOBVITE_DOCUMENTS_PREFIX = os.getenv('JOBVITE_DOCUMENTS_PREFIX', 'jobvite/documents/')

def upload_document_from_base64(
    content_base64: str,
    filename: str,
    tenant_id: int,
    candidate_id: int,
    doc_type: str = 'attachment'
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Upload a document from base64 content to S3.
    
    Args:
        content_base64: Base64-encoded document content
        filename: Original filename
        tenant_id: Tenant ID
        candidate_id: Candidate ID
        doc_type: Document type (resume, cover_letter, attachment)
    
    Returns:
        Tuple of (s3_key, s3_url, public_url) or (None, None, None) on error
    """
    try:
        # Decode base64 content
        try:
            file_content = base64.b64decode(content_base64)
        except Exception as e:
            logger.error(f"Failed to decode base64 content: {e}")
            return None, None, None
        
        # Generate unique S3 key
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'pdf'
        safe_filename = filename.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # S3 key: jobvite/documents/{tenant_id}/{candidate_id}/{timestamp}_{unique_id}_{filename}
        s3_key = f"{JOBVITE_DOCUMENTS_PREFIX}{tenant_id}/{candidate_id}/{timestamp}_{unique_id}_{safe_filename}"
        
        # Determine content type
        content_type_map = {
            'pdf': 'application/pdf',
            'doc': 'application/msword',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'txt': 'text/plain',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png'
        }
        content_type = content_type_map.get(file_extension, 'application/octet-stream')
        
        # Upload to S3
        file_obj = BytesIO(file_content)
        s3_client.upload_fileobj(
            file_obj,
            JOBVITE_DOCUMENTS_BUCKET,
            s3_key,
            ExtraArgs={
                'ContentType': content_type,
                'ServerSideEncryption': 'AES256',
                'Metadata': {
                    'tenant_id': str(tenant_id),
                    'candidate_id': str(candidate_id),
                    'doc_type': doc_type,
                    'original_filename': filename
                }
            }
        )
        
        # Generate URLs
        s3_url = f"s3://{JOBVITE_DOCUMENTS_BUCKET}/{s3_key}"
        region = os.getenv('AWS_REGION', 'ap-south-1')
        public_url = f"https://{JOBVITE_DOCUMENTS_BUCKET}.s3.{region}.amazonaws.com/{s3_key}"
        
        logger.info(f"Successfully uploaded document to S3: {s3_key} (size: {len(file_content)} bytes)")
        
        return s3_key, s3_url, public_url
        
    except Exception as e:
        logger.error(f"Error uploading document to S3: {e}", exc_info=True)
        return None, None, None

def get_document_from_s3(s3_key: str) -> Optional[bytes]:
    """
    Retrieve a document from S3.
    
    Args:
        s3_key: S3 key of the document
    
    Returns:
        Document content as bytes, or None on error
    """
    try:
        response = s3_client.get_object(
            Bucket=JOBVITE_DOCUMENTS_BUCKET,
            Key=s3_key
        )
        return response['Body'].read()
    except Exception as e:
        logger.error(f"Error retrieving document from S3: {s3_key}: {e}")
        return None

def get_document_url(s3_key: str, expires_in: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for document access.
    
    Args:
        s3_key: S3 key of the document
        expires_in: URL expiration time in seconds (default: 1 hour)
    
    Returns:
        Presigned URL or None on error
    """
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': JOBVITE_DOCUMENTS_BUCKET,
                'Key': s3_key
            },
            ExpiresIn=expires_in
        )
        return url
    except Exception as e:
        logger.error(f"Error generating presigned URL for {s3_key}: {e}")
        return None

def delete_document_from_s3(s3_key: str) -> bool:
    """
    Delete a document from S3.
    
    Args:
        s3_key: S3 key of the document
    
    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client.delete_object(
            Bucket=JOBVITE_DOCUMENTS_BUCKET,
            Key=s3_key
        )
        logger.info(f"Successfully deleted document from S3: {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error deleting document from S3: {s3_key}: {e}")
        return False

