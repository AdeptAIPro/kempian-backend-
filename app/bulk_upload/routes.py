"""
Bulk Resume Upload Routes
Handles bulk upload of resumes to S3 bucket "bulkupload"
"""
import os
import uuid
import boto3
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.simple_logger import get_logger
from app.models import db, User
from app.search.routes import get_user_from_jwt, get_jwt_payload

logger = get_logger("bulk_upload")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-south-1')
)

# S3 bucket for bulk uploads
BULK_UPLOAD_BUCKET = "adeptai-payroll"
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
MAX_FILES_PER_UPLOAD = int(os.getenv('MAX_FILES_PER_UPLOAD', 100))  # Default: 100 files

bulk_upload_bp = Blueprint('bulk_upload', __name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file_to_s3(file, bucket_name, s3_key):
    """Upload a single file to S3"""
    try:
        # Determine content type based on file extension
        filename = file.filename or s3_key
        extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'pdf'
        content_type_map = {
            'pdf': 'application/pdf',
            'doc': 'application/msword',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'txt': 'text/plain'
        }
        content_type = content_type_map.get(extension, 'application/pdf')
        
        # Use file's content_type if available, otherwise use mapped type
        if hasattr(file, 'content_type') and file.content_type:
            content_type = file.content_type
        
        s3_client.upload_fileobj(
            file,
            bucket_name,
            s3_key,
            ExtraArgs={
                'ContentType': content_type,
                'ServerSideEncryption': 'AES256'
            }
        )
        return True, None
    except Exception as e:
        logger.error(f"Error uploading file to S3: {str(e)}")
        return False, str(e)

@bulk_upload_bp.route('/api/bulk-upload/resumes', methods=['POST'])
def bulk_upload_resumes():
    """
    Bulk upload resumes to S3 bucket "bulkupload"
    Accepts multiple files in a single request
    """
    try:
        # Get user from JWT token (optional - allows any user)
        user = None
        user_email = None
        user_id = None
        
        try:
            payload = get_jwt_payload()
            if payload:
                user, tenant_id = get_user_from_jwt(payload)
                if user:
                    user_email = user.email if hasattr(user, 'email') else None
                    user_id = user.id if hasattr(user, 'id') else None
        except Exception as e:
            logger.warning(f"Could not get user from JWT: {e}")
            # Continue without user - allow public access as requested
        
        # Check if files are present
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # Check file count limit
        if len(files) > MAX_FILES_PER_UPLOAD:
            return jsonify({
                'success': False,
                'error': f'Too many files. Maximum {MAX_FILES_PER_UPLOAD} files allowed per upload. You selected {len(files)} files.'
            }), 400
        
        # Process each file
        results = []
        successful_uploads = []
        failed_uploads = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                failed_uploads.append({
                    'filename': file.filename,
                    'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
                })
                continue
            
            try:
                # Generate unique S3 key - store directly in bucket root
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                file_extension = file.filename.rsplit('.', 1)[1].lower()
                secure_name = secure_filename(file.filename)
                unique_id = str(uuid.uuid4())[:8]
                
                # Store directly in bucket root (no subfolders)
                s3_key = f"{timestamp}_{unique_id}_{secure_name}"
                
                # Upload to S3
                success, error = upload_file_to_s3(file, BULK_UPLOAD_BUCKET, s3_key)
                
                if success:
                    # Get S3 URL
                    s3_url = f"s3://{BULK_UPLOAD_BUCKET}/{s3_key}"
                    public_url = f"https://{BULK_UPLOAD_BUCKET}.s3.{os.getenv('AWS_REGION', 'ap-south-1')}.amazonaws.com/{s3_key}"
                    
                    successful_uploads.append({
                        'filename': file.filename,
                        's3_key': s3_key,
                        's3_url': s3_url,
                        'public_url': public_url,
                        'size': file.content_length if hasattr(file, 'content_length') else None,
                        'uploaded_at': timestamp
                    })
                    
                    logger.info(f"Successfully uploaded {file.filename} to S3: {s3_key}")
                else:
                    failed_uploads.append({
                        'filename': file.filename,
                        'error': error or 'Unknown upload error'
                    })
                    
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                failed_uploads.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        # Prepare response
        response_data = {
            'success': len(failed_uploads) == 0,
            'total_files': len(files),
            'successful_uploads': len(successful_uploads),
            'failed_uploads': len(failed_uploads),
            'uploads': successful_uploads,
            'failures': failed_uploads
        }
        
        if len(successful_uploads) > 0:
            status_code = 200 if len(failed_uploads) == 0 else 207  # 207 Multi-Status
            return jsonify(response_data), status_code
        else:
            return jsonify(response_data), 400
            
    except Exception as e:
        logger.error(f"Error in bulk upload endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@bulk_upload_bp.route('/api/bulk-upload/status', methods=['GET'])
def get_upload_status():
    """
    Get status of bulk upload functionality
    """
    try:
        # Check S3 connection
        try:
            s3_client.head_bucket(Bucket=BULK_UPLOAD_BUCKET)
            bucket_accessible = True
        except Exception as e:
            logger.warning(f"Cannot access S3 bucket: {e}")
            bucket_accessible = False
        
        return jsonify({
            'success': True,
            'bucket_name': BULK_UPLOAD_BUCKET,
            'bucket_accessible': bucket_accessible,
            'allowed_extensions': list(ALLOWED_EXTENSIONS),
            'max_file_size': current_app.config.get('MAX_CONTENT_LENGTH', 10485760),  # 10MB default
            'max_files_per_upload': MAX_FILES_PER_UPLOAD
        }), 200
        
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

