"""Routes for candidate file upload to DynamoDB"""

import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.simple_logger import get_logger
from app.models import User
import jwt

from .service import upload_file_to_dynamodb

logger = get_logger("candidate_upload")

upload_bp = Blueprint('candidate_upload', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'json'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_jwt_payload():
    """Extract JWT payload from request"""
    auth = request.headers.get('Authorization', None)
    if not auth:
        logger.error("No Authorization header found")
        return None
    
    # Check if it's a Bearer token
    if not auth.startswith('Bearer '):
        logger.error(f"Invalid Authorization header format: {auth}")
        return None
    
    try:
        token = auth.split(' ')[1]
        if not token or len(token.split('.')) != 3:
            logger.error(f"Invalid JWT token format: {token}")
            return None
        
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload
    except Exception as e:
        logger.error(f"Error decoding JWT: {str(e)}", exc_info=True)
        return None


def get_user_from_jwt(payload):
    """Get user from JWT payload"""
    if not payload:
        logger.error("get_user_from_jwt: No payload provided")
        return None
    
    email = payload.get('email')
    
    if not email:
        logger.error("get_user_from_jwt: No email in payload")
        return None
    
    user = User.query.filter_by(email=email).first()
    return user


@upload_bp.route('/upload', methods=['POST'])
def upload_candidates():
    """
    Upload candidate CSV or JSON file to DynamoDB
    
    Expected request:
    - multipart/form-data with 'file' field containing the CSV/JSON file
    
    Returns:
    - JSON response with upload results
    """
    try:
        # Authenticate user
        payload = get_jwt_payload()
        if not payload:
            logger.error('Unauthorized: No JWT payload')
            return jsonify({'error': 'Unauthorized: No JWT payload'}), 403
        
        user = get_user_from_jwt(payload)
        if not user:
            logger.error(f'User not found for payload: {payload}')
            return jsonify({'error': 'User not found'}), 404
        
        logger.info(f"Upload request from user: {user.email}")
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided. Please upload a CSV or JSON file.'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get file type
        file_type = file.filename.rsplit('.', 1)[1].lower()
        
        # Read file content
        try:
            file_content = file.read()
            
            # Check file size
            if len(file_content) > MAX_FILE_SIZE:
                return jsonify({
                    'error': f'File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024 * 1024):.0f}MB'
                }), 400
            
            logger.info(f"Processing {file_type.upper()} file: {file.filename} ({len(file_content)} bytes)")
            
            # Upload to DynamoDB
            result = upload_file_to_dynamodb(file_content, file_type)
            
            if result['success']:
                logger.info(f"Successfully uploaded candidates for user {user.email}")
                return jsonify({
                    'success': True,
                    'message': result['message'],
                    'results': result['results'],
                    'filename': secure_filename(file.filename)
                }), 200
            else:
                logger.error(f"Upload failed for user {user.email}: {result.get('error')}")
                return jsonify({
                    'success': False,
                    'error': result.get('message', 'Upload failed'),
                    'details': result.get('error')
                }), 400
                
        except Exception as e:
            logger.error(f"Error reading file: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'Error reading file: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in upload endpoint: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@upload_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        from .service import table, dynamodb
        
        if dynamodb and table:
            return jsonify({
                'status': 'healthy',
                'dynamodb_connected': True,
                'table_name': table.name,
                'region': os.getenv('AWS_REGION', 'ap-south-1')
            }), 200
        else:
            return jsonify({
                'status': 'unhealthy',
                'dynamodb_connected': False,
                'message': 'DynamoDB not initialized. Check AWS credentials.'
            }), 503
            
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

