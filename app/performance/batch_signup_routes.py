# Batch Jobseeker Signup API Routes
# High-performance endpoints for bulk jobseeker registration

import time
import logging
from flask import Blueprint, request, jsonify, current_app
from app.simple_logger import get_logger
from app.models import db, User
from app.search.routes import get_user_from_jwt, get_jwt_payload
from .batch_jobseeker_signup import signup_jobseekers_fast, create_jobseeker_data_from_dict

logger = get_logger("batch_signup_api")

# Create blueprint
batch_signup_bp = Blueprint('batch_signup', __name__)

@batch_signup_bp.route('/jobseekers/batch-signup', methods=['POST'])
def batch_signup_jobseekers():
    """
    Batch signup endpoint for multiple jobseekers
    Handles 1500+ jobseekers with parallel processing
    """
    try:
        # Get user from JWT token (admin only)
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user is admin
        if not user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        jobseeker_data_list = data.get('jobseekers', [])
        max_concurrent = data.get('max_concurrent', 50)
        batch_size = data.get('batch_size', 100)
        
        if not jobseeker_data_list:
            return jsonify({'error': 'No jobseeker data provided'}), 400
        
        # Validate jobseeker data
        validation_errors = validate_jobseeker_data(jobseeker_data_list)
        if validation_errors:
            return jsonify({
                'error': 'Validation failed',
                'details': validation_errors
            }), 400
        
        # Start batch signup
        start_time = time.time()
        
        # Run async signup
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                signup_jobseekers_fast(
                    jobseeker_data_list,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size
                )
            )
        finally:
            loop.close()
        
        total_time = time.time() - start_time
        
        # Process results
        successful_signups = [r for r in results if r.success]
        failed_signups = [r for r in results if not r.success]
        
        response = {
            'message': 'Batch signup completed',
            'summary': {
                'total_jobseekers': len(jobseeker_data_list),
                'successful_signups': len(successful_signups),
                'failed_signups': len(failed_signups),
                'success_rate': (len(successful_signups) / len(jobseeker_data_list)) * 100,
                'total_time': total_time,
                'signups_per_minute': (len(jobseeker_data_list) / total_time) * 60
            },
            'results': [
                {
                    'email': r.email,
                    'success': r.success,
                    'user_id': r.user_id,
                    'error_message': r.error_message,
                    'processing_time': r.processing_time
                }
                for r in results
            ],
            'performance': {
                'max_concurrent': max_concurrent,
                'batch_size': batch_size,
                'timestamp': time.time()
            }
        }
        
        logger.info(f"Batch signup completed: {len(successful_signups)}/{len(jobseeker_data_list)} successful in {total_time:.2f}s")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in batch signup: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@batch_signup_bp.route('/jobseekers/batch-signup-status', methods=['GET'])
def get_batch_signup_status():
    """
    Get status of batch signup operations
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get jobseeker statistics
        total_jobseekers = db.session.query(User).filter(
            User.role == 'job_seeker'
        ).count()
        
        recent_jobseekers = db.session.query(User).filter(
            User.role == 'job_seeker',
            User.created_at >= time.time() - 3600  # Last hour
        ).count()
        
        response = {
            'total_jobseekers': total_jobseekers,
            'recent_jobseekers': recent_jobseekers,
            'timestamp': time.time()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting batch signup status: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@batch_signup_bp.route('/jobseekers/validate-batch', methods=['POST'])
def validate_batch_jobseekers():
    """
    Validate jobseeker data before batch signup
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        jobseeker_data_list = data.get('jobseekers', [])
        
        if not jobseeker_data_list:
            return jsonify({'error': 'No jobseeker data provided'}), 400
        
        # Validate jobseeker data
        validation_errors = validate_jobseeker_data(jobseeker_data_list)
        
        response = {
            'valid': len(validation_errors) == 0,
            'total_jobseekers': len(jobseeker_data_list),
            'validation_errors': validation_errors,
            'timestamp': time.time()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error validating batch jobseekers: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@batch_signup_bp.route('/jobseekers/estimate-time', methods=['POST'])
def estimate_batch_signup_time():
    """
    Estimate time required for batch signup
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        jobseeker_count = data.get('jobseeker_count', 0)
        max_concurrent = data.get('max_concurrent', 50)
        batch_size = data.get('batch_size', 100)
        
        if jobseeker_count <= 0:
            return jsonify({'error': 'Invalid jobseeker count'}), 400
        
        # Calculate estimates
        estimates = calculate_signup_time_estimates(jobseeker_count, max_concurrent, batch_size)
        
        response = {
            'jobseeker_count': jobseeker_count,
            'configuration': {
                'max_concurrent': max_concurrent,
                'batch_size': batch_size
            },
            'estimates': estimates,
            'timestamp': time.time()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error estimating batch signup time: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def validate_jobseeker_data(jobseeker_data_list: list) -> list:
    """
    Validate jobseeker data for batch signup
    """
    errors = []
    
    required_fields = ['email', 'password', 'first_name', 'last_name', 'phone', 'location', 'resume_file_path', 'resume_filename']
    
    for i, jobseeker_data in enumerate(jobseeker_data_list):
        jobseeker_errors = []
        
        # Check required fields
        for field in required_fields:
            if field not in jobseeker_data or not jobseeker_data[field]:
                jobseeker_errors.append(f"Missing required field: {field}")
        
        # Validate email format
        if 'email' in jobseeker_data:
            email = jobseeker_data['email']
            if '@' not in email or '.' not in email:
                jobseeker_errors.append("Invalid email format")
        
        # Validate password strength
        if 'password' in jobseeker_data:
            password = jobseeker_data['password']
            if len(password) < 8:
                jobseeker_errors.append("Password must be at least 8 characters")
        
        # Validate file extension
        if 'resume_filename' in jobseeker_data:
            filename = jobseeker_data['resume_filename']
            allowed_extensions = ['pdf', 'doc', 'docx', 'txt']
            file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            if file_extension not in allowed_extensions:
                jobseeker_errors.append(f"Invalid file extension: {file_extension}")
        
        if jobseeker_errors:
            errors.append({
                'index': i,
                'email': jobseeker_data.get('email', 'Unknown'),
                'errors': jobseeker_errors
            })
    
    return errors

def calculate_signup_time_estimates(jobseeker_count: int, max_concurrent: int, batch_size: int) -> dict:
    """
    Calculate time estimates for batch signup
    """
    # Base time per signup (in seconds)
    base_time_per_signup = 3.0  # Average time for single signup
    
    # Calculate batches needed
    batches_needed = (jobseeker_count + batch_size - 1) // batch_size
    
    # Calculate time per batch
    time_per_batch = (batch_size / max_concurrent) * base_time_per_signup
    
    # Calculate total time
    total_time_seconds = batches_needed * time_per_batch
    
    # Convert to different time units
    total_time_minutes = total_time_seconds / 60
    total_time_hours = total_time_minutes / 60
    
    # Calculate signups per minute
    signups_per_minute = (jobseeker_count / total_time_seconds) * 60
    
    return {
        'total_time_seconds': round(total_time_seconds, 2),
        'total_time_minutes': round(total_time_minutes, 2),
        'total_time_hours': round(total_time_hours, 2),
        'batches_needed': batches_needed,
        'time_per_batch_seconds': round(time_per_batch, 2),
        'signups_per_minute': round(signups_per_minute, 2),
        'estimated_success_rate': 95.0  # Based on historical data
    }
