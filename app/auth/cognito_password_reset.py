import boto3
import os
import logging
from flask import Blueprint, request, jsonify, current_app
from app.simple_logger import get_logger
from botocore.exceptions import ClientError
from app.models import User
from app import db

logger = get_logger("auth")

cognito_password_reset_bp = Blueprint('cognito_password_reset', __name__)

# Initialize Cognito client with explicit region to avoid NoRegionError
cognito_client = boto3.client(
    'cognito-idp',
    region_name=os.getenv('AWS_REGION', os.getenv('COGNITO_REGION', 'ap-south-1'))
)

@cognito_password_reset_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Request password reset using Cognito"""
    try:
        data = request.get_json()
        if not data or not data.get('email'):
            return jsonify({'error': 'Email is required'}), 400
        
        email = data['email'].lower().strip()
        
        # Check if user exists in our database
        user = User.query.filter_by(email=email).first()
        if not user:
            # Don't reveal if user exists or not for security
            return jsonify({'message': 'If an account with this email exists, a password reset code has been sent'}), 200
        
        # Use Cognito to send password reset code
        try:
            response = cognito_client.forgot_password(
                ClientId=current_app.config.get('COGNITO_CLIENT_ID'),
                Username=email
            )
            
            logger.info(f"Password reset code sent via Cognito for user {email}")
            return jsonify({
                'message': 'Password reset code has been sent to your email',
                'delivery_medium': response.get('CodeDeliveryDetails', {}).get('DeliveryMedium', 'EMAIL'),
                'destination': response.get('CodeDeliveryDetails', {}).get('Destination', email)
            }), 200
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'UserNotFoundException':
                # Don't reveal if user exists or not for security
                return jsonify({'message': 'If an account with this email exists, a password reset code has been sent'}), 200
            elif error_code == 'LimitExceededException':
                return jsonify({'error': 'Too many password reset attempts. Please try again later.'}), 429
            elif error_code == 'InvalidParameterException':
                return jsonify({'error': 'Invalid email format'}), 400
            else:
                logger.error(f"Cognito forgot password error: {e}")
                return jsonify({'error': 'Failed to send password reset code. Please try again.'}), 500
                
    except Exception as e:
        logger.error(f"Error in forgot_password: {e}")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

@cognito_password_reset_bp.route('/confirm-forgot-password', methods=['POST'])
def confirm_forgot_password():
    """Confirm password reset using Cognito"""
    try:
        data = request.get_json()
        if not data or not data.get('email') or not data.get('code') or not data.get('new_password'):
            return jsonify({'error': 'Email, code, and new password are required'}), 400
        
        email = data['email'].lower().strip()
        code = data['code'].strip()
        new_password = data['new_password']
        
        # Validate password strength
        if len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        # Check if user exists in our database
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'Invalid email or code'}), 400
        
        # Use Cognito to confirm password reset
        try:
            response = cognito_client.confirm_forgot_password(
                ClientId=current_app.config.get('COGNITO_CLIENT_ID'),
                Username=email,
                ConfirmationCode=code,
                Password=new_password
            )
            
            # Update user's password hash in our database if needed
            # Note: Cognito handles the actual password, but we might want to sync
            user.updated_at = db.func.now()
            db.session.commit()
            
            logger.info(f"Password reset confirmed via Cognito for user {email}")
            return jsonify({
                'message': 'Password reset successfully. You can now log in with your new password.',
                'success': True
            }), 200
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'CodeMismatchException':
                return jsonify({'error': 'Invalid or expired reset code'}), 400
            elif error_code == 'ExpiredCodeException':
                return jsonify({'error': 'Reset code has expired. Please request a new one.'}), 400
            elif error_code == 'InvalidPasswordException':
                return jsonify({'error': 'Password does not meet requirements'}), 400
            elif error_code == 'LimitExceededException':
                return jsonify({'error': 'Too many attempts. Please try again later.'}), 429
            else:
                logger.error(f"Cognito confirm forgot password error: {e}")
                return jsonify({'error': 'Failed to reset password. Please try again.'}), 500
                
    except Exception as e:
        logger.error(f"Error in confirm_forgot_password: {e}")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

@cognito_password_reset_bp.route('/resend-code', methods=['POST'])
def resend_code():
    """Resend password reset code using Cognito"""
    try:
        data = request.get_json()
        if not data or not data.get('email'):
            return jsonify({'error': 'Email is required'}), 400
        
        email = data['email'].lower().strip()
        
        # Check if user exists in our database
        user = User.query.filter_by(email=email).first()
        if not user:
            # Don't reveal if user exists or not for security
            return jsonify({'message': 'If an account with this email exists, a new password reset code has been sent'}), 200
        
        # Use Cognito to resend password reset code
        try:
            response = cognito_client.forgot_password(
                ClientId=current_app.config.get('COGNITO_CLIENT_ID'),
                Username=email
            )
            
            logger.info(f"Password reset code resent via Cognito for user {email}")
            return jsonify({
                'message': 'New password reset code has been sent to your email',
                'delivery_medium': response.get('CodeDeliveryDetails', {}).get('DeliveryMedium', 'EMAIL'),
                'destination': response.get('CodeDeliveryDetails', {}).get('Destination', email)
            }), 200
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'UserNotFoundException':
                # Don't reveal if user exists or not for security
                return jsonify({'message': 'If an account with this email exists, a new password reset code has been sent'}), 200
            elif error_code == 'LimitExceededException':
                return jsonify({'error': 'Too many attempts. Please try again later.'}), 429
            else:
                logger.error(f"Cognito resend code error: {e}")
                return jsonify({'error': 'Failed to resend code. Please try again.'}), 500
                
    except Exception as e:
        logger.error(f"Error in resend_code: {e}")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

@cognito_password_reset_bp.route('/check-user-exists', methods=['POST'])
def check_user_exists():
    """Check if user exists in Cognito (for frontend validation)"""
    try:
        data = request.get_json()
        if not data or not data.get('email'):
            return jsonify({'error': 'Email is required'}), 400
        
        email = data['email'].lower().strip()
        
        # Check if user exists in our database
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'exists': False, 'message': 'No account found with this email'}), 200
        
        # Check if user exists in Cognito
        try:
            response = cognito_client.admin_get_user(
                UserPoolId=current_app.config.get('COGNITO_USER_POOL_ID'),
                Username=email
            )
            
            user_status = response['UserStatus']
            return jsonify({
                'exists': True,
                'user_status': user_status,
                'message': 'User found'
            }), 200
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'UserNotFoundException':
                return jsonify({'exists': False, 'message': 'No account found with this email'}), 200
            else:
                logger.error(f"Error checking user in Cognito: {e}")
                return jsonify({'error': 'Unable to verify user account'}), 500
                
    except Exception as e:
        logger.error(f"Error in check_user_exists: {e}")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

# Add this to your app initialization
def init_cognito_password_reset(app):
    """Initialize Cognito password reset system"""
    app.register_blueprint(cognito_password_reset_bp, url_prefix='/auth')
