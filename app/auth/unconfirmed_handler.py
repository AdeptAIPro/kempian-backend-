"""
Unconfirmed User Handler
Handles users who failed OTP confirmation and are stuck in UNCONFIRMED state
"""

import boto3
from botocore.exceptions import ClientError
from app.simple_logger import get_logger
from app.auth.cognito import COGNITO_USER_POOL_ID, COGNITO_CLIENT_ID, COGNITO_CLIENT_SECRET, get_secret_hash

logger = get_logger("unconfirmed_handler")

cognito_client = boto3.client('cognito-idp', region_name='ap-south-1')

def check_user_status(email: str) -> dict:
    """
    Check the current status of a user in Cognito
    Returns: {
        'exists': bool,
        'status': str,  # 'UNCONFIRMED', 'CONFIRMED', 'FORCE_CHANGE_PASSWORD', etc.
        'user_attributes': dict,
        'can_resend_confirmation': bool,
        'can_reset_password': bool
    }
    """
    try:
        response = cognito_client.admin_get_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email
        )
        
        user_status = response.get('UserStatus', 'UNKNOWN')
        user_attributes = {attr['Name']: attr['Value'] for attr in response.get('UserAttributes', [])}
        
        return {
            'exists': True,
            'status': user_status,
            'user_attributes': user_attributes,
            'can_resend_confirmation': user_status == 'UNCONFIRMED',
            'can_reset_password': user_status in ['UNCONFIRMED', 'CONFIRMED'],
            'needs_password_reset': user_status == 'FORCE_CHANGE_PASSWORD'
        }
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'UserNotFoundException':
            return {
                'exists': False,
                'status': 'NOT_FOUND',
                'user_attributes': {},
                'can_resend_confirmation': False,
                'can_reset_password': False,
                'needs_password_reset': False
            }
        else:
            logger.error(f"Error checking user status for {email}: {str(e)}")
            raise e

def resend_confirmation_code(email: str) -> dict:
    """
    Resend confirmation code for unconfirmed users
    """
    try:
        response = cognito_client.resend_confirmation_code(
            ClientId=COGNITO_CLIENT_ID,
            Username=email,
            SecretHash=get_secret_hash(email)
        )
        
        logger.info(f"Confirmation code resent for {email}")
        return {
            'success': True,
            'message': 'Confirmation code sent successfully. Please check your email.',
            'delivery_medium': response.get('CodeDeliveryDetails', {}).get('DeliveryMedium', 'EMAIL')
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'UserNotFoundException':
            return {
                'success': False,
                'message': 'No account found with this email address.',
                'error_code': error_code
            }
        elif error_code == 'InvalidParameterException':
            return {
                'success': False,
                'message': 'This account is already confirmed.',
                'error_code': error_code
            }
        elif error_code == 'LimitExceededException':
            return {
                'success': False,
                'message': 'Too many attempts. Please try again later.',
                'error_code': error_code
            }
        else:
            logger.error(f"Error resending confirmation for {email}: {str(e)}")
            return {
                'success': False,
                'message': f'Failed to resend confirmation: {e.response["Error"]["Message"]}',
                'error_code': error_code
            }

def initiate_password_reset(email: str) -> dict:
    """
    Initiate password reset for unconfirmed users
    This will delete the unconfirmed user and allow them to signup again
    """
    try:
        # First check if user exists and is unconfirmed
        user_status = check_user_status(email)
        
        if not user_status['exists']:
            return {
                'success': False,
                'message': 'No account found with this email address.',
                'error_code': 'UserNotFoundException'
            }
        
        if user_status['status'] != 'UNCONFIRMED':
            return {
                'success': False,
                'message': 'This account is already confirmed. Use regular password reset instead.',
                'error_code': 'InvalidUserStatus'
            }
        
        # Delete the unconfirmed user
        cognito_client.admin_delete_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email
        )
        
        logger.info(f"Deleted unconfirmed user {email} to allow re-signup")
        
        return {
            'success': True,
            'message': 'Account reset successfully. You can now sign up again with this email.',
            'action': 'deleted_unconfirmed_user'
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"Error resetting password for {email}: {str(e)}")
        return {
            'success': False,
            'message': f'Failed to reset account: {e.response["Error"]["Message"]}',
            'error_code': error_code
        }

def confirm_signup_with_reset(email: str, code: str) -> dict:
    """
    Confirm signup with code, with fallback to reset if confirmation fails
    """
    try:
        # Try to confirm the signup
        response = cognito_client.confirm_sign_up(
            ClientId=COGNITO_CLIENT_ID,
            Username=email,
            ConfirmationCode=code,
            SecretHash=get_secret_hash(email)
        )
        
        return {
            'success': True,
            'message': 'Account confirmed successfully!',
            'action': 'confirmed'
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        
        if error_code == 'CodeMismatchException':
            return {
                'success': False,
                'message': 'Invalid confirmation code. Please check your email and try again.',
                'error_code': error_code,
                'can_retry': True
            }
        elif error_code == 'ExpiredCodeException':
            return {
                'success': False,
                'message': 'Confirmation code expired. Please request a new one.',
                'error_code': error_code,
                'can_retry': True
            }
        elif error_code == 'NotAuthorizedException':
            # User might be already confirmed or in wrong state
            user_status = check_user_status(email)
            if user_status['status'] == 'CONFIRMED':
                return {
                    'success': True,
                    'message': 'Account is already confirmed! You can log in now.',
                    'action': 'already_confirmed'
                }
            else:
                return {
                    'success': False,
                    'message': 'Account is in an invalid state. Please contact support.',
                    'error_code': error_code,
                    'can_reset': True
                }
        else:
            logger.error(f"Error confirming signup for {email}: {str(e)}")
            return {
                'success': False,
                'message': f'Confirmation failed: {e.response["Error"]["Message"]}',
                'error_code': error_code,
                'can_reset': True
            }

def get_recovery_options(email: str) -> dict:
    """
    Get available recovery options for a user
    """
    user_status = check_user_status(email)
    
    if not user_status['exists']:
        return {
            'email': email,
            'exists': False,
            'options': ['signup'],
            'message': 'No account found. You can sign up for a new account.'
        }
    
    options = []
    message = ""
    
    if user_status['status'] == 'UNCONFIRMED':
        options = ['resend_confirmation', 'reset_account']
        message = "Your account is not confirmed. You can either resend the confirmation code or reset your account to sign up again."
    elif user_status['status'] == 'CONFIRMED':
        options = ['login', 'forgot_password']
        message = "Your account is confirmed. You can log in or reset your password if needed."
    elif user_status['status'] == 'FORCE_CHANGE_PASSWORD':
        options = ['reset_password']
        message = "Your account requires a password reset. Please use the password reset option."
    else:
        options = ['contact_support']
        message = f"Your account is in an unusual state ({user_status['status']}). Please contact support."
    
    return {
        'email': email,
        'exists': True,
        'status': user_status['status'],
        'options': options,
        'message': message,
        'can_resend_confirmation': user_status['can_resend_confirmation'],
        'can_reset_password': user_status['can_reset_password']
    }
