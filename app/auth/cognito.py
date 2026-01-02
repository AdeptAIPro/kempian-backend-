import os
import boto3
from botocore.exceptions import ClientError
from app.simple_logger import get_logger

logger = get_logger("cognito")

COGNITO_REGION = os.getenv('COGNITO_REGION')
COGNITO_USER_POOL_ID = os.getenv('COGNITO_USER_POOL_ID')
COGNITO_CLIENT_ID = os.getenv('COGNITO_CLIENT_ID')
COGNITO_CLIENT_SECRET = os.getenv('COGNITO_CLIENT_SECRET')

cognito_client = boto3.client('cognito-idp', region_name=COGNITO_REGION)

def get_secret_hash(username):
    import hmac, hashlib, base64
    message = username + COGNITO_CLIENT_ID
    dig = hmac.new(
        str(COGNITO_CLIENT_SECRET).encode('utf-8'),
        msg=message.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(dig).decode()

def cognito_signup(email, password, tenant_id=None, role='owner', user_type=None, full_name=None, first_name=None, last_name=None):
    user_attributes = [
        {'Name': 'email', 'Value': email},
        {'Name': 'custom:role', 'Value': role}
    ]
    if user_type:
        user_attributes.append({'Name': 'custom:user_type', 'Value': user_type})
    if full_name:
        user_attributes.append({'Name': 'name', 'Value': full_name})
    if first_name:
        user_attributes.append({'Name': 'given_name', 'Value': first_name})
    if last_name:
        user_attributes.append({'Name': 'family_name', 'Value': last_name})
    if tenant_id:
        user_attributes.append({'Name': 'custom:tenant_id', 'Value': str(tenant_id)})
    try:
        response = cognito_client.sign_up(
            ClientId=COGNITO_CLIENT_ID,
            Username=email,
            Password=password,
            UserAttributes=user_attributes
        )
        return response
    except ClientError as e:
        raise e

def cognito_confirm_signup(email, code):
    try:
        response = cognito_client.confirm_sign_up(
            ClientId=COGNITO_CLIENT_ID,
            Username=email,
            ConfirmationCode=code
        )
        return response
    except ClientError as e:
        raise e

def cognito_login(email, password):
    try:
        auth_params = {
            'USERNAME': email,
            'PASSWORD': password
        }
        
        # Add SECRET_HASH if client secret is configured
        if COGNITO_CLIENT_SECRET:
            auth_params['SECRET_HASH'] = get_secret_hash(email)
        
        response = cognito_client.initiate_auth(
            ClientId=COGNITO_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters=auth_params
        )
        
        # Check if challenge is required (e.g., NEW_PASSWORD_REQUIRED for temporary passwords)
        if 'ChallengeName' in response:
            # Return challenge info so login endpoint can handle it
            return {
                'ChallengeName': response['ChallengeName'],
                'Session': response['Session'],
                'ChallengeParameters': response.get('ChallengeParameters', {})
            }
        
        return response['AuthenticationResult']
    except ClientError as e:
        raise e

def cognito_admin_create_user(email, tenant_id, role='subuser', temp_password=None):
    import random, string
    if not temp_password:
        temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12)) + '1!A'
    try:
        # Extract name from email (use email prefix as name)
        name = email.split('@')[0]
        
        # Set both custom:role and custom:user_type to ensure proper role assignment
        # For employees, both should be 'employee'; for subusers, both should be 'subuser'
        user_attributes = [
            {'Name': 'email', 'Value': email},
            {'Name': 'email_verified', 'Value': 'true'},  # Mark email as verified
            {'Name': 'name', 'Value': name},  # Include name attribute to avoid missing name error
            {'Name': 'custom:tenant_id', 'Value': str(tenant_id)},
            {'Name': 'custom:role', 'Value': role},
            {'Name': 'custom:user_type', 'Value': role}  # Set user_type to match role for consistency
        ]
        
        response = cognito_client.admin_create_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email,
            UserAttributes=user_attributes,
            TemporaryPassword=temp_password,
            MessageAction='SUPPRESS'  # We'll send our own invite email
        )
        return temp_password, response
    except ClientError as e:
        raise e

def cognito_admin_update_user_attributes(email, attributes):
    try:
        # Resolve email to username first
        username = resolve_email_to_username(email)
        
        # ADMIN ROLE PROTECTION - Check if trying to change admin role
        if 'custom:role' in attributes:
            # Get current user attributes to check current role
            current_user = cognito_client.admin_get_user(
                UserPoolId=COGNITO_USER_POOL_ID,
                Username=username
            )
            current_attrs = {attr['Name']: attr['Value'] for attr in current_user['UserAttributes']}
            current_role = current_attrs.get('custom:role', '')
            
            # BLOCK any role change for admin accounts
            if current_role == 'admin' and attributes['custom:role'] != 'admin':
                raise Exception(f"Admin role cannot be changed from 'admin' to '{attributes['custom:role']}' for security reasons")
            
            # BLOCK any role change for owner accounts
            if current_role == 'owner' and attributes['custom:role'] != 'owner':
                raise Exception(f"Owner role cannot be changed from 'owner' to '{attributes['custom:role']}' for security reasons")
        
        user_attrs = [{'Name': k, 'Value': v} for k, v in attributes.items()]
        response = cognito_client.admin_update_user_attributes(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=username,
            UserAttributes=user_attrs
        )
        return response
    except ClientError as e:
        raise e

def resolve_email_to_username(email):
    """
    Resolve email address to Cognito username (UUID) with proper pagination support
    """
    try:
        logger.info(f"[RESOLVE] Resolving email to username: {email}")
        
        # Use paginator to handle large user pools efficiently
        paginator = cognito_client.get_paginator('list_users')
        page_iterator = paginator.paginate(
            UserPoolId=COGNITO_USER_POOL_ID,
            PaginationConfig={
                'MaxItems': 10000,  # Handle large user pools
                'PageSize': 60      # Process in chunks
            }
        )
        
        total_users_checked = 0
        for page in page_iterator:
            users = page.get('Users', [])
            total_users_checked += len(users)
            logger.info(f"[RESOLVE] Checking page with {len(users)} users (total checked: {total_users_checked})")
            
            # Find user with matching email (case-insensitive)
            for user in users:
                user_attrs = {attr['Name']: attr['Value'] for attr in user.get('Attributes', [])}
                user_email = user_attrs.get('email', '')
                
                if user_email.lower() == email.lower():
                    logger.info(f"[RESOLVE] Found matching user: {user_email} -> {user['Username']}")
                    return user['Username']
        
        # If no user found with matching email, raise an error
        logger.error(f"[RESOLVE] No user found with email: {email} (checked {total_users_checked} users)")
        raise Exception(f"User with email {email} not found")
            
    except ClientError as e:
        logger.error(f"[RESOLVE] Cognito error resolving email {email}: {e}")
        raise e

def get_user_by_email(email):
    """
    Get user attributes by email address
    """
    try:
        username = resolve_email_to_username(email)
        response = cognito_client.admin_get_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=username
        )
        return response
    except ClientError as e:
        raise e 