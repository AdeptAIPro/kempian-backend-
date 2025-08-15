import os
import boto3
from botocore.exceptions import ClientError

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
        response = cognito_client.initiate_auth(
            ClientId=COGNITO_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': email,
                'PASSWORD': password
            }
        )
        return response['AuthenticationResult']
    except ClientError as e:
        raise e

def cognito_admin_create_user(email, tenant_id, role='subuser', temp_password=None):
    import random, string
    if not temp_password:
        temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12)) + '1!A'
    try:
        response = cognito_client.admin_create_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email,
            UserAttributes=[
                {'Name': 'email', 'Value': email},
                {'Name': 'custom:tenant_id', 'Value': str(tenant_id)},
                {'Name': 'custom:role', 'Value': role}
            ],
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
    Resolve email address to Cognito username (UUID)
    """
    try:
        # List users and find the one with matching email (case-insensitive)
        response = cognito_client.list_users(
            UserPoolId=COGNITO_USER_POOL_ID,
            Limit=60
        )
        
        # Find user with matching email (case-insensitive)
        for user in response['Users']:
            user_attrs = {attr['Name']: attr['Value'] for attr in user.get('Attributes', [])}
            user_email = user_attrs.get('email', '')
            if user_email.lower() == email.lower():
                return user['Username']
        
        if response['Users']:
            return response['Users'][0]['Username']
        else:
            raise Exception(f"User with email {email} not found")
            
    except ClientError as e:
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