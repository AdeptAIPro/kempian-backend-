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
            SecretHash=get_secret_hash(email),
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
            SecretHash=get_secret_hash(email),
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
                'PASSWORD': password,
                'SECRET_HASH': get_secret_hash(email)
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
        user_attrs = [{'Name': k, 'Value': v} for k, v in attributes.items()]
        response = cognito_client.admin_update_user_attributes(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email,
            UserAttributes=user_attrs
        )
        return response
    except ClientError as e:
        raise e 