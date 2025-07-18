import logging
from flask import Blueprint, request, jsonify
from .cognito import cognito_signup, cognito_confirm_signup, cognito_login
import os
import boto3

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

COGNITO_REGION = os.getenv('COGNITO_REGION')
COGNITO_CLIENT_ID = os.getenv('COGNITO_CLIENT_ID')
COGNITO_CLIENT_SECRET = os.getenv('CLIENT_SECRET')

def get_secret_hash(username):
    import hmac, hashlib, base64
    message = username + COGNITO_CLIENT_ID
    dig = hmac.new(
        str(COGNITO_CLIENT_SECRET).encode('utf-8'),
        msg=message.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(dig).decode()

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    if not email or not password or not first_name or not last_name:
        return jsonify({'error': 'Email, password, first name, and last name required'}), 400
    full_name = f"{first_name} {last_name}"
    try:
        cognito_signup(email, password, role='owner', full_name=full_name, first_name=first_name, last_name=last_name)
        return jsonify({'message': 'Signup successful. Please check your email for confirmation code.'}), 201
    except Exception as e:
        logger.error(f"Error in /auth/signup: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@auth_bp.route('/confirm', methods=['POST'])
def confirm():
    data = request.get_json()
    email = data.get('email')
    code = data.get('code')
    if not email or not code:
        return jsonify({'error': 'Email and code required'}), 400
    try:
        cognito_confirm_signup(email, code)
        # --- Allocate Starter Plan if user has no tenant ---
        from app.models import db, Tenant, User, Plan
        from sqlalchemy.exc import SQLAlchemyError
        from app.auth.cognito import cognito_admin_update_user_attributes
        # Check if user already exists in DB
        user = User.query.filter_by(email=email).first()
        tenant_id = None
        if not user:
            # Find Starter plan
            starter_plan = Plan.query.filter_by(name="Starter").first()
            if not starter_plan:
                return jsonify({'error': 'Starter plan not found. Please contact support.'}), 500
            # Create tenant
            tenant = Tenant(
                plan_id=starter_plan.id,
                stripe_customer_id="",
                stripe_subscription_id="",
                status="active"
            )
            db.session.add(tenant)
            db.session.commit()
            # Create user as owner
            user = User(tenant_id=tenant.id, email=email, role="owner")
            db.session.add(user)
            db.session.commit()
            tenant_id = tenant.id
        else:
            tenant_id = user.tenant_id
        # Always update Cognito user with tenant_id
        try:
            cognito_admin_update_user_attributes(email, {"custom:tenant_id": str(tenant_id)})
        except Exception as e:
            logger.error(f"Failed to update Cognito tenant_id for {email}: {e}")
        # --- End allocation ---
        return jsonify({'message': 'Confirmation successful.'}), 200
    except Exception as e:
        logger.error(f"Error in /auth/confirm: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@auth_bp.route('/confirm-temp-password', methods=['POST'])
def confirm_temp_password():
    data = request.get_json()
    email = data.get('email')
    temp_password = data.get('temp_password')
    new_password = data.get('new_password')
    
    if not email or not temp_password or not new_password:
        return jsonify({'error': 'Email, temporary password, and new password required'}), 400
    
    try:
        # First, authenticate with temporary password
        cognito_client = boto3.client('cognito-idp', region_name=COGNITO_REGION)
        
        # Initiate auth with temporary password
        auth_response = cognito_client.initiate_auth(
            ClientId=COGNITO_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': email,
                'PASSWORD': temp_password,
                'SECRET_HASH': get_secret_hash(email)
            }
        )
        
        # If we get here, the temp password is correct
        # Now change the password
        cognito_client.respond_to_auth_challenge(
            ClientId=COGNITO_CLIENT_ID,
            ChallengeName='NEW_PASSWORD_REQUIRED',
            Session=auth_response['Session'],
            ChallengeResponses={
                'USERNAME': email,
                'NEW_PASSWORD': new_password,
                'SECRET_HASH': get_secret_hash(email)
            }
        )
        
        return jsonify({'message': 'Password changed successfully. You can now log in.'}), 200
        
    except Exception as e:
        logger.error(f"Error in /auth/confirm-temp-password: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    try:
        tokens = cognito_login(email, password)
        # Fetch user attributes from Cognito
        from .cognito import cognito_client, COGNITO_USER_POOL_ID, cognito_admin_update_user_attributes
        user_info = cognito_client.admin_get_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email
        )
        attrs = {attr['Name']: attr['Value'] for attr in user_info['UserAttributes']}
        user = {
            "id": attrs.get("sub"),
            "email": attrs.get("email"),
            "firstName": attrs.get("given_name", ""),
            "lastName": attrs.get("family_name", ""),
            "role": attrs.get("custom:role", "")
        }
        # --- Ensure Starter Plan and Tenant after login ---
        from app.models import db, Tenant, User as DBUser, Plan
        db_user = DBUser.query.filter_by(email=email).first()
        tenant_id = None
        if not db_user:
            # Find Starter plan
            starter_plan = Plan.query.filter_by(name="Starter").first()
            if not starter_plan:
                return jsonify({'error': 'Starter plan not found. Please contact support.'}), 500
            # Create tenant
            tenant = Tenant(
                plan_id=starter_plan.id,
                stripe_customer_id="",
                stripe_subscription_id="",
                status="active"
            )
            db.session.add(tenant)
            db.session.commit()
            # Create user as owner
            db_user = DBUser(tenant_id=tenant.id, email=email, role="owner")
            db.session.add(db_user)
            db.session.commit()
            tenant_id = tenant.id
        else:
            tenant_id = db_user.tenant_id
        # Always update Cognito user with tenant_id
        try:
            cognito_admin_update_user_attributes(email, {"custom:tenant_id": str(tenant_id)})
        except Exception as e:
            logger.error(f"Failed to update Cognito tenant_id for {email}: {e}")
        # --- End ensure Starter plan ---
        return jsonify({
            "access_token": tokens.get("AccessToken"),
            "id_token": tokens.get("IdToken"),
            "refresh_token": tokens.get("RefreshToken"),
            "user": user
        }), 200
    except Exception as e:
        logger.error(f"Error in /auth/login: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 401

@auth_bp.route('/resend-confirmation', methods=['POST'])
def resend_confirmation():
    data = request.get_json()
    email = data.get('email')
    if not email:
        return jsonify({'error': 'Email required'}), 400
    try:
        cognito_client = boto3.client('cognito-idp', region_name=COGNITO_REGION)
        resp = cognito_client.resend_confirmation_code(
            ClientId=COGNITO_CLIENT_ID,
            Username=email,
            SecretHash=get_secret_hash(email)
        )
        return jsonify({'message': 'Confirmation code resent.'}), 200
    except Exception as e:
        logger.error(f"Error in /auth/resend-confirmation: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400 