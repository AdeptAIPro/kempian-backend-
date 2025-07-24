import logging
from flask import Blueprint, request, jsonify
from werkzeug.exceptions import HTTPException
import logging
from .cognito import cognito_signup, cognito_confirm_signup, cognito_login, cognito_admin_update_user_attributes
from .cognito import cognito_client, COGNITO_USER_POOL_ID, COGNITO_REGION
import os
import boto3
from datetime import datetime
from app.utils import get_current_user
from app.models import db, Tenant, User, Plan, UserSocialLinks
from sqlalchemy.exc import SQLAlchemyError

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
    role = data.get('role', 'job_seeker')  # default to job_seeker
    print(f"[DEBUG] /signup received role: {role} for email: {email}")
    if not email or not password or not first_name or not last_name:
        return jsonify({'error': 'Email, password, first name, and last name required'}), 400
    full_name = f"{first_name} {last_name}"
    try:
        # For employers/recruiters/admin, they get 'owner' or 'admin' role in tenant system
        if role == 'admin':
            tenant_role = 'admin'
        elif role in ['employer', 'recruiter']:
            tenant_role = 'owner'
        else:
            tenant_role = role
        cognito_signup(email, password, role=tenant_role, user_type=role, full_name=full_name, first_name=first_name, last_name=last_name)
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
            # Get the original user type from Cognito
            cognito_client = boto3.client('cognito-idp', region_name=COGNITO_REGION)
            user_info = cognito_client.admin_get_user(
                UserPoolId=COGNITO_USER_POOL_ID,
                Username=email
            )
            attrs = {attr['Name']: attr['Value'] for attr in user_info['UserAttributes']}
            original_user_type = attrs.get("custom:user_type", attrs.get("custom:role", "owner"))
            print(f"[DEBUG] /confirm Cognito custom:user_type for {email} is {original_user_type}")
            # Always store the original user type for display, even if system role is owner/admin
            db_user = User(tenant_id=tenant.id, email=email, role=tenant_role, user_type=original_user_type)
            print(f"[DEBUG] /confirm storing user_type={original_user_type} for {email} in DB")
            db.session.add(db_user)
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
        
        # --- Ensure Starter Plan and Tenant after login ---
        db_user = User.query.filter_by(email=email).first()
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
            # Get the original user type from Cognito
            user_info = cognito_client.admin_get_user(
                UserPoolId=COGNITO_USER_POOL_ID,
                Username=email
            )
            attrs = {attr['Name']: attr['Value'] for attr in user_info['UserAttributes']}
            original_role = attrs.get("custom:role", "owner")
            
            # Store the original user type for display purposes
            user_type = original_role if original_role in ['job_seeker', 'employee', 'recruiter', 'employer'] else None
            
            db_user = User(tenant_id=tenant.id, email=email, role="owner", user_type=user_type)
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
        
        user = {
            "id": attrs.get("sub"),
            "email": attrs.get("email"),
            "firstName": attrs.get("given_name", ""),
            "lastName": attrs.get("family_name", ""),
            "role": attrs.get("custom:role", ""),
            "userType": attrs.get("custom:user_type", attrs.get("custom:role", ""))
        }
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

@auth_bp.route('/user/social-links', methods=['POST'])
def save_social_links():
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    data = request.get_json()
    linkedin = data.get('linkedin')
    facebook = data.get('facebook')
    x = data.get('x')
    github = data.get('github')
    social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
    if social_links:
        social_links.linkedin = linkedin
        social_links.facebook = facebook
        social_links.x = x
        social_links.github = github
        social_links.updated_at = datetime.utcnow()
    else:
        social_links = UserSocialLinks(
            user_id=user.id,
            linkedin=linkedin,
            facebook=facebook,
            x=x,
            github=github
        )
        db.session.add(social_links)
    db.session.commit()
    return jsonify({'message': 'Social links saved successfully.'}), 200

@auth_bp.route('/user/social-links', methods=['GET'])
def get_social_links():
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
    if not social_links:
        return jsonify({'linkedin': '', 'facebook': '', 'x': '', 'github': ''}), 200
    return jsonify({
        'linkedin': social_links.linkedin or '',
        'facebook': social_links.facebook or '',
        'x': social_links.x or '',
        'github': social_links.github or ''
    }), 200 

@auth_bp.route('/admin/social-links', methods=['GET'])
def admin_get_all_social_links():
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email') or user_jwt.get('custom:role') != 'admin':
        return jsonify({'error': 'Forbidden'}), 403
    all_links = []
    for social in UserSocialLinks.query.all():
        user = User.query.get(social.user_id)
        all_links.append({
            'user_id': social.user_id,
            'email': user.email if user else '',
            'linkedin': social.linkedin or '',
            'facebook': social.facebook or '',
            'x': social.x or '',
            'github': social.github or ''
        })
    return jsonify({'social_links': all_links}), 200 