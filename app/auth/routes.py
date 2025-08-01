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
from jose import jwt
import requests
import json
from app.utils.trial_manager import create_user_trial

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

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

# Helper to get Cognito public keys (cache for production)
COGNITO_KEYS_URL = f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}/.well-known/jwks.json'
_cognito_jwk_cache = None

def get_cognito_jwk():
    global _cognito_jwk_cache
    if _cognito_jwk_cache is None:
        resp = requests.get(COGNITO_KEYS_URL)
        _cognito_jwk_cache = resp.json()
    return _cognito_jwk_cache

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
        # --- Allocate Free Trial Plan if user has no tenant ---
        # Check if user already exists in DB
        user = User.query.filter_by(email=email).first()
        tenant_id = None
        if not user:
            # Find Free Trial plan
            trial_plan = Plan.query.filter_by(name="Free Trial").first()
            if not trial_plan:
                return jsonify({'error': 'Free Trial plan not found. Please contact support.'}), 500
            # Create tenant
            tenant = Tenant(
                plan_id=trial_plan.id,
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
            
            # Determine tenant role based on user type
            if original_user_type == 'admin':
                tenant_role = 'admin'
            elif original_user_type in ['employer', 'recruiter']:
                tenant_role = 'owner'
            else:
                tenant_role = original_user_type
            
            # Always store the original user type for display, even if system role is owner/admin
            db_user = User(tenant_id=tenant.id, email=email, role=tenant_role, user_type=original_user_type)
            print(f"[DEBUG] /confirm storing user_type={original_user_type} for {email} in DB")
            db.session.add(db_user)
            db.session.commit()
            
            # Create trial for new user
            trial = create_user_trial(db_user.id)
            if trial:
                print(f"[DEBUG] /confirm created trial for user {db_user.id}")
            else:
                print(f"[DEBUG] /confirm failed to create trial for user {db_user.id}")
            
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
    requested_role = data.get('role')  # Add role parameter for role change requests
    
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
        
        stored_role = attrs.get("custom:role", "")
        
        # Role validation and change logic
        if requested_role and requested_role != stored_role:
            logger.info(f"Role change requested: {email} from {stored_role} to {requested_role}")
            
            # Define allowed role changes (hierarchical permissions)
            # Note: 'owner' and 'admin' are system roles that can access any user role
            allowed_role_changes = {
                'admin': ['job_seeker', 'employee', 'recruiter', 'employer'],
                'owner': ['job_seeker', 'employee', 'recruiter', 'employer'],  # Add owner role
                'employer': ['job_seeker', 'employee', 'recruiter'],
                'recruiter': ['job_seeker', 'employee'],
                'employee': ['job_seeker'],
                'job_seeker': []  # Job seekers can't change to other roles
            }
            
            # Check if role change is allowed
            role_change_allowed = False
            if stored_role in allowed_role_changes and requested_role in allowed_role_changes[stored_role]:
                role_change_allowed = True
            elif stored_role in ['admin', 'owner'] and requested_role in ['job_seeker', 'employee', 'recruiter', 'employer']:
                # Admin/owner can access any role
                role_change_allowed = True
            
            if role_change_allowed:
                # Update Cognito with new role
                try:
                    cognito_admin_update_user_attributes(email, {
                        "custom:role": requested_role,
                        "custom:user_type": requested_role
                    })
                    stored_role = requested_role
                    logger.info(f"Role change successful: {email} changed to {requested_role}")
                except Exception as e:
                    logger.error(f"Failed to update Cognito role for {email}: {e}")
                    return jsonify({'error': 'Failed to update role. Please try again.'}), 500
            else:
                logger.warning(f"Role change denied: {email} from {stored_role} to {requested_role}")
                return jsonify({'error': f'Role change from {stored_role} to {requested_role} not allowed'}), 403
        
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
            "role": stored_role,
            "userType": attrs.get("custom:user_type", stored_role)
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

@auth_bp.route('/cognito-social-login', methods=['POST'])
def cognito_social_login():
    try:
        data = request.get_json()
        # Handle both frontend format (idToken, accessToken) and backend format (id_token, access_token)
        id_token = data.get('id_token') or data.get('idToken')
        access_token = data.get('access_token') or data.get('accessToken')
        state = data.get('state')  # Get state parameter from frontend
        role_fallback = data.get('role_fallback') or data.get('role')  # Get role fallback from frontend
        
        if not id_token:
            return jsonify({'error': 'ID token is required'}), 400
        if not access_token:
            return jsonify({'error': 'Access token is required'}), 400
            
        logger.info("🔍 Starting Cognito social login verification...")
        
        # Extract role from state if provided
        role_from_state = None
        if state:
            try:
                import base64
                import json
                state_data = json.loads(base64.b64decode(state).decode('utf-8'))
                role_from_state = state_data.get('role')
            except Exception as e:
                logger.warning(f"Could not decode state parameter: {e}")
        
        # 1. JWT verification logic
        # logger.info("🔍 Decoding JWT header to get KID...")
        header = jwt.get_unverified_header(id_token)
        kid = header.get('kid')
        # logger.info(f"🔑 JWT KID: {kid}")
        
        if not kid:
            logger.error("❌ No KID found in JWT header")
            return jsonify({'error': 'Invalid token format'}), 401
        
        # Get JWKS
        # logger.info("🔍 Fetching JWKS from Cognito...")
        jwks = get_cognito_jwk()
        
        # Find the matching key
        key = None
        for jwk in jwks.get('keys', []):
            if jwk['kid'] == kid:
                key = jwk
                break
        
        if not key:
            logger.error(f"❌ No matching key found for KID: {kid}")
            return jsonify({'error': 'No matching key found for token'}), 401
        
        # logger.info("✅ Found matching JWK key")
        
        # Verify the token
        # logger.info("🔍 Verifying JWT token...")
        # logger.info(f"🔑 Backend COGNITO_CLIENT_ID: {COGNITO_CLIENT_ID}")
        # logger.info(f"🔑 JWT token audience (from token): {jwt.get_unverified_claims(id_token).get('aud')}")
        
        claims = jwt.decode(
            id_token,
            key,
            algorithms=['RS256'],
            audience=COGNITO_CLIENT_ID,
            issuer=f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}',
            access_token=access_token  # Add access token for at_hash validation
        )
        
        # logger.info(f"✅ JWT verification successful")
        # logger.info(f"📋 Claims: {claims}")
        
        email = claims.get('email')
        sub = claims.get('sub')
        first_name = claims.get('given_name', '')
        last_name = claims.get('family_name', '')
        
        # Use role from state, then fallback, then default
        role = role_from_state or role_fallback or claims.get('custom:role', 'job_seeker')
        user_type = claims.get('custom:user_type', role)
        

        
        # 2. User creation logic (reuse from /login)
        # logger.info("🔍 Checking if user exists in database...")
        db_user = User.query.filter_by(email=email).first()
        tenant_id = None
        
        if not db_user:
            # logger.info("👤 User not found in database, creating new user...")
            starter_plan = Plan.query.filter_by(name="Starter").first()
            if not starter_plan:
                logger.error("❌ Starter plan not found")
                return jsonify({'error': 'Starter plan not found. Please contact support.'}), 500
            
            # logger.info("🏢 Creating new tenant...")
            tenant = Tenant(
                plan_id=starter_plan.id,
                stripe_customer_id="",
                stripe_subscription_id="",
                status="active"
            )
            db.session.add(tenant)
            db.session.commit()
            # logger.info(f"✅ Tenant created with ID: {tenant.id}")
            
            # logger.info("👤 Creating new user...")
            db_user = User(tenant_id=tenant.id, email=email, role="owner", user_type=user_type)
            db.session.add(db_user)
            db.session.commit()
            tenant_id = tenant.id
            # logger.info(f"✅ User created successfully with ID: {db_user.id}")
        else:
            # logger.info(f"✅ User already exists in database with ID: {db_user.id}")
            tenant_id = db_user.tenant_id
            
        # 3. Update Cognito user attributes with the role if it's different
        if role_from_state and role_from_state != claims.get('custom:role'):
            try:
                # logger.info(f"🔄 Updating Cognito user role from {claims.get('custom:role')} to {role_from_state}")
                cognito_admin_update_user_attributes(email, {
                    "custom:role": role_from_state,
                    "custom:user_type": role_from_state
                })
                # Update the role variables
                role = role_from_state
                user_type = role_from_state
            except Exception as e:
                logger.error(f"Failed to update Cognito role for {email}: {e}")
            
        # 4. Return user info
        user = {
            "id": sub,
            "email": email,
            "firstName": first_name,
            "lastName": last_name,
            "role": role,
            "userType": user_type
        }
        
        # logger.info(f"✅ Returning user data: {user}")
        return jsonify({
            "id_token": id_token,
            "token": id_token,  # Add token field for frontend compatibility
            "user": user
        }), 200
        
    except jwt.JWTError as e:
        logger.error(f"❌ JWT verification error in /auth/cognito-social-login: {str(e)}", exc_info=True)
        return jsonify({'error': f'Invalid token: {str(e)}'}), 401
    except Exception as e:
        logger.error(f"❌ Error in /auth/cognito-social-login: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 401

@auth_bp.route('/social-login', methods=['POST'])
def social_login():
    """Frontend-compatible social login endpoint that redirects to cognito-social-login"""
    return cognito_social_login()

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