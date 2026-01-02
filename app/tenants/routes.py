import logging
from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import Tenant, User, Plan, db
from app.auth.cognito import cognito_admin_create_user
from app.emails.ses import send_invite_email
from flask import current_app
import jwt
import os
from urllib.parse import quote

logger = get_logger("tenants")

tenants_bp = Blueprint('tenants', __name__)

# Helper: decode JWT and check role/tenant

def get_jwt_payload():
    auth = request.headers.get('Authorization', None)
    if not auth:
        return None
    token = auth.split(' ')[1]
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload
    except Exception as e:
        logger.error(f"Error decoding JWT: {str(e)}", exc_info=True)
        return None

@tenants_bp.route('/<int:tenant_id>/users', methods=['GET'])
def get_tenant_users(tenant_id):
    try:
        payload = get_jwt_payload()
        if not payload or int(payload.get('custom:tenant_id', 0)) != tenant_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        users = User.query.filter_by(tenant_id=tenant_id).all()
        users_data = []
        for user in users:
            users_data.append({
                'id': user.id,
                'email': user.email,
                'role': user.role,
                'user_type': user.user_type or user.role,  # Include user_type to identify employees
                'created_at': user.created_at.isoformat() if user.created_at and hasattr(user.created_at, 'isoformat') and callable(getattr(user.created_at, 'isoformat', None)) else None
            })
        
        return jsonify({'users': users_data}), 200
    except Exception as e:
        logger.error(f"Error in /tenants/<tenant_id>/users: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@tenants_bp.route('/<int:tenant_id>/plan', methods=['GET'])
def get_tenant_plan(tenant_id):
    try:
        payload = get_jwt_payload()
        if not payload or int(payload.get('custom:tenant_id', 0)) != tenant_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        tenant = Tenant.query.get(tenant_id)
        if not tenant:
            return jsonify({'error': 'Tenant not found'}), 404
        
        plan = Plan.query.get(tenant.plan_id)
        if not plan:
            return jsonify({'error': 'Plan not found'}), 404
        
        return jsonify({
            'plan': {
                'id': plan.id,
                'name': plan.name,
                'max_subaccounts': plan.max_subaccounts,
                'jd_quota_per_month': plan.jd_quota_per_month
            }
        }), 200
    except Exception as e:
        logger.error(f"Error in /tenants/<tenant_id>/plan: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@tenants_bp.route('/<int:tenant_id>/subusers', methods=['POST'])
def invite_subuser(tenant_id):
    try:
        payload = get_jwt_payload()
        user_role = payload.get('custom:role') if payload else None
        user_tenant_id = int(payload.get('custom:tenant_id', 0)) if payload else 0
        
        # Allow both 'owner' and 'admin' roles to invite subusers
        if not payload or user_role not in ['owner', 'admin'] or user_tenant_id != tenant_id:
            logger.warning(f"Unauthorized invite attempt - role: {user_role}, tenant_id: {user_tenant_id}, requested_tenant_id: {tenant_id}")
            return jsonify({'error': 'Unauthorized'}), 403
        data = request.get_json()
        sub_email = data.get('email')
        if not sub_email:
            return jsonify({'error': 'Email required'}), 400
        # Count subusers
        subuser_count = User.query.filter_by(tenant_id=tenant_id, role='subuser').count()
        tenant = Tenant.query.get(tenant_id)
        if not tenant:
            return jsonify({'error': 'Tenant not found'}), 404
        plan = Plan.query.get(tenant.plan_id)
        if not plan:
            return jsonify({'error': 'Plan not found'}), 404
        
        # Prevent team member addition for Free Trial plan
        if plan.name == "Free Trial" or plan.max_subaccounts == 0:
            return jsonify({'error': 'Team members are not available for Free Trial plan. Please upgrade to add team members.'}), 403
        
        if subuser_count >= plan.max_subaccounts:
            return jsonify({'error': 'Max sub-accounts reached.'}), 400
        
        # Check if user already exists in database
        existing_user = User.query.filter_by(email=sub_email).first()
        if existing_user:
            if existing_user.tenant_id != tenant_id:
                return jsonify({'error': 'User already exists with a different tenant'}), 400
            if existing_user.role != 'subuser':
                return jsonify({'error': 'User already exists with a different role'}), 400
            logger.info(f"User {sub_email} already exists in database, skipping creation")
        else:
            # Create User in database first (before Cognito to ensure consistency)
            try:
                db_user = User(
                    tenant_id=tenant_id,
                    email=sub_email,
                    role='subuser',
                    user_type='subuser'
                )
                db.session.add(db_user)
                db.session.commit()
                logger.info(f"Created User in database: {sub_email} with tenant_id {tenant_id}")
            except Exception as db_error:
                db.session.rollback()
                logger.error(f"Error creating user in database: {str(db_error)}")
                # Check if user was created by another process
                existing_user = User.query.filter_by(email=sub_email).first()
                if not existing_user:
                    return jsonify({'error': f'Failed to create user in database: {str(db_error)}'}), 500
        
        # Create Cognito subuser
        try:
            temp_password, _ = cognito_admin_create_user(sub_email, tenant_id, role='subuser')
        except Exception as cognito_error:
            # If Cognito creation fails but DB user exists, we should handle it
            error_msg = str(cognito_error)
            if 'UsernameExistsException' in error_msg or 'AliasExistsException' in error_msg:
                logger.warning(f"User {sub_email} already exists in Cognito, continuing with invite link generation")
                # Generate a temporary password for existing user
                import random, string
                temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12)) + '1!A'
            else:
                logger.error(f"Error creating user in Cognito: {error_msg}")
                return jsonify({'error': f'Failed to create user in Cognito: {error_msg}'}), 500
        
        # Generate invite link (for example)
        invite_code = temp_password  # In real use, generate a secure code
        
        # Get frontend URL (local for development, production for production)
        def get_frontend_url():
            frontend_url = os.getenv('FRONTEND_URL')
            if frontend_url:
                return frontend_url
            flask_env = os.getenv('FLASK_ENV', '').lower()
            flask_debug = os.getenv('FLASK_DEBUG', '').lower()
            is_development = (
                flask_env == 'development' or 
                flask_debug == 'true' or 
                flask_debug == '1' or
                os.getenv('ENVIRONMENT', '').lower() == 'development' or
                os.getenv('ENV', '').lower() == 'development'
            )
            if is_development:
                local_port = os.getenv('FRONTEND_PORT', '5173')
                return f'http://localhost:{local_port}'
            else:
                return 'https://kempian.ai'
        
        frontend_url = get_frontend_url()
        encoded_email = quote(sub_email, safe='')
        encoded_code = quote(invite_code, safe='')
        invite_link = (
            f"{frontend_url}/invite?"
            f"email={encoded_email}&username={encoded_email}&code={encoded_code}"
        )
        
        # Send SES invite - handle errors gracefully
        email_sent = False
        email_error = None
        try:
            email_sent = send_invite_email(sub_email, invite_link)
            if not email_sent:
                logger.warning(f"Failed to send invite email to {sub_email}, but invite link generated")
        except Exception as email_ex:
            logger.error(f"Exception while sending invite email to {sub_email}: {str(email_ex)}")
            email_error = str(email_ex)
        
        # Always return the invite link even if email fails
        # The user can share the link manually
        response_data = {
            'invite_link': invite_link,
            'email_sent': email_sent
        }
        
        if email_error:
            response_data['email_warning'] = f"Email could not be sent: {email_error}. Please share the invite link manually."
        
        return jsonify(response_data), 201
    except Exception as e:
        logger.error(f"Error in /tenants/<tenant_id>/subusers: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 