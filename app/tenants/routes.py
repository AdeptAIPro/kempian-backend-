import logging
from flask import Blueprint, request, jsonify
from app.models import Tenant, User, Plan, db
from app.auth.cognito import cognito_admin_create_user
from app.emails.ses import send_invite_email
from flask import current_app
import jwt
import os

logger = logging.getLogger(__name__)

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
                'created_at': user.created_at.isoformat() if user.created_at else None
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
        if not payload or payload.get('custom:role') != 'owner' or int(payload.get('custom:tenant_id', 0)) != tenant_id:
            return jsonify({'error': 'Unauthorized'}), 403
        data = request.get_json()
        sub_email = data.get('email')
        if not sub_email:
            return jsonify({'error': 'Email required'}), 400
        # Count subusers
        subuser_count = User.query.filter_by(tenant_id=tenant_id, role='subuser').count()
        tenant = Tenant.query.get(tenant_id)
        plan = Plan.query.get(tenant.plan_id)
        if subuser_count >= plan.max_subaccounts:
            return jsonify({'error': 'Max sub-accounts reached.'}), 400
        # Create Cognito subuser
        temp_password, _ = cognito_admin_create_user(sub_email, tenant_id, role='subuser')
        # Generate invite link (for example)
        invite_code = temp_password  # In real use, generate a secure code
        invite_link = f"{os.getenv('FRONTEND_URL')}/invite?username={sub_email}&code={invite_code}"
        # Send SES invite
        send_invite_email(sub_email, invite_link)
        return jsonify({'invite_link': invite_link}), 201
    except Exception as e:
        logger.error(f"Error in /tenants/<tenant_id>/subusers: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 