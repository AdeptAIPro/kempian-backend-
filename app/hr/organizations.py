from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import Tenant, User, Plan, OrganizationMetadata, db
from sqlalchemy.orm import joinedload
from datetime import datetime


hr_organizations_bp = Blueprint('hr_organizations', __name__)


@hr_organizations_bp.route('/', methods=['GET'], strict_slashes=False)
def list_organizations():
    user = get_current_user_flexible() or get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Get user's current organization
    current_tenant_id = get_current_tenant_id() or db_user.tenant_id
    
    # For owners/admins: Show their current organization + organizations they created
    # For others: Show only their current organization
    if db_user.role in ['owner', 'admin']:
        # Get organizations created by this user
        user_created_orgs = OrganizationMetadata.query.filter_by(created_by=db_user.id).all()
        user_created_tenant_ids = [org.tenant_id for org in user_created_orgs]
        
        # Show ONLY organizations created by this user (not their current org if they didn't create it)
        if user_created_tenant_ids:
            tenants = Tenant.query.options(joinedload(Tenant.organization_metadata)).filter(
                Tenant.id.in_(user_created_tenant_ids)
            ).order_by(Tenant.id.desc()).all()
        else:
            # If no organizations created yet, show empty list
            tenants = []
    else:
        # Regular users see only their own organization
        if current_tenant_id:
            tenants = Tenant.query.options(joinedload(Tenant.organization_metadata)).filter_by(id=current_tenant_id).all()
        else:
            tenants = []
    
    results = []
    for t in tenants:
        org_data = {
            'id': t.id,
            'status': t.status,
            'plan': t.plan.name if getattr(t, 'plan', None) else None,
            'stripe_customer_id': t.stripe_customer_id,
            'stripe_subscription_id': t.stripe_subscription_id
        }
        # Get organization name from metadata if exists
        if t.organization_metadata:
            org_data['name'] = t.organization_metadata.name
        else:
            org_data['name'] = f'Organization {t.id}'
        results.append(org_data)
    
    return jsonify({'organizations': results}), 200


@hr_organizations_bp.route('/', methods=['POST'], strict_slashes=False)
def create_organization():
    """Create a new organization (tenant) - only for owners"""
    user = get_current_user_flexible() or get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only owners can create organizations
    if db_user.role not in ['owner', 'admin']:
        return jsonify({'error': 'Only owners and admins can create organizations'}), 403
    
    data = request.get_json()
    organization_name = data.get('name')
    if not organization_name:
        return jsonify({'error': 'Organization name is required'}), 400
    
    try:
        # Find Free Trial plan
        trial_plan = Plan.query.filter_by(name="Free Trial").first()
        if not trial_plan:
            return jsonify({'error': 'Free Trial plan not found. Please contact support.'}), 500
        
        # Create new tenant (organization)
        new_tenant = Tenant(
            plan_id=trial_plan.id,
            stripe_customer_id="",
            stripe_subscription_id="",
            status="active"
        )
        db.session.add(new_tenant)
        db.session.flush()  # Flush to get the tenant ID
        
        # Create organization metadata to store the name
        org_metadata = OrganizationMetadata(
            tenant_id=new_tenant.id,
            name=organization_name,
            created_by=db_user.id  # Track who created this organization
        )
        db.session.add(org_metadata)
        db.session.commit()
        
        return jsonify({
            'organization': {
                'id': new_tenant.id,
                'name': organization_name,
                'status': new_tenant.status,
                'plan': trial_plan.name,
                'created_at': new_tenant.created_at.isoformat() if new_tenant.created_at else None
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to create organization: {str(e)}'}), 500


