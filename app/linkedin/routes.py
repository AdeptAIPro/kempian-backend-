from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.linkedin.controller import connect_linkedin, get_linkedin_jobs, get_job_applicants, import_candidates_to_system, disconnect_linkedin
from app.simple_logger import get_logger

logger = get_logger(__name__)

# Create LinkedIn blueprint
linkedin_bp = Blueprint('linkedin', __name__)

@linkedin_bp.route('/connect', methods=['POST'])
# Temporarily removed @jwt_required() to allow Cognito tokens
def connect_linkedin_endpoint():
    """Exchange OAuth code for LinkedIn access token"""
    try:
        # Get user ID from token manually (for Cognito compatibility)
        from app.utils import get_current_user
        user = get_current_user()
        user_id = user.get('sub') or user.get('id') if user else None
        
        request_data = request.get_json()
        
        logger.info(f"LinkedIn connect request - user_id: {user_id}")
        logger.info(f"LinkedIn connect request data: {request_data}")
        
        if not request_data:
            logger.error("No request data received")
            return jsonify({'error': 'Request data required'}), 400
        
        # Check if we have the required fields
        if not request_data.get('code'):
            logger.error("No code in request data")
            return jsonify({'error': 'Authorization code required'}), 400
        
        # Check for tenantId or organizationId
        tenant_id = request_data.get('tenantId')
        org_id = request_data.get('organizationId')
        if not tenant_id and not org_id:
            logger.error(f"Neither tenantId nor organizationId provided. Request data: {request_data}")
            return jsonify({'error': 'Tenant ID or Organization ID required'}), 400
        
        # Add user info to request data
        if user_id:
            request_data['user_id'] = user_id
        request_data['user'] = user
        
        result, status_code = connect_linkedin(request_data, {'id': user_id} if user_id else {})
        return jsonify(result), status_code
        
    except Exception as error:
        logger.error(f"Error in connect_linkedin_endpoint: {error}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@linkedin_bp.route('/jobs', methods=['GET'])
# Temporarily removed @jwt_required() to allow Cognito tokens
def get_linkedin_jobs_endpoint():
    """Fetch user's LinkedIn job postings"""
    try:
        # Get user ID from token manually (for Cognito compatibility)
        from app.utils import get_current_user
        user = get_current_user()
        user_id = user.get('sub') or user.get('id') if user else None
        
        # Accept both organizationId and tenantId (they are the same)
        organization_id = request.args.get('organizationId') or request.args.get('tenantId')
        
        if not organization_id:
            return jsonify({'error': 'Organization ID or Tenant ID required'}), 400
        
        result, status_code = get_linkedin_jobs(organization_id, {'id': user_id} if user_id else {}, tenant_id=request.args.get('tenantId'))
        return jsonify(result), status_code
        
    except Exception as error:
        logger.error(f"Error in get_linkedin_jobs_endpoint: {error}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@linkedin_bp.route('/jobs/<job_id>/applicants', methods=['GET'])
# Temporarily removed @jwt_required() to allow Cognito tokens
def get_job_applicants_endpoint(job_id):
    """Fetch applicants for a specific LinkedIn job"""
    try:
        # Get user ID from token manually (for Cognito compatibility)
        from app.utils import get_current_user
        user = get_current_user()
        user_id = user.get('sub') or user.get('id') if user else None
        
        # Accept both organizationId and tenantId (they are the same)
        organization_id = request.args.get('organizationId') or request.args.get('tenantId')
        
        if not organization_id:
            return jsonify({'error': 'Organization ID or Tenant ID required'}), 400
        
        result, status_code = get_job_applicants(job_id, organization_id, {'id': user_id} if user_id else {}, tenant_id=request.args.get('tenantId'))
        return jsonify(result), status_code
        
    except Exception as error:
        logger.error(f"Error in get_job_applicants_endpoint: {error}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@linkedin_bp.route('/import-candidates', methods=['POST'])
# Temporarily removed @jwt_required() to allow Cognito tokens
def import_candidates_endpoint():
    """Import LinkedIn candidates to the system"""
    try:
        # Get user ID from token manually (for Cognito compatibility)
        from app.utils import get_current_user
        user = get_current_user()
        user_id = user.get('sub') or user.get('id') if user else None
        
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({'error': 'Request data required'}), 400
        
        candidates = request_data.get('candidates', [])
        # Accept both organizationId and tenantId (they are the same)
        organization_id = request_data.get('organizationId') or request_data.get('tenantId')
        
        if not organization_id:
            return jsonify({'error': 'Organization ID or Tenant ID required'}), 400
        
        if not candidates:
            return jsonify({'error': 'Candidates data required'}), 400
        
        tenant_id = request_data.get('tenantId')
        result, status_code = import_candidates_to_system(candidates, organization_id, {'id': user_id} if user_id else {}, tenant_id=tenant_id)
        return jsonify(result), status_code
        
    except Exception as error:
        logger.error(f"Error in import_candidates_endpoint: {error}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@linkedin_bp.route('/status', methods=['GET'])
def get_linkedin_status():
    """Check if LinkedIn is connected for an organization"""
    try:
        # Get tenantId from request parameter or tenant context
        from flask import g
        tenant_id = request.args.get('tenantId')
        
        # Debug logging
        logger.info(f"LinkedIn status - tenantId from request: {tenant_id}")
        logger.info(f"LinkedIn status - tenant_id from context: {getattr(g, 'tenant_id', None)}")
        
        # If no tenantId provided, try to get from tenant context
        if not tenant_id:
            context_tenant_id = getattr(g, 'tenant_id', None)
            tenant_id = str(context_tenant_id) if context_tenant_id else None
            logger.info(f"LinkedIn status - using context tenant_id: {tenant_id}")
        
        if not tenant_id:
            logger.error("LinkedIn status - No tenantId found")
            return jsonify({'error': 'Tenant ID required'}), 400
        
        # Check if LinkedIn is connected for this tenant
        from app import db
        from sqlalchemy import text
        
        # Query the linkedin_integrations table
        with db.engine.connect() as conn:
            result = conn.execute(
                text("SELECT id, organization_id, created_at FROM linkedin_integrations WHERE organization_id = :tenant_id"),
                {"tenant_id": tenant_id}
            )
            integration = result.fetchone()
        
        if integration:
            logger.info(f"LinkedIn is connected for tenant_id: {tenant_id}")
            return jsonify({
                'connected': True,
                'message': 'LinkedIn is connected',
                'tenant_id': tenant_id
            })
        else:
            logger.info(f"LinkedIn is NOT connected for tenant_id: {tenant_id}")
            return jsonify({
                'connected': False,
                'message': 'LinkedIn is not connected',
                'tenant_id': tenant_id
            })
            
    except Exception as error:
        logger.error(f"Error in get_linkedin_status: {error}")
        return jsonify({'error': 'Internal server error'}), 500

@linkedin_bp.route('/disconnect', methods=['POST'])
# Temporarily removed @jwt_required() to allow Cognito tokens
def disconnect_linkedin_endpoint():
    """Disconnect LinkedIn integration"""
    try:
        # Get user ID from token manually (for Cognito compatibility)
        from app.utils import get_current_user
        user = get_current_user()
        user_id = user.get('sub') or user.get('id') if user else None
        
        request_data = request.get_json() or {}
        
        # Accept both organizationId and tenantId (they are the same)
        # Check both request body and query parameters
        tenant_id = request_data.get('tenantId') or request_data.get('organizationId') or request.args.get('tenantId') or request.args.get('organizationId')
        
        if not tenant_id:
            logger.error("Neither tenantId nor organizationId provided in disconnect request")
            return jsonify({'error': 'Tenant ID or Organization ID required'}), 400
        
        logger.info(f"LinkedIn disconnect request - tenant_id: {tenant_id}")
        
        result, status_code = disconnect_linkedin(None, {'id': user_id} if user_id else {}, tenant_id=tenant_id)
        return jsonify(result), status_code
        
    except Exception as error:
        logger.error(f"Error in disconnect_linkedin_endpoint: {error}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500
