from flask import Blueprint, request, jsonify, g
from app.simple_logger import get_logger
import os
import requests
from datetime import datetime, timedelta
from app.models import StafferlinkIntegration, User, db
from app.utils import get_current_user
import base64
import logging

logger = get_logger("stafferlink")

stafferlink_bp = Blueprint('stafferlink', __name__)

class StafferlinkAPI:
    def __init__(self, api_key, agency_id, email):
        self.api_key = api_key
        self.agency_id = agency_id
        self.email = email
        self.base_url = "https://api.stafferlink.com"
        self.headers = {
            "APIKey": api_key,
            "TargetSystem": "FSM",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def validate_credentials(self):
        """Validate credentials by making a test API call"""
        try:
            # Test with a simple endpoint that requires authentication
            response = requests.get(
                f"{self.base_url}/fsm/agency/orders/newandmodified?LastModifiedMinutes=1",
                headers=self.headers,
                timeout=20
            )
            logger.info(f"Stafferlink validation response status: {response.status_code}")
            
            if response.status_code == 200:
                return True, "Credentials validated successfully"
            elif response.status_code == 401:
                return False, "Invalid API credentials"
            elif response.status_code == 403:
                return False, "Access denied - check agency ID"
            else:
                return False, f"API error: {response.status_code}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Stafferlink API request failed: {e}")
            return False, f"Connection failed: {str(e)}"

    def get_orders(self, last_modified_minutes=30):
        """Get orders from Stafferlink API"""
        try:
            params = f"LastModifiedMinutes={last_modified_minutes}&ExcludeContracts=false&ExcludePerDiem=false&ExcludePermPlacement=false"
            response = requests.get(
                f"{self.base_url}/fsm/agency/orders/newandmodified?{params}",
                headers=self.headers,
                timeout=20
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch orders: {response.status_code}")
                return {"error": f"API error: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Stafferlink orders request failed: {e}")
            return {"error": str(e)}

    def get_order_details(self, order_id):
        """Get specific order details"""
        try:
            response = requests.get(
                f"{self.base_url}/fsm/agency/orders/{order_id}",
                headers=self.headers,
                timeout=20
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch order details: {response.status_code}")
                return {"error": f"API error: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Stafferlink order details request failed: {e}")
            return {"error": str(e)}

# POST /integrations/stafferlink/connect
@stafferlink_bp.route('/integrations/stafferlink/connect', methods=['POST'])
def stafferlink_connect():
    logger.info('HIT /integrations/stafferlink/connect')
    user_jwt = get_current_user()
    logger.info(f'user_jwt: {user_jwt}')
    
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get tenant_id from context or user lookup
    tenant_id = getattr(g, 'tenant_id', None)
    if not tenant_id:
        # Fallback: get tenant_id from user lookup
        temp_user = User.query.filter_by(email=user_jwt['email']).first()
        if temp_user:
            tenant_id = temp_user.tenant_id
        else:
            logger.error(f'User not found: {user_jwt.get("email")}')
            return jsonify({'error': 'User not found'}), 404
    
    # Filter user by both email and tenant_id to ensure correct tenant
    user = User.query.filter_by(email=user_jwt['email'], tenant_id=tenant_id).first()
    if not user:
        logger.error(f'User not found for email={user_jwt.get("email")}, tenant_id={tenant_id}')
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    logger.info(f'Received data: {data}')
    
    stafferlink_email = data.get('email')
    stafferlink_api_key = data.get('apiKey')
    stafferlink_agency_id = data.get('agencyId')
    
    if not stafferlink_email or not stafferlink_api_key or not stafferlink_agency_id:
        logger.error('Missing fields in request')
        return jsonify({'error': 'Missing required fields: email, apiKey, agencyId'}), 400
    
    # Validate credentials with Stafferlink API
    api = StafferlinkAPI(stafferlink_api_key, stafferlink_agency_id, stafferlink_email)
    is_valid, message = api.validate_credentials()
    
    if not is_valid:
        logger.error(f'Stafferlink validation failed: {message}')
        return jsonify({'error': f'Invalid credentials: {message}'}), 401
    
    # Save integration
    integration = StafferlinkIntegration.query.filter_by(user_id=user.id).first()
    if integration:
        integration.stafferlink_email = stafferlink_email
        integration.stafferlink_api_key = stafferlink_api_key
        integration.stafferlink_agency_id = stafferlink_agency_id
    else:
        integration = StafferlinkIntegration(
            user_id=user.id,
            stafferlink_email=stafferlink_email,
            stafferlink_api_key=stafferlink_api_key,
            stafferlink_agency_id=stafferlink_agency_id
        )
        db.session.add(integration)
    
    db.session.commit()
    logger.info('Stafferlink integration saved.')
    return jsonify({'message': 'Stafferlink integration saved successfully'}), 200

# POST /integrations/stafferlink/connect/test
@stafferlink_bp.route('/integrations/stafferlink/connect/test', methods=['POST'])
def stafferlink_test_connection():
    logger.info('HIT /integrations/stafferlink/connect/test')
    user_jwt = get_current_user()
    
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    stafferlink_email = data.get('email')
    stafferlink_api_key = data.get('apiKey')
    stafferlink_agency_id = data.get('agencyId')
    
    if not stafferlink_email or not stafferlink_api_key or not stafferlink_agency_id:
        return jsonify({'error': 'Missing required fields: email, apiKey, agencyId'}), 400
    
    # Test credentials
    api = StafferlinkAPI(stafferlink_api_key, stafferlink_agency_id, stafferlink_email)
    is_valid, message = api.validate_credentials()
    
    if is_valid:
        return jsonify({'ok': True, 'message': message}), 200
    else:
        return jsonify({'ok': False, 'error': message}), 400

# GET /integrations/stafferlink/status
@stafferlink_bp.route('/integrations/stafferlink/status', methods=['GET'])
def stafferlink_status():
    logger.info('HIT /integrations/stafferlink/status')
    user_jwt = get_current_user()
    
    if not user_jwt or not user_jwt.get('email'):
        logger.error('No user_jwt or email')
        return jsonify({'connected': False}), 200
    
    email = user_jwt.get('email')
    
    # Get tenant_id from context or user lookup
    tenant_id = getattr(g, 'tenant_id', None)
    if not tenant_id:
        # Fallback: get tenant_id from user lookup
        temp_user = User.query.filter_by(email=email).first()
        if temp_user:
            tenant_id = temp_user.tenant_id
            logger.info(f'Got tenant_id from user lookup: {tenant_id} for email: {email}')
        else:
            logger.error(f'User not found: {email}')
            return jsonify({'connected': False}), 200
    else:
        logger.info(f'Got tenant_id from context: {tenant_id} for email: {email}')
    
    # Directly query integration with join to User, filtering by email AND tenant_id
    # This ensures we only find integrations for users in the correct tenant
    integration = db.session.query(StafferlinkIntegration).join(
        User, StafferlinkIntegration.user_id == User.id
    ).filter(
        User.email == email,
        User.tenant_id == tenant_id
    ).first()
    
    if integration:
        # Double-check: verify the integration's user actually belongs to the correct tenant
        integration_user = User.query.get(integration.user_id)
        if integration_user and integration_user.tenant_id == tenant_id and integration_user.email == email:
            logger.info(f'Stafferlink integration verified for email={email}, tenant_id={tenant_id}, user_id={integration.user_id}.')
            return jsonify({
                'connected': True, 
                'email': integration.stafferlink_email,
                'agencyId': integration.stafferlink_agency_id
            }), 200
        else:
            logger.warning(f'Stafferlink integration found but user verification failed: email={email}, tenant_id={tenant_id}, integration_user_id={integration.user_id}, integration_user_tenant_id={integration_user.tenant_id if integration_user else None}')
            return jsonify({'connected': False}), 200
    else:
        logger.info(f'Stafferlink integration NOT found for email={email}, tenant_id={tenant_id}.')
        return jsonify({'connected': False}), 200

# GET /integrations/stafferlink/orders
@stafferlink_bp.route('/integrations/stafferlink/orders', methods=['GET'])
def stafferlink_orders():
    logger.info('HIT /integrations/stafferlink/orders')
    user_jwt = get_current_user()
    
    if not user_jwt or not user_jwt.get('email'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get tenant_id from context or user lookup
    tenant_id = getattr(g, 'tenant_id', None)
    if not tenant_id:
        # Fallback: get tenant_id from user lookup
        temp_user = User.query.filter_by(email=user_jwt['email']).first()
        if temp_user:
            tenant_id = temp_user.tenant_id
        else:
            return jsonify({'error': 'User not found'}), 404
    
    # Filter user by both email and tenant_id to ensure correct tenant
    user = User.query.filter_by(email=user_jwt['email'], tenant_id=tenant_id).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    integration = StafferlinkIntegration.query.filter_by(user_id=user.id).first()
    if not integration:
        return jsonify({'error': 'Stafferlink integration not found'}), 404
    
    # Get orders using saved credentials
    api = StafferlinkAPI(
        integration.stafferlink_api_key,
        integration.stafferlink_agency_id,
        integration.stafferlink_email
    )
    
    last_modified_minutes = request.args.get('last_modified_minutes', 30, type=int)
    orders = api.get_orders(last_modified_minutes)
    
    if 'error' in orders:
        return jsonify({'error': orders['error']}), 500
    
    # Transform the response to ensure it has the expected structure
    if isinstance(orders, dict):
        # Check if it has 'Orders' key (Stafferlink format)
        if 'Orders' in orders and isinstance(orders['Orders'], list):
            orders = {'orders': orders['Orders']}
        elif 'orders' not in orders:
            orders = {'orders': [orders] if orders else []}
    elif isinstance(orders, list):
        # If it's already a list, wrap it
        orders = {'orders': orders}
    else:
        # Fallback
        orders = {'orders': []}
    
    return jsonify(orders), 200

# GET /integrations/stafferlink/orders/<order_id>
@stafferlink_bp.route('/integrations/stafferlink/orders/<order_id>', methods=['GET'])
def stafferlink_order_details(order_id):
    logger.info(f'HIT /integrations/stafferlink/orders/{order_id}')
    user_jwt = get_current_user()
    
    if not user_jwt or not user_jwt.get('email'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get tenant_id from context or user lookup
    tenant_id = getattr(g, 'tenant_id', None)
    if not tenant_id:
        # Fallback: get tenant_id from user lookup
        temp_user = User.query.filter_by(email=user_jwt['email']).first()
        if temp_user:
            tenant_id = temp_user.tenant_id
        else:
            return jsonify({'error': 'User not found'}), 404
    
    # Filter user by both email and tenant_id to ensure correct tenant
    user = User.query.filter_by(email=user_jwt['email'], tenant_id=tenant_id).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    integration = StafferlinkIntegration.query.filter_by(user_id=user.id).first()
    if not integration:
        return jsonify({'error': 'Stafferlink integration not found'}), 404
    
    # Get order details using saved credentials
    api = StafferlinkAPI(
        integration.stafferlink_api_key,
        integration.stafferlink_agency_id,
        integration.stafferlink_email
    )
    
    order_details = api.get_order_details(order_id)
    
    if 'error' in order_details:
        return jsonify({'error': order_details['error']}), 500
    
    return jsonify(order_details), 200

# POST /integrations/stafferlink/disconnect
@stafferlink_bp.route('/integrations/stafferlink/disconnect', methods=['POST'])
def stafferlink_disconnect():
    logger.info('HIT /integrations/stafferlink/disconnect')
    user_jwt = get_current_user()
    
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get tenant_id from context or user lookup
    tenant_id = getattr(g, 'tenant_id', None)
    if not tenant_id:
        # Fallback: get tenant_id from user lookup
        temp_user = User.query.filter_by(email=user_jwt['email']).first()
        if temp_user:
            tenant_id = temp_user.tenant_id
        else:
            logger.error(f'User not found: {user_jwt.get("email")}')
            return jsonify({'error': 'User not found'}), 404
    
    # Filter user by both email and tenant_id to ensure correct tenant
    user = User.query.filter_by(email=user_jwt['email'], tenant_id=tenant_id).first()
    if not user:
        logger.error(f'User not found for email={user_jwt.get("email")}, tenant_id={tenant_id}')
        return jsonify({'error': 'User not found'}), 404
    
    integration = StafferlinkIntegration.query.filter_by(user_id=user.id).first()
    if integration:
        db.session.delete(integration)
        db.session.commit()
        logger.info(f'Stafferlink integration deleted for user_id={user.id}, tenant_id={tenant_id}.')
        return jsonify({'message': 'Stafferlink integration disconnected successfully'}), 200
    else:
        logger.info(f'No Stafferlink integration to delete for user_id={user.id}, tenant_id={tenant_id}.')
        return jsonify({'message': 'No Stafferlink integration found'}), 200
