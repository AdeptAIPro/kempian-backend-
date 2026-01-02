from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
import os
import requests
from datetime import datetime, timedelta
from app.models import CeipalIntegration, User, db
from app.utils import get_current_user
import base64
import logging
logger = get_logger("ceipal")

ceipal_bp = Blueprint('ceipal', __name__)

class CeipalAuth:
    def __init__(self):
        self.auth_url = "https://api.ceipal.com/v1/createAuthtoken/"
        self.email = os.getenv("CEIPAL_EMAIL")
        self.password = os.getenv("CEIPAL_PASSWORD")
        self.api_key = os.getenv("CEIPAL_API_KEY")
        self.token = None
        self.token_expiry = None

    def authenticate(self):
        payload = {"email": self.email, "password": self.password, "api_key": self.api_key, "json": "1"}
        try:
            response = requests.post(self.auth_url, json=payload)
            response.raise_for_status()
            data = response.json()
            if "access_token" in data:
                self.token = data["access_token"]
                self.token_expiry = datetime.now() + timedelta(hours=1)
                return True
        except Exception as e:
            print(f"CEIPAL auth error: {str(e)}")
        return False

    def get_token(self):
        if not self.token or datetime.now() >= self.token_expiry:
            if not self.authenticate():
                return None
        return self.token

class CeipalJobPostingsAPI:
    def __init__(self, auth):
        self.auth = auth
        self.base_url = "https://api.ceipal.com"
        self.job_postings_endpoint = "/getCustomJobPostingDetails/Z3RkUkt2OXZJVld2MjFpOVRSTXoxZz09/e6e04af381e7f42eeb7f942c8bf5ab6d"
        self.job_details_endpoint = "/v1/getJobPostingDetails/"

    def get_job_postings(self, paging_length=30):
        token = self.auth.get_token()
        if not token:
            return []
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{self.base_url}{self.job_postings_endpoint}", headers=headers, params={"paging_length": paging_length})
        return response.json().get("results", [])

    def get_job_details(self, job_code):
        token = self.auth.get_token()
        if not token:
            return {}
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{self.base_url}{self.job_details_endpoint}", headers=headers, params={"job_id": job_code})
        return response.json()

@ceipal_bp.route('/api/v1/ceipal/jobs', methods=['GET'])
def get_ceipal_jobs():
    try:
        count = int(request.args.get('count', 50))
        auth = CeipalAuth()
        if not auth.authenticate():
            return jsonify({"error": "Authentication failed"}), 401
        jobs = CeipalJobPostingsAPI(auth).get_job_postings(paging_length=count)
        return jsonify({"jobs": jobs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@ceipal_bp.route('/api/v1/ceipal/getJobDetails', methods=['GET'])
def get_job_details():
    job_code = request.args.get('job_code')
    if not job_code:
        return jsonify({"error": "Missing job_code parameter"}), 400
    try:
        job = CeipalJobPostingsAPI(CeipalAuth()).get_job_details(job_code)
        return jsonify({"job_details": job})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# POST /integrations/ceipal/connect
@ceipal_bp.route('/integrations/ceipal/connect', methods=['POST'])
def ceipal_connect():
    logger.info('HIT /integrations/ceipal/connect')
    user_jwt = get_current_user()
    logger.info(f'user_jwt: {user_jwt}')
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        logger.error(f'User not found: {user_jwt.get("email")}')
        return jsonify({'error': 'User not found'}), 404
    data = request.get_json()
    logger.info(f'Received data: {data}')
    ceipal_email = data.get('email')
    ceipal_api_key = data.get('apiKey')
    ceipal_password = data.get('name')  # frontend uses 'name' for password
    if not ceipal_email or not ceipal_api_key or not ceipal_password:
        logger.error('Missing fields in request')
        return jsonify({'error': 'Missing fields'}), 400
    # Validate credentials with Ceipal API
    payload = {"email": ceipal_email, "password": ceipal_password, "api_key": ceipal_api_key, "json": "1"}
    try:
        resp = requests.post("https://api.ceipal.com/v1/createAuthtoken/", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "access_token" not in data:
            logger.error('Ceipal authentication failed: No access_token')
            return jsonify({'error': 'Invalid Ceipal credentials'}), 401
    except Exception as e:
        logger.error(f'Ceipal authentication error: {e}')
        return jsonify({'error': 'Invalid Ceipal credentials'}), 401
    # Encrypt password (base64 for demo)
    enc_password = base64.b64encode(ceipal_password.encode()).decode()
    integration = CeipalIntegration.query.filter_by(user_id=user.id).first()
    if integration:
        integration.ceipal_email = ceipal_email
        integration.ceipal_api_key = ceipal_api_key
        integration.ceipal_password = enc_password
    else:
        integration = CeipalIntegration(
            user_id=user.id,
            ceipal_email=ceipal_email,
            ceipal_api_key=ceipal_api_key,
            ceipal_password=enc_password
        )
        db.session.add(integration)
    db.session.commit()
    logger.info('Ceipal integration saved.')
    return jsonify({'message': 'Ceipal integration saved.'}), 200

# GET /integrations/ceipal/status
@ceipal_bp.route('/integrations/ceipal/status', methods=['GET'])
def ceipal_status():
    logger.info('HIT /integrations/ceipal/status')
    user_jwt = get_current_user()
    logger.info(f'user_jwt: {user_jwt}')
    if not user_jwt or not user_jwt.get('email'):
        logger.error('No user_jwt or email')
        return jsonify({'connected': False}), 200
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        logger.error(f'User not found: {user_jwt.get("email")}')
        return jsonify({'connected': False}), 200
    integration = CeipalIntegration.query.filter_by(user_id=user.id).first()
    if integration:
        logger.info('Ceipal integration found.')
        return jsonify({'connected': True, 'email': integration.ceipal_email}), 200
    else:
        logger.info('Ceipal integration not found.')
        return jsonify({'connected': False}), 200

# GET /integrations/ceipal/jobs
@ceipal_bp.route('/integrations/ceipal/jobs', methods=['GET'])
def ceipal_jobs():
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    integration = CeipalIntegration.query.filter_by(user_id=user.id).first()
    if not integration:
        return jsonify({'error': 'Ceipal integration not found'}), 404
    ceipal_email = integration.ceipal_email
    ceipal_api_key = integration.ceipal_api_key
    ceipal_password = base64.b64decode(integration.ceipal_password.encode()).decode()
    # Authenticate and fetch jobs
    payload = {"email": ceipal_email, "password": ceipal_password, "api_key": ceipal_api_key, "json": "1"}
    try:
        resp = requests.post("https://api.ceipal.com/v1/createAuthtoken/", json=payload)
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        if not token:
            return jsonify({'error': 'Failed to get Ceipal token'}), 400
        headers = {"Authorization": f"Bearer {token}"}
        jobs_resp = requests.get("https://api.ceipal.com/getCustomJobPostingDetails/Z3RkUkt2OXZJVld2MjFpOVRSTXoxZz09/e6e04af381e7f42eeb7f942c8bf5ab6d", headers=headers, params={"paging_length": 10})
        jobs_resp.raise_for_status()
        return jsonify(jobs_resp.json()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# POST /integrations/ceipal/disconnect
@ceipal_bp.route('/integrations/ceipal/disconnect', methods=['POST'])
def ceipal_disconnect():
    logger.info('HIT /integrations/ceipal/disconnect')
    user_jwt = get_current_user()
    logger.info(f'user_jwt: {user_jwt}')
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        logger.error(f'User not found: {user_jwt.get("email")}')
        return jsonify({'error': 'User not found'}), 404
    integration = CeipalIntegration.query.filter_by(user_id=user.id).first()
    if integration:
        db.session.delete(integration)
        db.session.commit()
        logger.info('Ceipal integration deleted.')
        return jsonify({'message': 'Ceipal integration disconnected.'}), 200
    else:
        logger.info('No Ceipal integration to delete.')
        return jsonify({'message': 'No Ceipal integration found.'}), 200 