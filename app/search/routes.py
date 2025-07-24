import logging
from flask import Blueprint, request, jsonify
from app.models import Tenant, User, Plan, JDSearchLog, TenantAlert, db
from app.emails.ses import send_quota_alert_email
from datetime import datetime
import jwt
import os
from .service import semantic_match

logger = logging.getLogger(__name__)

search_bp = Blueprint('search', __name__)

# Helper: decode JWT and get tenant/user

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

@search_bp.route('', methods=['POST'])
def jd_search():
    try:
        payload = get_jwt_payload()
        if not payload:
            logger.error('Unauthorized: No JWT payload')
            return jsonify({'error': 'Unauthorized: No JWT payload'}), 403
        tenant_id = int(payload.get('custom:tenant_id', 0))
        cognito_sub = payload.get('sub')
        email = payload.get('email')
        # Ensure user exists in DB
        user = User.query.filter_by(email=email, tenant_id=tenant_id).first()
        if not user:
            logger.error(f'User not found: email={email}, tenant_id={tenant_id}')
            return jsonify({'error': f'User not found: email={email}, tenant_id={tenant_id}'}), 404
        user_id = user.id
        # Count logs for this tenant this month
        now = datetime.utcnow()
        month_str = now.strftime('%Y-%m')
        first_of_month = datetime(now.year, now.month, 1)
        plan = None
        tenant = Tenant.query.get(tenant_id)
        if tenant:
            plan = Plan.query.get(tenant.plan_id)
        if not plan:
            logger.error(f'Plan not found for tenant_id={tenant_id}')
            return jsonify({'error': f'Plan not found for tenant_id={tenant_id}'}), 400
        count = JDSearchLog.query.filter(
            JDSearchLog.tenant_id == tenant_id,
            JDSearchLog.searched_at >= first_of_month
        ).count()
        quota = plan.jd_quota_per_month
        if count + 1 > quota:
            logger.error(f'Monthly quota exceeded: used={count}, quota={quota}, tenant_id={tenant_id}')
            return jsonify({'error': 'Monthly quota exceeded.', 'used': count, 'quota': quota, 'remaining': 0}), 429
        # 80% alert
        percent = int(((count + 1) / quota) * 100)
        if percent >= 80:
            alert = TenantAlert.query.filter_by(tenant_id=tenant_id, alert_type='quota_80', alert_month=month_str).first()
            if not alert:
                # Instead of sending email, log a warning
                owner = User.query.filter_by(tenant_id=tenant_id, role='owner').first()
                if owner:
                    logger.warning(f"Quota alert: {percent}% used for tenant {tenant_id}, owner: {owner.email}")
                alert = TenantAlert(tenant_id=tenant_id, alert_type='quota_80', alert_month=month_str)
                db.session.add(alert)
                db.session.commit()
        # Log search
        log = JDSearchLog(tenant_id=tenant_id, user_id=user_id)
        db.session.add(log)
        db.session.commit()
        # Call semantic matching
        job_desc = request.get_json().get('job_description')
        results = semantic_match(job_desc)
        return jsonify(results), 200
    except Exception as e:
        logger.error(f"Error in /search: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@search_bp.route('/quota', methods=['GET'])
def get_quota():
    try:
        payload = get_jwt_payload()
        if not payload:
            logger.error('Unauthorized: No JWT payload')
            return jsonify({'error': 'Unauthorized: No JWT payload'}), 403
        tenant_id = int(payload.get('custom:tenant_id', 0))
        email = payload.get('email')
        user = User.query.filter_by(email=email, tenant_id=tenant_id).first()
        if not user:
            logger.error(f'User not found: email={email}, tenant_id={tenant_id}')
            return jsonify({'error': f'User not found: email={email}, tenant_id={tenant_id}'}), 404
        now = datetime.utcnow()
        first_of_month = datetime(now.year, now.month, 1)
        tenant = Tenant.query.get(tenant_id)
        plan = Plan.query.get(tenant.plan_id) if tenant else None
        if not plan:
            logger.error(f'Plan not found for tenant_id={tenant_id}')
            return jsonify({'error': f'Plan not found for tenant_id={tenant_id}'}), 400
        count = JDSearchLog.query.filter(
            JDSearchLog.tenant_id == tenant_id,
            JDSearchLog.searched_at >= first_of_month
        ).count()
        quota = plan.jd_quota_per_month
        remaining = max(0, quota - count)
        return jsonify({'quota': quota, 'used': count, 'remaining': remaining}), 200
    except Exception as e:
        logger.error(f"Error in /search/quota: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 