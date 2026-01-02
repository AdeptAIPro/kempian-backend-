"""
Webhook handlers for Jobvite webhooks.
Tenant resolution: via companyId in URL path.
"""

from flask import Blueprint, request, jsonify
from app.models import JobviteSettings, JobviteWebhookLog, db
from app.jobvite.crypto import verify_webhook_signature
from app.jobvite.client_v2 import JobviteV2Client
from app.jobvite.utils import get_base_urls
from app.jobvite.logging_utils import log_webhook_event
from app.simple_logger import get_logger
from datetime import datetime
from typing import Optional

logger = get_logger("jobvite_webhooks")

webhook_bp = Blueprint('jobvite_webhooks', __name__)

def resolve_tenant_from_company_id(company_id: str) -> Optional[JobviteSettings]:
    """
    Resolve tenant settings from companyId.
    
    Args:
        company_id: Jobvite company ID from webhook URL path
    
    Returns:
        JobviteSettings if found, None otherwise
    """
    return JobviteSettings.query.filter_by(
        company_id=company_id,
        is_active=True
    ).first()

@webhook_bp.route('/webhooks/jobvite/<company_id>/candidate', methods=['POST'])
def candidate_webhook(company_id: str):
    """
    Handle candidate webhook (workflowUpdate, etc.).
    
    Tenant resolution: companyId from URL path.
    """
    raw_body = request.get_data()
    signature_header = request.headers.get('X-Jobvite-Event-Signature')
    
    # Resolve tenant from companyId
    settings = resolve_tenant_from_company_id(company_id)
    if not settings:
        logger.warning(f"Webhook received for unknown companyId: {company_id}")
        return jsonify({'error': 'Invalid company ID'}), 404
    
    # Verify signature
    signature_valid = False
    if signature_header and settings.webhook_signing_key_encrypted:
        signing_key = settings.decrypt_webhook_key()
        if signing_key:
            signature_valid = verify_webhook_signature(raw_body, signature_header, signing_key)
            if not signature_valid:
                logger.warning(f"Invalid webhook signature for companyId: {company_id}")
        else:
            logger.warning(f"No webhook signing key for companyId: {company_id}")
    else:
        logger.warning(f"Missing signature header or signing key for companyId: {company_id}")
    
    # Parse payload
    try:
        payload = request.get_json()
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        payload = {}
    
    event_type = payload.get('eventType', 'unknown')
    candidate_id = payload.get('candidateId')
    application_id = payload.get('applicationId')
    entity_id = candidate_id or application_id
    
    # Log webhook (always log, even if signature invalid)
    webhook_log = JobviteWebhookLog(
        tenant_id=settings.tenant_id,
        event_type=event_type,
        source='candidate',
        jobvite_entity_id=entity_id,
        payload=payload,
        signature=signature_header,
        signature_valid=signature_valid,
        processed=False
    )
    db.session.add(webhook_log)
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving webhook log: {e}")
        return jsonify({'error': 'Failed to process webhook'}), 500
    
    # Return 401 if signature invalid
    if not signature_valid:
        return jsonify({'error': 'Invalid signature'}), 401
    
    # Process webhook (async or inline)
    try:
        # Fetch updated candidate from Jobvite
        base_urls = get_base_urls(settings.environment)
        client = JobviteV2Client(
            api_key=settings.api_key,
            api_secret=settings.decrypt_secret(),
            company_id=settings.company_id,
            base_url=base_urls['v2']
        )
        
        candidate_data = None
        if candidate_id:
            candidate_data = client.get_candidate(candidate_id=candidate_id)
        elif application_id:
            candidate_data = client.get_candidate(application_id=application_id)
        
        if candidate_data:
            # Upsert to JobviteCandidate table
            from app.models import JobviteCandidate
            from app.jobvite.sync import _upsert_candidate_from_jobvite_data
            
            _upsert_candidate_from_jobvite_data(
                tenant_id=settings.tenant_id,
                candidate_data=candidate_data,
                db=db
            )
        
        webhook_log.processed = True
        db.session.commit()
        
        log_webhook_event(
            tenant_id=settings.tenant_id,
            company_id=company_id,
            event_type=event_type,
            source='candidate',
            signature_valid=signature_valid,
            processed=True
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing webhook: {error_msg}")
        webhook_log.error_message = error_msg
        db.session.commit()
        
        log_webhook_event(
            tenant_id=settings.tenant_id,
            company_id=company_id,
            event_type=event_type,
            source='candidate',
            signature_valid=signature_valid,
            processed=False,
            error=error_msg
        )
    
    return jsonify({'success': True}), 200

@webhook_bp.route('/webhooks/jobvite/<company_id>/job', methods=['POST'])
def job_webhook(company_id: str):
    """
    Handle job webhook (jobUpdate, etc.).
    
    Similar to candidate_webhook but for jobs.
    """
    raw_body = request.get_data()
    signature_header = request.headers.get('X-Jobvite-Event-Signature')
    
    # Resolve tenant
    settings = resolve_tenant_from_company_id(company_id)
    if not settings:
        return jsonify({'error': 'Invalid company ID'}), 404
    
    # Verify signature
    signature_valid = False
    if signature_header and settings.webhook_signing_key_encrypted:
        signing_key = settings.decrypt_webhook_key()
        if signing_key:
            signature_valid = verify_webhook_signature(raw_body, signature_header, signing_key)
    
    # Parse payload
    try:
        payload = request.get_json()
    except Exception:
        payload = {}
    
    event_type = payload.get('eventType', 'unknown')
    job_id = payload.get('jobId')
    
    # Log webhook
    webhook_log = JobviteWebhookLog(
        tenant_id=settings.tenant_id,
        event_type=event_type,
        source='job',
        jobvite_entity_id=job_id,
        payload=payload,
        signature=signature_header,
        signature_valid=signature_valid,
        processed=False
    )
    db.session.add(webhook_log)
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving webhook log: {e}")
        return jsonify({'error': 'Failed to process webhook'}), 500
    
    if not signature_valid:
        return jsonify({'error': 'Invalid signature'}), 401
    
    # Process webhook
    try:
        base_urls = get_base_urls(settings.environment)
        client = JobviteV2Client(
            api_key=settings.api_key,
            api_secret=settings.decrypt_secret(),
            company_id=settings.company_id,
            base_url=base_urls['v2']
        )
        
        if job_id:
            job_data = client.get_job(job_id=job_id)
            if job_data:
                # Upsert to JobviteJob table
                from app.models import JobviteJob
                from app.jobvite.sync import _upsert_job_from_jobvite_data
                
                _upsert_job_from_jobvite_data(
                    tenant_id=settings.tenant_id,
                    job_data=job_data,
                    db=db
                )
        
        webhook_log.processed = True
        db.session.commit()
    except Exception as e:
        logger.error(f"Error processing job webhook: {e}")
        webhook_log.error_message = str(e)
        db.session.commit()
    
    return jsonify({'success': True}), 200

