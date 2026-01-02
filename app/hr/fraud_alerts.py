"""
Fraud Alerts API
Manage fraud alerts for review and approval
"""
from flask import Blueprint, request, jsonify
from app.models import db, FraudAlert, User, PayRun, PaymentTransaction
from app.auth_utils import get_current_user_flexible, get_current_tenant_id
from app.simple_logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

fraud_alerts_bp = Blueprint('fraud_alerts', __name__)


def _auth_or_401():
    user = get_current_user_flexible()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


@fraud_alerts_bp.route('/', methods=['GET'], strict_slashes=False)
def list_fraud_alerts():
    """
    List fraud alerts with comprehensive filtering
    
    Query Parameters:
    - status: 'pending', 'reviewed', 'approved', 'rejected' (optional)
    - severity: 'low', 'medium', 'high', 'critical' (optional)
    - pay_run_id: Filter by pay run (optional)
    - date_from: ISO date string (optional)
    - date_to: ISO date string (optional)
    - limit: Number of results (default: 100, max: 1000)
    - offset: Pagination offset (default: 0)
    """
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Check permissions - only admin/owner can view fraud alerts
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view fraud alerts'}), 403
    
    tenant_id = get_current_tenant_id()
    
    # Get query parameters
    pay_run_id = request.args.get('pay_run_id', type=int)
    status = request.args.get('status')  # 'pending', 'reviewed', 'approved', 'rejected'
    severity = request.args.get('severity')  # 'low', 'medium', 'high', 'critical'
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    limit = min(request.args.get('limit', 100, type=int), 1000)
    offset = request.args.get('offset', 0, type=int)
    
    # Build query
    query = FraudAlert.query.filter_by(tenant_id=tenant_id)
    
    if pay_run_id:
        query = query.filter_by(pay_run_id=pay_run_id)
    
    if status:
        # Map 'open' to 'pending' for backward compatibility
        if status == 'open':
            status = 'pending'
        query = query.filter_by(status=status)
    
    if severity:
        query = query.filter_by(severity=severity)
    
    # Date range filtering
    if date_from:
        try:
            from datetime import datetime
            date_from_obj = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            query = query.filter(FraudAlert.created_at >= date_from_obj)
        except ValueError:
            return jsonify({'error': 'Invalid date_from format. Use ISO 8601 format.'}), 400
    
    if date_to:
        try:
            from datetime import datetime
            date_to_obj = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            query = query.filter(FraudAlert.created_at <= date_to_obj)
        except ValueError:
            return jsonify({'error': 'Invalid date_to format. Use ISO 8601 format.'}), 400
    
    # Get total count before pagination
    total_count = query.count()
    
    # Apply pagination and ordering
    alerts = query.order_by(FraudAlert.created_at.desc()).offset(offset).limit(limit).all()
    
    return jsonify({
        'alerts': [alert.to_dict() for alert in alerts],
        'total': total_count,
        'limit': limit,
        'offset': offset,
        'has_more': (offset + limit) < total_count
    }), 200


@fraud_alerts_bp.route('/<int:alert_id>/review', methods=['POST'], strict_slashes=False)
def review_fraud_alert(alert_id):
    """
    Review/Override a fraud alert (STRICT ADMIN-ONLY)
    
    Request Body:
    {
        "decision": "approve" | "reject" (required),
        "review_notes": "string" (mandatory),
        "override": true | false (optional, default: false)
    }
    
    Rules:
    - Only admin/owner can review
    - review_notes is mandatory
    - If rejected → payout is permanently blocked
    - If approved → payout may resume
    - All actions are immutable in audit logs
    """
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # STRICT: Only admin/owner can review fraud alerts
    if db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'Only administrators can review fraud alerts'}), 403
    
    alert = FraudAlert.query.get(alert_id)
    if not alert:
        return jsonify({'error': 'Fraud alert not found'}), 404
    
    # Check tenant
    tenant_id = get_current_tenant_id()
    if alert.tenant_id != tenant_id and db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to view this alert'}), 403
    
    data = request.get_json() or {}
    decision = data.get('decision')  # 'approve' or 'reject'
    review_notes = data.get('review_notes', '').strip()
    override = data.get('override', False)
    
    # Validation
    if decision not in ['approve', 'reject']:
        return jsonify({'error': 'Decision must be "approve" or "reject"'}), 400
    
    if not review_notes:
        return jsonify({'error': 'review_notes is mandatory and cannot be empty'}), 400
    
    # Audit: Log before making changes
    from app.utils.payment_security import PaymentAuditLogger
    audit_logger = PaymentAuditLogger()
    
    audit_logger.log_security_event('fraud_alert_review', {
        'alert_id': alert_id,
        'decision': decision,
        'override': override,
        'risk_score': float(alert.risk_score),
        'severity': alert.severity,
        'previous_status': alert.status
    }, db_user.id)
    
    # Update alert (immutable audit trail)
    old_status = alert.status
    alert.status = 'approved' if decision == 'approve' else 'rejected'
    alert.reviewed_by = db_user.id
    alert.reviewed_at = datetime.utcnow()
    alert.review_notes = review_notes
    
    # If rejected, block the associated payment transaction
    if decision == 'reject' and alert.payment_transaction_id:
        from app.models import PaymentTransaction
        transaction = PaymentTransaction.query.get(alert.payment_transaction_id)
        if transaction:
            transaction.requires_manual_review = True
            transaction.reviewed_by = db_user.id
            transaction.reviewed_at = datetime.utcnow()
            transaction.review_notes = f"Blocked by fraud alert {alert_id}: {review_notes}"
            # Mark as failed if not already
            if transaction.status in ['pending', 'processing']:
                transaction.status = 'failed'
                transaction.failure_reason = f"Blocked due to fraud alert rejection: {review_notes}"
    
    # If approved, allow payment to proceed (if still pending)
    if decision == 'approve' and alert.payment_transaction_id:
        from app.models import PaymentTransaction
        transaction = PaymentTransaction.query.get(alert.payment_transaction_id)
        if transaction and transaction.status == 'pending' and transaction.requires_manual_review:
            transaction.requires_manual_review = False
            transaction.reviewed_by = db_user.id
            transaction.reviewed_at = datetime.utcnow()
            transaction.review_notes = f"Approved by fraud alert review {alert_id}: {review_notes}"
    
    db.session.commit()
    
    logger.info(
        f"Fraud alert {alert_id} {decision}d by admin {db_user.id} (user_id={db_user.id}). "
        f"Risk score: {alert.risk_score}, Override: {override}, Notes: {review_notes[:50]}"
    )
    
    # Final audit log
    audit_logger.log_security_event('fraud_alert_reviewed', {
        'alert_id': alert_id,
        'decision': decision,
        'new_status': alert.status,
        'reviewer_id': db_user.id
    }, db_user.id)
    
    return jsonify({
        'alert': alert.to_dict(),
        'message': f'Fraud alert {decision}d successfully',
        'irreversible': True,
        'audit_logged': True
    }), 200


@fraud_alerts_bp.route('/<int:alert_id>', methods=['GET'], strict_slashes=False)
def get_fraud_alert(alert_id):
    """Get a single fraud alert"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    alert = FraudAlert.query.get(alert_id)
    if not alert:
        return jsonify({'error': 'Fraud alert not found'}), 404
    
    # Check tenant
    tenant_id = get_current_tenant_id()
    if alert.tenant_id != tenant_id and db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to view this alert'}), 403
    
    return jsonify({'alert': alert.to_dict()}), 200

