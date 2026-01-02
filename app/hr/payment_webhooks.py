"""
Payment Webhook Handlers
Handles webhook callbacks from payment gateways with security verification
"""
from flask import Blueprint, request, jsonify
from app.models import db, PaymentTransaction, PayrollSettings
from app.utils.payment_security import WebhookVerification, PaymentAuditLogger, DataMasking
from app.simple_logger import get_logger
from datetime import datetime

logger = get_logger(__name__)
audit_logger = PaymentAuditLogger()

payment_webhooks_bp = Blueprint('payment_webhooks', __name__)


@payment_webhooks_bp.route('/razorpay', methods=['POST'], strict_slashes=False)
def razorpay_webhook():
    """
    Handle Razorpay webhook callbacks
    Verifies signature and updates payment status
    """
    try:
        # Get webhook signature from headers
        webhook_signature = request.headers.get('X-Razorpay-Signature')
        if not webhook_signature:
            logger.warning("Razorpay webhook received without signature")
            return jsonify({'error': 'Missing signature'}), 400
        
        # Get raw payload
        payload = request.get_data(as_text=True)
        
        # Get webhook secret from settings - try to find by transaction tenant_id first
        # Parse webhook payload to get payout_id first
        try:
            event_data = request.get_json()
        except Exception as e:
            logger.error(f"Failed to parse webhook payload: {str(e)}")
            return jsonify({'error': 'Invalid JSON payload'}), 400
        
        payout_id = None
        if event_data and isinstance(event_data, dict):
            payout_data = event_data.get('payload', {})
            if isinstance(payout_data, dict):
                payout_obj = payout_data.get('payout', {})
                if isinstance(payout_obj, dict):
                    payout_id = payout_obj.get('id')
        
        # Try to find settings by transaction tenant_id
        settings = None
        if payout_id:
            transaction = PaymentTransaction.query.filter_by(gateway_payout_id=payout_id).first()
            if transaction:
                settings = PayrollSettings.query.filter_by(tenant_id=transaction.tenant_id).first()
        
        # Fallback: get from first tenant with Razorpay configured
        if not settings:
            settings = PayrollSettings.query.filter(
                PayrollSettings.payment_gateway == 'razorpay',
                PayrollSettings.razorpay_webhook_secret.isnot(None)
            ).first()
        
        if not settings:
            logger.error("Razorpay webhook secret not found")
            return jsonify({'error': 'Webhook secret not configured'}), 500
        
        # Use webhook secret (separate from key secret)
        from app.utils.payment_security import PaymentEncryption
        encryption = PaymentEncryption()
        webhook_secret = settings.razorpay_webhook_secret
        if not webhook_secret:
            # Fallback to key_secret for backward compatibility (deprecated)
            logger.warning("Using key_secret as webhook secret (deprecated - configure webhook_secret)")
            webhook_secret = settings.razorpay_key_secret
        
        if webhook_secret and webhook_secret.startswith('enc:'):
            webhook_secret = encryption.decrypt(webhook_secret[4:])
        
        # Verify webhook signature
        is_valid = WebhookVerification.verify_razorpay_webhook(
            payload,
            webhook_signature,
            webhook_secret
        )
        
        if not is_valid:
            audit_logger.log_security_event('invalid_webhook_signature', {
                'gateway': 'razorpay',
                'ip': request.remote_addr
            })
            logger.warning(f"Invalid Razorpay webhook signature from {request.remote_addr}")
            return jsonify({'error': 'Invalid signature'}), 401
        
        # Parse webhook payload (already parsed above)
        if not event_data or not isinstance(event_data, dict):
            logger.error("Invalid webhook payload structure")
            return jsonify({'error': 'Invalid webhook payload'}), 400
        
        event_type = event_data.get('event')
        payload_data = event_data.get('payload', {})
        if not isinstance(payload_data, dict):
            logger.error("Invalid payload structure in webhook")
            return jsonify({'error': 'Invalid payload structure'}), 400
        
        payout_obj = payload_data.get('payout', {})
        if not isinstance(payout_obj, dict):
            logger.error("Invalid payout object in webhook")
            return jsonify({'error': 'Invalid payout object'}), 400
        
        payout_id = payout_obj.get('id')
        status = payout_obj.get('status', '').lower()
        
        # Find transaction by gateway payout ID
        transaction = PaymentTransaction.query.filter_by(
            gateway_payout_id=payout_id
        ).first()
        
        if not transaction:
            logger.warning(f"Transaction not found for payout_id: {payout_id}")
            return jsonify({'error': 'Transaction not found'}), 404
        
        # Update transaction status
        old_status = transaction.status
        
        if status == 'processed':
            transaction.status = 'success'
            transaction.completed_at = datetime.utcnow()
            
            # Update payslip status
            from app.models import PayRunPayslip
            payrun_payslip = PayRunPayslip.query.filter_by(
                pay_run_id=transaction.pay_run_id,
                payslip_id=transaction.payslip_id
            ).first()
            if payrun_payslip:
                payrun_payslip.payment_status = 'processed'
            
            audit_logger.log_payment_success(
                transaction.id,
                payout_id,
                float(transaction.amount)
            )
            
        elif status == 'failed' or status == 'reversed':
            transaction.status = 'failed'
            transaction.failure_reason = payout_obj.get('failure_reason', 'Payment failed')
            transaction.completed_at = datetime.utcnow()
            
            # Update payslip status
            from app.models import PayRunPayslip
            payrun_payslip = PayRunPayslip.query.filter_by(
                pay_run_id=transaction.pay_run_id,
                payslip_id=transaction.payslip_id
            ).first()
            if payrun_payslip:
                payrun_payslip.payment_status = 'failed'
            
            audit_logger.log_payment_failure(
                transaction.id,
                payout_obj.get('failure_reason', 'Payment failed'),
                payout_obj
            )
            
        elif status == 'queued' or status == 'pending':
            transaction.status = 'processing'
        
        # Update gateway response
        transaction.gateway_response = payout_obj
        transaction.processed_at = datetime.utcnow()
        
        db.session.commit()
        
        logger.info(
            f"Webhook processed: transaction_id={transaction.id}, "
            f"payout_id={payout_id}, status={status}, old_status={old_status}"
        )
        
        return jsonify({'success': True, 'transaction_id': transaction.id}), 200
        
    except Exception as e:
        logger.error(f"Error processing Razorpay webhook: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@payment_webhooks_bp.route('/health', methods=['GET'], strict_slashes=False)
def webhook_health():
    """Health check for webhook endpoint"""
    return jsonify({'status': 'ok', 'service': 'payment_webhooks'}), 200

