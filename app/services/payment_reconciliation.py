"""
Payment Reconciliation Service
Automatically reconciles stuck payments by checking status with Razorpay
"""
import requests
from datetime import datetime, timedelta
from sqlalchemy import and_, or_
from app.models import db, PaymentTransaction, PayrollSettings
from app.simple_logger import get_logger
from app.utils.payment_security import PaymentEncryption, PaymentAuditLogger

logger = get_logger(__name__)
audit_logger = PaymentAuditLogger()


class PaymentReconciliationService:
    """Service for reconciling payment statuses with Razorpay"""
    
    def __init__(self, tenant_id=None):
        self.tenant_id = tenant_id
    
    def _get_razorpay_auth(self, settings):
        """Get Razorpay Basic Auth header"""
        import base64
        encryption = PaymentEncryption()
        
        key_id = settings.razorpay_key_id
        key_secret = settings.razorpay_key_secret
        if key_secret and key_secret.startswith('enc:'):
            key_secret = encryption.decrypt(key_secret[4:])
        
        credentials = f"{key_id}:{key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    def reconcile_stuck_payments(self, hours_threshold=2, limit=100):
        """
        Reconcile payments stuck in 'processing' or 'pending' state
        
        Args:
            hours_threshold: Hours after which a payment is considered stuck
            limit: Maximum number of payments to reconcile in one run
        
        Returns:
            dict: Reconciliation results
        """
        results = {
            'processed': 0,
            'updated': 0,
            'failed': 0,
            'errors': []
        }
        
        # Find stuck payments
        threshold_time = datetime.utcnow() - timedelta(hours=hours_threshold)
        
        # Build query - handle both processed_at and initiated_at
        stuck_payments = PaymentTransaction.query.filter(
            PaymentTransaction.status.in_(['processing', 'pending']),
            or_(
                PaymentTransaction.processed_at < threshold_time,
                and_(
                    PaymentTransaction.processed_at.is_(None),
                    PaymentTransaction.initiated_at < threshold_time
                )
            ),
            PaymentTransaction.gateway == 'razorpay',
            PaymentTransaction.gateway_payout_id.isnot(None)
        ).limit(limit).all()
        
        if not stuck_payments:
            logger.info("No stuck payments found for reconciliation")
            return results
        
        logger.info(f"Found {len(stuck_payments)} stuck payments to reconcile")
        
        # Group by tenant_id to batch API calls
        payments_by_tenant = {}
        for payment in stuck_payments:
            tenant_id = payment.tenant_id
            if tenant_id not in payments_by_tenant:
                payments_by_tenant[tenant_id] = []
            payments_by_tenant[tenant_id].append(payment)
        
        # Reconcile payments for each tenant
        for tenant_id, payments in payments_by_tenant.items():
            try:
                settings = PayrollSettings.query.filter_by(tenant_id=tenant_id).first()
                if not settings or settings.payment_gateway != 'razorpay':
                    logger.warning(f"Razorpay not configured for tenant {tenant_id}")
                    continue
                
                headers = {
                    "Authorization": self._get_razorpay_auth(settings),
                    "Content-Type": "application/json"
                }
                
                for payment in payments:
                    try:
                        results['processed'] += 1
                        
                        # Fetch latest status from Razorpay
                        response = requests.get(
                            f"https://api.razorpay.com/v1/payouts/{payment.gateway_payout_id}",
                            headers=headers,
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            payout_data = response.json()
                            razorpay_status = payout_data.get('status', '').lower()
                            old_status = payment.status
                            
                            # Only update if status has changed
                            if old_status in ['processing', 'pending']:
                                # Update status based on Razorpay status
                                if razorpay_status == 'processed':
                                    payment.status = 'success'
                                    payment.completed_at = datetime.utcnow()
                                    
                                    # Update payslip status
                                    from app.models import PayRunPayslip
                                    payrun_payslip = PayRunPayslip.query.filter_by(
                                        pay_run_id=payment.pay_run_id,
                                        payslip_id=payment.payslip_id
                                    ).first()
                                    if payrun_payslip:
                                        payrun_payslip.payment_status = 'processed'
                                    
                                    results['updated'] += 1
                                    logger.info(
                                        f"Reconciled payment {payment.id}: {old_status} -> success "
                                        f"(payout_id: {payment.gateway_payout_id})"
                                    )
                                    
                                elif razorpay_status in ['failed', 'reversed', 'cancelled']:
                                    payment.status = 'failed'
                                    payment.failure_reason = payout_data.get('failure_reason', 'Payment failed')
                                    payment.completed_at = datetime.utcnow()
                                    
                                    # Update payslip status
                                    from app.models import PayRunPayslip
                                    payrun_payslip = PayRunPayslip.query.filter_by(
                                        pay_run_id=payment.pay_run_id,
                                        payslip_id=payment.payslip_id
                                    ).first()
                                    if payrun_payslip:
                                        payrun_payslip.payment_status = 'failed'
                                    
                                    results['updated'] += 1
                                    logger.info(
                                        f"Reconciled payment {payment.id}: {old_status} -> failed "
                                        f"(payout_id: {payment.gateway_payout_id})"
                                    )
                                
                                # Update gateway response
                                payment.gateway_response = payout_data
                                payment.processed_at = datetime.utcnow()
                            
                        elif response.status_code == 404:
                            # Payout not found - mark as failed (only if still in processing state)
                            if payment.status in ['processing', 'pending']:
                                payment.status = 'failed'
                                payment.failure_reason = 'Payout not found in Razorpay'
                                payment.completed_at = datetime.utcnow()
                                results['updated'] += 1
                                logger.warning(
                                    f"Payment {payment.id} payout not found in Razorpay: {payment.gateway_payout_id}"
                                )
                        else:
                            error_data = response.json() if response.text else {}
                            error_msg = error_data.get('error', {}).get('description', 'Unknown error')
                            results['failed'] += 1
                            results['errors'].append(f"Payment {payment.id}: {error_msg}")
                            logger.error(
                                f"Failed to reconcile payment {payment.id}: "
                                f"{response.status_code} - {error_msg}"
                            )
                    
                    except Exception as e:
                        results['failed'] += 1
                        results['errors'].append(f"Payment {payment.id}: {str(e)}")
                        logger.error(f"Error reconciling payment {payment.id}: {str(e)}")
                
                # Commit changes for this tenant
                db.session.commit()
                
            except Exception as e:
                logger.error(f"Error reconciling payments for tenant {tenant_id}: {str(e)}")
                results['errors'].append(f"Tenant {tenant_id}: {str(e)}")
        
        logger.info(
            f"Reconciliation complete: {results['processed']} processed, "
            f"{results['updated']} updated, {results['failed']} failed"
        )
        
        return results


def reconcile_all_tenants(hours_threshold=2, limit=100):
    """Reconcile payments for all tenants with Razorpay configured"""
    tenants = PayrollSettings.query.filter(
        PayrollSettings.payment_gateway == 'razorpay',
        PayrollSettings.razorpay_key_id.isnot(None)
    ).all()
    
    total_results = {
        'tenants_processed': 0,
        'total_processed': 0,
        'total_updated': 0,
        'total_failed': 0,
        'errors': []
    }
    
    for settings in tenants:
        try:
            service = PaymentReconciliationService(tenant_id=settings.tenant_id)
            results = service.reconcile_stuck_payments(hours_threshold=hours_threshold, limit=limit)
            
            total_results['tenants_processed'] += 1
            total_results['total_processed'] += results['processed']
            total_results['total_updated'] += results['updated']
            total_results['total_failed'] += results['failed']
            total_results['errors'].extend(results['errors'])
            
        except Exception as e:
            logger.error(f"Error processing tenant {settings.tenant_id}: {str(e)}")
            total_results['errors'].append(f"Tenant {settings.tenant_id}: {str(e)}")
    
    return total_results

