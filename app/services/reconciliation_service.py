"""
Reconciliation Service
Reconciles payment status with Razorpay API to handle webhook failures
"""
import requests
from datetime import datetime, timedelta
from app.models import db, PaymentTransaction, PayrollSettings, PayRun, PayRunPayslip
from app.simple_logger import get_logger
from app.utils.payment_security import PaymentEncryption

logger = get_logger(__name__)


class ReconciliationService:
    """Reconcile payment status with Razorpay"""
    
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id
        self.settings = PayrollSettings.query.filter_by(tenant_id=tenant_id).first()
        if not self.settings or self.settings.payment_gateway != 'razorpay':
            raise ValueError("Razorpay not configured for this tenant")
    
    def _get_razorpay_auth(self):
        """Get Razorpay Basic Auth header"""
        import base64
        encryption = PaymentEncryption()
        
        key_id = self.settings.razorpay_key_id
        key_secret = self.settings.razorpay_key_secret
        
        if key_secret and key_secret.startswith('enc:'):
            key_secret = encryption.decrypt(key_secret[4:])
        
        credentials = f"{key_id}:{key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    def reconcile_payment(self, transaction_id):
        """
        Reconcile a single payment transaction
        
        Returns: (updated: bool, status: str, error: str)
        """
        transaction = PaymentTransaction.query.get(transaction_id)
        if not transaction:
            return False, None, "Transaction not found"
        
        if not transaction.gateway_payout_id:
            return False, transaction.status, "No gateway payout ID"
        
        try:
            headers = {
                "Authorization": self._get_razorpay_auth(),
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"https://api.razorpay.com/v1/payouts/{transaction.gateway_payout_id}",
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get('error', {}).get('description', 'Failed to fetch payout status')
                logger.error(f"Reconciliation failed for transaction {transaction_id}: {error_msg}")
                return False, transaction.status, error_msg
            
            payout_data = response.json()
            razorpay_status = payout_data.get('status', '').lower()
            old_status = transaction.status
            
            # Map Razorpay status to our status
            if razorpay_status == 'processed':
                new_status = 'success'
            elif razorpay_status == 'failed' or razorpay_status == 'reversed':
                new_status = 'failed'
            elif razorpay_status == 'queued' or razorpay_status == 'pending':
                new_status = 'processing'
            else:
                new_status = transaction.status  # Keep current status
            
            # Update if status changed
            if new_status != old_status:
                transaction.status = new_status
                transaction.gateway_response = payout_data
                transaction.processed_at = datetime.utcnow()
                
                if new_status == 'success':
                    transaction.completed_at = datetime.utcnow()
                    # Update payslip status
                    payrun_payslip = PayRunPayslip.query.filter_by(
                        pay_run_id=transaction.pay_run_id,
                        payslip_id=transaction.payslip_id
                    ).first()
                    if payrun_payslip:
                        payrun_payslip.payment_status = 'processed'
                        if payrun_payslip.payslip:
                            payrun_payslip.payslip.status = 'paid'
                
                elif new_status == 'failed':
                    transaction.failure_reason = payout_data.get('failure_reason', 'Payment failed')
                    # Update payslip status
                    payrun_payslip = PayRunPayslip.query.filter_by(
                        pay_run_id=transaction.pay_run_id,
                        payslip_id=transaction.payslip_id
                    ).first()
                    if payrun_payslip:
                        payrun_payslip.payment_status = 'failed'
                
                db.session.commit()
                
                logger.info(
                    f"Reconciled transaction {transaction_id}: {old_status} → {new_status} "
                    f"(Razorpay: {razorpay_status})"
                )
                
                return True, new_status, None
            else:
                return False, old_status, None  # No change
            
        except Exception as e:
            logger.error(f"Error reconciling transaction {transaction_id}: {str(e)}")
            return False, transaction.status, str(e)
    
    def reconcile_payrun(self, payrun_id):
        """
        Reconcile all payments in a payrun
        
        Returns: {
            'reconciled': int,
            'updated': int,
            'errors': list
        }
        """
        transactions = PaymentTransaction.query.filter_by(
            pay_run_id=payrun_id,
            status__in=['pending', 'processing']  # Only reconcile pending/processing
        ).all()
        
        results = {
            'reconciled': len(transactions),
            'updated': 0,
            'errors': []
        }
        
        for txn in transactions:
            updated, status, error = self.reconcile_payment(txn.id)
            if updated:
                results['updated'] += 1
            if error:
                results['errors'].append({
                    'transaction_id': txn.id,
                    'error': error
                })
        
        # Update payrun status based on reconciliation
        self._update_payrun_status(payrun_id)
        
        return results
    
    def reconcile_stuck_payments(self, hours_threshold=2):
        """
        Reconcile payments stuck in processing for more than threshold hours
        
        Returns: {
            'found': int,
            'reconciled': int,
            'updated': int
        }
        """
        threshold_time = datetime.utcnow() - timedelta(hours=hours_threshold)
        
        stuck_transactions = PaymentTransaction.query.filter(
            PaymentTransaction.status.in_(['pending', 'processing']),
            PaymentTransaction.initiated_at < threshold_time,
            PaymentTransaction.gateway_payout_id.isnot(None)
        ).all()
        
        results = {
            'found': len(stuck_transactions),
            'reconciled': 0,
            'updated': 0
        }
        
        for txn in stuck_transactions:
            updated, status, error = self.reconcile_payment(txn.id)
            results['reconciled'] += 1
            if updated:
                results['updated'] += 1
        
        return results
    
    def _update_payrun_status(self, payrun_id):
        """Update payrun status based on payment statuses"""
        payrun = PayRun.query.get(payrun_id)
        if not payrun:
            return
        
        transactions = PaymentTransaction.query.filter_by(pay_run_id=payrun_id).all()
        
        if not transactions:
            return
        
        success_count = sum(1 for t in transactions if t.status == 'success')
        failed_count = sum(1 for t in transactions if t.status == 'failed')
        processing_count = sum(1 for t in transactions if t.status in ['pending', 'processing'])
        total_count = len(transactions)
        
        # Update payrun statistics
        payrun.payments_successful = success_count
        payrun.payments_failed = failed_count
        payrun.payments_pending = processing_count
        
        # Update payrun status
        if payrun.status == 'payout_initiated':
            if success_count == total_count:
                payrun.status = 'completed'
                payrun.processed_at = datetime.utcnow()
            elif failed_count == total_count:
                payrun.status = 'failed'
            elif success_count > 0 or failed_count > 0:
                payrun.status = 'partially_completed'
        
        db.session.commit()

