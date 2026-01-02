"""
Cron Job: Reconcile Payment Status
Runs every 15-30 minutes to reconcile payment status with Razorpay
Handles webhook failures and stuck payments
"""
from app import create_app
from app.models import db, PaymentTransaction, PayrollSettings
from app.services.reconciliation_service import ReconciliationService
from app.simple_logger import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)


def reconcile_all_tenants():
    """Reconcile payments for all tenants with Razorpay configured"""
    app = create_app()
    
    with app.app_context():
        # Get all tenants with Razorpay configured
        settings = PayrollSettings.query.filter(
            PayrollSettings.payment_gateway == 'razorpay',
            PayrollSettings.razorpay_key_id.isnot(None)
        ).all()
        
        total_reconciled = 0
        total_updated = 0
        
        for setting in settings:
            try:
                reconciliation = ReconciliationService(tenant_id=setting.tenant_id)
                
                # Reconcile stuck payments (stuck > 2 hours)
                results = reconciliation.reconcile_stuck_payments(hours_threshold=2)
                
                total_reconciled += results['reconciled']
                total_updated += results['updated']
                
                if results['updated'] > 0:
                    logger.info(
                        f"Tenant {setting.tenant_id}: Reconciled {results['reconciled']} payments, "
                        f"updated {results['updated']}"
                    )
                    
            except Exception as e:
                logger.error(f"Error reconciling tenant {setting.tenant_id}: {str(e)}")
                continue
        
        logger.info(
            f"Reconciliation complete: {total_reconciled} payments checked, "
            f"{total_updated} statuses updated"
        )


if __name__ == '__main__':
    reconcile_all_tenants()

