"""
Cron Job: Sync Wallet Balance
Runs periodically to sync employer wallet balance from Razorpay
"""
from app import create_app
from app.models import db, PayrollSettings
from app.services.wallet_balance_service import WalletBalanceService
from app.simple_logger import get_logger

logger = get_logger(__name__)


def sync_all_wallets():
    """Sync wallet balance for all tenants with Razorpay configured"""
    app = create_app()
    
    with app.app_context():
        # Get all tenants with Razorpay configured
        settings = PayrollSettings.query.filter(
            PayrollSettings.payment_gateway == 'razorpay',
            PayrollSettings.razorpay_key_id.isnot(None),
            PayrollSettings.razorpay_key_secret.isnot(None)
        ).all()
        
        synced_count = 0
        failed_count = 0
        
        for setting in settings:
            try:
                wallet_service = WalletBalanceService(tenant_id=setting.tenant_id)
                
                success, error = wallet_service.sync_balance_from_razorpay(
                    setting.razorpay_key_id,
                    setting.razorpay_key_secret
                )
                
                if success:
                    synced_count += 1
                    logger.info(f"Synced balance for tenant {setting.tenant_id}")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to sync balance for tenant {setting.tenant_id}: {error}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error syncing wallet for tenant {setting.tenant_id}: {str(e)}")
                continue
        
        logger.info(
            f"Wallet sync complete: {synced_count} synced, {failed_count} failed"
        )


if __name__ == '__main__':
    sync_all_wallets()

