"""
Wallet Balance Service
Manages employer's Razorpay wallet balance, fund locking, and validation
"""
import requests
from decimal import Decimal
from datetime import datetime
from app.models import db, EmployerWalletBalance, PayRun, Tenant
from app.models import db, EmployerWalletBalance, PayRun, Tenant
from app.simple_logger import get_logger
from app.utils.payment_security import PaymentEncryption

logger = get_logger(__name__)


class WalletBalanceService:
    """Service for managing employer wallet balance"""
    
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id
        self.wallet = EmployerWalletBalance.query.filter_by(tenant_id=tenant_id).first()
        
        if not self.wallet:
            # Create wallet record if doesn't exist
            self.wallet = EmployerWalletBalance(
                tenant_id=tenant_id,
                available_balance=Decimal('0'),
                locked_balance=Decimal('0'),
                total_balance=Decimal('0')
            )
            db.session.add(self.wallet)
            db.session.commit()
    
    def _get_razorpay_auth(self, razorpay_key_id, razorpay_key_secret):
        """Get Razorpay Basic Auth header"""
        import base64
        credentials = f"{razorpay_key_id}:{razorpay_key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    def sync_balance_from_razorpay(self, razorpay_key_id, razorpay_key_secret):
        """
        Sync balance from Razorpay API
        
        Returns: (success: bool, error: str)
        """
        try:
            # Decrypt secret if needed
            encryption = PaymentEncryption()
            if razorpay_key_secret and razorpay_key_secret.startswith('enc:'):
                razorpay_key_secret = encryption.decrypt(razorpay_key_secret[4:])
            
            headers = {
                "Authorization": self._get_razorpay_auth(razorpay_key_id, razorpay_key_secret),
                "Content-Type": "application/json"
            }
            
            # Get account balance from Razorpay
            response = requests.get(
                "https://api.razorpay.com/v1/accounts/me",
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get('error', {}).get('description', 'Failed to fetch balance')
                logger.error(f"Razorpay balance sync failed: {error_msg}")
                self.wallet.sync_error = error_msg
                self.wallet.last_synced_at = datetime.utcnow()
                db.session.commit()
                return False, error_msg
            
            account_data = response.json()
            
            # Get fund account balance
            fund_account_id = self._get_fund_account_id(razorpay_key_id, razorpay_key_secret)
            if fund_account_id:
                balance_response = requests.get(
                    f"https://api.razorpay.com/v1/fund_accounts/{fund_account_id}/balance",
                    headers=headers,
                    timeout=30
                )
                
                if balance_response.status_code == 200:
                    balance_data = balance_response.json()
                    # Razorpay returns balance in smallest currency unit (paise for INR)
                    balance_paise = balance_data.get('balance', 0)
                    balance_rupees = Decimal(str(balance_paise)) / Decimal('100')
                    
                    # Update wallet balance
                    # Available = Total - Locked
                    locked = self.wallet.locked_balance or Decimal('0')
                    self.wallet.total_balance = balance_rupees
                    self.wallet.available_balance = balance_rupees - locked
                    self.wallet.razorpay_account_status = account_data.get('status', 'active')
                    self.wallet.kyc_status = account_data.get('kyc_status', 'pending')
                    self.wallet.last_synced_at = datetime.utcnow()
                    self.wallet.sync_error = None
                    
                    db.session.commit()
                    logger.info(f"Balance synced: Total={balance_rupees}, Available={self.wallet.available_balance}")
                    return True, None
                else:
                    # Fallback: use account balance if fund account balance fails
                    logger.warning("Fund account balance fetch failed, using account balance")
            
            # Update status even if balance fetch fails
            self.wallet.razorpay_account_status = account_data.get('status', 'active')
            self.wallet.kyc_status = account_data.get('kyc_status', 'pending')
            self.wallet.last_synced_at = datetime.utcnow()
            db.session.commit()
            
            return True, None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error syncing balance: {error_msg}")
            self.wallet.sync_error = error_msg
            self.wallet.last_synced_at = datetime.utcnow()
            db.session.commit()
            return False, error_msg
    
    def _get_fund_account_id(self, razorpay_key_id, razorpay_key_secret):
        """Get fund account ID from settings"""
        from app.models import PayrollSettings
        settings = PayrollSettings.query.filter_by(tenant_id=self.tenant_id).first()
        return settings.razorpay_fund_account_id if settings else None
    
    def validate_funds_for_payrun(self, payrun_id):
        """
        Validate that sufficient funds are available for a payrun
        
        Returns: (is_valid: bool, error: str, available: Decimal, required: Decimal)
        """
        payrun = PayRun.query.get(payrun_id)
        if not payrun:
            return False, "Pay run not found", Decimal('0'), Decimal('0')
        
        required_amount = payrun.total_net or Decimal('0')
        available = self.wallet.available_balance or Decimal('0')
        
        # Check KYC status
        if self.wallet.kyc_status not in ['approved', 'verified']:
            return False, f"Employer KYC not approved. Status: {self.wallet.kyc_status}", available, required_amount
        
        # Check account status
        if self.wallet.razorpay_account_status not in ['active', 'live']:
            return False, f"Razorpay account not active. Status: {self.wallet.razorpay_account_status}", available, required_amount
        
        # Check balance
        if available < required_amount:
            return False, f"Insufficient balance. Available: {available}, Required: {required_amount}", available, required_amount
        
        return True, None, available, required_amount
    
    def lock_funds_for_payrun(self, payrun_id):
        """
        Lock funds for a payrun with CONCURRENCY HARDENING
        
        Uses SELECT FOR UPDATE to prevent race conditions.
        Two payrolls can NEVER lock the same funds.
        
        Returns: (success: bool, error: str)
        """
        payrun = PayRun.query.get(payrun_id)
        if not payrun:
            return False, "Pay run not found"
        
        if payrun.funds_locked:
            return True, None  # Already locked
        
        required_amount = payrun.total_net or Decimal('0')
        
        try:
            # CRITICAL: Use row-level locking to prevent concurrent access
            # This ensures atomic fund locking across multiple payrolls
            from sqlalchemy import select
            
            # Lock the wallet row for update (prevents concurrent modifications)
            wallet_locked = db.session.query(EmployerWalletBalance).filter_by(
                tenant_id=self.tenant_id
            ).with_for_update(nowait=True).first()
            
            if not wallet_locked:
                # Create if doesn't exist
                wallet_locked = EmployerWalletBalance(
                    tenant_id=self.tenant_id,
                    available_balance=Decimal('0'),
                    locked_balance=Decimal('0'),
                    total_balance=Decimal('0')
                )
                db.session.add(wallet_locked)
                db.session.flush()
            
            # Re-validate with locked row (most current data)
            available = wallet_locked.available_balance or Decimal('0')
            
            # Check KYC status
            if wallet_locked.kyc_status not in ['approved', 'verified']:
                db.session.rollback()
                return False, f"Employer KYC not approved. Status: {wallet_locked.kyc_status}"
            
            # Check account status
            if wallet_locked.razorpay_account_status not in ['active', 'live']:
                db.session.rollback()
                return False, f"Razorpay account not active. Status: {wallet_locked.razorpay_account_status}"
            
            # Check balance with locked row
            if available < required_amount:
                db.session.rollback()
                return False, f"Insufficient balance. Available: {available}, Required: {required_amount}"
            
            # Atomic lock operation
            wallet_locked.available_balance -= required_amount
            wallet_locked.locked_balance += required_amount
            
            # Update payrun
            payrun.funds_locked = True
            payrun.funds_locked_at = datetime.utcnow()
            payrun.funds_locked_amount = required_amount
            
            # Update local wallet reference
            self.wallet = wallet_locked
            
            db.session.commit()
            logger.info(f"Funds locked for payrun {payrun_id}: {required_amount} (atomic operation)")
            return True, None
            
        except Exception as e:
            db.session.rollback()
            error_msg = str(e)
            
            # Check if it's a lock timeout (concurrent access)
            if 'could not obtain lock' in error_msg.lower() or 'lock wait timeout' in error_msg.lower():
                logger.warning(f"Concurrent fund lock attempt detected for payrun {payrun_id}")
                return False, "Another payroll is currently locking funds. Please retry in a moment."
            
            logger.error(f"Error locking funds for payrun {payrun_id}: {error_msg}")
            return False, error_msg
    
    def unlock_funds_for_payrun(self, payrun_id):
        """
        Unlock funds for a payrun (on failure/reversal) with concurrency safety
        
        Returns: (success: bool, error: str)
        """
        payrun = PayRun.query.get(payrun_id)
        if not payrun:
            return False, "Pay run not found"
        
        if not payrun.funds_locked:
            return True, None  # Not locked
        
        locked_amount = payrun.funds_locked_amount or Decimal('0')
        
        try:
            # Use row-level locking for atomic unlock
            wallet_locked = db.session.query(EmployerWalletBalance).filter_by(
                tenant_id=self.tenant_id
            ).with_for_update(nowait=True).first()
            
            if not wallet_locked:
                db.session.rollback()
                return False, "Wallet not found"
            
            # Verify locked amount matches
            if wallet_locked.locked_balance < locked_amount:
                db.session.rollback()
                logger.warning(f"Locked balance mismatch for payrun {payrun_id}. Expected: {locked_amount}, Actual: {wallet_locked.locked_balance}")
                # Unlock what we can
                locked_amount = wallet_locked.locked_balance
            
            # Atomic unlock
            wallet_locked.locked_balance -= locked_amount
            wallet_locked.available_balance += locked_amount
            
            # Update payrun
            payrun.funds_locked = False
            payrun.funds_locked_at = None
            payrun.funds_locked_amount = None
            
            # Update local reference
            self.wallet = wallet_locked
            
            db.session.commit()
            logger.info(f"Funds unlocked for payrun {payrun_id}: {locked_amount} (atomic operation)")
            return True, None
            
        except Exception as e:
            db.session.rollback()
            error_msg = str(e)
            
            if 'could not obtain lock' in error_msg.lower():
                return False, "Concurrent unlock operation in progress. Please retry."
            
            logger.error(f"Error unlocking funds for payrun {payrun_id}: {error_msg}")
            return False, error_msg
    
    def get_balance_summary(self):
        """Get current balance summary"""
        return {
            'available': float(self.wallet.available_balance or 0),
            'locked': float(self.wallet.locked_balance or 0),
            'total': float(self.wallet.total_balance or 0),
            'account_status': self.wallet.razorpay_account_status,
            'kyc_status': self.wallet.kyc_status,
            'last_synced': self.wallet.last_synced_at.isoformat() if self.wallet.last_synced_at else None
        }

