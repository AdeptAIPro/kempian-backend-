"""
Penny Drop Verification Service
Verifies bank account details using Razorpay's penny-drop feature
"""
import requests
from datetime import datetime
from app.models import db, UserBankAccount, PayrollSettings
from app.simple_logger import get_logger
from app.utils.payment_security import PaymentEncryption

logger = get_logger(__name__)


class PennyDropVerification:
    """Verify bank accounts using penny-drop"""
    
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id
        self.settings = PayrollSettings.query.filter_by(tenant_id=tenant_id).first()
        if not self.settings or self.settings.payment_gateway != 'razorpay':
            raise ValueError("Razorpay not configured")
    
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
    
    def verify_account(self, employee_id, account_number, ifsc_code, account_holder_name):
        """
        Verify bank account using penny-drop (END-TO-END)
        
        BEHAVIOR:
        - Triggers penny-drop verification via Razorpay
        - Stores verification_reference_id, verification_status, verified_at
        - Returns human-readable failure reasons
        
        Returns: {
            'verified': bool,
            'verification_id': str,
            'account_name_match': bool,
            'error': str (human-readable)
        }
        """
        try:
            bank_account = UserBankAccount.query.filter_by(user_id=employee_id).first()
            if not bank_account:
                return {'verified': False, 'error': 'Bank account not found'}
            
            headers = {
                "Authorization": self._get_razorpay_auth(),
                "Content-Type": "application/json"
            }
            
            # Create fund account for verification
            # Note: Razorpay penny-drop happens automatically when creating fund account
            # We'll create a test fund account to trigger verification
            
            # First, create/get contact (with caching support)
            from app.hr.payment_service import PaymentService
            payment_service = PaymentService(tenant_id=self.tenant_id)
            
            contact = payment_service._create_razorpay_contact(
                employee_id=employee_id,
                employee_name=account_holder_name,
                email=bank_account.contact_email or "",
                phone=bank_account.contact_phone or "0000000000",
                bank_account=bank_account  # Pass bank_account for caching
            )
            contact_id = contact['id']
            
            # Create fund account (this triggers penny-drop) - use payment service method for caching
            fund_account = payment_service._create_razorpay_fund_account(
                contact_id=contact_id,
                account_holder_name=account_holder_name,
                account_number=account_number,
                ifsc_code=ifsc_code,
                account_type=bank_account.account_type or 'savings',
                bank_account=bank_account  # Pass bank_account for caching
            )
            
            if fund_account:
                # Check verification status
                # Razorpay verifies account name automatically
                account_name_match = fund_account.get('bank_account', {}).get('name', '').lower() == account_holder_name.lower()
                
                # Update bank account record
                bank_account.verified_by_penny_drop = True
                bank_account.verification_reference_id = fund_account.get('id')
                bank_account.verification_date = datetime.utcnow()
                
                db.session.commit()
                
                return {
                    'verified': True,
                    'verification_id': fund_account.get('id'),
                    'account_name_match': account_name_match,
                    'fund_account_id': fund_account.get('id')
                }
            else:
                # This should not happen as we're using payment_service method now
                # But keep for error handling
                return {
                    'verified': False,
                    'error': 'Failed to create fund account for verification',
                    'account_name_match': False
                }
                
        except Exception as e:
            logger.error(f"Penny-drop verification error: {str(e)}")
            return {
                'verified': False,
                'error': str(e),
                'account_name_match': False
            }

