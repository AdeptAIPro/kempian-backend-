"""
Payment Security Utilities
Handles encryption, decryption, and security for payment data
"""
import os
import base64
import hashlib
import hmac
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from app.simple_logger import get_logger

logger = get_logger(__name__)


class PaymentEncryption:
    """Encryption utility for sensitive payment data"""
    
    def __init__(self):
        # Get encryption key from environment or use default (for development only)
        encryption_key = os.environ.get('PAYMENT_ENCRYPTION_KEY')
        if not encryption_key:
            # Generate a key from SECRET_KEY - require it in production
            secret_key = os.environ.get('SECRET_KEY')
            if not secret_key:
                # Only allow default in development mode
                if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('ENV') == 'production':
                    raise ValueError("SECRET_KEY environment variable is required in production for payment encryption")
                logger.warning("Using development secret key - NOT SAFE FOR PRODUCTION")
                secret_key = 'dev-secret-key-change-in-production'
            
            # Get salt from environment or generate a consistent one from secret_key
            encryption_salt = os.environ.get('PAYMENT_ENCRYPTION_SALT')
            if encryption_salt:
                salt = encryption_salt.encode()
            else:
                # Generate a deterministic salt from secret key hash (better than hardcoded)
                # In production, PAYMENT_ENCRYPTION_SALT should be set
                if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('ENV') == 'production':
                    logger.warning("PAYMENT_ENCRYPTION_SALT not set - using derived salt. Set this for better security.")
                salt = hashlib.sha256(f"{secret_key}_payment_salt".encode()).digest()[:16]
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))
        else:
            key = encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
        
        self.cipher = Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt sensitive data"""
        try:
            if not plaintext:
                return ""
            encrypted = self.cipher.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            raise
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt sensitive data"""
        try:
            if not ciphertext:
                return ""
            decoded = base64.urlsafe_b64decode(ciphertext.encode())
            decrypted = self.cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            raise


class DataMasking:
    """Utility for masking sensitive data in logs and responses"""
    
    @staticmethod
    def mask_account_number(account_number: str) -> str:
        """Mask account number: show only last 4 digits"""
        if not account_number or len(account_number) < 4:
            return "****"
        return "****" + account_number[-4:]
    
    @staticmethod
    def mask_ifsc(ifsc: str) -> str:
        """Mask IFSC: show only first 2 and last 2 characters"""
        if not ifsc or len(ifsc) < 4:
            return "****"
        return ifsc[:2] + "***" + ifsc[-2:]
    
    @staticmethod
    def mask_api_key(key: str) -> str:
        """Mask API key: show only first 8 characters"""
        if not key or len(key) < 8:
            return "****"
        return key[:8] + "****"
    
    @staticmethod
    def mask_phone(phone: str) -> str:
        """Mask phone number: show only last 4 digits"""
        if not phone or len(phone) < 4:
            return "****"
        return "****" + phone[-4:]


class WebhookVerification:
    """Verify webhook signatures from payment gateways"""
    
    @staticmethod
    def verify_razorpay_webhook(payload: str, signature: str, secret: str) -> bool:
        """
        Verify Razorpay webhook signature
        Razorpay uses HMAC SHA256 with the webhook secret
        """
        try:
            expected_signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(expected_signature, signature)
        except Exception as e:
            logger.error(f"Webhook verification error: {str(e)}")
            return False
    
    @staticmethod
    def verify_razorpay_webhook_v2(payload: bytes, signature: str, secret: str) -> bool:
        """
        Verify Razorpay webhook signature (v2 format)
        For newer Razorpay webhooks
        """
        try:
            expected_signature = hmac.new(
                secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(expected_signature, signature)
        except Exception as e:
            logger.error(f"Webhook verification error: {str(e)}")
            return False


class PaymentAuditLogger:
    """Audit logging for payment operations"""
    
    @staticmethod
    def log_payment_initiated(transaction_id: int, employee_id: int, amount: float, currency: str, user_id: int):
        """Log payment initiation"""
        logger.info(
            f"PAYMENT_INITIATED: transaction_id={transaction_id}, "
            f"employee_id={employee_id}, amount={amount} {currency}, "
            f"initiated_by={user_id}"
        )
    
    @staticmethod
    def log_payment_success(transaction_id: int, gateway_transaction_id: str, amount: float):
        """Log successful payment"""
        logger.info(
            f"PAYMENT_SUCCESS: transaction_id={transaction_id}, "
            f"gateway_id={gateway_transaction_id}, amount={amount}"
        )
    
    @staticmethod
    def log_payment_failure(transaction_id: int, reason: str, gateway_response: dict = None):
        """Log payment failure"""
        logger.error(
            f"PAYMENT_FAILURE: transaction_id={transaction_id}, "
            f"reason={reason}, gateway_response={gateway_response}"
        )
    
    @staticmethod
    def log_payment_retry(transaction_id: int, original_transaction_id: int, user_id: int):
        """Log payment retry"""
        logger.warning(
            f"PAYMENT_RETRY: transaction_id={transaction_id}, "
            f"original_transaction_id={original_transaction_id}, "
            f"retried_by={user_id}"
        )
    
    @staticmethod
    def log_security_event(event_type: str, details: dict, user_id: int = None):
        """Log security-related events"""
        logger.warning(
            f"SECURITY_EVENT: type={event_type}, "
            f"user_id={user_id}, details={details}"
        )


class FraudDetection:
    """Basic fraud detection for payment transactions"""
    
    @staticmethod
    def check_amount_threshold(amount: float, currency: str = 'INR') -> dict:
        """
        Check if amount exceeds suspicious thresholds
        Returns: {'suspicious': bool, 'reason': str}
        """
        # Define thresholds (adjust based on business rules)
        thresholds = {
            'INR': {
                'warning': 100000,  # 1 lakh
                'critical': 500000  # 5 lakhs
            },
            'USD': {
                'warning': 10000,
                'critical': 50000
            }
        }
        
        threshold = thresholds.get(currency, thresholds['INR'])
        
        if amount > threshold['critical']:
            return {
                'suspicious': True,
                'reason': f'Amount {amount} {currency} exceeds critical threshold',
                'severity': 'critical'
            }
        elif amount > threshold['warning']:
            return {
                'suspicious': True,
                'reason': f'Amount {amount} {currency} exceeds warning threshold',
                'severity': 'warning'
            }
        
        return {'suspicious': False}
    
    @staticmethod
    def check_velocity(employee_id: int, recent_payments: list) -> dict:
        """
        Check payment velocity (too many payments in short time)
        Returns: {'suspicious': bool, 'reason': str}
        """
        if len(recent_payments) > 5:  # More than 5 payments in recent period
            return {
                'suspicious': True,
                'reason': f'High payment velocity for employee {employee_id}',
                'severity': 'warning'
            }
        
        return {'suspicious': False}
    
    @staticmethod
    def validate_bank_details(account_number: str, ifsc_code: str) -> dict:
        """
        Basic validation of bank details
        Returns: {'valid': bool, 'errors': list}
        """
        errors = []
        
        # Validate IFSC format (11 characters, alphanumeric)
        if not ifsc_code or len(ifsc_code) != 11:
            errors.append("IFSC code must be exactly 11 characters")
        elif not ifsc_code[:4].isalpha():
            errors.append("IFSC code must start with 4 letters")
        elif not ifsc_code[4:].isalnum():
            errors.append("IFSC code format invalid")
        
        # Validate account number (9-18 digits typically)
        if not account_number or len(account_number) < 9 or len(account_number) > 18:
            errors.append("Account number must be between 9 and 18 digits")
        elif not account_number.isdigit():
            errors.append("Account number must contain only digits")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

