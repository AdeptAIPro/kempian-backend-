"""
Payment Data Validators
Validates bank account details, IFSC codes, account numbers, etc.
"""
import re
from app.simple_logger import get_logger

logger = get_logger(__name__)


class PaymentValidators:
    """Validators for payment-related data"""
    
    # IFSC code pattern: 4 letters + 0 + 6 digits (e.g., HDFC0001234)
    IFSC_PATTERN = re.compile(r'^[A-Z]{4}0[A-Z0-9]{6}$')
    
    # Account number: 9-18 digits for Indian banks
    ACCOUNT_NUMBER_PATTERN = re.compile(r'^\d{9,18}$')
    
    # Account holder name: letters, spaces, dots, hyphens (min 2 chars, max 100)
    ACCOUNT_HOLDER_NAME_PATTERN = re.compile(r'^[a-zA-Z\s\.\-]{2,100}$')
    
    @staticmethod
    def validate_ifsc_code(ifsc_code):
        """
        Validate IFSC code format
        
        Args:
            ifsc_code: IFSC code to validate
        
        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        if not ifsc_code:
            return False, "IFSC code is required"
        
        ifsc_code = ifsc_code.strip().upper()
        
        if len(ifsc_code) != 11:
            return False, "IFSC code must be exactly 11 characters"
        
        if not PaymentValidators.IFSC_PATTERN.match(ifsc_code):
            return False, "Invalid IFSC code format. Format: AAAA0XXXXXX (4 letters, 0, 6 alphanumeric)"
        
        return True, None
    
    @staticmethod
    def validate_account_number(account_number):
        """
        Validate bank account number format
        
        Args:
            account_number: Account number to validate
        
        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        if not account_number:
            return False, "Account number is required"
        
        account_number = account_number.strip().replace(' ', '').replace('-', '')
        
        if not account_number.isdigit():
            return False, "Account number must contain only digits"
        
        if len(account_number) < 9:
            return False, "Account number must be at least 9 digits"
        
        if len(account_number) > 18:
            return False, "Account number must not exceed 18 digits"
        
        if not PaymentValidators.ACCOUNT_NUMBER_PATTERN.match(account_number):
            return False, "Invalid account number format"
        
        return True, None
    
    @staticmethod
    def validate_account_holder_name(name):
        """
        Validate account holder name
        
        Args:
            name: Account holder name to validate
        
        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        if not name:
            return False, "Account holder name is required"
        
        name = name.strip()
        
        if len(name) < 2:
            return False, "Account holder name must be at least 2 characters"
        
        if len(name) > 100:
            return False, "Account holder name must not exceed 100 characters"
        
        if not PaymentValidators.ACCOUNT_HOLDER_NAME_PATTERN.match(name):
            return False, "Account holder name contains invalid characters. Only letters, spaces, dots, and hyphens are allowed"
        
        return True, None
    
    @staticmethod
    def validate_payment_amount(amount, currency='INR'):
        """
        Validate payment amount
        
        Args:
            amount: Payment amount
            currency: Currency code
        
        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        try:
            amount_float = float(amount)
        except (ValueError, TypeError):
            return False, "Invalid amount format"
        
        if amount_float <= 0:
            return False, "Payment amount must be greater than zero"
        
        # Minimum amount: ₹1
        if currency == 'INR' and amount_float < 1:
            return False, "Minimum payment amount is ₹1"
        
        # Maximum amount: ₹1 crore (10,000,000)
        if currency == 'INR' and amount_float > 10000000:
            return False, "Maximum payment amount is ₹1,00,00,000 (1 crore)"
        
        return True, None
    
    @staticmethod
    def validate_bank_account_details(account_number, ifsc_code, account_holder_name):
        """
        Validate all bank account details at once
        
        Args:
            account_number: Bank account number
            ifsc_code: IFSC code
            account_holder_name: Account holder name
        
        Returns:
            dict: {
                'valid': bool,
                'errors': list of error messages
            }
        """
        errors = []
        
        # Validate IFSC
        ifsc_valid, ifsc_error = PaymentValidators.validate_ifsc_code(ifsc_code)
        if not ifsc_valid:
            errors.append(ifsc_error)
        
        # Validate account number
        acc_valid, acc_error = PaymentValidators.validate_account_number(account_number)
        if not acc_valid:
            errors.append(acc_error)
        
        # Validate account holder name
        name_valid, name_error = PaymentValidators.validate_account_holder_name(account_holder_name)
        if not name_valid:
            errors.append(name_error)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

