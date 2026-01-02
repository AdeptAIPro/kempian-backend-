"""
Razorpay Error Code Mapper
Maps Razorpay error codes to user-friendly messages
"""
from app.simple_logger import get_logger

logger = get_logger(__name__)


class RazorpayErrorMapper:
    """Maps Razorpay API errors to user-friendly messages"""
    
    # Error code mappings
    ERROR_MESSAGES = {
        # Authentication errors
        'BAD_REQUEST_ERROR': 'Invalid request. Please check your Razorpay configuration.',
        'GATEWAY_ERROR': 'Payment gateway error. Please try again later.',
        'SERVER_ERROR': 'Razorpay server error. Please try again later.',
        
        # Fund account errors
        'INVALID_FUND_ACCOUNT': 'Invalid fund account. Please verify your fund account ID.',
        'FUND_ACCOUNT_NOT_FOUND': 'Fund account not found. Please check your fund account ID.',
        'FUND_ACCOUNT_INACTIVE': 'Fund account is inactive. Please activate it in Razorpay dashboard.',
        
        # Balance errors
        'INSUFFICIENT_BALANCE': 'Insufficient balance in Razorpay wallet. Please add funds.',
        'LOW_BALANCE': 'Low balance in Razorpay wallet. Please add funds.',
        
        # Bank account errors
        'INVALID_BANK_ACCOUNT': 'Invalid bank account details. Please verify account number and IFSC code.',
        'INVALID_IFSC': 'Invalid IFSC code. Please check the IFSC code.',
        'ACCOUNT_NOT_FOUND': 'Bank account not found. Please verify account details.',
        'ACCOUNT_CLOSED': 'Bank account is closed. Please use a different account.',
        'ACCOUNT_FROZEN': 'Bank account is frozen. Please contact your bank.',
        
        # KYC errors
        'KYC_NOT_APPROVED': 'KYC verification not completed. Please complete KYC in Razorpay dashboard.',
        'KYC_PENDING': 'KYC verification is pending. Please complete KYC in Razorpay dashboard.',
        'KYC_REJECTED': 'KYC verification was rejected. Please contact Razorpay support.',
        
        # Payment errors
        'PAYMENT_FAILED': 'Payment failed. Please check bank account details and try again.',
        'PAYMENT_REJECTED': 'Payment was rejected by the bank. Please contact your bank.',
        'PAYMENT_TIMEOUT': 'Payment request timed out. Please try again.',
        'PAYMENT_CANCELLED': 'Payment was cancelled.',
        
        # Rate limit errors
        'RATE_LIMIT_EXCEEDED': 'Too many requests. Please wait a moment and try again.',
        
        # Generic errors
        'UNKNOWN_ERROR': 'An unknown error occurred. Please contact support.',
    }
    
    # Error description patterns (for partial matching)
    ERROR_PATTERNS = {
        'insufficient': 'INSUFFICIENT_BALANCE',
        'low balance': 'LOW_BALANCE',
        'invalid ifsc': 'INVALID_IFSC',
        'invalid account': 'INVALID_BANK_ACCOUNT',
        'account not found': 'ACCOUNT_NOT_FOUND',
        'kyc': 'KYC_NOT_APPROVED',
        'fund account': 'INVALID_FUND_ACCOUNT',
        'timeout': 'PAYMENT_TIMEOUT',
        'rate limit': 'RATE_LIMIT_EXCEEDED',
    }
    
    @classmethod
    def get_user_friendly_message(cls, error_code=None, error_description=None, error_data=None):
        """
        Get user-friendly error message from Razorpay error
        
        Args:
            error_code: Razorpay error code
            error_description: Error description from Razorpay
            error_data: Full error data dict
        
        Returns:
            str: User-friendly error message
        """
        # Try to get error code from error_data if not provided
        if error_data and isinstance(error_data, dict):
            if not error_code:
                error_code = error_data.get('error', {}).get('code')
            if not error_description:
                error_description = error_data.get('error', {}).get('description')
        
        # First, try exact match with error code
        if error_code and error_code in cls.ERROR_MESSAGES:
            return cls.ERROR_MESSAGES[error_code]
        
        # Then, try pattern matching with description
        if error_description:
            error_desc_lower = error_description.lower()
            for pattern, mapped_code in cls.ERROR_PATTERNS.items():
                if pattern in error_desc_lower:
                    return cls.ERROR_MESSAGES.get(mapped_code, error_description)
        
        # If no match, return the original description or a generic message
        if error_description:
            return error_description
        
        return cls.ERROR_MESSAGES.get('UNKNOWN_ERROR', 'An error occurred. Please try again.')
    
    @classmethod
    def map_razorpay_exception(cls, exception):
        """
        Map a Razorpay exception to user-friendly message
        
        Args:
            exception: Exception object or error dict
        
        Returns:
            str: User-friendly error message
        """
        if isinstance(exception, dict):
            return cls.get_user_friendly_message(error_data=exception)
        elif hasattr(exception, 'error'):
            # Razorpay SDK exception
            error_data = exception.error if hasattr(exception, 'error') else {}
            return cls.get_user_friendly_message(error_data=error_data)
        else:
            error_str = str(exception)
            # Try to extract error code/description from string
            return cls.get_user_friendly_message(error_description=error_str)

