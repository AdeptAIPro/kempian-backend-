"""
Twilio SMS Service
"""
from twilio.rest import Client
from app.simple_logger import get_logger
import os
from datetime import datetime

logger = get_logger('twilio_sms')

def get_twilio_client():
    """Get Twilio client instance"""
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    
    if not account_sid or not auth_token:
        logger.error("[Twilio SMS] Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN")
        return None
    
    try:
        return Client(account_sid, auth_token)
    except Exception as e:
        logger.error(f"[Twilio SMS] Failed to create Twilio client: {str(e)}")
        return None

def send_sms(to_phone, message_body, from_phone=None):
    """
    Send SMS via Twilio
    
    Args:
        to_phone: Recipient phone number (format: +1234567890)
        message_body: Message content
        from_phone: Sender phone number (defaults to TWILIO_PHONE_NUMBER)
    
    Returns:
        dict with 'success', 'message_sid', 'error' keys
    """
    try:
        client = get_twilio_client()
        if not client:
            return {
                'success': False,
                'error': 'Twilio client not initialized. Check credentials.'
            }
        
        # Get sender phone number
        if not from_phone:
            from_phone = os.getenv('TWILIO_PHONE_NUMBER')
        
        if not from_phone:
            return {
                'success': False,
                'error': 'TWILIO_PHONE_NUMBER not configured'
            }
        
        # Validate phone number format
        if not to_phone.startswith('+'):
            to_phone = '+' + to_phone.lstrip('+')
        
        logger.info(f"[Twilio SMS] Sending SMS to {to_phone} from {from_phone}")
        
        # Send message
        message = client.messages.create(
            body=message_body,
            from_=from_phone,
            to=to_phone
        )
        
        logger.info(f"[Twilio SMS] SMS sent successfully. SID: {message.sid}")
        
        return {
            'success': True,
            'message_sid': message.sid,
            'status': message.status,
            'error': None
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[Twilio SMS] Failed to send SMS: {error_msg}")
        
        # Provide user-friendly error messages for common Twilio errors
        if 'unverified' in error_msg.lower() or '21608' in error_msg:
            user_friendly_error = (
                "Trial accounts can only send messages to verified phone numbers. "
                "Please verify the recipient's phone number in your Twilio account, "
                "or upgrade to a paid account to send to unverified numbers."
            )
            return {
                'success': False,
                'error': user_friendly_error,
                'technical_error': error_msg,
                'message_sid': None
            }
        elif '401' in error_msg or 'unauthorized' in error_msg.lower():
            return {
                'success': False,
                'error': 'Twilio authentication failed. Please check your account credentials.',
                'technical_error': error_msg,
                'message_sid': None
            }
        
        return {
            'success': False,
            'error': error_msg,
            'message_sid': None
        }

def get_message_status(message_sid):
    """
    Get SMS message status from Twilio
    
    Args:
        message_sid: Twilio message SID
    
    Returns:
        dict with message status information
    """
    try:
        client = get_twilio_client()
        if not client:
            return {
                'success': False,
                'error': 'Twilio client not initialized'
            }
        
        message = client.messages(message_sid).fetch()
        
        return {
            'success': True,
            'status': message.status,
            'date_sent': message.date_sent.isoformat() if message.date_sent else None,
            'error_code': message.error_code,
            'error_message': message.error_message
        }
        
    except Exception as e:
        logger.error(f"[Twilio SMS] Failed to get message status: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

