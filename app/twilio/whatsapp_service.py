"""
Twilio WhatsApp Service
"""
from twilio.rest import Client
from app.simple_logger import get_logger
import os
import json
from datetime import datetime

logger = get_logger('twilio_whatsapp')

def get_twilio_client():
    """Get Twilio client instance"""
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    
    if not account_sid or not auth_token:
        logger.error("[Twilio WhatsApp] Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN")
        return None
    
    try:
        return Client(account_sid, auth_token)
    except Exception as e:
        logger.error(f"[Twilio WhatsApp] Failed to create Twilio client: {str(e)}")
        return None

def send_whatsapp(to_phone, message_body, from_phone=None, content_sid=None):
    """
    Send WhatsApp message via Twilio
    
    Args:
        to_phone: Recipient phone number (format: +1234567890)
        message_body: Message content (or template variables if using content_sid)
        from_phone: Sender WhatsApp number (defaults to TWILIO_WHATSAPP_NUMBER)
        content_sid: Content template SID for approved templates (optional)
    
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
        
        # Get sender WhatsApp number
        if not from_phone:
            from_phone = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')  # Default sandbox
        
        # Ensure WhatsApp format
        if not from_phone.startswith('whatsapp:'):
            from_phone = 'whatsapp:' + from_phone.lstrip('whatsapp:')
        
        # Validate phone number format
        if not to_phone.startswith('whatsapp:'):
            to_phone = 'whatsapp:' + to_phone.lstrip('whatsapp:+').lstrip('+')
        
        logger.info(f"[Twilio WhatsApp] Sending WhatsApp to {to_phone} from {from_phone}")
        
        # Prepare message parameters
        # Note: 'from' is a Python keyword, so Twilio API uses 'from_'
        message_params = {
            'from_': from_phone,
            'to': to_phone
        }
        
        # Use content template if provided, otherwise use body
        if content_sid:
            message_params['content_sid'] = content_sid
            # If using template, message_body should be JSON with variables
            if isinstance(message_body, dict):
                message_params['content_variables'] = json.dumps(message_body)
            else:
                logger.warning("[Twilio WhatsApp] content_sid provided but message_body is not a dict")
        else:
            message_params['body'] = message_body
        
        # Send message
        message = client.messages.create(**message_params)
        
        logger.info(f"[Twilio WhatsApp] WhatsApp sent successfully. SID: {message.sid}")
        
        return {
            'success': True,
            'message_sid': message.sid,
            'status': message.status,
            'error': None
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[Twilio WhatsApp] Failed to send WhatsApp: {error_msg}")
        
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
        elif 'unexpected keyword argument' in error_msg.lower():
            # This should not happen after the fix, but handle it gracefully
            return {
                'success': False,
                'error': 'Twilio API error. Please check your configuration.',
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
    Get WhatsApp message status from Twilio
    
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
        logger.error(f"[Twilio WhatsApp] Failed to get message status: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

