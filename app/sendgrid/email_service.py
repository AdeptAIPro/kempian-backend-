"""
SendGrid Email Service
"""
from sendgrid import SendGridAPIClient, SendGridException
from sendgrid.helpers.mail import Mail, Email, Content
from app.simple_logger import get_logger
import os
from datetime import datetime

logger = get_logger('sendgrid_email')

def get_sendgrid_client():
    """Get SendGrid client instance"""
    api_key = os.getenv('SENDGRID_API_KEY')
    
    if not api_key:
        logger.error("[SendGrid] Missing SENDGRID_API_KEY")
        return None
    
    try:
        return SendGridAPIClient(api_key)
    except Exception as e:
        logger.error(f"[SendGrid] Failed to create SendGrid client: {str(e)}")
        return None

def send_email(to_email, subject, html_content, text_content=None, from_email=None, from_name=None, reply_to=None):
    """
    Send email via SendGrid
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        html_content: HTML email body
        text_content: Plain text email body (optional)
        from_email: Sender email (defaults to SENDGRID_FROM_EMAIL)
        from_name: Sender name (defaults to SENDGRID_FROM_NAME)
        reply_to: Reply-to email address (optional)
    
    Returns:
        dict with 'success', 'message_id', 'error' keys
    """
    try:
        sg = get_sendgrid_client()
        if not sg:
            return {
                'success': False,
                'error': 'SendGrid client not initialized. Check API key.'
            }
        
        # Get sender information
        if not from_email:
            from_email = os.getenv('SENDGRID_FROM_EMAIL')
        
        if not from_name:
            from_name = os.getenv('SENDGRID_FROM_NAME', 'Kempian AI')
        
        if not from_email:
            return {
                'success': False,
                'error': 'SENDGRID_FROM_EMAIL not configured'
            }
        
        logger.info(f"[SendGrid] Sending email to {to_email} from {from_email}")
        
        # Create email message
        message = Mail(
            from_email=Email(from_email, from_name),
            to_emails=to_email,
            subject=subject,
            html_content=html_content
        )
        
        # Add plain text content if provided
        if text_content:
            message.content = Content("text/plain", text_content)
        
        # Add reply-to if provided
        if reply_to:
            message.reply_to = Email(reply_to)
        
        # Send email
        response = sg.send(message)
        
        # Extract message ID from response headers
        message_id = None
        if hasattr(response, 'headers') and 'X-Message-Id' in response.headers:
            message_id = response.headers['X-Message-Id']
        
        logger.info(f"[SendGrid] Email sent successfully. Status: {response.status_code}, Message ID: {message_id}")
        
        return {
            'success': True,
            'status_code': response.status_code,
            'message_id': message_id,
            'error': None
        }
        
    except SendGridException as e:
        error_msg = str(e)
        status_code = getattr(e, 'status_code', None) if hasattr(e, 'status_code') else None
        logger.error(f"[SendGrid] Failed to send email: {error_msg} (Status: {status_code})")
        
        # Provide user-friendly error messages for common SendGrid errors
        if status_code == 401 or '401' in error_msg or 'unauthorized' in error_msg.lower():
            return {
                'success': False,
                'error': 'SendGrid authentication failed. Please check your API key.',
                'technical_error': error_msg,
                'message_id': None
            }
        elif status_code == 403 or '403' in error_msg or 'forbidden' in error_msg.lower():
            return {
                'success': False,
                'error': 'SendGrid access denied. Please check your account permissions.',
                'technical_error': error_msg,
                'message_id': None
            }
        elif 'api key' in error_msg.lower() or 'api_key' in error_msg.lower():
            return {
                'success': False,
                'error': 'SendGrid API key is missing or invalid. Please configure SENDGRID_API_KEY.',
                'technical_error': error_msg,
                'message_id': None
            }
        
        return {
            'success': False,
            'error': error_msg,
            'message_id': None
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[SendGrid] Failed to send email: {error_msg}")
        
        # Check for HTTP errors in the error message
        if '401' in error_msg or 'unauthorized' in error_msg.lower():
            return {
                'success': False,
                'error': 'SendGrid authentication failed. Please check your API key.',
                'technical_error': error_msg,
                'message_id': None
            }
        elif '403' in error_msg or 'forbidden' in error_msg.lower():
            return {
                'success': False,
                'error': 'SendGrid access denied. Please check your account permissions.',
                'technical_error': error_msg,
                'message_id': None
            }
        
        return {
            'success': False,
            'error': error_msg,
            'message_id': None
        }

def send_email_with_template(to_email, template_id, template_data, from_email=None, from_name=None):
    """
    Send email using SendGrid dynamic template
    
    Args:
        to_email: Recipient email address
        template_id: SendGrid template ID
        template_data: Dictionary of template variables
        from_email: Sender email (defaults to SENDGRID_FROM_EMAIL)
        from_name: Sender name (defaults to SENDGRID_FROM_NAME)
    
    Returns:
        dict with 'success', 'message_id', 'error' keys
    """
    try:
        sg = get_sendgrid_client()
        if not sg:
            return {
                'success': False,
                'error': 'SendGrid client not initialized. Check API key.'
            }
        
        # Get sender information
        if not from_email:
            from_email = os.getenv('SENDGRID_FROM_EMAIL')
        
        if not from_name:
            from_name = os.getenv('SENDGRID_FROM_NAME', 'Kempian AI')
        
        if not from_email:
            return {
                'success': False,
                'error': 'SENDGRID_FROM_EMAIL not configured'
            }
        
        logger.info(f"[SendGrid] Sending templated email to {to_email} using template {template_id}")
        
        # Create email message with template
        message = Mail(
            from_email=Email(from_email, from_name),
            to_emails=to_email
        )
        
        message.template_id = template_id
        message.dynamic_template_data = template_data
        
        # Send email
        response = sg.send(message)
        
        # Extract message ID from response headers
        message_id = None
        if hasattr(response, 'headers') and 'X-Message-Id' in response.headers:
            message_id = response.headers['X-Message-Id']
        
        logger.info(f"[SendGrid] Templated email sent successfully. Status: {response.status_code}, Message ID: {message_id}")
        
        return {
            'success': True,
            'status_code': response.status_code,
            'message_id': message_id,
            'error': None
        }
        
    except SendGridException as e:
        error_msg = str(e)
        status_code = getattr(e, 'status_code', None) if hasattr(e, 'status_code') else None
        logger.error(f"[SendGrid] Failed to send templated email: {error_msg} (Status: {status_code})")
        
        # Provide user-friendly error messages for common SendGrid errors
        if status_code == 401 or '401' in error_msg or 'unauthorized' in error_msg.lower():
            return {
                'success': False,
                'error': 'SendGrid authentication failed. Please check your API key.',
                'technical_error': error_msg,
                'message_id': None
            }
        elif status_code == 403 or '403' in error_msg or 'forbidden' in error_msg.lower():
            return {
                'success': False,
                'error': 'SendGrid access denied. Please check your account permissions.',
                'technical_error': error_msg,
                'message_id': None
            }
        elif 'api key' in error_msg.lower() or 'api_key' in error_msg.lower():
            return {
                'success': False,
                'error': 'SendGrid API key is missing or invalid. Please configure SENDGRID_API_KEY.',
                'technical_error': error_msg,
                'message_id': None
            }
        
        return {
            'success': False,
            'error': error_msg,
            'message_id': None
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[SendGrid] Failed to send templated email: {error_msg}")
        
        # Check for HTTP errors in the error message
        if '401' in error_msg or 'unauthorized' in error_msg.lower():
            return {
                'success': False,
                'error': 'SendGrid authentication failed. Please check your API key.',
                'technical_error': error_msg,
                'message_id': None
            }
        elif '403' in error_msg or 'forbidden' in error_msg.lower():
            return {
                'success': False,
                'error': 'SendGrid access denied. Please check your account permissions.',
                'technical_error': error_msg,
                'message_id': None
            }
        
        return {
            'success': False,
            'error': error_msg,
            'message_id': None
        }

