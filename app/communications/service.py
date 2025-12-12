"""
Communication Service Layer
Handles sending messages via Email (SendGrid/Hostinger SMTP), SMS, and WhatsApp
"""
from app.simple_logger import get_logger
from app.models import MessageTemplate, CandidateCommunication, CommunicationReply, db
from app.sendgrid.email_service import send_email, send_email_with_template
from app.emails.smtp import send_email_via_smtp
from app.twilio.sms_service import send_sms
from app.twilio.whatsapp_service import send_whatsapp
from datetime import datetime
import json
import re

logger = get_logger('communications')

def replace_template_variables(template_body, variables):
    """
    Replace template variables in message body
    
    Args:
        template_body: Template string with variables like {{candidate_name}}
        variables: Dictionary of variable values
    
    Returns:
        String with variables replaced
    """
    try:
        result = template_body
        for key, value in variables.items():
            # Replace {{variable_name}} or {variable_name}
            pattern = r'\{\{?\s*' + re.escape(key) + r'\s*\}?\}?'
            result = re.sub(pattern, str(value), result, flags=re.IGNORECASE)
        return result
    except Exception as e:
        logger.error(f"[Communications] Error replacing template variables: {str(e)}")
        return template_body

def get_template(user_id, template_id=None, channel=None, is_default=False):
    """
    Get message template
    
    Args:
        user_id: User ID
        template_id: Template ID (optional)
        channel: Channel type (optional)
        is_default: Whether to get default template (optional)
    
    Returns:
        MessageTemplate object or None
    """
    try:
        query = MessageTemplate.query.filter_by(user_id=user_id, is_active=True)
        
        if template_id:
            query = query.filter_by(id=template_id)
        
        if channel:
            query = query.filter_by(channel=channel)
        
        if is_default:
            query = query.filter_by(is_default=True)
        
        return query.first()
    except Exception as e:
        logger.error(f"[Communications] Error getting template: {str(e)}")
        return None

def send_candidate_message(user_id, candidate_id, candidate_name, candidate_email, 
                          candidate_phone, channel, template_id=None, custom_message=None,
                          subject=None, email_provider='sendgrid', variables=None):
    """
    Send message to candidate via Email, SMS, or WhatsApp
    
    Args:
        user_id: User ID sending the message
        candidate_id: Candidate ID
        candidate_name: Candidate name
        candidate_email: Candidate email (for email channel)
        candidate_phone: Candidate phone (for SMS/WhatsApp)
        channel: 'email', 'sms', or 'whatsapp'
        template_id: Template ID to use (optional)
        custom_message: Custom message text (optional, overrides template)
        subject: Email subject (for email channel)
        email_provider: 'sendgrid' or 'smtp' (for email channel)
        variables: Dictionary of template variables
    
    Returns:
        dict with 'success', 'communication_id', 'error' keys
    """
    try:
        # Validate candidate_id
        if not candidate_id:
            return {
                'success': False,
                'error': 'Candidate ID is required'
            }
        
        # Convert candidate_id to string and check it's not empty
        candidate_id_str = str(candidate_id).strip()
        if not candidate_id_str:
            return {
                'success': False,
                'error': 'Candidate ID cannot be empty'
            }
        
        # Validate channel
        if channel not in ['email', 'sms', 'whatsapp']:
            return {
                'success': False,
                'error': f'Invalid channel: {channel}'
            }
        
        # Validate recipient based on channel
        if channel == 'email' and not candidate_email:
            return {
                'success': False,
                'error': 'Candidate email is required for email channel'
            }
        
        if channel in ['sms', 'whatsapp'] and not candidate_phone:
            return {
                'success': False,
                'error': f'Candidate phone is required for {channel} channel'
            }
        
        # Get template if template_id provided
        template = None
        message_body = custom_message
        message_subject = subject
        
        if template_id:
            template = get_template(user_id, template_id=template_id)
            if not template:
                return {
                    'success': False,
                    'error': f'Template {template_id} not found'
                }
            
            # Prepare variables
            template_vars = variables or {}
            template_vars.update({
                'candidate_name': candidate_name or 'Candidate',
                'candidate_email': candidate_email or '',
                'candidate_phone': candidate_phone or '',
                'company_name': 'Our Company',  # TODO: Get from user
                'sender_name': 'Recruiter',  # TODO: Get from user
            })
            
            # Replace template variables
            message_body = replace_template_variables(template.body, template_vars)
            if template.subject:
                message_subject = replace_template_variables(template.subject, template_vars)
        
        if not message_body:
            return {
                'success': False,
                'error': 'Message body is required'
            }
        
        # Create communication record
        communication = CandidateCommunication(
            user_id=user_id,
            candidate_id=candidate_id_str,
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            candidate_phone=candidate_phone,
            channel=channel,
            template_id=template_id,
            message_subject=message_subject,
            message_body=message_body,
            status='pending'
        )
        db.session.add(communication)
        db.session.flush()  # Get the ID
        
        # Send message based on channel
        send_result = None
        
        if channel == 'email':
            if email_provider == 'sendgrid':
                # Use SendGrid
                send_result = send_email(
                    to_email=candidate_email,
                    subject=message_subject or 'Message from Kempian AI',
                    html_content=message_body,
                    text_content=message_body  # Simple text version
                )
                if send_result.get('success'):
                    communication.sendgrid_message_id = send_result.get('message_id')
            else:
                # Use Hostinger SMTP
                send_result = {
                    'success': send_email_via_smtp(
                        to_email=candidate_email,
                        subject=message_subject or 'Message from Kempian AI',
                        body_html=message_body,
                        body_text=message_body
                    ),
                    'message_id': None
                }
        
        elif channel == 'sms':
            send_result = send_sms(
                to_phone=candidate_phone,
                message_body=message_body
            )
            if send_result.get('success'):
                communication.twilio_message_sid = send_result.get('message_sid')
        
        elif channel == 'whatsapp':
            send_result = send_whatsapp(
                to_phone=candidate_phone,
                message_body=message_body
            )
            if send_result.get('success'):
                communication.twilio_message_sid = send_result.get('message_sid')
        
        # Update communication record
        if send_result and send_result.get('success'):
            communication.status = 'sent'
            communication.sent_at = datetime.utcnow()
            communication.delivery_status = send_result.get('status', 'sent')
        else:
            communication.status = 'failed'
            communication.error_message = send_result.get('error', 'Unknown error') if send_result else 'Send failed'
        
        db.session.commit()
        
        return {
            'success': send_result.get('success', False) if send_result else False,
            'communication_id': communication.id,
            'message_sid': send_result.get('message_sid') if send_result else None,
            'message_id': send_result.get('message_id') if send_result else None,
            'error': send_result.get('error') if send_result and not send_result.get('success') else None
        }
        
    except Exception as e:
        db.session.rollback()
        error_msg = str(e)
        logger.error(f"[Communications] Error sending message: {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }

def update_communication_status(communication_id, status, delivery_status=None, 
                                error_message=None, delivered_at=None, read_at=None):
    """
    Update communication status
    
    Args:
        communication_id: Communication ID
        status: New status
        delivery_status: Delivery status (optional)
        error_message: Error message (optional)
        delivered_at: Delivery timestamp (optional)
        read_at: Read timestamp (optional)
    
    Returns:
        bool: Success status
    """
    try:
        communication = CandidateCommunication.query.get(communication_id)
        if not communication:
            return False
        
        communication.status = status
        if delivery_status:
            communication.delivery_status = delivery_status
        if error_message:
            communication.error_message = error_message
        if delivered_at:
            communication.delivered_at = delivered_at
        if read_at:
            communication.read_at = read_at
        
        communication.updated_at = datetime.utcnow()
        db.session.commit()
        
        return True
    except Exception as e:
        db.session.rollback()
        logger.error(f"[Communications] Error updating communication status: {str(e)}")
        return False

def add_communication_reply(communication_id, reply_content, channel, 
                           candidate_phone=None, candidate_email=None, twilio_message_sid=None):
    """
    Add reply to a communication
    
    Args:
        communication_id: Communication ID
        reply_content: Reply message content
        channel: Channel type
        candidate_phone: Candidate phone (for SMS/WhatsApp)
        candidate_email: Candidate email (for email)
        twilio_message_sid: Twilio message SID (optional)
    
    Returns:
        dict with 'success', 'reply_id', 'error' keys
    """
    try:
        communication = CandidateCommunication.query.get(communication_id)
        if not communication:
            return {
                'success': False,
                'error': 'Communication not found'
            }
        
        # Create reply record
        reply = CommunicationReply(
            communication_id=communication_id,
            candidate_phone=candidate_phone,
            candidate_email=candidate_email,
            reply_content=reply_content,
            channel=channel,
            twilio_message_sid=twilio_message_sid
        )
        db.session.add(reply)
        
        # Update communication
        communication.status = 'replied'
        communication.replied_at = datetime.utcnow()
        communication.reply_content = reply_content
        communication.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return {
            'success': True,
            'reply_id': reply.id
        }
    except Exception as e:
        db.session.rollback()
        error_msg = str(e)
        logger.error(f"[Communications] Error adding reply: {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }

