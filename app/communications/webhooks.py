"""
Webhook Handlers for Twilio and SendGrid
Handle incoming messages and status updates
"""
from flask import Blueprint, request, jsonify, current_app
from app.simple_logger import get_logger
from app.models import db, CandidateCommunication, CommunicationReply
from app.communications.service import update_communication_status, add_communication_reply
from datetime import datetime
import os

logger = get_logger('communications_webhooks')

webhooks_bp = Blueprint('communication_webhooks', __name__)

# Twilio Webhook Routes
@webhooks_bp.route('/twilio/sms', methods=['POST'])
def twilio_sms_webhook():
    """Handle Twilio SMS webhook (status updates and incoming messages)"""
    try:
        data = request.form.to_dict()
        
        message_sid = data.get('MessageSid')
        message_status = data.get('MessageStatus')
        from_number = data.get('From')
        to_number = data.get('To')
        body = data.get('Body', '')
        
        logger.info(f"[Twilio SMS Webhook] Message SID: {message_sid}, Status: {message_status}")
        
        # Find communication by message SID
        if message_sid:
            communication = CandidateCommunication.query.filter_by(
                twilio_message_sid=message_sid
            ).first()
            
            if communication:
                # Update status based on message status
                status_map = {
                    'queued': 'pending',
                    'sent': 'sent',
                    'delivered': 'delivered',
                    'undelivered': 'failed',
                    'failed': 'failed',
                    'read': 'read'
                }
                
                new_status = status_map.get(message_status, communication.status)
                
                # Update communication
                update_communication_status(
                    communication_id=communication.id,
                    status=new_status,
                    delivery_status=message_status
                )
                
                # If this is an incoming message (from candidate)
                if from_number and body and from_number != to_number:
                    # This is a reply
                    add_communication_reply(
                        communication_id=communication.id,
                        reply_content=body,
                        channel='sms',
                        candidate_phone=from_number,
                        twilio_message_sid=message_sid
                    )
        
        # Return TwiML response (required by Twilio)
        return '<?xml version="1.0" encoding="UTF-8"?><Response></Response>', 200
        
    except Exception as e:
        logger.error(f"[Twilio SMS Webhook] Error: {str(e)}")
        return '<?xml version="1.0" encoding="UTF-8"?><Response></Response>', 200

@webhooks_bp.route('/twilio/whatsapp', methods=['POST'])
def twilio_whatsapp_webhook():
    """Handle Twilio WhatsApp webhook (status updates and incoming messages)"""
    try:
        data = request.form.to_dict()
        
        message_sid = data.get('MessageSid')
        message_status = data.get('MessageStatus')
        from_number = data.get('From', '').replace('whatsapp:', '')
        to_number = data.get('To', '').replace('whatsapp:', '')
        body = data.get('Body', '')
        
        logger.info(f"[Twilio WhatsApp Webhook] Message SID: {message_sid}, Status: {message_status}")
        
        # Find communication by message SID
        if message_sid:
            communication = CandidateCommunication.query.filter_by(
                twilio_message_sid=message_sid
            ).first()
            
            if communication:
                # Update status based on message status
                status_map = {
                    'queued': 'pending',
                    'sent': 'sent',
                    'delivered': 'delivered',
                    'undelivered': 'failed',
                    'failed': 'failed',
                    'read': 'read'
                }
                
                new_status = status_map.get(message_status, communication.status)
                
                # Update communication
                update_communication_status(
                    communication_id=communication.id,
                    status=new_status,
                    delivery_status=message_status
                )
                
                # If this is an incoming message (from candidate)
                if from_number and body and from_number != to_number:
                    # This is a reply
                    add_communication_reply(
                        communication_id=communication.id,
                        reply_content=body,
                        channel='whatsapp',
                        candidate_phone=from_number,
                        twilio_message_sid=message_sid
                    )
        
        # Return TwiML response (required by Twilio)
        return '<?xml version="1.0" encoding="UTF-8"?><Response></Response>', 200
        
    except Exception as e:
        logger.error(f"[Twilio WhatsApp Webhook] Error: {str(e)}")
        return '<?xml version="1.0" encoding="UTF-8"?><Response></Response>', 200

# SendGrid Webhook Routes
@webhooks_bp.route('/sendgrid', methods=['POST'])
def sendgrid_webhook():
    """Handle SendGrid webhook (email events)"""
    try:
        # SendGrid sends events as JSON array
        events = request.get_json()
        
        if not isinstance(events, list):
            events = [events]
        
        for event in events:
            event_type = event.get('event')
            message_id = event.get('sg_message_id')
            email = event.get('email')
            timestamp = event.get('timestamp')
            
            logger.info(f"[SendGrid Webhook] Event: {event_type}, Message ID: {message_id}")
            
            # Find communication by message ID
            if message_id:
                # SendGrid message ID format: <message_id@domain>
                # Extract the message ID part
                message_id_clean = message_id.split('@')[0] if '@' in message_id else message_id
                
                communication = CandidateCommunication.query.filter(
                    CandidateCommunication.sendgrid_message_id.like(f'%{message_id_clean}%')
                ).first()
                
                if communication:
                    # Map SendGrid events to our status
                    status_map = {
                        'processed': 'sent',
                        'delivered': 'delivered',
                        'open': 'read',
                        'click': 'read',
                        'bounce': 'failed',
                        'dropped': 'failed',
                        'spamreport': 'failed',
                        'unsubscribe': 'failed'
                    }
                    
                    new_status = status_map.get(event_type, communication.status)
                    
                    # Update communication
                    if event_type == 'delivered':
                        delivered_at = datetime.fromtimestamp(timestamp) if timestamp else datetime.utcnow()
                        update_communication_status(
                            communication_id=communication.id,
                            status=new_status,
                            delivery_status=event_type,
                            delivered_at=delivered_at
                        )
                    elif event_type == 'open':
                        read_at = datetime.fromtimestamp(timestamp) if timestamp else datetime.utcnow()
                        update_communication_status(
                            communication_id=communication.id,
                            status=new_status,
                            delivery_status=event_type,
                            read_at=read_at
                        )
                    else:
                        update_communication_status(
                            communication_id=communication.id,
                            status=new_status,
                            delivery_status=event_type
                        )
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        logger.error(f"[SendGrid Webhook] Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

