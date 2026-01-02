"""
Communication API Routes
Handle template management, message sending, and communication history
"""
from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import db, MessageTemplate, CandidateCommunication, CommunicationReply
from app.communications.service import (
    send_candidate_message,
    get_template,
    update_communication_status,
    add_communication_reply
)
from app.utils import get_current_user
from datetime import datetime
from sqlalchemy import desc

logger = get_logger('communications_routes')

communications_bp = Blueprint('communications', __name__)

# Template Management Routes
@communications_bp.route('/templates', methods=['GET'])
def list_templates():
    """List all message templates for the current user"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Get user ID from user object
        user_id = user.get('id') or user.get('user_id')
        if not user_id:
            # Try to get from database
            from app.models import User
            email = user.get('email')
            if email:
                db_user = User.query.filter_by(email=email).first()
                if db_user:
                    user_id = db_user.id
                else:
                    return jsonify({'error': 'User not found'}), 404
            else:
                return jsonify({'error': 'User ID not found'}), 404
        
        # Get query parameters
        channel = request.args.get('channel', None)
        is_default = request.args.get('is_default', None)
        
        # Build query
        query = MessageTemplate.query.filter_by(user_id=user_id, is_active=True)
        
        if channel:
            query = query.filter_by(channel=channel)
        
        if is_default is not None:
            query = query.filter_by(is_default=is_default.lower() == 'true')
        
        templates = query.order_by(desc(MessageTemplate.created_at)).all()
        
        return jsonify({
            'templates': [template.to_dict() for template in templates],
            'total': len(templates)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing templates: {str(e)}")
        return jsonify({'error': 'Failed to list templates'}), 500

@communications_bp.route('/templates', methods=['POST'])
def create_template():
    """Create a new message template"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Get user ID
        user_id = user.get('id') or user.get('user_id')
        if not user_id:
            from app.models import User
            email = user.get('email')
            if email:
                db_user = User.query.filter_by(email=email).first()
                if db_user:
                    user_id = db_user.id
                else:
                    return jsonify({'error': 'User not found'}), 404
            else:
                return jsonify({'error': 'User ID not found'}), 404
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'channel', 'body']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Validate channel
        if data['channel'] not in ['email', 'sms', 'whatsapp']:
            return jsonify({'error': 'Invalid channel. Must be email, sms, or whatsapp'}), 400
        
        # Create template
        template = MessageTemplate(
            user_id=user_id,
            name=data['name'],
            channel=data['channel'],
            subject=data.get('subject'),
            body=data['body'],
            is_default=data.get('is_default', False),
            is_active=True
        )
        
        db.session.add(template)
        db.session.commit()
        
        logger.info(f"Template created: {template.id} by user {user_id}")
        
        return jsonify({
            'message': 'Template created successfully',
            'template': template.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating template: {str(e)}")
        return jsonify({'error': 'Failed to create template'}), 500

@communications_bp.route('/templates/<int:template_id>', methods=['GET'])
def get_template_details(template_id):
    """Get a specific template"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        template = MessageTemplate.query.get_or_404(template_id)
        
        # Check ownership
        user_id = user.get('id') or user.get('user_id')
        if not user_id:
            from app.models import User
            email = user.get('email')
            if email:
                db_user = User.query.filter_by(email=email).first()
                if db_user:
                    user_id = db_user.id
        
        if template.user_id != user_id:
            return jsonify({'error': 'Template not found'}), 404
        
        return jsonify({
            'template': template.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting template: {str(e)}")
        return jsonify({'error': 'Failed to get template'}), 500

@communications_bp.route('/templates/<int:template_id>', methods=['PUT'])
def update_template(template_id):
    """Update a message template"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        template = MessageTemplate.query.get_or_404(template_id)
        
        # Check ownership
        user_id = user.get('id') or user.get('user_id')
        if not user_id:
            from app.models import User
            email = user.get('email')
            if email:
                db_user = User.query.filter_by(email=email).first()
                if db_user:
                    user_id = db_user.id
        
        if template.user_id != user_id:
            return jsonify({'error': 'Template not found'}), 404
        
        data = request.get_json()
        
        # Update fields
        if 'name' in data:
            template.name = data['name']
        if 'subject' in data:
            template.subject = data['subject']
        if 'body' in data:
            template.body = data['body']
        if 'is_default' in data:
            template.is_default = data['is_default']
        if 'is_active' in data:
            template.is_active = data['is_active']
        
        template.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Template updated: {template_id} by user {user_id}")
        
        return jsonify({
            'message': 'Template updated successfully',
            'template': template.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating template: {str(e)}")
        return jsonify({'error': 'Failed to update template'}), 500

@communications_bp.route('/templates/<int:template_id>', methods=['DELETE'])
def delete_template(template_id):
    """Delete a message template"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        template = MessageTemplate.query.get_or_404(template_id)
        
        # Check ownership
        user_id = user.get('id') or user.get('user_id')
        if not user_id:
            from app.models import User
            email = user.get('email')
            if email:
                db_user = User.query.filter_by(email=email).first()
                if db_user:
                    user_id = db_user.id
        
        if template.user_id != user_id:
            return jsonify({'error': 'Template not found'}), 404
        
        # Soft delete
        template.is_active = False
        template.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Template deleted: {template_id} by user {user_id}")
        
        return jsonify({'message': 'Template deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting template: {str(e)}")
        return jsonify({'error': 'Failed to delete template'}), 500

# Message Sending Routes
@communications_bp.route('/send', methods=['POST'])
def send_message():
    """Send a message to a candidate via Email, SMS, or WhatsApp"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Get user ID
        user_id = user.get('id') or user.get('user_id')
        if not user_id:
            from app.models import User
            email = user.get('email')
            if email:
                db_user = User.query.filter_by(email=email).first()
                if db_user:
                    user_id = db_user.id
                else:
                    return jsonify({'error': 'User not found'}), 404
            else:
                return jsonify({'error': 'User ID not found'}), 404
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        # Validate candidate_id - must be present and non-empty
        candidate_id = data.get('candidate_id')
        if not candidate_id:
            return jsonify({'error': 'candidate_id is required'}), 400
        
        # Convert candidate_id to string and check it's not empty
        candidate_id_str = str(candidate_id).strip()
        if not candidate_id_str:
            return jsonify({'error': 'candidate_id cannot be empty'}), 400
        
        # Validate channel
        channel = data.get('channel')
        if not channel:
            return jsonify({'error': 'channel is required'}), 400
        
        if channel not in ['email', 'sms', 'whatsapp']:
            return jsonify({'error': 'Invalid channel. Must be email, sms, or whatsapp'}), 400
        
        # Call service function
        result = send_candidate_message(
            user_id=user_id,
            candidate_id=candidate_id_str,
            candidate_name=data.get('candidate_name'),
            candidate_email=data.get('candidate_email'),
            candidate_phone=data.get('candidate_phone'),
            channel=channel,
            template_id=data.get('template_id'),
            custom_message=data.get('custom_message'),
            subject=data.get('subject'),
            email_provider=data.get('email_provider', 'sendgrid'),
            variables=data.get('variables')
        )
        
        if result.get('success'):
            return jsonify({
                'message': 'Message sent successfully',
                'communication_id': result.get('communication_id'),
                'message_sid': result.get('message_sid'),
                'message_id': result.get('message_id')
            }), 200
        else:
            return jsonify({
                'error': result.get('error', 'Failed to send message')
            }), 400
        
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        return jsonify({'error': 'Failed to send message'}), 500

# Communication History Routes
@communications_bp.route('/history', methods=['GET'])
def get_communication_history():
    """Get communication history for the current user"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Get user ID
        user_id = user.get('id') or user.get('user_id')
        if not user_id:
            from app.models import User
            email = user.get('email')
            if email:
                db_user = User.query.filter_by(email=email).first()
                if db_user:
                    user_id = db_user.id
                else:
                    return jsonify({'error': 'User not found'}), 404
            else:
                return jsonify({'error': 'User ID not found'}), 404
        
        # Get query parameters
        candidate_id = request.args.get('candidate_id', None)
        channel = request.args.get('channel', None)
        status = request.args.get('status', None)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        # Build query
        query = CandidateCommunication.query.filter_by(user_id=user_id)
        
        if candidate_id:
            query = query.filter_by(candidate_id=candidate_id)
        
        if channel:
            query = query.filter_by(channel=channel)
        
        if status:
            query = query.filter_by(status=status)
        
        # Paginate results
        pagination = query.order_by(desc(CandidateCommunication.created_at)).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'communications': [comm.to_dict() for comm in pagination.items],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
            'per_page': per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting communication history: {str(e)}")
        return jsonify({'error': 'Failed to get communication history'}), 500

@communications_bp.route('/history/<int:communication_id>', methods=['GET'])
def get_communication_details(communication_id):
    """Get details of a specific communication"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        communication = CandidateCommunication.query.get_or_404(communication_id)
        
        # Check ownership
        user_id = user.get('id') or user.get('user_id')
        if not user_id:
            from app.models import User
            email = user.get('email')
            if email:
                db_user = User.query.filter_by(email=email).first()
                if db_user:
                    user_id = db_user.id
        
        if communication.user_id != user_id:
            return jsonify({'error': 'Communication not found'}), 404
        
        return jsonify({
            'communication': communication.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting communication details: {str(e)}")
        return jsonify({'error': 'Failed to get communication details'}), 500

@communications_bp.route('/history/<int:communication_id>/replies', methods=['GET'])
def get_communication_replies(communication_id):
    """Get replies for a specific communication"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        communication = CandidateCommunication.query.get_or_404(communication_id)
        
        # Check ownership
        user_id = user.get('id') or user.get('user_id')
        if not user_id:
            from app.models import User
            email = user.get('email')
            if email:
                db_user = User.query.filter_by(email=email).first()
                if db_user:
                    user_id = db_user.id
        
        if communication.user_id != user_id:
            return jsonify({'error': 'Communication not found'}), 404
        
        # Get replies
        replies = CommunicationReply.query.filter_by(
            communication_id=communication_id
        ).order_by(desc(CommunicationReply.created_at)).all()
        
        return jsonify({
            'replies': [reply.to_dict() for reply in replies],
            'total': len(replies)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting communication replies: {str(e)}")
        return jsonify({'error': 'Failed to get communication replies'}), 500

