from flask import Blueprint, request, jsonify, g
from app.simple_logger import get_logger
from app.models import db, SupportTicket, User
from app.utils import get_current_user, get_current_user_flexible
from app.utils.admin_auth import require_admin_auth
from datetime import datetime
from sqlalchemy import desc, or_

logger = get_logger("support")

support_bp = Blueprint('support', __name__)

# Admin email recipients (for future email implementation)
ADMIN_EMAILS = [
    'vinit@adeptaipro.com',
    'abhi@adeptaipro.com',
    'contact@kempian.ai'
]


@support_bp.route('/tickets', methods=['POST'])
def create_ticket():
    """Create a new support ticket"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        # Get user from database
        user_email = user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found'}), 400
        
        db_user = User.query.filter_by(email=user_email).first()
        if not db_user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        message = data.get('message', '').strip()
        subject = data.get('subject', '').strip()
        
        # Validate message
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if len(message) < 10:
            return jsonify({'error': 'Message must be at least 10 characters long'}), 400
        
        # Create ticket
        ticket = SupportTicket(
            user_id=db_user.id,
            tenant_id=db_user.tenant_id,
            user_email=user_email,
            subject=subject or f"Support Request - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            message=message,
            status='open',
            priority='medium'
        )
        
        db.session.add(ticket)
        db.session.commit()
        
        # TODO: Send email notification to admins
        # from app.emails.support import send_support_ticket_notification_email
        # send_support_ticket_notification_email(ticket)
        
        logger.info(f"Support ticket created: {ticket.id} by user {user_email}")
        
        return jsonify({
            'message': 'Support ticket created successfully',
            'ticket': ticket.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating support ticket: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create support ticket'}), 500


@support_bp.route('/tickets', methods=['GET'])
@require_admin_auth
def list_tickets():
    """List all support tickets (Admin only)"""
    try:
        # Get query parameters
        status = request.args.get('status')
        priority = request.args.get('priority')
        search = request.args.get('search', '').strip()
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Build query
        query = SupportTicket.query
        
        # Filter by status
        if status and status in ['open', 'in_progress', 'resolved', 'closed']:
            query = query.filter(SupportTicket.status == status)
        
        # Filter by priority
        if priority and priority in ['low', 'medium', 'high']:
            query = query.filter(SupportTicket.priority == priority)
        
        # Search by user email or ticket ID
        if search:
            try:
                ticket_id = int(search)
                query = query.filter(SupportTicket.id == ticket_id)
            except ValueError:
                query = query.filter(SupportTicket.user_email.ilike(f'%{search}%'))
        
        # Order by created_at descending
        query = query.order_by(desc(SupportTicket.created_at))
        
        # Paginate
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        tickets = pagination.items
        
        return jsonify({
            'tickets': [ticket.to_dict() for ticket in tickets],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing support tickets: {e}")
        return jsonify({'error': 'Failed to retrieve support tickets'}), 500


@support_bp.route('/tickets/<int:ticket_id>', methods=['GET'])
@require_admin_auth
def get_ticket(ticket_id):
    """Get a single support ticket (Admin only)"""
    try:
        ticket = SupportTicket.query.get_or_404(ticket_id)
        return jsonify({'ticket': ticket.to_dict()}), 200
        
    except Exception as e:
        logger.error(f"Error retrieving support ticket {ticket_id}: {e}")
        return jsonify({'error': 'Failed to retrieve support ticket'}), 500


@support_bp.route('/tickets/<int:ticket_id>/reply', methods=['POST'])
@require_admin_auth
def reply_to_ticket(ticket_id):
    """Admin reply to a support ticket"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = user.get('email')
        db_user = User.query.filter_by(email=user_email).first()
        if not db_user:
            return jsonify({'error': 'User not found'}), 404
        
        ticket = SupportTicket.query.get_or_404(ticket_id)
        
        data = request.get_json()
        reply_message = data.get('reply', '').strip()
        new_status = data.get('status', ticket.status)
        
        # Validate reply
        if not reply_message:
            return jsonify({'error': 'Reply message is required'}), 400
        
        if len(reply_message) < 5:
            return jsonify({'error': 'Reply must be at least 5 characters long'}), 400
        
        # Validate status
        if new_status not in ['open', 'in_progress', 'resolved', 'closed']:
            return jsonify({'error': 'Invalid status'}), 400
        
        # Update ticket
        ticket.admin_reply = reply_message
        ticket.replied_by = db_user.id
        ticket.replied_at = datetime.utcnow()
        ticket.status = new_status
        ticket.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # TODO: Send email to user with admin's reply
        # from app.emails.support import send_support_ticket_reply_email
        # send_support_ticket_reply_email(ticket)
        
        logger.info(f"Admin {user_email} replied to ticket {ticket_id}")
        
        return jsonify({
            'message': 'Reply sent successfully',
            'ticket': ticket.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error replying to ticket {ticket_id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to send reply'}), 500


@support_bp.route('/tickets/my-tickets', methods=['GET'])
def get_my_tickets():
    """Get current user's own tickets"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found'}), 400
        
        db_user = User.query.filter_by(email=user_email).first()
        if not db_user:
            return jsonify({'error': 'User not found'}), 404
        
        tickets = SupportTicket.query.filter_by(user_id=db_user.id)\
            .order_by(desc(SupportTicket.created_at))\
            .all()
        
        return jsonify({
            'tickets': [ticket.to_dict() for ticket in tickets]
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving user tickets: {e}")
        return jsonify({'error': 'Failed to retrieve tickets'}), 500


@support_bp.route('/tickets/<int:ticket_id>/status', methods=['PATCH'])
@require_admin_auth
def update_ticket_status(ticket_id):
    """Update ticket status (Admin only)"""
    try:
        ticket = SupportTicket.query.get_or_404(ticket_id)
        
        data = request.get_json()
        new_status = data.get('status')
        
        if new_status not in ['open', 'in_progress', 'resolved', 'closed']:
            return jsonify({'error': 'Invalid status'}), 400
        
        ticket.status = new_status
        ticket.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Ticket {ticket_id} status updated to {new_status}")
        
        return jsonify({
            'message': 'Status updated successfully',
            'ticket': ticket.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating ticket status: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update status'}), 500


@support_bp.route('/tickets/<int:ticket_id>', methods=['DELETE'])
@require_admin_auth
def delete_ticket(ticket_id):
    """Delete a support ticket (Admin only)"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = user.get('email')
        ticket = SupportTicket.query.get_or_404(ticket_id)
        
        # Store ticket info for logging before deletion
        ticket_id_log = ticket.id
        ticket_user_email = ticket.user_email
        
        db.session.delete(ticket)
        db.session.commit()
        
        logger.info(f"Admin {user_email} deleted ticket {ticket_id_log} (user: {ticket_user_email})")
        
        return jsonify({
            'message': 'Ticket deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting ticket {ticket_id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete ticket'}), 500
