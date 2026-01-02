from flask import Blueprint, request, jsonify, g, send_file
from app.simple_logger import get_logger
from app.models import db, SupportTicket, TicketAttachment, User
from app.utils import get_current_user, get_current_user_flexible
from app.utils.admin_auth import require_admin_auth
from datetime import datetime, timedelta
from sqlalchemy import desc, or_
import os
from werkzeug.utils import secure_filename
from app.emails.smtp import send_email_via_smtp

logger = get_logger("support")

support_bp = Blueprint('support', __name__)

# Admin email recipients (for future email implementation)
ADMIN_EMAILS = [
    'vinit@adeptaipro.com',
    'abhi@adeptaipro.com',
    'rushikesh@adeptaipro.com',
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
        category = data.get('category', '').strip()
        source = data.get('source', 'help_widget')
        source_url = data.get('source_url', '').strip()
        tags = data.get('tags', [])
        
        # Validate message
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if len(message) < 10:
            return jsonify({'error': 'Message must be at least 10 characters long'}), 400
        
        # Validate category if provided
        valid_categories = ['bug', 'feature_request', 'question', 'billing', 'technical', 'account', 'integration', 'other']
        if category and category not in valid_categories:
            return jsonify({'error': f'Invalid category. Must be one of: {", ".join(valid_categories)}'}), 400
        
        # Validate source
        valid_sources = ['help_widget', 'email', 'dashboard', 'api', 'other']
        if source not in valid_sources:
            source = 'help_widget'
        
        # Create ticket
        ticket = SupportTicket(
            user_id=db_user.id,
            tenant_id=db_user.tenant_id,
            user_email=user_email,
            subject=subject or f"Support Request - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            message=message,
            status='open',
            priority='medium',
            category=category if category else None,
            source=source,
            source_url=source_url if source_url else None,
            tags=tags if tags else None
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


@support_bp.route('/enterprise-request', methods=['POST'])
def create_enterprise_request():
    """Create an enterprise plan request ticket and send emails to admins"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        # Get user from database
        user_email = user.get('email')
        user_name = user.get('name', '')
        if not user_email:
            return jsonify({'error': 'User email not found'}), 400
        
        db_user = User.query.filter_by(email=user_email).first()
        if not db_user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        plan_name = data.get('plan_name', 'Enterprise')
        additional_info = data.get('additional_info', '').strip()
        requirements = data.get('requirements', {})
        
        # Format requirements for ticket message
        requirements_text = ""
        if requirements:
            requirements_text = f"""
Requirements Details:
- Company Name: {requirements.get('companyName', 'Not provided')}
- Contact Name: {requirements.get('contactName', 'Not provided')}
- Email: {requirements.get('email', 'Not provided')}
- Phone: {requirements.get('phone', 'Not provided')}
- Team Size: {requirements.get('teamSize', 'Not provided')}
- Expected Monthly Usage: {requirements.get('expectedUsage', 'Not provided')}
- Current Challenges: {requirements.get('currentChallenges', 'Not provided')}
- Timeline for Implementation: {requirements.get('timeline', 'Not provided')}
- Additional Requirements: {requirements.get('additionalRequirements', 'Not provided')}
"""
        elif additional_info:
            requirements_text = f"\nAdditional Information:\n{additional_info}"
        
        # Create detailed message for the ticket
        message = f"""Enterprise Plan Request

User Details:
- Name: {user_name or 'Not provided'}
- Email: {user_email}
- User ID: {db_user.id}
- Tenant ID: {db_user.tenant_id}

Plan Requested: {plan_name}
{requirements_text}

This request was submitted from the pricing page.
"""
        
        # Create support ticket with high priority
        ticket = SupportTicket(
            user_id=db_user.id,
            tenant_id=db_user.tenant_id,
            user_email=user_email,
            subject=f"Enterprise Plan Request - {user_email}",
            message=message,
            status='open',
            priority='high',
            category='billing',
            source='dashboard',
            source_url='/pricing',
            tags=['enterprise', 'plan_request']
        )
        
        db.session.add(ticket)
        db.session.commit()
        
        # Prepare email content
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Plan Request</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8fafc;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: #ffffff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 30px;
            text-align: center;
            color: white;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 600;
        }}
        .content {{
            padding: 40px 30px;
        }}
        .alert-box {{
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .alert-box h2 {{
            margin: 0 0 10px 0;
            font-size: 24px;
        }}
        .info-box {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .data-table td {{
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
        }}
        .data-table td:first-child {{
            font-weight: 600;
            color: #6b7280;
            width: 200px;
        }}
        .data-table tr:last-child td {{
            border-bottom: none;
        }}
        .cta-button {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 16px;
            margin: 20px 0;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enterprise Plan Request</h1>
        </div>
        
        <div class="content">
            <div class="alert-box">
                <h2>New Enterprise Plan Request</h2>
                <p>Received on: {timestamp}</p>
            </div>
            
            <p>Hello Admin Team,</p>
            
            <p>A user has requested information about the Enterprise plan. Please review the details below and follow up with the user.</p>
            
            <div class="info-box">
                <h3 style="margin-top: 0; color: #667eea;">User Information</h3>
                <table class="data-table">
                    <tr>
                        <td>Name:</td>
                        <td>{user_name or 'Not provided'}</td>
                    </tr>
                    <tr>
                        <td>Email:</td>
                        <td><a href="mailto:{user_email}">{user_email}</a></td>
                    </tr>
                    <tr>
                        <td>User ID:</td>
                        <td>{db_user.id}</td>
                    </tr>
                    <tr>
                        <td>Tenant ID:</td>
                        <td>{db_user.tenant_id}</td>
                    </tr>
                    <tr>
                        <td>Plan Requested:</td>
                        <td><strong>{plan_name}</strong></td>
                    </tr>
                    <tr>
                        <td>Support Ticket ID:</td>
                        <td>#{ticket.id}</td>
                    </tr>
                </table>
            </div>
            
            {f'''<div class="info-box">
                <h3 style="margin-top: 0; color: #667eea;">Requirements & Details</h3>
                <table class="data-table">
                    <tr><td>Company Name:</td><td>{requirements.get('companyName', 'Not provided')}</td></tr>
                    <tr><td>Contact Name:</td><td>{requirements.get('contactName', 'Not provided')}</td></tr>
                    <tr><td>Email:</td><td><a href="mailto:{requirements.get('email', user_email)}">{requirements.get('email', user_email)}</a></td></tr>
                    <tr><td>Phone:</td><td>{requirements.get('phone', 'Not provided')}</td></tr>
                    <tr><td>Team Size:</td><td>{requirements.get('teamSize', 'Not provided')}</td></tr>
                    <tr><td>Expected Monthly Usage:</td><td>{requirements.get('expectedUsage', 'Not provided')}</td></tr>
                    <tr><td>Current Challenges:</td><td>{requirements.get('currentChallenges', 'Not provided')}</td></tr>
                    <tr><td>Timeline:</td><td>{requirements.get('timeline', 'Not provided')}</td></tr>
                    <tr><td>Additional Requirements:</td><td>{requirements.get('additionalRequirements', 'Not provided')}</td></tr>
                </table>
            </div>''' if requirements and len(requirements) > 0 else (f'<div class="info-box"><h3 style="margin-top: 0; color: #667eea;">Additional Information</h3><p>{additional_info}</p></div>' if additional_info else '')}
            
            <div class="info-box">
                <p style="margin:0;"><strong>Action Required:</strong> Please contact the user to discuss Enterprise plan options, pricing, and implementation details.</p>
            </div>
            
            <div style="text-align: center;">
                <a href="mailto:{user_email}?subject=Re: Enterprise Plan Request" class="cta-button">
                    Contact User
                </a>
            </div>
            
            <p>You can view and manage this ticket in the admin support dashboard.</p>
            
            <p>Best regards,<br>Kempian AI System</p>
        </div>
        
        <div class="footer">
            <p>© 2024 Kempian AI. All rights reserved.</p>
            <p>This is an automated notification from the Kempian AI platform.</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Format text body with requirements
        requirements_text_section = ""
        if requirements:
            requirements_text_section = f"""
Requirements & Details:
- Company Name: {requirements.get('companyName', 'Not provided')}
- Contact Name: {requirements.get('contactName', 'Not provided')}
- Email: {requirements.get('email', user_email)}
- Phone: {requirements.get('phone', 'Not provided')}
- Team Size: {requirements.get('teamSize', 'Not provided')}
- Expected Monthly Usage: {requirements.get('expectedUsage', 'Not provided')}
- Current Challenges: {requirements.get('currentChallenges', 'Not provided')}
- Timeline: {requirements.get('timeline', 'Not provided')}
- Additional Requirements: {requirements.get('additionalRequirements', 'Not provided')}
"""
        elif additional_info:
            requirements_text_section = f"\nAdditional Information:\n{additional_info}\n"
        
        text_body = f"""
Enterprise Plan Request - Kempian AI
Received on: {timestamp}

A user has requested information about the Enterprise plan.

User Information:
- Name: {user_name or 'Not provided'}
- Email: {user_email}
- User ID: {db_user.id}
- Tenant ID: {db_user.tenant_id}
- Plan Requested: {plan_name}
- Support Ticket ID: #{ticket.id}
{requirements_text_section}

Action Required: Please contact the user to discuss Enterprise plan options, pricing, and implementation details.

Contact the user: {user_email}

You can view and manage this ticket in the admin support dashboard.

---
© 2024 Kempian AI. All rights reserved.
This is an automated notification from the Kempian AI platform.
"""
        
        # Send emails to both admin addresses
        email_subject = f"Enterprise Plan Request - {user_email}"
        enterprise_emails = ['vinit@adeptaipro.com', 'abhi@adeptaipro.com', 'contact@kempian.ai']
        
        email_success = True
        for admin_email in enterprise_emails:
            success = send_email_via_smtp(
                to_email=admin_email,
                subject=email_subject,
                body_html=html_body,
                body_text=text_body,
                reply_to=user_email
            )
            if not success:
                email_success = False
                logger.error(f"Failed to send enterprise request email to {admin_email}")
            else:
                logger.info(f"Enterprise request email sent successfully to {admin_email}")
        
        logger.info(f"Enterprise plan request ticket created: {ticket.id} by user {user_email}")
        
        return jsonify({
            'message': 'Enterprise request submitted successfully',
            'ticket': ticket.to_dict(),
            'emails_sent': email_success
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating enterprise request: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create enterprise request'}), 500


@support_bp.route('/tickets', methods=['GET'])
@require_admin_auth
def list_tickets():
    """List all support tickets (Admin only)"""
    try:
        # Get query parameters
        status = request.args.get('status')
        priority = request.args.get('priority')
        category = request.args.get('category')
        assigned_to = request.args.get('assigned_to', type=int)
        source = request.args.get('source')
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
        
        # Filter by category
        if category:
            query = query.filter(SupportTicket.category == category)
        
        # Filter by assigned_to
        if assigned_to:
            query = query.filter(SupportTicket.assigned_to == assigned_to)
        
        # Filter by source
        if source:
            query = query.filter(SupportTicket.source == source)
        
        # Search by user email or ticket ID
        if search:
            try:
                ticket_id = int(search)
                query = query.filter(SupportTicket.id == ticket_id)
            except ValueError:
                query = query.filter(
                    or_(
                        SupportTicket.user_email.ilike(f'%{search}%'),
                        SupportTicket.subject.ilike(f'%{search}%'),
                        SupportTicket.message.ilike(f'%{search}%')
                    )
                )
        
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
        internal_notes = data.get('internal_notes', '').strip()
        
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
        
        # Set first_response_at if this is the first reply
        if not ticket.first_response_at:
            ticket.first_response_at = datetime.utcnow()
        
        # Update internal notes if provided
        if internal_notes is not None:
            ticket.internal_notes = internal_notes if internal_notes else None
            ticket.notes_updated_by = db_user.id
            ticket.notes_updated_at = datetime.utcnow()
        
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
def update_ticket_status(ticket_id):
    """Update ticket status (Admin or ticket owner)"""
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
        
        ticket = SupportTicket.query.get_or_404(ticket_id)
        
        # Check if user is admin/owner
        is_admin = db_user.role in ['admin', 'owner'] or user_email in ['vinit@adeptaipro.com', 'abhi@adeptaipro.com', 'rushikesh@adeptaipro.com', 'contact@kempian.ai']
        is_owner = ticket.user_id == db_user.id
        
        # Only admins can update ticket status
        if not is_admin:
            return jsonify({'error': 'Only administrators can update ticket status'}), 403
        
        data = request.get_json()
        new_status = data.get('status')
        
        if new_status not in ['open', 'in_progress', 'resolved', 'closed']:
            return jsonify({'error': 'Invalid status'}), 400
        
        ticket.status = new_status
        ticket.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Ticket {ticket_id} status updated to {new_status} by {user_email} ({'admin' if is_admin else 'owner'})")
        
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


@support_bp.route('/tickets/<int:ticket_id>/assign', methods=['POST'])
@require_admin_auth
def assign_ticket(ticket_id):
    """Assign a ticket to an admin (Admin only)"""
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
        assign_to_user_id = data.get('assigned_to')
        
        if assign_to_user_id:
            assignee = User.query.get(assign_to_user_id)
            if not assignee:
                return jsonify({'error': 'Assignee user not found'}), 404
            ticket.assigned_to = assign_to_user_id
            ticket.assigned_at = datetime.utcnow()
        else:
            ticket.assigned_to = None
            ticket.assigned_at = None
        
        ticket.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Ticket {ticket_id} assigned to user {assign_to_user_id} by {user_email}")
        
        return jsonify({
            'message': 'Ticket assigned successfully',
            'ticket': ticket.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error assigning ticket {ticket_id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to assign ticket'}), 500


@support_bp.route('/tickets/<int:ticket_id>/internal-notes', methods=['PATCH'])
@require_admin_auth
def update_internal_notes(ticket_id):
    """Update internal notes for a ticket (Admin only)"""
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
        internal_notes = data.get('internal_notes', '').strip()
        
        ticket.internal_notes = internal_notes if internal_notes else None
        ticket.notes_updated_by = db_user.id
        ticket.notes_updated_at = datetime.utcnow()
        ticket.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        logger.info(f"Internal notes updated for ticket {ticket_id} by {user_email}")
        
        return jsonify({
            'message': 'Internal notes updated successfully',
            'ticket': ticket.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating internal notes for ticket {ticket_id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update internal notes'}), 500


@support_bp.route('/tickets/<int:ticket_id>/tags', methods=['PATCH'])
@require_admin_auth
def update_ticket_tags(ticket_id):
    """Update tags for a ticket (Admin only)"""
    try:
        ticket = SupportTicket.query.get_or_404(ticket_id)
        data = request.get_json()
        tags = data.get('tags', [])
        
        # Validate tags (should be array of strings)
        if not isinstance(tags, list):
            return jsonify({'error': 'Tags must be an array'}), 400
        
        # Limit tag length and count
        if len(tags) > 10:
            return jsonify({'error': 'Maximum 10 tags allowed'}), 400
        
        validated_tags = []
        for tag in tags:
            if isinstance(tag, str) and len(tag.strip()) > 0 and len(tag.strip()) <= 50:
                validated_tags.append(tag.strip())
        
        ticket.tags = validated_tags if validated_tags else None
        ticket.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Tags updated successfully',
            'ticket': ticket.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating tags for ticket {ticket_id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update tags'}), 500


@support_bp.route('/tickets/<int:ticket_id>/rating', methods=['POST'])
def rate_ticket(ticket_id):
    """Rate a ticket (Ticket owner only)"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = user.get('email')
        db_user = User.query.filter_by(email=user_email).first()
        if not db_user:
            return jsonify({'error': 'User not found'}), 404
        
        ticket = SupportTicket.query.get_or_404(ticket_id)
        
        # Only ticket owner can rate
        if ticket.user_id != db_user.id:
            return jsonify({'error': 'You can only rate your own tickets'}), 403
        
        # Only resolved/closed tickets can be rated
        if ticket.status not in ['resolved', 'closed']:
            return jsonify({'error': 'You can only rate resolved or closed tickets'}), 400
        
        data = request.get_json()
        rating = data.get('rating')
        feedback = data.get('feedback', '').strip()
        
        if not rating or not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        
        ticket.rating = rating
        ticket.feedback = feedback if feedback else None
        ticket.rated_at = datetime.utcnow()
        ticket.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        logger.info(f"Ticket {ticket_id} rated {rating} stars by {user_email}")
        
        return jsonify({
            'message': 'Rating submitted successfully',
            'ticket': ticket.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error rating ticket {ticket_id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to submit rating'}), 500


@support_bp.route('/tickets/<int:ticket_id>/attachments', methods=['POST'])
def upload_attachment(ticket_id):
    """Upload an attachment to a ticket"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = user.get('email')
        db_user = User.query.filter_by(email=user_email).first()
        if not db_user:
            return jsonify({'error': 'User not found'}), 404
        
        ticket = SupportTicket.query.get_or_404(ticket_id)
        
        # Check permissions: user can upload to their own tickets, admin can upload to any
        is_admin = db_user.role in ['admin', 'owner'] or user_email in ['vinit@adeptaipro.com', 'abhi@adeptaipro.com', 'rushikesh@adeptaipro.com', 'contact@kempian.ai']
        is_owner = ticket.user_id == db_user.id
        
        if not (is_admin or is_owner):
            return jsonify({'error': 'You can only upload attachments to your own tickets'}), 403
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file size (max 10MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': 'File size exceeds 10MB limit'}), 400
        
        # Allowed file types
        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.xls', '.xlsx', '.csv'}
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(allowed_extensions)}'}), 400
        
        # Determine content type
        content_type_map = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.zip': 'application/zip',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.csv': 'text/csv',
        }
        content_type = content_type_map.get(file_ext, 'application/octet-stream')
        
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads', 'tickets', str(ticket_id))
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save file
        file.save(file_path)
        
        # Store relative path in database
        relative_path = f"uploads/tickets/{ticket_id}/{unique_filename}"

        # Check if this is an internal attachment (admin-only)
        # Get is_internal from form data (multipart/form-data)
        is_internal = False
        if is_admin:
            is_internal_str = request.form.get('is_internal', 'false')
            is_internal = is_internal_str.lower() == 'true'
        
        # Create attachment record
        attachment = TicketAttachment(
            ticket_id=ticket.id,
            file_name=filename,
            file_path=relative_path,
            file_size=file_size,
            content_type=content_type,
            uploaded_by=db_user.id,
            is_internal=is_internal
        )
        
        db.session.add(attachment)
        db.session.commit()
        
        logger.info(f"Attachment uploaded to ticket {ticket_id} by {user_email}: {filename} (is_internal: {is_internal})")
        logger.debug(f"Attachment details: {attachment.to_dict()}")
        
        return jsonify({
            'message': 'Attachment uploaded successfully',
            'attachment': attachment.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error uploading attachment to ticket {ticket_id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to upload attachment'}), 500


@support_bp.route('/tickets/<int:ticket_id>/attachments', methods=['GET'])
def get_ticket_attachments(ticket_id):
    """Get all attachments for a ticket"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = user.get('email')
        db_user = User.query.filter_by(email=user_email).first()
        if not db_user:
            return jsonify({'error': 'User not found'}), 404
        
        ticket = SupportTicket.query.get_or_404(ticket_id)
        
        # Check permissions
        is_admin = db_user.role in ['admin', 'owner'] or user_email in ['vinit@adeptaipro.com', 'abhi@adeptaipro.com', 'rushikesh@adeptaipro.com', 'contact@kempian.ai']
        is_owner = ticket.user_id == db_user.id
        
        if not (is_admin or is_owner):
            return jsonify({'error': 'You can only view attachments for your own tickets'}), 403
        
        # Get attachments (filter out internal ones for non-admins)
        if is_admin:
            attachments = TicketAttachment.query.filter_by(ticket_id=ticket_id).all()
        else:
            attachments = TicketAttachment.query.filter_by(ticket_id=ticket_id, is_internal=False).all()
        
        logger.info(f"Retrieved {len(attachments)} attachments for ticket {ticket_id} (admin: {is_admin}, owner: {is_owner})")
        
        attachments_data = [att.to_dict() for att in attachments]
        logger.debug(f"Attachments data: {attachments_data}")
        
        return jsonify({
            'attachments': attachments_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving attachments for ticket {ticket_id}: {e}")
        return jsonify({'error': 'Failed to retrieve attachments'}), 500


@support_bp.route('/tickets/<int:ticket_id>/attachments/<int:attachment_id>', methods=['DELETE'])
def delete_attachment(ticket_id, attachment_id):
    """Delete an attachment (Admin or uploader only)"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = user.get('email')
        db_user = User.query.filter_by(email=user_email).first()
        if not db_user:
            return jsonify({'error': 'User not found'}), 404
        
        ticket = SupportTicket.query.get_or_404(ticket_id)
        attachment = TicketAttachment.query.get_or_404(attachment_id)
        
        if attachment.ticket_id != ticket_id:
            return jsonify({'error': 'Attachment does not belong to this ticket'}), 400
        
        # Check permissions
        is_admin = db_user.role in ['admin', 'owner'] or user_email in ['vinit@adeptaipro.com', 'abhi@adeptaipro.com', 'rushikesh@adeptaipro.com', 'contact@kempian.ai']
        is_uploader = attachment.uploaded_by == db_user.id
        
        if not (is_admin or is_uploader):
            return jsonify({'error': 'You can only delete your own attachments'}), 403
        
        # Delete file from filesystem
        try:
            full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), attachment.file_path)
            if os.path.exists(full_path):
                os.remove(full_path)
        except Exception as e:
            logger.warning(f"Could not delete file {attachment.file_path}: {e}")
        
        # Delete from database
        db.session.delete(attachment)
        db.session.commit()
        
        logger.info(f"Attachment {attachment_id} deleted from ticket {ticket_id} by {user_email}")
        
        return jsonify({
            'message': 'Attachment deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting attachment {attachment_id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete attachment'}), 500


@support_bp.route('/tickets/<int:ticket_id>/attachments/<int:attachment_id>/download', methods=['GET'])
def download_attachment(ticket_id, attachment_id):
    """Download an attachment file"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = user.get('email')
        db_user = User.query.filter_by(email=user_email).first()
        if not db_user:
            return jsonify({'error': 'User not found'}), 404
        
        ticket = SupportTicket.query.get_or_404(ticket_id)
        attachment = TicketAttachment.query.get_or_404(attachment_id)
        
        if attachment.ticket_id != ticket_id:
            return jsonify({'error': 'Attachment does not belong to this ticket'}), 400
        
        # Check permissions
        is_admin = db_user.role in ['admin', 'owner'] or user_email in ['vinit@adeptaipro.com', 'abhi@adeptaipro.com', 'rushikesh@adeptaipro.com', 'contact@kempian.ai']
        is_owner = ticket.user_id == db_user.id
        
        # Admins can see all attachments, users can only see non-internal ones
        if not (is_admin or is_owner):
            return jsonify({'error': 'You can only download attachments from your own tickets'}), 403
        
        if attachment.is_internal and not is_admin:
            return jsonify({'error': 'You do not have permission to download this attachment'}), 403
        
        # Get full file path
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        full_path = os.path.join(base_dir, attachment.file_path)
        
        if not os.path.exists(full_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Send file
        return send_file(
            full_path,
            as_attachment=True,
            download_name=attachment.file_name,
            mimetype=attachment.content_type
        )
        
    except Exception as e:
        logger.error(f"Error downloading attachment {attachment_id}: {e}")
        return jsonify({'error': 'Failed to download attachment'}), 500


@support_bp.route('/tickets/<int:ticket_id>/attachments/<int:attachment_id>/view', methods=['GET'])
def view_attachment(ticket_id, attachment_id):
    """View an attachment file (without forcing download)"""
    try:
        user = get_current_user_flexible() or get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = user.get('email')
        db_user = User.query.filter_by(email=user_email).first()
        if not db_user:
            return jsonify({'error': 'User not found'}), 404
        
        ticket = SupportTicket.query.get_or_404(ticket_id)
        attachment = TicketAttachment.query.get_or_404(attachment_id)
        
        if attachment.ticket_id != ticket_id:
            return jsonify({'error': 'Attachment does not belong to this ticket'}), 400
        
        # Check permissions
        is_admin = db_user.role in ['admin', 'owner'] or user_email in ['vinit@adeptaipro.com', 'abhi@adeptaipro.com', 'rushikesh@adeptaipro.com', 'contact@kempian.ai']
        is_owner = ticket.user_id == db_user.id
        
        # Admins can see all attachments, users can only see non-internal ones
        if not (is_admin or is_owner):
            return jsonify({'error': 'You can only view attachments from your own tickets'}), 403
        
        if attachment.is_internal and not is_admin:
            return jsonify({'error': 'You do not have permission to view this attachment'}), 403
        
        # Get full file path
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        full_path = os.path.join(base_dir, attachment.file_path)
        
        if not os.path.exists(full_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Send file without forcing download (as_attachment=False)
        return send_file(
            full_path,
            as_attachment=False,
            download_name=attachment.file_name,
            mimetype=attachment.content_type
        )
        
    except Exception as e:
        logger.error(f"Error viewing attachment {attachment_id}: {e}")
        return jsonify({'error': 'Failed to view attachment'}), 500