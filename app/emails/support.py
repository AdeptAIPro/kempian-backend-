"""
Email service for support tickets

This module contains email functions for support ticket notifications.
Currently, these functions are placeholders and will be implemented later.

Admin Email Recipients:
- vinit@adeptaipro.com
- abhi@adeptaipro.com
- contact@kempian.ai
"""

from app.simple_logger import get_logger

logger = get_logger("support_emails")

# Admin email recipients
ADMIN_EMAILS = [
    'vinit@adeptaipro.com',
    'abhi@adeptaipro.com',
    'contact@kempian.ai'
]


def send_support_ticket_notification_email(ticket):
    """
    Send email notification to admins when a new support ticket is created.
    
    Args:
        ticket: SupportTicket model instance
        
    TODO: Implement email sending functionality
    - Send to all admin emails in ADMIN_EMAILS
    - Subject: "New Support Ticket: [Ticket ID] - [Subject]"
    - Include ticket details, user info, message
    - Include link to admin support page
    """
    logger.info(f"TODO: Send support ticket notification email for ticket {ticket.id}")
    logger.info(f"Would send to: {', '.join(ADMIN_EMAILS)}")
    logger.info(f"Ticket details: ID={ticket.id}, User={ticket.user_email}, Subject={ticket.subject}")
    
    # TODO: Implement email sending
    # Example structure:
    # from app.emails.ses import send_email  # or smtp/sendgrid
    # for admin_email in ADMIN_EMAILS:
    #     send_email(
    #         to=admin_email,
    #         subject=f"New Support Ticket: #{ticket.id} - {ticket.subject}",
    #         html_body=render_template('support_ticket_notification.html', ticket=ticket)
    #     )
    
    pass


def send_support_ticket_reply_email(ticket):
    """
    Send email to user when admin replies to their support ticket.
    
    Args:
        ticket: SupportTicket model instance with admin_reply populated
        
    TODO: Implement email sending functionality
    - Send to ticket creator's email (ticket.user_email)
    - Subject: "Re: Your Support Ticket - [Ticket ID]"
    - Include admin's reply and next steps
    - Include ticket reference
    """
    logger.info(f"TODO: Send support ticket reply email for ticket {ticket.id}")
    logger.info(f"Would send to: {ticket.user_email}")
    logger.info(f"Reply from: {ticket.replier.email if ticket.replier else 'Unknown'}")
    
    # TODO: Implement email sending
    # Example structure:
    # from app.emails.ses import send_email  # or smtp/sendgrid
    # send_email(
    #     to=ticket.user_email,
    #     subject=f"Re: Your Support Ticket - #{ticket.id}",
    #     html_body=render_template('support_ticket_reply.html', ticket=ticket)
    # )
    
    pass

