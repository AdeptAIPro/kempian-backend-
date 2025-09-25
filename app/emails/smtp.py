"""
SMTP Email Service as fallback for AWS SES
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.simple_logger import get_logger
import os

logger = get_logger('emails')

def send_email_via_smtp(to_email, subject, body_html, body_text=None):
    """
    Send email via SMTP (Gmail, Hostinger, Outlook, etc.)
    """
    try:
        # SMTP Configuration - Update these with your email provider
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_username = os.getenv('SMTP_USERNAME', '')
        smtp_password = os.getenv('SMTP_PASSWORD', '')
        from_email = os.getenv('SMTP_FROM_EMAIL', smtp_username)
        
        if not smtp_username or not smtp_password:
            logger.error("[SMTP] SMTP credentials not configured")
            return False
        
        logger.info(f"[SMTP] Using server: {smtp_server}:{smtp_port}")
        logger.info(f"[SMTP] From: {from_email}, To: {to_email}")
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        
        # Add text and HTML parts
        if body_text:
            text_part = MIMEText(body_text, 'plain')
            msg.attach(text_part)
        
        html_part = MIMEText(body_html, 'html')
        msg.attach(html_part)
        
        # Create secure connection and send email
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"[SMTP] Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"[SMTP] Failed to send email: {str(e)}")
        return False

def send_application_status_email_smtp(to_email, candidate_name, job_title, company_name, job_location, applied_date, status):
    """
    Send application status email via SMTP
    """
    logger.info(f"[SMTP] Sending application status email to {to_email}")
    
    # Status configurations
    status_configs = {
        'hired': {
            'subject': f"Congratulations! You're hired for {job_title} at {company_name}",
            'color': '#28a745',
            'icon': '🎉'
        },
        'shortlisted': {
            'subject': f"Great news! You've been shortlisted for {job_title} at {company_name}",
            'color': '#17a2b8',
            'icon': '⭐'
        },
        'rejected': {
            'subject': f"Update on your application for {job_title} at {company_name}",
            'color': '#dc3545',
            'icon': '📝'
        },
        'reviewed': {
            'subject': f"Your application for {job_title} at {company_name} has been reviewed",
            'color': '#ffc107',
            'icon': '👀'
        }
    }
    
    config = status_configs.get(status, status_configs['reviewed'])
    
    # Format applied date
    if isinstance(applied_date, str):
        formatted_date = applied_date
    else:
        formatted_date = applied_date.strftime('%B %d, %Y')
    
    # Create HTML email
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{config['subject']}</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: {config['color']}; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;">
            <h1 style="margin: 0; font-size: 24px;">{config['icon']} {config['subject']}</h1>
        </div>
        
        <div style="background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; border: 1px solid #dee2e6;">
            <p style="font-size: 16px; margin-bottom: 20px;">Dear {candidate_name},</p>
            
            <p style="font-size: 16px; margin-bottom: 20px;">
                We're excited to inform you about the status of your job application.
            </p>
            
            <div style="background: white; padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 4px solid {config['color']};">
                <h3 style="margin-top: 0; color: {config['color']};">Application Details</h3>
                <p><strong>Position:</strong> {job_title}</p>
                <p><strong>Company:</strong> {company_name}</p>
                <p><strong>Location:</strong> {job_location}</p>
                <p><strong>Applied Date:</strong> {formatted_date}</p>
                <p><strong>Status:</strong> <span style="color: {config['color']}; font-weight: bold;">{status.title()}</span></p>
            </div>
            
            <p style="font-size: 16px; margin-bottom: 20px;">
                Thank you for your interest in joining our team. We appreciate the time and effort you put into your application.
            </p>
            
            <p style="font-size: 16px; margin-bottom: 0;">
                Best regards,<br>
                The {company_name} Team
            </p>
        </div>
    </body>
    </html>
    """
    
    # Create text version
    text_body = f"""
{config['subject']}

Dear {candidate_name},

We're excited to inform you about the status of your job application.

Application Details:
- Position: {job_title}
- Company: {company_name}
- Location: {job_location}
- Applied Date: {formatted_date}
- Status: {status.title()}

Thank you for your interest in joining our team. We appreciate the time and effort you put into your application.

Best regards,
The {company_name} Team
    """
    
    return send_email_via_smtp(to_email, config['subject'], html_body, text_body)

def send_application_confirmation_email_smtp(to_email, candidate_name, job_title, company_name, job_location, applied_date):
    """
    Send application confirmation email via SMTP
    """
    logger.info(f"[SMTP] Sending application confirmation email to {to_email}")
    
    # Format applied date
    if isinstance(applied_date, str):
        formatted_date = applied_date
    else:
        formatted_date = applied_date.strftime('%B %d, %Y at %I:%M %p')
    
    # Create subject
    subject = f"Thank you for applying to {job_title} at {company_name}"
    
    # Create HTML email
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{subject}</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: #28a745; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;">
            <h1 style="margin: 0; font-size: 24px;">🎉 Application Received!</h1>
        </div>
        
        <div style="background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; border: 1px solid #dee2e6;">
            <p style="font-size: 16px; margin-bottom: 20px;">Dear {candidate_name},</p>
            
            <p style="font-size: 16px; margin-bottom: 20px;">
                Thank you for your interest in joining our team! We have successfully received your application.
            </p>
            
            <div style="background: white; padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 4px solid #28a745;">
                <h3 style="margin-top: 0; color: #28a745;">Application Details</h3>
                <p><strong>Position:</strong> {job_title}</p>
                <p><strong>Company:</strong> {company_name}</p>
                <p><strong>Location:</strong> {job_location}</p>
                <p><strong>Applied Date:</strong> {formatted_date}</p>
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 6px; margin: 20px 0;">
                <h4 style="margin-top: 0; color: #1976d2;">What happens next?</h4>
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li>Our team will review your application</li>
                    <li>We'll contact you within 5-7 business days</li>
                    <li>If shortlisted, we'll schedule an interview</li>
                </ul>
            </div>
            
            <p style="font-size: 16px; margin-bottom: 20px;">
                We appreciate the time and effort you put into your application. We look forward to learning more about your qualifications and experience.
            </p>
            
            <p style="font-size: 16px; margin-bottom: 0;">
                Best regards,<br>
                The {company_name} Team
            </p>
        </div>
    </body>
    </html>
    """
    
    # Create text version
    text_body = f"""
{subject}

Dear {candidate_name},

Thank you for your interest in joining our team! We have successfully received your application.

Application Details:
- Position: {job_title}
- Company: {company_name}
- Location: {job_location}
- Applied Date: {formatted_date}

What happens next?
- Our team will review your application
- We'll contact you within 5-7 business days
- If shortlisted, we'll schedule an interview

We appreciate the time and effort you put into your application. We look forward to learning more about your qualifications and experience.

Best regards,
The {company_name} Team
    """
    
    return send_email_via_smtp(to_email, subject, html_body, text_body)

def send_interview_invitation_email_smtp(to_email, candidate_name, job_title, company_name, job_location, interview_date, meeting_link, meeting_type, interviewer_name, interview_notes):
    """
    Send interview invitation email via SMTP
    """
    logger.info(f"[SMTP] Sending interview invitation email to {to_email}")
    
    # Format interview date and time
    if isinstance(interview_date, str):
        formatted_date = interview_date
    else:
        formatted_date = interview_date.strftime('%B %d, %Y at %I:%M %p')
    
    # Get meeting type display name
    meeting_type_names = {
        'zoom': 'Zoom',
        'google_meet': 'Google Meet',
        'teams': 'Microsoft Teams',
        'other': 'Video Call'
    }
    meeting_type_display = meeting_type_names.get(meeting_type, meeting_type.title())
    
    # Create subject
    subject = f"Interview Invitation - {job_title} at {company_name}"
    
    # Create HTML email
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{subject}</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: #1976d2; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;">
            <h1 style="margin: 0; font-size: 24px;">🎯 Interview Invitation</h1>
        </div>
        
        <div style="background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; border: 1px solid #dee2e6;">
            <p style="font-size: 16px; margin-bottom: 20px;">Dear {candidate_name},</p>
            
            <p style="font-size: 16px; margin-bottom: 20px;">
                Congratulations! We are pleased to invite you for an interview for the position of <strong>{job_title}</strong> at {company_name}.
            </p>
            
            <div style="background: white; padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 4px solid #1976d2;">
                <h3 style="margin-top: 0; color: #1976d2;">Interview Details</h3>
                <p><strong>Position:</strong> {job_title}</p>
                <p><strong>Company:</strong> {company_name}</p>
                <p><strong>Location:</strong> {job_location}</p>
                <p><strong>Date & Time:</strong> {formatted_date}</p>
                <p><strong>Platform:</strong> {meeting_type_display}</p>
                <p><strong>Interviewer:</strong> {interviewer_name}</p>
            </div>
            
            <div style="background: #e8f5e8; padding: 15px; border-radius: 6px; margin: 20px 0;">
                <h4 style="margin-top: 0; color: #2e7d32;">Meeting Link</h4>
                <p style="margin: 10px 0;">
                    <a href="{meeting_link}" style="color: #1976d2; text-decoration: none; font-weight: bold; word-break: break-all;">
                        {meeting_link}
                    </a>
                </p>
                <p style="font-size: 14px; color: #666; margin: 10px 0 0 0;">
                    Please click the link above to join the interview at the scheduled time.
                </p>
            </div>
            
            {f'<div style="background: #fff3e0; padding: 15px; border-radius: 6px; margin: 20px 0;"><h4 style="margin-top: 0; color: #f57c00;">Additional Notes</h4><p style="margin: 10px 0;">{interview_notes}</p></div>' if interview_notes else ''}
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 6px; margin: 20px 0;">
                <h4 style="margin-top: 0; color: #1976d2;">What to expect?</h4>
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li>Technical discussion about the role</li>
                    <li>Questions about your experience and skills</li>
                    <li>Opportunity to ask questions about the company</li>
                    <li>Duration: Approximately 45-60 minutes</li>
                </ul>
            </div>
            
            <p style="font-size: 16px; margin-bottom: 20px;">
                Please confirm your attendance by replying to this email. If you have any questions or need to reschedule, please let us know as soon as possible.
            </p>
            
            <p style="font-size: 16px; margin-bottom: 0;">
                We look forward to speaking with you!<br>
                Best regards,<br>
                The {company_name} Team
            </p>
        </div>
    </body>
    </html>
    """
    
    # Create text version
    text_body = f"""
{subject}

Dear {candidate_name},

Congratulations! We are pleased to invite you for an interview for the position of {job_title} at {company_name}.

Interview Details:
- Position: {job_title}
- Company: {company_name}
- Location: {job_location}
- Date & Time: {formatted_date}
- Platform: {meeting_type_display}
- Interviewer: {interviewer_name}

Meeting Link: {meeting_link}

{f'Additional Notes: {interview_notes}' if interview_notes else ''}

What to expect?
- Technical discussion about the role
- Questions about your experience and skills
- Opportunity to ask questions about the company
- Duration: Approximately 45-60 minutes

Please confirm your attendance by replying to this email. If you have any questions or need to reschedule, please let us know as soon as possible.

We look forward to speaking with you!

Best regards,
The {company_name} Team
    """
    
    return send_email_via_smtp(to_email, subject, html_body, text_body)
