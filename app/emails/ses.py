# import boto3
# import os
# from flask import render_template_string
# from app.simple_logger import get_logger

# SES_REGION = os.getenv('SES_REGION')
# SES_FROM_EMAIL = os.getenv('SES_FROM_EMAIL')

# # Initialize SES client with error handling
# def get_ses_client():
#     """Get SES client with proper error handling"""
#     try:
#         if not SES_REGION or not SES_FROM_EMAIL:
#             get_logger('emails').error("SES configuration missing: SES_REGION or SES_FROM_EMAIL not set")
#             return None
        
#         return boto3.client('ses', region_name=SES_REGION)
#     except Exception as e:
#         get_logger('emails').error(f"Failed to initialize SES client: {str(e)}")
#         return None

# ses_client = get_ses_client()

# TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')

# def load_template(filename):
#     with open(os.path.join(TEMPLATES_DIR, filename), 'r') as f:
#         return f.read()

# def send_invite_email(to_email, invite_link):
#     """Send invite email to candidate"""
#     try:
#         client = get_ses_client()
#         if not client:
#             get_logger('emails').error("SES client not available for invite email")
#             return False
            
#     subject = "You're invited to Talent-Match!"
#     template = load_template('invite.html')
#     body_html = render_template_string(template, invite_link=invite_link)
        
#         client.send_email(
#         Source=SES_FROM_EMAIL,
#         Destination={'ToAddresses': [to_email]},
#         Message={
#             'Subject': {'Data': subject},
#             'Body': {'Html': {'Data': body_html}}
#         }
#     )
        
#         get_logger('emails').info(f"Invite email sent to {to_email}")
#         return True
        
#     except Exception as e:
#         get_logger('emails').error(f"Failed to send invite email: {str(e)}")
#         return False

# def send_quota_alert_email(to_email, percent):
#     """Send quota alert email"""
#     try:
#         client = get_ses_client()
#         if not client:
#             get_logger('emails').error("SES client not available for quota alert email")
#             return False
            
#     subject = f"Talent-Match: {percent}% Quota Used"
#     template = load_template('quota_alert.html')
#     body_html = render_template_string(template, percent=percent)
        
#         client.send_email(
#         Source=SES_FROM_EMAIL,
#         Destination={'ToAddresses': [to_email]},
#         Message={
#             'Subject': {'Data': subject},
#             'Body': {'Html': {'Data': body_html}}
#         }
#     )
        
#         get_logger('emails').info(f"Quota alert email sent to {to_email}")
#         return True
        
#     except Exception as e:
#         get_logger('emails').error(f"Failed to send quota alert email: {str(e)}")
#         return False

# def send_application_status_email(to_email, candidate_name, job_title, company_name, job_location, applied_date, status):
#     """Send application status update email to candidate"""
#     logger = get_logger('emails')
#     logger.info(f"[EMAIL] send_application_status_email called with:")
#     logger.info(f"   - to_email: {to_email}")
#     logger.info(f"   - candidate_name: {candidate_name}")
#     logger.info(f"   - job_title: {job_title}")
#     logger.info(f"   - company_name: {company_name}")
#     logger.info(f"   - job_location: {job_location}")
#     logger.info(f"   - applied_date: {applied_date}")
#     logger.info(f"   - status: {status}")
    
#     # Try SMTP first (primary method)
#     logger.info(f"[SMTP] Attempting to send email via SMTP (primary method)...")
#     try:
#         from .smtp import send_application_status_email_smtp
#         smtp_result = send_application_status_email_smtp(
#             to_email, candidate_name, job_title, company_name, 
#             job_location, applied_date, status
#         )
#         if smtp_result:
#             logger.info(f"[SUCCESS] Email sent successfully via SMTP to {to_email}")
#             return True
#         else:
#             logger.warning(f"[SMTP_FAILED] SMTP failed, trying AWS SES fallback...")
#     except Exception as smtp_error:
#         logger.warning(f"[SMTP_ERROR] SMTP failed: {str(smtp_error)}, trying AWS SES fallback...")
    
#     # Fallback to AWS SES
#     logger.info(f"[SES] Attempting to send email via AWS SES (fallback method)...")
#     try:
#         client = get_ses_client()
#         if not client:
#             logger.error("SES client not available for application status email")
#             return False
            
#         # Map status to template and subject
#         status_config = {
#             'reviewed': {
#                 'template': 'application_reviewed.html',
#                 'subject': f'Application Update: {job_title} at {company_name}'
#             },
#             'shortlisted': {
#                 'template': 'application_shortlisted.html',
#                 'subject': f'Congratulations! You\'ve been shortlisted for {job_title} at {company_name}'
#             },
#             'rejected': {
#                 'template': 'application_rejected.html',
#                 'subject': f'Application Update: {job_title} at {company_name}'
#             },
#             'hired': {
#                 'template': 'application_hired.html',
#                 'subject': f'Congratulations! You\'re hired for {job_title} at {company_name}'
#             }
#         }
        
#         if status not in status_config:
#             get_logger('emails').error(f"Unknown status for email: {status}")
#             return False
            
#         config = status_config[status]
#         template = load_template(config['template'])
        
#         # Format the applied date
#         from datetime import datetime
#         if isinstance(applied_date, str):
#             try:
#                 applied_date_obj = datetime.fromisoformat(applied_date.replace('Z', '+00:00'))
#                 formatted_date = applied_date_obj.strftime('%B %d, %Y')
#             except:
#                 formatted_date = applied_date
#         else:
#             formatted_date = applied_date.strftime('%B %d, %Y') if applied_date else 'N/A'
        
#         body_html = render_template_string(template, 
#             candidate_name=candidate_name,
#             job_title=job_title,
#             company_name=company_name,
#             job_location=job_location,
#             applied_date=formatted_date
#         )
        
#         logger.info(f"[SES] Sending email via SES...")
#         logger.info(f"   - From: {SES_FROM_EMAIL}")
#         logger.info(f"   - To: {to_email}")
#         logger.info(f"   - Subject: {config['subject']}")
        
#         client.send_email(
#             Source=SES_FROM_EMAIL,
#             Destination={'ToAddresses': [to_email]},
#             Message={
#                 'Subject': {'Data': config['subject']},
#                 'Body': {'Html': {'Data': body_html}}
#             }
#         )
        
#         logger.info(f"[SUCCESS] Application status email sent successfully to {to_email} for status: {status}")
#         return True
        
#     except Exception as e:
#         error_msg = str(e)
#         logger.error(f"[ERROR] Both SMTP and AWS SES failed to send email to {to_email}")
#         logger.error(f"[SES_ERROR] AWS SES error: {error_msg}")
        
#         # Log specific SES errors for debugging
#         if "MessageRejected" in error_msg and "not verified" in error_msg:
#             logger.error(f"[SES_ERROR] Email address {to_email} is not verified in AWS SES.")
#         elif "MessageRejected" in error_msg and "sandbox" in error_msg.lower():
#             logger.error(f"[SES_ERROR] AWS SES is in sandbox mode. Only verified email addresses can receive emails.")
#         elif "InvalidParameterValue" in error_msg:
#             logger.error(f"[SES_ERROR] Invalid email parameters. Check email format and content.")
#         else:
#             import traceback
#             logger.error(f"[TRACEBACK] Full traceback: {traceback.format_exc()}")
        
#         return False 

# def send_application_confirmation_email(to_email, candidate_name, job_title, company_name, job_location, applied_date):
#     """Send application confirmation email to candidate"""
#     logger = get_logger('emails')
#     logger.info(f"[EMAIL] send_application_confirmation_email called with:")
#     logger.info(f"   - to_email: {to_email}")
#     logger.info(f"   - candidate_name: {candidate_name}")
#     logger.info(f"   - job_title: {job_title}")
#     logger.info(f"   - company_name: {company_name}")
#     logger.info(f"   - job_location: {job_location}")
#     logger.info(f"   - applied_date: {applied_date}")
    
#     # Try SMTP first (primary method)
#     logger.info(f"[SMTP] Attempting to send application confirmation via SMTP (primary method)...")
#     try:
#         from .smtp import send_application_confirmation_email_smtp
#         smtp_result = send_application_confirmation_email_smtp(
#             to_email, candidate_name, job_title, company_name, 
#             job_location, applied_date
#         )
#         if smtp_result:
#             logger.info(f"[SUCCESS] Application confirmation email sent successfully via SMTP to {to_email}")
#             return True
#         else:
#             logger.warning(f"[SMTP_FAILED] SMTP failed, trying AWS SES fallback...")
#     except Exception as smtp_error:
#         logger.warning(f"[SMTP_ERROR] SMTP failed: {str(smtp_error)}, trying AWS SES fallback...")
    
#     # Fallback to AWS SES
#     logger.info(f"[SES] Attempting to send application confirmation via AWS SES (fallback method)...")
#     try:
#         client = get_ses_client()
#         if not client:
#             logger.error("SES client not available for application confirmation email")
#             return False
            
#         # Format applied date
#         if isinstance(applied_date, str):
#             formatted_date = applied_date
#         else:
#             formatted_date = applied_date.strftime('%B %d, %Y at %I:%M %p')
        
#         # Create HTML email content
#         subject = f"Thank you for applying to {job_title} at {company_name}"
        
#         html_body = f"""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <meta charset="UTF-8">
#             <title>{subject}</title>
#         </head>
#         <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
#             <div style="background: #28a745; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;">
#                 <h1 style="margin: 0; font-size: 24px;">🎉 Application Received!</h1>
#             </div>
            
#             <div style="background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; border: 1px solid #dee2e6;">
#                 <p style="font-size: 16px; margin-bottom: 20px;">Dear {candidate_name},</p>
                
#                 <p style="font-size: 16px; margin-bottom: 20px;">
#                     Thank you for your interest in joining our team! We have successfully received your application.
#                 </p>
                
#                 <div style="background: white; padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 4px solid #28a745;">
#                     <h3 style="margin-top: 0; color: #28a745;">Application Details</h3>
#                     <p><strong>Position:</strong> {job_title}</p>
#                     <p><strong>Company:</strong> {company_name}</p>
#                     <p><strong>Location:</strong> {job_location}</p>
#                     <p><strong>Applied Date:</strong> {formatted_date}</p>
#                 </div>
                
#                 <div style="background: #e3f2fd; padding: 15px; border-radius: 6px; margin: 20px 0;">
#                     <h4 style="margin-top: 0; color: #1976d2;">What happens next?</h4>
#                     <ul style="margin: 10px 0; padding-left: 20px;">
#                         <li>Our team will review your application</li>
#                         <li>We'll contact you within 5-7 business days</li>
#                         <li>If shortlisted, we'll schedule an interview</li>
#                     </ul>
#                 </div>
                
#                 <p style="font-size: 16px; margin-bottom: 20px;">
#                     We appreciate the time and effort you put into your application. We look forward to learning more about your qualifications and experience.
#                 </p>
                
#                 <p style="font-size: 16px; margin-bottom: 0;">
#                     Best regards,<br>
#                     The {company_name} Team
#                 </p>
#             </div>
#         </body>
#         </html>
#         """
        
#         # Create text version
#         text_body = f"""
# {subject}

# Dear {candidate_name},

# Thank you for your interest in joining our team! We have successfully received your application.

# Application Details:
# - Position: {job_title}
# - Company: {company_name}
# - Location: {job_location}
# - Applied Date: {formatted_date}

# What happens next?
# - Our team will review your application
# - We'll contact you within 5-7 business days
# - If shortlisted, we'll schedule an interview

# We appreciate the time and effort you put into your application. We look forward to learning more about your qualifications and experience.

# Best regards,
# The {company_name} Team
#         """
        
#         logger.info(f"[SES] Sending application confirmation email via SES...")
#         logger.info(f"   - From: {SES_FROM_EMAIL}")
#         logger.info(f"   - To: {to_email}")
#         logger.info(f"   - Subject: {subject}")
        
#         client.send_email(
#             Source=SES_FROM_EMAIL,
#             Destination={'ToAddresses': [to_email]},
#             Message={
#                 'Subject': {'Data': subject},
#                 'Body': {
#                     'Html': {'Data': html_body},
#                     'Text': {'Data': text_body}
#                 }
#             }
#         )
        
#         logger.info(f"[SUCCESS] Application confirmation email sent successfully to {to_email}")
#         return True
        
#     except Exception as e:
#         error_msg = str(e)
#         logger.error(f"[ERROR] Both SMTP and AWS SES failed to send application confirmation email to {to_email}")
#         logger.error(f"[SES_ERROR] AWS SES error: {error_msg}")
        
#         # Log specific SES errors for debugging
#         if "MessageRejected" in error_msg and "not verified" in error_msg:
#             logger.error(f"[SES_ERROR] Email address {to_email} is not verified in AWS SES.")
#         elif "MessageRejected" in error_msg and "sandbox" in error_msg.lower():
#             logger.error(f"[SES_ERROR] AWS SES is in sandbox mode. Only verified email addresses can receive emails.")
#         elif "InvalidParameterValue" in error_msg:
#             logger.error(f"[SES_ERROR] Invalid email parameters. Check email format and content.")
#         else:
#             import traceback
#             logger.error(f"[TRACEBACK] Full traceback: {traceback.format_exc()}")
        
#         return False 
import boto3
import os
from typing import Optional, List
from flask import render_template_string
from app.simple_logger import get_logger

SES_REGION = os.getenv('SES_REGION')
SES_FROM_EMAIL = os.getenv('SES_FROM_EMAIL')

# Initialize SES client with error handling
def get_ses_client():
    """Get SES client with proper error handling"""
    try:
        if not SES_REGION or not SES_FROM_EMAIL:
            get_logger('emails').error("SES configuration missing: SES_REGION or SES_FROM_EMAIL not set")
            return None
        
        return boto3.client('ses', region_name=SES_REGION)
    except Exception as e:
        get_logger('emails').error(f"Failed to initialize SES client: {str(e)}")
        return None

ses_client = get_ses_client()

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')

def load_template(filename):
    with open(os.path.join(TEMPLATES_DIR, filename), 'r') as f:
        return f.read()

def _render_simple_html(title: str, body_lines: List[str]) -> str:
    lines = ''.join(f"<p style=\"margin:0 0 12px\">{line}</p>" for line in body_lines)
    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset=\"UTF-8\"><title>{title}</title></head>
    <body style=\"font-family: Arial, sans-serif; line-height:1.6; color:#333; max-width:600px; margin:0 auto; padding:20px;\">
      <div style=\"background:#2563eb; color:white; padding:16px 20px; border-radius:8px 8px 0 0;\">
        <h1 style=\"margin:0; font-size:20px;\">{title}</h1>
      </div>
      <div style=\"background:#f8fafc; padding:24px; border:1px solid #e2e8f0; border-top:0; border-radius:0 0 8px 8px;\">
        {lines}
      </div>
    </body>
    </html>
    """

def send_welcome_email(to_email: str, first_name: Optional[str] = None) -> bool:
    """Send a simple welcome/thanks-for-signup email.

    Returns True on success, False otherwise.
    """
    logger = get_logger('emails')
    subject = "Welcome to Kempian!"
    greeting_name = (first_name or '').strip() or 'there'
    html_body = _render_simple_html(
        subject,
        [
            f"Hi {greeting_name},",
            "Thanks for signing up with Kempian. We're excited to have you onboard!",
            "You can log in anytime to explore features and get started.",
            "If you have any questions, just reply to this email.",
            "— The Kempian Team",
        ],
    )

    text_body = (
        f"{subject}\n\n"
        f"Hi {greeting_name},\n\n"
        "Thanks for signing up with Kempian. We're excited to have you onboard!\n"
        "You can log in anytime to explore features and get started.\n"
        "If you have any questions, just reply to this email.\n\n"
        "— The Kempian Team\n"
    )

    # Try SMTP first
    try:
        from .smtp import send_generic_email_smtp  # optional helper if available
        try:
            smtp_ok = send_generic_email_smtp(to_email, subject, html_body, text_body)
        except Exception:
            smtp_ok = False
        if smtp_ok:
            logger.info(f"[SUCCESS] Welcome email sent via SMTP to {to_email}")
            return True
    except Exception:
        # SMTP helper not available; proceed to SES
        pass

    # Fallback to SES
    try:
        client = get_ses_client()
        if not client:
            logger.error("SES client not available for welcome email")
            return False

        client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {
                    'Html': {'Data': html_body},
                    'Text': {'Data': text_body}
                }
            }
        )
        logger.info(f"[SUCCESS] Welcome email sent via SES to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send welcome email to {to_email}: {e}")
        return False

def send_invite_email(to_email, invite_link):
    """Send invite email - tries SMTP (Hostinger) first, then falls back to SES"""
    logger = get_logger('emails')
    
    subject = "You're invited to Kempian!"
    
    # Load email template
    try:
        template = load_template('invite.html')
        body_html = render_template_string(template, invite_link=invite_link)
    except Exception as template_error:
        logger.warning(f"Failed to load invite template, using simple HTML: {template_error}")
        # Fallback simple HTML template
        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"><title>{subject}</title></head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: #2563eb; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;">
                <h1 style="margin: 0; font-size: 24px;">You're Invited!</h1>
            </div>
            <div style="background: #f8fafc; padding: 30px; border: 1px solid #e2e8f0; border-top: 0; border-radius: 0 0 8px 8px;">
                <p style="font-size: 16px; margin-bottom: 20px;">You've been invited to join Kempian!</p>
                <p style="font-size: 16px; margin-bottom: 20px;">Click the link below to complete your registration:</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{invite_link}" style="background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: bold;">Complete Registration</a>
                </div>
                <p style="font-size: 14px; color: #666; margin-top: 30px;">Or copy and paste this link into your browser:</p>
                <p style="font-size: 12px; color: #999; word-break: break-all;">{invite_link}</p>
            </div>
        </body>
        </html>
        """
    
    # Create plain text version
    body_text = f"""
You're Invited to Kempian!

You've been invited to join Kempian!

Click the link below to complete your registration:
{invite_link}

If you have any questions, please contact support.

Best regards,
The Kempian Team
    """
    
    # Try SMTP first (Hostinger or configured SMTP)
    logger.info(f"[SMTP] Attempting to send invite email via SMTP (primary method)...")
    try:
        from .smtp import send_email_via_smtp
        smtp_result = send_email_via_smtp(to_email, subject, body_html, body_text)
        if smtp_result:
            logger.info(f"[SUCCESS] Invite email sent successfully via SMTP to {to_email}")
            return True
        else:
            logger.warning(f"[SMTP_FAILED] SMTP failed, trying AWS SES fallback...")
    except Exception as smtp_error:
        logger.warning(f"[SMTP_ERROR] SMTP failed: {str(smtp_error)}, trying AWS SES fallback...")
    
    # Fallback to AWS SES
    logger.info(f"[SES] Attempting to send invite email via AWS SES (fallback method)...")
    try:
        # Check SES configuration
        if not SES_REGION or not SES_FROM_EMAIL:
            logger.error(f"SES configuration missing: SES_REGION={SES_REGION}, SES_FROM_EMAIL={SES_FROM_EMAIL}")
            return False
            
        client = get_ses_client()
        if not client:
            logger.error("SES client not available for invite email")
            return False
        
        logger.info(f"Attempting to send invite email to {to_email} from {SES_FROM_EMAIL}")
        
        response = client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {
                    'Html': {'Data': body_html},
                    'Text': {'Data': body_text}
                }
            }
        )
        
        logger.info(f"[SUCCESS] Invite email sent successfully via SES to {to_email}. MessageId: {response.get('MessageId', 'N/A')}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ERROR] Both SMTP and AWS SES failed to send invite email to {to_email}")
        logger.error(f"[SES_ERROR] AWS SES error: {error_msg}")
        # Log more details for debugging
        if "InvalidParameterValue" in error_msg:
            logger.error(f"SES InvalidParameterValue - Check email format: {to_email}")
        elif "MessageRejected" in error_msg:
            logger.error(f"SES MessageRejected - Email might be in spam or SES sandbox mode")
        elif "Throttling" in error_msg or "Rate exceeded" in error_msg:
            logger.error(f"SES Rate limit exceeded - Too many emails sent")
        return False

def send_quota_alert_email(to_email, percent):
    """Send quota alert email"""
    try:
        client = get_ses_client()
        if not client:
            get_logger('emails').error("SES client not available for quota alert email")
            return False
            
        subject = f"Talent-Match: {percent}% Quota Used"
        template = load_template('quota_alert.html')
        body_html = render_template_string(template, percent=percent)
        
        client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {'Html': {'Data': body_html}}
            }
        )
        
        get_logger('emails').info(f"Quota alert email sent to {to_email}")
        return True
        
    except Exception as e:
        get_logger('emails').error(f"Failed to send quota alert email: {str(e)}")
        return False

def send_application_status_email(to_email, candidate_name, job_title, company_name, job_location, applied_date, status):
    """Send application status update email to candidate"""
    logger = get_logger('emails')
    logger.info(f"[EMAIL] send_application_status_email called with:")
    logger.info(f"   - to_email: {to_email}")
    logger.info(f"   - candidate_name: {candidate_name}")
    logger.info(f"   - job_title: {job_title}")
    logger.info(f"   - company_name: {company_name}")
    logger.info(f"   - job_location: {job_location}")
    logger.info(f"   - applied_date: {applied_date}")
    logger.info(f"   - status: {status}")
    
    # Try SMTP first (primary method)
    logger.info(f"[SMTP] Attempting to send email via SMTP (primary method)...")
    try:
        from .smtp import send_application_status_email_smtp
        smtp_result = send_application_status_email_smtp(
            to_email, candidate_name, job_title, company_name, 
            job_location, applied_date, status
        )
        if smtp_result:
            logger.info(f"[SUCCESS] Email sent successfully via SMTP to {to_email}")
            return True
        else:
            logger.warning(f"[SMTP_FAILED] SMTP failed, trying AWS SES fallback...")
    except Exception as smtp_error:
        logger.warning(f"[SMTP_ERROR] SMTP failed: {str(smtp_error)}, trying AWS SES fallback...")
    
    # Fallback to AWS SES
    logger.info(f"[SES] Attempting to send email via AWS SES (fallback method)...")
    try:
        client = get_ses_client()
        if not client:
            logger.error("SES client not available for application status email")
            return False
            
        # Map status to template and subject
        status_config = {
            'reviewed': {
                'template': 'application_reviewed.html',
                'subject': f'Application Update: {job_title} at {company_name}'
            },
            'shortlisted': {
                'template': 'application_shortlisted.html',
                'subject': f'Congratulations! You\'ve been shortlisted for {job_title} at {company_name}'
            },
            'rejected': {
                'template': 'application_rejected.html',
                'subject': f'Application Update: {job_title} at {company_name}'
            },
            'hired': {
                'template': 'application_hired.html',
                'subject': f'Congratulations! You\'re hired for {job_title} at {company_name}'
            }
        }
        
        if status not in status_config:
            get_logger('emails').error(f"Unknown status for email: {status}")
            return False
            
        config = status_config[status]
        template = load_template(config['template'])
        
        # Format the applied date
        from datetime import datetime
        if isinstance(applied_date, str):
            try:
                applied_date_obj = datetime.fromisoformat(applied_date.replace('Z', '+00:00'))
                formatted_date = applied_date_obj.strftime('%B %d, %Y')
            except:
                formatted_date = applied_date
        else:
            formatted_date = applied_date.strftime('%B %d, %Y') if applied_date else 'N/A'
        
        body_html = render_template_string(template, 
            candidate_name=candidate_name,
            job_title=job_title,
            company_name=company_name,
            job_location=job_location,
            applied_date=formatted_date
        )
        
        logger.info(f"[SES] Sending email via SES...")
        logger.info(f"   - From: {SES_FROM_EMAIL}")
        logger.info(f"   - To: {to_email}")
        logger.info(f"   - Subject: {config['subject']}")
        
        client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': config['subject']},
                'Body': {'Html': {'Data': body_html}}
            }
        )
        
        logger.info(f"[SUCCESS] Application status email sent successfully to {to_email} for status: {status}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ERROR] Both SMTP and AWS SES failed to send email to {to_email}")
        logger.error(f"[SES_ERROR] AWS SES error: {error_msg}")
        
        # Log specific SES errors for debugging
        if "MessageRejected" in error_msg and "not verified" in error_msg:
            logger.error(f"[SES_ERROR] Email address {to_email} is not verified in AWS SES.")
        elif "MessageRejected" in error_msg and "sandbox" in error_msg.lower():
            logger.error(f"[SES_ERROR] AWS SES is in sandbox mode. Only verified email addresses can receive emails.")
        elif "InvalidParameterValue" in error_msg:
            logger.error(f"[SES_ERROR] Invalid email parameters. Check email format and content.")
        else:
            import traceback
            logger.error(f"[TRACEBACK] Full traceback: {traceback.format_exc()}")
        
        return False 

def send_application_confirmation_email(to_email, candidate_name, job_title, company_name, job_location, applied_date):
    """Send application confirmation email to candidate"""
    logger = get_logger('emails')
    logger.info(f"[EMAIL] send_application_confirmation_email called with:")
    logger.info(f"   - to_email: {to_email}")
    logger.info(f"   - candidate_name: {candidate_name}")
    logger.info(f"   - job_title: {job_title}")
    logger.info(f"   - company_name: {company_name}")
    logger.info(f"   - job_location: {job_location}")
    logger.info(f"   - applied_date: {applied_date}")
    
    # Try SMTP first (primary method)
    logger.info(f"[SMTP] Attempting to send application confirmation via SMTP (primary method)...")
    try:
        from .smtp import send_application_confirmation_email_smtp
        smtp_result = send_application_confirmation_email_smtp(
            to_email, candidate_name, job_title, company_name, 
            job_location, applied_date
        )
        if smtp_result:
            logger.info(f"[SUCCESS] Application confirmation email sent successfully via SMTP to {to_email}")
            return True
        else:
            logger.warning(f"[SMTP_FAILED] SMTP failed, trying AWS SES fallback...")
    except Exception as smtp_error:
        logger.warning(f"[SMTP_ERROR] SMTP failed: {str(smtp_error)}, trying AWS SES fallback...")
    
    # Fallback to AWS SES
    logger.info(f"[SES] Attempting to send application confirmation via AWS SES (fallback method)...")
    try:
        client = get_ses_client()
        if not client:
            logger.error("SES client not available for application confirmation email")
            return False
            
        # Format applied date
        if isinstance(applied_date, str):
            formatted_date = applied_date
        else:
            formatted_date = applied_date.strftime('%B %d, %Y at %I:%M %p')
        
        # Create HTML email content
        subject = f"Thank you for applying to {job_title} at {company_name}"
        
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
        
        logger.info(f"[SES] Sending application confirmation email via SES...")
        logger.info(f"   - From: {SES_FROM_EMAIL}")
        logger.info(f"   - To: {to_email}")
        logger.info(f"   - Subject: {subject}")
        
        client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {
                    'Html': {'Data': html_body},
                    'Text': {'Data': text_body}
                }
            }
        )
        
        logger.info(f"[SUCCESS] Application confirmation email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ERROR] Both SMTP and AWS SES failed to send application confirmation email to {to_email}")
        logger.error(f"[SES_ERROR] AWS SES error: {error_msg}")
        
        # Log specific SES errors for debugging
        if "MessageRejected" in error_msg and "not verified" in error_msg:
            logger.error(f"[SES_ERROR] Email address {to_email} is not verified in AWS SES.")
        elif "MessageRejected" in error_msg and "sandbox" in error_msg.lower():
            logger.error(f"[SES_ERROR] AWS SES is in sandbox mode. Only verified email addresses can receive emails.")
        elif "InvalidParameterValue" in error_msg:
            logger.error(f"[SES_ERROR] Invalid email parameters. Check email format and content.")
        else:
            import traceback
            logger.error(f"[TRACEBACK] Full traceback: {traceback.format_exc()}")
        
        return False

def send_payslip_email(to_email, employee_name, pay_period_start, pay_period_end, net_pay, currency, payslip_url):
    """Send payslip email to employee"""
    logger = get_logger('emails')
    logger.info(f"[EMAIL] send_payslip_email called for {to_email}")
    
    # Handle date formatting
    if isinstance(pay_period_start, str):
        try:
            from datetime import datetime
            pay_period_start = datetime.strptime(pay_period_start, '%Y-%m-%d').date()
        except:
            pass
    
    if isinstance(pay_period_end, str):
        try:
            from datetime import datetime
            pay_period_end = datetime.strptime(pay_period_end, '%Y-%m-%d').date()
        except:
            pass
    
    subject = f"Your Payslip for {pay_period_start.strftime('%B %Y') if hasattr(pay_period_start, 'strftime') else pay_period_start}"
    
    # Create HTML email content
    start_str = pay_period_start.strftime('%B %d, %Y') if hasattr(pay_period_start, 'strftime') else str(pay_period_start)
    end_str = pay_period_end.strftime('%B %d, %Y') if hasattr(pay_period_end, 'strftime') else str(pay_period_end)
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{subject}</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: #2563eb; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;">
            <h1 style="margin: 0; font-size: 24px;">📄 Your Payslip is Ready</h1>
        </div>
        <div style="background: #f8fafc; padding: 30px; border: 1px solid #e2e8f0; border-top: 0; border-radius: 0 0 8px 8px;">
            <p style="font-size: 16px; margin-bottom: 20px;">Dear {employee_name or 'Employee'},</p>
            <p style="font-size: 16px; margin-bottom: 20px;">
                Your payslip for the period <strong>{start_str} - {end_str}</strong> is now available.
            </p>
            <div style="background: white; padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 4px solid #2563eb;">
                <h3 style="margin-top: 0; color: #2563eb;">Net Pay</h3>
                <p style="font-size: 24px; font-weight: bold; color: #059669; margin: 10px 0;">{currency} {float(net_pay):.2f}</p>
            </div>
            <div style="text-align: center; margin: 30px 0;">
                <a href="{payslip_url}" style="background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: bold;">View Payslip</a>
            </div>
            <p style="font-size: 14px; color: #666; margin-top: 30px;">
                If you have any questions about your payslip, please contact your HR department.
            </p>
            <p style="font-size: 14px; margin-top: 20px;">
                Best regards,<br>
                Payroll Team
            </p>
        </div>
    </body>
    </html>
    """
    
    # Create plain text version
    text_body = f"""
{subject}

Dear {employee_name or 'Employee'},

Your payslip for the period {start_str} - {end_str} is now available.

Net Pay: {currency} {float(net_pay):.2f}

View your payslip: {payslip_url}

If you have any questions about your payslip, please contact your HR department.

Best regards,
Payroll Team
    """
    
    # Try SMTP first
    try:
        from .smtp import send_email_via_smtp
        smtp_result = send_email_via_smtp(to_email, subject, html_body, text_body)
        if smtp_result:
            logger.info(f"[SUCCESS] Payslip email sent via SMTP to {to_email}")
            return True
    except Exception as smtp_error:
        logger.warning(f"[SMTP_ERROR] SMTP failed: {str(smtp_error)}, trying AWS SES fallback...")
    
    # Fallback to SES
    try:
        client = get_ses_client()
        if not client:
            logger.error("SES client not available for payslip email")
            return False
        
        client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {
                    'Html': {'Data': html_body},
                    'Text': {'Data': text_body}
                }
            }
        )
        
        logger.info(f"[SUCCESS] Payslip email sent via SES to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send payslip email to {to_email}: {e}")
        return False