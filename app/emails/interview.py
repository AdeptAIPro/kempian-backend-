"""
Interview invitation email functions
"""
from app.simple_logger import get_logger
import os

logger = get_logger('emails')

def send_interview_invitation_email(to_email, candidate_name, job_title, company_name, job_location, interview_date, meeting_link, meeting_type, interviewer_name, interview_notes):
    """Send interview invitation email to candidate"""
    logger.info(f"[EMAIL] send_interview_invitation_email called with:")
    logger.info(f"   - to_email: {to_email}")
    logger.info(f"   - candidate_name: {candidate_name}")
    logger.info(f"   - job_title: {job_title}")
    logger.info(f"   - company_name: {company_name}")
    logger.info(f"   - job_location: {job_location}")
    logger.info(f"   - interview_date: {interview_date}")
    logger.info(f"   - meeting_link: {meeting_link}")
    logger.info(f"   - meeting_type: {meeting_type}")
    logger.info(f"   - interviewer_name: {interviewer_name}")
    
    # Try SMTP first (primary method)
    logger.info(f"[SMTP] Attempting to send interview invitation via SMTP (primary method)...")
    try:
        from .smtp import send_interview_invitation_email_smtp
        smtp_result = send_interview_invitation_email_smtp(
            to_email, candidate_name, job_title, company_name, 
            job_location, interview_date, meeting_link, meeting_type, 
            interviewer_name, interview_notes
        )
        if smtp_result:
            logger.info(f"[SUCCESS] Interview invitation email sent successfully via SMTP to {to_email}")
            return True
        else:
            logger.warning(f"[SMTP_FAILED] SMTP failed, trying AWS SES fallback...")
    except Exception as smtp_error:
        logger.warning(f"[SMTP_ERROR] SMTP failed: {str(smtp_error)}, trying AWS SES fallback...")
    
    # Fallback to AWS SES
    logger.info(f"[SES] Attempting to send interview invitation via AWS SES (fallback method)...")
    try:
        from .ses import get_ses_client, SES_FROM_EMAIL
        client = get_ses_client()
        if not client:
            logger.error("SES client not available for interview invitation email")
            return False
            
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
        
        # Create HTML email content
        subject = f"Interview Invitation - {job_title} at {company_name}"
        
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
        
        logger.info(f"[SES] Sending interview invitation email via SES...")
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
        
        logger.info(f"[SUCCESS] Interview invitation email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ERROR] Both SMTP and AWS SES failed to send interview invitation email to {to_email}")
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
