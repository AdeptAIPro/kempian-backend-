"""
SMTP Email Service as fallback for AWS SES
"""
import smtplib
import ssl
from typing import Optional, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.simple_logger import get_logger
import os

logger = get_logger('emails')

def send_email_via_smtp(to_email, subject, body_html, body_text=None, reply_to=None):
    """
    Send email via SMTP (Gmail, Hostinger, Outlook, etc.)
    """
    try:
        # SMTP Configuration - Default to Hostinger SMTP
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.hostinger.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))  # 587 for TLS, 465 for SSL
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
        
        # Add Reply-To header if provided
        if reply_to:
            msg['Reply-To'] = reply_to
        
        # Add text and HTML parts
        if body_text:
            text_part = MIMEText(body_text, 'plain')
            msg.attach(text_part)
        
        html_part = MIMEText(body_html, 'html')
        msg.attach(html_part)
        
        # Create secure connection and send email
        # Handle both TLS (port 587) and SSL (port 465)
        context = ssl.create_default_context()
        
        if smtp_port == 465:
            # SSL connection for port 465
            logger.info(f"[SMTP] Using SSL connection (port 465)")
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
        else:
            # TLS connection for port 587 or other ports
            logger.info(f"[SMTP] Using TLS connection (port {smtp_port})")
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls(context=context)
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
        
        logger.info(f"[SMTP] Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"[SMTP] Failed to send email: {str(e)}")
        return False

def send_admin_notification_email(new_user_email: str, new_user_role: str, new_user_name: str = None):
    """
    Send notification email to admin when a new user signs up
    """
    try:
        from datetime import datetime
        
        admin_email = "vinit@adeptaipro.com"  # Can be changed to send to multiple admins
        
        # Create HTML email body
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>New User Signup Notification</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #e9ecef;
                }}
                .logo {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2563eb;
                    margin-bottom: 10px;
                }}
                .notification-badge {{
                    background: linear-gradient(135deg, #10b981, #059669);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 14px;
                    font-weight: 600;
                    display: inline-block;
                    margin-bottom: 20px;
                }}
                .user-info {{
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    border-left: 4px solid #2563eb;
                }}
                .info-row {{
                    display: flex;
                    justify-content: space-between;
                    margin: 10px 0;
                    padding: 8px 0;
                    border-bottom: 1px solid #e9ecef;
                }}
                .info-row:last-child {{
                    border-bottom: none;
                }}
                .label {{
                    font-weight: 600;
                    color: #6b7280;
                }}
                .value {{
                    color: #1f2937;
                }}
                .role-badge {{
                    background: #dbeafe;
                    color: #1e40af;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: 600;
                    text-transform: capitalize;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #e9ecef;
                    text-align: center;
                    color: #6b7280;
                    font-size: 14px;
                }}
                .action-button {{
                    background: linear-gradient(135deg, #2563eb, #1d4ed8);
                    color: white;
                    padding: 12px 24px;
                    border-radius: 8px;
                    text-decoration: none;
                    font-weight: 600;
                    display: inline-block;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">Kempian</div>
                    <div class="notification-badge">New User Signup Alert</div>
                </div>
                
                <h2 style="color: #1f2937; margin-bottom: 20px;">New User Registration</h2>
                
                <div class="user-info">
                    <div class="info-row">
                        <span class="label">Email Address:</span>
                        <span class="value">{new_user_email}</span>
                    </div>
                    <div class="info-row">
                        <span class="label">User Role:</span>
                        <span class="value">
                            <span class="role-badge">{new_user_role.replace('_', ' ').title()}</span>
                        </span>
                    </div>
                    {f'<div class="info-row"><span class="label">Full Name:</span><span class="value">{new_user_name}</span></div>' if new_user_name else ''}
                    <div class="info-row">
                        <span class="label">Signup Time:</span>
                        <span class="value">{datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}</span>
                    </div>
                </div>
                
                <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 15px; margin: 20px 0;">
                    <strong style="color: #92400e;">Next Steps:</strong>
                    <ul style="margin: 10px 0; color: #92400e;">
                        <li>Review the new user's profile and information</li>
                        <li>Monitor their activity and engagement</li>
                        <li>Consider reaching out for onboarding if needed</li>
                    </ul>
                </div>
                
                <div style="text-align: center;">
                    <a href="https://kempian.ai/admin/dashboard" class="action-button">
                        View Admin Dashboard
                    </a>
                </div>
                
                <div class="footer">
                    <p>This is an automated notification from Kempian's user registration system.</p>
                    <p>If you have any questions, please contact the development team.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        text_body = f"""
        NEW USER SIGNUP NOTIFICATION
        ===========================
        
        A new user has registered on Kempian:
        
        Email: {new_user_email}
        Role: {new_user_role.replace('_', ' ').title()}
        {f'Name: {new_user_name}' if new_user_name else ''}
        Signup Time: {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}
        
        Next Steps:
        - Review the new user's profile and information
        - Monitor their activity and engagement
        - Consider reaching out for onboarding if needed
        
        Admin Dashboard: https://app.kempian.ai/admin/dashboard
        
        This is an automated notification from Kempian's user registration system.
        """
        
        # Send the email
        success = send_email_via_smtp(
            to_email=admin_email,
            subject=f"New User Signup: {new_user_role.replace('_', ' ').title()} - {new_user_email}",
            body_html=html_body,
            body_text=text_body
        )
        
        if success:
            logger.info(f"[ADMIN NOTIFICATION] Successfully sent new user notification to {admin_email}")
        else:
            logger.error(f"[ADMIN NOTIFICATION] Failed to send notification to {admin_email}")
            
        return success
        
    except Exception as e:
        logger.error(f"[ADMIN NOTIFICATION] Error sending admin notification: {str(e)}")
        return False

def send_welcome_email_smtp(to_email, user_name, role=None):
    """Send welcome email to new users via SMTP with role-specific content"""
    try:
        # Role-specific content
        role_configs = {
            'employer': {
                'subject': 'Welcome to Kempian AI - Transform Your Hiring Process',
                'greeting': f'Hi <strong>{user_name}</strong>,',
                'intro': 'Welcome to <span class="highlight">Kempian AI</span>! We\'re thrilled to have you join our community of forward-thinking employers who are revolutionizing the way they find and hire top talent.',
                'cta_text': 'Start Hiring Now',
                'cta_link': 'https://kempian.ai/dashboard'
            },
            'recruiter': {
                'subject': 'Welcome to Kempian AI - Your Recruitment Partner',
                'greeting': f'Hi <strong>{user_name}</strong>,',
                'intro': 'Welcome to <span class="highlight">Kempian AI</span>! We\'re excited to have you join our community of innovative recruiters who are transforming the talent acquisition landscape.',
                'cta_text': 'Start Recruiting',
                'cta_link': 'https://kempian.ai/dashboard'
            },
            'job_seeker': {
                'subject': 'Welcome to Kempian AI - Your Career Journey Starts Here',
                'greeting': f'Hi <strong>{user_name}</strong>,',
                'intro': 'Welcome to <span class="highlight">Kempian AI</span>! We\'re excited to have you join our platform where top employers discover exceptional talent like you.',
                'cta_text': 'Complete Your Profile',
                'cta_link': 'https://kempian.ai/profile'
            },
            'admin': {
                'subject': 'Welcome to Kempian AI - Admin Access Granted',
                'greeting': f'Hi <strong>{user_name}</strong>,',
                'intro': 'Welcome to <span class="highlight">Kempian AI</span>! You now have administrative access to our powerful AI-driven talent matching platform.',
                'cta_text': 'Access Admin Panel',
                'cta_link': 'https://kempian.ai/admin'
            }
        }
        
        # Default config for unknown roles
        config = role_configs.get(role, role_configs['employer'])
        subject = config['subject']
        
        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to Kempian AI</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center; color: white; }}
                .header h1 {{ margin: 0; font-size: 28px; font-weight: 600; }}
                .content {{ padding: 40px 30px; }}
                .welcome-box {{ background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 25px; border-radius: 12px; margin-bottom: 30px; text-align: center; }}
                .welcome-box h2 {{ margin: 0 0 15px 0; font-size: 24px; }}
                .welcome-box p {{ margin: 0; font-size: 16px; opacity: 0.9; }}
                .features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
                .feature {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                .feature-icon {{ font-size: 32px; margin-bottom: 10px; }}
                .cta-button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; margin: 20px 0; }}
                .cta-button:hover {{ background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%); }}
                .footer {{ background: #f8f9fa; padding: 30px; text-align: center; color: #6c757d; font-size: 14px; }}
                .highlight {{ color: #667eea; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Kempian AI</h1>
                </div>
                
                <div class="content">
                    <div class="welcome-box">
                        <h2>Welcome aboard, {user_name}!</h2>
                        <p>Thank you for joining the future of AI-powered talent matching</p>
                    </div>
                    
                    <p>{config['greeting']}</p>
                    
                    <p>{config['intro']}</p>
                    
                    <div class="features">
                        <div class="feature">
                            <h3>AI-Powered Matching</h3>
                            <p>76% accuracy in finding the perfect candidates</p>
                        </div>
                        <div class="feature">
                            <h3>Lightning Fast</h3>
                            <p>Reduce time-to-hire by 70%</p>
                        </div>
                        <div class="feature">
                            <h3>Smart Automation</h3>
                            <p>Automated screening and ranking</p>
                        </div>
                        <div class="feature">
                            <h3>App Store</h3>
                            <p>Integrations with your favorite tools</p>
                        </div>
                    </div>
                    
                    <!-- YouTube Video Thumbnail Link -->
                    <div style="margin: 30px 0; text-align: center;">
                        <h3 style="color: #667eea; font-size: 18px; margin-bottom: 15px; font-weight: 600;">Watch Our Platform Demo</h3>
                        <a href="https://www.youtube.com/watch?v=3KKOBI_Qz7w" style="text-decoration: none; display: inline-block;">
                            <div style="position: relative; max-width: 100%; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                                <img 
                                    src="https://img.youtube.com/vi/3KKOBI_Qz7w/maxresdefault.jpg" 
                                    alt="Kempian AI Platform Demo" 
                                    style="width: 100%; height: auto; display: block;"
                                />
                                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.8); border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center;">
                                    <div style="width: 0; height: 0; border-left: 20px solid white; border-top: 12px solid transparent; border-bottom: 12px solid transparent; margin-left: 4px;"></div>
                                </div>
                            </div>
                        </a>
                        <p style="color: #6c757d; font-size: 14px; margin-top: 10px;">Click to watch how Kempian AI transforms your hiring process</p>
                    </div>
                    
                    <div style="text-align: center;">
                        <a href="{config['cta_link']}" class="cta-button">{config['cta_text']}</a>
                    </div>
                    
                    <p>Your account is now active and ready to use. Here's what you can do next:</p>
                    <ul>
                        <li>Create your first job posting</li>
                        <li>Search our extensive talent database</li>
                        <li>Track your hiring metrics</li>
                        <li>Connect with top candidates</li>
                    </ul>
                    
                    <p>If you have any questions or need assistance getting started, our support team is here to help at <a href="mailto:support@kempian.ai">support@kempian.ai</a>.</p>
                    
                    <p>Welcome to the future of recruitment!</p>
                    
                    <p>Best regards,<br>
                    <strong>The Kempian AI Team</strong></p>
                </div>
                
                <div class="footer">
                    <p>© 2024 Kempian AI. All rights reserved.</p>
                    <p>You received this email because you signed up for Kempian AI.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        body_text = f"""
        Welcome to Kempian AI!

        Hi {user_name},

        {config['intro'].replace('<span class="highlight">', '').replace('</span>', '')}

        Your account is now active and ready to use. Here's what you can do next:
        - Create your first job posting
        - Search our extensive talent database
        - Track your hiring metrics
        - Connect with top candidates

        Features you'll love:
        - AI-Powered Matching: 76% accuracy in finding the perfect candidates
        - Lightning Fast: Reduce time-to-hire by 70%
        - Smart Automation: Automated screening and ranking
        - App Store: Integrations with your favorite tools

        Watch our demo video: https://www.youtube.com/watch?v=3KKOBI_Qz7w
        
        {config['cta_text']}: {config['cta_link']}

        If you have any questions or need assistance getting started, our support team is here to help at support@kempian.ai.

        Welcome to the future of recruitment!

        Best regards,
        The Kempian AI Team

        ---
        © 2024 Kempian AI. All rights reserved.
        You received this email because you signed up for Kempian AI.
        """
        
        return send_email_via_smtp(to_email, subject, body_html, body_text)
        
    except Exception as e:
        logger.error(f"[SMTP] Failed to send welcome email: {str(e)}")
        return False

def send_onboarding_thanks_email_smtp(to_email: str, contact_name: str = "", company_name: str = "", highlights: Optional[List[str]] = None):
    """Send a polished onboarding thank-you email via SMTP."""
    try:
        first_name = (contact_name or "").strip().split(' ')[0] if contact_name else "there"
        subject = "Thank You for Onboarding with Kempian AI"
        hi_list = highlights or []
        hi_items = ''.join([f"<li>• {h}</li>" for h in hi_list])
        company_line = f" at <strong>{company_name}</strong>" if company_name else ""
        highlights_html = ''
        if hi_items:
            highlights_html = (
                '<div class="card">'
                '<p style="margin:0"><strong>Your highlights</strong></p>'
                f'<ul style="margin:8px 0 0 18px; padding:0;">{hi_items}</ul>'
                '</div>'
            )

        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset=\"utf-8\" />
          <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
          <title>{subject}</title>
          <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#f8fafc; margin:0; padding:0; }}
            .container {{ max-width:600px; margin:0 auto; background:#ffffff; box-shadow:0 4px 10px rgba(0,0,0,0.06); }}
            .header {{ background:linear-gradient(135deg,#2563eb,#7c3aed); color:#fff; padding:32px 28px; text-align:center; }}
            .header h1 {{ margin:0; font-size:24px; font-weight:700; }}
            .content {{ padding:28px; color:#111827; }}
            .card {{ background:#f3f4f6; border-left:4px solid #2563eb; padding:18px; border-radius:10px; margin:16px 0; }}
            .cta {{ display:inline-block; padding:12px 20px; background:linear-gradient(135deg,#2563eb,#1d4ed8); color:#fff; text-decoration:none; border-radius:8px; font-weight:600; }}
            .muted {{ color:#6b7280; font-size:13px; }}
          </style>
        </head>
        <body>
          <div class=\"container\"> 
            <div class=\"header\"><h1>Thank you for onboarding KempianAI!</h1></div>
            <div class=\"content\">
              <p>Hi <strong>{first_name}</strong>,</p>
              <p>We’ve received your onboarding details and our team has started setting things up for you. You’ll get a follow-up from us shortly with next steps.</p>
              <div class=\"card\">
                <p style=\"margin:0\"><strong>What happens next?</strong></p>
                <ul style=\"margin:8px 0 0 18px; padding:0;\">
                  <li>We review your requirements and configure your workspace</li>
                  <li>We align integrations and preferences you selected</li>
                  <li>We share timelines and a success checklist</li>
                </ul>
              </div>
              {highlights_html}
              <p>You can track your details or update preferences anytime from your profile.</p>
              <p style=\"margin:20px 0\"><a class=\"cta\" href=\"https://kempian.ai/profile\">View your profile</a></p>
              <p class=\"muted\">Need help? Reply to this email or reach us at support@kempian.ai.</p>
              <p class=\"muted\">© 2024 Kempian AI</p>
            </div>
          </div>
        </body>
        </html>
        """
        body_text = f"""
        Thanks for onboarding with Kempian AI

        Hi {first_name},
        We’ve received your onboarding details{(' for ' + company_name) if company_name else ''}.
        Next steps:
        - We review your requirements and configure your workspace
        - We align integrations and preferences you selected
        - We share timelines and a success checklist

        View your profile: https://kempian.ai/profile
        Need help? support@kempian.ai
        """
        return send_email_via_smtp(to_email, subject, body_html, body_text)
    except Exception as e:
        logger.error(f"[SMTP] Failed to send onboarding thanks email: {str(e)}")
        return False

def send_onboarding_notification_to_support_smtp(user_email: str, user_name: str, company_name: str, onboarding_data: dict):
    """Send onboarding notification email to support@kempian.ai with full onboarding details"""
    try:
        from datetime import datetime
        support_email = "support@kempian.ai"
        subject = f"New Onboarding Submission - {company_name or user_email}"
        
        # Format onboarding data for display
        def format_field(label: str, value: any) -> str:
            if value is None or value == '' or (isinstance(value, (list, set)) and len(value) == 0):
                return f'<tr><td style="padding:8px; border-bottom:1px solid #e5e7eb;"><strong>{label}:</strong></td><td style="padding:8px; border-bottom:1px solid #e5e7eb; color:#6b7280;">Not provided</td></tr>'
            
            if isinstance(value, (list, set)):
                display_value = ', '.join(str(v) for v in value)
            elif isinstance(value, bool):
                display_value = 'Yes' if value else 'No'
            else:
                display_value = str(value)
            
            return f'<tr><td style="padding:8px; border-bottom:1px solid #e5e7eb;"><strong>{label}:</strong></td><td style="padding:8px; border-bottom:1px solid #e5e7eb;">{display_value}</td></tr>'
        
        # Build data table rows
        data_rows = []
        data_rows.append(format_field('Contact Name', onboarding_data.get('contactName') or onboarding_data.get('full_name')))
        data_rows.append(format_field('Email', user_email))
        data_rows.append(format_field('Phone', onboarding_data.get('phone')))
        data_rows.append(format_field('Company Name', company_name or onboarding_data.get('company_name')))
        data_rows.append(format_field('Website', onboarding_data.get('website')))
        data_rows.append(format_field('Company Size', onboarding_data.get('companySize')))
        data_rows.append(format_field('Location', onboarding_data.get('location') or onboarding_data.get('address')))
        data_rows.append(format_field('Existing Tools', onboarding_data.get('tools', [])))
        data_rows.append(format_field('Business Goals', onboarding_data.get('goals') or onboarding_data.get('business_goals')))
        data_rows.append(format_field('Services Interested In', onboarding_data.get('services', [])))
        data_rows.append(format_field('Project Scope', onboarding_data.get('scope') or onboarding_data.get('project_scope')))
        data_rows.append(format_field('Timeline', onboarding_data.get('timeline')))
        data_rows.append(format_field('Budget', onboarding_data.get('budget')))
        data_rows.append(format_field('Billing Address', onboarding_data.get('billingAddress')))
        data_rows.append(format_field('Payment Method', onboarding_data.get('paymentMethod')))
        data_rows.append(format_field('Data Consent', onboarding_data.get('consentData', False)))
        data_rows.append(format_field('Communication Consent', onboarding_data.get('consentCommunication', False)))
        data_rows.append(format_field('Additional Notes', onboarding_data.get('notes')))
        
        data_table = ''.join(data_rows)
        
        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>New Onboarding Submission</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
                .container {{ max-width: 800px; margin: 0 auto; background-color: #ffffff; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center; }}
                .header h1 {{ color: #ffffff; margin: 0; font-size: 28px; font-weight: 600; }}
                .content {{ padding: 40px 30px; }}
                .info-box {{ background: #f8f9fa; border-left: 4px solid #667eea; padding: 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
                .data-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .data-table tr:first-child td {{ padding-top: 16px; }}
                .data-table tr:last-child td {{ border-bottom: none; padding-bottom: 16px; }}
                .highlight {{ color: #667eea; font-weight: 600; }}
                .footer {{ background: #f8f9fa; padding: 30px; text-align: center; color: #6c757d; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>New Onboarding Submission</h1>
                </div>
                
                <div class="content">
                    <p>Hello Support Team,</p>
                    
                    <p>A new client has submitted their onboarding form. Please review the details below:</p>
                    
                    <div class="info-box">
                        <p style="margin:0 0 10px 0;"><strong>Client Information:</strong></p>
                        <p style="margin:0;"><strong>Name:</strong> {user_name or 'Not provided'}</p>
                        <p style="margin:0;"><strong>Email:</strong> {user_email}</p>
                        <p style="margin:0;"><strong>Company:</strong> {company_name or 'Not provided'}</p>
                        <p style="margin:0;"><strong>Submitted:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}</p>
                    </div>
                    
                    <h3 style="color: #333; margin-top: 30px;">Onboarding Details:</h3>
                    
                    <table class="data-table">
                        {data_table}
                    </table>
                    
                    <div class="info-box">
                        <p style="margin:0;"><strong>Action Required:</strong> Please review this submission and reach out to the client within 2 business days to discuss next steps and configure their workspace.</p>
                    </div>
                    
                    <p>You can view all onboarding submissions in the admin dashboard.</p>
                    
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
        
        body_text = f"""
        New Onboarding Submission - Kempian AI
        
        A new client has submitted their onboarding form.
        
        Client Information:
        Name: {user_name or 'Not provided'}
        Email: {user_email}
        Company: {company_name or 'Not provided'}
        Submitted: {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}
        
        Onboarding Details:
        Contact Name: {onboarding_data.get('contactName') or onboarding_data.get('full_name') or 'Not provided'}
        Email: {user_email}
        Phone: {onboarding_data.get('phone') or 'Not provided'}
        Company Name: {company_name or onboarding_data.get('company_name') or 'Not provided'}
        Website: {onboarding_data.get('website') or 'Not provided'}
        Company Size: {onboarding_data.get('companySize') or 'Not provided'}
        Location: {onboarding_data.get('location') or onboarding_data.get('address') or 'Not provided'}
        Existing Tools: {', '.join(str(v) for v in (onboarding_data.get('tools') or [])) or 'Not provided'}
        Business Goals: {onboarding_data.get('goals') or onboarding_data.get('business_goals') or 'Not provided'}
        Services: {', '.join(str(v) for v in (onboarding_data.get('services') or [])) or 'Not provided'}
        Project Scope: {onboarding_data.get('scope') or onboarding_data.get('project_scope') or 'Not provided'}
        Timeline: {onboarding_data.get('timeline') or 'Not provided'}
        Budget: {onboarding_data.get('budget') or 'Not provided'}
        Billing Address: {onboarding_data.get('billingAddress') or 'Not provided'}
        Payment Method: {onboarding_data.get('paymentMethod') or 'Not provided'}
        Data Consent: {'Yes' if onboarding_data.get('consentData') else 'No'}
        Communication Consent: {'Yes' if onboarding_data.get('consentCommunication') else 'No'}
        Additional Notes: {onboarding_data.get('notes') or 'Not provided'}
        
        Action Required: Please review this submission and reach out to the client within 2 business days.
        """
        
        return send_email_via_smtp(support_email, subject, body_html, body_text)
    except Exception as e:
        logger.error(f"[SMTP] Failed to send onboarding notification to support: {str(e)}")
        return False

def send_trial_ending_reminder_email_smtp(to_email, user_name, days_remaining, trial_end_date):
    """Send trial ending reminder email via SMTP"""
    try:
        # Format trial end date
        if isinstance(trial_end_date, str):
            from datetime import datetime
            trial_end_date = datetime.fromisoformat(trial_end_date.replace('Z', '+00:00'))
        
        formatted_end_date = trial_end_date.strftime('%B %d, %Y at %I:%M %p UTC')
        
        subject = f"Your Kempian AI Trial Ends in {days_remaining} Day{'s' if days_remaining != 1 else ''}"
        
        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trial Ending Soon</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center; }}
                .header h1 {{ color: #ffffff; margin: 0; font-size: 28px; font-weight: 600; }}
                .content {{ padding: 40px 30px; }}
                .alert-box {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; text-align: center; }}
                .alert-box h2 {{ margin: 0 0 10px 0; font-size: 24px; }}
                .alert-box p {{ margin: 0; font-size: 16px; opacity: 0.9; }}
                .info-box {{ background: #f8f9fa; border-left: 4px solid #667eea; padding: 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
                .cta-button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; margin: 20px 0; }}
                .cta-button:hover {{ background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%); }}
                .features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
                .feature {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                .feature-icon {{ font-size: 32px; margin-bottom: 10px; }}
                .footer {{ background: #f8f9fa; padding: 30px; text-align: center; color: #6c757d; font-size: 14px; }}
                .highlight {{ color: #667eea; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Trial Ending Soon</h1>
                </div>
                
                <div class="content">
                    <div class="alert-box">
                        <h2>Your trial ends in {days_remaining} day{'s' if days_remaining != 1 else ''}!</h2>
                        <p>Trial expires on {formatted_end_date}</p>
                    </div>
                    
                    <p>Hi {user_name},</p>
                    
                    <p>We hope you've been enjoying your <span class="highlight">Kempian AI</span> trial experience! Your free trial is ending soon, and we don't want you to lose access to our powerful AI-driven talent matching platform.</p>
                    
                    <div class="info-box">
                        <h3>What happens when your trial ends?</h3>
                        <ul>
                            <li>You'll lose access to AI-powered candidate matching</li>
                            <li>No more job description analysis and optimization</li>
                            <li>Limited access to our talent database</li>
                            <li>No more automated candidate screening</li>
                        </ul>
                    </div>
                    
                    <div style="text-align: center;">
                        <a href="https://kempian.ai/plans" class="cta-button">Upgrade Now & Continue Your Success</a>
                    </div>
                    
                    <div class="features">
                        <div class="feature">
                            <h3>Precise Matching</h3>
                            <p>76% accuracy in finding the right candidates</p>
                        </div>
                        <div class="feature">
                            <h3>Lightning Fast</h3>
                            <p>Reduce time-to-hire by 70%</p>
                        </div>
                        <div class="feature">
                            <h3>AI-Powered</h3>
                            <p>Advanced algorithms for better results</p>
                        </div>
                    </div>
                    
                    <p>Don't let your hiring momentum slow down. Upgrade now to continue finding the perfect candidates for your open positions.</p>
                    
                    <p>If you have any questions, feel free to reach out to our support team at <a href="mailto:support@kempian.ai">support@kempian.ai</a>.</p>
                    
                    <p>Best regards,<br>The Kempian AI Team</p>
                </div>
                
                <div class="footer">
                    <p>© 2024 Kempian AI. All rights reserved.</p>
                    <p>You received this email because you have an active trial account.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        body_text = f"""
        Trial Ending Soon - Kempian AI
        
        Hi {user_name},
        
        Your Kempian AI trial ends in {days_remaining} day{'s' if days_remaining != 1 else ''}!
        Trial expires on: {formatted_end_date}
        
        What happens when your trial ends?
        - You'll lose access to AI-powered candidate matching
        - No more job description analysis and optimization
        - Limited access to our talent database
        - No more automated candidate screening
        
        Upgrade now to continue your success:
        https://kempian.ai/plans
        
        Features you'll keep with a paid plan:
        - 76% accuracy in finding the right candidates
        - Reduce time-to-hire by 70%
        - Advanced AI algorithms for better results
        
        Questions? Contact us at support@kempian.ai
        
        Best regards,
        The Kempian AI Team
        """
        
        return send_email_via_smtp(to_email, subject, body_html, body_text)
        
    except Exception as e:
        logger.error(f"[SMTP] Failed to send trial reminder email: {str(e)}")
        return False

def send_trial_expired_email_smtp(to_email, user_name, trial_end_date):
    """Send trial expired email via SMTP"""
    try:
        # Format trial end date
        if isinstance(trial_end_date, str):
            from datetime import datetime
            trial_end_date = datetime.fromisoformat(trial_end_date.replace('Z', '+00:00'))
        
        formatted_end_date = trial_end_date.strftime('%B %d, %Y at %I:%M %p UTC')
        
        subject = "Your Kempian AI Trial Has Expired"
        
        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trial Expired</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; }}
                .header {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 40px 30px; text-align: center; }}
                .header h1 {{ color: #ffffff; margin: 0; font-size: 28px; font-weight: 600; }}
                .content {{ padding: 40px 30px; }}
                .expired-box {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; text-align: center; }}
                .expired-box h2 {{ margin: 0 0 10px 0; font-size: 24px; }}
                .expired-box p {{ margin: 0; font-size: 16px; opacity: 0.9; }}
                .info-box {{ background: #f8f9fa; border-left: 4px solid #ff6b6b; padding: 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
                .cta-button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; margin: 20px 0; }}
                .cta-button:hover {{ background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%); }}
                .benefits {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
                .benefit {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                .benefit-icon {{ font-size: 32px; margin-bottom: 10px; }}
                .footer {{ background: #f8f9fa; padding: 30px; text-align: center; color: #6c757d; font-size: 14px; }}
                .highlight {{ color: #667eea; font-weight: 600; }}
                .urgent {{ color: #ff6b6b; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Trial Expired</h1>
                </div>
                
                <div class="content">
                    <div class="expired-box">
                        <h2>Your trial has expired</h2>
                        <p>Expired on {formatted_end_date}</p>
                    </div>
                    
                    <p>Hi {user_name},</p>
                    
                    <p>Your <span class="highlight">Kempian AI</span> trial has expired, but don't worry - we're here to help you get back on track! We know you were making great progress with our platform.</p>
                    
                    <div class="info-box">
                        <h3>What you're missing now:</h3>
                        <ul>
                            <li>AI-powered candidate matching (76% accuracy)</li>
                            <li>Job description analysis and optimization</li>
                            <li>Access to our extensive talent database</li>
                            <li>Automated candidate screening and ranking</li>
                            <li>Advanced search filters and criteria</li>
                        </ul>
                    </div>
                    
                    <div style="text-align: center;">
                        <a href="https://kempian.ai/plans" class="cta-button">Renew Now & Restore Access</a>
                    </div>
                    
                    <div class="benefits">
                        <div class="benefit">
                            <h3>Precise Matching</h3>
                            <p>76% accuracy in finding the right candidates</p>
                        </div>
                        <div class="benefit">
                            <h3>Lightning Fast</h3>
                            <p>Reduce time-to-hire by 70%</p>
                        </div>
                        <div class="benefit">
                            <h3>AI-Powered</h3>
                            <p>Advanced algorithms for better results</p>
                        </div>
                    </div>
                    
                    <p><span class="urgent">Special Offer:</span> Get 20% off your first month when you upgrade within 48 hours!</p>
                    
                    <p>Don't let your hiring momentum slow down. Upgrade now to restore full access and continue finding the perfect candidates.</p>
                    
                    <p>Questions? Our support team is here to help at <a href="mailto:support@kempian.ai">support@kempian.ai</a>.</p>
                    
                    <p>Best regards,<br>The Kempian AI Team</p>
                </div>
                
                <div class="footer">
                    <p>© 2024 Kempian AI. All rights reserved.</p>
                    <p>You received this email because your trial account has expired.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        body_text = f"""
        Trial Expired - Kempian AI
        
        Hi {user_name},
        
        Your Kempian AI trial has expired on {formatted_end_date}.
        
        What you're missing now:
        - AI-powered candidate matching (76% accuracy)
        - Job description analysis and optimization
        - Access to our extensive talent database
        - Automated candidate screening and ranking
        - Advanced search filters and criteria
        
        Special Offer: Get 20% off your first month when you upgrade within 48 hours!
        
        Renew now to restore access:
        https://kempian.ai/plans
        
        Questions? Contact us at support@kempian.ai
        
        Best regards,
        The Kempian AI Team
        """
        
        return send_email_via_smtp(to_email, subject, body_html, body_text)
        
    except Exception as e:
        logger.error(f"[SMTP] Failed to send trial expired email: {str(e)}")
        return False

def send_application_status_email_smtp(to_email, candidate_name, job_title, company_name, job_location, applied_date, status):
    """
    Send application status email via SMTP
    """
    logger.info(f"[SMTP] Sending application status email to {to_email}")
    
    # Status configurations
    status_configs = {
        'hired': {
            'subject': f"Congratulations! You're Hired for {job_title} at {company_name}",
            'color': '#28a745',
            'icon': ''
        },
        'shortlisted': {
            'subject': f"Great News! You've Been Shortlisted for {job_title} at {company_name}",
            'color': '#17a2b8',
            'icon': ''
        },
        'rejected': {
            'subject': f"Update on Your Application for {job_title} at {company_name}",
            'color': '#dc3545',
            'icon': ''
        },
        'reviewed': {
            'subject': f"Your Application for {job_title} at {company_name} Has Been Reviewed",
            'color': '#ffc107',
            'icon': ''
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
            <h1 style="margin: 0; font-size: 24px;">{config['subject']}</h1>
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
            <h1 style="margin: 0; font-size: 24px;">Application Received</h1>
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

def send_candidate_shortlisted_email_smtp(to_email, candidate_name, job_title, company_name, job_description, contact_email):
    """
    Send candidate shortlisted email via SMTP
    """
    logger.info(f"[SMTP] Sending candidate shortlisted email to {to_email}")
    
    # Extract role from job description
    def extract_role_from_job_description(job_desc):
        """Extract the main role/title from job description"""
        if not job_desc:
            return job_title or "the position"
        
        # Common role patterns
        role_patterns = [
            r'(?:looking for|seeking|hiring)\s+(?:a\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:position|role|job)\s+(?:of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:developer|engineer|manager|analyst|designer|specialist|coordinator|director|lead|architect)',
            r'(?:Senior|Junior|Lead|Principal|Staff)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        import re
        for pattern in role_patterns:
            match = re.search(pattern, job_desc, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: look for common job titles
        common_titles = [
            'Software Engineer', 'Developer', 'Manager', 'Analyst', 'Designer',
            'Marketing Specialist', 'Sales Representative', 'HR Coordinator',
            'Data Scientist', 'Product Manager', 'Project Manager', 'Consultant'
        ]
        
        for title in common_titles:
            if title.lower() in job_desc.lower():
                return title
        
        return job_title or "the position"
    
    extracted_role = extract_role_from_job_description(job_description)
    
    # Create subject
    subject = f"Congratulations! You've Been Shortlisted for {extracted_role} at {company_name}"
    
    # Create HTML email
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{subject}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
            .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center; color: white; }}
            .header h1 {{ margin: 0; font-size: 28px; font-weight: 600; }}
            .content {{ padding: 40px 30px; }}
            .congratulations-box {{ background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 25px; border-radius: 12px; margin-bottom: 30px; text-align: center; }}
            .congratulations-box h2 {{ margin: 0 0 15px 0; font-size: 24px; }}
            .congratulations-box p {{ margin: 0; font-size: 16px; opacity: 0.9; }}
            .job-details {{ background: #f8f9fa; border-left: 4px solid #667eea; padding: 25px; margin: 25px 0; border-radius: 0 8px 8px 0; }}
            .job-details h3 {{ margin-top: 0; color: #667eea; font-size: 20px; }}
            .contact-box {{ background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 25px 0; text-align: center; }}
            .contact-box h4 {{ margin-top: 0; color: #1976d2; }}
            .cta-button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; margin: 15px 0; }}
            .cta-button:hover {{ background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%); }}
            .footer {{ background: #f8f9fa; padding: 30px; text-align: center; color: #6c757d; font-size: 14px; }}
            .highlight {{ color: #667eea; font-weight: 600; }}
            .next-steps {{ background: #fff3e0; padding: 20px; border-radius: 8px; margin: 25px 0; }}
            .next-steps h4 {{ margin-top: 0; color: #f57c00; }}
            .next-steps ul {{ margin: 10px 0; padding-left: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Congratulations!</h1>
            </div>
            
            <div class="content">
                <div class="congratulations-box">
                    <h2>You've been shortlisted!</h2>
                    <p>We're excited to inform you that you've been selected for the next stage of our hiring process.</p>
                </div>
                
                <p>Dear <strong>{candidate_name}</strong>,</p>
                
                <p>We are thrilled to inform you that you have been <span class="highlight">shortlisted</span> for the position of <strong>{extracted_role}</strong> at <strong>{company_name}</strong>!</p>
                
                <div class="job-details">
                    <h3>Position Details</h3>
                    <p><strong>Role:</strong> {extracted_role}</p>
                    <p><strong>Company:</strong> {company_name}</p>
                    <p><strong>Status:</strong> <span style="color: #28a745; font-weight: bold;">Shortlisted for Interview</span></p>
                </div>
                
                <p>Your application stood out among many candidates, and we believe you have the skills and experience we're looking for. We're impressed by your qualifications and would like to move forward with the next steps in our hiring process.</p>
                
                <div class="next-steps">
                    <h4>What Happens Next?</h4>
                    <ul>
                        <li>Our hiring team will review your application in detail</li>
                        <li>We'll contact you within 2-3 business days to schedule an interview</li>
                        <li>You may be asked to complete additional assessments or provide references</li>
                        <li>We'll keep you updated throughout the process</li>
                    </ul>
                </div>
                
                <div class="contact-box">
                    <h4>Have Questions?</h4>
                    <p>If you have any questions about the position or the next steps, please don't hesitate to contact us:</p>
                    <p><strong>Email:</strong> <a href="mailto:{contact_email}" style="color: #1976d2; text-decoration: none;">{contact_email}</a></p>
                </div>
                
                <div style="text-align: center;">
                    <a href="mailto:{contact_email}?subject=Re: {extracted_role} at {company_name} - Shortlisted" class="cta-button">
                        Contact Us Now
                    </a>
                </div>
                
                <p>We're excited about the possibility of you joining our team and look forward to learning more about you in the upcoming interview.</p>
                
                <p>Thank you for your interest in <strong>{company_name}</strong>!</p>
                
                <p>Best regards,<br>
                <strong>The {company_name} Hiring Team</strong></p>
            </div>
            
            <div class="footer">
                <p>© 2024 {company_name}. All rights reserved.</p>
                <p>This email was sent because you were shortlisted for a position through our recruitment process.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create text version
    text_body = f"""
{subject}

Dear {candidate_name},

We are thrilled to inform you that you have been SHORTLISTED for the position of {extracted_role} at {company_name}!

Your application stood out among many candidates, and we believe you have the skills and experience we're looking for. We're impressed by your qualifications and would like to move forward with the next steps in our hiring process.

Position Details:
- Role: {extracted_role}
- Company: {company_name}
- Status: Shortlisted for Interview

What happens next?
- Our hiring team will review your application in detail
- We'll contact you within 2-3 business days to schedule an interview
- You may be asked to complete additional assessments or provide references
- We'll keep you updated throughout the process

Have Questions?
If you have any questions about the position or the next steps, please don't hesitate to contact us:
Email: {contact_email}

We're excited about the possibility of you joining our team and look forward to learning more about you in the upcoming interview.

Thank you for your interest in {company_name}!

Best regards,
The {company_name} Hiring Team

---
© 2024 {company_name}. All rights reserved.
This email was sent because you were shortlisted for a position through our recruitment process.
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
    
    # Create subject (strip any accidental newlines or embedded descriptions)
    raw_subject = f"Interview Invitation - {job_title} at {company_name}"
    # Remove line breaks that cause "embedded header" issues in SMTP libraries
    subject = " ".join(str(raw_subject).splitlines())[:255]
    
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
            <h1 style="margin: 0; font-size: 24px;">Interview Invitation</h1>
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

def send_candidate_shortlisted_email_smtp(to_email, candidate_name, job_title, company_name, job_description, contact_email):
    """
    Send candidate shortlisted email via SMTP
    """
    logger.info(f"[SMTP] Sending candidate shortlisted email to {to_email}")
    
    # Extract role from job description
    def extract_role_from_job_description(job_desc):
        """Extract the main role/title from job description"""
        if not job_desc:
            return job_title or "the position"
        
        # Common role patterns
        role_patterns = [
            r'(?:looking for|seeking|hiring)\s+(?:a\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:position|role|job)\s+(?:of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:developer|engineer|manager|analyst|designer|specialist|coordinator|director|lead|architect)',
            r'(?:Senior|Junior|Lead|Principal|Staff)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        import re
        for pattern in role_patterns:
            match = re.search(pattern, job_desc, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: look for common job titles
        common_titles = [
            'Software Engineer', 'Developer', 'Manager', 'Analyst', 'Designer',
            'Marketing Specialist', 'Sales Representative', 'HR Coordinator',
            'Data Scientist', 'Product Manager', 'Project Manager', 'Consultant'
        ]
        
        for title in common_titles:
            if title.lower() in job_desc.lower():
                return title
        
        return job_title or "the position"
    
    extracted_role = extract_role_from_job_description(job_description)
    
    # Create subject
    subject = f"Congratulations! You've Been Shortlisted for {extracted_role} at {company_name}"
    
    # Create HTML email
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{subject}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
            .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center; color: white; }}
            .header h1 {{ margin: 0; font-size: 28px; font-weight: 600; }}
            .content {{ padding: 40px 30px; }}
            .congratulations-box {{ background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 25px; border-radius: 12px; margin-bottom: 30px; text-align: center; }}
            .congratulations-box h2 {{ margin: 0 0 15px 0; font-size: 24px; }}
            .congratulations-box p {{ margin: 0; font-size: 16px; opacity: 0.9; }}
            .job-details {{ background: #f8f9fa; border-left: 4px solid #667eea; padding: 25px; margin: 25px 0; border-radius: 0 8px 8px 0; }}
            .job-details h3 {{ margin-top: 0; color: #667eea; font-size: 20px; }}
            .contact-box {{ background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 25px 0; text-align: center; }}
            .contact-box h4 {{ margin-top: 0; color: #1976d2; }}
            .cta-button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; margin: 15px 0; }}
            .cta-button:hover {{ background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%); }}
            .footer {{ background: #f8f9fa; padding: 30px; text-align: center; color: #6c757d; font-size: 14px; }}
            .highlight {{ color: #667eea; font-weight: 600; }}
            .next-steps {{ background: #fff3e0; padding: 20px; border-radius: 8px; margin: 25px 0; }}
            .next-steps h4 {{ margin-top: 0; color: #f57c00; }}
            .next-steps ul {{ margin: 10px 0; padding-left: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Congratulations!</h1>
            </div>
            
            <div class="content">
                <div class="congratulations-box">
                    <h2>You've been shortlisted!</h2>
                    <p>We're excited to inform you that you've been selected for the next stage of our hiring process.</p>
                </div>
                
                <p>Dear <strong>{candidate_name}</strong>,</p>
                
                <p>We are thrilled to inform you that you have been <span class="highlight">shortlisted</span> for the position of <strong>{extracted_role}</strong> at <strong>{company_name}</strong>!</p>
                
                <div class="job-details">
                    <h3>Position Details</h3>
                    <p><strong>Role:</strong> {extracted_role}</p>
                    <p><strong>Company:</strong> {company_name}</p>
                    <p><strong>Status:</strong> <span style="color: #28a745; font-weight: bold;">Shortlisted for Interview</span></p>
                </div>
                
                <p>Your application stood out among many candidates, and we believe you have the skills and experience we're looking for. We're impressed by your qualifications and would like to move forward with the next steps in our hiring process.</p>
                
                <div class="next-steps">
                    <h4>What Happens Next?</h4>
                    <ul>
                        <li>Our hiring team will review your application in detail</li>
                        <li>We'll contact you within 2-3 business days to schedule an interview</li>
                        <li>You may be asked to complete additional assessments or provide references</li>
                        <li>We'll keep you updated throughout the process</li>
                    </ul>
                </div>
                
                <div class="contact-box">
                    <h4>Have Questions?</h4>
                    <p>If you have any questions about the position or the next steps, please don't hesitate to contact us:</p>
                    <p><strong>Email:</strong> <a href="mailto:{contact_email}" style="color: #1976d2; text-decoration: none;">{contact_email}</a></p>
                </div>
                
                <div style="text-align: center;">
                    <a href="mailto:{contact_email}?subject=Re: {extracted_role} at {company_name} - Shortlisted" class="cta-button">
                        Contact Us Now
                    </a>
                </div>
                
                <p>We're excited about the possibility of you joining our team and look forward to learning more about you in the upcoming interview.</p>
                
                <p>Thank you for your interest in <strong>{company_name}</strong>!</p>
                
                <p>Best regards,<br>
                <strong>The {company_name} Hiring Team</strong></p>
            </div>
            
            <div class="footer">
                <p>© 2024 {company_name}. All rights reserved.</p>
                <p>This email was sent because you were shortlisted for a position through our recruitment process.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create text version
    text_body = f"""
{subject}

Dear {candidate_name},

We are thrilled to inform you that you have been SHORTLISTED for the position of {extracted_role} at {company_name}!

Your application stood out among many candidates, and we believe you have the skills and experience we're looking for. We're impressed by your qualifications and would like to move forward with the next steps in our hiring process.

Position Details:
- Role: {extracted_role}
- Company: {company_name}
- Status: Shortlisted for Interview

What happens next?
- Our hiring team will review your application in detail
- We'll contact you within 2-3 business days to schedule an interview
- You may be asked to complete additional assessments or provide references
- We'll keep you updated throughout the process

Have Questions?
If you have any questions about the position or the next steps, please don't hesitate to contact us:
Email: {contact_email}

We're excited about the possibility of you joining our team and look forward to learning more about you in the upcoming interview.

Thank you for your interest in {company_name}!

Best regards,
The {company_name} Hiring Team

---
© 2024 {company_name}. All rights reserved.
This email was sent because you were shortlisted for a position through our recruitment process.
    """
    
    return send_email_via_smtp(to_email, subject, html_body, text_body)
