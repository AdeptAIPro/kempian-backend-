"""
Contact Form API Endpoint
Handles contact form submissions and sends emails to admin@kempian.ai
"""

from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.emails.smtp import send_email_via_smtp
from datetime import datetime
import re

logger = get_logger('contact')
contact_bp = Blueprint('contact', __name__)

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Validate phone number format (basic validation)"""
    if not phone:
        return True  # Phone is optional
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    # Check if it has at least 10 digits
    return len(digits_only) >= 10

@contact_bp.route('/api/contact', methods=['POST'])
def submit_contact_form():
    """
    Handle contact form submission
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'subject', 'message']
        missing_fields = [field for field in required_fields if not data.get(field, '').strip()]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Extract form data
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        company = data.get('company', '').strip()
        phone = data.get('phone', '').strip()
        subject = data.get('subject', '').strip()
        department = data.get('department', '').strip()
        message = data.get('message', '').strip()
        
        # Validate email format
        if not validate_email(email):
            return jsonify({
                'success': False,
                'error': 'Invalid email format'
            }), 400
        
        # Validate phone format (if provided)
        if phone and not validate_phone(phone):
            return jsonify({
                'success': False,
                'error': 'Invalid phone number format'
            }), 400
        
        # Validate message length
        if len(message) < 10:
            return jsonify({
                'success': False,
                'error': 'Message must be at least 10 characters long'
            }), 400
        
        # Create email content
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # HTML email body
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>New Contact Form Submission - Kempian AI</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    line-height: 1.6; 
                    color: #2d3748; 
                    background-color: #f7fafc; 
                    margin: 0; 
                    padding: 20px; 
                }}
                .email-container {{ 
                    max-width: 650px; 
                    margin: 0 auto; 
                    background: #ffffff; 
                    border-radius: 16px; 
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1); 
                    overflow: hidden; 
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 40px 30px; 
                    text-align: center; 
                    position: relative;
                    overflow: hidden;
                }}
                .header::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="0.5" fill="white" opacity="0.1"/><circle cx="90" cy="40" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>') repeat;
                    opacity: 0.3;
                }}
                .header-content {{ position: relative; z-index: 1; }}
                .header h1 {{ 
                    font-size: 28px; 
                    font-weight: 700; 
                    margin-bottom: 8px; 
                    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .header .subtitle {{ 
                    font-size: 16px; 
                    opacity: 0.9; 
                    font-weight: 300;
                }}
                .timestamp {{ 
                    background: rgba(255, 255, 255, 0.2); 
                    padding: 8px 16px; 
                    border-radius: 20px; 
                    display: inline-block; 
                    margin-top: 12px; 
                    font-size: 14px; 
                    font-weight: 500;
                }}
                .content {{ 
                    padding: 40px 30px; 
                    background: #ffffff; 
                }}
                .field {{ 
                    margin-bottom: 24px; 
                    background: #f8fafc; 
                    border-radius: 12px; 
                    padding: 20px; 
                    border: 1px solid #e2e8f0;
                    transition: all 0.3s ease;
                }}
                .field:hover {{
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                    transform: translateY(-1px);
                }}
                .field-label {{ 
                    font-weight: 600; 
                    color: #4a5568; 
                    font-size: 14px; 
                    text-transform: uppercase; 
                    letter-spacing: 0.5px; 
                    margin-bottom: 8px; 
                    display: flex; 
                    align-items: center; 
                    gap: 8px;
                }}
                .field-value {{ 
                    color: #2d3748; 
                    font-size: 16px; 
                    font-weight: 500; 
                    word-break: break-word;
                }}
                .field-value a {{ 
                    color: #667eea; 
                    text-decoration: none; 
                    font-weight: 600;
                }}
                .field-value a:hover {{ 
                    text-decoration: underline; 
                }}
                .message-field {{ 
                    background: linear-gradient(135deg, #f0fff4 0%, #f7fafc 100%); 
                    border: 1px solid #c6f6d5; 
                }}
                .message-content {{ 
                    background: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    white-space: pre-wrap; 
                    font-size: 15px; 
                    line-height: 1.7; 
                    border-left: 4px solid #48bb78; 
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                }}
                .priority-badge {{
                    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                    color: white;
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    display: inline-block;
                    margin-left: 12px;
                }}
                .footer {{ 
                    background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); 
                    padding: 30px; 
                    text-align: center; 
                    color: #718096; 
                    font-size: 14px; 
                    border-top: 1px solid #e2e8f0;
                }}
                .footer p {{ 
                    margin-bottom: 8px; 
                }}
                .footer .brand {{ 
                    color: #667eea; 
                    font-weight: 600; 
                }}
                .action-buttons {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 24px;
                    margin: 0 8px;
                    border-radius: 8px;
                    text-decoration: none;
                    font-weight: 600;
                    font-size: 14px;
                    transition: all 0.3s ease;
                }}
                .btn-primary {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                .btn-secondary {{
                    background: #f7fafc;
                    color: #4a5568;
                    border: 1px solid #e2e8f0;
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                }}
                .divider {{
                    height: 1px;
                    background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%);
                    margin: 30px 0;
                }}
                @media (max-width: 600px) {{
                    .email-container {{ margin: 10px; border-radius: 12px; }}
                    .header, .content, .footer {{ padding: 20px; }}
                    .field {{ padding: 16px; }}
                    .header h1 {{ font-size: 24px; }}
                }}
            </style>
        </head>
        <body>
            <div class="email-container">
            <div class="header">
                    <div class="header-content">
                <h1>📧 New Contact Form Submission</h1>
                        <p class="subtitle">Kempian AI Contact Form</p>
                        <div class="timestamp">Received on {timestamp}</div>
                    </div>
            </div>
            
            <div class="content">
                <div class="field">
                        <div class="field-label">
                            👤 Contact Information
                        </div>
                        <div class="field-value">
                            <strong>{name}</strong>
                        </div>
                </div>
                
                <div class="field">
                        <div class="field-label">
                            📧 Email Address
                        </div>
                    <div class="field-value">
                            <a href="mailto:{email}">{email}</a>
                    </div>
                </div>
                
                    {f'<div class="field"><div class="field-label">🏢 Company</div><div class="field-value">{company}</div></div>' if company else ''}
                
                    {f'<div class="field"><div class="field-label">📞 Phone Number</div><div class="field-value">{phone}</div></div>' if phone else ''}
                
                <div class="field">
                        <div class="field-label">
                            📋 Subject
                            <span class="priority-badge">New Inquiry</span>
                        </div>
                    <div class="field-value">{subject}</div>
                </div>
                
                    {f'<div class="field"><div class="field-label">🏢 Department</div><div class="field-value">{department}</div></div>' if department else ''}
                    
                    <div class="divider"></div>
                
                <div class="field message-field">
                        <div class="field-label">
                            💬 Message Details
                        </div>
                    <div class="message-content">{message}</div>
                </div>
                    
                    <div class="action-buttons">
                        <a href="mailto:{email}?subject=Re: {subject}" class="btn btn-primary">
                            Reply to {name}
                        </a>
                        <a href="mailto:contact@kempian.ai" class="btn btn-secondary">
                            Forward to Team
                        </a>
                    </div>
            </div>
            
            <div class="footer">
                    <p>This message was sent from the <span class="brand">Kempian AI</span> contact form.</p>
                <p>Reply directly to this email to respond to the inquiry.</p>
                    <p style="margin-top: 16px; font-size: 12px; color: #a0aec0;">
                        © 2024 Kempian AI. All rights reserved.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Text email body
        text_body = f"""
New Contact Form Submission - Kempian AI
Received on: {timestamp}

Name: {name}
Email: {email}
{f'Company: {company}' if company else ''}
{f'Phone: {phone}' if phone else ''}
Subject: {subject}
{f'Department: {department}' if department else ''}

Message:
{message}

---
This message was sent from the Kempian AI contact form.
Reply directly to this email to respond to the inquiry.
        """
        
        # Send email to contact@kempian.ai
        email_subject = f"Contact Form: {subject} - {name}"
        success = send_email_via_smtp(
            to_email='contact@kempian.ai',
            subject=email_subject,
            body_html=html_body,
            body_text=text_body,
            reply_to=email  # Set reply-to header so replies go directly to the person
        )
        
        if success:
            logger.info(f"Contact form submitted successfully by {name} ({email})")
            return jsonify({
                'success': True,
                'message': 'Thank you for your message! We will get back to you within 24 hours.'
            })
        else:
            logger.error(f"Failed to send contact form email for {name} ({email})")
            return jsonify({
                'success': False,
                'error': 'Failed to send email. Please try again later.'
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing contact form: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your request. Please try again later.'
        }), 500

@contact_bp.route('/api/contact/health', methods=['GET'])
def contact_health():
    """Health check for contact service"""
    return jsonify({
        'status': 'healthy',
        'service': 'contact',
        'timestamp': datetime.now().isoformat()
    })
