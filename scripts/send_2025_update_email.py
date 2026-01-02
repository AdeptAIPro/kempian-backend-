#!/usr/bin/env python3
"""
Simple script to send the 2025 Kempian.ai update email
Uses the existing backend email service (app.emails.smtp)
Make sure your .env file has:
SMTP_SERVER=smtp.hostinger.com
SMTP_PORT=587
SMTP_USERNAME=contact@kempian.ai
SMTP_PASSWORD=your_password
SMTP_FROM_EMAIL=contact@kempian.ai
"""

import os
import sys

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# HTML email content from Email 3.html (with responsive fixes)
HTML_EMAIL_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Kempian.ai 2025 Update</title>
    <!--[if mso]>
    <style type="text/css">
        body, table, td {font-family: Arial, sans-serif !important;}
    </style>
    <![endif]-->
</head>
<body style="margin: 0; padding: 0; background-color: #f5f7fa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; -webkit-text-size-adjust: 100%; -ms-text-size-adjust: 100%;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color: #f5f7fa; padding: 20px 10px;">
        <tr>
            <td align="center" style="padding: 0;">
                <table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" style="background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); max-width: 600px; width: 100%;">
                    
                    <!-- Header with gradient -->
                     <tr>
                        <td style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 30px 20px 25px 20px; text-align: left;">
                           <img
                                src="https://raw.githubusercontent.com/abhayawach/photo/refs/heads/main/Kempian_%20logo1.png"
                                alt="Kempian.ai"
                                width="200"
                                style="max-width: 200px; width: 100%; height: auto; display: block;"
                            />

                            <div style="font-size: 14px; color: rgba(255,255,255,0.9); font-weight: 500; margin-top: 10px;">2025 Year in Review</div>
                        </td>
                    </tr>
                    
                    <!-- Main Content -->
                    <tr>
                        <td style="padding: 30px 20px;">
                            
                            <p style="margin: 0 0 20px 0; font-size: 15px; line-height: 1.6; color: #2d3748;">Hello,</p>
                            
                            <p style="margin: 0 0 25px 0; font-size: 15px; line-height: 1.6; color: #2d3748;">As the year comes to a close, I wanted to share a short update on how Kempian.ai has progressed in 2025.</p>
                            
                            <p style="margin: 0 0 25px 0; font-size: 15px; line-height: 1.6; color: #2d3748;">We are now positioned as an <strong style="color: #1a202c;">AI-Powered Workforce Intelligence Platform</strong>, focused on supporting enterprise workforce decisions with explainable, system-agnostic intelligence.</p>
                            
                            <!-- Section: This year we -->
                            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 25px;">
                                <tr>
                                    <td>
                                        <div style="display: inline-block; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: #ffffff; font-size: 12px; font-weight: 600; padding: 6px 12px; border-radius: 6px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">This year, we:</div>
                                        <div style="background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 15px 18px; border-radius: 6px;">
                                            <p style="margin: 0 0 12px 0; font-size: 14px; line-height: 1.5; color: #2d3748;">
                                                <span style="color: #3b82f6; font-weight: 700; margin-right: 6px;">✓</span>Advanced from MVP to <strong>enterprise-grade platform design</strong>
                                            </p>
                                            <p style="margin: 0 0 12px 0; font-size: 14px; line-height: 1.5; color: #2d3748;">
                                                <span style="color: #3b82f6; font-weight: 700; margin-right: 6px;">✓</span>Began building a unified <strong>Source → Hire → Pay</strong> workflow
                                            </p>
                                            <p style="margin: 0 0 12px 0; font-size: 14px; line-height: 1.5; color: #2d3748;">
                                                <span style="color: #3b82f6; font-weight: 700; margin-right: 6px;">✓</span>Integrated <strong>AI Talent Matchmaking, ATS, Payroll, and Compliance</strong>
                                            </p>
                                            <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #2d3748;">
                                                <span style="color: #3b82f6; font-weight: 700; margin-right: 6px;">✓</span>Ran live pilots and demos with enterprise teams
                                            </p>
                                        </div>
                                    </td>
                                </tr>
                            </table>
                            
                            <!-- Section: Why it's important -->
                            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 25px;">
                                <tr>
                                    <td>
                                        <div style="display: inline-block; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: #ffffff; font-size: 12px; font-weight: 600; padding: 6px 12px; border-radius: 6px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Why it's important</div>
                                        <p style="margin: 0; font-size: 15px; line-height: 1.6; color: #2d3748;">Kempian is shaping up as an intelligence layer for workforce operations helping organizations reduce risk, improve decision quality, and operate more efficiently at scale.</p>
                                    </td>
                                </tr>
                            </table>
                            
                            <!-- Founder's reflection -->
                            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 25px;">
                                <tr>
                                    <td>
                                        <div style="background-color: #dbeafe; padding: 20px; border-radius: 8px; border-left: 4px solid #1e3a8a;">
                                            <p style="margin: 0; font-size: 15px; line-height: 1.6; color: #2d3748; font-style: italic;">From a founder's standpoint, 2025 was about execution discipline. The company now behaves less like a product experiment and more like long-term infrastructure.</p>
                                        </div>
                                    </td>
                                </tr>
                            </table>
                            
                            <p style="margin: 0 0 20px 0; font-size: 15px; line-height: 1.6; color: #2d3748;">If you'd like to stay close to the journey or explore involvement, you can find more here:</p>
                            
                            <!-- CTA Section -->
                            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 25px;">
                                <tr>
                                    <td align="center" style="padding: 18px; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 8px; border: 2px solid #3b82f6;">
                                        <div style="font-size: 13px; color: #1e3a8a; font-weight: 600; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Stay Connected</div>
                                        <a href="https://wefunder.com/kempian.ai" style="display: inline-block; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: #ffffff; text-decoration: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; font-size: 15px; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);">View Our WeFunder Page →</a>
                                    </td>
                                </tr>
                            </table>
                            
                            <!-- Signature -->
                            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top: 30px; padding-top: 25px; border-top: 2px solid #e2e8f0;">
                                <tr>
                                    <td>
                                        <p style="margin: 0 0 8px 0; font-size: 15px; color: #4a5568;">Thanks for staying connected,</p>
                                        <p style="margin: 0 0 4px 0; font-size: 17px; font-weight: 700; color: #1a202c;">Abhishek Mathur</p>
                                        <p style="margin: 0 0 12px 0; font-size: 14px; color: #718096;">Founder & CTO, <span style="color: #3b82f6; font-weight: 600;">Kempian.ai</span></p>
                                        <a href="https://www.linkedin.com/company/kempian-ai/" style="display: inline-block; color: #3b82f6; text-decoration: none; font-size: 14px; font-weight: 500; margin-top: 8px;">
                                            <span style="background-color: #eff6ff; padding: 6px 12px; border-radius: 6px; border: 1px solid #3b82f6;">🔗 Connect on LinkedIn</span>
                                        </a>
                                    </td>
                                </tr>
                            </table>
                            
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f7fafc; padding: 20px; text-align: center; border-top: 1px solid #e2e8f0;">
                            <p style="margin: 0; font-size: 12px; color: #718096;">© 2025 Kempian.ai • Building the future of workforce intelligence</p>
                        </td>
                    </tr>
                    
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""

# Plain text version
TEXT_EMAIL_CONTENT = """Kempian.ai 2025 Year in Review

Hello,

As the year comes to a close, We wanted to share a short update on how Kempian.ai has progressed in 2025.

We are now positioned as an AI-Powered Workforce Intelligence Platform, focused on supporting enterprise workforce decisions with explainable, system-agnostic intelligence.

THIS YEAR, WE:
✓ Advanced from MVP to enterprise-grade platform design
✓ Began building a unified Source → Hire → Pay workflow
✓ Integrated AI Talent Matchmaking, ATS, Payroll, and Compliance
✓ Ran live pilots and demos with enterprise teams

WHY IT'S IMPORTANT
Kempian is shaping up as an intelligence layer for workforce operations helping organizations reduce risk, improve decision quality, and operate more efficiently at scale.

From a founder's standpoint, 2025 was about execution discipline. The company now behaves less like a product experiment and more like long-term infrastructure.

If you'd like to stay close to the journey or explore involvement, you can find more here:
Stay Connected: https://wefunder.com/kempian.ai

Thanks for staying connected,
Abhishek Mathur
Founder & CTO, Kempian.ai
Connect on LinkedIn: https://www.linkedin.com/company/kempian-ai/

© 2025 Kempian.ai • Building the future of workforce intelligence
"""


def main():
    """Main function to send the 2025 update email using existing backend service"""
    print("=" * 60)
    print("Kempian.ai 2025 Update Email Sender")
    print("Using existing backend email service")
    print("=" * 60)
    print()
    
    # Import the existing email service - exactly as used elsewhere
    try:
        from app.emails.smtp import send_email_via_smtp
    except ImportError as e:
        print(f"✗ Error: Could not import email service: {e}")
        print("Make sure you're running this from the backend directory")
        return
    
    # Check credentials from environment (same as existing flow)
    smtp_username = os.getenv('SMTP_USERNAME', '')
    smtp_password = os.getenv('SMTP_PASSWORD', '')
    
    if not smtp_username or not smtp_password:
        print("⚠ Error: SMTP credentials not found in environment variables.")
        print("\nPlease ensure your .env file contains:")
        print("  SMTP_SERVER=smtp.hostinger.com")
        print("  SMTP_PORT=587")
        print("  SMTP_USERNAME=contact@kempian.ai")
        print("  SMTP_PASSWORD=your_password")
        print("  SMTP_FROM_EMAIL=contact@kempian.ai")
        return
    
    print(f"SMTP Server: {os.getenv('SMTP_SERVER', 'smtp.hostinger.com')}")
    print(f"SMTP Port: {os.getenv('SMTP_PORT', '587')}")
    print(f"SMTP Username: {smtp_username}")
    print(f"From Email: {os.getenv('SMTP_FROM_EMAIL', smtp_username)}")
    print()
    print("=" * 60)
    print("Sending emails...")
    print("=" * 60)
    print()
    
    # Recipients
    recipients = [
        "vinit@adeptaipro.com",
        "abhi@adeptaipro.com"
    ]
    
    subject = "Kempian.ai 2025 Year in Review"
    
    # Send email to each recipient - using exact same method as other working emails
    success_count = 0
    for recipient in recipients:
        print(f"Sending email to {recipient}...")
        try:
            # Call the function exactly as it's called in other working places
            result = send_email_via_smtp(
                to_email=recipient,
                subject=subject,
                body_html=HTML_EMAIL_CONTENT,
                body_text=TEXT_EMAIL_CONTENT
            )
            
            if result:
                print(f"✓ Email sent successfully to {recipient}")
                success_count += 1
            else:
                print(f"✗ Failed to send email to {recipient}")
        except Exception as e:
            print(f"✗ Error sending to {recipient}: {str(e)}")
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total recipients: {len(recipients)}")
    print(f"Successfully sent: {success_count}")
    print(f"Failed: {len(recipients) - success_count}")
    print()
    
    if success_count == len(recipients):
        print("✓ All emails sent successfully!")
    else:
        print("✗ Some emails failed to send.")
        print("\nIf other emails work but this one doesn't, the issue might be:")
        print("1. Email content size or formatting")
        print("2. Special characters in the HTML")
        print("3. Server-side content filtering")


if __name__ == "__main__":
    main()
