"""
Script to send profile completion reminder email to a specific candidate email address.
Useful for manual sends, testing, or targeting specific users.

Usage:
    # Send to a specific email
    python -m scripts.send_profile_reminder_to_specific_user user@example.com
    
    # Or using direct path
    python backend/scripts/send_profile_reminder_to_specific_user.py user@example.com
    
    # Interactive mode (will prompt for email)
    python backend/scripts/send_profile_reminder_to_specific_user.py
"""

import sys
import os
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.models import db, User, CandidateProfile
from app.emails.smtp import send_email_via_smtp
from app.simple_logger import get_logger

logger = get_logger("profile_reminder_specific")

def is_profile_incomplete(user):
    """
    Check if a jobseeker's profile is incomplete.
    Returns True if profile is incomplete, False if complete.
    """
    # Check if user is a jobseeker
    if user.user_type != 'job_seeker' and user.role != 'job_seeker':
        return False
    
    # Check if user has a candidate profile
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    
    if not profile:
        # No profile at all - definitely incomplete
        return True
    
    # Check if profile has required fields
    # Profile is complete if it has: full_name, email (from user), and (skills OR experience_years)
    has_full_name = bool(profile.full_name and profile.full_name.strip())
    has_email = bool(user.email and user.email.strip())  # Email comes from User model
    
    # Check if has skills (via relationship) or experience_years
    has_skills = len(profile.skills) > 0 if profile.skills else False
    has_experience = profile.experience_years is not None
    
    # Profile is incomplete if missing any required field
    is_incomplete = not (has_full_name and has_email and (has_skills or has_experience))
    
    return is_incomplete

def send_profile_completion_reminder_email(to_email, user_name, login_method="email"):
    """
    Send a reminder email to jobseekers to complete their profile.
    
    Args:
        to_email: Email address of the jobseeker
        user_name: Name of the user (or email prefix if name not available)
        login_method: "google" or "email" - how they logged in
    """
    try:
        # Determine greeting name
        display_name = user_name if user_name else to_email.split('@')[0]
        first_name = display_name.split(' ')[0] if ' ' in display_name else display_name
        
        # Subject line
        subject = "📝 Complete Your Profile on Kempian AI - Unlock More Opportunities!"
        
        # Login method specific messaging
        login_context = ""
        if login_method == "google":
            login_context = "We noticed you signed up with Google, but your profile isn't complete yet."
        else:
            login_context = "We noticed you created an account, but your profile isn't complete yet."
        
        # HTML email body
        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Complete Your Profile</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center; color: white; }}
                .header h1 {{ margin: 0; font-size: 28px; font-weight: 600; }}
                .content {{ padding: 40px 30px; }}
                .reminder-box {{ background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 25px; border-radius: 12px; margin-bottom: 30px; text-align: center; }}
                .reminder-box h2 {{ margin: 0 0 15px 0; font-size: 24px; }}
                .reminder-box p {{ margin: 0; font-size: 16px; opacity: 0.9; }}
                .info-box {{ background: #f8f9fa; border-left: 4px solid #667eea; padding: 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
                .benefits {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
                .benefit {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                .benefit-icon {{ font-size: 32px; margin-bottom: 10px; }}
                .cta-button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; margin: 20px 0; }}
                .cta-button:hover {{ background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%); }}
                .steps {{ background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 25px 0; }}
                .steps h4 {{ margin-top: 0; color: #1976d2; }}
                .steps ul {{ margin: 10px 0; padding-left: 20px; }}
                .footer {{ background: #f8f9fa; padding: 30px; text-align: center; color: #6c757d; font-size: 14px; }}
                .highlight {{ color: #667eea; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📝 Complete Your Profile</h1>
                </div>
                
                <div class="content">
                    <div class="reminder-box">
                        <h2>Your Profile Needs Attention</h2>
                        <p>Complete your profile to unlock all opportunities on Kempian AI</p>
                    </div>
                    
                    <p>Hi <strong>{first_name}</strong>,</p>
                    
                    <p>{login_context} To get the most out of Kempian AI and connect with top employers, you'll need to complete your profile.</p>
                    
                    <div class="info-box">
                        <h3>Why complete your profile?</h3>
                        <p>Employers are actively searching for candidates like you, but they can only find you if your profile is complete. A complete profile helps you:</p>
                    </div>
                    
                    <div class="benefits">
                        <div class="benefit">
                            <div class="benefit-icon">🎯</div>
                            <h3>Get Matched</h3>
                            <p>Our AI matches you with relevant job opportunities</p>
                        </div>
                        <div class="benefit">
                            <div class="benefit-icon">👀</div>
                            <h3>Be Discovered</h3>
                            <p>Employers can find and contact you directly</p>
                        </div>
                        <div class="benefit">
                            <div class="benefit-icon">📈</div>
                            <h3>Stand Out</h3>
                            <p>Complete profiles get 3x more views from employers</p>
                        </div>
                        <div class="benefit">
                            <div class="benefit-icon">⚡</div>
                            <h3>Save Time</h3>
                            <p>Apply faster with your profile pre-filled</p>
                        </div>
                    </div>
                    
                    <div class="steps">
                        <h4>🚀 Quick Steps to Complete Your Profile:</h4>
                        <ul>
                            <li><strong>Upload Your Resume</strong> - We'll automatically extract your skills and experience</li>
                            <li><strong>Add Your Skills</strong> - Highlight what you're good at</li>
                            <li><strong>Update Your Experience</strong> - Showcase your work history</li>
                            <li><strong>Add Your Education</strong> - Include your qualifications</li>
                            <li><strong>Set Your Preferences</strong> - Tell us what you're looking for</li>
                        </ul>
                    </div>
                    
                    <div style="text-align: center;">
                        <a href="https://kempian.ai/oauth-resume-upload" class="cta-button">
                            Complete My Profile Now
                        </a>
                    </div>
                    
                    <p>It only takes a few minutes to complete your profile, and it could be the key to your next career opportunity!</p>
                    
                    <p>If you have any questions or need help, our support team is here to assist you at <a href="mailto:support@kempian.ai">support@kempian.ai</a>.</p>
                    
                    <p>Best regards,<br>
                    <strong>The Kempian AI Team</strong></p>
                </div>
                
                <div class="footer">
                    <p>© 2024 Kempian AI. All rights reserved.</p>
                    <p>You received this email because your profile on Kempian AI is incomplete.</p>
                    <p>Don't want to receive these reminders? <a href="https://kempian.ai/profile">Update your preferences</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text version
        body_text = f"""
Complete Your Profile on Kempian AI - Unlock More Opportunities!

Hi {first_name},

{login_context} To get the most out of Kempian AI and connect with top employers, you'll need to complete your profile.

Why complete your profile?
Employers are actively searching for candidates like you, but they can only find you if your profile is complete. A complete profile helps you:

- Get Matched: Our AI matches you with relevant job opportunities
- Be Discovered: Employers can find and contact you directly
- Stand Out: Complete profiles get 3x more views from employers
- Save Time: Apply faster with your profile pre-filled

Quick Steps to Complete Your Profile:
1. Upload Your Resume - We'll automatically extract your skills and experience
2. Add Your Skills - Highlight what you're good at
3. Update Your Experience - Showcase your work history
4. Add Your Education - Include your qualifications
5. Set Your Preferences - Tell us what you're looking for

Complete your profile now: https://kempian.ai/oauth-resume-upload

It only takes a few minutes to complete your profile, and it could be the key to your next career opportunity!

If you have any questions or need help, our support team is here to assist you at support@kempian.ai.

Best regards,
The Kempian AI Team

---
© 2024 Kempian AI. All rights reserved.
You received this email because your profile on Kempian AI is incomplete.
Don't want to receive these reminders? Update your preferences: https://kempian.ai/profile
        """
        
        # Send the email
        success = send_email_via_smtp(
            to_email=to_email,
            subject=subject,
            body_html=body_html,
            body_text=body_text
        )
        
        if success:
            logger.info(f"[PROFILE REMINDER] Successfully sent reminder email to {to_email}")
        else:
            logger.error(f"[PROFILE REMINDER] Failed to send reminder email to {to_email}")
        
        return success
        
    except Exception as e:
        logger.error(f"[PROFILE REMINDER] Error sending email to {to_email}: {str(e)}")
        return False

def determine_login_method(user):
    """
    Determine how the user logged in (Google OAuth or email).
    
    Args:
        user: User object
    
    Returns:
        "google" or "email"
    """
    # Check if user has linkedin_id (OAuth indicator) or password_hash
    # Google OAuth users typically don't have password_hash
    if user.linkedin_id:
        return "google"
    elif not user.password_hash:
        # No password hash might indicate OAuth
        return "google"
    else:
        return "email"

def send_reminder_to_email(email_address, force_send=False):
    """
    Send profile completion reminder to a specific email address.
    
    Args:
        email_address: Email address of the candidate
        force_send: If True, send email even if profile is complete
    
    Returns:
        dict with status information
    """
    try:
        # Find user by email
        user = User.query.filter_by(email=email_address).first()
        
        if not user:
            return {
                'success': False,
                'error': f'No user found with email: {email_address}',
                'email': email_address
            }
        
        # Check if user is a jobseeker
        if user.user_type != 'job_seeker' and user.role != 'job_seeker':
            return {
                'success': False,
                'error': f'User {email_address} is not a jobseeker (user_type: {user.user_type}, role: {user.role})',
                'email': email_address,
                'user_type': user.user_type,
                'role': user.role
            }
        
        # Check profile completion status
        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        is_incomplete = is_profile_incomplete(user)
        
        if not is_incomplete and not force_send:
            return {
                'success': False,
                'error': f'Profile for {email_address} is already complete. Use force_send=True to send anyway.',
                'email': email_address,
                'has_profile': profile is not None,
                'is_complete': True
            }
        
        # Determine login method
        login_method = determine_login_method(user)
        
        # Get user name
        user_name = user.email.split('@')[0]  # Default to email prefix
        if profile and profile.full_name:
            user_name = profile.full_name
        
        # Send the email
        success = send_profile_completion_reminder_email(
            to_email=user.email,
            user_name=user_name,
            login_method=login_method
        )
        
        if success:
            return {
                'success': True,
                'message': f'Reminder email sent successfully to {email_address}',
                'email': email_address,
                'user_name': user_name,
                'login_method': login_method,
                'has_profile': profile is not None,
                'is_incomplete': is_incomplete
            }
        else:
            return {
                'success': False,
                'error': f'Failed to send email to {email_address}',
                'email': email_address
            }
            
    except Exception as e:
        logger.error(f"[PROFILE REMINDER] Error processing email {email_address}: {str(e)}")
        return {
            'success': False,
            'error': f'Error: {str(e)}',
            'email': email_address
        }

def main():
    """
    Main function to send reminder email to a specific candidate.
    """
    app = create_app()
    
    with app.app_context():
        # Get email from command line argument or prompt
        email_address = None
        
        if len(sys.argv) > 1:
            email_address = sys.argv[1].strip()
        else:
            # Interactive mode - prompt for email
            print("\n" + "="*60)
            print("Profile Completion Reminder - Send to Specific User")
            print("="*60)
            email_address = input("\nEnter candidate email address: ").strip()
        
        if not email_address:
            print("Error: Email address is required")
            print("\nUsage:")
            print("  python -m scripts.send_profile_reminder_to_specific_user user@example.com")
            print("  python backend/scripts/send_profile_reminder_to_specific_user.py user@example.com")
            sys.exit(1)
        
        # Check for force flag
        force_send = '--force' in sys.argv or '-f' in sys.argv
        
        logger.info(f"[PROFILE REMINDER] Sending reminder to specific user: {email_address}")
        if force_send:
            logger.info("[PROFILE REMINDER] Force send enabled - will send even if profile is complete")
        
        # Send the reminder
        result = send_reminder_to_email(email_address, force_send=force_send)
        
        # Print results
        print("\n" + "="*60)
        print("Result:")
        print("="*60)
        
        if result['success']:
            print(f"✓ SUCCESS: {result['message']}")
            print(f"  Email: {result['email']}")
            print(f"  User Name: {result.get('user_name', 'N/A')}")
            print(f"  Login Method: {result.get('login_method', 'N/A')}")
            print(f"  Has Profile: {result.get('has_profile', False)}")
            print(f"  Profile Incomplete: {result.get('is_incomplete', False)}")
            logger.info(f"[PROFILE REMINDER] ✓ Successfully sent to {email_address}")
        else:
            print(f"✗ FAILED: {result['error']}")
            if 'user_type' in result:
                print(f"  User Type: {result.get('user_type')}")
                print(f"  Role: {result.get('role')}")
            if 'is_complete' in result:
                print(f"  Profile Complete: {result.get('is_complete')}")
            logger.error(f"[PROFILE REMINDER] ✗ Failed to send to {email_address}: {result['error']}")
        
        print("="*60 + "\n")
        
        # Exit with appropriate code
        sys.exit(0 if result['success'] else 1)

if __name__ == "__main__":
    main()

