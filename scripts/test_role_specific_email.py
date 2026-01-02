import os
import sys

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # Basic sanity check for SMTP envs
    required = [
        'SMTP_SERVER', 'SMTP_PORT', 'SMTP_USERNAME', 'SMTP_PASSWORD', 'SMTP_FROM_EMAIL'
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"Missing SMTP env vars: {', '.join(missing)}")
        sys.exit(1)

    to_email = os.getenv('TEST_TO_EMAIL', 'vinit@adeptaipro.com')
    user_name = os.getenv('TEST_USER_NAME', 'Vinit')
    test_role = os.getenv('TEST_ROLE', 'employer')  # employer, recruiter, job_seeker, admin

    from app.emails.smtp import send_welcome_email_smtp

    print(f"Sending role-specific welcome email to {to_email}...")
    print(f"Role: {test_role}")
    print(f"User: {user_name}")
    
    ok = send_welcome_email_smtp(to_email, user_name, test_role)
    if ok:
        print(f"SUCCESS: Role-specific welcome email sent to {to_email}")
        print(f"\nEmail includes:")
        print(f"  • Role-specific subject and content for {test_role}")
        print(f"  • 4 symmetric features (AI Matching, Lightning Fast, Smart Automation, App Store)")
        print(f"  • YouTube video thumbnail (ID: 3KKOBI_Qz7w)")
        print(f"  • Role-specific CTA button and link")
        sys.exit(0)
    else:
        print(f"FAILED: Could not send email to {to_email}")
        sys.exit(2)

if __name__ == '__main__':
    main()
