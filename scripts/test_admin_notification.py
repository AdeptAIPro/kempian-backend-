#!/usr/bin/env python3
"""
Test script for admin notification email functionality
"""
import os
import sys
from dotenv import load_dotenv

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'backend', '.env'))

def test_admin_notification():
    """Test the admin notification email functionality"""
    try:
        from app.emails.smtp import send_admin_notification_email
        
        print("🧪 Testing Admin Notification Email...")
        print("=" * 50)
        
        # Test data
        test_email = "test.user@example.com"
        test_role = "employer"
        test_name = "John Doe"
        
        print(f"📧 Test User Email: {test_email}")
        print(f"👤 Test User Role: {test_role}")
        print(f"📝 Test User Name: {test_name}")
        print(f"📬 Admin Email: vinit@adeptaipro.com")
        print()
        
        # Send test notification
        success = send_admin_notification_email(test_email, test_role, test_name)
        
        if success:
            print("✅ SUCCESS: Admin notification email sent successfully!")
            print("📬 Check vinit@adeptaipro.com for the notification email")
        else:
            print("❌ FAILED: Admin notification email could not be sent")
            print("🔍 Check SMTP configuration and logs for details")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("🔍 Make sure SMTP credentials are configured in .env file")

def test_notify_admins_function():
    """Test the notify_admins_new_user function"""
    try:
        from app.utils import notify_admins_new_user
        
        print("\n🧪 Testing notify_admins_new_user Function...")
        print("=" * 50)
        
        # Test data
        test_email = "test.user2@example.com"
        test_role = "job_seeker"
        test_name = "Jane Smith"
        
        print(f"📧 Test User Email: {test_email}")
        print(f"👤 Test User Role: {test_role}")
        print(f"📝 Test User Name: {test_name}")
        print()
        
        # Call the function
        notify_admins_new_user(test_email, test_role, test_name)
        
        print("✅ SUCCESS: notify_admins_new_user function executed!")
        print("📬 Check vinit@adeptaipro.com for the notification email")
        print("📊 Check admin activity logs for the log entry")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    print("🚀 Kempian Admin Notification Test")
    print("=" * 50)
    
    # Test the email function directly
    test_admin_notification()
    
    # Test the utility function
    test_notify_admins_function()
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")
    print("📧 If successful, check vinit@adeptaipro.com for notification emails")
