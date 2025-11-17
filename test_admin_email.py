"""
Test script to verify admin notification email sending
"""
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_admin_notification():
    """Test the admin notification email"""
    try:
        print("Testing admin notification email...")
        
        # Test import
        from app.utils import notify_admins_new_user
        print("[OK] Successfully imported notify_admins_new_user")
        
        # Test email sending
        print("\nSending test notification email...")
        notify_admins_new_user(
            email="test@example.com",
            role="recruiter",
            name="Test User"
        )
        
        print("[OK] Admin notification function called successfully!")
        print("[OK] Check vinit@adeptaipro.com for the email")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_admin_notification()

