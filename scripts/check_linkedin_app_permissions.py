"""
Script to check LinkedIn app permissions and provide guidance.
Run this to verify if your LinkedIn app has the required permissions.
"""
import os
import sys

def check_linkedin_permissions():
    """Check if LinkedIn app has required permissions"""
    
    print("=" * 60)
    print("LINKEDIN APP PERMISSIONS CHECKER")
    print("=" * 60)
    print()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    client_id = os.getenv('LINKEDIN_CLIENT_ID')
    client_secret = os.getenv('LINKEDIN_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("❌ ERROR: LinkedIn credentials not found in environment")
        print()
        print("Make sure you have:")
        print("  - LINKEDIN_CLIENT_ID")
        print("  - LINKEDIN_CLIENT_SECRET")
        print("  in your .env file")
        return
    
    print(f"✅ LinkedIn Client ID found: {client_id[:10]}...")
    print()
    
    print("-" * 60)
    print("HOW TO CHECK YOUR LINKEDIN APP PERMISSIONS:")
    print("-" * 60)
    print()
    
    print("1. Go to: https://www.linkedin.com/developers/apps")
    print()
    print(f"2. Find your app with Client ID: {client_id}")
    print()
    print("3. Click on your app and check the 'Products' tab")
    print()
    print("4. Look for these products:")
    print("   ✅ Marketing Developer Platform - REQUIRED for organization data")
    print("   ✅ Sign In with LinkedIn - For authentication")
    print()
    print("5. Check 'Auth' tab for redirect URIs:")
    print("   ✅ http://localhost:8081/linkedin-callback (development)")
    print("   ✅ https://your-domain.com/linkedin-callback (production)")
    print()
    
    print("-" * 60)
    print("IF 'Marketing Developer Platform' IS MISSING:")
    print("-" * 60)
    print()
    print("You need to apply for this product:")
    print()
    print("1. In your app, go to 'Products' tab")
    print("2. Find 'Marketing Developer Platform'")
    print("3. Click 'Request Access'")
    print("4. Fill out the application:")
    print("   - Use Case: 'Fetch job postings and applicants from company page'")
    print("   - Company Page: Select your organization's LinkedIn page")
    print("   - Website: Your company website")
    print("   - Privacy Policy: Your privacy policy URL")
    print("   - Terms: Your terms of service URL")
    print()
    print("5. Submit and wait 5-7 business days for approval")
    print()
    
    print("-" * 60)
    print("CURRENT REQUIRED SCOPES IN YOUR APP:")
    print("-" * 60)
    print()
    print("When connecting, these scopes are requested:")
    print("   - openid")
    print("   - profile")
    print("   - email")
    print("   - w_member_social")
    print("   - w_organization_social (REQUIRES Marketing Developer Platform)")
    print("   - r_organization_social (REQUIRES Marketing Developer Platform)")
    print()
    
    print("-" * 60)
    print("ALTERNATIVE: WORK WITHOUT ORGANIZATION DATA")
    print("-" * 60)
    print()
    print("If you don't want to wait for approval, you can:")
    print("1. Remove w_organization_social and r_organization_social scopes")
    print("2. Use only personal profile data")
    print("3. Skip fetching organization jobs/applicants")
    print()
    print("To do this, edit: src/components/linkedin/LinkedInIntegration.tsx")
    print("Change line ~130 to:")
    print('   const scope = "openid profile email w_member_social";')
    print()
    
    print("=" * 60)
    print("For detailed steps, see: LINKEDIN_ORGANIZATION_PERMISSIONS_GUIDE.md")
    print("=" * 60)

if __name__ == '__main__':
    try:
        check_linkedin_permissions()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

