#!/usr/bin/env python3
"""
Test script for candidate shortlisted email functionality
"""

import os
import sys
import requests
import json
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_email_function():
    """Test the email function directly"""
    try:
        from backend.app.emails.smtp import send_candidate_shortlisted_email_smtp
        
        # Test data
        test_data = {
            'to_email': 'vinit@adeptaipro.com',
            'candidate_name': 'Vinit Kumar',
            'job_title': 'Senior Software Engineer',
            'company_name': 'Kempian AI',
            'job_description': 'We are looking for a Senior Software Engineer with 5+ years of experience in React, Node.js, and Python. The ideal candidate should have strong problem-solving skills and experience with cloud technologies.',
            'contact_email': 'hr@kempian.ai'
        }
        
        print("Testing email function...")
        print(f"Sending test email to: {test_data['to_email']}")
        print(f"Candidate: {test_data['candidate_name']}")
        print(f"Position: {test_data['job_title']}")
        print(f"Company: {test_data['company_name']}")
        print("-" * 50)
        
        # Send the email
        success = send_candidate_shortlisted_email_smtp(**test_data)
        
        if success:
            print("✅ Email sent successfully!")
            print("Check your inbox at vinit@adeptaipro.com")
        else:
            print("❌ Failed to send email")
            print("Please check your SMTP configuration in the .env file")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_api_endpoint():
    """Test the API endpoint"""
    try:
        # Test API endpoint
        api_url = "http://localhost:8000/talent/contact-candidate"
        
        # You'll need to get a valid JWT token for this test
        # For now, we'll just test the email function directly
        print("API endpoint test requires authentication token")
        print("Use the email function test instead")
        
    except Exception as e:
        print(f"❌ API test error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Candidate Shortlisted Email Functionality")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('backend'):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Test the email function
    test_email_function()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("\nTo test the full API endpoint:")
    print("1. Start the backend server: cd backend && python main.py")
    print("2. Get a valid JWT token from the frontend")
    print("3. Use the contact candidate button in the UI")
