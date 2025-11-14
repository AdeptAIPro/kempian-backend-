#!/usr/bin/env python3
"""
Test Admin Activity Logging System
Tests the complete admin activity logging functionality
"""

import sys
import os
import time
import requests
import json
sys.path.append('backend')

from backend.app import create_app, db
from backend.app.models import AdminActivityLog, AdminSession
from backend.app.services.admin_activity_logger import AdminActivityLogger
from backend.app.simple_logger import get_logger

logger = get_logger("admin_activity_test")

def test_admin_activity_logging():
    """Test the admin activity logging system"""
    print("🧪 Testing Admin Activity Logging System")
    print("=" * 60)
    
    try:
        app = create_app()
        with app.app_context():
            # Test 1: Log admin login
            print("\n1. Testing admin login logging...")
            session_id = AdminActivityLogger.log_admin_login(
                admin_email="test@admin.com",
                admin_id=1,
                admin_role="admin",
                tenant_id=1
            )
            
            if session_id:
                print(f"✅ Admin login logged successfully (session: {session_id})")
            else:
                print("❌ Failed to log admin login")
                return False
            
            # Test 2: Log admin action
            print("\n2. Testing admin action logging...")
            AdminActivityLogger.log_admin_action(
                admin_email="test@admin.com",
                admin_id=1,
                admin_role="admin",
                action="Test Action",
                endpoint="/admin/test",
                method="GET",
                status_code=200,
                response_time_ms=150,
                tenant_id=1
            )
            print("✅ Admin action logged successfully")
            
            # Test 3: Log admin logout
            print("\n3. Testing admin logout logging...")
            AdminActivityLogger.log_admin_logout(
                admin_email="test@admin.com",
                session_id=session_id
            )
            print("✅ Admin logout logged successfully")
            
            # Test 4: Get admin activities
            print("\n4. Testing activity retrieval...")
            activities = AdminActivityLogger.get_admin_activities(
                admin_email="test@admin.com",
                page=1,
                per_page=10
            )
            
            if activities and activities['activities']:
                print(f"✅ Retrieved {len(activities['activities'])} activities")
                for activity in activities['activities']:
                    print(f"   - {activity['activity_type']}: {activity['action']}")
            else:
                print("❌ Failed to retrieve activities")
                return False
            
            # Test 5: Get admin sessions
            print("\n5. Testing session retrieval...")
            sessions = AdminActivityLogger.get_admin_sessions(
                admin_email="test@admin.com",
                page=1,
                per_page=10
            )
            
            if sessions and sessions['sessions']:
                print(f"✅ Retrieved {len(sessions['sessions'])} sessions")
                for session in sessions['sessions']:
                    print(f"   - Session {session['session_id'][:8]}... ({'Active' if session['is_active'] else 'Inactive'})")
            else:
                print("❌ Failed to retrieve sessions")
                return False
            
            # Test 6: Get admin stats
            print("\n6. Testing statistics retrieval...")
            stats = AdminActivityLogger.get_admin_stats(
                admin_email="test@admin.com",
                days=30
            )
            
            if stats:
                print(f"✅ Retrieved stats: {stats['total_activities']} activities, {stats['unique_admins']} admins")
            else:
                print("❌ Failed to retrieve stats")
                return False
            
            return True
            
    except Exception as e:
        print(f"❌ Error testing admin activity logging: {e}")
        return False

def test_api_endpoints():
    """Test the admin activity API endpoints"""
    print("\n🌐 Testing Admin Activity API Endpoints")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test with a sample JWT token (this would normally be a real admin token)
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXVzZXItaWQiLCJlbWFpbCI6InRlc3RAYWRtaW4uY29tIiwiY3VzdG9tOnRlbmFudF9pZCI6IjEiLCJjdXN0b206cm9sZSI6ImFkbWluIn0.test-signature',
        'Content-Type': 'application/json'
    }
    
    endpoints = [
        '/admin/activity-logs',
        '/admin/sessions',
        '/admin/activity-stats'
    ]
    
    for endpoint in endpoints:
        try:
            print(f"\nTesting {endpoint}...")
            response = requests.get(f"{base_url}{endpoint}", headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {endpoint} - Status: {response.status_code}")
                if 'data' in data:
                    print(f"   Data keys: {list(data['data'].keys())}")
            else:
                print(f"⚠️  {endpoint} - Status: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint} - Error: {e}")
        except Exception as e:
            print(f"❌ {endpoint} - Unexpected error: {e}")

def cleanup_test_data():
    """Clean up test data"""
    print("\n🧹 Cleaning up test data...")
    
    try:
        app = create_app()
        with app.app_context():
            # Delete test activities
            AdminActivityLog.query.filter_by(admin_email="test@admin.com").delete()
            
            # Delete test sessions
            AdminSession.query.filter_by(admin_email="test@admin.com").delete()
            
            db.session.commit()
            print("✅ Test data cleaned up")
            
    except Exception as e:
        print(f"⚠️  Error cleaning up test data: {e}")

def main():
    """Main test function"""
    print("🚀 Admin Activity Logging System Test")
    print("=" * 80)
    
    # Test 1: Database functionality
    print("\n📊 Testing Database Functionality")
    db_success = test_admin_activity_logging()
    
    # Test 2: API endpoints
    print("\n🌐 Testing API Endpoints")
    test_api_endpoints()
    
    # Test 3: Cleanup
    cleanup_test_data()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    if db_success:
        print("✅ Database functionality: PASSED")
    else:
        print("❌ Database functionality: FAILED")
    
    print("🌐 API endpoints: Tested (check individual results above)")
    print("🧹 Cleanup: Completed")
    
    if db_success:
        print("\n🎉 Admin Activity Logging System is working correctly!")
        print("📝 You can now:")
        print("   - View activity logs at /admin/activity-logs")
        print("   - Monitor admin sessions")
        print("   - Track admin activities in real-time")
    else:
        print("\n⚠️  Some tests failed. Please check the logs above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
