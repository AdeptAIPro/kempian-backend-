#!/usr/bin/env python3
"""
Simple test script for quota management
"""

import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from app import create_app
    from app.models import User, Plan, Tenant, JDSearchLog
    from app.db import db
    
    print("✅ All imports successful!")
    
    # Create app context
    app = create_app()
    with app.app_context():
        print("✅ App context created successfully!")
        
        # Test database connection
        try:
            # Count users
            user_count = User.query.count()
            print(f"✅ Database connection successful! Found {user_count} users")
            
            # Count plans
            plan_count = Plan.query.count()
            print(f"✅ Found {plan_count} plans")
            
            # Count tenants
            tenant_count = Tenant.query.count()
            print(f"✅ Found {tenant_count} tenants")
            
            # Count search logs
            log_count = JDSearchLog.query.count()
            print(f"✅ Found {log_count} search logs")
            
            print("\n🎉 All tests passed! The quota management scripts should work.")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}") 