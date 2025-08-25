#!/usr/bin/env python3
"""
Test script for the KPI system
Run this after setting up the KPI tables to verify everything works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models import db, User, UserKPIs, UserSkillGap, UserLearningPath

def test_kpi_system():
    """Test the KPI system functionality"""
    app = create_app()
    
    with app.app_context():
        try:
            print("🧪 Testing KPI System...")
            print("=" * 50)
            
            # Test 1: Check if tables exist
            print("1. Checking database tables...")
            try:
                # Try to query each table
                kpi_count = UserKPIs.query.count()
                skill_gap_count = UserSkillGap.query.count()
                learning_path_count = UserLearningPath.query.count()
                
                print(f"   ✅ UserKPIs table: {kpi_count} records")
                print(f"   ✅ UserSkillGap table: {skill_gap_count} records")
                print(f"   ✅ UserLearningPath table: {learning_path_count} records")
                
            except Exception as e:
                print(f"   ❌ Table check failed: {str(e)}")
                return False
            
            # Test 2: Check if users exist
            print("\n2. Checking user data...")
            users = User.query.limit(5).all()
            if users:
                print(f"   ✅ Found {len(users)} users")
                for user in users:
                    print(f"      - User {user.id}: {user.email} ({user.role})")
            else:
                print("   ⚠️  No users found in database")
            
            # Test 3: Check KPI data
            print("\n3. Checking KPI data...")
            kpis = UserKPIs.query.all()
            if kpis:
                print(f"   ✅ Found {len(kpis)} KPI records")
                for kpi in kpis:
                    print(f"      - User {kpi.user_id}: Role Fit {kpi.role_fit_score}%, Benchmark {kpi.career_benchmark}")
            else:
                print("   ⚠️  No KPI records found")
            
            # Test 4: Check skill gaps
            print("\n4. Checking skill gaps...")
            skill_gaps = UserSkillGap.query.all()
            if skill_gaps:
                print(f"   ✅ Found {len(skill_gaps)} skill gap records")
                for gap in skill_gaps:
                    print(f"      - {gap.skill_name}: {gap.current_level}% → {gap.target_level}% ({gap.priority} priority)")
            else:
                print("   ⚠️  No skill gap records found")
            
            # Test 5: Check learning paths
            print("\n5. Checking learning paths...")
            learning_paths = UserLearningPath.query.all()
            if learning_paths:
                print(f"   ✅ Found {len(learning_paths)} learning path records")
                for path in learning_paths:
                    print(f"      - {path.pathway_name}: {path.progress}% complete ({path.total_duration})")
            else:
                print("   ⚠️  No learning path records found")
            
            # Test 6: API endpoint test simulation
            print("\n6. Simulating API endpoint logic...")
            try:
                from app.analytics.kpi_routes import create_default_user_kpis, calculate_role_fit_score
                
                if users:
                    test_user = users[0]
                    print(f"   Testing with user: {test_user.email}")
                    
                    # Test KPI creation
                    user_kpis = UserKPIs.query.filter_by(user_id=test_user.id).first()
                    if not user_kpis:
                        print("   Creating default KPIs for test user...")
                        user_kpis = create_default_user_kpis(test_user)
                        print(f"   ✅ Created KPIs: Role Fit {user_kpis.role_fit_score}%")
                    else:
                        print(f"   ✅ Existing KPIs: Role Fit {user_kpis.role_fit_score}%")
                    
                    # Test role fit calculation
                    role_fit = calculate_role_fit_score(test_user)
                    print(f"   ✅ Calculated Role Fit: {role_fit}%")
                    
                else:
                    print("   ⚠️  Skipping API simulation (no users)")
                    
            except Exception as e:
                print(f"   ❌ API simulation failed: {str(e)}")
            
            print("\n" + "=" * 50)
            print("🎉 KPI System Test Complete!")
            
            # Summary
            print("\n📊 Summary:")
            print(f"   - Database tables: ✅ Working")
            print(f"   - User data: {'✅ Found' if users else '⚠️  None'}")
            print(f"   - KPI data: {'✅ Found' if kpis else '⚠️  None'}")
            print(f"   - Skill gaps: {'✅ Found' if skill_gaps else '⚠️  None'}")
            print(f"   - Learning paths: {'✅ Found' if learning_paths else '⚠️  None'}")
            
            if not users:
                print("\n💡 Recommendation: Create some test users first to see full KPI functionality")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            return False

def create_test_user():
    """Create a test user if none exist"""
    app = create_app()
    
    with app.app_context():
        try:
            users = User.query.limit(1).all()
            if not users:
                print("Creating test user...")
                
                # Create a test user
                test_user = User(
                    tenant_id=1,  # Assuming tenant 1 exists
                    email="test@example.com",
                    role="job_seeker",
                    user_type="job_seeker"
                )
                db.session.add(test_user)
                db.session.commit()
                
                print("✅ Test user created: test@example.com")
                return True
            else:
                print("✅ Users already exist")
                return True
                
        except Exception as e:
            print(f"❌ Failed to create test user: {str(e)}")
            return False

if __name__ == "__main__":
    print("🚀 KPI System Test Runner")
    print("=" * 50)
    
    # Create test user if needed
    if create_test_user():
        # Run the tests
        test_kpi_system()
    else:
        print("❌ Cannot proceed without test user")
        sys.exit(1)
