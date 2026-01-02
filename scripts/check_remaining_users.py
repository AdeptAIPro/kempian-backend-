#!/usr/bin/env python3
"""
Quick script to check remaining users in database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models import User, CandidateProfile, UserKPIs, UserSkillGap, UserLearningPath

def check_remaining_users():
    """Check what users remain in the database"""
    app = create_app()
    
    with app.app_context():
        try:
            print("🔍 Checking remaining users in database...")
            
            # Count users
            total_users = User.query.count()
            print(f"Total users: {total_users}")
            
            # Count users with profiles
            users_with_profiles = CandidateProfile.query.count()
            print(f"Users with profiles: {users_with_profiles}")
            
            # Count KPI data
            kpis_count = UserKPIs.query.count()
            skill_gaps_count = UserSkillGap.query.count()
            learning_paths_count = UserLearningPath.query.count()
            
            print(f"\nRemaining KPI data:")
            print(f"  - User KPIs: {kpis_count}")
            print(f"  - Skill Gaps: {skill_gaps_count}")
            print(f"  - Learning Paths: {learning_paths_count}")
            
            # Show remaining users
            if total_users > 0:
                print(f"\nRemaining users:")
                users = User.query.all()
                for user in users:
                    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
                    has_resume = "Yes" if (profile and profile.resume_s3_key and profile.resume_filename) else "No"
                    print(f"  - {user.email} (ID: {user.id}) - Resume: {has_resume}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking users: {str(e)}")
            return False

if __name__ == "__main__":
    check_remaining_users()
