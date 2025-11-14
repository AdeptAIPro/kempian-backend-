#!/usr/bin/env python3
"""
Script to clean up user profiles without resumes and predefined KPI data
This removes users who don't have actual resume files uploaded or have predefined data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User, UserKPIs, UserSkillGap, UserLearningPath, LearningModule, LearningCourse, UserAchievement, UserGoal, UserSchedule, CandidateProfile

def cleanup_predefined_profiles():
    """Clean up predefined user profiles and KPI data"""
    app = create_app()
    
    with app.app_context():
        try:
            print("🧹 Cleaning up user profiles without resumes...")
            
            # Get all users
            users = User.query.all()
            print(f"Found {len(users)} users in database")
            
            # Identify users to remove (those without resumes or with predefined data)
            users_to_remove = []
            
            for user in users:
                # Check if user has a candidate profile
                candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
                
                if candidate_profile:
                    # Check if user has an actual resume file uploaded
                    has_resume = candidate_profile.resume_s3_key and candidate_profile.resume_filename
                    
                    if not has_resume:
                        users_to_remove.append(user)
                        print(f"  - {user.email}: No resume uploaded (profile exists but no resume file)")
                    
                    # Also check for predefined KPI data patterns
                    kpis = UserKPIs.query.filter_by(user_id=user.id).first()
                    if kpis:
                        if (kpis.role_fit_score == 75.0 and 
                            kpis.career_benchmark == 'Top 30%' and
                            kpis.industry_targeting == 2):
                            if user not in users_to_remove:
                                users_to_remove.append(user)
                                print(f"  - {user.email}: Has predefined KPI data")
                    
                    # Check for predefined skill gaps
                    skill_gaps = UserSkillGap.query.filter_by(user_id=user.id).all()
                    if skill_gaps:
                        predefined_skills = ['Advanced SQL', 'Tableau', 'Python', 'Machine Learning', 'Statistical Analysis']
                        if all(gap.skill_name in predefined_skills for gap in skill_gaps):
                            if user not in users_to_remove:
                                users_to_remove.append(user)
                                print(f"  - {user.email}: Has predefined skill gaps")
                else:
                    # User has no profile at all - this is fine, keep them
                    pass
            
            print(f"\nIdentified {len(users_to_remove)} users to remove (no resume or predefined data)")
            
            if not users_to_remove:
                print("✅ No profiles found to clean up")
                return True
            
            # Confirm deletion
            print("\nUsers to be removed:")
            for user in users_to_remove:
                print(f"  - {user.email} (ID: {user.id})")
            
            # Remove KPI data first (due to foreign key constraints)
            # Delete in correct order: child tables first, then parent tables
            print("\nRemoving KPI data...")
            for user in users_to_remove:
                # Remove child tables first (due to foreign key constraints)
                # 1. Remove learning course data
                LearningCourse.query.filter(
                    LearningCourse.module_id.in_(
                        db.session.query(LearningModule.id).filter(
                            LearningModule.learning_path_id.in_(
                                db.session.query(UserLearningPath.id).filter_by(user_id=user.id)
                            )
                        )
                    )
                ).delete(synchronize_session=False)
                
                # 2. Remove learning module data
                LearningModule.query.filter(
                    LearningModule.learning_path_id.in_(
                        db.session.query(UserLearningPath.id).filter_by(user_id=user.id)
                    )
                ).delete(synchronize_session=False)
                
                # 3. Remove candidate-related data (skills, education, experience, certifications)
                from app.models import CandidateSkill, CandidateEducation, CandidateExperience, CandidateCertification
                
                # Get candidate profile first
                candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
                if candidate_profile:
                    # Remove candidate skills
                    CandidateSkill.query.filter_by(profile_id=candidate_profile.id).delete()
                    # Remove candidate education
                    CandidateEducation.query.filter_by(profile_id=candidate_profile.id).delete()
                    # Remove candidate experience
                    CandidateExperience.query.filter_by(profile_id=candidate_profile.id).delete()
                    # Remove candidate certifications
                    CandidateCertification.query.filter_by(profile_id=candidate_profile.id).delete()
                
                # 4. Remove other user-related data
                # Remove JD search logs
                from app.models import JDSearchLog
                JDSearchLog.query.filter_by(user_id=user.id).delete()
                
                # Remove user social links
                from app.models import UserSocialLinks
                UserSocialLinks.query.filter_by(user_id=user.id).delete()
                
                # Remove user trials
                from app.models import UserTrial
                UserTrial.query.filter_by(user_id=user.id).delete()
                
                # Remove ceipal integrations
                from app.models import CeipalIntegration
                CeipalIntegration.query.filter_by(user_id=user.id).delete()
                
                # Remove user images
                from app.models import UserImage
                UserImage.query.filter_by(user_id=user.id).delete()
                
                # 5. Now remove parent tables
                UserKPIs.query.filter_by(user_id=user.id).delete()
                UserSkillGap.query.filter_by(user_id=user.id).delete()
                UserLearningPath.query.filter_by(user_id=user.id).delete()
                UserAchievement.query.filter_by(user_id=user.id).delete()
                UserGoal.query.filter_by(user_id=user.id).delete()
                UserSchedule.query.filter_by(user_id=user.id).delete()
                
                # 6. Remove candidate profile
                CandidateProfile.query.filter_by(user_id=user.id).delete()
                
                print(f"  ✅ Removed KPI data for {user.email}")
            
            # Remove users
            print("\nRemoving users...")
            for user in users_to_remove:
                db.session.delete(user)
                print(f"  ✅ Removed user {user.email}")
            
            # Commit changes
            db.session.commit()
            
            print(f"\n🎉 Successfully cleaned up {len(users_to_remove)} profiles!")
            
            # Show remaining users
            remaining_users = User.query.all()
            print(f"\nRemaining users: {len(remaining_users)}")
            for user in remaining_users:
                print(f"  - {user.email}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cleaning up profiles: {str(e)}")
            db.session.rollback()
            return False

def verify_cleanup():
    """Verify that predefined data has been cleaned up"""
    app = create_app()
    
    with app.app_context():
        try:
            print("\n🔍 Verifying cleanup...")
            
            # Check remaining KPI data
            kpis_count = UserKPIs.query.count()
            skill_gaps_count = UserSkillGap.query.count()
            learning_paths_count = UserLearningPath.query.count()
            
            print(f"Remaining data:")
            print(f"  - User KPIs: {kpis_count}")
            print(f"  - Skill Gaps: {skill_gaps_count}")
            print(f"  - Learning Paths: {learning_paths_count}")
            
            if kpis_count == 0 and skill_gaps_count == 0 and learning_paths_count == 0:
                print("✅ All predefined KPI data has been cleaned up!")
            else:
                print("⚠️  Some KPI data still remains")
            
            return True
            
        except Exception as e:
            print(f"❌ Error verifying cleanup: {str(e)}")
            return False

if __name__ == "__main__":
    print("🚀 Profile Cleanup Script - Remove Users Without Resumes")
    print("=" * 60)
    
    # Clean up predefined profiles
    if cleanup_predefined_profiles():
        # Verify cleanup
        verify_cleanup()
        print("\n🎉 Cleanup process completed successfully!")
    else:
        print("\n❌ Cleanup process failed!")
        sys.exit(1)
