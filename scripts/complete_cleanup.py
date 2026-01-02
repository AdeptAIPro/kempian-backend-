#!/usr/bin/env python3
"""
Script to complete the cleanup by removing users without resumes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User, CandidateProfile, UserKPIs, UserSkillGap, UserLearningPath, LearningModule, LearningCourse, UserAchievement, UserGoal, UserSchedule

def complete_cleanup():
    """Complete the cleanup by removing users without resumes"""
    app = create_app()
    
    with app.app_context():
        try:
            print("🧹 Completing cleanup - removing users without resumes...")
            
            # Get all users
            users = User.query.all()
            print(f"Found {len(users)} users in database")
            
            # Identify users to remove (those without resumes)
            users_to_remove = []
            
            for user in users:
                profile = CandidateProfile.query.filter_by(user_id=user.id).first()
                if profile:
                    # Check if user has an actual resume file uploaded
                    has_resume = profile.resume_s3_key and profile.resume_filename
                    
                    if not has_resume:
                        users_to_remove.append(user)
                        print(f"  - {user.email}: No resume uploaded")
                else:
                    # User has no profile - this is fine, keep them
                    pass
            
            print(f"\nIdentified {len(users_to_remove)} users without resumes to remove")
            
            if not users_to_remove:
                print("✅ No users without resumes found to clean up")
                return True
            
            # Confirm deletion
            print("\nUsers to be removed:")
            for user in users_to_remove:
                print(f"  - {user.email} (ID: {user.id})")
            
            # Remove KPI data first (due to foreign key constraints)
            print("\nRemoving KPI data...")
            for user in users_to_remove:
                try:
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
                    
                except Exception as e:
                    print(f"  ❌ Error removing data for {user.email}: {str(e)}")
                    continue
            
            # Remove users
            print("\nRemoving users...")
            for user in users_to_remove:
                try:
                    db.session.delete(user)
                    print(f"  ✅ Removed user {user.email}")
                except Exception as e:
                    print(f"  ❌ Error removing user {user.email}: {str(e)}")
                    continue
            
            # Commit changes
            print("\nCommitting changes...")
            db.session.commit()
            
            print(f"\n🎉 Successfully cleaned up {len(users_to_remove)} users without resumes!")
            
            # Show remaining users
            remaining_users = User.query.all()
            print(f"\nRemaining users: {len(remaining_users)}")
            for user in remaining_users:
                profile = CandidateProfile.query.filter_by(user_id=user.id).first()
                has_resume = "Yes" if (profile and profile.resume_s3_key and profile.resume_filename) else "No"
                print(f"  - {user.email} - Resume: {has_resume}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error completing cleanup: {str(e)}")
            db.session.rollback()
            return False

if __name__ == "__main__":
    print("🚀 Complete Cleanup Script - Remove Users Without Resumes")
    print("=" * 60)
    
    # Complete the cleanup
    if complete_cleanup():
        print("\n🎉 Cleanup process completed successfully!")
    else:
        print("\n❌ Cleanup process failed!")
        sys.exit(1)
