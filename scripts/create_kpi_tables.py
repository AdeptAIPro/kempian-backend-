#!/usr/bin/env python3
"""
Script to create KPI tables in the database
Run this script after starting the backend to create the new KPI tables
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models import db, UserKPIs, UserSkillGap, UserLearningPath, LearningModule, LearningCourse, UserAchievement, UserGoal, UserSchedule

def create_kpi_tables():
    """Create all KPI-related tables"""
    app = create_app()
    
    with app.app_context():
        try:
            print("Creating KPI tables...")
            
            # Create tables
            db.create_all()
            
            print("✅ KPI tables created successfully!")
            print("\nCreated tables:")
            print("- user_kpis")
            print("- user_skill_gaps")
            print("- user_learning_paths")
            print("- learning_modules")
            print("- learning_courses")
            print("- user_achievements")
            print("- user_goals")
            print("- user_schedules")
            
        except Exception as e:
            print(f"❌ Error creating KPI tables: {str(e)}")
            return False
    
    return True

def seed_sample_data():
    """Seed sample data for testing"""
    app = create_app()
    
    with app.app_context():
        try:
            print("\nSeeding sample data...")
            
            # Check if we have any users to seed data for
            from app.models import User
            users = User.query.limit(5).all()
            
            if not users:
                print("No users found. Skipping sample data seeding.")
                return
            
            print(f"Found {len(users)} users. Seeding sample data...")
            
            for user in users:
                # Create sample KPIs
                if not UserKPIs.query.filter_by(user_id=user.id).first():
                    kpis = UserKPIs(
                        user_id=user.id,
                        role_fit_score=75.0,
                        career_benchmark='Top 30%',
                        industry_targeting=2,
                        experience_level='Above Average',
                        skills_learned=8,
                        jobs_applied=15,
                        courses_completed=3,
                        learning_streak=7
                    )
                    db.session.add(kpis)
                
                # Create sample skill gaps
                if not UserSkillGap.query.filter_by(user_id=user.id).first():
                    skill_gaps = [
                        UserSkillGap(
                            user_id=user.id,
                            skill_name='Advanced SQL',
                            current_level=60,
                            target_level=90,
                            priority='High',
                            role_target='Senior Data Analyst'
                        ),
                        UserSkillGap(
                            user_id=user.id,
                            skill_name='Tableau',
                            current_level=40,
                            target_level=85,
                            priority='High',
                            role_target='Senior Data Analyst'
                        ),
                        UserSkillGap(
                            user_id=user.id,
                            skill_name='Python',
                            current_level=70,
                            target_level=90,
                            priority='Medium',
                            role_target='Senior Data Analyst'
                        )
                    ]
                    for gap in skill_gaps:
                        db.session.add(gap)
                
                # Create sample learning path
                if not UserLearningPath.query.filter_by(user_id=user.id).first():
                    learning_path = UserLearningPath(
                        user_id=user.id,
                        pathway_name='Data Analyst Mastery Path',
                        pathway_description='Comprehensive roadmap to become a senior data analyst',
                        total_duration='4-6 months',
                        progress=35.0,
                        is_active=True
                    )
                    db.session.add(learning_path)
                    db.session.flush()  # Get the ID
                    
                    # Create modules
                    modules = [
                        LearningModule(
                            learning_path_id=learning_path.id,
                            title='SQL Fundamentals',
                            status='completed',
                            duration='2 weeks',
                            order_index=1
                        ),
                        LearningModule(
                            learning_path_id=learning_path.id,
                            title='Python for Data Analysis',
                            status='in-progress',
                            duration='3 weeks',
                            order_index=2
                        ),
                        LearningModule(
                            learning_path_id=learning_path.id,
                            title='Tableau Mastery',
                            status='upcoming',
                            duration='2 weeks',
                            order_index=3
                        )
                    ]
                    for module in modules:
                        db.session.add(module)
                
                # Create sample achievements
                if not UserAchievement.query.filter_by(user_id=user.id).first():
                    achievements = [
                        UserAchievement(
                            user_id=user.id,
                            title='SQL Master',
                            description='Completed advanced SQL course',
                            type='skill',
                            points=150
                        ),
                        UserAchievement(
                            user_id=user.id,
                            title='Profile Complete',
                            description='Added all required sections',
                            type='profile',
                            points=100
                        )
                    ]
                    for achievement in achievements:
                        db.session.add(achievement)
                
                # Create sample goals
                if not UserGoal.query.filter_by(user_id=user.id).first():
                    goals = [
                        UserGoal(
                            user_id=user.id,
                            title='Complete Python Module',
                            progress=65,
                            deadline='2 days',
                            priority='high'
                        ),
                        UserGoal(
                            user_id=user.id,
                            title='Apply to 5 Jobs',
                            progress=40,
                            deadline='5 days',
                            priority='medium'
                        )
                    ]
                    for goal in goals:
                        db.session.add(goal)
                
                # Create sample schedule
                if not UserSchedule.query.filter_by(user_id=user.id).first():
                    from datetime import datetime, timedelta
                    
                    schedules = [
                        UserSchedule(
                            user_id=user.id,
                            title='Python Study Session',
                            description='Complete Python module exercises',
                            event_date=datetime.utcnow() + timedelta(hours=2),
                            duration_minutes=90,
                            event_type='study'
                        ),
                        UserSchedule(
                            user_id=user.id,
                            title='Mock Interview Practice',
                            description='Practice common data analyst questions',
                            event_date=datetime.utcnow() + timedelta(days=1),
                            duration_minutes=60,
                            event_type='interview'
                        )
                    ]
                    for schedule in schedules:
                        db.session.add(schedule)
            
            db.session.commit()
            print("✅ Sample data seeded successfully!")
            
        except Exception as e:
            print(f"❌ Error seeding sample data: {str(e)}")
            db.session.rollback()

if __name__ == "__main__":
    print("🚀 KPI Tables Setup Script")
    print("=" * 40)
    
    # Create tables
    if create_kpi_tables():
        # Seed sample data
        seed_sample_data()
        print("\n🎉 KPI system setup complete!")
    else:
        print("\n❌ KPI system setup failed!")
        sys.exit(1)
