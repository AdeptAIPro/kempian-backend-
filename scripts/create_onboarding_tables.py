#!/usr/bin/env python3
"""
Script to create Onboarding tables in the database
Run this script after starting the backend to create the onboarding tables

Usage:
  python backend/scripts/create_onboarding_tables.py

Optional seeding (mark a specific email as required):
  python backend/scripts/create_onboarding_tables.py user@example.com
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models import db, OnboardingFlag, OnboardingSubmission, User

def create_onboarding_tables():
    """Create onboarding-related tables"""
    app = create_app()
    with app.app_context():
        try:
            print("Creating Onboarding tables...")
            # Create any missing tables declared in models (including onboarding ones)
            db.create_all()
            print("✅ Onboarding tables created successfully!")
            print("\nCreated tables (if missing):")
            print("- onboarding_flags")
            print("- onboarding_submissions")
        except Exception as e:
            print(f"❌ Error creating onboarding tables: {str(e)}")
            return False
    return True

def seed_onboarding_flag(email: str):
    """Optionally seed a user to require onboarding by email"""
    app = create_app()
    with app.app_context():
        try:
            user = User.query.filter_by(email=email).first()
            if not user:
                print(f"⚠️  No user found with email: {email}")
                return
            flag = OnboardingFlag.query.filter_by(user_id=user.id).first()
            if not flag:
                flag = OnboardingFlag(user_id=user.id, required=True, completed=False)
                db.session.add(flag)
            else:
                flag.required = True
                flag.completed = False
            db.session.commit()
            print(f"✅ Marked onboarding required for {email} (user_id={user.id})")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error seeding onboarding flag: {str(e)}")

if __name__ == "__main__":
    print("🚀 Onboarding Tables Setup Script")
    print("=" * 40)
    if create_onboarding_tables():
        if len(sys.argv) > 1 and sys.argv[1]:
            seed_onboarding_flag(sys.argv[1])
        print("\n🎉 Onboarding setup complete!")
    else:
        print("\n❌ Onboarding setup failed!")
        sys.exit(1)


