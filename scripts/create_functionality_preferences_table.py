#!/usr/bin/env python3
"""
Script to create functionality preferences table in the database
Run this script after starting the backend to create the user_functionality_preferences table
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models import db, UserFunctionalityPreferences, User

def create_functionality_preferences_table():
    """Create the functionality preferences table"""
    app = create_app()
    
    with app.app_context():
        try:
            print("Creating functionality preferences table...")
            
            # Create table
            db.create_all()
            
            print("✅ Functionality preferences table created successfully!")
            print("\nCreated table:")
            print("- user_functionality_preferences")
            print("\nTable structure:")
            print("  - id: INTEGER PRIMARY KEY")
            print("  - user_id: INTEGER (FK to users.id, UNIQUE)")
            print("  - functionalities: JSON (array of functionality IDs)")
            print("  - created_at: DATETIME")
            print("  - updated_at: DATETIME")
            
        except Exception as e:
            print(f"❌ Error creating functionality preferences table: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def seed_sample_preferences():
    """Seed sample functionality preferences for existing users"""
    app = create_app()
    
    with app.app_context():
        try:
            print("\nSeeding sample functionality preferences...")
            
            # Get all non-job-seeker users
            users = User.query.filter(
                ~User.user_type.in_(['job_seeker', 'jobseeker']),
                User.role.notin_(['job_seeker', 'jobseeker'])
            ).limit(10).all()
            
            if not users:
                print("No eligible users found. Skipping sample data seeding.")
                return
            
            print(f"Found {len(users)} eligible users. Seeding sample preferences...")
            
            # Available functionalities
            available_functionalities = [
                'talent_matchmaker',
                'payroll',
                'agentic_ai',
                'compliance',
                'candidate_search',
                'document_management',
                'scheduling',
                'analytics',
                'communications',
                'workflow_automation'
            ]
            
            # Sample preference sets for different user types
            sample_preferences = {
                'employer': ['talent_matchmaker', 'candidate_search', 'analytics', 'scheduling'],
                'recruiter': ['talent_matchmaker', 'candidate_search', 'communications', 'analytics'],
                'admin': ['talent_matchmaker', 'payroll', 'agentic_ai', 'analytics', 'compliance'],
                'owner': ['talent_matchmaker', 'payroll', 'agentic_ai', 'compliance', 'analytics', 'document_management'],
            }
            
            seeded_count = 0
            for user in users:
                # Skip if user already has preferences
                existing = UserFunctionalityPreferences.query.filter_by(user_id=user.id).first()
                if existing:
                    continue
                
                # Determine user type
                user_type = user.user_type or user.role
                
                # Get sample preferences based on user type, or use default
                preferences = sample_preferences.get(user_type, ['talent_matchmaker', 'analytics'])
                
                # Create preferences entry
                user_prefs = UserFunctionalityPreferences(
                    user_id=user.id,
                    functionalities=preferences
                )
                db.session.add(user_prefs)
                seeded_count += 1
                
                print(f"  ✓ Seeded preferences for {user.email} ({user_type}): {len(preferences)} functionalities")
            
            if seeded_count > 0:
                db.session.commit()
                print(f"\n✅ Seeded functionality preferences for {seeded_count} users!")
            else:
                print("\nℹ️  All users already have preferences set.")
            
        except Exception as e:
            print(f"❌ Error seeding sample preferences: {str(e)}")
            import traceback
            traceback.print_exc()
            db.session.rollback()

def reset_all_preferences():
    """Reset all functionality preferences (use with caution!)"""
    app = create_app()
    
    with app.app_context():
        try:
            confirm = input("\n⚠️  WARNING: This will delete ALL functionality preferences. Type 'YES' to confirm: ")
            if confirm != 'YES':
                print("Operation cancelled.")
                return
            
            count = UserFunctionalityPreferences.query.count()
            UserFunctionalityPreferences.query.delete()
            db.session.commit()
            
            print(f"✅ Deleted {count} functionality preference entries.")
            
        except Exception as e:
            print(f"❌ Error resetting preferences: {str(e)}")
            import traceback
            traceback.print_exc()
            db.session.rollback()

def show_statistics():
    """Show statistics about functionality preferences"""
    app = create_app()
    
    with app.app_context():
        try:
            total_prefs = UserFunctionalityPreferences.query.count()
            total_users = User.query.filter(
                ~User.user_type.in_(['job_seeker', 'jobseeker']),
                User.role.notin_(['job_seeker', 'jobseeker'])
            ).count()
            
            print("\n📊 Functionality Preferences Statistics:")
            print("=" * 50)
            print(f"Total users (non-job-seekers): {total_users}")
            print(f"Users with preferences: {total_prefs}")
            print(f"Users without preferences: {total_users - total_prefs}")
            
            if total_prefs > 0:
                # Count functionality usage
                from collections import Counter
                all_functionalities = []
                
                prefs = UserFunctionalityPreferences.query.all()
                for pref in prefs:
                    if pref.functionalities:
                        all_functionalities.extend(pref.functionalities)
                
                if all_functionalities:
                    usage_count = Counter(all_functionalities)
                    print("\nMost popular functionalities:")
                    for func, count in usage_count.most_common():
                        percentage = (count / total_prefs) * 100
                        print(f"  - {func}: {count} users ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error showing statistics: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Functionality Preferences Table Setup Script")
    print("=" * 50)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'reset':
            reset_all_preferences()
        elif command == 'stats':
            show_statistics()
        elif command == 'seed':
            app = create_app()
            with app.app_context():
                seed_sample_preferences()
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  python create_functionality_preferences_table.py         - Create table and seed sample data")
            print("  python create_functionality_preferences_table.py seed     - Only seed sample data")
            print("  python create_functionality_preferences_table.py stats   - Show statistics")
            print("  python create_functionality_preferences_table.py reset  - Reset all preferences (CAUTION!)")
    else:
        # Default: create table and seed sample data
        if create_functionality_preferences_table():
            # Ask if user wants to seed sample data
            response = input("\nWould you like to seed sample functionality preferences? (y/n): ").lower()
            if response == 'y' or response == 'yes':
                seed_sample_preferences()
            
            # Show statistics
            show_statistics()
            
            print("\n🎉 Functionality preferences system setup complete!")
            print("\n💡 Tips:")
            print("  - Run with 'stats' argument to see usage statistics")
            print("  - Run with 'seed' argument to only seed sample data")
            print("  - Users can set their preferences through the UI")
        else:
            print("\n❌ Functionality preferences table setup failed!")
            sys.exit(1)

