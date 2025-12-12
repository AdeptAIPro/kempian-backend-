"""
Script to list all jobseekers who don't have a profile.
Useful for auditing and identifying users who need to complete their profiles.

Usage:
    # Display list in console
    python -m scripts.list_jobseekers_without_profile
    
    # Export to CSV
    python -m scripts.list_jobseekers_without_profile --export csv
    
    # Export to JSON
    python -m scripts.list_jobseekers_without_profile --export json
    
    # Show detailed information
    python -m scripts.list_jobseekers_without_profile --detailed
"""

import sys
import os
import csv
import json
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.models import db, User, CandidateProfile
from app.simple_logger import get_logger

logger = get_logger("list_jobseekers")

def get_jobseekers_without_profile(detailed=False):
    """
    Get all jobseekers who don't have a candidate profile.
    
    Args:
        detailed: If True, include more information about each user
    
    Returns:
        List of dictionaries with user information
    """
    try:
        # Query all jobseekers
        jobseekers = User.query.filter(
            (User.user_type == 'job_seeker') | (User.role == 'job_seeker')
        ).all()
        
        users_without_profile = []
        
        for user in jobseekers:
            # Check if user has a candidate profile
            profile = CandidateProfile.query.filter_by(user_id=user.id).first()
            
            if not profile:
                # User has no profile
                user_info = {
                    'id': user.id,
                    'email': user.email,
                    'user_type': user.user_type,
                    'role': user.role,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'updated_at': user.updated_at.isoformat() if user.updated_at else None,
                    'days_since_signup': (datetime.utcnow() - user.created_at).days if user.created_at else None,
                    'has_password': bool(user.password_hash),
                    'has_linkedin_id': bool(user.linkedin_id),
                    'login_method': 'google' if user.linkedin_id or not user.password_hash else 'email',
                    'tenant_id': user.tenant_id
                }
                
                if detailed:
                    # Add more detailed information
                    user_info.update({
                        'company_name': user.company_name,
                        'linkedin_id': user.linkedin_id,
                    })
                
                users_without_profile.append(user_info)
        
        return users_without_profile
        
    except Exception as e:
        logger.error(f"[LIST JOBSEEKERS] Error getting jobseekers without profile: {str(e)}")
        return []

def display_summary(users):
    """
    Display summary statistics.
    """
    if not users:
        print("\n" + "="*70)
        print("No jobseekers without profiles found!")
        print("="*70 + "\n")
        return
    
    # Calculate statistics
    total = len(users)
    google_logins = sum(1 for u in users if u['login_method'] == 'google')
    email_logins = sum(1 for u in users if u['login_method'] == 'email')
    
    # Group by days since signup
    signup_groups = {
        '0-1 days': 0,
        '2-7 days': 0,
        '8-30 days': 0,
        '31-90 days': 0,
        '90+ days': 0
    }
    
    for user in users:
        days = user.get('days_since_signup')
        if days is None:
            continue
        elif days <= 1:
            signup_groups['0-1 days'] += 1
        elif days <= 7:
            signup_groups['2-7 days'] += 1
        elif days <= 30:
            signup_groups['8-30 days'] += 1
        elif days <= 90:
            signup_groups['31-90 days'] += 1
        else:
            signup_groups['90+ days'] += 1
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total jobseekers without profile: {total}")
    print(f"  - Google OAuth logins: {google_logins}")
    print(f"  - Email logins: {email_logins}")
    print("\nSignup Distribution:")
    for group, count in signup_groups.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  - {group}: {count} ({percentage:.1f}%)")
    print("="*70 + "\n")

def display_list(users, detailed=False):
    """
    Display the list of users in a formatted table.
    """
    if not users:
        print("No jobseekers without profiles found.")
        return
    
    print("\n" + "="*120)
    print("JOBSEEKERS WITHOUT PROFILE")
    print("="*120)
    
    if detailed:
        # Detailed view
        print(f"{'ID':<6} {'Email':<40} {'Login':<8} {'Signup Date':<12} {'Days Ago':<10} {'User Type':<12}")
        print("-" * 120)
        
        for user in users:
            signup_date = user['created_at'][:10] if user['created_at'] else 'N/A'
            days_ago = str(user['days_since_signup']) if user['days_since_signup'] is not None else 'N/A'
            login_method = user['login_method'].upper()
            user_type = user.get('user_type', 'N/A')
            
            print(f"{user['id']:<6} {user['email']:<40} {login_method:<8} {signup_date:<12} {days_ago:<10} {user_type:<12}")
    else:
        # Simple view
        print(f"{'ID':<6} {'Email':<50} {'Login':<8} {'Days Since Signup':<18}")
        print("-" * 90)
        
        for user in users:
            days_ago = str(user['days_since_signup']) if user['days_since_signup'] is not None else 'N/A'
            login_method = user['login_method'].upper()
            
            print(f"{user['id']:<6} {user['email']:<50} {login_method:<8} {days_ago:<18}")
    
    print("="*120 + "\n")

def export_to_csv(users, filename=None):
    """
    Export users to CSV file.
    """
    if not users:
        print("No data to export.")
        return False
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jobseekers_without_profile_{timestamp}.csv"
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'email', 'user_type', 'role', 'login_method', 
                         'created_at', 'days_since_signup', 'has_password', 
                         'has_linkedin_id', 'tenant_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for user in users:
                writer.writerow({
                    'id': user['id'],
                    'email': user['email'],
                    'user_type': user['user_type'],
                    'role': user['role'],
                    'login_method': user['login_method'],
                    'created_at': user['created_at'],
                    'days_since_signup': user['days_since_signup'],
                    'has_password': user['has_password'],
                    'has_linkedin_id': user['has_linkedin_id'],
                    'tenant_id': user['tenant_id']
                })
        
        print(f"✓ Exported {len(users)} records to {filename}")
        logger.info(f"[LIST JOBSEEKERS] Exported {len(users)} records to {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error exporting to CSV: {str(e)}")
        logger.error(f"[LIST JOBSEEKERS] Error exporting to CSV: {str(e)}")
        return False

def export_to_json(users, filename=None):
    """
    Export users to JSON file.
    """
    if not users:
        print("No data to export.")
        return False
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jobseekers_without_profile_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(users, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported {len(users)} records to {filename}")
        logger.info(f"[LIST JOBSEEKERS] Exported {len(users)} records to {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error exporting to JSON: {str(e)}")
        logger.error(f"[LIST JOBSEEKERS] Error exporting to JSON: {str(e)}")
        return False

def main():
    """
    Main function to list jobseekers without profiles.
    """
    app = create_app()
    
    with app.app_context():
        logger.info("[LIST JOBSEEKERS] Starting script to list jobseekers without profiles...")
        
        # Parse command line arguments
        detailed = '--detailed' in sys.argv or '-d' in sys.argv
        export_format = None
        
        if '--export' in sys.argv:
            idx = sys.argv.index('--export')
            if idx + 1 < len(sys.argv):
                export_format = sys.argv[idx + 1].lower()
        elif '-e' in sys.argv:
            idx = sys.argv.index('-e')
            if idx + 1 < len(sys.argv):
                export_format = sys.argv[idx + 1].lower()
        
        # Get jobseekers without profile
        users = get_jobseekers_without_profile(detailed=detailed)
        
        logger.info(f"[LIST JOBSEEKERS] Found {len(users)} jobseekers without profiles")
        
        # Display summary
        display_summary(users)
        
        # Display list
        if not export_format:
            display_list(users, detailed=detailed)
        
        # Export if requested
        if export_format == 'csv':
            export_to_csv(users)
        elif export_format == 'json':
            export_to_json(users)
        elif export_format:
            print(f"✗ Unknown export format: {export_format}")
            print("Supported formats: csv, json")
        
        # Print email list for easy copy-paste
        if users and not export_format:
            print("="*70)
            print("EMAIL LIST (for easy copy-paste):")
            print("="*70)
            emails = [user['email'] for user in users]
            print(', '.join(emails))
            print("="*70 + "\n")
        
        logger.info("[LIST JOBSEEKERS] Script completed.")

if __name__ == "__main__":
    main()

