#!/usr/bin/env python3
"""
Quota Management Script
=======================

This script allows you to set or reset quota limits for users by email.
It works with the existing Plan-based quota system where users inherit
quota from their tenant's plan.

It can be used to:
- Set a specific quota limit for a user's plan
- Reset a user's quota usage (by updating their plan)
- View current quota status
- List all users with their quota information

Usage:
    python manage_quota.py --email user@example.com --action set --limit 100
    python manage_quota.py --email user@example.com --action reset
    python manage_quota.py --email user@example.com --action view
    python manage_quota.py --action list
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import create_app
from app.models import User, Plan, Tenant, JDSearchLog
from app.db import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quota_management.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class QuotaManager:
    def __init__(self):
        """Initialize the quota manager with database connection."""
        try:
            self.app = create_app()
            self.app.app_context().push()
            self.db = db
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            sys.exit(1)

    def get_user_by_email(self, email):
        """Get user by email address."""
        try:
            user = User.query.filter_by(email=email).first()
            if user:
                logger.info(f"✅ Found user: {user.email}")
                return user
            else:
                logger.warning(f"⚠️ User not found: {email}")
                return None
        except Exception as e:
            logger.error(f"❌ Error finding user: {e}")
            return None

    def get_user_quota_info(self, user):
        """Get quota information for a user based on their tenant's plan."""
        try:
            if not user.tenant or not user.tenant.plan:
                return None, "User has no tenant or plan assigned"
            
            plan = user.tenant.plan
            quota_limit = plan.jd_quota_per_month
            
            # Count used quota (search logs for this month)
            current_month = datetime.utcnow().strftime('%Y-%m')
            used_quota = JDSearchLog.query.filter(
                JDSearchLog.user_id == user.id,
                JDSearchLog.searched_at >= datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            ).count()
            
            return {
                'plan_name': plan.name,
                'plan_id': plan.id,
                'quota_limit': quota_limit,
                'used_quota': used_quota,
                'remaining_quota': max(0, quota_limit - used_quota),
                'usage_percent': round((used_quota / quota_limit * 100), 2) if quota_limit > 0 else 0,
                'tenant_id': user.tenant.id,
                'tenant_status': user.tenant.status
            }, None
            
        except Exception as e:
            logger.error(f"❌ Error getting quota info: {e}")
            return None, f"Error getting quota info: {e}"

    def set_quota_limit(self, email, limit):
        """Set quota limit for a user by updating their plan."""
        try:
            user = self.get_user_by_email(email)
            if not user:
                return False, f"User not found: {email}"

            if not user.tenant:
                return False, f"User {email} has no tenant assigned"

            # Get current plan info
            current_plan = user.tenant.plan
            if not current_plan:
                return False, f"User {email} has no plan assigned"

            old_limit = current_plan.jd_quota_per_month
            
            # Update the plan's quota
            current_plan.jd_quota_per_month = limit
            current_plan.updated_at = datetime.utcnow()
            
            self.db.session.commit()
            
            logger.info(f"✅ Quota limit updated for {email}: {old_limit} → {limit}")
            return True, f"Quota limit set to {limit} for {email} (Plan: {current_plan.name})"

        except Exception as e:
            logger.error(f"❌ Error setting quota limit: {e}")
            self.db.session.rollback()
            return False, f"Error setting quota limit: {e}"

    def reset_quota(self, email):
        """Reset user's quota usage by clearing search logs for current month."""
        try:
            user = self.get_user_by_email(email)
            if not user:
                return False, f"User not found: {email}"

            # Count current usage
            current_month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            old_used = JDSearchLog.query.filter(
                JDSearchLog.user_id == user.id,
                JDSearchLog.searched_at >= current_month_start
            ).count()

            if old_used == 0:
                return True, f"Quota already reset for {email} (usage: 0)"

            # Delete search logs for current month
            deleted_count = JDSearchLog.query.filter(
                JDSearchLog.user_id == user.id,
                JDSearchLog.searched_at >= current_month_start
            ).delete()
            
            self.db.session.commit()
            
            logger.info(f"✅ Quota reset for {email}: {old_used} → 0 (deleted {deleted_count} logs)")
            return True, f"Quota reset for {email} (usage: {old_used} → 0, deleted {deleted_count} logs)"

        except Exception as e:
            logger.error(f"❌ Error resetting quota: {e}")
            self.db.session.rollback()
            return False, f"Error resetting quota: {e}"

    def view_quota(self, email):
        """View current quota status for a user."""
        try:
            user = self.get_user_by_email(email)
            if not user:
                return False, f"User not found: {email}"

            quota_info, error = self.get_user_quota_info(user)
            if error:
                return False, error

            logger.info(f"✅ Quota info retrieved for {email}")
            return True, quota_info

        except Exception as e:
            logger.error(f"❌ Error viewing quota: {e}")
            return False, f"Error viewing quota: {e}"

    def list_all_quotas(self):
        """List all users with their quota information."""
        try:
            users = User.query.all()
            
            quota_list = []
            for user in users:
                quota_info, error = self.get_user_quota_info(user)
                if error:
                    quota_list.append({
                        'email': user.email,
                        'plan_name': 'N/A',
                        'quota_limit': 0,
                        'used_quota': 0,
                        'remaining_quota': 0,
                        'usage_percent': 0,
                        'error': error
                    })
                else:
                    quota_list.append({
                        'email': user.email,
                        'plan_name': quota_info['plan_name'],
                        'quota_limit': quota_info['quota_limit'],
                        'used_quota': quota_info['used_quota'],
                        'remaining_quota': quota_info['remaining_quota'],
                        'usage_percent': quota_info['usage_percent'],
                        'tenant_status': quota_info['tenant_status']
                    })

            logger.info(f"✅ Retrieved quota info for {len(quota_list)} users")
            return True, quota_list

        except Exception as e:
            logger.error(f"❌ Error listing quotas: {e}")
            return False, f"Error listing quotas: {e}"

    def bulk_reset_quotas(self, emails):
        """Reset quotas for multiple users."""
        results = []
        for email in emails:
            success, message = self.reset_quota(email.strip())
            results.append({'email': email, 'success': success, 'message': message})
        return results

    def bulk_set_quotas(self, email_limit_pairs):
        """Set quota limits for multiple users."""
        results = []
        for email, limit in email_limit_pairs:
            success, message = self.set_quota_limit(email.strip(), int(limit))
            results.append({'email': email, 'limit': limit, 'success': success, 'message': message})
        return results

def main():
    """Main function to handle command line arguments and execute quota operations."""
    parser = argparse.ArgumentParser(
        description='Manage user quota limits (Plan-based system)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_quota.py --email user@example.com --action set --limit 100
  python manage_quota.py --email user@example.com --action reset
  python manage_quota.py --email user@example.com --action view
  python manage_quota.py --action list
  python manage_quota.py --action bulk-reset --file emails.txt
  python manage_quota.py --action bulk-set --file quota_config.txt
        """
    )

    parser.add_argument('--email', type=str, help='User email address')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['set', 'reset', 'view', 'list', 'bulk-reset', 'bulk-set'],
                       help='Action to perform')
    parser.add_argument('--limit', type=int, help='Quota limit to set')
    parser.add_argument('--file', type=str, help='File containing emails or email:limit pairs')

    args = parser.parse_args()

    # Validate arguments
    if args.action in ['set', 'reset', 'view'] and not args.email:
        logger.error("❌ Email is required for set, reset, and view actions")
        sys.exit(1)

    if args.action == 'set' and not args.limit:
        logger.error("❌ Limit is required for set action")
        sys.exit(1)

    if args.action in ['bulk-reset', 'bulk-set'] and not args.file:
        logger.error("❌ File is required for bulk actions")
        sys.exit(1)

    # Initialize quota manager
    manager = QuotaManager()

    try:
        if args.action == 'set':
            success, message = manager.set_quota_limit(args.email, args.limit)
            if success:
                logger.info(f"✅ {message}")
            else:
                logger.error(f"❌ {message}")
                sys.exit(1)

        elif args.action == 'reset':
            success, message = manager.reset_quota(args.email)
            if success:
                logger.info(f"✅ {message}")
            else:
                logger.error(f"❌ {message}")
                sys.exit(1)

        elif args.action == 'view':
            success, result = manager.view_quota(args.email)
            if success:
                print("\n" + "="*70)
                print(f"QUOTA STATUS FOR: {args.email}")
                print("="*70)
                print(f"Plan:         {result['plan_name']}")
                print(f"Tenant ID:    {result['tenant_id']}")
                print(f"Tenant Status: {result['tenant_status']}")
                print(f"Quota Limit:  {result['quota_limit']}")
                print(f"Used:         {result['used_quota']}")
                print(f"Remaining:    {result['remaining_quota']}")
                print(f"Usage:        {result['usage_percent']}%")
                print("="*70)
            else:
                logger.error(f"❌ {result}")
                sys.exit(1)

        elif args.action == 'list':
            success, quotas = manager.list_all_quotas()
            if success:
                print("\n" + "="*120)
                print(f"QUOTA STATUS FOR ALL USERS ({len(quotas)} total)")
                print("="*120)
                print(f"{'Email':<30} {'Plan':<15} {'Limit':<8} {'Used':<8} {'Remaining':<10} {'Usage %':<8} {'Status':<10}")
                print("-"*120)
                for quota in quotas:
                    if 'error' in quota:
                        print(f"{quota['email']:<30} {'ERROR':<15} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<8} {'N/A':<10}")
                    else:
                        print(f"{quota['email']:<30} {quota['plan_name']:<15} {quota['quota_limit']:<8} "
                              f"{quota['used_quota']:<8} {quota['remaining_quota']:<10} "
                              f"{quota['usage_percent']:<8.1f}% {quota['tenant_status']:<10}")
                print("="*120)
            else:
                logger.error(f"❌ {quotas}")
                sys.exit(1)

        elif args.action == 'bulk-reset':
            try:
                with open(args.file, 'r') as f:
                    emails = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                logger.info(f"🔄 Resetting quotas for {len(emails)} users...")
                results = manager.bulk_reset_quotas(emails)
                
                print("\n" + "="*80)
                print("BULK RESET RESULTS")
                print("="*80)
                for result in results:
                    status = "✅" if result['success'] else "❌"
                    print(f"{status} {result['email']}: {result['message']}")
                print("="*80)
                
            except FileNotFoundError:
                logger.error(f"❌ File not found: {args.file}")
                sys.exit(1)

        elif args.action == 'bulk-set':
            try:
                with open(args.file, 'r') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                email_limit_pairs = []
                for line in lines:
                    if ':' in line:
                        email, limit = line.split(':', 1)
                        email_limit_pairs.append((email.strip(), int(limit.strip())))
                    else:
                        logger.warning(f"⚠️ Skipping invalid line: {line}")
                
                logger.info(f"🔄 Setting quotas for {len(email_limit_pairs)} users...")
                results = manager.bulk_set_quotas(email_limit_pairs)
                
                print("\n" + "="*80)
                print("BULK SET RESULTS")
                print("="*80)
                for result in results:
                    status = "✅" if result['success'] else "❌"
                    print(f"{status} {result['email']} (limit: {result['limit']}): {result['message']}")
                print("="*80)
                
            except FileNotFoundError:
                logger.error(f"❌ File not found: {args.file}")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⚠️ Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 