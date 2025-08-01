#!/usr/bin/env python3
"""
Interactive Quota Management Script
==================================

This script provides an interactive interface for managing user quotas.
It works with the existing Plan-based quota system where users inherit
quota from their tenant's plan.

Usage:
    python interactive_quota.py
"""

import os
import sys
import logging
from datetime import datetime

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import create_app
from app.models import User, Plan, Tenant, JDSearchLog
from app.db import db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveQuotaManager:
    def __init__(self):
        """Initialize the interactive quota manager."""
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
            return user
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
            current_month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            used_quota = JDSearchLog.query.filter(
                JDSearchLog.user_id == user.id,
                JDSearchLog.searched_at >= current_month_start
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
            
            return True, f"Quota limit updated: {old_limit} → {limit} (Plan: {current_plan.name})"

        except Exception as e:
            logger.error(f"❌ Error setting quota limit: {e}")
            self.db.session.rollback()
            return False, f"Error: {e}"

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
            
            return True, f"Quota reset: {old_used} → 0 (deleted {deleted_count} logs)"

        except Exception as e:
            logger.error(f"❌ Error resetting quota: {e}")
            self.db.session.rollback()
            return False, f"Error: {e}"

    def view_quota(self, email):
        """View current quota status for a user."""
        try:
            user = self.get_user_by_email(email)
            if not user:
                return False, f"User not found: {email}"

            quota_info, error = self.get_user_quota_info(user)
            if error:
                return False, error

            return True, quota_info

        except Exception as e:
            logger.error(f"❌ Error viewing quota: {e}")
            return False, f"Error: {e}"

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

            return True, quota_list

        except Exception as e:
            logger.error(f"❌ Error listing quotas: {e}")
            return False, f"Error: {e}"

    def show_menu(self):
        """Display the main menu."""
        print("\n" + "="*50)
        print("🎯 QUOTA MANAGEMENT SYSTEM")
        print("="*50)
        print("1. Set quota limit for user")
        print("2. Reset user quota (clear current month usage)")
        print("3. View user quota status")
        print("4. List all users and quotas")
        print("5. Exit")
        print("="*50)

    def run(self):
        """Run the interactive quota manager."""
        print("🚀 Starting Interactive Quota Manager...")
        
        while True:
            self.show_menu()
            
            try:
                choice = input("\n📝 Enter your choice (1-5): ").strip()
                
                if choice == '1':
                    self.handle_set_quota()
                elif choice == '2':
                    self.handle_reset_quota()
                elif choice == '3':
                    self.handle_view_quota()
                elif choice == '4':
                    self.handle_list_quotas()
                elif choice == '5':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

    def handle_set_quota(self):
        """Handle setting quota limit."""
        print("\n🎯 SET QUOTA LIMIT")
        print("-" * 30)
        
        email = input("📧 Enter user email: ").strip()
        if not email:
            print("❌ Email is required")
            return
            
        try:
            limit = int(input("📊 Enter quota limit: ").strip())
            if limit < 0:
                print("❌ Limit must be positive")
                return
        except ValueError:
            print("❌ Invalid limit. Please enter a number.")
            return
            
        success, message = self.set_quota_limit(email, limit)
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")

    def handle_reset_quota(self):
        """Handle resetting quota."""
        print("\n🔄 RESET USER QUOTA")
        print("-" * 30)
        
        email = input("📧 Enter user email: ").strip()
        if not email:
            print("❌ Email is required")
            return
            
        success, message = self.reset_quota(email)
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")

    def handle_view_quota(self):
        """Handle viewing quota status."""
        print("\n👁️ VIEW QUOTA STATUS")
        print("-" * 30)
        
        email = input("📧 Enter user email: ").strip()
        if not email:
            print("❌ Email is required")
            return
            
        success, result = self.view_quota(email)
        if success:
            print("\n" + "="*70)
            print(f"📊 QUOTA STATUS: {email}")
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
            print(f"❌ {result}")

    def handle_list_quotas(self):
        """Handle listing all quotas."""
        print("\n📋 LISTING ALL QUOTAS")
        print("-" * 30)
        
        success, quotas = self.list_all_quotas()
        if success:
            print("\n" + "="*120)
            print(f"📊 QUOTA STATUS FOR ALL USERS ({len(quotas)} total)")
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
            print(f"❌ {quotas}")

def main():
    """Main function to run the interactive quota manager."""
    try:
        manager = InteractiveQuotaManager()
        manager.run()
    except Exception as e:
        logger.error(f"❌ Failed to start quota manager: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 