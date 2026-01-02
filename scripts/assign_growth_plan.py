"""
Script to assign Growth Plan to users by email address.
Usage: python -m scripts.assign_growth_plan
"""
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables to skip heavy initialization
os.environ['SKIP_HEAVY_INIT'] = '1'
os.environ['SKIP_SEARCH_INIT'] = '1'
os.environ['MIGRATION_MODE'] = '1'

from flask import Flask
from app.db import db
from app.models import User, Plan, Tenant, SubscriptionHistory
from app.config import Config
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def create_minimal_app():
    """Create a minimal Flask app without heavy dependencies"""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app

def assign_growth_plan_to_emails(emails):
    """Assign Growth Plan to users by email addresses"""
    app = create_minimal_app()
    
    with app.app_context():
        try:
            # Find Growth Plan - prefer monthly billing cycle, then use latest (highest ID)
            # First try to get monthly Growth plan
            growth_plan = Plan.query.filter(
                Plan.name.ilike("growth"),
                Plan.billing_cycle == 'monthly'
            ).order_by(Plan.id.desc()).first()
            
            if not growth_plan:
                # Fallback to any Growth plan (yearly or monthly)
                growth_plan = Plan.query.filter(Plan.name.ilike("growth")).order_by(Plan.id.desc()).first()
            
            if not growth_plan:
                # Try "Growth Plan"
                growth_plan = Plan.query.filter(Plan.name.ilike("growth plan")).order_by(Plan.id.desc()).first()
            
            if not growth_plan:
                print("[ERROR] Growth Plan not found in database")
                print("Available plans:")
                plans = Plan.query.order_by(Plan.name, Plan.id.desc()).all()
                seen_names = set()
                for plan in plans:
                    if plan.name not in seen_names:
                        print(f"  - {plan.name} (ID: {plan.id})")
                        seen_names.add(plan.name)
                    else:
                        print(f"  - {plan.name} (ID: {plan.id}) [DUPLICATE]")
                return False
            
            # Check if there are duplicates and warn
            all_growth_plans = Plan.query.filter(Plan.name.ilike("growth")).order_by(Plan.id.desc()).all()
            if len(all_growth_plans) > 1:
                print(f"[WARNING] Found {len(all_growth_plans)} Growth plan(s). Using the latest one (ID: {growth_plan.id})")
                for dup_plan in all_growth_plans[1:]:
                    print(f"  [NOTE] Duplicate Growth plan found (ID: {dup_plan.id}) - will not be used")
            
            print(f"[SUCCESS] Found Growth Plan: {growth_plan.name} (ID: {growth_plan.id})")
            
            success_count = 0
            failed_count = 0
            
            for email in emails:
                email = email.strip().lower()
                print(f"\nProcessing email: {email}")
                
                # Find user by email
                user = User.query.filter_by(email=email).first()
                
                if not user:
                    print(f"[WARNING] User not found: {email}")
                    failed_count += 1
                    continue
                
                # Get user's tenant
                tenant = db.session.get(Tenant, user.tenant_id)
                
                if not tenant:
                    print(f"[WARNING] Tenant not found for user: {email}")
                    failed_count += 1
                    continue
                
                # Get current plan
                current_plan = db.session.get(Plan, tenant.plan_id)
                current_plan_name = current_plan.name if current_plan else "Unknown"
                
                print(f"   Current plan: {current_plan_name} (ID: {tenant.plan_id})")
                print(f"   Tenant ID: {tenant.id}")
                
                # Update tenant plan
                old_plan_id = tenant.plan_id
                tenant.plan_id = growth_plan.id
                tenant.updated_at = datetime.utcnow()  # Update timestamp to trigger frontend refresh
                
                # Create subscription history record
                action = "upgraded" if old_plan_id != growth_plan.id else "created"
                subscription_history = SubscriptionHistory(
                    tenant_id=tenant.id,
                    user_id=user.id,
                    action=action,
                    from_plan_id=old_plan_id if old_plan_id != growth_plan.id else None,
                    to_plan_id=growth_plan.id,
                    reason=f"Manually assigned via script",
                    effective_date=datetime.utcnow()
                )
                db.session.add(subscription_history)
                
                try:
                    db.session.commit()
                    print(f"[SUCCESS] Successfully assigned Growth Plan to {email}")
                    print(f"   Previous plan: {current_plan_name} (ID: {old_plan_id})")
                    print(f"   New plan: {growth_plan.name} (ID: {growth_plan.id})")
                    print(f"   Subscription history record created")
                    success_count += 1
                except Exception as e:
                    db.session.rollback()
                    print(f"[ERROR] Failed to update plan for {email}: {e}")
                    failed_count += 1
            
            print("\n" + "="*60)
            print(f"Summary:")
            print(f"  [SUCCESS] Successfully assigned: {success_count}")
            print(f"  [ERROR] Failed: {failed_count}")
            print("="*60)
            
            return success_count > 0
            
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return False

if __name__ == "__main__":
    # Email addresses to assign Growth Plan
    emails = [
        "vivi@peopleconnectstaffing.com",
        "vinit@adeptaipro.com"
    ]
    
    print("="*60)
    print("Assigning Growth Plan to Users")
    print("="*60)
    print(f"Emails to process: {len(emails)}")
    for email in emails:
        print(f"  - {email}")
    print("="*60)
    print()
    
    success = assign_growth_plan_to_emails(emails)
    
    if success:
        print("\n[SUCCESS] Script completed successfully!")
        sys.exit(0)
    else:
        print("\n[ERROR] Script completed with errors!")
        sys.exit(1)

