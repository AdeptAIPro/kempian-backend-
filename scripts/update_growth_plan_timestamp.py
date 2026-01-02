"""
Script to update tenant.updated_at timestamp and create subscription history
for users who already have Growth Plan assigned.
This ensures the frontend recognizes the plan change.
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

def update_growth_plan_users():
    """Update timestamp and create history for users with Growth Plan"""
    app = create_minimal_app()
    
    with app.app_context():
        try:
            # Find Growth Plan
            growth_plan = Plan.query.filter(Plan.name.ilike("growth")).order_by(Plan.id.desc()).first()
            
            if not growth_plan:
                print("[ERROR] Growth Plan not found in database")
                return False
            
            print(f"[INFO] Found Growth Plan: {growth_plan.name} (ID: {growth_plan.id})")
            
            # Find all tenants with Growth Plan
            tenants_with_growth = Tenant.query.filter_by(plan_id=growth_plan.id).all()
            
            if not tenants_with_growth:
                print("[INFO] No tenants found with Growth Plan")
                return True
            
            print(f"[INFO] Found {len(tenants_with_growth)} tenant(s) with Growth Plan")
            
            success_count = 0
            failed_count = 0
            
            for tenant in tenants_with_growth:
                try:
                    # Get the first user for this tenant to create history
                    user = User.query.filter_by(tenant_id=tenant.id).first()
                    
                    if not user:
                        print(f"[WARNING] No user found for tenant ID: {tenant.id}")
                        continue
                    
                    # Update tenant timestamp
                    tenant.updated_at = datetime.utcnow()
                    
                    # Check if history record already exists for this upgrade
                    existing_history = SubscriptionHistory.query.filter_by(
                        tenant_id=tenant.id,
                        to_plan_id=growth_plan.id
                    ).first()
                    
                    if not existing_history:
                        # Create subscription history record
                        subscription_history = SubscriptionHistory(
                            tenant_id=tenant.id,
                            user_id=user.id,
                            action="upgraded",
                            from_plan_id=None,  # We don't know the previous plan
                            to_plan_id=growth_plan.id,
                            reason="Manually assigned via script",
                            effective_date=datetime.utcnow()
                        )
                        db.session.add(subscription_history)
                        print(f"[INFO] Created history record for tenant ID: {tenant.id} (user: {user.email})")
                    else:
                        print(f"[INFO] History record already exists for tenant ID: {tenant.id}")
                    
                    db.session.commit()
                    success_count += 1
                    
                except Exception as e:
                    db.session.rollback()
                    print(f"[ERROR] Failed to update tenant ID {tenant.id}: {e}")
                    failed_count += 1
            
            print("\n" + "="*60)
            print(f"Summary:")
            print(f"  [SUCCESS] Updated: {success_count}")
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
    print("="*60)
    print("Updating Growth Plan Users (Timestamp & History)")
    print("="*60)
    print()
    
    success = update_growth_plan_users()
    
    if success:
        print("\n[SUCCESS] Script completed successfully!")
        print("[NOTE] Users may need to refresh their browser to see the updated plan")
        sys.exit(0)
    else:
        print("\n[ERROR] Script completed with errors!")
        sys.exit(1)

