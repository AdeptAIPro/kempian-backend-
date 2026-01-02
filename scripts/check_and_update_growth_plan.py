"""
Script to check and update Growth Plan max_subaccounts value.
Usage: python -m scripts.check_and_update_growth_plan
"""
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variable to skip heavy initialization
os.environ['SKIP_HEAVY_INIT'] = '1'

from flask import Flask
from app.db import db
from app.models import Plan, Tenant, User
from app.config import Config
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def create_minimal_app():
    """Create a minimal Flask app without heavy dependencies"""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app

def check_and_update_growth_plan():
    """Check and update Growth Plan max_subaccounts"""
    app = create_minimal_app()
    
    with app.app_context():
        try:
            # Find Growth Plan
            growth_plan = Plan.query.filter_by(name="Growth").first()
            if not growth_plan:
                growth_plan = Plan.query.filter(Plan.name.ilike("growth")).first()
            
            if not growth_plan:
                print("[ERROR] Growth Plan not found in database")
                print("Available plans:")
                plans = Plan.query.all()
                for plan in plans:
                    print(f"  - {plan.name} (ID: {plan.id}) - max_subaccounts: {plan.max_subaccounts}")
                return False
            
            print(f"[SUCCESS] Found Growth Plan: {growth_plan.name} (ID: {growth_plan.id})")
            print(f"   Current max_subaccounts: {growth_plan.max_subaccounts}")
            print(f"   jd_quota_per_month: {growth_plan.jd_quota_per_month}")
            
            # Check if max_subaccounts is 0 or less than expected
            if growth_plan.max_subaccounts == 0:
                print("\n[WARNING] max_subaccounts is 0, updating to 3 (as shown in dashboard)")
                growth_plan.max_subaccounts = 3
                db.session.commit()
                print("[SUCCESS] Updated max_subaccounts to 3")
            elif growth_plan.max_subaccounts < 3:
                print(f"\n[WARNING] max_subaccounts is {growth_plan.max_subaccounts}, updating to 3")
                growth_plan.max_subaccounts = 3
                db.session.commit()
                print(f"[SUCCESS] Updated max_subaccounts from {growth_plan.max_subaccounts} to 3")
            else:
                print(f"\n[INFO] max_subaccounts is already set to {growth_plan.max_subaccounts}, no update needed")
            
            # Check tenants with Growth plan
            print("\n" + "="*60)
            print("Checking tenants with Growth plan:")
            print("="*60)
            
            tenants_with_growth = Tenant.query.filter_by(plan_id=growth_plan.id).all()
            print(f"Found {len(tenants_with_growth)} tenant(s) with Growth plan")
            
            for tenant in tenants_with_growth:
                users = User.query.filter_by(tenant_id=tenant.id).all()
                owner_count = sum(1 for u in users if u.role == 'owner')
                subuser_count = sum(1 for u in users if u.role == 'subuser')
                total_count = len(users)
                
                print(f"\n  Tenant ID: {tenant.id}")
                print(f"    Total users: {total_count}")
                print(f"    Owners: {owner_count}")
                print(f"    Subusers: {subuser_count}")
                print(f"    Max allowed: {growth_plan.max_subaccounts}")
                print(f"    Can add more: {subuser_count < growth_plan.max_subaccounts}")
            
            print("\n" + "="*60)
            print("[SUCCESS] Script completed successfully!")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return False

if __name__ == "__main__":
    print("="*60)
    print("Checking and Updating Growth Plan max_subaccounts")
    print("="*60)
    print()
    
    success = check_and_update_growth_plan()
    
    if success:
        print("\n[SUCCESS] Script completed successfully!")
        sys.exit(0)
    else:
        print("\n[ERROR] Script completed with errors!")
        sys.exit(1)

