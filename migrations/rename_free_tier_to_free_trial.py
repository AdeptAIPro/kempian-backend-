"""
Migration Script: Rename Free Tier to Free Trial
This script renames the "Free Tier" plan to "Free Trial" to match existing code references.

Usage:
    python backend/migrations/rename_free_tier_to_free_trial.py
    OR
    cd backend && python migrations/rename_free_tier_to_free_trial.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Set environment variable to skip heavy initialization
os.environ['SKIP_HEAVY_INIT'] = '1'
os.environ['SKIP_SEARCH_INIT'] = '1'
os.environ['MIGRATION_MODE'] = '1'

from flask import Flask
from app.db import db
from app.models import Plan
from app.config import Config
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(backend_dir, '.env'))

def create_minimal_app():
    """Create a minimal Flask app without heavy dependencies"""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app

def rename_free_tier_to_free_trial():
    """Rename Free Tier plan to Free Trial"""
    app = create_minimal_app()
    
    with app.app_context():
        try:
            print("[INFO] Starting plan name migration...")
            print("[INFO] Renaming 'Free Tier' to 'Free Trial'...")
            
            # Find all plans named "Free Tier"
            free_tier_plans = Plan.query.filter_by(name="Free Tier").all()
            
            if not free_tier_plans:
                print("[WARNING] No 'Free Tier' plans found in database.")
                print("[INFO] Checking for 'Free Trial' plans...")
                free_trial_plans = Plan.query.filter_by(name="Free Trial").all()
                if free_trial_plans:
                    print(f"[INFO] Found {len(free_trial_plans)} 'Free Trial' plan(s) already in database.")
                    for plan in free_trial_plans:
                        print(f"  - Free Trial (ID: {plan.id})")
                    print("[SUCCESS] No changes needed - 'Free Trial' plan already exists!")
                    return True
                else:
                    print("[ERROR] Neither 'Free Tier' nor 'Free Trial' plan found!")
                    return False
            
            print(f"[INFO] Found {len(free_tier_plans)} 'Free Tier' plan(s) to rename.")
            
            # Check if "Free Trial" already exists
            existing_free_trial = Plan.query.filter_by(name="Free Trial").first()
            if existing_free_trial:
                print(f"[WARNING] 'Free Trial' plan already exists (ID: {existing_free_trial.id})")
                print("[INFO] Will keep the existing 'Free Trial' and delete 'Free Tier' duplicates.")
                
                # If we have multiple Free Tier plans, we need to handle them
                if len(free_tier_plans) > 1:
                    print(f"[INFO] Found {len(free_tier_plans)} 'Free Tier' plans. Keeping the latest one (highest ID).")
                    # Sort by ID descending and keep the first one
                    free_tier_plans.sort(key=lambda p: p.id, reverse=True)
                    plan_to_rename = free_tier_plans[0]
                    plans_to_delete = free_tier_plans[1:]
                    
                    # Delete duplicates
                    for dup_plan in plans_to_delete:
                        print(f"[INFO] Deleting duplicate Free Tier plan (ID: {dup_plan.id})")
                        db.session.delete(dup_plan)
                    
                    # Rename the kept one
                    print(f"[INFO] Renaming Free Tier plan (ID: {plan_to_rename.id}) to Free Trial...")
                    plan_to_rename.name = "Free Trial"
                else:
                    # Just rename the single Free Tier plan
                    print(f"[INFO] Renaming Free Tier plan (ID: {free_tier_plans[0].id}) to Free Trial...")
                    free_tier_plans[0].name = "Free Trial"
            else:
                # No Free Trial exists, rename Free Tier to Free Trial
                if len(free_tier_plans) > 1:
                    print(f"[WARNING] Found {len(free_tier_plans)} 'Free Tier' plans. Keeping the latest one (highest ID).")
                    # Sort by ID descending and keep the first one
                    free_tier_plans.sort(key=lambda p: p.id, reverse=True)
                    plan_to_rename = free_tier_plans[0]
                    plans_to_delete = free_tier_plans[1:]
                    
                    # Delete duplicates
                    for dup_plan in plans_to_delete:
                        print(f"[INFO] Deleting duplicate Free Tier plan (ID: {dup_plan.id})")
                        db.session.delete(dup_plan)
                    
                    # Rename the kept one
                    print(f"[INFO] Renaming Free Tier plan (ID: {plan_to_rename.id}) to Free Trial...")
                    plan_to_rename.name = "Free Trial"
                else:
                    # Just rename the single Free Tier plan
                    print(f"[INFO] Renaming Free Tier plan (ID: {free_tier_plans[0].id}) to Free Trial...")
                    free_tier_plans[0].name = "Free Trial"
            
            # Commit changes
            db.session.commit()
            print("[SUCCESS] Successfully renamed 'Free Tier' to 'Free Trial'!")
            
            # Verify the change
            free_trial_plan = Plan.query.filter_by(name="Free Trial").first()
            if free_trial_plan:
                print(f"\n[VERIFICATION] Free Trial plan confirmed:")
                print(f"  - Name: {free_trial_plan.name}")
                print(f"  - ID: {free_trial_plan.id}")
                print(f"  - Price: ${free_trial_plan.price_cents / 100:.2f}")
                print(f"  - Job Postings: {free_trial_plan.jd_quota_per_month if free_trial_plan.jd_quota_per_month != -1 else 'Unlimited'}")
                print(f"  - Recruiters: {free_trial_plan.max_subaccounts if free_trial_plan.max_subaccounts != -1 else 'Unlimited'}")
            
            return True
            
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error renaming plan: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    success = rename_free_tier_to_free_trial()
    if success:
        print("\n[SUCCESS] Migration completed successfully!")
        sys.exit(0)
    else:
        print("\n[ERROR] Migration failed!")
        sys.exit(1)

