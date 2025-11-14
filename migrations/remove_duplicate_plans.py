"""
Migration Script: Remove Duplicate Plans
This script removes duplicate plans from the database, keeping only the latest (highest ID) for each plan name.

Usage:
    python backend/migrations/remove_duplicate_plans.py
    OR
    cd backend && python migrations/remove_duplicate_plans.py
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
from app.models import Plan, Tenant
from app.config import Config
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv(os.path.join(backend_dir, '.env'))

def create_minimal_app():
    """Create a minimal Flask app without heavy dependencies"""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app

def remove_duplicate_plans():
    """Remove duplicate plans, keeping only the latest (highest ID) for each plan name"""
    app = create_minimal_app()
    
    with app.app_context():
        try:
            print("[INFO] Starting duplicate plan cleanup...")
            
            # Step 1: Get all plans and group by name
            print("\n[INFO] Analyzing existing plans...")
            all_plans = Plan.query.order_by(Plan.id).all()
            
            if not all_plans:
                print("[INFO] No plans found in database. Nothing to clean up.")
                return
            
            # Group plans by name
            plans_by_name = defaultdict(list)
            for plan in all_plans:
                plans_by_name[plan.name].append(plan)
            
            print(f"[INFO] Found {len(all_plans)} total plan(s) with {len(plans_by_name)} unique plan name(s)")
            
            # Step 2: Identify duplicates and plan to keep
            plans_to_keep = {}
            plans_to_delete = []
            
            for plan_name, plans_list in plans_by_name.items():
                if len(plans_list) > 1:
                    # Multiple plans with same name - keep the one with highest ID
                    plans_list.sort(key=lambda p: p.id, reverse=True)
                    keep_plan = plans_list[0]  # Highest ID
                    delete_plans = plans_list[1:]  # All others
                    
                    plans_to_keep[plan_name] = keep_plan
                    plans_to_delete.extend(delete_plans)
                    
                    print(f"\n[INFO] Found {len(plans_list)} duplicate(s) for '{plan_name}':")
                    print(f"  [KEEP] Plan ID: {keep_plan.id} (created: {keep_plan.created_at})")
                    for dup_plan in delete_plans:
                        print(f"  [DELETE] Plan ID: {dup_plan.id} (created: {dup_plan.created_at})")
                else:
                    # Only one plan with this name - keep it
                    plans_to_keep[plan_name] = plans_list[0]
                    print(f"\n[INFO] Plan '{plan_name}' has no duplicates (ID: {plans_list[0].id})")
            
            if not plans_to_delete:
                print("\n[SUCCESS] No duplicate plans found. Database is clean!")
                return
            
            print(f"\n[INFO] Total plans to delete: {len(plans_to_delete)}")
            
            # Step 3: Get tenants using plans that will be deleted
            delete_plan_ids = [p.id for p in plans_to_delete]
            tenants_to_reassign = Tenant.query.filter(Tenant.plan_id.in_(delete_plan_ids)).all()
            
            if tenants_to_reassign:
                print(f"\n[INFO] Found {len(tenants_to_reassign)} tenant(s) using plans that will be deleted")
                print("[INFO] Reassigning tenants to the latest plan with the same name...")
                
                # Reassign tenants to the kept plan with the same name
                reassignment_map = {}
                for tenant in tenants_to_reassign:
                    old_plan = Plan.query.get(tenant.plan_id)
                    if old_plan and old_plan.name in plans_to_keep:
                        new_plan = plans_to_keep[old_plan.name]
                        if tenant.plan_id != new_plan.id:
                            tenant.plan_id = new_plan.id
                            reassignment_map[old_plan.name] = reassignment_map.get(old_plan.name, 0) + 1
                
                if reassignment_map:
                    db.session.commit()
                    print("[INFO] Tenant reassignments:")
                    for plan_name, count in reassignment_map.items():
                        print(f"  - {count} tenant(s) reassigned from old '{plan_name}' to new '{plan_name}' (ID: {plans_to_keep[plan_name].id})")
                else:
                    print("[INFO] No tenant reassignments needed")
            else:
                print("[INFO] No tenants using plans that will be deleted")
            
            # Step 4: Delete duplicate plans
            print(f"\n[INFO] Deleting {len(plans_to_delete)} duplicate plan(s)...")
            deleted_count = 0
            
            with db.session.no_autoflush:
                for plan in plans_to_delete:
                    plan_name = plan.name
                    plan_id = plan.id
                    print(f"[INFO] Deleting duplicate plan: {plan_name} (ID: {plan_id})")
                    db.session.delete(plan)
                    deleted_count += 1
            
            # Commit all deletions
            db.session.commit()
            
            print(f"\n[SUCCESS] Successfully removed {deleted_count} duplicate plan(s)!")
            print(f"\n[INFO] Remaining plans in database:")
            remaining_plans = Plan.query.order_by(Plan.name, Plan.id).all()
            for plan in remaining_plans:
                price_dollars = plan.price_cents / 100
                quota = "Unlimited" if plan.jd_quota_per_month == -1 else plan.jd_quota_per_month
                subaccounts = "Unlimited" if plan.max_subaccounts == -1 else plan.max_subaccounts
                print(f"  - {plan.name} (ID: {plan.id}): ${price_dollars:.2f}/month, {quota} job postings, {subaccounts} recruiters")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error removing duplicate plans: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    remove_duplicate_plans()

