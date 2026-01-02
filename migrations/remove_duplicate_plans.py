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
from app.models import Plan, Tenant, SubscriptionTransaction, SubscriptionHistory
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
            
            # Step 1: Get all plans and group by name AND billing_cycle
            print("\n[INFO] Analyzing existing plans...")
            all_plans = Plan.query.order_by(Plan.id).all()
            
            if not all_plans:
                print("[INFO] No plans found in database. Nothing to clean up.")
                return
            
            # Group plans by (name, billing_cycle) tuple
            plans_by_key = defaultdict(list)
            for plan in all_plans:
                key = (plan.name, plan.billing_cycle or 'monthly')
                plans_by_key[key].append(plan)
            
            print(f"[INFO] Found {len(all_plans)} total plan(s) with {len(plans_by_key)} unique plan name/billing cycle combination(s)")
            
            # Step 2: Identify duplicates and plan to keep
            plans_to_keep = {}
            plans_to_delete = []
            
            for (plan_name, billing_cycle), plans_list in plans_by_key.items():
                if len(plans_list) > 1:
                    # Multiple plans with same name and billing_cycle - keep the one with highest ID
                    plans_list.sort(key=lambda p: p.id, reverse=True)
                    keep_plan = plans_list[0]  # Highest ID
                    delete_plans = plans_list[1:]  # All others
                    
                    plans_to_keep[(plan_name, billing_cycle)] = keep_plan
                    plans_to_delete.extend(delete_plans)
                    
                    print(f"\n[INFO] Found {len(plans_list)} duplicate(s) for '{plan_name}' ({billing_cycle}):")
                    print(f"  [KEEP] Plan ID: {keep_plan.id} (created: {keep_plan.created_at})")
                    for dup_plan in delete_plans:
                        print(f"  [DELETE] Plan ID: {dup_plan.id} (created: {dup_plan.created_at})")
                else:
                    # Only one plan with this name and billing_cycle - keep it
                    plans_to_keep[(plan_name, billing_cycle)] = plans_list[0]
                    print(f"\n[INFO] Plan '{plan_name}' ({billing_cycle}) has no duplicates (ID: {plans_list[0].id})")
            
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
                
                # Reassign tenants to the kept plan with the same name and billing_cycle
                reassignment_map = {}
                for tenant in tenants_to_reassign:
                    old_plan = Plan.query.get(tenant.plan_id)
                    if old_plan:
                        key = (old_plan.name, old_plan.billing_cycle or 'monthly')
                        if key in plans_to_keep:
                            new_plan = plans_to_keep[key]
                            if tenant.plan_id != new_plan.id:
                                tenant.plan_id = new_plan.id
                                plan_key_str = f"{old_plan.name} ({old_plan.billing_cycle or 'monthly'})"
                                reassignment_map[plan_key_str] = reassignment_map.get(plan_key_str, 0) + 1
                
                if reassignment_map:
                    db.session.commit()
                    print("[INFO] Tenant reassignments:")
                    for plan_key_str, count in reassignment_map.items():
                        print(f"  - {count} tenant(s) reassigned to new '{plan_key_str}'")
                else:
                    print("[INFO] No tenant reassignments needed")
            else:
                print("[INFO] No tenants using plans that will be deleted")
            
            # Step 4: Reassign transaction and history records
            print(f"\n[INFO] Reassigning transaction and history records...")
            trans_reassigned = 0
            hist_reassigned = 0
            
            # Create mapping of old plan IDs to new plan IDs
            plan_id_mapping = {}
            for old_plan in plans_to_delete:
                key = (old_plan.name, old_plan.billing_cycle or 'monthly')
                if key in plans_to_keep:
                    plan_id_mapping[old_plan.id] = plans_to_keep[key].id
            
            for old_plan_id, new_plan_id in plan_id_mapping.items():
                # Update transactions
                trans_count = SubscriptionTransaction.query.filter_by(plan_id=old_plan_id).count()
                if trans_count > 0:
                    SubscriptionTransaction.query.filter_by(plan_id=old_plan_id).update(
                        {'plan_id': new_plan_id}, 
                        synchronize_session=False
                    )
                    trans_reassigned += trans_count
                
                # Update previous_plan_id in transactions
                trans_prev_count = SubscriptionTransaction.query.filter_by(previous_plan_id=old_plan_id).count()
                if trans_prev_count > 0:
                    SubscriptionTransaction.query.filter_by(previous_plan_id=old_plan_id).update(
                        {'previous_plan_id': new_plan_id}, 
                        synchronize_session=False
                    )
                
                # Update history records
                hist_from_count = SubscriptionHistory.query.filter_by(from_plan_id=old_plan_id).count()
                if hist_from_count > 0:
                    SubscriptionHistory.query.filter_by(from_plan_id=old_plan_id).update(
                        {'from_plan_id': new_plan_id}, 
                        synchronize_session=False
                    )
                
                hist_to_count = SubscriptionHistory.query.filter_by(to_plan_id=old_plan_id).count()
                if hist_to_count > 0:
                    SubscriptionHistory.query.filter_by(to_plan_id=old_plan_id).update(
                        {'to_plan_id': new_plan_id}, 
                        synchronize_session=False
                    )
                    hist_reassigned += hist_to_count
            
            if trans_reassigned > 0 or hist_reassigned > 0:
                db.session.commit()
                print(f"[INFO] Reassigned {trans_reassigned} transaction(s) and {hist_reassigned} history record(s).")
            
            # Step 5: Delete duplicate plans
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
            remaining_plans = Plan.query.order_by(Plan.name, Plan.billing_cycle, Plan.id).all()
            for plan in remaining_plans:
                price_dollars = plan.price_cents / 100
                quota = "Unlimited" if plan.jd_quota_per_month == -1 else plan.jd_quota_per_month
                subaccounts = "Unlimited" if plan.max_subaccounts == -1 else plan.max_subaccounts
                billing = plan.billing_cycle or 'monthly'
                print(f"  - {plan.name} ({billing}) (ID: {plan.id}): ${price_dollars:.2f}/{billing}, {quota} AI candidate matching, {subaccounts} recruiters")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error removing duplicate plans: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    remove_duplicate_plans()

