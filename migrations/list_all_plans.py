"""
Script to list all plans in the database with details
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
from collections import defaultdict

# Load environment variables
load_dotenv(os.path.join(backend_dir, '.env'))

def create_minimal_app():
    """Create a minimal Flask app without heavy dependencies"""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app

def list_all_plans():
    """List all plans in the database"""
    app = create_minimal_app()
    
    with app.app_context():
        try:
            print("="*80)
            print("ALL PLANS IN DATABASE")
            print("="*80)
            
            all_plans = Plan.query.order_by(Plan.id).all()
            
            if not all_plans:
                print("\n[INFO] No plans found in database.")
                return
            
            print(f"\n[INFO] Total plans: {len(all_plans)}\n")
            
            # Group by name and billing cycle
            plans_by_key = defaultdict(list)
            for plan in all_plans:
                key = (plan.name, plan.billing_cycle or 'monthly')
                plans_by_key[key].append(plan)
            
            # Check for duplicates
            duplicates_found = False
            for (name, billing), plans_list in plans_by_key.items():
                if len(plans_list) > 1:
                    duplicates_found = True
                    print(f"\n[WARNING] DUPLICATE FOUND: {name} ({billing})")
                    for p in plans_list:
                        print(f"  - ID: {p.id}, Price: ${p.price_cents/100:.2f}, Stripe ID: {p.stripe_price_id}, Created: {p.created_at}")
            
            if not duplicates_found:
                print("[INFO] No duplicates found.\n")
            
            # List all plans
            print("\n" + "-"*80)
            print("DETAILED PLAN LIST:")
            print("-"*80)
            
            for plan in all_plans:
                price_dollars = plan.price_cents / 100
                quota = "Unlimited" if plan.jd_quota_per_month == -1 else plan.jd_quota_per_month
                subaccounts = "Unlimited" if plan.max_subaccounts == -1 else plan.max_subaccounts
                billing = plan.billing_cycle or 'monthly'
                
                print(f"\nPlan ID: {plan.id}")
                print(f"  Name: {plan.name}")
                print(f"  Billing Cycle: {billing}")
                print(f"  Price: ${price_dollars:.2f}")
                print(f"  Stripe Price ID: {plan.stripe_price_id}")
                print(f"  Quota: {quota} AI candidate matching per month")
                print(f"  Recruiters: {subaccounts}")
                print(f"  Created: {plan.created_at}")
                print(f"  Updated: {plan.updated_at}")
            
            print("\n" + "="*80)
            
        except Exception as e:
            print(f"\n[ERROR] Error listing plans: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    list_all_plans()

