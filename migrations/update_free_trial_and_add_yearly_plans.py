"""
Migration Script: Update Free Trial and Add Yearly Plans
This script:
1. Updates Free Trial plan to remove team member option (max_subaccounts = 0)
2. Creates yearly plans with 2 months discount (10 months price for 12 months)

Usage:
    python backend/migrations/update_free_trial_and_add_yearly_plans.py
    OR
    cd backend && python migrations/update_free_trial_and_add_yearly_plans.py
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
from datetime import datetime

# Load environment variables
load_dotenv(os.path.join(backend_dir, '.env'))

def create_minimal_app():
    """Create a minimal Flask app without heavy dependencies"""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app

def update_free_trial_and_add_yearly_plans():
    """Update Free Trial plan and add yearly plans with 2 months discount"""
    app = create_minimal_app()
    
    with app.app_context():
        try:
            print("[INFO] Starting Free Trial update and yearly plans migration...")
            
            # Step 1: Update Free Trial plan to remove team member option
            print("\n[INFO] Updating Free Trial plan...")
            free_trial_plan = Plan.query.filter_by(name="Free Trial").first()
            if free_trial_plan:
                old_max_subaccounts = free_trial_plan.max_subaccounts
                free_trial_plan.max_subaccounts = 0
                db.session.commit()
                print(f"[SUCCESS] Updated Free Trial plan:")
                print(f"  - max_subaccounts: {old_max_subaccounts} -> 0 (team members disabled)")
            else:
                print("[WARNING] Free Trial plan not found. Creating it...")
                free_trial_plan = Plan(
                    name='Free Trial',
                    price_cents=0,
                    stripe_price_id='price_free_tier',
                    jd_quota_per_month=5,
                    max_subaccounts=0,  # No team members for free trial
                    is_trial=False,
                    trial_days=0,
                    billing_cycle='monthly'
                )
                db.session.add(free_trial_plan)
                db.session.commit()
                print(f"[SUCCESS] Created Free Trial plan with max_subaccounts=0")
            
            # Step 2: Get existing monthly plans to create yearly versions
            print("\n[INFO] Checking for existing monthly plans...")
            from sqlalchemy import and_
            monthly_plans = Plan.query.filter(
                and_(
                    Plan.billing_cycle == 'monthly',
                    Plan.name != 'Free Trial',
                    Plan.name != 'Enterprise'
                )
            ).all()
            
            if not monthly_plans:
                print("[WARNING] No monthly plans found. Please run update_pricing_plans.py first.")
                return
            
            print(f"[INFO] Found {len(monthly_plans)} monthly plan(s) to create yearly versions for")
            
            # Step 3: Create yearly plans with 2 months discount
            # 2 months discount = 10 months price for 12 months
            yearly_plans_created = []
            yearly_plans_updated = []
            
            for monthly_plan in monthly_plans:
                # Check if yearly plan already exists
                existing_yearly = Plan.query.filter_by(
                    name=monthly_plan.name,
                    billing_cycle='yearly'
                ).first()
                
                # Calculate yearly price: monthly price * 10 (2 months free)
                yearly_price_cents = monthly_plan.price_cents * 10
                
                # Generate Stripe price ID for yearly plan
                yearly_stripe_id = monthly_plan.stripe_price_id.replace('_monthly', '_yearly')
                if '_monthly' not in monthly_plan.stripe_price_id:
                    yearly_stripe_id = f"{monthly_plan.stripe_price_id}_yearly"
                
                if existing_yearly:
                    # Update existing yearly plan
                    old_price = existing_yearly.price_cents
                    existing_yearly.price_cents = yearly_price_cents
                    existing_yearly.stripe_price_id = yearly_stripe_id
                    existing_yearly.jd_quota_per_month = monthly_plan.jd_quota_per_month
                    existing_yearly.max_subaccounts = monthly_plan.max_subaccounts
                    db.session.commit()
                    yearly_plans_updated.append({
                        'name': monthly_plan.name,
                        'old_price': old_price,
                        'new_price': yearly_price_cents
                    })
                    print(f"[SUCCESS] Updated yearly plan: {monthly_plan.name}")
                    print(f"  - Price: ${old_price/100:.2f} -> ${yearly_price_cents/100:.2f} (2 months discount)")
                else:
                    # Create new yearly plan
                    yearly_plan = Plan(
                        name=monthly_plan.name,
                        price_cents=yearly_price_cents,
                        stripe_price_id=yearly_stripe_id,
                        jd_quota_per_month=monthly_plan.jd_quota_per_month,
                        max_subaccounts=monthly_plan.max_subaccounts,
                        is_trial=monthly_plan.is_trial,
                        trial_days=monthly_plan.trial_days,
                        billing_cycle='yearly'
                    )
                    db.session.add(yearly_plan)
                    db.session.commit()
                    yearly_plans_created.append({
                        'name': monthly_plan.name,
                        'price': yearly_price_cents,
                        'monthly_price': monthly_plan.price_cents
                    })
                    print(f"[SUCCESS] Created yearly plan: {monthly_plan.name}")
                    print(f"  - Monthly: ${monthly_plan.price_cents/100:.2f}/month")
                    print(f"  - Yearly: ${yearly_price_cents/100:.2f}/year (2 months free)")
            
            print(f"\n[SUCCESS] Migration completed!")
            print(f"  - Updated Free Trial: max_subaccounts set to 0")
            print(f"  - Created {len(yearly_plans_created)} yearly plan(s)")
            print(f"  - Updated {len(yearly_plans_updated)} yearly plan(s)")
            
            # Display summary
            print("\n" + "="*80)
            print("YEARLY PLANS SUMMARY")
            print("="*80)
            
            all_yearly_plans = Plan.query.filter_by(billing_cycle='yearly').all()
            for plan in all_yearly_plans:
                monthly_equivalent = plan.price_cents / 10
                monthly_savings = (plan.price_cents / 12) - monthly_equivalent
                print(f"\n{plan.name} (Yearly):")
                print(f"  - Yearly Price: ${plan.price_cents/100:.2f}/year")
                print(f"  - Monthly Equivalent: ${monthly_equivalent/100:.2f}/month")
                print(f"  - Savings: ${monthly_savings/100:.2f}/month (2 months free)")
                print(f"  - Stripe Price ID: {plan.stripe_price_id}")
            
            print("\n" + "="*80)
            print("[NOTE] Please update stripe_price_id values with actual Stripe price IDs from your Stripe dashboard.")
            print("="*80)
            
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error updating plans: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    update_free_trial_and_add_yearly_plans()

