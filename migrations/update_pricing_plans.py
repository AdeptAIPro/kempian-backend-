"""
Migration Script: Update Pricing Plans
This script updates the pricing plans in the database with the latest pricing information:
- Freelancer: $29/recruiter, up to 10 job postings/month
- Starter: $49/recruiter, up to 50 job postings/month
- Growth (Small Team): $99 for up to 5 recruiters, unlimited job posts
- Professional: $299/recruiter, unlimited placements
- Enterprise: Custom (Starting at $499/recruiter), unlimited seats

Usage:
    python backend/migrations/update_pricing_plans.py
    OR
    cd backend && python migrations/update_pricing_plans.py
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

def update_pricing_plans():
    """Update pricing plans in the database - removes all old plans and creates only the new ones"""
    # Create minimal app without heavy model loading
    app = create_minimal_app()
    
    with app.app_context():
        try:
            print("[INFO] Starting pricing plan migration...")
            
            # Define the new pricing plans based on the latest pricing table
            new_plan_names = ['Free Trial', 'Freelancer', 'Starter', 'Growth', 'Professional', 'Enterprise']
            plans_data = [
                {
                    'name': 'Free Trial',
                    'price_cents': 0,  # Free
                    'stripe_price_id': 'price_free_tier',  # Placeholder - update with actual Stripe price ID
                    'jd_quota_per_month': 5,  # Limited free tier - 5 job postings per month
                    'max_subaccounts': 1,  # 1 recruiter
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Freelancer',
                    'price_cents': 2900,  # $29.00
                    'stripe_price_id': 'price_freelancer_monthly',  # Placeholder - update with actual Stripe price ID
                    'jd_quota_per_month': 10,  # up to 10 job postings per month
                    'max_subaccounts': 1,  # 1 recruiter
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Starter',
                    'price_cents': 4900,  # $49.00
                    'stripe_price_id': 'price_starter_monthly',  # Placeholder - update with actual Stripe price ID
                    'jd_quota_per_month': 50,  # up to 50 job postings per month
                    'max_subaccounts': 1,  # 1 recruiter
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Growth',
                    'price_cents': 9900,  # $99.00 for up to 5 recruiters
                    'stripe_price_id': 'price_growth_monthly',  # Placeholder - update with actual Stripe price ID
                    'jd_quota_per_month': -1,  # -1 means unlimited job posts
                    'max_subaccounts': 5,  # up to 5 recruiters
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Professional',
                    'price_cents': 29900,  # $299.00 per recruiter
                    'stripe_price_id': 'price_professional_monthly',  # Placeholder - update with actual Stripe price ID
                    'jd_quota_per_month': -1,  # -1 means unlimited placements
                    'max_subaccounts': 10,  # reasonable default, can be adjusted
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Enterprise',
                    'price_cents': 49900,  # $499.00 per recruiter (starting price)
                    'stripe_price_id': 'price_enterprise_custom',  # Placeholder - update with actual Stripe price ID
                    'jd_quota_per_month': -1,  # -1 means unlimited
                    'max_subaccounts': -1,  # -1 means unlimited seats
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                }
            ]
            
            # Step 1: Get all existing plans and tenants
            print("\n[INFO] Checking existing plans...")
            all_existing_plans = Plan.query.all()
            existing_plan_ids = [p.id for p in all_existing_plans]
            existing_plan_names = [p.name for p in all_existing_plans]
            
            print(f"[INFO] Found {len(all_existing_plans)} existing plan(s) to delete")
            if len(all_existing_plans) > 0:
                print(f"[INFO] Plan names: {', '.join(set(existing_plan_names))}")
            
            # Step 2: Get tenants using existing plans
            tenants_with_plans = []
            if existing_plan_ids:
                from app.models import Tenant
                tenants_with_plans = Tenant.query.filter(Tenant.plan_id.in_(existing_plan_ids)).all()
                
                if tenants_with_plans:
                    print(f"[INFO] Found {len(tenants_with_plans)} tenants using existing plans.")
                else:
                    print("[INFO] No tenants using existing plans.")
            
            # Step 3: Create fresh new plans first
            print("\n[INFO] Creating fresh new pricing plans...")
            created_plans = []
            new_plan_ids = []
            for plan_data in plans_data:
                print(f"[INFO] Creating plan: {plan_data['name']}")
                new_plan = Plan(**plan_data)
                db.session.add(new_plan)
                created_plans.append(plan_data['name'])
            
            # Commit new plans to get their IDs
            db.session.commit()
            
            # Get the IDs of newly created plans (keep only the newest one for each name)
            # Use no_autoflush to prevent premature flushes during queries
            with db.session.no_autoflush:
                seen_plan_names = set()
                for plan_name in new_plan_names:
                    if plan_name not in seen_plan_names:
                        # Get the newest plan with this name (highest ID = most recently created)
                        plan = Plan.query.filter_by(name=plan_name).order_by(Plan.id.desc()).first()
                        if plan:
                            new_plan_ids.append(plan.id)
                            seen_plan_names.add(plan_name)
                            print(f"[INFO] Keeping plan: {plan_name} (ID: {plan.id})")
            
            print(f"[INFO] Created {len(created_plans)} new plan(s) with IDs: {new_plan_ids}\n")
            
            # Step 4: Reassign tenants to new Free Trial plan FIRST (before deleting any plans)
            if tenants_with_plans:
                # Get the Free Trial plan ID from the new plans we just created
                free_trial_plan_id = None
                for plan_id in new_plan_ids:
                    plan = db.session.get(Plan, plan_id)  # Use get() which doesn't require a query
                    if plan and plan.name == 'Free Trial':
                        free_trial_plan_id = plan_id
                        break
                
                # If not found, try querying (shouldn't happen but just in case)
                if not free_trial_plan_id:
                    free_trial_plan = Plan.query.filter_by(name='Free Trial').order_by(Plan.id.desc()).first()
                    if free_trial_plan:
                        free_trial_plan_id = free_trial_plan.id
                
                if free_trial_plan_id:
                    print(f"[INFO] Reassigning {len(tenants_with_plans)} tenant(s) to new Free Trial plan (ID: {free_trial_plan_id})...")
                    # Update tenants in batches using ORM
                    batch_size = 50
                    updated_count = 0
                    
                    for i in range(0, len(tenants_with_plans), batch_size):
                        batch = tenants_with_plans[i:i + batch_size]
                        for tenant in batch:
                            tenant.plan_id = free_trial_plan_id
                        db.session.commit()  # Commit each batch
                        updated_count += len(batch)
                        print(f"[INFO] Updated {updated_count}/{len(tenants_with_plans)} tenants...")
                    
                    print(f"[INFO] Successfully reassigned {len(tenants_with_plans)} tenant(s) to Free Trial plan.")
                else:
                    print("[ERROR] Free Trial plan not found after creation! This should not happen.")
            
            # Step 5: Delete ALL old plans (everything except the new ones we just created)
            print("\n[INFO] Deleting ALL old plans (including duplicates)...")
            deleted_plans = []
            
            # Delete all plans that are NOT in our new plan IDs list
            # Use no_autoflush to prevent issues during deletion
            with db.session.no_autoflush:
                all_plans_now = Plan.query.all()
                for plan in all_plans_now:
                    if plan.id not in new_plan_ids:
                        plan_name = plan.name
                        plan_id = plan.id
                        print(f"[INFO] Deleting plan: {plan_name} (ID: {plan_id})")
                        db.session.delete(plan)
                        deleted_plans.append(f"{plan_name} (ID: {plan_id})")
            
            # Commit all deletions at once
            db.session.commit()
            print(f"[INFO] Deleted {len(deleted_plans)} old plan(s).\n")
            deleted_count = len(deleted_plans)
            
            print(f"\n[SUCCESS] Successfully updated pricing plans!")
            print(f"   - Deleted: {deleted_count} old plan(s)")
            print(f"   - Created: {len(created_plans)} new plan(s)")
            print("\n" + "="*80)
            print("CURRENT PLANS IN DATABASE - DETAILED INFORMATION")
            print("="*80)
            
            # Detailed plan information mapping
            plan_details = {
                'Free Trial': {
                    'description': 'Limited free tier with basic features. Perfect for trying out the platform.',
                    'ai_capabilities': 'Basic AI matching for job descriptions and candidates.',
                    'notes': 'Free tier with limited features. Upgrade to unlock more capabilities.'
                },
                'Freelancer': {
                    'description': 'AI job-description generator, basic candidate search & matching, up to 10 job postings per month, basic compliance checklist, standard email/SMS templates.',
                    'ai_capabilities': 'Basic AI matching for job descriptions and candidates.',
                    'notes': 'Ideal for solo recruiters; additional job posts $5 each; email support.'
                },
                'Starter': {
                    'description': 'AI job-description generator and AI matchmaker, candidate search, up to 50 job posts per month, basic compliance management.',
                    'ai_capabilities': 'AI job-description generator and basic AI matching.',
                    'notes': 'Affordable entry for small teams; includes standard support and community resources.'
                },
                'Growth': {
                    'description': 'Everything in Starter plus unlimited job posts, team collaboration & interview scheduling, up to 3 integrations (e.g., LinkedIn, Indeed, email), simple onboarding workflows.',
                    'ai_capabilities': 'Enhanced AI candidate ranking and matching with predictive suggestions.',
                    'notes': 'Suited for small-mid agencies; additional seats $15/month; includes chat support.'
                },
                'Professional': {
                    'description': 'Everything in Growth plus full ATS pipeline with customizable stages, cross-border compliance, onboarding & payroll integration, advanced analytics dashboard, API access, Slack/Teams notifications.',
                    'ai_capabilities': 'Advanced AI including behavioural insights, predictive analytics and fairness controls.',
                    'notes': 'Best for mid-large agencies; unlimited placements; dedicated support available.'
                },
                'Enterprise': {
                    'description': 'All modules (Jobs, ATS, Onboarding, Compliance, Payroll, Finance, Background Checks, Analytics, Integration Hub), unlimited seats, custom workflows, advanced AI models, bias auditing, single sign-on, dedicated account manager and SLA.',
                    'ai_capabilities': 'Dedicated AI models with customizable algorithms and bias auditing.',
                    'notes': 'Custom contracts; per-hire fees ($50–$150) optional; premium connectors and on-prem/VPC deployments available.'
                }
            }
            
            for plan_data in plans_data:
                plan = Plan.query.filter_by(name=plan_data['name']).first()
                if plan:
                    price_dollars = plan.price_cents / 100
                    quota = "Unlimited" if plan.jd_quota_per_month == -1 else f"{plan.jd_quota_per_month} job postings"
                    subaccounts = "Unlimited seats" if plan.max_subaccounts == -1 else f"{plan.max_subaccounts} recruiter(s)"
                    
                    details = plan_details.get(plan.name, {})
                    
                    print(f"\n{'─'*80}")
                    print(f"PLAN: {plan.name}")
                    print(f"{'─'*80}")
                    print(f"  Monthly Price:     ${price_dollars:.2f} / recruiter")
                    print(f"  Job Postings:      {quota} per month")
                    print(f"  Recruiters:        {subaccounts}")
                    print(f"  Billing Cycle:     {plan.billing_cycle.capitalize()}")
                    print(f"  Stripe Price ID:   {plan.stripe_price_id}")
                    print(f"  Plan ID:           {plan.id}")
                    
                    if details:
                        print(f"\n  Included Modules & Features:")
                        print(f"    {details.get('description', 'N/A')}")
                        print(f"\n  AI Capabilities:")
                        print(f"    {details.get('ai_capabilities', 'N/A')}")
                        print(f"\n  Notes:")
                        print(f"    {details.get('notes', 'N/A')}")
            
            print(f"\n{'─'*80}")
            print("\n[NOTE] Please update stripe_price_id values with actual Stripe price IDs from your Stripe dashboard.")
            print("="*80)
            
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error updating plans: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    update_pricing_plans()

