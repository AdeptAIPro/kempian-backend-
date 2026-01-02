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
            # Yearly plans get 20% discount (2 months free per year)
            # Formula: monthly_price * 12 * 0.8 = yearly_price
            new_plan_names = ['Free Trial', 'Freelancer', 'Starter', 'Growth', 'Professional', 'Enterprise']
            
            # Base monthly plans data - each plan has a unique price ID
            monthly_plans_data = [
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
                    'stripe_price_id': 'price_1RrJkOKVj190YQJbUPk9z136',  # Unique price ID for Freelancer
                    'jd_quota_per_month': 40,  # up to 10 job postings per month
                    'max_subaccounts': 1,  # 1 recruiter
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Starter',
                    'price_cents': 4900,  # $49.00
                    'stripe_price_id': 'price_1RHnLzKVj190YQJbKhavdyiI',  # Unique price ID for Starter
                    'jd_quota_per_month': 50,  # up to 50 job postings per month
                    'max_subaccounts': 1,  # 1 recruiter
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Growth',
                    'price_cents': 19900,  # $99.00 for up to 5 recruiters
                    'stripe_price_id': 'price_1RHnPkKVj190YQJbKtOCaZgj',  # Unique price ID for Growth
                    'jd_quota_per_month': -1,  # -1 means unlimited job posts
                    'max_subaccounts': 5,  # up to 5 recruiters
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Professional',
                    'price_cents': 29900,  # $299.00 per recruiter
                    'stripe_price_id': 'price_1RjMKNKVj190YQJbhBuhcjfQ',  # Unique price ID for Professional
                    'jd_quota_per_month': -1,  # -1 means unlimited placements
                    'max_subaccounts': 10,  # reasonable default, can be adjusted
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Enterprise',
                    'price_cents': 59900,  # $499.00 per recruiter (starting price)
                    'stripe_price_id': 'price_1RjMMuKVj190YQJbBX2LhoTs',  # Unique price ID for Enterprise
                    'jd_quota_per_month': -1,  # -1 means unlimited
                    'max_subaccounts': -1,  # -1 means unlimited seats
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                }
            ]
            
            # Create yearly plans from monthly plans (20% discount)
            yearly_plans_data = []
            for monthly_plan in monthly_plans_data:
                # Skip Free Trial for yearly (it's free anyway)
                if monthly_plan['name'] == 'Free Trial':
                    continue
                
                # Calculate yearly price: monthly * 12 * 0.8 (20% discount)
                yearly_price = int(monthly_plan['price_cents'] * 12 * 0.8)
                
                # Map Stripe price IDs for yearly plans
                # IMPORTANT: Configure separate price IDs for yearly plans here
                # Format: 'monthly_price_id': 'yearly_price_id'
                # If a monthly price ID is not in this mapping, it will use the monthly price ID as fallback
                yearly_stripe_id_mapping = {
                    # Freelancer - Monthly: price_1RrJkOKVj190YQJbUPk9z136
                    'price_1Shlm2KVj190YQJbh5X3xz29': 'price_1Shlm2KVj190YQJbh5X3xz29',  # Update with actual yearly price ID
                    
                    # Starter - Monthly: price_1RHnLzKVj190YQJbKhavdyiI
                    'price_1Shm5yKVj190YQJb3R5Ijrie': 'price_1Shm5yKVj190YQJb3R5Ijrie',  # Update with actual yearly price ID
                    
                    # Growth - Monthly: price_1RHnPkKVj190YQJbKtOCaZgj
                    'price_1Shm8VKVj190YQJbGWSfJwSX': 'price_1Shm8VKVj190YQJbGWSfJwSX',  # Update with actual yearly price ID
                    
                    # Professional - Monthly: price_1RA9hQQLZiir9RFCqgZKpVaV
                    'price_1Shm7HKVj190YQJbmRtzlPZ8': 'price_1Shm7HKVj190YQJbmRtzlPZ8',  # Update with actual yearly price ID
                    
                    # Enterprise - Monthly: price_1RjMMuKVj190YQJbBX2LhoTs
                    'price_1Shm9KKVj190YQJbXgA5tsJW': 'price_1Shm9KKVj190YQJbXgA5tsJW',  # Update with actual yearly price ID
                    
                    # Free tier (no yearly plan)
                    'price_free_tier': 'price_free_tier'
                }
                
                # Get yearly price ID from mapping, or use monthly price ID as fallback
                yearly_stripe_price_id = yearly_stripe_id_mapping.get(
                    monthly_plan['stripe_price_id'], 
                    monthly_plan['stripe_price_id']  # Fallback to monthly price ID if not in mapping
                )
                
                yearly_plan = {
                    'name': monthly_plan['name'],
                    'price_cents': yearly_price,
                    'stripe_price_id': yearly_stripe_price_id,
                    'jd_quota_per_month': monthly_plan['jd_quota_per_month'],
                    'max_subaccounts': monthly_plan['max_subaccounts'],
                    'is_trial': monthly_plan['is_trial'],
                    'trial_days': monthly_plan['trial_days'],
                    'billing_cycle': 'yearly'
                }
                yearly_plans_data.append(yearly_plan)
            
            # Combine monthly and yearly plans
            plans_data = monthly_plans_data + yearly_plans_data
            
            # Step 1: Get all existing plans and create mapping
            print("\n[INFO] Checking existing plans...")
            all_existing_plans = Plan.query.all()
            existing_plan_names = [p.name for p in all_existing_plans]
            
            print(f"[INFO] Found {len(all_existing_plans)} existing plan(s) that will be deleted")
            if len(all_existing_plans) > 0:
                print(f"[INFO] Plan names: {', '.join(set(existing_plan_names))}")
            
            # Step 2: Get all tenants and their current plans (for reassignment mapping)
            from app.models import Tenant, SubscriptionTransaction, SubscriptionHistory
            all_tenants = Tenant.query.all()
            tenants_with_plans = [t for t in all_tenants if t.plan_id]
            
            if tenants_with_plans:
                print(f"[INFO] Found {len(tenants_with_plans)} tenant(s) that need plan reassignment.")
            else:
                print("[INFO] No tenants found.")
            
            # Step 3: Create mapping of old plans to new plans (by name and billing_cycle)
            # This will be used to reassign tenants automatically
            old_plan_mapping = {}
            for old_plan in all_existing_plans:
                plan_key = (old_plan.name, old_plan.billing_cycle)
                old_plan_mapping[old_plan.id] = plan_key
            
            # Step 4: Create fresh new plans
            print("\n[INFO] Creating fresh new pricing plans...")
            created_plans = []
            new_plans_map = {}  # Map (name, billing_cycle) -> plan_id
            
            for plan_data in plans_data:
                plan_display_name = f"{plan_data['name']} ({plan_data['billing_cycle']})"
                print(f"[INFO] Creating plan: {plan_display_name}")
                new_plan = Plan(**plan_data)
                db.session.add(new_plan)
                created_plans.append(plan_display_name)
            
            # Commit new plans to get their IDs
            db.session.commit()
            
            # Build mapping of (name, billing_cycle) -> new plan ID
            for plan_data in plans_data:
                plan_key = (plan_data['name'], plan_data['billing_cycle'])
                if plan_key not in new_plans_map:
                    # Get the newest plan with this name and billing_cycle (highest ID = most recently created)
                    plan = Plan.query.filter_by(
                        name=plan_data['name'], 
                        billing_cycle=plan_data['billing_cycle']
                    ).order_by(Plan.id.desc()).first()
                    if plan:
                        new_plans_map[plan_key] = plan.id
                        print(f"[INFO] New plan: {plan.name} ({plan.billing_cycle}) (ID: {plan.id})")
            
            print(f"[INFO] Created {len(created_plans)} new plan(s)\n")
            
            # Step 5: Reassign ALL tenants from old plans to new plans (matching by name and billing_cycle)
            # This automatically transfers users to the equivalent new plan
            reassigned_count = 0
            if tenants_with_plans:
                print(f"[INFO] Automatically reassigning {len(tenants_with_plans)} tenant(s) to new plans...")
                
                fallback_plan_id = new_plans_map.get(('Free Trial', 'monthly'))
                
                # Use bulk update for efficiency
                for old_plan_id, plan_key in old_plan_mapping.items():
                    new_plan_id = new_plans_map.get(plan_key)
                    
                    if new_plan_id:
                        # Reassign all tenants with this old plan to the new plan
                        tenant_count = Tenant.query.filter_by(plan_id=old_plan_id).count()
                        if tenant_count > 0:
                            Tenant.query.filter_by(plan_id=old_plan_id).update(
                                {'plan_id': new_plan_id}, 
                                synchronize_session=False
                            )
                            reassigned_count += tenant_count
                            print(f"[INFO] Reassigned {tenant_count} tenant(s) from {plan_key[0]} ({plan_key[1]}) old plan (ID: {old_plan_id}) to new plan (ID: {new_plan_id})")
                    else:
                        # No matching new plan found, assign to Free Trial as fallback
                        if fallback_plan_id:
                            tenant_count = Tenant.query.filter_by(plan_id=old_plan_id).count()
                            if tenant_count > 0:
                                Tenant.query.filter_by(plan_id=old_plan_id).update(
                                    {'plan_id': fallback_plan_id}, 
                                    synchronize_session=False
                                )
                                reassigned_count += tenant_count
                                print(f"[WARNING] Reassigned {tenant_count} tenant(s) from {plan_key[0]} ({plan_key[1]}) to Free Trial (no matching new plan found)")
                
                # Commit tenant reassignments
                if reassigned_count > 0:
                    db.session.commit()
                    print(f"[INFO] Successfully reassigned {reassigned_count} tenant(s) to new plans.\n")
                else:
                    print("[INFO] No tenants needed reassignment.\n")
            
            # Step 6: Reassign all transaction and history records
            print("[INFO] Reassigning transaction and history records...")
            trans_reassigned = 0
            hist_reassigned = 0
            
            for old_plan_id, plan_key in old_plan_mapping.items():
                new_plan_id = new_plans_map.get(plan_key)
                if new_plan_id:
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
                print(f"[INFO] Reassigned {trans_reassigned} transaction(s) and {hist_reassigned} history record(s).\n")
            
            # Step 7: Delete ALL old plans (now that all references have been reassigned)
            print("\n[INFO] Deleting ALL old plans...")
            deleted_plans = []
            new_plan_id_set = set(new_plans_map.values())
            fallback_plan_id = new_plans_map.get(('Free Trial', 'monthly'))
            
            # Get all plans and delete those that are not in our new plans
            all_plans_now = Plan.query.all()
            for plan in all_plans_now:
                # Skip if this is one of our new plans
                if plan.id in new_plan_id_set:
                    continue
                
                # Delete all old plans
                plan_name = plan.name
                plan_id = plan.id
                billing_cycle = plan.billing_cycle
                
                # Final safety check - make sure no tenants reference this plan
                tenant_check = Tenant.query.filter_by(plan_id=plan_id).count()
                if tenant_check > 0:
                    print(f"[WARNING] Plan {plan_name} (ID: {plan_id}) still has {tenant_check} tenant(s) - force reassigning...")
                    # Force reassign to new plan
                    plan_key = (plan_name, billing_cycle)
                    new_plan_id = new_plans_map.get(plan_key, fallback_plan_id)
                    if new_plan_id:
                        Tenant.query.filter_by(plan_id=plan_id).update(
                            {'plan_id': new_plan_id}, 
                            synchronize_session=False
                        )
                        db.session.commit()
                        print(f"[INFO] Force reassigned {tenant_check} tenant(s) to plan {new_plan_id}")
                
                print(f"[INFO] Deleting old plan: {plan_name} ({billing_cycle}) (ID: {plan_id})")
                db.session.delete(plan)
                deleted_plans.append(f"{plan_name} ({billing_cycle}) (ID: {plan_id})")
            
            # Commit all deletions
            db.session.commit()
            print(f"[INFO] Deleted {len(deleted_plans)} old plan(s).\n")
            deleted_count = len(deleted_plans)
            
            # Step 8: Verify we only have the correct plans and no duplicates
            print("\n[INFO] Verifying final plan state...")
            all_final_plans = Plan.query.all()
            plan_counts = {}
            for plan in all_final_plans:
                key = (plan.name, plan.billing_cycle)
                if key not in plan_counts:
                    plan_counts[key] = []
                plan_counts[key].append(plan.id)
            
            # Check for duplicates (shouldn't happen, but just in case)
            duplicates_found = False
            skipped_plans = []
            for key, ids in plan_counts.items():
                if len(ids) > 1:
                    duplicates_found = True
                    print(f"[WARNING] Found {len(ids)} duplicate plans for {key[0]} ({key[1]}): IDs {ids}")
                    # Keep only the newest one (highest ID)
                    ids_sorted = sorted(ids)
                    keep_id = ids_sorted[-1]  # Highest ID (newest plan)
                    ids_to_delete = ids_sorted[:-1]  # All except the highest ID
                    
                    for dup_id in ids_to_delete:
                        dup_plan = db.session.get(Plan, dup_id)
                        if dup_plan:
                            # Reassign all references using bulk updates
                            tenant_count = Tenant.query.filter_by(plan_id=dup_id).count()
                            if tenant_count > 0:
                                print(f"[INFO] Reassigning {tenant_count} tenant(s) from duplicate plan {dup_id} to new plan {keep_id}...")
                                Tenant.query.filter_by(plan_id=dup_id).update({'plan_id': keep_id}, synchronize_session=False)
                            
                            trans_count = SubscriptionTransaction.query.filter_by(plan_id=dup_id).count()
                            if trans_count > 0:
                                SubscriptionTransaction.query.filter_by(plan_id=dup_id).update({'plan_id': keep_id}, synchronize_session=False)
                            
                            trans_prev_count = SubscriptionTransaction.query.filter_by(previous_plan_id=dup_id).count()
                            if trans_prev_count > 0:
                                SubscriptionTransaction.query.filter_by(previous_plan_id=dup_id).update({'previous_plan_id': keep_id}, synchronize_session=False)
                            
                            hist_from_count = SubscriptionHistory.query.filter_by(from_plan_id=dup_id).count()
                            if hist_from_count > 0:
                                SubscriptionHistory.query.filter_by(from_plan_id=dup_id).update({'from_plan_id': keep_id}, synchronize_session=False)
                            
                            hist_to_count = SubscriptionHistory.query.filter_by(to_plan_id=dup_id).count()
                            if hist_to_count > 0:
                                SubscriptionHistory.query.filter_by(to_plan_id=dup_id).update({'to_plan_id': keep_id}, synchronize_session=False)
                            
                            # Commit all reassignments BEFORE deleting
                            if tenant_count > 0 or trans_count > 0 or trans_prev_count > 0 or hist_from_count > 0 or hist_to_count > 0:
                                db.session.commit()
                                print(f"[INFO] Committed all reassignments for duplicate plan {dup_id}")
                            
                            # Verify no references remain
                            final_tenant_check = Tenant.query.filter_by(plan_id=dup_id).count()
                            final_trans_check = SubscriptionTransaction.query.filter_by(plan_id=dup_id).count()
                            final_hist_check = SubscriptionHistory.query.filter_by(to_plan_id=dup_id).count()
                            
                            if final_tenant_check > 0 or final_trans_check > 0 or final_hist_check > 0:
                                print(f"[ERROR] Cannot delete plan {dup_id}: Still has references (tenants: {final_tenant_check}, transactions: {final_trans_check}, history: {final_hist_check})")
                                skipped_plans.append(f"{dup_plan.name} ({dup_plan.billing_cycle}) (ID: {dup_id}) - still has references")
                                continue
                            
                            # Now safe to delete the duplicate plan
                            print(f"[INFO] Deleting duplicate plan: {dup_plan.name} ({dup_plan.billing_cycle}) (ID: {dup_id})")
                            db.session.delete(dup_plan)
                            deleted_plans.append(f"{dup_plan.name} ({dup_plan.billing_cycle}) (ID: {dup_id}) - duplicate")
            
            # Commit any remaining duplicate plan deletions
            if duplicates_found:
                db.session.commit()
                if skipped_plans:
                    print(f"[WARNING] Could not delete {len(skipped_plans)} duplicate plan(s) due to remaining references")
                else:
                    print(f"[INFO] Cleaned up all duplicate plans.\n")
            
            # Final verification - ensure no yearly Free Trial exists
            yearly_free_trial = Plan.query.filter_by(name='Free Trial', billing_cycle='yearly').first()
            if yearly_free_trial:
                print(f"[WARNING] Found yearly Free Trial plan (ID: {yearly_free_trial.id}), deleting...")
                db.session.delete(yearly_free_trial)
                deleted_plans.append(f"Free Trial (yearly) (ID: {yearly_free_trial.id}) - invalid")
                db.session.commit()
            
            print(f"[INFO] Final verification complete. Total plans in database: {Plan.query.count()}")
            
            # Final summary
            print(f"\n[SUCCESS] Successfully updated pricing plans!")
            print(f"   - Deleted: {deleted_count} old plan(s)")
            print(f"   - Created: {len(created_plans)} new plan(s) (monthly + yearly)")
            if reassigned_count > 0:
                print(f"   - Reassigned: {reassigned_count} tenant(s) to new plans automatically")
            print(f"   - Final plan count: {Plan.query.count()} plan(s)")
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
                    'description': 'AI job-description generator, basic candidate search & matching, up to 10 job postings per month, standard email/SMS templates.',
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
            
            # Display plans grouped by name, showing both monthly and yearly
            for plan_name in new_plan_names:
                monthly_plan = Plan.query.filter_by(name=plan_name, billing_cycle='monthly').first()
                yearly_plan = Plan.query.filter_by(name=plan_name, billing_cycle='yearly').first()
                
                if monthly_plan:
                    price_dollars = monthly_plan.price_cents / 100
                    quota = "Unlimited" if monthly_plan.jd_quota_per_month == -1 else f"{monthly_plan.jd_quota_per_month} job postings"
                    subaccounts = "Unlimited seats" if monthly_plan.max_subaccounts == -1 else f"{monthly_plan.max_subaccounts} recruiter(s)"
                    
                    details = plan_details.get(monthly_plan.name, {})
                    
                    print(f"\n{'─'*80}")
                    print(f"PLAN: {monthly_plan.name}")
                    print(f"{'─'*80}")
                    
                    # Monthly plan info
                    print(f"  MONTHLY BILLING:")
                    print(f"    Price:            ${price_dollars:.2f} / month")
                    print(f"    Stripe Price ID:  {monthly_plan.stripe_price_id}")
                    print(f"    Plan ID:          {monthly_plan.id}")
                    
                    # Yearly plan info (if exists)
                    if yearly_plan:
                        yearly_price_dollars = yearly_plan.price_cents / 100
                        monthly_equivalent = yearly_price_dollars / 12
                        savings = (price_dollars * 12) - yearly_price_dollars
                        print(f"\n  YEARLY BILLING (20% discount):")
                        print(f"    Price:            ${yearly_price_dollars:.2f} / year (${monthly_equivalent:.2f} / month)")
                        print(f"    Savings:          ${savings:.2f} per year")
                        print(f"    Stripe Price ID:  {yearly_plan.stripe_price_id}")
                        print(f"    Plan ID:          {yearly_plan.id}")
                    
                    # Common plan features
                    print(f"\n  FEATURES:")
                    print(f"    Job Postings:      {quota} per month")
                    print(f"    Recruiters:        {subaccounts}")
                    
                    if details:
                        print(f"\n  Included Modules & Features:")
                        print(f"    {details.get('description', 'N/A')}")
                        print(f"\n  AI Capabilities:")
                        print(f"    {details.get('ai_capabilities', 'N/A')}")
                        print(f"\n  Notes:")
                        print(f"    {details.get('notes', 'N/A')}")
            
            print(f"\n{'─'*80}")
            print("\n[NOTE] Stripe price IDs configured:")
            print("\n  MONTHLY PLANS:")
            print("    - Free Trial: price_free_tier")
            print("    - Freelancer: price_1RrJkOKVj190YQJbUPk9z136")
            print("    - Starter: price_1RHnLzKVj190YQJbKhavdyiI")
            print("    - Growth: price_1RHnPkKVj190YQJbKtOCaZgj")
            print("    - Professional: price_1RA9hQQLZiir9RFCqgZKpVaV")
            print("    - Enterprise: price_1RjMMuKVj190YQJbBX2LhoTs")
            print("\n  YEARLY PLANS:")
            print("    - Configure separate price IDs in 'yearly_stripe_id_mapping' dictionary")
            print("    - Currently using same IDs as monthly (update mapping to use different IDs)")
            print("    - See lines 133-150 in this file to configure yearly price IDs")
            print("="*80)
            
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error updating plans: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    update_pricing_plans()

