#!/usr/bin/env python3
"""
Quick database setup script - Creates all tables and optionally seeds initial data.

Usage:
    python scripts/setup_database.py
    python scripts/setup_database.py --seed-plans    # Also seed default plans
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import *  # noqa: F401, F403


def create_tables():
    """Create all database tables."""
    app = create_app()
    
    with app.app_context():
        try:
            print("Creating all database tables...")
            db.create_all()
            print("✅ All tables created successfully!")
            
            # Verify critical tables
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            critical = ['plans', 'tenants', 'subscription_transactions', 'users']
            missing = [t for t in critical if t not in tables]
            
            if missing:
                print(f"⚠️  Warning: Missing critical tables: {missing}")
                return False
            
            print(f"✅ Verified {len(critical)} critical tables exist")
            print(f"📊 Total tables: {len(tables)}")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def seed_default_plans():
    """Seed default subscription plans."""
    app = create_app()
    
    with app.app_context():
        from app.models import Plan
        
        try:
            # Check if plans already exist
            existing = Plan.query.count()
            if existing > 0:
                print(f"⚠️  {existing} plan(s) already exist. Skipping seed.")
                return True
            
            print("Seeding default plans...")
            
            default_plans = [
                {
                    'name': 'Free Trial',
                    'price_cents': 0,
                    'stripe_price_id': 'free_trial',
                    'jd_quota_per_month': 10,
                    'max_subaccounts': 1,
                    'is_trial': True,
                    'trial_days': 30,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Starter',
                    'price_cents': 4900,  # $49.00
                    'stripe_price_id': 'price_starter_monthly',  # Update with actual Stripe price ID
                    'jd_quota_per_month': 50,
                    'max_subaccounts': 3,
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Growth',
                    'price_cents': 9900,  # $99.00
                    'stripe_price_id': 'price_growth_monthly',  # Update with actual Stripe price ID
                    'jd_quota_per_month': 200,
                    'max_subaccounts': 10,
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                },
                {
                    'name': 'Professional',
                    'price_cents': 29900,  # $299.00
                    'stripe_price_id': 'price_professional_monthly',  # Update with actual Stripe price ID
                    'jd_quota_per_month': 1000,
                    'max_subaccounts': 50,
                    'is_trial': False,
                    'trial_days': 0,
                    'billing_cycle': 'monthly'
                }
            ]
            
            for plan_data in default_plans:
                plan = Plan(**plan_data)
                db.session.add(plan)
            
            db.session.commit()
            print(f"✅ Seeded {len(default_plans)} default plans")
            print("⚠️  Note: Update stripe_price_id values with actual Stripe price IDs")
            return True
            
        except Exception as e:
            print(f"❌ Error seeding plans: {e}")
            db.session.rollback()
            return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup database tables')
    parser.add_argument('--seed-plans', action='store_true', 
                       help='Also seed default subscription plans')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATABASE SETUP")
    print("=" * 60)
    
    if create_tables():
        if args.seed_plans:
            seed_default_plans()
        print("\n✨ Setup complete!")
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)

