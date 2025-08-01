from app import create_app
from app.models import db, Plan
from app.config import Config
import logging

logger = logging.getLogger(__name__)

def migrate_database():
    app = create_app()
    
    with app.app_context():
        print("Starting database migration...")
        
        # Add missing columns to plans table
        try:
            db.engine.execute("""
                ALTER TABLE plans 
                ADD COLUMN is_trial BOOLEAN DEFAULT FALSE,
                ADD COLUMN trial_days INTEGER DEFAULT 0,
                ADD COLUMN billing_cycle VARCHAR(20) DEFAULT 'monthly'
            """)
            print("✅ Added missing columns to plans table")
        except Exception as e:
            print(f"⚠️  Plans table columns may already exist: {e}")
        
        # Create user_trials table
        try:
            db.engine.execute("""
                CREATE TABLE user_trials (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL UNIQUE,
                    trial_start_date DATETIME NOT NULL,
                    trial_end_date DATETIME NOT NULL,
                    searches_used_today INT DEFAULT 0 NOT NULL,
                    last_search_date DATE,
                    is_active BOOLEAN DEFAULT TRUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            print("✅ Created user_trials table")
        except Exception as e:
            print(f"⚠️  user_trials table may already exist: {e}")
        
        # Update existing plans with new data
        plans_data = [
            {
                "name": "Free Trial",
                "price_cents": 0,
                "stripe_price_id": "price_free_trial",
                "jd_quota_per_month": 5,
                "max_subaccounts": 1,
                "is_trial": True,
                "trial_days": 7,
                "billing_cycle": "monthly"
            },
            {
                "name": "Recruiters plan",
                "price_cents": 2900,
                "stripe_price_id": "price_1RjMKNKVj190YQJbhBuhcjfQ",
                "jd_quota_per_month": 100,
                "max_subaccounts": 2,
                "is_trial": False,
                "trial_days": 0,
                "billing_cycle": "monthly"
            },
            {
                "name": "Recruiters plan (Yearly)",
                "price_cents": 29000,
                "stripe_price_id": "price_1RjMKNKVj190YQJbhBuhcjfQ_yearly",
                "jd_quota_per_month": 100,
                "max_subaccounts": 2,
                "is_trial": False,
                "trial_days": 0,
                "billing_cycle": "yearly"
            },
            {
                "name": "Pro",
                "price_cents": 290000,
                "stripe_price_id": "price_1RjMKNKVj190YQJbhBuhcjfQ",
                "jd_quota_per_month": 500,
                "max_subaccounts": 3,
                "is_trial": False,
                "trial_days": 0,
                "billing_cycle": "monthly"
            },
            {
                "name": "Pro (Yearly)",
                "price_cents": 2900000,
                "stripe_price_id": "price_1RjMKNKVj190YQJbhBuhcjfQ_pro_yearly",
                "jd_quota_per_month": 500,
                "max_subaccounts": 3,
                "is_trial": False,
                "trial_days": 0,
                "billing_cycle": "yearly"
            },
            {
                "name": "Business",
                "price_cents": 59900,
                "stripe_price_id": "price_1RjMMuKVj190YQJbBX2LhoTs",
                "jd_quota_per_month": 2500,
                "max_subaccounts": 10,
                "is_trial": False,
                "trial_days": 0,
                "billing_cycle": "monthly"
            },
            {
                "name": "Business (Yearly)",
                "price_cents": 599000,
                "stripe_price_id": "price_1RjMMuKVj190YQJbBX2LhoTs_yearly",
                "jd_quota_per_month": 2500,
                "max_subaccounts": 10,
                "is_trial": False,
                "trial_days": 0,
                "billing_cycle": "yearly"
            }
        ]
        
        for plan_data in plans_data:
            plan = Plan.query.filter_by(name=plan_data["name"]).first()
            if plan:
                # Update existing plan
                plan.price_cents = plan_data["price_cents"]
                plan.stripe_price_id = plan_data["stripe_price_id"]
                plan.jd_quota_per_month = plan_data["jd_quota_per_month"]
                plan.max_subaccounts = plan_data["max_subaccounts"]
                plan.is_trial = plan_data.get("is_trial", False)
                plan.trial_days = plan_data.get("trial_days", 0)
                plan.billing_cycle = plan_data.get("billing_cycle", "monthly")
                print(f"✅ Updated plan: {plan.name}")
            else:
                # Create new plan
                plan = Plan(**plan_data)
                db.session.add(plan)
                print(f"✅ Added plan: {plan.name}")
        
        db.session.commit()
        print("✅ Database migration completed successfully!")

if __name__ == "__main__":
    migrate_database() 