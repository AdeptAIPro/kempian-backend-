from app import create_app
from app.models import db, Plan
from app.config import Config

# Replace these with your actual Stripe Price IDs
PLANS = [
    {
        "name": "Free Trial",
        "price_cents": 0,
        "stripe_price_id": "price_free_trial",
        "jd_quota_per_month": 5,  # 5 searches per day for 7 days = 35 total
        "max_subaccounts": 1,
        "is_trial": True,
        "trial_days": 7,
    },
    {
        "name": "Recruiters plan",
        "price_cents": 2900,
        "stripe_price_id": "price_1RrJkOKVj190YQJbUPk9z136",
        "jd_quota_per_month": 50,
        "max_subaccounts": 1,
        "is_trial": False,
        "trial_days": 0,
    },
    {
        "name": "Recruiters plan (Yearly)",
        "price_cents": 30000,  # $290/year (about 2 months free)
        "stripe_price_id": "price_1RrJmPKVj190YQJb78dUzdLF",
        "jd_quota_per_month": 50,
        "max_subaccounts": 1,
        "is_trial": False,
        "trial_days": 0,
        "billing_cycle": "yearly",
    },
    {
        "name": "Basic plan",
        "price_cents": 4900,
        "stripe_price_id": "price_1RrJnhKVj190YQJb87TyY5EV",
        "jd_quota_per_month": 50,
        "max_subaccounts": 1,
        "is_trial": False,
        "trial_days": 0,
    },
    {
        "name": "Basic plan (Yearly)",
        "price_cents": 49000,  # $290/year (about 2 months free)
        "stripe_price_id": "price_1RrLbSKVj190YQJbh9kmeNQn",
        "jd_quota_per_month": 50,
        "max_subaccounts": 1,
        "is_trial": False,
        "trial_days": 0,
        "billing_cycle": "yearly",
    },
    {
        "name": "Growth",
        "price_cents": 29900,
        "stripe_price_id": "price_1RjMKNKVj190YQJbhBuhcjfQ",
        "jd_quota_per_month": 400,
        "max_subaccounts": 3,
        "is_trial": False,
        "trial_days": 0,
    },
    {
        "name": "Growth (Yearly)",
        "price_cents": 305000,  # $2900/year (about 2 months free)
        "stripe_price_id": "price_1RrLhfKVj190YQJbHkF6HzWQ",
        "jd_quota_per_month": 400,
        "max_subaccounts": 3,
        "is_trial": False,
        "trial_days": 0,
        "billing_cycle": "yearly",
    },
    {
        "name": "Professional",
        "price_cents": 59900,
        "stripe_price_id": "price_1RjMMuKVj190YQJbBX2LhoTs",
        "jd_quota_per_month": 2000,
        "max_subaccounts": 10,
        "is_trial": False,
        "trial_days": 0,
    },
    {
        "name": "Professional (Yearly)",
        "price_cents": 610000,  # $5990/year (about 2 months free)
        "stripe_price_id": "price_1RrLjiKVj190YQJbEyviX8kJ",
        "jd_quota_per_month": 2000,
        "max_subaccounts": 10,
        "is_trial": False,
        "trial_days": 0,
        "billing_cycle": "yearly",
    },
]

app = create_app()

with app.app_context():
    print("ACTUAL SQLALCHEMY_DATABASE_URI:", Config.SQLALCHEMY_DATABASE_URI)
    # Create all tables
    db.create_all()
    print("All tables created.")

    # Seed plans
    for plan_data in PLANS:
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
            print(f"Updated plan: {plan.name}")
        else:
            # Create new plan
            plan = Plan(**plan_data)
            db.session.add(plan)
            print(f"Added plan: {plan.name}")
    db.session.commit()
    print("Plans seeded/updated.") 