from app import create_app
from app.models import db, Plan
from app.config import Config

# Replace these with your actual Stripe Price IDs
PLANS = [
    # {
    #     "name": "Starter",
    #     "price_cents": 0,
    #     "stripe_price_id": "price_1STARTERID",
    #     "jd_quota_per_month": 50,
    #     "max_subaccounts": 1,
    # },
    {
        "name": "Pro",
        "price_cents": 29900,
        "stripe_price_id": "price_1RjMKNKVj190YQJbhBuhcjfQ",
        "jd_quota_per_month": 500,
        "max_subaccounts": 3,
    },
    {
        "name": "Business",
        "price_cents": 59900,
        "stripe_price_id": "price_1RjMMuKVj190YQJbBX2LhoTs",
        "jd_quota_per_month": 2500,
        "max_subaccounts": 10,
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
            print(f"Updated plan: {plan.name}")
        else:
            # Create new plan
            plan = Plan(**plan_data)
            db.session.add(plan)
            print(f"Added plan: {plan.name}")
    db.session.commit()
    print("Plans seeded/updated.") 