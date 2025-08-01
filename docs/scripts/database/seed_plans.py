from app import create_app
from app.models import db, Plan

app = create_app()

with app.app_context():
    plans = [
        Plan(name='Starter', price_cents=0, stripe_price_id='price_free', jd_quota_per_month=10, max_subaccounts=1),
        Plan(name='Pro', price_cents=4900, stripe_price_id='price_pro', jd_quota_per_month=100, max_subaccounts=5),
        Plan(name='Business', price_cents=19900, stripe_price_id='price_business', jd_quota_per_month=1000, max_subaccounts=20),
    ]
    for plan in plans:
        if not Plan.query.filter_by(name=plan.name).first():
            db.session.add(plan)
    db.session.commit()
    print('Plans seeded.') 