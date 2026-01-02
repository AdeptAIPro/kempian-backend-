from app.models import TenantAlert, db, Tenant, User
from app.simple_logger import get_logger
from app.emails.ses import send_quota_alert_email
from datetime import datetime

def reset_monthly_alerts():
    month_str = datetime.utcnow().strftime('%Y-%m')
    # Delete all quota_80 alerts for previous month
    TenantAlert.query.filter(TenantAlert.alert_type == 'quota_80').delete()
    db.session.commit()
    print('Monthly quota alerts reset.')
    # Optionally, send summary emails to all owners
    # for tenant in Tenant.query.all():
    #     owner = User.query.filter_by(tenant_id=tenant.id, role='owner').first()
    #     if owner:
    #         send_quota_alert_email(owner.email, 0)  # Or send a summary

if __name__ == '__main__':
    reset_monthly_alerts() 