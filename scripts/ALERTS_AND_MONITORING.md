# 📊 ALERTS & MONITORING CONFIGURATION

**Purpose**: Proactive monitoring and alerting for payment system

---

## 🚨 CRITICAL ALERTS

### Alert 1: Payroll Stuck > X Minutes

**Trigger**: Payrun in `payout_initiated` for > 30 minutes

**Severity**: 🔴 **HIGH**

**Action**:
1. Check payment statuses
2. Run reconciliation
3. Check Razorpay dashboard
4. Escalate if > 2 hours

**Implementation**:
```python
# Check every 5 minutes
if payrun.status == 'payout_initiated':
    minutes_stuck = (now - payrun.updated_at).total_seconds() / 60
    if minutes_stuck > 30:
        send_alert('payroll_stuck', {
            'payrun_id': payrun.id,
            'minutes_stuck': minutes_stuck
        })
```

---

### Alert 2: Reconciliation Backlog

**Trigger**: > 20 payments stuck in processing

**Severity**: 🟡 **MEDIUM**

**Action**:
1. Check reconciliation cron status
2. Run manual reconciliation
3. Investigate webhook delivery

**Implementation**:
```python
stuck_count = PaymentTransaction.query.filter(
    PaymentTransaction.status.in_(['pending', 'processing']),
    PaymentTransaction.initiated_at < threshold_time
).count()

if stuck_count > 20:
    send_alert('reconciliation_backlog', {
        'stuck_count': stuck_count
    })
```

---

### Alert 3: High Fraud Alert Rate

**Trigger**: > 10 fraud alerts per hour

**Severity**: 🟡 **MEDIUM**

**Action**:
1. Review recent alerts
2. Check for pattern
3. Investigate if systemic issue

**Implementation**:
```python
recent_alerts = FraudAlert.query.filter(
    FraudAlert.created_at >= one_hour_ago
).count()

if recent_alerts > 10:
    send_alert('high_fraud_rate', {
        'alerts_per_hour': recent_alerts
    })
```

---

### Alert 4: Razorpay API Failure Rate

**Trigger**: > 10% API failures in last hour

**Severity**: 🔴 **HIGH**

**Action**:
1. Check Razorpay status page
2. Verify API keys
3. Check network connectivity
4. Escalate if persistent

**Implementation**:
```python
# Track API calls and failures
failure_rate = failed_calls / total_calls
if failure_rate > 0.10:
    send_alert('razorpay_api_failure', {
        'failure_rate': failure_rate,
        'failed_calls': failed_calls
    })
```

---

### Alert 5: Payment Success Rate Drop

**Trigger**: Success rate < 90% in last hour

**Severity**: 🔴 **HIGH**

**Action**:
1. Review failed payments
2. Check failure reasons
3. Identify pattern
4. Fix root cause

**Implementation**:
```python
success_rate = successful_payments / total_payments
if success_rate < 0.90:
    send_alert('low_success_rate', {
        'success_rate': success_rate,
        'failed_count': failed_payments
    })
```

---

### Alert 6: Low Wallet Balance

**Trigger**: Available balance < ₹10,000

**Severity**: 🟡 **MEDIUM**

**Action**:
1. Notify employer
2. Request fund addition
3. Block new payrolls if < required amount

**Implementation**:
```python
if wallet.available_balance < 10000:
    send_alert('low_balance', {
        'available_balance': wallet.available_balance
    })
```

---

### Alert 7: Critical Fraud Alert

**Trigger**: Fraud alert with `critical` severity

**Severity**: 🔴 **CRITICAL**

**Action**:
1. Immediate review required
2. Block payment if not reviewed
3. Escalate to management

**Implementation**:
```python
if alert.severity == 'critical' and alert.status == 'pending':
    send_alert('critical_fraud', {
        'alert_id': alert.id,
        'risk_score': alert.risk_score
    })
```

---

## 📧 ALERT DELIVERY

### Email Alerts:

```python
def send_alert(alert_type, details):
    recipient = os.getenv('ALERT_EMAIL', 'admin@yourcompany.com')
    subject = f"[PAYROLL ALERT] {alert_type}"
    body = format_alert_message(alert_type, details)
    send_email(recipient, subject, body)
```

### Slack Alerts (Optional):

```python
def send_slack_alert(alert_type, details):
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    if webhook_url:
        payload = {
            'text': f"🚨 {alert_type}",
            'attachments': [{'text': format_details(details)}]
        }
        requests.post(webhook_url, json=payload)
```

---

## 📊 METRICS DASHBOARD

### Key Metrics to Track:

1. **Payment Success Rate** (Target: > 95%)
2. **Average Processing Time** (Target: < 5 minutes)
3. **Stuck Payment Count** (Target: 0)
4. **Fraud Alert Rate** (Target: < 5/hour)
5. **Reconciliation Backlog** (Target: 0)
6. **Webhook Delivery Rate** (Target: > 99%)
7. **API Failure Rate** (Target: < 1%)

### Dashboard Implementation:

```python
def get_payment_metrics():
    return {
        'success_rate': calculate_success_rate(),
        'stuck_count': count_stuck_payments(),
        'fraud_rate': calculate_fraud_rate(),
        'reconciliation_backlog': count_stuck_for_reconciliation(),
        'api_failure_rate': calculate_api_failure_rate()
    }
```

---

## 🔄 MONITORING SCHEDULE

### Every 5 Minutes:
- Check for stuck payruns
- Check for critical fraud alerts

### Every 15 Minutes:
- Check reconciliation backlog
- Check API failure rate

### Every Hour:
- Calculate success rates
- Check fraud alert rate
- Check wallet balances

### Daily:
- Generate metrics report
- Review alert history
- Check system health

---

## 🛠️ IMPLEMENTATION SCRIPT

Create `backend/scripts/monitor_payments.py`:

```python
#!/usr/bin/env python3
"""Payment System Monitoring Script"""
from app import create_app, db
from app.models import PayRun, PaymentTransaction, FraudAlert
from datetime import datetime, timedelta
import os

def check_stuck_payruns():
    """Check for payruns stuck in payout_initiated"""
    threshold = datetime.utcnow() - timedelta(minutes=30)
    stuck = PayRun.query.filter(
        PayRun.status == 'payout_initiated',
        PayRun.updated_at < threshold
    ).all()
    
    for payrun in stuck:
        send_alert('payroll_stuck', {'payrun_id': payrun.id})

def check_reconciliation_backlog():
    """Check for payments needing reconciliation"""
    threshold = datetime.utcnow() - timedelta(hours=2)
    stuck = PaymentTransaction.query.filter(
        PaymentTransaction.status.in_(['pending', 'processing']),
        PaymentTransaction.initiated_at < threshold
    ).count()
    
    if stuck > 20:
        send_alert('reconciliation_backlog', {'count': stuck})

def check_fraud_alerts():
    """Check for high fraud alert rate"""
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    recent = FraudAlert.query.filter(
        FraudAlert.created_at >= one_hour_ago
    ).count()
    
    if recent > 10:
        send_alert('high_fraud_rate', {'count': recent})

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        check_stuck_payruns()
        check_reconciliation_backlog()
        check_fraud_alerts()
```

---

## 📋 ALERT CONFIGURATION

### Environment Variables:

```bash
# Alert thresholds
PAYMENT_STUCK_ALERT_MINUTES=30
RECONCILIATION_BACKLOG_THRESHOLD=20
FRAUD_ALERT_RATE_THRESHOLD=10
PAYMENT_SUCCESS_RATE_THRESHOLD=0.90
LOW_BALANCE_THRESHOLD=10000

# Alert delivery
ALERT_EMAIL=admin@yourcompany.com
SLACK_WEBHOOK_URL=
ENABLE_ALERTS=True
```

---

**Note**: Set up monitoring before first production payroll. Alerts prevent disasters.

