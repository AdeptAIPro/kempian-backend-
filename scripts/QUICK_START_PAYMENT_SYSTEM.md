# 🚀 QUICK START: PAYROLL PAYMENT SYSTEM

**Complete setup guide in 5 steps**

---

## ⚡ STEP 1: Environment Configuration (5 minutes)

```bash
# 1. Copy example file
cd backend
cp .env.example .env

# 2. Edit .env and fill in these REQUIRED values:
```

### Required Environment Variables:

```bash
# Database
DATABASE_URL=mysql://user:password@localhost:3306/kempian_db

# Security (generate strong keys)
SECRET_KEY=<generate-strong-32-char-key>
JWT_SECRET_KEY=${SECRET_KEY}

# Razorpay (get from https://dashboard.razorpay.com/app/keys)
RAZORPAY_KEY_ID=rzp_live_xxxxxxxxxxxxx
RAZORPAY_KEY_SECRET=xxxxxxxxxxxxxxxxxxxxxxxx
RAZORPAY_WEBHOOK_SECRET=<configure-in-razorpay-dashboard>

# Application
APP_URL=https://yourdomain.com
FRONTEND_URL=https://yourdomain.com
```

**Generate SECRET_KEY:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## ⚡ STEP 2: Database Setup (2 minutes)

```bash
# Run the table creation script
cd backend
python scripts/create_payment_tables_complete.py
```

**Expected Output:**
```
✅ Database connection successful
✅ Created table 'employer_wallet_balances'
✅ Created table 'fraud_alerts'
✅ Created table 'payment_transactions'
✅ Added columns to existing tables
✅ Database setup complete
```

---

## ⚡ STEP 3: Razorpay Configuration (10 minutes)

### 3.1 Get API Keys
1. Login to Razorpay Dashboard: https://dashboard.razorpay.com
2. Go to Settings → API Keys
3. Copy **Key ID** and **Key Secret** (use LIVE keys for production)
4. Add to `.env` file

### 3.2 Configure Webhook
1. Go to Settings → Webhooks
2. Click "Add New Webhook"
3. Set URL: `https://yourdomain.com/api/hr/payments/webhooks/razorpay`
4. Select Events:
   - `payout.processed`
   - `payout.failed`
   - `payout.reversed`
5. Copy **Webhook Secret** to `.env` as `RAZORPAY_WEBHOOK_SECRET`

### 3.3 Configure in App
1. Login to your app as admin
2. Go to Payroll Settings
3. Enter Razorpay Key ID and Secret
4. Save settings

---

## ⚡ STEP 4: Test Payment (15 minutes)

### 4.1 Pre-Test Checklist
- [ ] Database tables created
- [ ] Environment variables set
- [ ] Razorpay keys configured
- [ ] Webhook endpoint accessible
- [ ] At least one employee with verified bank account

### 4.2 Send Test Payment
1. Create a test payrun with ₹1 amount
2. Approve the payrun
3. Validate funds
4. Process payment
5. Monitor status

### 4.3 Verify
- [ ] Payment status updated to `success`
- [ ] Webhook received (check logs)
- [ ] Employee received ₹1 in bank account
- [ ] Reconciliation works (run manually)

---

## ⚡ STEP 5: Production Hardening (30 minutes)

### 5.1 Set Up Cron Jobs

```bash
# Add to crontab (crontab -e)
# Reconciliation (every 15 minutes)
*/15 * * * * cd /path/to/backend && python cron/reconcile_payments.py >> logs/cron.log 2>&1

# Balance Sync (every hour)
0 * * * * cd /path/to/backend && python cron/sync_wallet_balance.py >> logs/cron.log 2>&1

# Monitoring (every 5 minutes)
*/5 * * * * cd /path/to/backend && python scripts/monitor_payments.py >> logs/monitor.log 2>&1
```

### 5.2 Configure Alerts

Edit `.env`:
```bash
# Alert configuration
ALERT_EMAIL=admin@yourcompany.com
SLACK_WEBHOOK_URL=<optional>
PAYMENT_STUCK_ALERT_MINUTES=30
FRAUD_ALERT_RATE_THRESHOLD=10
```

### 5.3 Review Documentation
- [ ] Read `PAYROLL_INCIDENT_RUNBOOK.md`
- [ ] Review `LEGAL_PAYMENT_DISCLAIMERS.md`
- [ ] Check `FIRST_PAYROLL_SAFETY.md`
- [ ] Set up monitoring per `ALERTS_AND_MONITORING.md`

---

## ✅ VERIFICATION CHECKLIST

Before first real payroll:

### Technical:
- [ ] Database tables exist
- [ ] Environment variables configured
- [ ] Razorpay LIVE keys configured
- [ ] Webhook endpoint tested
- [ ] ₹1 test payment successful
- [ ] Webhook received and processed
- [ ] Reconciliation tested

### Operational:
- [ ] Team trained on runbook
- [ ] Alert emails configured
- [ ] Support contacts defined
- [ ] Legal disclaimers added to UI
- [ ] First payroll safety enabled

### Compliance:
- [ ] Terms of Service updated
- [ ] Privacy Policy updated
- [ ] Payment disclaimers added
- [ ] Employee consent obtained

---

## 🆘 TROUBLESHOOTING

### Database Connection Failed
```bash
# Check database is running
mysql -u user -p -e "SELECT 1"

# Verify DATABASE_URL in .env
# Format: mysql://user:password@host:port/database
```

### Razorpay API Errors
```bash
# Check API keys are correct
# Verify you're using LIVE keys (not test)
# Check Razorpay account is active
# Verify KYC is completed
```

### Webhook Not Received
```bash
# Check webhook URL is accessible
curl https://yourdomain.com/api/hr/payments/webhooks/razorpay

# Verify webhook secret matches Razorpay dashboard
# Check firewall/security group allows Razorpay IPs
# Review webhook logs
```

### Payment Stuck
```bash
# Run manual reconciliation
POST /api/hr/payruns/{id}/reconcile

# Check Razorpay dashboard
# Review PAYROLL_INCIDENT_RUNBOOK.md
```

---

## 📚 DOCUMENTATION INDEX

- **Setup**: This file (`QUICK_START_PAYMENT_SYSTEM.md`)
- **Environment**: `backend/.env.example`
- **Database**: `backend/scripts/create_payment_tables_complete.py`
- **Incidents**: `backend/PAYROLL_INCIDENT_RUNBOOK.md`
- **Legal**: `backend/LEGAL_PAYMENT_DISCLAIMERS.md`
- **Safety**: `backend/FIRST_PAYROLL_SAFETY.md`
- **Monitoring**: `backend/ALERTS_AND_MONITORING.md`
- **Gaps**: `backend/OPERATIONAL_GAPS_COMPLETION.md`

---

## 🎯 NEXT STEPS

1. **Complete Setup** (Steps 1-4 above)
2. **Run Test Payment** (₹1 to one employee)
3. **Review Runbook** (familiarize team)
4. **Set Up Alerts** (email/Slack)
5. **First Payroll** (use safety mechanisms)

---

**You're ready! 🚀**

