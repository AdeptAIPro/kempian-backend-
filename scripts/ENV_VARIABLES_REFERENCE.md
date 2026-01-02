# 📋 ENVIRONMENT VARIABLES REFERENCE

**Complete list of all environment variables for the payment system**

---

## 🔴 REQUIRED (Must Set)

### Database
```bash
DATABASE_URL=mysql://user:password@host:3306/database
# OR
SQLALCHEMY_DATABASE_URI=mysql://user:password@host:3306/database
```

### Security
```bash
SECRET_KEY=<32+ character random string>
# Generate: python -c "import secrets; print(secrets.token_hex(32))"
```

### Razorpay (Payment Gateway)
```bash
RAZORPAY_KEY_ID=rzp_live_xxxxxxxxxxxxx
RAZORPAY_KEY_SECRET=xxxxxxxxxxxxxxxxxxxxxxxx
RAZORPAY_WEBHOOK_SECRET=<from-razorpay-dashboard>
```

### Stripe (Subscription Payments)
```bash
# Stripe Secret Key (for API calls)
STRIPE_SECRET_KEY=sk_live_xxxxxxxxxxxxx  # Production
# OR for testing: sk_test_xxxxxxxxxxxxx

# Stripe Webhook Secret (for webhook signature verification)
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxx  # Get from Stripe Dashboard → Webhooks
```

---

## 🟡 RECOMMENDED (Should Set)

### Application URLs
```bash
APP_URL=https://yourdomain.com
FRONTEND_URL=https://yourdomain.com
```

### JWT Configuration
```bash
JWT_SECRET_KEY=${SECRET_KEY}
# OR for RS256:
JWT_PUBLIC_KEY=<public-key-pem>
```

### Payment Encryption
```bash
PAYMENT_ENCRYPTION_KEY=<fernet-key>
# Generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# If not set, will derive from SECRET_KEY (less secure)
```

---

## 🟢 OPTIONAL (Nice to Have)

### Payment Configuration
```bash
# Default payment mode
PAYMENT_MODE_DEFAULT=NEFT

# First payroll safety limits
FIRST_PAYROLL_MAX_PER_EMPLOYEE=100000
FIRST_PAYROLL_MAX_TOTAL=1000000
```

### Reconciliation
```bash
# Hours threshold for stuck payments
RECONCILIATION_HOURS_THRESHOLD=2
```

### Cron Jobs
```bash
# Enable/disable cron jobs
ENABLE_RECONCILIATION_CRON=True
ENABLE_BALANCE_SYNC_CRON=True

# Cron schedules (cron format)
RECONCILIATION_CRON_SCHEDULE=*/15 * * * *
BALANCE_SYNC_CRON_SCHEDULE=0 * * * *
```

### Alerting
```bash
# Email for alerts
ALERT_EMAIL=admin@yourcompany.com

# Slack webhook (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Alert thresholds
PAYMENT_STUCK_ALERT_MINUTES=30
FRAUD_ALERT_RATE_THRESHOLD=10
PAYMENT_SUCCESS_RATE_THRESHOLD=0.90
LOW_BALANCE_THRESHOLD=10000
```

### Logging
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=logs/kempian_app.log
```

### Legal/Compliance
```bash
# Payment disclaimer text
PAYMENT_DISCLAIMER=Payments are processed through Razorpay. Settlement may take 1-2 business days.

# Terms URLs
TERMS_OF_SERVICE_URL=https://yourcompany.com/terms
PRIVACY_POLICY_URL=https://yourcompany.com/privacy
```

---

## 🔵 DEVELOPMENT ONLY

### Flask Configuration
```bash
FLASK_ENV=development
DEBUG=True  # NEVER use in production
```

### AWS (if using)
```bash
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
```

### Redis (if using)
```bash
REDIS_URL=redis://localhost:6379/0
```

### Email (for notifications)
```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM_EMAIL=noreply@yourcompany.com
```

---

## 📝 COMPLETE .env TEMPLATE

See `backend/.env.example` for the complete template with all variables and descriptions.

---

## 🔐 SECURITY NOTES

1. **NEVER commit `.env` file** to version control
2. **Use strong random keys** for all SECRET_* variables
3. **Rotate keys periodically** (every 90 days recommended)
4. **Use different keys** for development/staging/production
5. **Encrypt sensitive values** before storing in database
6. **Restrict file permissions**: `chmod 600 .env`

---

## ✅ VERIFICATION

Check your environment is configured correctly:

```bash
# Test database connection
python -c "from app import create_app, db; app = create_app(); app.app_context().push(); db.session.execute('SELECT 1')"

# Verify Razorpay keys (will fail if invalid)
python -c "import requests; requests.get('https://api.razorpay.com/v1/account', auth=('YOUR_KEY_ID', 'YOUR_KEY_SECRET'))"
```

---

## 🆘 TROUBLESHOOTING

### "SECRET_KEY not set"
- Set `SECRET_KEY` in `.env`
- Generate: `python -c "import secrets; print(secrets.token_hex(32))"`

### "Database connection failed"
- Check `DATABASE_URL` format
- Verify database is running
- Check credentials

### "Razorpay authentication failed"
- Verify `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET`
- Check you're using LIVE keys (not test)
- Verify account is active

### "Webhook signature verification failed"
- Check `RAZORPAY_WEBHOOK_SECRET` matches Razorpay dashboard
- Verify webhook URL is correct
- Check webhook events are selected

### "Stripe webhook signature verification failed"
- Check `STRIPE_WEBHOOK_SECRET` matches Stripe dashboard
- Verify webhook URL is correct: `https://yourdomain.com/webhook/stripe`
- Check webhook events are selected in Stripe Dashboard
- Ensure using correct secret for environment (test vs live)

---

**For complete setup instructions, see `QUICK_START_PAYMENT_SYSTEM.md`**

