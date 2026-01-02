# Payroll System Environment Configuration

## Required Environment Variables

### Security (REQUIRED FOR PRODUCTION)

```bash
# Secret key for application - REQUIRED
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=your-secret-key-change-in-production

# Payment encryption salt - RECOMMENDED for production
# Generate with: python -c "import secrets; print(secrets.token_hex(16))"
PAYMENT_ENCRYPTION_SALT=your-salt-value
```

### Database (REQUIRED)

```bash
# PostgreSQL connection string
DATABASE_URL=postgresql://user:password@localhost:5432/kempian
```

### AWS Cognito (Required for employee authentication)

```bash
AWS_COGNITO_USER_POOL_ID=us-east-1_xxxxxxxxx
AWS_COGNITO_CLIENT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### AWS SES (Required for email notifications)

```bash
AWS_SES_REGION=us-east-1
SES_SENDER_EMAIL=noreply@yourdomain.com
```

### Application Settings

```bash
FLASK_ENV=development  # Set to 'production' in production
ENV=development
FRONTEND_URL=http://localhost:5173
JWT_SECRET_KEY=your-jwt-secret-key
```

## Razorpay Configuration

**Note:** Razorpay credentials are stored per-tenant in the database (encrypted).
Configure them via the **Payroll Settings UI**, not environment variables.

Required Razorpay setup:
1. Create Razorpay account at https://razorpay.com
2. Enable Payouts feature in dashboard
3. Complete KYC verification
4. Generate API keys in Settings > API Keys
5. Configure webhook endpoint: `https://your-domain.com/api/hr/payments/webhooks/razorpay`
6. Enable penny-drop verification for bank account verification

## Database Setup

Run migrations to create payroll tables:

```bash
cd backend
python migrations/create_all_payroll_tables.py
```

## Production Checklist

- [ ] Set `FLASK_ENV=production` and `ENV=production`
- [ ] Set strong, unique `SECRET_KEY`
- [ ] Set `PAYMENT_ENCRYPTION_SALT`
- [ ] Configure AWS Cognito credentials
- [ ] Configure AWS SES for email
- [ ] Run database migrations
- [ ] Configure Razorpay in Payroll Settings UI
- [ ] Enable HTTPS
- [ ] Configure CORS origins in `backend/app/__init__.py`

