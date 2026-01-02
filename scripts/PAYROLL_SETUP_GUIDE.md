# Payroll System Setup & Testing Guide

## Overview

This guide provides step-by-step instructions for setting up and testing the complete payroll system end-to-end.

---

## 1. Prerequisites

### Backend Requirements
- Python 3.9+
- PostgreSQL database
- AWS account with Cognito and SES configured
- Razorpay business account (for payment processing)

### Frontend Requirements
- Node.js 18+
- npm or yarn

---

## 2. Database Setup

### Run Migrations

```bash
cd backend
python migrations/create_all_payroll_tables.py
```

This creates the following tables:
- `employee_profiles` - Employee details
- `organization_metadata` - Organization info
- `timesheets` - Work hour tracking
- `payslips` - Generated payslips
- `tax_configurations` - Tax rules
- `employee_tax_profiles` - Employee W-4 info
- `deduction_types` - Deduction definitions
- `employee_deductions` - Employee deduction enrollments
- `pay_runs` - Pay run batches
- `pay_run_payslips` - Links payslips to pay runs
- `payroll_settings` - Tenant payroll configuration
- `holiday_calendars` - Holiday definitions
- `leave_types` - Leave type definitions
- `leave_balances` - Employee leave balances
- `leave_requests` - Leave requests
- `employer_wallet_balances` - Razorpay wallet tracking
- `fraud_alerts` - Fraud detection alerts
- `payment_transactions` - Payment records
- `user_bank_accounts` - Employee bank accounts

---

## 3. Environment Configuration

See `PAYROLL_ENV_CONFIG.md` for required environment variables.

### Minimum Required Variables:

```bash
SECRET_KEY=<strong-secret-key>
DATABASE_URL=postgresql://user:pass@localhost:5432/kempian
AWS_COGNITO_USER_POOL_ID=<cognito-pool-id>
AWS_COGNITO_CLIENT_ID=<cognito-client-id>
AWS_REGION=us-east-1
FRONTEND_URL=http://localhost:5173
```

---

## 4. Backend Startup

```bash
cd backend
pip install -r requirements.txt
python wsgi.py
```

Server runs on `http://localhost:8000`

---

## 5. Frontend Startup

```bash
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`

---

## 6. End-to-End Testing Checklist

### 6.1 Organization Setup
- [ ] Create an organization via `/dashboard/payroll`
- [ ] Verify organization appears in the list

### 6.2 Employee Management
- [ ] Add new employee via "Add Employee" form
- [ ] Verify Cognito user is created
- [ ] Verify invite email is sent
- [ ] Employee can log in with temporary password
- [ ] Employee can change password on first login

### 6.3 Payroll Settings
- [ ] Configure payroll settings (pay frequency, overtime rules)
- [ ] Add Razorpay API credentials
- [ ] Test Razorpay connection
- [ ] Validate fund account

### 6.4 Tax Configuration
- [ ] Add federal tax configuration
- [ ] Add state tax configuration (if applicable)
- [ ] Add FICA (Social Security & Medicare) configurations
- [ ] Create employee tax profiles

### 6.5 Deductions Setup
- [ ] Create deduction types (health insurance, 401k, etc.)
- [ ] Assign deductions to employees

### 6.6 Timesheet Management
- [ ] Employee submits timesheet
- [ ] Admin/employer approves timesheet
- [ ] Verify hours calculation (regular, overtime)

### 6.7 Payslip Generation
- [ ] Generate payslip for employee
- [ ] Verify calculations (gross, taxes, deductions, net)
- [ ] Send payslip via email
- [ ] Download/print payslip

### 6.8 Pay Run Processing
- [ ] Create new pay run
- [ ] Add payslips to pay run
- [ ] Approve pay run
- [ ] Validate funds (check wallet balance)
- [ ] Process payments
- [ ] Verify payment transactions

### 6.9 Bank Account Verification
- [ ] Add employee bank account
- [ ] Trigger penny-drop verification
- [ ] Verify 72-hour cooldown after changes

### 6.10 Fraud Detection
- [ ] Verify high-amount alerts are created
- [ ] Review and approve/reject fraud alerts

### 6.11 Compliance Reports
- [ ] Generate W-2 forms
- [ ] Generate 1099 forms (for contractors)
- [ ] Generate quarterly report
- [ ] Generate annual report
- [ ] Download CSV exports

### 6.12 Leave Management
- [ ] Create leave types
- [ ] Employee submits leave request
- [ ] Approve/reject leave request
- [ ] Verify balance deduction

---

## 7. API Endpoints Reference

### Employees
- `GET /api/hr/employees` - List employees
- `POST /api/hr/employees` - Create employee
- `PUT /api/hr/employees/:id` - Update employee
- `DELETE /api/hr/employees/:id` - Delete employee
- `POST /api/hr/employees/:id/invite` - Send invite
- `POST /api/hr/employees/:id/verify-bank-account` - Verify bank

### Timesheets
- `GET /api/hr/timesheets` - List timesheets
- `POST /api/hr/timesheets` - Create timesheet
- `PUT /api/hr/timesheets/:id` - Update timesheet
- `POST /api/hr/timesheets/:id/approve` - Approve
- `POST /api/hr/timesheets/:id/reject` - Reject
- `GET /api/hr/timesheets/summary` - Summary

### Payslips
- `POST /api/hr/payslips/generate/:employee_id` - Generate payslip
- `GET /api/hr/payslips/:id` - Get payslip
- `GET /api/hr/payslips/employee/:id/history` - History
- `POST /api/hr/payslips/:id/send-email` - Send email
- `POST /api/hr/payslips/bulk-generate` - Bulk generate

### Pay Runs
- `GET /api/hr/payruns` - List pay runs
- `POST /api/hr/payruns` - Create pay run
- `POST /api/hr/payruns/:id/approve` - Approve
- `POST /api/hr/payruns/:id/process` - Process
- `GET /api/hr/payruns/:id/payments` - Get payments
- `POST /api/hr/payruns/reconcile-payments` - Reconcile
- `POST /api/hr/payruns/:id/force-resolve` - Force resolve

### Payroll Settings
- `GET /api/hr/payroll-settings` - Get settings
- `PUT /api/hr/payroll-settings` - Update settings
- `POST /api/hr/payroll-settings/test-razorpay-connection` - Test
- `POST /api/hr/payroll-settings/validate-fund-account` - Validate

### Tax Management
- `GET /api/hr/tax/config` - List tax configs
- `POST /api/hr/tax/config` - Create config
- `PUT /api/hr/tax/config/:id` - Update config
- `GET /api/hr/tax/employee/:id/profile` - Get profile
- `POST /api/hr/tax/employee/:id/profile` - Create/update profile
- `POST /api/hr/tax/calculate` - Calculate tax

### Deductions
- `GET /api/hr/deductions/types` - List types
- `POST /api/hr/deductions/types` - Create type
- `GET /api/hr/deductions/employee/:id` - Get deductions
- `POST /api/hr/deductions/employee/:id` - Add deduction
- `DELETE /api/hr/deductions/employee/:id/:ded_id` - Remove
- `POST /api/hr/deductions/calculate` - Calculate

### Leave Management
- `GET /api/hr/leave/types` - List leave types
- `POST /api/hr/leave/types` - Create type
- `GET /api/hr/leave/employee/:id/balance` - Get balance
- `GET /api/hr/leave/employee/:id/requests` - Get requests
- `POST /api/hr/leave/employee/:id/requests` - Create request
- `POST /api/hr/leave/requests/:id/approve` - Approve
- `POST /api/hr/leave/requests/:id/reject` - Reject

### Compliance
- `GET /api/hr/compliance/w2/:year` - W-2 forms
- `GET /api/hr/compliance/1099/:year` - 1099 forms
- `GET /api/hr/compliance/quarterly` - Quarterly report
- `GET /api/hr/compliance/annual` - Annual report

### Fraud Alerts
- `GET /api/hr/fraud-alerts` - List alerts
- `POST /api/hr/fraud-alerts/:id/review` - Review alert

### Employee Payments (Employee View)
- `GET /api/hr/employee-payments/my-payments` - My payments
- `GET /api/hr/employee-payments/my-payments/:id` - Payment details

### Webhooks
- `POST /api/hr/payments/webhooks/razorpay` - Razorpay webhook

---

## 8. Troubleshooting

### Common Issues

**"Authentication required" error**
- Ensure JWT token is included in Authorization header
- Check Cognito configuration

**"User not found" error**
- User may not exist in database
- Check if Cognito user is synced with database

**Razorpay connection fails**
- Verify API key and secret
- Ensure Razorpay Payouts is enabled
- Check KYC status

**Email not sent**
- Verify SES configuration
- Check sender email is verified
- Check SES sandbox restrictions

**Payment fails**
- Check wallet balance
- Verify bank account details
- Check fraud alerts
- Verify 72-hour cooldown for bank changes

---

## 9. Security Best Practices

1. **Never expose API keys** in frontend code
2. **Use HTTPS** in production
3. **Set strong SECRET_KEY** and encryption salt
4. **Enable rate limiting** for payment endpoints
5. **Monitor fraud alerts** regularly
6. **Audit payment transactions** periodically
7. **Keep Razorpay KYC up to date**

---

## 10. Support

For issues or questions:
1. Check the logs in `backend/logs/`
2. Review API responses for error messages
3. Check Razorpay dashboard for payment status
4. Verify AWS CloudWatch for Cognito/SES issues

