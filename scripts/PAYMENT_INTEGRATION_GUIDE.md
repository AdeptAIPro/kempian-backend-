# Indian Bank-to-Bank Payment Integration Guide

## Overview

This implementation enables direct bank-to-bank transfers for payroll payments to Indian banks using the Razorpay Payouts API. The system supports NEFT, RTGS, IMPS, and UPI payment modes.

## Features

1. **Indian Bank Account Support**: Added IFSC code, account type, and branch details to employee bank accounts
2. **Razorpay Integration**: Full integration with Razorpay Payouts API for automated transfers
3. **Payment Tracking**: Complete transaction tracking with status updates
4. **Bulk Processing**: Process multiple payments in a single pay run
5. **Status Monitoring**: Real-time payment status checking

## Database Changes

### New Fields Added

1. **UserBankAccount Model**:
   - `ifsc_code` (String): Indian Financial System Code (11 characters)
   - `account_type` (String): 'savings', 'current', or 'salary'
   - `bank_branch` (String): Branch name
   - `bank_address` (Text): Branch address

2. **PayrollSettings Model**:
   - `payment_gateway` (String): 'razorpay', 'cashfree', 'paytm', or 'manual'
   - `razorpay_key_id` (String): Razorpay API key ID
   - `razorpay_key_secret` (String): Razorpay API secret (encrypted in production)
   - `razorpay_fund_account_id` (String): Company's fund account ID
   - `payment_mode` (String): 'NEFT', 'RTGS', 'IMPS', or 'UPI' (default: 'NEFT')

3. **New PaymentTransaction Model**:
   - Tracks individual bank transfer transactions
   - Stores gateway responses and status
   - Links to pay runs and payslips

## Setup Instructions

### 1. Razorpay Account Setup

1. Sign up for a Razorpay account at https://razorpay.com
2. Complete KYC verification
3. Enable Payouts feature in your dashboard
4. Get your API keys:
   - Key ID: Found in Settings > API Keys
   - Key Secret: Generated when you create API keys
   - Fund Account ID: Your company's bank account registered with Razorpay

### 2. Configure Payment Settings

Update payroll settings via API:

```bash
PUT /api/hr/payroll-settings
{
  "payment_gateway": "razorpay",
  "razorpay_key_id": "rzp_test_xxxxx",
  "razorpay_key_secret": "your_secret_key",
  "razorpay_fund_account_id": "fa_xxxxx",
  "payment_mode": "NEFT"
}
```

### 3. Add Employee Bank Details

Ensure employees have Indian bank account details:

```bash
PUT /api/hr/employees/{employee_id}/bank-account
{
  "account_holder_name": "John Doe",
  "account_number": "1234567890",
  "ifsc_code": "HDFC0001234",
  "bank_name": "HDFC Bank",
  "account_type": "savings",
  "bank_branch": "Mumbai Main Branch"
}
```

## API Endpoints

### Process Pay Run with Payments

```bash
POST /api/hr/payruns/{payrun_id}/process
{
  "status": "processing",
  "initiate_payments": true
}
```

This will:
1. Create Razorpay contacts for each employee
2. Create fund accounts for each bank account
3. Initiate payouts for all payslips
4. Track all transactions in PaymentTransaction table

### Get Payment Transactions

```bash
GET /api/hr/payruns/{payrun_id}/payments
```

Returns all payment transactions for a pay run.

### Check Payment Status

```bash
GET /api/hr/payruns/payments/{transaction_id}/status
```

Checks the current status with Razorpay and updates the transaction record.

## Payment Modes

- **NEFT**: National Electronic Funds Transfer (batch processing, 30 min - 2 hours)
- **RTGS**: Real-Time Gross Settlement (real-time, for amounts > ₹2 lakhs)
- **IMPS**: Immediate Payment Service (instant, 24/7)
- **UPI**: Unified Payments Interface (instant, 24/7)

## Payment Flow

1. **Pay Run Creation**: Create pay run with payslips
2. **Approval**: Approve the pay run
3. **Processing**: Call process endpoint with `initiate_payments: true`
4. **Payment Service**:
   - Creates/updates Razorpay contacts
   - Creates/updates fund accounts
   - Initiates payouts
   - Tracks transactions
5. **Status Updates**: Payments move through: pending → processing → success/failed
6. **Completion**: Mark pay run as completed after verification

## Error Handling

- Failed payments are tracked with failure reasons
- Transactions can be retried
- Gateway errors are logged for debugging
- Partial success is supported (some payments succeed, others fail)

## Security Considerations

1. **API Keys**: Store Razorpay keys securely (use environment variables or encrypted storage)
2. **Encryption**: Encrypt sensitive data in production
3. **Access Control**: Only authorized users can process payments
4. **Audit Trail**: All transactions are logged with timestamps

## Testing

### Test Mode

Use Razorpay test keys (rzp_test_*) for development:
- Test fund accounts are provided by Razorpay
- Test payouts are simulated
- No real money is transferred

### Production

1. Complete Razorpay KYC
2. Use live API keys (rzp_live_*)
3. Register your company's bank account
4. Test with small amounts first

## Monitoring

- Check payment status regularly
- Monitor failed transactions
- Review Razorpay dashboard for reconciliation
- Set up alerts for payment failures

## Support

- Razorpay Documentation: https://razorpay.com/docs/payouts/
- Razorpay Support: support@razorpay.com
- API Status: https://status.razorpay.com

## Future Enhancements

1. Support for other payment gateways (Cashfree, Paytm)
2. Automatic reconciliation
3. Payment retry mechanisms
4. Webhook integration for status updates
5. Bulk payout optimization
6. Payment scheduling

