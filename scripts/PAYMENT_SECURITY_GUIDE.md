# Payment Security and Privacy Guide

## Overview

This document outlines the comprehensive security and privacy measures implemented for bank-to-bank payment processing in compliance with RBI guidelines and industry best practices.

## Security Features

### 1. Data Encryption

#### At Rest Encryption
- **API Keys**: Razorpay API keys are encrypted using Fernet (symmetric encryption) before storage
- **Encryption Key**: Derived from `SECRET_KEY` environment variable using PBKDF2
- **Format**: Encrypted values are prefixed with `enc:` for identification

#### In Transit Encryption
- All API communications use HTTPS/TLS 1.2+
- Razorpay API endpoints use secure connections
- Webhook payloads are verified using HMAC signatures

### 2. Authentication & Authorization

#### Multi-Layer Access Control
- **User Authentication**: JWT-based authentication required for all payment operations
- **Role-Based Access**: Only admin, owner, employer, and recruiter roles can process payments
- **Tenant Isolation**: Payments are scoped to tenant/organization

#### API Key Management
- API keys stored encrypted in database
- Keys are decrypted only when needed for API calls
- Never logged or exposed in error messages

### 3. Webhook Security

#### Signature Verification
- All Razorpay webhooks are verified using HMAC SHA256
- Invalid signatures are rejected and logged as security events
- Constant-time comparison prevents timing attacks

#### Webhook Endpoint
- Protected endpoint: `/api/hr/payments/webhooks/razorpay`
- Requires valid signature in `X-Razorpay-Signature` header
- Logs all webhook attempts for audit

### 4. Fraud Detection

#### Amount Thresholds
- **Warning Threshold**: ₹1,00,000 (1 lakh)
- **Critical Threshold**: ₹5,00,000 (5 lakhs)
- Critical amounts require manual approval

#### Velocity Checks
- Monitors payment frequency per employee
- Flags suspicious patterns (e.g., >5 payments in short period)

#### Bank Details Validation
- IFSC code format validation (11 characters, alphanumeric)
- Account number validation (9-18 digits)
- Real-time validation before processing

### 5. Data Masking

#### Sensitive Data Protection
- **Account Numbers**: Masked in logs (shows only last 4 digits)
- **IFSC Codes**: Masked in logs (shows first 2 and last 2 characters)
- **API Keys**: Masked in logs (shows only first 8 characters)
- **Phone Numbers**: Masked in logs (shows only last 4 digits)

#### Response Sanitization
- Sensitive data never exposed in API responses
- Error messages don't leak sensitive information
- Audit logs contain masked data only

### 6. Audit Logging

#### Comprehensive Logging
- **Payment Initiation**: Logs who initiated, when, and for whom
- **Payment Success**: Logs transaction IDs and amounts
- **Payment Failures**: Logs failure reasons and gateway responses
- **Security Events**: Logs suspicious activities and invalid attempts
- **Retry Attempts**: Logs all payment retries

#### Log Format
```
PAYMENT_INITIATED: transaction_id=123, employee_id=456, amount=50000 INR, initiated_by=789
PAYMENT_SUCCESS: transaction_id=123, gateway_id=pout_xxx, amount=50000
PAYMENT_FAILURE: transaction_id=123, reason=Insufficient funds
SECURITY_EVENT: type=suspicious_amount, user_id=789, details={...}
```

### 7. Transaction Limits

#### Safety Limits
- **Minimum Amount**: ₹1 (configurable)
- **Maximum Amount**: ₹1,00,00,000 (1 crore) per transaction
- **Bulk Limit**: Maximum 100 payments per batch
- **Daily Limits**: Can be configured per tenant

### 8. Error Handling

#### Secure Error Messages
- Generic error messages for users
- Detailed errors logged server-side only
- No sensitive data in error responses
- Stack traces disabled in production

#### Error Recovery
- Failed transactions can be retried
- Automatic retry with exponential backoff (future enhancement)
- Manual intervention for critical failures

## Privacy Measures

### 1. Data Minimization
- Only collect necessary bank account information
- Store minimal data required for transactions
- Regular data cleanup of old transactions

### 2. Access Controls
- Employees can only view their own payment history
- Admins can view all payments within their organization
- Audit trail of all data access

### 3. Data Retention
- Transaction data retained as per regulatory requirements
- Configurable retention policies
- Secure deletion of expired data

### 4. Compliance
- **RBI Guidelines**: Compliance with Reserve Bank of India regulations
- **KYC/AML**: Support for Know Your Customer and Anti-Money Laundering
- **GDPR**: Data protection measures (if applicable)
- **PCI DSS**: Payment Card Industry compliance (for card payments)

## Security Best Practices

### 1. Environment Variables
```bash
# Required for encryption
SECRET_KEY=your-secret-key-here

# Optional: Custom encryption key
PAYMENT_ENCRYPTION_KEY=your-encryption-key-here
```

### 2. Production Checklist
- [ ] Use strong `SECRET_KEY` (minimum 32 characters)
- [ ] Enable HTTPS only (disable HTTP)
- [ ] Configure webhook secret in Razorpay dashboard
- [ ] Set up monitoring and alerts
- [ ] Regular security audits
- [ ] Backup encryption keys securely
- [ ] Enable two-factor authentication for admin accounts
- [ ] Regular dependency updates
- [ ] Penetration testing

### 3. Monitoring
- Monitor all payment transactions
- Alert on suspicious activities
- Track failed payment rates
- Monitor API response times
- Review audit logs regularly

### 4. Incident Response
- Document all security incidents
- Immediate revocation of compromised keys
- Notification procedures
- Recovery procedures

## API Security

### Request Security
- All requests require authentication
- Rate limiting on payment endpoints
- Input validation on all fields
- SQL injection prevention (parameterized queries)
- XSS prevention (input sanitization)

### Response Security
- No sensitive data in responses
- Proper HTTP status codes
- CORS configuration
- Security headers (X-Frame-Options, etc.)

## Webhook Security

### Configuration
1. Set webhook secret in Razorpay dashboard
2. Configure webhook URL: `https://yourdomain.com/api/hr/payments/webhooks/razorpay`
3. Enable webhook events: `payout.processed`, `payout.failed`, `payout.reversed`

### Verification Process
1. Receive webhook with signature
2. Extract signature from `X-Razorpay-Signature` header
3. Compute HMAC SHA256 of payload with secret
4. Compare signatures using constant-time comparison
5. Reject if signatures don't match

## Testing Security

### Test Scenarios
- Invalid webhook signatures
- Malformed requests
- Excessive amounts
- Invalid bank details
- Unauthorized access attempts
- SQL injection attempts
- XSS attempts

### Security Testing Tools
- OWASP ZAP for vulnerability scanning
- Burp Suite for penetration testing
- Custom security test suite

## Compliance Documentation

### Required Documentation
- Security policy
- Data protection policy
- Incident response plan
- Audit logs
- Compliance certificates

### Regular Reviews
- Quarterly security audits
- Annual penetration testing
- Regular compliance reviews
- Dependency vulnerability scans

## Support

For security concerns or incidents:
1. Report immediately to security team
2. Do not log sensitive information
3. Follow incident response procedures
4. Document all actions taken

## References

- [Razorpay Security Best Practices](https://razorpay.com/docs/security/)
- [RBI Digital Payment Guidelines](https://www.rbi.org.in/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [PCI DSS Requirements](https://www.pcisecuritystandards.org/)

