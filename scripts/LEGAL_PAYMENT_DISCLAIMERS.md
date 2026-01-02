# ⚖️ LEGAL & PAYMENT DISCLAIMERS

**For**: Terms of Service, Privacy Policy, UI Disclaimers

---

## 📄 PAYMENT DISCLAIMER (For UI)

### Short Version (Employee Dashboard):

```
Payments are processed through Razorpay, a licensed payment aggregator. 
Settlement typically occurs within 1-2 business days. 
For payment issues, please contact your HR department.
```

### Full Version (Terms of Service):

```
PAYMENT PROCESSING DISCLAIMER

1. Payment Method
   Payments are processed through Razorpay Payouts API, a licensed payment 
   aggregator authorized by the Reserve Bank of India (RBI). These are 
   Razorpay-mediated bank transfers, not direct bank-to-bank transfers.

2. Settlement Timeline
   - NEFT: Typically 1-2 business days
   - RTGS: Same day (if initiated before cutoff)
   - IMPS: Real-time (subject to bank availability)
   
   Settlement timing is controlled by Razorpay and the banking system, 
   not by [Your Company Name].

3. Liability
   [Your Company Name] is not liable for:
   - Delays caused by banking system
   - Gateway downtime or failures
   - Incorrect bank account details provided by employee
   - Bank rejections or holds
   - Weekend/holiday processing delays

4. Employee Responsibility
   Employees are responsible for:
   - Providing accurate bank account details
   - Verifying bank account before payroll
   - Notifying HR of bank account changes (72-hour notice required)
   - Verifying receipt of payment in bank statement

5. Dispute Resolution
   For payment disputes:
   - Contact HR within 7 days of expected payment date
   - Provide transaction reference ID
   - Allow 2-3 business days for investigation
   
   [Your Company Name] will work with Razorpay to resolve issues but 
   cannot guarantee immediate resolution.

6. Refund Policy
   In case of incorrect payment:
   - Employee must notify within 24 hours
   - Refund processing: 5-7 business days
   - Refund subject to bank processing times
```

---

## 🔒 PRIVACY & DATA PROTECTION

### Bank Account Data:

```
BANK ACCOUNT INFORMATION

1. Data Collection
   We collect bank account details (account number, IFSC, account holder name) 
   solely for processing salary payments.

2. Data Storage
   - Bank account numbers are stored encrypted
   - IFSC codes are stored securely
   - Data is masked in logs and UI

3. Data Sharing
   Bank account details are shared ONLY with:
   - Razorpay (payment processor)
   - Your bank (for payment processing)
   
   We do NOT share bank details with third parties for marketing or 
   any other purpose.

4. Data Retention
   Bank account details are retained as long as:
   - Employee is active
   - Required for tax/compliance purposes
   - Required for payment dispute resolution

5. Right to Deletion
   Employees can request deletion of bank account details after:
   - Employment termination
   - Final payment processed
   - All disputes resolved
```

---

## 📋 TERMS OF SERVICE ADDITIONS

### Section: Payment Processing

Add to your Terms of Service:

```markdown
## 8. PAYMENT PROCESSING

### 8.1 Payment Gateway
Payments are processed through Razorpay, a licensed payment aggregator. 
[Your Company] acts as an intermediary and does not directly process 
bank transfers.

### 8.2 Payment Authorization
By providing bank account details, you authorize:
- [Your Company] to initiate payments to your account
- Razorpay to process payments on our behalf
- Your bank to credit payments to your account

### 8.3 Payment Timing
- Payments are initiated on scheduled pay dates
- Actual credit to your account depends on:
  - Banking system processing
  - Payment mode (NEFT/RTGS/IMPS)
  - Weekend/holiday schedules
  - Bank processing times

### 8.4 Payment Failures
If payment fails:
- You will be notified via email/SMS
- Payment will be retried automatically (up to 3 times)
- If all retries fail, contact HR for manual processing

### 8.5 Disputes
Payment disputes must be reported within 7 days of expected payment date.
We will investigate and resolve within 2-3 business days.

### 8.6 Liability Limitations
[Your Company] is not liable for:
- Delays caused by banking system or payment gateway
- Incorrect bank details provided by employee
- Bank account freezes or holds
- Technical failures beyond our control

Maximum liability is limited to the disputed payment amount.
```

---

## 🎨 UI DISCLAIMER PLACEMENT

### Employee Payment History Page:

Add banner at top:

```html
<div class="payment-disclaimer-banner">
  <p>
    <strong>Payment Processing:</strong> Payments are processed through 
    Razorpay. Settlement may take 1-2 business days. For issues, 
    contact HR.
  </p>
</div>
```

### Bank Account Verification Page:

Add notice:

```html
<div class="verification-notice">
  <p>
    Bank account verification is mandatory for payroll payments. 
    A 72-hour cooldown applies after changing bank details. 
    Payments are processed through Razorpay.
  </p>
</div>
```

### Payroll Settings Page:

Add disclaimer:

```html
<div class="gateway-disclaimer">
  <p>
    <strong>Payment Gateway:</strong> We use Razorpay for payment processing. 
    This is a Razorpay-mediated transfer, not a direct bank-to-bank transfer. 
    Your bank account details are encrypted and shared only with Razorpay 
    for payment processing.
  </p>
</div>
```

---

## 📧 EMAIL TEMPLATES

### Payment Failed Notification:

```
Subject: Payment Failed - Action Required

Dear [Employee Name],

Your salary payment for [Pay Period] has failed.

Reason: [Failure Reason]

Next Steps:
1. Verify your bank account details are correct
2. Ensure bank account is active
3. Contact HR if issue persists

Reference ID: [Transaction ID]

For support, contact: hr@yourcompany.com

[Your Company Name]
```

### Payment Delayed Notification:

```
Subject: Payment Processing - Slight Delay

Dear [Employee Name],

Your salary payment for [Pay Period] is being processed.

Status: Processing
Expected Credit: [Date] (1-2 business days)

This delay may be due to:
- Banking system processing
- Weekend/holiday schedules
- Payment mode (NEFT typically takes 1-2 days)

If payment is not credited by [Date + 2 days], please contact HR.

Reference ID: [Transaction ID]

[Your Company Name]
```

---

## ✅ COMPLIANCE CHECKLIST

Before Launch:

- [ ] Terms of Service updated with payment disclaimer
- [ ] Privacy Policy updated with bank data handling
- [ ] UI disclaimers added to payment pages
- [ ] Email templates prepared
- [ ] Support contact information provided
- [ ] Refund policy documented
- [ ] Dispute resolution process defined

---

## 📞 SUPPORT INFORMATION

### For Employees:

- **Payment Issues**: hr@yourcompany.com
- **Bank Account Changes**: hr@yourcompany.com
- **Payment Disputes**: support@yourcompany.com

### For HR/Admin:

- **Razorpay Support**: support@razorpay.com
- **Technical Issues**: tech@yourcompany.com
- **Compliance Questions**: legal@yourcompany.com

---

**Note**: Consult with legal counsel before finalizing disclaimers for production use.

