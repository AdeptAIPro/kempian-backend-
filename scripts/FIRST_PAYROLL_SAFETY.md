# 🛡️ FIRST PAYROLL SAFETY MECHANISMS

**Purpose**: Prevent disasters on first payroll run

---

## 🔒 SAFETY SWITCHES

### 1. Per-Employee Amount Limit

**First Payroll Only**:
- Maximum amount per employee: ₹1,00,000 (configurable)
- If amount exceeds → Requires admin approval
- Bypass after first successful payroll

**Implementation**:
```python
# In payment_service.py
FIRST_PAYROLL_MAX_PER_EMPLOYEE = int(os.getenv('FIRST_PAYROLL_MAX_PER_EMPLOYEE', 100000))

def process_payment(...):
    # Check if first payroll
    if is_first_payroll(tenant_id):
        if amount > FIRST_PAYROLL_MAX_PER_EMPLOYEE:
            raise ValueError(f"First payroll limit exceeded. Max: ₹{FIRST_PAYROLL_MAX_PER_EMPLOYEE}")
```

---

### 2. Total Payroll Amount Limit

**First Payroll Only**:
- Maximum total payroll: ₹10,00,000 (configurable)
- If total exceeds → Requires admin approval
- Bypass after first successful payroll

**Implementation**:
```python
FIRST_PAYROLL_MAX_TOTAL = int(os.getenv('FIRST_PAYROLL_MAX_TOTAL', 1000000))

def validate_funds_for_payrun(...):
    if is_first_payroll(tenant_id):
        if total_net > FIRST_PAYROLL_MAX_TOTAL:
            return False, f"First payroll total limit exceeded. Max: ₹{FIRST_PAYROLL_MAX_TOTAL}"
```

---

### 3. Manual Confirmation Banner

**UI Implementation**:
- Show warning banner on first payroll
- Require admin to check "I understand this is the first payroll"
- Show summary of:
  - Number of employees
  - Total amount
  - Payment mode
  - Estimated settlement time

---

### 4. Test Payment Requirement

**Before First Real Payroll**:
- Require at least one ₹1 test payment
- Verify:
  - Webhook received
  - Status updated correctly
  - Employee received payment
  - Reconciliation works

---

## 📋 FIRST PAYROLL CHECKLIST

### Pre-Flight Checks:

- [ ] At least one test payment successful
- [ ] All bank accounts verified (penny-drop)
- [ ] No active cooldowns
- [ ] Wallet balance sufficient
- [ ] KYC status approved
- [ ] Razorpay account active
- [ ] Webhook endpoint tested
- [ ] Reconciliation tested
- [ ] Admin confirmation received

### During First Payroll:

- [ ] Monitor payment statuses closely
- [ ] Check webhook delivery
- [ ] Verify first few payments successful
- [ ] Watch for fraud alerts
- [ ] Monitor reconciliation

### Post-First Payroll:

- [ ] Verify all payments successful
- [ ] Confirm employees received money
- [ ] Review any fraud alerts
- [ ] Check audit logs
- [ ] Document any issues

---

## 🚨 FIRST PAYROLL DRY-RUN CHECKLIST

### Day Before:

1. **Test Payment**:
   - [ ] Send ₹1 test payment to one employee
   - [ ] Verify webhook received
   - [ ] Verify status updated
   - [ ] Verify employee received money
   - [ ] Test reconciliation

2. **Configuration Check**:
   - [ ] Razorpay keys configured (live mode)
   - [ ] Webhook secret configured
   - [ ] Webhook URL accessible
   - [ ] Payment mode set (NEFT recommended)

3. **Data Verification**:
   - [ ] All bank accounts verified
   - [ ] No test employees in payroll
   - [ ] Amounts verified
   - [ ] Employee count verified

### Day Of:

1. **Pre-Payroll**:
   - [ ] Sync wallet balance
   - [ ] Verify sufficient balance
   - [ ] Check KYC status
   - [ ] Review fraud alerts (should be none)

2. **During Payroll**:
   - [ ] Monitor payment initiation
   - [ ] Watch for immediate failures
   - [ ] Check first few statuses
   - [ ] Monitor webhook delivery

3. **Post-Payroll**:
   - [ ] Run reconciliation
   - [ ] Verify all statuses updated
   - [ ] Check for any failures
   - [ ] Confirm employees notified

---

## ⚙️ CONFIGURATION

### Environment Variables:

```bash
# First Payroll Safety Limits
FIRST_PAYROLL_MAX_PER_EMPLOYEE=100000
FIRST_PAYROLL_MAX_TOTAL=1000000

# Enable/Disable first payroll checks
ENABLE_FIRST_PAYROLL_SAFETY=True
```

### Database Flag:

Add to `tenants` table:
```sql
ALTER TABLE tenants 
ADD COLUMN first_payroll_completed BOOLEAN NOT NULL DEFAULT FALSE;
```

---

## 🔄 BYPASSING SAFETY (Admin Only)

**When to Bypass**:
- First payroll is small (< limits)
- All checks manually verified
- Emergency situation (with approval)

**How to Bypass**:
1. Admin sets `FIRST_PAYROLL_BYPASS=true` in .env
2. OR: Mark tenant as `first_payroll_completed=true` in DB
3. **Warning**: Only bypass if absolutely necessary

---

**Remember**: First payroll mistakes are expensive. Better safe than sorry.

