# 🚨 PAYROLL INCIDENT RUNBOOK

**Version**: 1.0  
**Last Updated**: 2024  
**For**: Operations Team, HR Admins, Support Staff

---

## 📋 TABLE OF CONTENTS

1. [Common Incidents](#common-incidents)
2. [Step-by-Step Actions](#step-by-step-actions)
3. [When to Use Force-Resolve](#when-to-use-force-resolve)
4. [When NOT to Retry](#when-not-to-retry)
5. [Escalation Matrix](#escalation-matrix)
6. [Prevention Checklist](#prevention-checklist)

---

## 🔥 COMMON INCIDENTS

### Incident 1: Payroll Stuck in PAYOUT_INITIATED

**Symptoms**:
- Payrun status: `payout_initiated`
- Payments show `processing` for > 30 minutes
- No webhook received
- Employees haven't received money

**Severity**: 🔴 **HIGH** (if > 2 hours)

**Actions**:
1. Check payment transactions status
2. Run manual reconciliation
3. Check Razorpay dashboard
4. If still stuck > 2 hours → Use force-resolve

---

### Incident 2: Fraud Alert Fired at Critical Time

**Symptoms**:
- Fraud alert with `critical` severity
- Payroll blocked
- Multiple employees affected

**Severity**: 🔴 **CRITICAL** (if payroll day)

**Actions**:
1. Review fraud alert immediately
2. Check fraud flags and risk score
3. If false positive → Approve with notes
4. If genuine → Reject and block payment
5. Notify affected employees

---

### Incident 3: Employee Says Money Not Received

**Symptoms**:
- Employee reports no payment
- System shows `success`
- No bank statement entry

**Severity**: 🟡 **MEDIUM**

**Actions**:
1. Check payment transaction status
2. Verify gateway payout ID
3. Check Razorpay dashboard
4. Run reconciliation for that transaction
5. If confirmed failed → Retry payment
6. If confirmed success → Check bank account details

---

### Incident 4: Razorpay API Down / Gateway Failure

**Symptoms**:
- All API calls failing
- 500/503 errors from Razorpay
- Webhooks not arriving

**Severity**: 🔴 **HIGH** (if during payroll)

**Actions**:
1. Check Razorpay status page
2. Wait 15-30 minutes (usually temporary)
3. If persistent → Mark payrun for manual payout
4. Notify employees of delay
5. Retry when service restored

---

### Incident 5: Insufficient Balance Error

**Symptoms**:
- Fund validation fails
- Error: "Insufficient balance"
- Payroll cannot proceed

**Severity**: 🟡 **MEDIUM**

**Actions**:
1. Check wallet balance
2. Sync balance from Razorpay
3. If still insufficient → Employer must add funds
4. After funds added → Retry fund validation
5. Do NOT proceed without sufficient balance

---

### Incident 6: Partial Payouts (Some Success, Some Failed)

**Symptoms**:
- Payrun status: `partially_completed`
- Some employees paid, others not
- Mixed success/failure

**Severity**: 🟡 **MEDIUM**

**Actions**:
1. Review failed transactions
2. Check failure reasons
3. Fix issues (bank details, verification, etc.)
4. Retry failed payments
5. Update payrun status when all complete

---

## 📝 STEP-BY-STEP ACTIONS

### Action 1: Check Payment Status

**When**: Any payment issue

**Steps**:
1. Go to Pay Run Management
2. Click on affected payrun
3. View Payment Transactions
4. Check status of each transaction
5. Note gateway payout IDs

**API**: `GET /api/hr/payruns/{id}/payments`

---

### Action 2: Run Manual Reconciliation

**When**: Payments stuck in processing

**Steps**:
1. Go to Pay Run Management
2. Click "Reconcile" button
3. Wait for reconciliation to complete
4. Check updated statuses
5. Review any errors

**API**: `POST /api/hr/payruns/{id}/reconcile`

**Expected Result**: Statuses updated from Razorpay

---

### Action 3: Check Razorpay Dashboard

**When**: Need to verify gateway status

**Steps**:
1. Login to Razorpay Dashboard
2. Go to Payouts section
3. Search by payout ID (from transaction)
4. Check status and failure reason
5. Compare with internal status

**Note**: Razorpay status is source of truth

---

### Action 4: Retry Failed Payment

**When**: Payment failed, issue fixed

**Steps**:
1. Verify issue is fixed (bank details, verification, etc.)
2. Go to Payment Transactions
3. Click "Retry" on failed transaction
4. Monitor status
5. Verify success

**API**: `POST /api/hr/payruns/payments/{id}/retry`

**Rules**:
- Only retry if issue is fixed
- Check retry count (max 3)
- Do NOT retry if fraud alert exists

---

### Action 5: Review Fraud Alert

**When**: Fraud alert blocks payment

**Steps**:
1. Go to Fraud Alerts
2. Open alert details
3. Review fraud flags and risk score
4. Check employee history
5. Make decision: Approve or Reject
6. Add mandatory review notes (min 20 chars)
7. Submit decision

**API**: `POST /api/hr/fraud-alerts/{id}/review`

**Warning**: Rejecting permanently blocks payment

---

### Action 6: Force-Resolve Payrun

**When**: Payrun stuck, reconciliation failed, manual intervention needed

**Steps**:
1. **VERIFY**: Payrun stuck > 2 hours
2. **VERIFY**: Reconciliation attempted
3. **VERIFY**: Razorpay dashboard checked
4. Go to Pay Run Management
5. Click "Force Resolve"
6. Select resolution:
   - `force_complete`: If payments confirmed successful
   - `force_fail`: If payments confirmed failed
   - `mark_for_manual_payout`: If manual processing needed
7. Enter reason (min 20 chars)
8. Confirm irreversible action
9. Submit

**API**: `POST /api/hr/payruns/{id}/force-resolve`

**⚠️ CRITICAL**: This is irreversible. Use only as last resort.

---

## ✅ WHEN TO USE FORCE-RESOLVE

### ✅ USE FORCE-RESOLVE WHEN:

1. **Payrun stuck > 2 hours** AND
   - Reconciliation attempted
   - Razorpay dashboard checked
   - No webhook received
   - Manual verification confirms status

2. **Razorpay confirmed status** AND
   - Internal status mismatched
   - Reconciliation failed to update
   - Need to sync manually

3. **Manual payout required** AND
   - Gateway issue
   - Special circumstances
   - Need to process outside system

### ❌ DO NOT USE FORCE-RESOLVE WHEN:

1. **Payrun < 2 hours old**
   - Wait for webhook
   - Run reconciliation first

2. **Status unclear**
   - Check Razorpay dashboard first
   - Verify actual payment status

3. **Fraud alert pending**
   - Review fraud alert first
   - Do not bypass security

4. **Funds not validated**
   - Ensure funds are locked
   - Do not skip validation

---

## ❌ WHEN NOT TO RETRY

### ❌ DO NOT RETRY WHEN:

1. **Fraud Alert Exists**
   - Review alert first
   - Do not retry if rejected

2. **Bank Account Unverified**
   - Verify account first (penny-drop)
   - Do not retry unverified accounts

3. **Cooldown Active**
   - Wait for 72-hour cooldown
   - Do not retry during cooldown

4. **Max Retries Reached**
   - Check retry count
   - Manual intervention needed

5. **Insufficient Balance**
   - Add funds first
   - Do not retry without balance

6. **Invalid Bank Details**
   - Fix bank details first
   - Do not retry with wrong details

---

## 📞 ESCALATION MATRIX

### Level 1: Support Team (0-30 minutes)

**Handles**:
- Status checks
- Basic reconciliation
- Simple retries
- Employee queries

**Actions**:
- Check transaction status
- Run reconciliation
- Retry failed payments (if safe)

---

### Level 2: HR Admin (30 minutes - 2 hours)

**Handles**:
- Fraud alert reviews
- Complex payment issues
- Partial payouts
- Bank account verification

**Actions**:
- Review fraud alerts
- Approve/reject alerts
- Force-resolve (with approval)
- Verify bank accounts

---

### Level 3: Technical Team (2+ hours)

**Handles**:
- System issues
- API failures
- Database problems
- Integration issues

**Actions**:
- Investigate system errors
- Fix technical issues
- Database recovery
- API debugging

---

### Level 4: Management (Critical/Financial Impact)

**Handles**:
- Large payment failures
- Security incidents
- Compliance issues
- Legal matters

**Actions**:
- Executive decisions
- Legal consultation
- Compliance reporting
- Customer communication

---

## 🛡️ PREVENTION CHECKLIST

### Before Each Payroll:

- [ ] Wallet balance synced
- [ ] KYC status verified (approved)
- [ ] Razorpay account active
- [ ] All bank accounts verified (penny-drop)
- [ ] No active cooldowns
- [ ] No pending fraud alerts
- [ ] Webhook endpoint accessible
- [ ] Cron jobs running

### Daily Checks:

- [ ] Reconciliation cron running
- [ ] Balance sync cron running
- [ ] No stuck payruns (> 2 hours)
- [ ] No high fraud alert rate
- [ ] Payment success rate > 95%
- [ ] Webhook delivery normal

### Weekly Checks:

- [ ] Review fraud alerts
- [ ] Check payment success rates
- [ ] Verify reconciliation logs
- [ ] Review audit logs
- [ ] Test webhook endpoint

---

## 📊 METRICS TO MONITOR

### Critical Metrics:

1. **Payment Success Rate**
   - Target: > 95%
   - Alert if: < 90%

2. **Stuck Payment Count**
   - Target: 0
   - Alert if: > 5

3. **Fraud Alert Rate**
   - Target: < 5 per hour
   - Alert if: > 10 per hour

4. **Reconciliation Backlog**
   - Target: 0
   - Alert if: > 20

5. **Webhook Delivery Rate**
   - Target: > 99%
   - Alert if: < 95%

---

## 🔗 QUICK REFERENCE

### API Endpoints:

- **List Payments**: `GET /api/hr/payruns/{id}/payments`
- **Reconcile**: `POST /api/hr/payruns/{id}/reconcile`
- **Retry Payment**: `POST /api/hr/payruns/payments/{id}/retry`
- **Force-Resolve**: `POST /api/hr/payruns/{id}/force-resolve`
- **Review Fraud Alert**: `POST /api/hr/fraud-alerts/{id}/review`
- **Check Status**: `GET /api/hr/payruns/payments/{id}/status`

### Razorpay Dashboard:
- URL: https://dashboard.razorpay.com
- Section: Payouts → Transactions

### Support Contacts:
- **Razorpay Support**: support@razorpay.com
- **Internal Tech Team**: tech@yourcompany.com
- **HR Admin**: hr@yourcompany.com

---

## ⚠️ CRITICAL WARNINGS

1. **NEVER** skip fraud alert review
2. **NEVER** force-resolve without verification
3. **NEVER** retry without fixing root cause
4. **NEVER** bypass security checks
5. **ALWAYS** document actions in audit log

---

**Remember**: When in doubt, escalate. Better safe than sorry with real money.

