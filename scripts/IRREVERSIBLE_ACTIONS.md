# Irreversible Actions - Production Safety

## Critical Actions That Cannot Be Undone

### 1. Force Resolve Payrun
**Endpoint**: `POST /api/hr/payruns/{id}/force-resolve`

**Irreversible Because**:
- Changes payrun status permanently
- Updates payment transaction statuses
- Unlocks funds
- Full audit trail required

**Protection**:
- Admin-only access
- Mandatory reason (min 20 chars)
- `confirm_irreversible` flag required
- Full audit logging

### 2. Fraud Alert Rejection
**Endpoint**: `POST /api/hr/fraud-alerts/{id}/review` (decision: "reject")

**Irreversible Because**:
- Permanently blocks associated payment
- Marks transaction as failed
- Cannot be auto-retried

**Protection**:
- Admin-only access
- Mandatory review_notes
- Full audit logging

### 3. Funds Locking
**Action**: Locking funds for payrun

**Irreversible Because**:
- Atomic operation with SELECT FOR UPDATE
- Cannot be double-locked
- Prevents concurrent access

**Protection**:
- Row-level locking
- Transaction isolation
- Automatic rollback on failure

### 4. Payment Completion
**Action**: Payment marked as success

**Irreversible Because**:
- Money has been transferred
- Cannot be undone (only reversed)

**Protection**:
- Webhook verification
- Reconciliation checks
- Audit logging

### 5. Bank Account Verification
**Action**: Penny-drop verification

**Irreversible Because**:
- Verification status is permanent
- Cannot be "unverified" without new verification

**Protection**:
- 72-hour cooldown after change
- Full audit trail

## Audit Requirements

All irreversible actions MUST log:
- User ID
- Timestamp
- Reason/Notes
- Before/After state
- IP address (if available)

