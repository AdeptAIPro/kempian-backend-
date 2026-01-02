# API Documentation - Production Payment System

## Base URL
`/api/hr`

---

## Fraud Alerts API

### List Fraud Alerts
**GET** `/fraud-alerts`

**Query Parameters:**
- `status` (optional): `pending`, `reviewed`, `approved`, `rejected`, or `open` (maps to pending)
- `severity` (optional): `low`, `medium`, `high`, `critical`
- `pay_run_id` (optional): Filter by pay run ID
- `date_from` (optional): ISO 8601 date string
- `date_to` (optional): ISO 8601 date string
- `limit` (optional): Number of results (default: 100, max: 1000)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "alerts": [
    {
      "id": 1,
      "pay_run_id": 123,
      "payment_transaction_id": 456,
      "employee_id": 789,
      "alert_type": "duplicate_account",
      "severity": "high",
      "risk_score": 75.5,
      "flags": [...],
      "status": "pending",
      "reviewed_by": null,
      "reviewed_at": null,
      "review_notes": null,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 10,
  "limit": 100,
  "offset": 0,
  "has_more": false
}
```

### Review Fraud Alert
**POST** `/fraud-alerts/{id}/review`

**Request Body:**
```json
{
  "decision": "approve" | "reject",
  "review_notes": "Detailed reason for decision (mandatory, min 20 chars)",
  "override": true | false
}
```

**Rules:**
- Only admin/owner can review
- `review_notes` is mandatory
- If rejected → payout is permanently blocked
- If approved → payout may resume
- Action is irreversible

**Response:**
```json
{
  "alert": {...},
  "message": "Fraud alert approved successfully",
  "irreversible": true,
  "audit_logged": true
}
```

### Get Fraud Alert
**GET** `/fraud-alerts/{id}`

**Response:**
```json
{
  "alert": {...}
}
```

---

## Bank Account Verification API

### Verify Bank Account
**POST** `/employees/{employee_id}/verify-bank-account`

**Request Body:**
```json
{
  "account_number": "1234567890",
  "ifsc_code": "HDFC0001234",
  "account_holder_name": "John Doe"
}
```

**Response (Success):**
```json
{
  "verified": true,
  "verification_id": "fa_xxxxx",
  "account_name_match": true,
  "verification_date": "2024-01-01T00:00:00Z",
  "message": "Bank account verified successfully via penny-drop",
  "cooldown_active": false
}
```

**Response (Failure):**
```json
{
  "verified": false,
  "error": "Human-readable error message",
  "technical_error": "Technical details (debug mode only)",
  "account_name_match": false,
  "verification_id": "fa_xxxxx",
  "cooldown_active": false
}
```

**Response (Cooldown Active):**
```json
{
  "verified": false,
  "error": "Bank account changed recently. 72-hour cooldown period active...",
  "cooldown_active": true,
  "cooldown_until": "2024-01-04T00:00:00Z",
  "hours_remaining": 24.5
}
```

---

## Pay Run Management API

### Force Resolve Payrun (Admin Only)
**POST** `/payruns/{id}/force-resolve`

**Request Body:**
```json
{
  "resolution": "force_complete" | "force_fail" | "mark_for_manual_payout",
  "reason": "Detailed reason (mandatory, min 20 chars)",
  "confirm_irreversible": true
}
```

**Rules:**
- Only admin/owner can use
- `reason` is mandatory (min 20 chars)
- `confirm_irreversible` must be true
- Action is irreversible
- Full audit trail required

**Response:**
```json
{
  "pay_run": {...},
  "message": "Pay run force-resolved to completed",
  "irreversible": true,
  "audit_logged": true,
  "resolution": "force_complete"
}
```

---

## Employee Payment Transparency API

### Get My Payments
**GET** `/employee-payments/my-payments`

**Query Parameters:**
- `limit` (optional): Number of results (default: 10, max: 50)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "payments": [
    {
      "id": 1,
      "pay_run_id": 123,
      "pay_period": {
        "start": "2024-01-01",
        "end": "2024-01-31"
      },
      "pay_date": "2024-02-01",
      "amount": 50000.00,
      "currency": "INR",
      "status": "success",
      "payment_mode": "NEFT",
      "initiated_at": "2024-02-01T10:00:00Z",
      "completed_at": "2024-02-01T10:30:00Z",
      "failure_reason": null,
      "reference_id": "pout****1234",
      "purpose_code": "SALARY"
    }
  ],
  "latest_payment": {
    "status": "success",
    "amount": 50000.00,
    "currency": "INR",
    "date": "2024-02-01T10:30:00Z",
    "reference_id": "pout****1234"
  },
  "total": 5,
  "limit": 10,
  "offset": 0,
  "has_more": false
}
```

### Get Payment Detail
**GET** `/employee-payments/my-payments/{transaction_id}`

**Response:**
```json
{
  "payment": {
    "id": 1,
    "pay_run_id": 123,
    "payslip_id": 456,
    "amount": 50000.00,
    "currency": "INR",
    "status": "success",
    "payment_mode": "NEFT",
    "purpose_code": "SALARY",
    "initiated_at": "2024-02-01T10:00:00Z",
    "completed_at": "2024-02-01T10:30:00Z",
    "failure_reason": null,
    "reference_id": "pout****1234",
    "pay_run": {...},
    "payslip": {...}
  }
}
```

---

## Authentication

All endpoints require JWT authentication via `Authorization: Bearer <token>` header.

---

## Error Responses

**400 Bad Request:**
```json
{
  "error": "Human-readable error message"
}
```

**401 Unauthorized:**
```json
{
  "error": "Authentication required"
}
```

**403 Forbidden:**
```json
{
  "error": "You do not have permission to perform this action"
}
```

**404 Not Found:**
```json
{
  "error": "Resource not found"
}
```

**500 Internal Server Error:**
```json
{
  "error": "An unexpected error occurred"
}
```

