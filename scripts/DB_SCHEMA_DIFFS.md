# Database Schema Diffs

## New Tables

### 1. employer_wallet_balances
```sql
CREATE TABLE employer_wallet_balances (
    id INT PRIMARY KEY AUTO_INCREMENT,
    tenant_id INT NOT NULL UNIQUE,
    available_balance DECIMAL(15, 2) NOT NULL DEFAULT 0,
    locked_balance DECIMAL(15, 2) NOT NULL DEFAULT 0,
    total_balance DECIMAL(15, 2) NOT NULL DEFAULT 0,
    razorpay_account_status VARCHAR(50),
    razorpay_account_id VARCHAR(255),
    kyc_status VARCHAR(50),
    last_synced_at DATETIME,
    sync_error TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    INDEX idx_tenant_id (tenant_id)
);
```

### 2. fraud_alerts
```sql
CREATE TABLE fraud_alerts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    tenant_id INT NOT NULL,
    pay_run_id INT,
    payment_transaction_id INT,
    employee_id INT NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    risk_score DECIMAL(5, 2) NOT NULL,
    flags JSON,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    reviewed_by INT,
    reviewed_at DATETIME,
    review_notes TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    FOREIGN KEY (pay_run_id) REFERENCES pay_runs(id),
    FOREIGN KEY (payment_transaction_id) REFERENCES payment_transactions(id),
    FOREIGN KEY (employee_id) REFERENCES users(id),
    FOREIGN KEY (reviewed_by) REFERENCES users(id),
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_pay_run_id (pay_run_id),
    INDEX idx_status (status),
    INDEX idx_employee_id (employee_id)
);
```

## Modified Tables

### 1. payment_transactions
```sql
ALTER TABLE payment_transactions
ADD COLUMN idempotency_key VARCHAR(255) UNIQUE,
ADD COLUMN retry_count INT NOT NULL DEFAULT 0,
ADD COLUMN max_retries INT NOT NULL DEFAULT 3,
ADD COLUMN last_retry_at DATETIME,
ADD COLUMN purpose_code VARCHAR(20) NOT NULL DEFAULT 'SALARY',
ADD COLUMN payout_category VARCHAR(20) NOT NULL DEFAULT 'salary',
ADD COLUMN fraud_risk_score DECIMAL(5, 2),
ADD COLUMN fraud_flags JSON,
ADD COLUMN requires_manual_review BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN reviewed_by INT,
ADD COLUMN reviewed_at DATETIME,
ADD COLUMN review_notes TEXT,
ADD INDEX idx_idempotency_key (idempotency_key),
ADD INDEX idx_fraud_review (requires_manual_review),
ADD FOREIGN KEY (reviewed_by) REFERENCES users(id);
```

### 2. pay_runs
```sql
ALTER TABLE pay_runs
MODIFY COLUMN status ENUM(
    'draft',
    'approval_pending',
    'funds_validated',
    'payout_initiated',
    'partially_completed',
    'completed',
    'failed',
    'reversed'
) NOT NULL DEFAULT 'draft',
ADD COLUMN funds_locked BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN funds_locked_at DATETIME,
ADD COLUMN funds_locked_amount DECIMAL(12, 2),
ADD COLUMN payments_initiated INT NOT NULL DEFAULT 0,
ADD COLUMN payments_successful INT NOT NULL DEFAULT 0,
ADD COLUMN payments_failed INT NOT NULL DEFAULT 0,
ADD COLUMN payments_pending INT NOT NULL DEFAULT 0,
ADD COLUMN funds_validated_by INT,
ADD COLUMN funds_validated_at DATETIME,
ADD COLUMN correlation_id VARCHAR(255),
ADD INDEX idx_correlation_id (correlation_id),
ADD FOREIGN KEY (funds_validated_by) REFERENCES users(id);
```

### 3. user_bank_accounts
```sql
ALTER TABLE user_bank_accounts
ADD COLUMN consent_given_at DATETIME,
ADD COLUMN consent_ip VARCHAR(45),
ADD COLUMN verified_by_penny_drop BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN verification_reference_id VARCHAR(255),
ADD COLUMN verification_date DATETIME,
ADD COLUMN last_updated_by INT,
ADD COLUMN bank_change_cooldown_until DATETIME,
ADD INDEX idx_verified (verified_by_penny_drop),
ADD INDEX idx_cooldown (bank_change_cooldown_until),
ADD FOREIGN KEY (last_updated_by) REFERENCES users(id);
```

## Indexes Added

- `idx_tenant_id` on `employer_wallet_balances`
- `idx_idempotency_key` on `payment_transactions`
- `idx_fraud_review` on `payment_transactions`
- `idx_correlation_id` on `pay_runs`
- `idx_verified` on `user_bank_accounts`
- `idx_cooldown` on `user_bank_accounts`
- `idx_status` on `fraud_alerts`

## Foreign Keys Added

- `employer_wallet_balances.tenant_id` → `tenants.id`
- `fraud_alerts.tenant_id` → `tenants.id`
- `fraud_alerts.pay_run_id` → `pay_runs.id`
- `fraud_alerts.payment_transaction_id` → `payment_transactions.id`
- `fraud_alerts.employee_id` → `users.id`
- `fraud_alerts.reviewed_by` → `users.id`
- `payment_transactions.reviewed_by` → `users.id`
- `pay_runs.funds_validated_by` → `users.id`
- `user_bank_accounts.last_updated_by` → `users.id`

