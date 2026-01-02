# Updated Database Migration Instructions

## New Models and Fields Added

This document includes ALL the production-grade fixes.

## 1. EmployerWalletBalance Table (NEW)

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## 2. FraudAlert Table (NEW)

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## 3. PaymentTransaction - New Fields

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

## 4. PayRun - New Fields

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

## 5. UserBankAccount - Consent Fields

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

## Complete Migration Script

```sql
-- 1. Create EmployerWalletBalance table
CREATE TABLE IF NOT EXISTS employer_wallet_balances (
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Create FraudAlert table
CREATE TABLE IF NOT EXISTS fraud_alerts (
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
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Update PaymentTransaction
ALTER TABLE payment_transactions
ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(255) UNIQUE,
ADD COLUMN IF NOT EXISTS retry_count INT NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS max_retries INT NOT NULL DEFAULT 3,
ADD COLUMN IF NOT EXISTS last_retry_at DATETIME,
ADD COLUMN IF NOT EXISTS purpose_code VARCHAR(20) NOT NULL DEFAULT 'SALARY',
ADD COLUMN IF NOT EXISTS payout_category VARCHAR(20) NOT NULL DEFAULT 'salary',
ADD COLUMN IF NOT EXISTS fraud_risk_score DECIMAL(5, 2),
ADD COLUMN IF NOT EXISTS fraud_flags JSON,
ADD COLUMN IF NOT EXISTS requires_manual_review BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS reviewed_by INT,
ADD COLUMN IF NOT EXISTS reviewed_at DATETIME,
ADD COLUMN IF NOT EXISTS review_notes TEXT;

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_idempotency_key ON payment_transactions(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_fraud_review ON payment_transactions(requires_manual_review);

-- 4. Update PayRun status enum and add fields
-- Note: You may need to drop and recreate the enum in MySQL
ALTER TABLE pay_runs
ADD COLUMN IF NOT EXISTS funds_locked BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS funds_locked_at DATETIME,
ADD COLUMN IF NOT EXISTS funds_locked_amount DECIMAL(12, 2),
ADD COLUMN IF NOT EXISTS payments_initiated INT NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS payments_successful INT NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS payments_failed INT NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS payments_pending INT NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS funds_validated_by INT,
ADD COLUMN IF NOT EXISTS funds_validated_at DATETIME,
ADD COLUMN IF NOT EXISTS correlation_id VARCHAR(255);

-- 5. Update UserBankAccount
ALTER TABLE user_bank_accounts
ADD COLUMN IF NOT EXISTS consent_given_at DATETIME,
ADD COLUMN IF NOT EXISTS consent_ip VARCHAR(45),
ADD COLUMN IF NOT EXISTS verified_by_penny_drop BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS verification_reference_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS verification_date DATETIME,
ADD COLUMN IF NOT EXISTS last_updated_by INT,
ADD COLUMN IF NOT EXISTS bank_change_cooldown_until DATETIME;
```

## Using Flask-Migrate

```bash
cd backend
flask db migrate -m "Add production-grade payment fixes: wallet balance, fraud detection, consent, idempotency"
flask db upgrade
```

## Verification

After migration, verify:

```sql
-- Check tables exist
SHOW TABLES LIKE '%wallet%';
SHOW TABLES LIKE '%fraud%';

-- Check PayRun status enum
SHOW COLUMNS FROM pay_runs LIKE 'status';

-- Check PaymentTransaction new fields
SHOW COLUMNS FROM payment_transactions;

-- Check UserBankAccount new fields
SHOW COLUMNS FROM user_bank_accounts;
```

