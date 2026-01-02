# Database Migration Instructions

## New Fields and Tables

This implementation adds new fields to existing tables and creates a new table. You need to run database migrations.

### 1. UserBankAccount Table - New Fields

Add these columns to `user_bank_accounts` table:

```sql
ALTER TABLE user_bank_accounts 
ADD COLUMN ifsc_code VARCHAR(11) NULL,
ADD COLUMN account_type VARCHAR(20) NULL,
ADD COLUMN bank_branch VARCHAR(255) NULL,
ADD COLUMN bank_address TEXT NULL;
```

### 2. PayrollSettings Table - New Fields

Add these columns to `payroll_settings` table:

```sql
ALTER TABLE payroll_settings
ADD COLUMN payment_gateway VARCHAR(50) NULL,
ADD COLUMN razorpay_key_id VARCHAR(255) NULL,
ADD COLUMN razorpay_key_secret VARCHAR(255) NULL,
ADD COLUMN razorpay_fund_account_id VARCHAR(255) NULL,
ADD COLUMN payment_mode VARCHAR(20) NOT NULL DEFAULT 'NEFT';
```

### 3. PaymentTransaction Table - New Table

Create the new `payment_transactions` table:

```sql
CREATE TABLE payment_transactions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    pay_run_id INT NOT NULL,
    payslip_id INT NOT NULL,
    employee_id INT NOT NULL,
    tenant_id INT NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    currency VARCHAR(10) NOT NULL DEFAULT 'INR',
    payment_mode VARCHAR(20) NOT NULL,
    beneficiary_name VARCHAR(255) NOT NULL,
    account_number VARCHAR(50) NOT NULL,
    ifsc_code VARCHAR(11) NULL,
    bank_name VARCHAR(255) NULL,
    gateway VARCHAR(50) NULL,
    gateway_transaction_id VARCHAR(255) NULL,
    gateway_payout_id VARCHAR(255) NULL,
    gateway_response JSON NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    failure_reason TEXT NULL,
    initiated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at DATETIME NULL,
    completed_at DATETIME NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (pay_run_id) REFERENCES pay_runs(id),
    FOREIGN KEY (payslip_id) REFERENCES payslips(id),
    FOREIGN KEY (employee_id) REFERENCES users(id),
    FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    INDEX idx_pay_run_id (pay_run_id),
    INDEX idx_payslip_id (payslip_id),
    INDEX idx_employee_id (employee_id),
    INDEX idx_status (status),
    INDEX idx_gateway_transaction_id (gateway_transaction_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## Using Flask-Migrate

If you're using Flask-Migrate, create a migration:

```bash
cd backend
flask db migrate -m "Add Indian bank payment support"
flask db upgrade
```

## Manual Migration

If you prefer to run SQL directly:

1. Connect to your database
2. Run the SQL statements above in order
3. Verify the changes:

```sql
-- Check UserBankAccount fields
DESCRIBE user_bank_accounts;

-- Check PayrollSettings fields
DESCRIBE payroll_settings;

-- Check PaymentTransaction table
DESCRIBE payment_transactions;
```

## Rollback (if needed)

If you need to rollback:

```sql
-- Remove PaymentTransaction table
DROP TABLE IF EXISTS payment_transactions;

-- Remove PayrollSettings fields
ALTER TABLE payroll_settings
DROP COLUMN payment_gateway,
DROP COLUMN razorpay_key_id,
DROP COLUMN razorpay_key_secret,
DROP COLUMN razorpay_fund_account_id,
DROP COLUMN payment_mode;

-- Remove UserBankAccount fields
ALTER TABLE user_bank_accounts
DROP COLUMN ifsc_code,
DROP COLUMN account_type,
DROP COLUMN bank_branch,
DROP COLUMN bank_address;
```

