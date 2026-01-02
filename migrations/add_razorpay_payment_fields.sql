-- SQL Migration Script: Add Razorpay Payment Fields
-- This script adds new fields for Razorpay payment integration
-- 
-- Usage:
--   For PostgreSQL:
--     psql -U your_user -d your_database -f add_razorpay_payment_fields.sql
--   
--   For MySQL:
--     mysql -u your_user -p your_database < add_razorpay_payment_fields.sql
--
--   For SQLite:
--     sqlite3 your_database.db < add_razorpay_payment_fields.sql

-- ============================================================================
-- Add fields to payroll_settings table
-- ============================================================================

-- Add razorpay_webhook_secret column (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'payroll_settings' 
        AND column_name = 'razorpay_webhook_secret'
    ) THEN
        ALTER TABLE payroll_settings 
        ADD COLUMN razorpay_webhook_secret VARCHAR(255) NULL;
        RAISE NOTICE 'Added column razorpay_webhook_secret to payroll_settings';
    ELSE
        RAISE NOTICE 'Column razorpay_webhook_secret already exists in payroll_settings';
    END IF;
END $$;

-- Add razorpay_fund_account_validated column (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'payroll_settings' 
        AND column_name = 'razorpay_fund_account_validated'
    ) THEN
        ALTER TABLE payroll_settings 
        ADD COLUMN razorpay_fund_account_validated BOOLEAN NOT NULL DEFAULT FALSE;
        RAISE NOTICE 'Added column razorpay_fund_account_validated to payroll_settings';
    ELSE
        RAISE NOTICE 'Column razorpay_fund_account_validated already exists in payroll_settings';
    END IF;
END $$;

-- Add razorpay_fund_account_validated_at column (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'payroll_settings' 
        AND column_name = 'razorpay_fund_account_validated_at'
    ) THEN
        ALTER TABLE payroll_settings 
        ADD COLUMN razorpay_fund_account_validated_at TIMESTAMP NULL;
        RAISE NOTICE 'Added column razorpay_fund_account_validated_at to payroll_settings';
    ELSE
        RAISE NOTICE 'Column razorpay_fund_account_validated_at already exists in payroll_settings';
    END IF;
END $$;

-- ============================================================================
-- Add fields to user_bank_accounts table
-- ============================================================================

-- Add razorpay_contact_id column (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'user_bank_accounts' 
        AND column_name = 'razorpay_contact_id'
    ) THEN
        ALTER TABLE user_bank_accounts 
        ADD COLUMN razorpay_contact_id VARCHAR(255) NULL;
        RAISE NOTICE 'Added column razorpay_contact_id to user_bank_accounts';
    ELSE
        RAISE NOTICE 'Column razorpay_contact_id already exists in user_bank_accounts';
    END IF;
END $$;

-- Add razorpay_fund_account_id column (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'user_bank_accounts' 
        AND column_name = 'razorpay_fund_account_id'
    ) THEN
        ALTER TABLE user_bank_accounts 
        ADD COLUMN razorpay_fund_account_id VARCHAR(255) NULL;
        RAISE NOTICE 'Added column razorpay_fund_account_id to user_bank_accounts';
    ELSE
        RAISE NOTICE 'Column razorpay_fund_account_id already exists in user_bank_accounts';
    END IF;
END $$;

-- Add razorpay_contact_created_at column (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'user_bank_accounts' 
        AND column_name = 'razorpay_contact_created_at'
    ) THEN
        ALTER TABLE user_bank_accounts 
        ADD COLUMN razorpay_contact_created_at TIMESTAMP NULL;
        RAISE NOTICE 'Added column razorpay_contact_created_at to user_bank_accounts';
    ELSE
        RAISE NOTICE 'Column razorpay_contact_created_at already exists in user_bank_accounts';
    END IF;
END $$;

-- Add razorpay_fund_account_created_at column (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'user_bank_accounts' 
        AND column_name = 'razorpay_fund_account_created_at'
    ) THEN
        ALTER TABLE user_bank_accounts 
        ADD COLUMN razorpay_fund_account_created_at TIMESTAMP NULL;
        RAISE NOTICE 'Added column razorpay_fund_account_created_at to user_bank_accounts';
    ELSE
        RAISE NOTICE 'Column razorpay_fund_account_created_at already exists in user_bank_accounts';
    END IF;
END $$;

-- ============================================================================
-- Verification Query
-- ============================================================================

-- Verify payroll_settings columns
SELECT 
    'payroll_settings' as table_name,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'payroll_settings'
AND column_name IN (
    'razorpay_webhook_secret',
    'razorpay_fund_account_validated',
    'razorpay_fund_account_validated_at'
)
ORDER BY column_name;

-- Verify user_bank_accounts columns
SELECT 
    'user_bank_accounts' as table_name,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'user_bank_accounts'
AND column_name IN (
    'razorpay_contact_id',
    'razorpay_fund_account_id',
    'razorpay_contact_created_at',
    'razorpay_fund_account_created_at'
)
ORDER BY column_name;

-- ============================================================================
-- MySQL Alternative (if PostgreSQL DO blocks don't work)
-- ============================================================================

-- For MySQL, use this instead:
/*
-- Add fields to payroll_settings table
ALTER TABLE payroll_settings 
ADD COLUMN IF NOT EXISTS razorpay_webhook_secret VARCHAR(255) NULL,
ADD COLUMN IF NOT EXISTS razorpay_fund_account_validated BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS razorpay_fund_account_validated_at DATETIME NULL;

-- Add fields to user_bank_accounts table
ALTER TABLE user_bank_accounts 
ADD COLUMN IF NOT EXISTS razorpay_contact_id VARCHAR(255) NULL,
ADD COLUMN IF NOT EXISTS razorpay_fund_account_id VARCHAR(255) NULL,
ADD COLUMN IF NOT EXISTS razorpay_contact_created_at DATETIME NULL,
ADD COLUMN IF NOT EXISTS razorpay_fund_account_created_at DATETIME NULL;
*/

-- ============================================================================
-- SQLite Alternative (SQLite doesn't support IF NOT EXISTS in ALTER TABLE)
-- ============================================================================

-- For SQLite, check columns first, then add:
/*
-- Check and add razorpay_webhook_secret
-- (Run this check in application code or manually verify)

ALTER TABLE payroll_settings 
ADD COLUMN razorpay_webhook_secret TEXT;

ALTER TABLE payroll_settings 
ADD COLUMN razorpay_fund_account_validated INTEGER NOT NULL DEFAULT 0;

ALTER TABLE payroll_settings 
ADD COLUMN razorpay_fund_account_validated_at TEXT;

ALTER TABLE user_bank_accounts 
ADD COLUMN razorpay_contact_id TEXT;

ALTER TABLE user_bank_accounts 
ADD COLUMN razorpay_fund_account_id TEXT;

ALTER TABLE user_bank_accounts 
ADD COLUMN razorpay_contact_created_at TEXT;

ALTER TABLE user_bank_accounts 
ADD COLUMN razorpay_fund_account_created_at TEXT;
*/


