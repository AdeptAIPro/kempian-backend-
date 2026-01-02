-- Migration: Add bank_branch column to user_bank_accounts table
-- Run this SQL directly in your MySQL database to fix the missing column error

-- Check if column exists first (optional - for verification)
-- SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME = 'user_bank_accounts' AND COLUMN_NAME = 'bank_branch';

-- Add the bank_branch column
ALTER TABLE user_bank_accounts
ADD COLUMN bank_branch VARCHAR(255) NULL
COMMENT 'Bank branch name';

-- Verify the column was added
SELECT 'bank_branch column added successfully' as status;
