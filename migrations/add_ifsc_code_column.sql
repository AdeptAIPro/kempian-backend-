-- Migration: Add ifsc_code column to user_bank_accounts table
-- Run this SQL directly in your MySQL database to fix the missing column error

-- Check if column exists first (optional - for verification)
-- SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME = 'user_bank_accounts' AND COLUMN_NAME = 'ifsc_code';

-- Add the ifsc_code column
ALTER TABLE user_bank_accounts
ADD COLUMN ifsc_code VARCHAR(11) NULL
COMMENT 'Indian Financial System Code';

-- Verify the column was added
SELECT 'ifsc_code column added successfully' as status;
