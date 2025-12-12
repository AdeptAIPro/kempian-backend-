-- Migration: Add service account fields to jobvite_settings table
-- Run this SQL directly in your database

-- Add service account fields
ALTER TABLE jobvite_settings 
ADD COLUMN IF NOT EXISTS service_account_username VARCHAR(255) NULL,
ADD COLUMN IF NOT EXISTS service_account_password_encrypted TEXT NULL;

-- Add index for faster lookups
CREATE INDEX IF NOT EXISTS idx_jobvite_settings_service_account 
ON jobvite_settings(service_account_username);

-- Verify migration
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'jobvite_settings' 
AND column_name IN ('service_account_username', 'service_account_password_encrypted');

