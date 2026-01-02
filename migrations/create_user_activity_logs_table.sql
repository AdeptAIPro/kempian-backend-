-- SQL Script to create user_activity_logs table
-- This script can be run directly in MySQL/MariaDB client
-- 
-- Usage:
--   mysql -u your_user -p your_database < create_user_activity_logs_table.sql
--   OR
--   Copy and paste this SQL into your database client

CREATE TABLE IF NOT EXISTS `user_activity_logs` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_email` VARCHAR(255) NOT NULL,
  `user_id` INT NOT NULL,
  `user_role` VARCHAR(50) NULL,
  `activity_type` VARCHAR(100) NOT NULL,
  `action` VARCHAR(200) NULL,
  `endpoint` VARCHAR(500) NULL,
  `method` VARCHAR(10) NULL,
  `resource_type` VARCHAR(100) NULL,
  `resource_id` VARCHAR(255) NULL,
  `ip_address` VARCHAR(45) NULL,
  `user_agent` TEXT NULL,
  `request_data` TEXT NULL,
  `status_code` INT NULL,
  `response_time_ms` INT NULL,
  `tenant_id` INT NULL,
  `session_id` VARCHAR(255) NULL,
  `error_message` TEXT NULL,
  `success` BOOLEAN NOT NULL DEFAULT TRUE,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  INDEX `idx_user_email` (`user_email`),
  INDEX `idx_user_id` (`user_id`),
  INDEX `idx_activity_type` (`activity_type`),
  INDEX `idx_endpoint` (`endpoint`),
  INDEX `idx_tenant_id` (`tenant_id`),
  INDEX `idx_session_id` (`session_id`),
  INDEX `idx_created_at` (`created_at`),
  CONSTRAINT `fk_user_activity_logs_user_id` 
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) 
    ON DELETE CASCADE 
    ON UPDATE CASCADE,
  CONSTRAINT `fk_user_activity_logs_tenant_id` 
    FOREIGN KEY (`tenant_id`) REFERENCES `tenants` (`id`) 
    ON DELETE SET NULL 
    ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Add comments to columns for documentation
ALTER TABLE `user_activity_logs` 
  MODIFY COLUMN `user_email` VARCHAR(255) NOT NULL COMMENT 'Email of the user who performed the activity',
  MODIFY COLUMN `user_id` INT NOT NULL COMMENT 'ID of the user who performed the activity',
  MODIFY COLUMN `user_role` VARCHAR(50) NULL COMMENT 'Role of the user (owner, subuser, job_seeker, employee, recruiter, employer, admin)',
  MODIFY COLUMN `activity_type` VARCHAR(100) NOT NULL COMMENT 'Type of activity (login, logout, search, upload, download, view, create, update, delete, etc.)',
  MODIFY COLUMN `action` VARCHAR(200) NULL COMMENT 'Specific action performed',
  MODIFY COLUMN `endpoint` VARCHAR(500) NULL COMMENT 'API endpoint accessed',
  MODIFY COLUMN `method` VARCHAR(10) NULL COMMENT 'HTTP method (GET, POST, PUT, DELETE, PATCH)',
  MODIFY COLUMN `resource_type` VARCHAR(100) NULL COMMENT 'Type of resource being acted upon (candidate, job, resume, profile, etc.)',
  MODIFY COLUMN `resource_id` VARCHAR(255) NULL COMMENT 'ID of the resource being acted upon',
  MODIFY COLUMN `ip_address` VARCHAR(45) NULL COMMENT 'IP address of the user (IPv4 or IPv6)',
  MODIFY COLUMN `user_agent` TEXT NULL COMMENT 'User agent string from the request',
  MODIFY COLUMN `request_data` TEXT NULL COMMENT 'JSON string of request data (sanitized)',
  MODIFY COLUMN `status_code` INT NULL COMMENT 'HTTP status code of the response',
  MODIFY COLUMN `response_time_ms` INT NULL COMMENT 'Response time in milliseconds',
  MODIFY COLUMN `tenant_id` INT NULL COMMENT 'Tenant ID for multitenancy',
  MODIFY COLUMN `session_id` VARCHAR(255) NULL COMMENT 'Session ID for tracking user sessions',
  MODIFY COLUMN `error_message` TEXT NULL COMMENT 'Error message if the action failed',
  MODIFY COLUMN `success` BOOLEAN NOT NULL DEFAULT TRUE COMMENT 'Whether the action was successful',
  MODIFY COLUMN `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when the activity was logged',
  MODIFY COLUMN `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Timestamp when the record was last updated';

-- Show success message
SELECT 'Table user_activity_logs created successfully!' AS message;

