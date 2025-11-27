-- SQL script to create integration_submissions table
-- Run this script in your database to create the table

CREATE TABLE IF NOT EXISTS integration_submissions (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    user_id INTEGER NOT NULL,
    user_email VARCHAR(255) NOT NULL,
    integration_type VARCHAR(255) NOT NULL,
    integration_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'in_progress',
    data TEXT,
    callback_url VARCHAR(500),
    source VARCHAR(100),
    saved_to_server BOOLEAN NOT NULL DEFAULT TRUE,
    submitted_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_integration_type (integration_type),
    INDEX idx_status (status),
    INDEX idx_submitted_at (submitted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

