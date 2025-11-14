-- Migration: Create Communication Tables for Twilio Integration
-- Description: Creates tables for message templates, candidate communications, and communication replies
-- Date: 2024

-- Table: message_templates
-- Stores pre-defined and custom message templates for Email, SMS, and WhatsApp
CREATE TABLE IF NOT EXISTS message_templates (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    user_id INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    channel ENUM('email', 'sms', 'whatsapp') NOT NULL,
    subject VARCHAR(500) NULL COMMENT 'For email only',
    body TEXT NOT NULL,
    variables TEXT NULL COMMENT 'JSON string of available variables',
    is_default BOOLEAN DEFAULT FALSE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_channel (channel),
    INDEX idx_is_default (is_default),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: candidate_communications
-- Stores all outgoing messages sent to candidates via Email, SMS, or WhatsApp
CREATE TABLE IF NOT EXISTS candidate_communications (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    user_id INTEGER NOT NULL,
    candidate_id VARCHAR(255) NOT NULL,
    candidate_name VARCHAR(255) NULL,
    candidate_email VARCHAR(255) NULL,
    candidate_phone VARCHAR(50) NULL,
    channel ENUM('email', 'sms', 'whatsapp') NOT NULL,
    template_id INTEGER NULL,
    message_subject VARCHAR(500) NULL COMMENT 'For email',
    message_body TEXT NOT NULL,
    twilio_message_sid VARCHAR(100) NULL COMMENT 'For SMS/WhatsApp tracking',
    sendgrid_message_id VARCHAR(100) NULL COMMENT 'For email tracking',
    status ENUM('pending', 'sent', 'delivered', 'failed', 'read', 'replied') DEFAULT 'pending' NOT NULL,
    delivery_status VARCHAR(100) NULL,
    error_message TEXT NULL,
    sent_at DATETIME NULL,
    delivered_at DATETIME NULL,
    read_at DATETIME NULL,
    replied_at DATETIME NULL,
    reply_content TEXT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (template_id) REFERENCES message_templates(id) ON DELETE SET NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_candidate_id (candidate_id),
    INDEX idx_channel (channel),
    INDEX idx_status (status),
    INDEX idx_template_id (template_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: communication_replies
-- Stores incoming replies from candidates
CREATE TABLE IF NOT EXISTS communication_replies (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    communication_id INTEGER NOT NULL,
    candidate_phone VARCHAR(50) NULL COMMENT 'For SMS/WhatsApp',
    candidate_email VARCHAR(255) NULL COMMENT 'For email',
    reply_content TEXT NOT NULL,
    channel ENUM('email', 'sms', 'whatsapp') NOT NULL,
    twilio_message_sid VARCHAR(100) NULL,
    received_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (communication_id) REFERENCES candidate_communications(id) ON DELETE CASCADE,
    INDEX idx_communication_id (communication_id),
    INDEX idx_channel (channel),
    INDEX idx_received_at (received_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

