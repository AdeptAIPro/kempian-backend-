-- Create linkedin_integrations table
CREATE TABLE IF NOT EXISTS linkedin_integrations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(255) NOT NULL,
    access_token TEXT NOT NULL,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_org (organization_id),
    INDEX idx_org_id (organization_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

