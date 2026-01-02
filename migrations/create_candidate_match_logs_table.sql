-- SQL Script to create candidate_match_logs table and update candidate_search_results
-- This script can be run directly on MySQL/MariaDB, PostgreSQL, or SQLite

-- Step 1: Add match_reasons column to candidate_search_results table (if it doesn't exist)
-- For MySQL/MariaDB:
ALTER TABLE candidate_search_results 
ADD COLUMN IF NOT EXISTS match_reasons TEXT NULL AFTER match_score;

-- For PostgreSQL (use this if MySQL version doesn't support IF NOT EXISTS):
-- DO $$ 
-- BEGIN
--     IF NOT EXISTS (
--         SELECT 1 FROM information_schema.columns 
--         WHERE table_name = 'candidate_search_results' 
--         AND column_name = 'match_reasons'
--     ) THEN
--         ALTER TABLE candidate_search_results ADD COLUMN match_reasons TEXT;
--     END IF;
-- END $$;

-- Step 2: Create candidate_match_logs table
CREATE TABLE IF NOT EXISTS candidate_match_logs (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    search_history_id INTEGER NULL,
    candidate_result_id INTEGER NULL,
    tenant_id INTEGER NOT NULL,
    user_id VARCHAR(128) NOT NULL,
    candidate_id VARCHAR(128) NOT NULL,
    candidate_name VARCHAR(255) NOT NULL,
    candidate_email VARCHAR(255) NULL,
    job_description TEXT NOT NULL,
    search_query TEXT NULL,
    search_criteria TEXT NULL,
    match_score FLOAT NOT NULL,
    match_reasons TEXT NOT NULL,
    match_explanation TEXT NULL,
    match_details TEXT NULL,
    algorithm_version VARCHAR(50) NULL,
    search_duration_ms INTEGER NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    FOREIGN KEY (search_history_id) REFERENCES candidate_search_history(id) ON DELETE SET NULL,
    FOREIGN KEY (candidate_result_id) REFERENCES candidate_search_results(id) ON DELETE SET NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- Indexes for better query performance
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_user_id (user_id),
    INDEX idx_candidate_id (candidate_id),
    INDEX idx_search_history_id (search_history_id),
    INDEX idx_created_at (created_at),
    INDEX idx_match_score (match_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- For PostgreSQL, use this instead:
/*
CREATE TABLE IF NOT EXISTS candidate_match_logs (
    id SERIAL PRIMARY KEY,
    search_history_id INTEGER NULL,
    candidate_result_id INTEGER NULL,
    tenant_id INTEGER NOT NULL,
    user_id VARCHAR(128) NOT NULL,
    candidate_id VARCHAR(128) NOT NULL,
    candidate_name VARCHAR(255) NOT NULL,
    candidate_email VARCHAR(255) NULL,
    job_description TEXT NOT NULL,
    search_query TEXT NULL,
    search_criteria TEXT NULL,
    match_score FLOAT NOT NULL,
    match_reasons TEXT NOT NULL,
    match_explanation TEXT NULL,
    match_details TEXT NULL,
    algorithm_version VARCHAR(50) NULL,
    search_duration_ms INTEGER NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (search_history_id) REFERENCES candidate_search_history(id) ON DELETE SET NULL,
    FOREIGN KEY (candidate_result_id) REFERENCES candidate_search_results(id) ON DELETE SET NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tenant_id ON candidate_match_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_user_id ON candidate_match_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_candidate_id ON candidate_match_logs(candidate_id);
CREATE INDEX IF NOT EXISTS idx_search_history_id ON candidate_match_logs(search_history_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON candidate_match_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_match_score ON candidate_match_logs(match_score);
*/

-- For SQLite, use this instead:
/*
CREATE TABLE IF NOT EXISTS candidate_match_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_history_id INTEGER NULL,
    candidate_result_id INTEGER NULL,
    tenant_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    candidate_name TEXT NOT NULL,
    candidate_email TEXT NULL,
    job_description TEXT NOT NULL,
    search_query TEXT NULL,
    search_criteria TEXT NULL,
    match_score REAL NOT NULL,
    match_reasons TEXT NOT NULL,
    match_explanation TEXT NULL,
    match_details TEXT NULL,
    algorithm_version TEXT NULL,
    search_duration_ms INTEGER NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (search_history_id) REFERENCES candidate_search_history(id) ON DELETE SET NULL,
    FOREIGN KEY (candidate_result_id) REFERENCES candidate_search_results(id) ON DELETE SET NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tenant_id ON candidate_match_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_user_id ON candidate_match_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_candidate_id ON candidate_match_logs(candidate_id);
CREATE INDEX IF NOT EXISTS idx_search_history_id ON candidate_match_logs(search_history_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON candidate_match_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_match_score ON candidate_match_logs(match_score);
*/

-- Verify table creation
SELECT 'Table candidate_match_logs created successfully' AS status;

-- Show table structure
DESCRIBE candidate_match_logs;

