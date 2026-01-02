-- SQL Script to create job_suggested_candidates table
-- This script can be run directly on MySQL/MariaDB, PostgreSQL, or SQLite
-- 
-- This table stores the top 3 suggested candidates for each job posting
-- to enable fast retrieval without re-running the matching algorithm

-- For MySQL/MariaDB:
CREATE TABLE IF NOT EXISTS job_suggested_candidates (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    job_id INTEGER NOT NULL UNIQUE,
    candidates_data TEXT NOT NULL COMMENT 'JSON string containing top 3 candidate data',
    algorithm_used VARCHAR(100) NULL COMMENT 'Name of the algorithm used for matching',
    generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    
    -- Indexes for better query performance
    INDEX idx_job_id (job_id),
    INDEX idx_generated_at (generated_at),
    INDEX idx_updated_at (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- For PostgreSQL, use this instead:
/*
CREATE TABLE IF NOT EXISTS job_suggested_candidates (
    id SERIAL PRIMARY KEY,
    job_id INTEGER NOT NULL UNIQUE,
    candidates_data TEXT NOT NULL,
    algorithm_used VARCHAR(100) NULL,
    generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_job_id ON job_suggested_candidates(job_id);
CREATE INDEX IF NOT EXISTS idx_generated_at ON job_suggested_candidates(generated_at);
CREATE INDEX IF NOT EXISTS idx_updated_at ON job_suggested_candidates(updated_at);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_job_suggested_candidates_updated_at 
    BEFORE UPDATE ON job_suggested_candidates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
*/

-- For SQLite, use this instead:
/*
CREATE TABLE IF NOT EXISTS job_suggested_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL UNIQUE,
    candidates_data TEXT NOT NULL,
    algorithm_used TEXT NULL,
    generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_job_id ON job_suggested_candidates(job_id);
CREATE INDEX IF NOT EXISTS idx_generated_at ON job_suggested_candidates(generated_at);
CREATE INDEX IF NOT EXISTS idx_updated_at ON job_suggested_candidates(updated_at);

-- SQLite doesn't support triggers for ON UPDATE CURRENT_TIMESTAMP
-- You'll need to update updated_at manually in your application code
*/

-- Verify table creation
SELECT 'Table job_suggested_candidates created successfully' AS status;

-- Show table structure
DESCRIBE job_suggested_candidates;

