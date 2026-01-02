-- Migration script to add AI insights columns to candidate_profiles table
-- 
-- This script adds the following columns:
-- - ai_career_insight (TEXT)
-- - benchmarking_data (JSON)
-- - recommended_courses (JSON)
-- - insights_generated_at (DATETIME)
--
-- Usage:
--   For PostgreSQL: psql -d your_database -f migrate_add_insights_columns.sql
--   For MySQL: mysql -u username -p database_name < migrate_add_insights_columns.sql
--   For SQLite: sqlite3 your_database.db < migrate_add_insights_columns.sql

-- PostgreSQL version
-- Uncomment and use if you're using PostgreSQL:

/*
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='candidate_profiles' AND column_name='ai_career_insight') THEN
        ALTER TABLE candidate_profiles ADD COLUMN ai_career_insight TEXT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='candidate_profiles' AND column_name='benchmarking_data') THEN
        ALTER TABLE candidate_profiles ADD COLUMN benchmarking_data JSON;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='candidate_profiles' AND column_name='recommended_courses') THEN
        ALTER TABLE candidate_profiles ADD COLUMN recommended_courses JSON;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='candidate_profiles' AND column_name='insights_generated_at') THEN
        ALTER TABLE candidate_profiles ADD COLUMN insights_generated_at TIMESTAMP;
    END IF;
END $$;
*/

-- MySQL/MariaDB version
-- Uncomment and use if you're using MySQL or MariaDB:

/*
-- Check and add ai_career_insight
SET @dbname = DATABASE();
SET @tablename = 'candidate_profiles';
SET @columnname = 'ai_career_insight';
SET @preparedStatement = (SELECT IF(
  (
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    WHERE
      (table_name = @tablename)
      AND (table_schema = @dbname)
      AND (column_name = @columnname)
  ) > 0,
  'SELECT 1',
  CONCAT('ALTER TABLE ', @tablename, ' ADD COLUMN ', @columnname, ' TEXT')
));
PREPARE alterIfNotExists FROM @preparedStatement;
EXECUTE alterIfNotExists;
DEALLOCATE PREPARE alterIfNotExists;

-- Check and add benchmarking_data
SET @columnname = 'benchmarking_data';
SET @preparedStatement = (SELECT IF(
  (
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    WHERE
      (table_name = @tablename)
      AND (table_schema = @dbname)
      AND (column_name = @columnname)
  ) > 0,
  'SELECT 1',
  CONCAT('ALTER TABLE ', @tablename, ' ADD COLUMN ', @columnname, ' JSON')
));
PREPARE alterIfNotExists FROM @preparedStatement;
EXECUTE alterIfNotExists;
DEALLOCATE PREPARE alterIfNotExists;

-- Check and add recommended_courses
SET @columnname = 'recommended_courses';
SET @preparedStatement = (SELECT IF(
  (
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    WHERE
      (table_name = @tablename)
      AND (table_schema = @dbname)
      AND (column_name = @columnname)
  ) > 0,
  'SELECT 1',
  CONCAT('ALTER TABLE ', @tablename, ' ADD COLUMN ', @columnname, ' JSON')
));
PREPARE alterIfNotExists FROM @preparedStatement;
EXECUTE alterIfNotExists;
DEALLOCATE PREPARE alterIfNotExists;

-- Check and add insights_generated_at
SET @columnname = 'insights_generated_at';
SET @preparedStatement = (SELECT IF(
  (
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    WHERE
      (table_name = @tablename)
      AND (table_schema = @dbname)
      AND (column_name = @columnname)
  ) > 0,
  'SELECT 1',
  CONCAT('ALTER TABLE ', @tablename, ' ADD COLUMN ', @columnname, ' DATETIME')
));
PREPARE alterIfNotExists FROM @preparedStatement;
EXECUTE alterIfNotExists;
DEALLOCATE PREPARE alterIfNotExists;
*/

-- Simple SQL version (use with caution - will fail if columns exist)
-- Uncomment and use if you're sure the columns don't exist:

-- For PostgreSQL:
-- ALTER TABLE candidate_profiles ADD COLUMN ai_career_insight TEXT;
-- ALTER TABLE candidate_profiles ADD COLUMN benchmarking_data JSON;
-- ALTER TABLE candidate_profiles ADD COLUMN recommended_courses JSON;
-- ALTER TABLE candidate_profiles ADD COLUMN insights_generated_at TIMESTAMP;

-- For MySQL/MariaDB:
-- ALTER TABLE candidate_profiles ADD COLUMN ai_career_insight TEXT;
-- ALTER TABLE candidate_profiles ADD COLUMN benchmarking_data JSON;
-- ALTER TABLE candidate_profiles ADD COLUMN recommended_courses JSON;
-- ALTER TABLE candidate_profiles ADD COLUMN insights_generated_at DATETIME;

-- For SQLite:
-- ALTER TABLE candidate_profiles ADD COLUMN ai_career_insight TEXT;
-- ALTER TABLE candidate_profiles ADD COLUMN benchmarking_data TEXT;
-- ALTER TABLE candidate_profiles ADD COLUMN recommended_courses TEXT;
-- ALTER TABLE candidate_profiles ADD COLUMN insights_generated_at DATETIME;

