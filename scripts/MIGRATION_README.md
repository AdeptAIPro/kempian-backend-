# Migration: Add Insights Columns to candidate_profiles Table

This migration adds the following columns to the `candidate_profiles` table to store AI-generated insights:

- `ai_career_insight` (TEXT) - AI-generated career summary
- `benchmarking_data` (JSON) - Industry benchmarking data
- `recommended_courses` (JSON) - Recommended courses from platforms
- `insights_generated_at` (DATETIME) - Timestamp when insights were generated

## Option 1: Python Script (Recommended)

The Python script automatically detects your database type and handles the migration safely.

### Usage:

```bash
cd backend
python migrate_add_insights_columns.py
```

### Features:
- ✅ Automatically detects database type (PostgreSQL, MySQL, SQLite)
- ✅ Checks if columns already exist (safe to run multiple times)
- ✅ Handles JSON type differences between databases
- ✅ Provides clear logging and error messages

## Option 2: SQL Script

If you prefer to run SQL directly, use the SQL script.

### For PostgreSQL:

```bash
psql -d your_database_name -f migrate_add_insights_columns.sql
```

Or connect to your database and run:

```sql
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
```

### For MySQL/MariaDB:

```bash
mysql -u username -p database_name < migrate_add_insights_columns.sql
```

Or connect and run:

```sql
ALTER TABLE candidate_profiles 
ADD COLUMN ai_career_insight TEXT,
ADD COLUMN benchmarking_data JSON,
ADD COLUMN recommended_courses JSON,
ADD COLUMN insights_generated_at DATETIME;
```

### For SQLite:

```bash
sqlite3 your_database.db < migrate_add_insights_columns.sql
```

Or:

```sql
ALTER TABLE candidate_profiles ADD COLUMN ai_career_insight TEXT;
ALTER TABLE candidate_profiles ADD COLUMN benchmarking_data TEXT;
ALTER TABLE candidate_profiles ADD COLUMN recommended_courses TEXT;
ALTER TABLE candidate_profiles ADD COLUMN insights_generated_at DATETIME;
```

## Verification

After running the migration, verify the columns were added:

### PostgreSQL:
```sql
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'candidate_profiles' 
AND column_name IN ('ai_career_insight', 'benchmarking_data', 'recommended_courses', 'insights_generated_at');
```

### MySQL:
```sql
SHOW COLUMNS FROM candidate_profiles 
WHERE Field IN ('ai_career_insight', 'benchmarking_data', 'recommended_courses', 'insights_generated_at');
```

### SQLite:
```sql
PRAGMA table_info(candidate_profiles);
```

## Rollback (if needed)

If you need to remove these columns:

```sql
-- PostgreSQL/MySQL/SQLite
ALTER TABLE candidate_profiles 
DROP COLUMN ai_career_insight,
DROP COLUMN benchmarking_data,
DROP COLUMN recommended_courses,
DROP COLUMN insights_generated_at;
```

## Notes

- The Python script is **idempotent** - safe to run multiple times
- All new columns are **nullable** - existing records won't be affected
- JSON columns in SQLite are stored as TEXT (SQLite doesn't have native JSON type)
- Make sure to backup your database before running migrations in production

