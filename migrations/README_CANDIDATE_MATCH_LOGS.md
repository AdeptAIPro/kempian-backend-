# Candidate Match Logs Table Migration

This migration creates the `candidate_match_logs` table and adds the `match_reasons` column to the `candidate_search_results` table.

## Quick Start

### Option 1: Python Script (Recommended)

```bash
# From project root
python backend/migrations/create_candidate_match_logs_table.py

# OR from backend directory
cd backend
python migrations/create_candidate_match_logs_table.py
```

### Option 2: SQL Script

```bash
# For MySQL/MariaDB
mysql -u your_username -p your_database < backend/migrations/create_candidate_match_logs_table.sql

# For PostgreSQL
psql -U your_username -d your_database -f backend/migrations/create_candidate_match_logs_table.sql

# For SQLite
sqlite3 your_database.db < backend/migrations/create_candidate_match_logs_table.sql
```

## What Gets Created

### 1. New Column in `candidate_search_results`
- **Column**: `match_reasons` (TEXT, nullable)
- **Purpose**: Stores JSON string of match reasons for each candidate result

### 2. New Table: `candidate_match_logs`
- **Purpose**: Long-term storage (2+ years) of detailed match logs for admin analytics
- **Key Features**:
  - Stores why each candidate was matched
  - Links to search history and candidate results
  - Includes match scores, reasons, explanations, and detailed breakdowns
  - Indexed for fast queries

## Table Structure

### candidate_match_logs

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| search_history_id | INTEGER | Foreign key to candidate_search_history |
| candidate_result_id | INTEGER | Foreign key to candidate_search_results |
| tenant_id | INTEGER | Foreign key to tenants |
| user_id | VARCHAR(128) | Cognito user ID |
| candidate_id | VARCHAR(128) | External candidate ID |
| candidate_name | VARCHAR(255) | Candidate name |
| candidate_email | VARCHAR(255) | Candidate email |
| job_description | TEXT | Job description used in search |
| search_query | TEXT | Actual search query |
| search_criteria | TEXT | JSON string of search filters |
| match_score | FLOAT | Match score (0.0 to 1.0) |
| match_reasons | TEXT | JSON string of match reasons |
| match_explanation | TEXT | Human-readable explanation |
| match_details | TEXT | JSON string of detailed scoring |
| algorithm_version | VARCHAR(50) | Algorithm version used |
| search_duration_ms | INTEGER | Search duration in milliseconds |
| created_at | DATETIME | Timestamp |

### Indexes
- `idx_tenant_id` - For filtering by tenant
- `idx_user_id` - For filtering by user
- `idx_candidate_id` - For finding all matches for a candidate
- `idx_search_history_id` - For linking to search history
- `idx_created_at` - For date range queries
- `idx_match_score` - For score-based filtering

## Verification

After running the migration, verify the table was created:

```sql
-- Check table exists
SHOW TABLES LIKE 'candidate_match_logs';

-- View table structure
DESCRIBE candidate_match_logs;

-- Check match_reasons column exists
DESCRIBE candidate_search_results;
```

## Troubleshooting

### Error: "Table already exists"
- This is normal if you've run the migration before
- The Python script will skip creation if the table exists
- The SQL script uses `CREATE TABLE IF NOT EXISTS`

### Error: "Column match_reasons already exists"
- The column may have been added manually
- The Python script will detect this and skip adding it

### Error: "Foreign key constraint fails"
- Make sure the referenced tables exist:
  - `candidate_search_history`
  - `candidate_search_results`
  - `tenants`

## Rollback

If you need to remove the table:

```sql
-- Remove the table (WARNING: This deletes all data!)
DROP TABLE IF EXISTS candidate_match_logs;

-- Remove the column (WARNING: This deletes all data in that column!)
ALTER TABLE candidate_search_results DROP COLUMN match_reasons;
```

## Next Steps

After running the migration:
1. The system will automatically start logging match reasons when candidates are saved
2. Access logs via admin routes: `/admin/candidate-match-logs`
3. View logs in the frontend admin dashboard

