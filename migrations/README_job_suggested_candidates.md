# Job Suggested Candidates Table Migration

This migration creates the `job_suggested_candidates` table to store the top 3 suggested candidates for each job posting.

## Table Purpose

The `job_suggested_candidates` table caches the results of the semantic matching algorithm, allowing:
- Fast retrieval of suggested candidates without re-running the algorithm
- Better user experience with instant results
- Ability to refresh suggestions when needed

## Migration Options

### Option 1: Python Script (Recommended)

This is the recommended approach as it's database-agnostic and uses SQLAlchemy:

```bash
# From the backend directory
python migrations/create_job_suggested_candidates_table.py

# Or from the project root
python backend/migrations/create_job_suggested_candidates_table.py
```

### Option 2: SQL Script

If you prefer to run SQL directly:

**For MySQL/MariaDB:**
```bash
mysql -u your_user -p your_database < backend/migrations/create_job_suggested_candidates_table.sql
```

**For PostgreSQL:**
```bash
psql -U your_user -d your_database -f backend/migrations/create_job_suggested_candidates_table.sql
```

**For SQLite:**
```bash
sqlite3 your_database.db < backend/migrations/create_job_suggested_candidates_table.sql
```

Note: You may need to uncomment the appropriate section in the SQL file for your database type.

## Table Structure

```
job_suggested_candidates
├── id (PRIMARY KEY, AUTO_INCREMENT)
├── job_id (UNIQUE, FOREIGN KEY -> jobs.id)
├── candidates_data (TEXT, JSON string of top 3 candidates)
├── algorithm_used (VARCHAR(100), algorithm name)
├── generated_at (DATETIME, when suggestions were first generated)
└── updated_at (DATETIME, when suggestions were last updated)
```

## Verification

After running the migration, verify the table was created:

**Python:**
```python
from app import create_app, db
from app.models import JobSuggestedCandidates

app = create_app()
with app.app_context():
    # Check if table exists
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print('job_suggested_candidates' in tables)  # Should print True
```

**SQL:**
```sql
DESCRIBE job_suggested_candidates;
-- or
SHOW CREATE TABLE job_suggested_candidates;
```

## Rollback (if needed)

If you need to remove the table:

```sql
DROP TABLE IF EXISTS job_suggested_candidates;
```

## Notes

- The table uses a UNIQUE constraint on `job_id` to ensure only one set of suggestions per job
- Foreign key constraint ensures data integrity with the `jobs` table
- The `candidates_data` field stores JSON, so ensure your database supports TEXT type
- Indexes are created on `job_id`, `generated_at`, and `updated_at` for better query performance

