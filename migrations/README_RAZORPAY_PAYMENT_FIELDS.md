# Razorpay Payment Fields Migration

This migration adds new fields required for Razorpay payment integration to existing database tables.

## New Fields Added

### `payroll_settings` Table
- `razorpay_webhook_secret` (VARCHAR(255), nullable) - Separate webhook secret for Razorpay webhook verification
- `razorpay_fund_account_validated` (BOOLEAN, default=False) - Tracks if the company's fund account ID has been validated
- `razorpay_fund_account_validated_at` (DATETIME, nullable) - Timestamp when fund account was validated

### `user_bank_accounts` Table
- `razorpay_contact_id` (VARCHAR(255), nullable) - Cached Razorpay contact ID to reduce API calls
- `razorpay_fund_account_id` (VARCHAR(255), nullable) - Cached Razorpay fund account ID to reduce API calls
- `razorpay_contact_created_at` (DATETIME, nullable) - Timestamp when contact was created in Razorpay
- `razorpay_fund_account_created_at` (DATETIME, nullable) - Timestamp when fund account was created in Razorpay

## Migration Options

### Option 1: Python Script (Recommended)

**For existing databases with tables already created:**

```bash
# From project root
python backend/migrations/add_razorpay_payment_fields.py

# OR from backend directory
cd backend
python migrations/add_razorpay_payment_fields.py
```

**What it does:**
- Checks if columns already exist (safe to run multiple times)
- Adds only missing columns
- Verifies the migration after completion
- Works with PostgreSQL, MySQL, and SQLite

### Option 2: SQL Script

**For PostgreSQL databases:**

```bash
psql -U your_user -d your_database -f backend/migrations/add_razorpay_payment_fields.sql
```

**For MySQL databases:**

Edit the SQL file to use MySQL syntax (see comments in the file), then:
```bash
mysql -u your_user -p your_database < backend/migrations/add_razorpay_payment_fields.sql
```

**For SQLite databases:**

SQLite doesn't support `IF NOT EXISTS` in `ALTER TABLE`, so use the Python script instead.

### Option 3: Create All Tables (New Installations)

**If you're setting up a fresh database:**

```bash
# From project root
python backend/create_all_tables.py
```

This will create all tables including the new fields (since they're already defined in the models).

## Verification

After running the migration, verify the columns were added:

### Using Python:
```python
from app import create_app, db
from sqlalchemy import inspect

app = create_app()
with app.app_context():
    inspector = inspect(db.engine)
    
    # Check payroll_settings columns
    payroll_cols = [col['name'] for col in inspector.get_columns('payroll_settings')]
    print('payroll_settings columns:', payroll_cols)
    
    # Check user_bank_accounts columns
    bank_cols = [col['name'] for col in inspector.get_columns('user_bank_accounts')]
    print('user_bank_accounts columns:', bank_cols)
```

### Using SQL:
```sql
-- PostgreSQL/MySQL
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'payroll_settings' 
AND column_name LIKE 'razorpay%';

SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'user_bank_accounts' 
AND column_name LIKE 'razorpay%';
```

## Rollback

If you need to remove these columns (not recommended for production):

```sql
-- PostgreSQL/MySQL
ALTER TABLE payroll_settings 
DROP COLUMN razorpay_webhook_secret,
DROP COLUMN razorpay_fund_account_validated,
DROP COLUMN razorpay_fund_account_validated_at;

ALTER TABLE user_bank_accounts 
DROP COLUMN razorpay_contact_id,
DROP COLUMN razorpay_fund_account_id,
DROP COLUMN razorpay_contact_created_at,
DROP COLUMN razorpay_fund_account_created_at;
```

## Troubleshooting

### Error: "Table does not exist"
- Run `python backend/create_all_tables.py` first to create all tables
- Then run the migration script to add the new columns

### Error: "Column already exists"
- This is safe to ignore - the script checks for existing columns
- The migration is idempotent (safe to run multiple times)

### Error: "Permission denied"
- Ensure your database user has `ALTER TABLE` permissions
- For production, use a database admin account

### Error: "Syntax error" (SQL script)
- The SQL script uses PostgreSQL syntax with `DO $$` blocks
- For MySQL, use the alternative syntax provided in comments
- For SQLite, use the Python script instead

## Next Steps

After running the migration:

1. **Configure Razorpay Settings:**
   - Go to Payroll Settings in the application
   - Enter your Razorpay API keys
   - Enter your Razorpay Fund Account ID
   - Test the connection
   - Validate the fund account

2. **Test Payment Processing:**
   - Create a test pay run
   - Verify funds validation works
   - Test payment processing

3. **Monitor:**
   - Check that webhook secret is being used for webhook verification
   - Verify fund account validation status is being tracked
   - Monitor payment processing logs

## Support

If you encounter issues:
1. Check the migration script output for specific error messages
2. Verify your database connection settings
3. Ensure you have the latest models.py with these field definitions
4. Check database logs for detailed error information


