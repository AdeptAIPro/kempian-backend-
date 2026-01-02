# User Activity Logs Table Migration

This guide explains how to create the `user_activity_logs` table for tracking all user activities across the platform.

## 📋 Prerequisites

- Python 3.8+ installed
- Flask application set up
- Database connection configured
- Virtual environment activated (recommended)

## 🗄️ Step 1: Create Database Table

### Option A: Using Python Script (Recommended)

**From project root:**
```bash
python backend/migrations/create_user_activity_logs_table.py
```

**OR from backend directory:**
```bash
cd backend
python migrations/create_user_activity_logs_table.py
```

**OR using the convenience script:**
```bash
python backend/create_user_activity_logs_table.py
```

### Option B: Using SQL Script Directly

If you prefer to run SQL directly in your database client:

1. Open `backend/migrations/create_user_activity_logs_table.sql`
2. Copy the SQL statements
3. Execute them in your MySQL/MariaDB client

**Using MySQL command line:**
```bash
mysql -u your_user -p your_database < backend/migrations/create_user_activity_logs_table.sql
```

### Option C: Using Flask Shell

You can also create the table using Flask's shell:

```bash
cd backend
python
```

```python
from app import create_app, db
from app.models import UserActivityLog

app = create_app()
with app.app_context():
    db.create_all()
    print("Table created successfully!")
```

## 🔍 Verify Table Creation

### Check if table exists:
```sql
SHOW TABLES LIKE 'user_activity_logs';
```

### View table structure:
```sql
DESCRIBE user_activity_logs;
```

### View indexes:
```sql
SHOW INDEXES FROM user_activity_logs;
```

## 📊 Table Structure

The `user_activity_logs` table includes:

- **User Information**: `user_email`, `user_id`, `user_role`
- **Activity Details**: `activity_type`, `action`, `endpoint`, `method`
- **Resource Information**: `resource_type`, `resource_id`
- **Request Details**: `ip_address`, `user_agent`, `request_data`
- **Response Details**: `status_code`, `response_time_ms`, `success`
- **Context**: `tenant_id`, `session_id`, `error_message`
- **Timestamps**: `created_at`, `updated_at`

## 🔑 Indexes

The table includes indexes on:
- `user_email` - For filtering by user email
- `user_id` - For filtering by user ID
- `activity_type` - For filtering by activity type
- `endpoint` - For filtering by API endpoint
- `tenant_id` - For multitenancy filtering
- `session_id` - For session tracking
- `created_at` - For date range queries

## ⚠️ Troubleshooting

### Error: "ModuleNotFoundError: No module named 'app'"
**Solution:** Make sure you're running from the project root or backend directory:
```bash
# From project root
python backend/migrations/create_user_activity_logs_table.py

# OR from backend directory
cd backend
python migrations/create_user_activity_logs_table.py
```

### Error: "Table 'user_activity_logs' doesn't exist"
**Solution:** Run the migration script:
```bash
python backend/migrations/create_user_activity_logs_table.py
```

### Error: "Table already exists"
**Solution:** This is normal if you've run the migration before. The script will detect this and show the existing table structure.

### Error: "Foreign key constraint fails"
**Solution:** Make sure the `users` and `tenants` tables exist before creating this table. The foreign keys reference:
- `users.id` (CASCADE on delete/update)
- `tenants.id` (SET NULL on delete, CASCADE on update)

## 📝 Next Steps

After creating the table:

1. **Start logging activities** by using the `@log_user_activity` decorator:
   ```python
   from app.utils.user_activity_logger import log_user_activity
   
   @user_bp.route('/some-action', methods=['POST'])
   @log_user_activity('create', action='Create resource', resource_type='resource')
   def create_resource():
       # Your route logic
       pass
   ```

2. **View user logs** at:
   - User view: `/activity-logs` or `/my-activity-logs`
   - Admin view: `/admin/user-activity-logs`

3. **Use the logging utility** for manual logging:
   ```python
   from app.utils.user_activity_logger import log_user_action
   
   log_user_action(
       user_id=user.id,
       user_email=user.email,
       user_role=user.role,
       activity_type='custom_action',
       action='Custom action description'
   )
   ```

## 🔒 Security Notes

- The `request_data` field stores sanitized request data (sensitive keys are redacted)
- IP addresses are stored for security auditing
- User agents are stored for device/browser tracking
- Error messages are limited to 500 characters

## 📈 Performance Considerations

- The table uses indexes on frequently queried columns
- Consider archiving old logs periodically (older than 1 year)
- Monitor table size and implement retention policies if needed

