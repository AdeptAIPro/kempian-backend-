# Quota Management Scripts

This directory contains scripts to manage user quota limits in the Kempian application.

## 📋 **Important Note**

These scripts work with the **existing Plan-based quota system** where:
- Users inherit quota from their tenant's plan (`jd_quota_per_month`)
- Quota usage is tracked via `JDSearchLog` entries
- No separate `Quota` model exists - quota is managed through the `Plan` model

## 📁 Files

- `scripts/manage_quota.py` - Command-line quota management script
- `scripts/interactive_quota.py` - Interactive quota management script (easier to use)
- `scripts/test_quota.py` - Simple test script to verify functionality
- `scripts/example_emails.txt` - Example file for bulk operations
- `scripts/example_quota_config.txt` - Example file for bulk quota setting

## 🚀 Quick Start

### Interactive Script (Recommended for beginners)

```bash
cd backend/docs/scripts
python interactive_quota.py
```

This will start an interactive menu where you can:
1. Set quota limits (updates the user's plan)
2. Reset user quotas (clears current month search logs)
3. View quota status (shows plan info + usage)
4. List all users and quotas

### Command-Line Script (For automation)

```bash
cd backend/docs/scripts
python manage_quota.py --help
```

## 📋 Available Actions

### 1. Set Quota Limit
Set a specific quota limit for a user by updating their plan's `jd_quota_per_month`.

**Interactive:**
- Choose option 1 from the menu
- Enter email and limit

**Command-line:**
```bash
python manage_quota.py --email user@example.com --action set --limit 100
```

### 2. Reset Quota
Reset a user's quota usage by clearing all search logs for the current month.

**Interactive:**
- Choose option 2 from the menu
- Enter email

**Command-line:**
```bash
python manage_quota.py --email user@example.com --action reset
```

### 3. View Quota Status
View current quota information for a user including plan details and usage.

**Interactive:**
- Choose option 3 from the menu
- Enter email

**Command-line:**
```bash
python manage_quota.py --email user@example.com --action view
```

### 4. List All Quotas
View quota information for all users.

**Interactive:**
- Choose option 4 from the menu

**Command-line:**
```bash
python manage_quota.py --action list
```

## 🔄 Bulk Operations

### Bulk Reset Quotas
Reset quotas for multiple users at once.

1. Create a file with email addresses (one per line):
```txt
user1@example.com
user2@example.com
user3@example.com
```

2. Run the command:
```bash
python manage_quota.py --action bulk-reset --file emails.txt
```

### Bulk Set Quotas
Set quota limits for multiple users at once.

1. Create a file with email:limit pairs:
```txt
user1@example.com:100
user2@example.com:200
user3@example.com:50
```

2. Run the command:
```bash
python manage_quota.py --action bulk-set --file quota_config.txt
```

## 📊 Example Output

### View Quota Status
```
======================================================================
QUOTA STATUS FOR: user@example.com
======================================================================
Plan:         Pro
Tenant ID:    123
Tenant Status: active
Quota Limit:  100
Used:         25
Remaining:    75
Usage:        25.0%
======================================================================
```

### List All Quotas
```
========================================================================================================================
QUOTA STATUS FOR ALL USERS (3 total)
========================================================================================================================
Email                          Plan            Limit    Used     Remaining  Usage %  Status    
--------------------------------------------------------------------------------
user1@example.com              Pro             100      25       75         25.0%    active    
user2@example.com              Business        200      150      50         75.0%    active    
user3@example.com              Starter         10       0        10         0.0%     active    
========================================================================================================================
```

## ⚙️ How It Works

### Plan-Based Quota System
- Users belong to **Tenants**
- Tenants have **Plans** with `jd_quota_per_month` limits
- Quota usage is tracked via **JDSearchLog** entries
- Monthly usage is calculated from search logs in the current month

### Database Relationships
```
User → Tenant → Plan (jd_quota_per_month)
User → JDSearchLog (tracks usage)
```

### Quota Calculation
- **Limit**: `plan.jd_quota_per_month`
- **Used**: Count of `JDSearchLog` entries for current month
- **Remaining**: `limit - used`
- **Usage %**: `(used / limit) * 100`

## ⚙️ Configuration

### Database Connection
The scripts automatically connect to the database using the Flask app configuration. Make sure:
1. Your database is running
2. Environment variables are set correctly
3. The Flask app can connect to the database

### Default Values
- New users inherit quota from their tenant's plan
- Quota usage starts at 0 (no search logs)
- Quota limits must be positive integers

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Error**
   ```
   ❌ Failed to connect to database: [error message]
   ```
   - Check if database is running
   - Verify database credentials
   - Ensure Flask app configuration is correct

2. **User Not Found**
   ```
   ❌ User not found: user@example.com
   ```
   - Verify the email address is correct
   - Check if the user exists in the database

3. **No Tenant/Plan Assigned**
   ```
   ❌ User has no tenant or plan assigned
   ```
   - Ensure user is assigned to a tenant
   - Ensure tenant has a plan assigned
   - Check tenant and plan relationships

4. **Permission Errors**
   ```
   ❌ Permission denied: [file path]
   ```
   - Ensure you have write permissions in the scripts directory
   - Check file permissions for log files

### Log Files
- `quota_management.log` - Detailed logs for the command-line script
- Console output - Real-time feedback for both scripts

## 🛡️ Security Notes

- These scripts require database access
- Use with caution in production environments
- Consider backing up the database before bulk operations
- Logs may contain sensitive information (email addresses)
- **Reset operations delete search logs** - use carefully

## 📝 Usage Examples

### Daily Operations
```bash
# Check quota status for a user
python manage_quota.py --email john@company.com --action view

# Reset quota for a user who exceeded their limit
python manage_quota.py --email john@company.com --action reset

# Increase quota limit for a VIP user
python manage_quota.py --email vip@company.com --action set --limit 500
```

### Monthly Operations
```bash
# Reset all user quotas at the start of the month
python manage_quota.py --action bulk-reset --file all_users.txt

# Set different quota tiers for different user types
python manage_quota.py --action bulk-set --file quota_tiers.txt
```

### Administrative Tasks
```bash
# Audit all user quotas
python manage_quota.py --action list

# Check specific user's usage
python manage_quota.py --email admin@company.com --action view
```

## 🔄 System Integration

### How Quota is Used in the Application
1. **Frontend**: Shows quota status in user interface
2. **Backend**: Checks quota before allowing searches
3. **Logging**: Records each search in `JDSearchLog`
4. **Monthly Reset**: Quota resets automatically each month

### Related Models
- `User`: User information
- `Tenant`: Organization/company
- `Plan`: Subscription plan with quota limits
- `JDSearchLog`: Tracks search usage

## 🤝 Support

If you encounter issues:
1. Check the log files for detailed error messages
2. Verify database connectivity
3. Ensure all required dependencies are installed
4. Check user-tenant-plan relationships
5. Contact the development team for assistance 