# Trial Expired Email System Setup Guide

This guide explains how to set up and use the trial expired email system to send notifications to all users whose free trials have ended.

## 📁 Files Created

### 1. Python Script
- **File**: `backend/scripts/send_trial_expired_emails.py`
- **Purpose**: Main script to send trial expired emails
- **Features**: 
  - Dry run mode for testing
  - Filtering by days overdue
  - Limit for testing
  - Comprehensive logging
  - Error handling

### 2. Shell Script (Linux/Mac)
- **File**: `backend/scripts/run_trial_expired_emails.sh`
- **Purpose**: Easy execution wrapper for Unix systems
- **Features**: Command line argument parsing, error handling

### 3. Batch Script (Windows)
- **File**: `backend/scripts/run_trial_expired_emails.bat`
- **Purpose**: Easy execution wrapper for Windows systems
- **Features**: Windows-compatible execution, error handling

## 🚀 Quick Start

### 1. Test the System (Dry Run)
```bash
# Linux/Mac
cd backend/scripts
./run_trial_expired_emails.sh --dry-run

# Windows
cd backend\scripts
run_trial_expired_emails.bat --dry-run
```

### 2. Send Emails to All Expired Trials
```bash
# Linux/Mac
./run_trial_expired_emails.sh

# Windows
run_trial_expired_emails.bat
```

### 3. Send to Recently Expired Trials Only
```bash
# Send to trials expired in last 7 days
./run_trial_expired_emails.sh --days-overdue 7

# Windows
run_trial_expired_emails.bat --days-overdue 7
```

## 📋 Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--dry-run` | Preview what would be sent without actually sending | `--dry-run` |
| `--days-overdue N` | Only send to trials expired within last N days | `--days-overdue 7` |
| `--limit N` | Maximum number of emails to send (for testing) | `--limit 10` |
| `--verbose, -v` | Enable verbose logging | `--verbose` |
| `--help, -h` | Show help message | `--help` |

## 🔧 Advanced Usage

### 1. Direct Python Execution
```bash
cd backend
python scripts/send_trial_expired_emails.py --dry-run --limit 5 --verbose
```

### 2. Filter by Days Overdue
```bash
# Only send to trials expired in last 3 days
python scripts/send_trial_expired_emails.py --days-overdue 3
```

### 3. Test with Limited Users
```bash
# Send to maximum 5 users for testing
python scripts/send_trial_expired_emails.py --limit 5
```

## ⏰ Scheduling with Cron (Linux/Mac)

### 1. Edit Crontab
```bash
crontab -e
```

### 2. Add Cron Job
```bash
# Run every day at 9:00 AM
0 9 * * * /path/to/your/project/backend/scripts/run_trial_expired_emails.sh

# Run every 6 hours
0 */6 * * * /path/to/your/project/backend/scripts/run_trial_expired_emails.sh

# Run only on weekdays at 10:00 AM
0 10 * * 1-5 /path/to/your/project/backend/scripts/run_trial_expired_emails.sh
```

### 3. Log Output
```bash
# Run with logging
0 9 * * * /path/to/your/project/backend/scripts/run_trial_expired_emails.sh >> /var/log/trial_emails.log 2>&1
```

## 🪟 Scheduling with Windows Task Scheduler

### 1. Open Task Scheduler
- Press `Win + R`, type `taskschd.msc`, press Enter

### 2. Create Basic Task
- Click "Create Basic Task"
- Name: "Trial Expired Emails"
- Description: "Send emails to users with expired trials"

### 3. Set Trigger
- Choose "Daily" or "Weekly" as needed
- Set time (e.g., 9:00 AM)

### 4. Set Action
- Action: "Start a program"
- Program: `python`
- Arguments: `C:\path\to\your\project\backend\scripts\send_trial_expired_emails.py`
- Start in: `C:\path\to\your\project\backend`

## 📊 Monitoring and Logs

### 1. Check Logs
```bash
# View recent logs
tail -f /var/log/trial_emails.log

# Check application logs
tail -f backend/logs/app.log
```

### 2. Monitor Database
```sql
-- Check expired trials
SELECT u.email, ut.trial_end_date, ut.is_active
FROM users u
JOIN user_trials ut ON u.id = ut.user_id
WHERE ut.trial_end_date < NOW()
AND ut.is_active = true;

-- Check recently processed trials
SELECT u.email, ut.trial_end_date, ut.is_active
FROM users u
JOIN user_trials ut ON u.id = ut.user_id
WHERE ut.is_active = false
AND ut.updated_at > NOW() - INTERVAL 1 DAY;
```

## 🛠️ Troubleshooting

### 1. Common Issues

#### Python Path Issues
```bash
# Check Python path
which python3
python3 --version

# Set PYTHONPATH if needed
export PYTHONPATH=/path/to/your/project:$PYTHONPATH
```

#### Database Connection Issues
```bash
# Check database connection
python -c "from app import create_app; app = create_app(); print('Database connected')"
```

#### Email Configuration Issues
```bash
# Test email configuration
python -c "from app.emails.trial_notifications import send_trial_expired_email; print('Email config OK')"
```

### 2. Debug Mode
```bash
# Run with verbose logging
python scripts/send_trial_expired_emails.py --dry-run --verbose --limit 1
```

### 3. Check Email Templates
```bash
# Verify email templates exist
ls -la backend/app/emails/
```

## 📈 Performance Considerations

### 1. Batch Processing
- The script processes trials in batches
- Large numbers of expired trials are handled efficiently
- Database connections are managed properly

### 2. Rate Limiting
- Consider email service rate limits
- Use `--limit` option for testing
- Monitor email service quotas

### 3. Memory Usage
- Script loads trials in batches
- Memory usage is optimized for large datasets
- Database queries are efficient

## 🔒 Security Considerations

### 1. Environment Variables
- Ensure email credentials are properly configured
- Use environment variables for sensitive data
- Never hardcode credentials in scripts

### 2. Access Control
- Limit script execution to authorized users
- Use proper file permissions
- Monitor script execution logs

### 3. Data Privacy
- Email addresses are handled securely
- Personal data is processed according to privacy policies
- Logs don't contain sensitive information

## 📞 Support

### 1. Log Analysis
```bash
# Check for errors
grep -i error backend/logs/app.log

# Check email sending
grep -i "trial_expired_emails" backend/logs/app.log
```

### 2. Manual Testing
```bash
# Test with specific user
python -c "
from app import create_app
from app.services.trial_notification_service import send_trial_notification_to_user
app = create_app()
with app.app_context():
    result = send_trial_notification_to_user(USER_ID, 'expired')
    print(f'Result: {result}')
"
```

### 3. Email Service Status
- Check SMTP/SES service status
- Verify email credentials
- Test email delivery

## 🎯 Best Practices

1. **Always test with `--dry-run` first**
2. **Use `--limit` for initial testing**
3. **Monitor logs regularly**
4. **Set up proper scheduling**
5. **Have backup email methods**
6. **Test email templates**
7. **Monitor email delivery rates**
8. **Keep scripts updated**

## 📝 Example Workflows

### Daily Production Run
```bash
# Run every day at 9 AM
0 9 * * * /path/to/project/backend/scripts/run_trial_expired_emails.sh
```

### Weekly Cleanup
```bash
# Run weekly to catch any missed trials
0 10 * * 0 /path/to/project/backend/scripts/run_trial_expired_emails.sh --days-overdue 7
```

### Testing New Features
```bash
# Test with limited users
python scripts/send_trial_expired_emails.py --dry-run --limit 5 --verbose
```

This system provides a robust, scalable solution for sending trial expired emails to all users whose free trials have ended.
