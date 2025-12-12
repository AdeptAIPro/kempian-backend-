# 🚨 Trial Notifications System - Complete Setup Guide

## 📋 Overview

This system automatically sends email notifications to users whose trials are ending in 1 day and after their trials have expired. It includes both SMTP and AWS SES email delivery with fallback support.

## 🎯 Features

### **Email Types:**
1. **Trial Ending Reminder** - Sent 1 day before trial expires
2. **Trial Expired** - Sent after trial has expired

### **Email Content:**
- **Beautiful HTML templates** with responsive design
- **Clear call-to-action buttons** to upgrade/renew
- **Feature highlights** and benefits
- **Special offers** for expired trials (20% off)
- **Professional branding** with Kempian AI styling

## 🔧 Setup Instructions

### **1. Backend Setup**

The trial notification system is already integrated into the backend. Here's what was added:

#### **Files Created:**
- `backend/app/emails/trial_notifications.py` - Main email functions
- `backend/app/services/trial_notification_service.py` - Service logic
- `backend/app/trial_notifications/routes.py` - API endpoints
- `backend/scripts/send_trial_notifications.py` - Cron script
- `backend/scripts/setup_trial_notifications_cron.sh` - Cron setup script

#### **Files Modified:**
- `backend/app/emails/smtp.py` - Added SMTP functions
- `backend/app/__init__.py` - Registered blueprint

### **2. API Endpoints**

The system provides these API endpoints:

#### **Check and Send Notifications:**
```bash
POST /trial-notifications/check-and-send
```
Manually trigger trial notification check and send emails.

#### **Send to Specific User:**
```bash
POST /trial-notifications/send-to-user
Content-Type: application/json

{
  "user_id": 123,
  "type": "reminder"  // or "expired"
}
```

#### **Get Statistics:**
```bash
GET /trial-notifications/stats
```
Returns statistics about trials ending soon and expired trials.

#### **Test Email:**
```bash
POST /trial-notifications/test-email
Content-Type: application/json

{
  "email": "test@example.com",
  "type": "reminder"  // or "expired"
}
```

### **3. Automated Scheduling**

#### **Option A: Cron Job (Recommended)**

1. **Make the setup script executable:**
```bash
chmod +x backend/scripts/setup_trial_notifications_cron.sh
```

2. **Run the setup script:**
```bash
cd backend
./scripts/setup_trial_notifications_cron.sh
```

This will:
- Set up a cron job to run every hour
- Make the notification script executable
- Test the script to ensure it works
- Log output to `/var/log/trial_notifications.log`

#### **Option B: Manual Cron Setup**

Add this line to your crontab (`crontab -e`):
```bash
0 * * * * cd /path/to/backend && python3 scripts/send_trial_notifications.py >> /var/log/trial_notifications.log 2>&1
```

### **4. Testing the System**

#### **Test Script:**
```bash
cd backend
python3 test_trial_notifications.py
```

#### **Test with Test Data:**
```bash
cd backend
python3 test_trial_notifications.py --create-test
```

#### **Manual API Testing:**
```bash
# Check stats
curl -X GET http://localhost:8000/trial-notifications/stats

# Send test email
curl -X POST http://localhost:8000/trial-notifications/test-email \
  -H "Content-Type: application/json" \
  -d '{"email": "your-email@example.com", "type": "reminder"}'

# Trigger notification check
curl -X POST http://localhost:8000/trial-notifications/check-and-send
```

## 📧 Email Templates

### **Trial Ending Reminder Email:**
- **Subject:** "🚨 Your Kempian AI trial ends in 1 day!"
- **Content:** 
  - Clear warning about trial ending
  - What happens when trial expires
  - Call-to-action to upgrade
  - Feature highlights
  - Support contact information

### **Trial Expired Email:**
- **Subject:** "😢 Your Kempian AI trial has expired - Don't lose your progress!"
- **Content:**
  - Confirmation of trial expiration
  - What features are now unavailable
  - Special offer (20% off first month)
  - Call-to-action to renew
  - Benefits of upgrading

## 🔍 Monitoring and Logs

### **Log Files:**
- **Cron logs:** `/var/log/trial_notifications.log`
- **Application logs:** Check your Flask app logs
- **Email logs:** Check SMTP/SES logs

### **Monitoring Commands:**
```bash
# Check cron logs
tail -f /var/log/trial_notifications.log

# Check if cron job is running
crontab -l

# Test the notification script manually
cd backend && python3 scripts/send_trial_notifications.py
```

## ⚙️ Configuration

### **Email Settings:**
The system uses the same email configuration as your existing setup:

- **SMTP Settings:** `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`
- **AWS SES Settings:** `SES_REGION`, `SES_FROM_EMAIL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

### **Trial Settings:**
- **Trial Duration:** 7 days (configured in `UserTrial` model)
- **Reminder Timing:** 1 day before expiration
- **Expired Timing:** Within 24 hours of expiration

## 🚀 Usage Examples

### **1. Check Current Stats:**
```python
from app.services.trial_notification_service import get_trial_notification_stats

stats = get_trial_notification_stats()
print(f"Trials ending soon: {stats['trials_ending_soon']}")
print(f"Expired trials: {stats['expired_trials']}")
```

### **2. Send Manual Notification:**
```python
from app.services.trial_notification_service import send_trial_notification_to_user

# Send reminder to user ID 123
success = send_trial_notification_to_user(123, "reminder")
```

### **3. Run Full Check:**
```python
from app.services.trial_notification_service import check_and_send_trial_notifications

success = check_and_send_trial_notifications()
```

## 🔧 Troubleshooting

### **Common Issues:**

1. **Cron job not running:**
   - Check if cron service is running: `systemctl status cron`
   - Check cron logs: `journalctl -u cron`
   - Verify script permissions: `ls -la backend/scripts/send_trial_notifications.py`

2. **Email not sending:**
   - Check SMTP/SES configuration
   - Verify email credentials
   - Check application logs for errors

3. **Database connection issues:**
   - Ensure database is running
   - Check database connection string
   - Verify user permissions

### **Debug Commands:**
```bash
# Test email configuration
python3 -c "from app.emails.smtp import send_email_via_smtp; print('SMTP OK')"

# Test database connection
python3 -c "from app import create_app; from app.models import User; app = create_app(); print('DB OK')"

# Check trial data
python3 -c "from app import create_app; from app.models import UserTrial; app = create_app(); print(f'Trials: {UserTrial.query.count()}')"
```

## 📊 Performance

### **Efficiency:**
- **Batch processing** - Processes all trials in one run
- **Duplicate prevention** - Prevents sending multiple emails to same user
- **Error handling** - Continues processing even if individual emails fail
- **Logging** - Comprehensive logging for monitoring

### **Scalability:**
- **Database queries** - Optimized queries with proper indexing
- **Email delivery** - Uses existing SMTP/SES infrastructure
- **Cron scheduling** - Runs every hour to balance timeliness and load

## 🎉 Success Metrics

After setup, you should see:
- ✅ Cron job running every hour
- ✅ Emails being sent to users with trials ending soon
- ✅ Emails being sent to users with expired trials
- ✅ Logs showing successful processing
- ✅ API endpoints responding correctly

## 📞 Support

If you encounter any issues:
1. Check the logs first
2. Run the test script
3. Verify email configuration
4. Check database connectivity
5. Contact the development team

---

**🎯 The trial notification system is now ready to help convert trial users to paid customers!**
