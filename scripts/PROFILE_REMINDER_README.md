# Profile Completion Reminder Scripts

This directory contains scripts to send reminder emails to jobseekers who haven't completed their profile, whether they signed up via Google OAuth or email login.

## Available Scripts

1. **`send_profile_completion_reminder.py`** - Sends reminders to ALL jobseekers with incomplete profiles (bulk send)
2. **`send_profile_reminder_to_specific_user.py`** - Sends reminder to a SPECIFIC candidate email address (targeted send)
3. **`list_jobseekers_without_profile.py`** - Lists all jobseekers who don't have a profile (audit/reporting)

## What It Does

- Finds all jobseekers with incomplete profiles
- Sends a personalized reminder email encouraging them to complete their profile
- Works for both Google OAuth and email-based logins
- Only sends to users who signed up at least 1 day ago (to avoid immediate spam)

## Profile Completion Criteria

A profile is considered **incomplete** if:
- The user doesn't have a `CandidateProfile` record, OR
- The profile is missing required fields:
  - `full_name` (required)
  - `email` (from User model, required)
  - Either `skills` (at least one skill) OR `experience_years` (not null)

## Usage

### Script 1: Send to All Incomplete Profiles (Bulk)

```bash
# From the backend directory
cd backend
python -m scripts.send_profile_completion_reminder

# Or using direct path
python backend/scripts/send_profile_completion_reminder.py
```

### Script 2: Send to Specific User (Targeted)

Send reminder to a specific candidate email:

```bash
# With email as argument
python -m scripts.send_profile_reminder_to_specific_user user@example.com

# Or using direct path
python backend/scripts/send_profile_reminder_to_specific_user.py user@example.com

# Interactive mode (will prompt for email)
python backend/scripts/send_profile_reminder_to_specific_user.py

# Force send even if profile is complete (use with caution)
python backend/scripts/send_profile_reminder_to_specific_user.py user@example.com --force
```

**Use cases for specific user script:**
- Testing the email template
- Sending to a specific user who requested help
- Manual follow-up for important candidates
- Debugging email delivery issues

### Script 3: List Jobseekers Without Profile (Audit)

List all jobseekers who don't have a profile:

```bash
# Display list in console
python -m scripts.list_jobseekers_without_profile

# Show detailed information
python -m scripts.list_jobseekers_without_profile --detailed

# Export to CSV
python -m scripts.list_jobseekers_without_profile --export csv

# Export to JSON
python -m scripts.list_jobseekers_without_profile --export json

# Or using direct path
python backend/scripts/list_jobseekers_without_profile.py --detailed
```

**Use cases for list script:**
- Auditing incomplete profiles
- Generating reports
- Identifying users who need follow-up
- Data analysis and statistics
- Exporting data for external analysis

### Schedule it (Cron/Windows Task Scheduler):

**Linux/Mac (Cron):**
```bash
# Run daily at 9 AM
0 9 * * * cd /path/to/backend && python -m scripts.send_profile_completion_reminder >> logs/profile_reminder.log 2>&1
```

**Windows (Task Scheduler):**
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., Daily at 9:00 AM)
4. Action: Start a program
5. Program: `python`
6. Arguments: `-m scripts.send_profile_completion_reminder`
7. Start in: `D:\kempian_3.0-my-new-branch\kempian_3.0-my-new-branch\backend`

## Email Content

The email includes:
- Friendly reminder about incomplete profile
- Benefits of completing profile (matching, discovery, etc.)
- Step-by-step guide to complete profile
- Direct link to profile completion page
- Support contact information

## Configuration

The script uses the existing SMTP email configuration from your environment variables:
- `SMTP_SERVER` (default: smtp.hostinger.com)
- `SMTP_PORT` (default: 587)
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`

## Customization

### Change the minimum days since signup:

Edit the `main()` function:
```python
# Only send to users who signed up at least 3 days ago
incomplete_users = get_incomplete_jobseekers(days_since_signup=3)
```

### Modify email content:

Edit the `send_profile_completion_reminder_email()` function to customize the HTML/text email templates.

## Logging

The script logs:
- Number of incomplete profiles found
- Success/failure for each email sent
- Summary statistics at the end

Logs are written to the application logger (check your log files).

## Example Output

### Bulk Script Output:

```
[PROFILE REMINDER] Starting profile completion reminder script...
[PROFILE REMINDER] Found 15 jobseekers with incomplete profiles
[PROFILE REMINDER] ✓ Sent to user1@example.com (login: google)
[PROFILE REMINDER] ✓ Sent to user2@example.com (login: email)
[PROFILE REMINDER] ========================================
[PROFILE REMINDER] Summary:
[PROFILE REMINDER]   Total incomplete profiles: 15
[PROFILE REMINDER]   Emails sent successfully: 14
[PROFILE REMINDER]   Emails failed: 1
[PROFILE REMINDER] ========================================
[PROFILE REMINDER] Script completed.
```

### Specific User Script Output:

```
============================================================
Profile Completion Reminder - Send to Specific User
============================================================

Enter candidate email address: john.doe@example.com

============================================================
Result:
============================================================
✓ SUCCESS: Reminder email sent successfully to john.doe@example.com
  Email: john.doe@example.com
  User Name: John Doe
  Login Method: google
  Has Profile: True
  Profile Incomplete: True
============================================================
```

**Error Example:**
```
============================================================
Result:
============================================================
✗ FAILED: Profile for user@example.com is already complete. Use force_send=True to send anyway.
  Email: user@example.com
  Has Profile: True
  Profile Complete: True
============================================================
```

### List Script Output:

```
======================================================================
SUMMARY STATISTICS
======================================================================
Total jobseekers without profile: 25
  - Google OAuth logins: 15
  - Email logins: 10

Signup Distribution:
  - 0-1 days: 5 (20.0%)
  - 2-7 days: 8 (32.0%)
  - 8-30 days: 7 (28.0%)
  - 31-90 days: 3 (12.0%)
  - 90+ days: 2 (8.0%)
======================================================================

================================================================================
JOBSEEKERS WITHOUT PROFILE
================================================================================
ID     Email                                        Login    Days Since Signup
--------------------------------------------------------------------------------
123    john.doe@example.com                        GOOGLE   5
124    jane.smith@example.com                      EMAIL    12
125    bob.jones@example.com                       GOOGLE   3
...
================================================================================

======================================================================
EMAIL LIST (for easy copy-paste):
======================================================================
john.doe@example.com, jane.smith@example.com, bob.jones@example.com, ...
======================================================================
```

## Notes

- The script respects the existing email sending infrastructure
- It won't send duplicate emails if run multiple times (no tracking of sent emails)
- Consider adding a "last_reminder_sent" field to User model if you want to prevent duplicate sends
- The script is safe to run multiple times - it will send to all incomplete profiles each time

## Troubleshooting

**No emails sent:**
- Check SMTP configuration in environment variables
- Verify SMTP credentials are correct
- Check application logs for errors

**Script fails to import:**
- Ensure you're running from the correct directory
- Check that all dependencies are installed
- Verify database connection is working

**Too many emails:**
- Adjust `days_since_signup` parameter
- Consider adding a cooldown period between reminders

