# Integrations Management Backend

This document describes the backend API for managing integration submissions.

## Database Setup

### Create the Table

Run the migration script to create the `integration_submissions` table:

```bash
python backend/migrations/create_integration_submissions_table.py
```

Or use the SQL script directly:

```sql
-- Run this in your MySQL database
source backend/migrations/create_integration_submissions_table.sql
```

## API Endpoints

### 1. Submit Integration Request

**POST** `/api/integrations/submit`

Submit a new integration request from a user.

**Request Body:**
```json
{
  "integrationType": "jobadder",
  "integrationName": "JobAdder",
  "userId": "user@example.com",
  "userEmail": "user@example.com",
  "status": "in_progress",
  "data": {
    "apiKey": "xxx",
    "companyId": "123",
    "jobId": "456"
  },
  "callbackUrl": "https://kempian.ai/api/integrations/jobadder/callback",
  "source": "integration_overview"
}
```

**Response:**
```json
{
  "success": true,
  "id": 1,
  "message": "JobAdder integration submitted successfully"
}
```

### 2. Get All Integrations (Admin Only)

**GET** `/api/integrations/all`

Retrieve all integration submissions. Requires admin role.

**Response:**
```json
{
  "success": true,
  "integrations": [
    {
      "id": 1,
      "userId": "user@example.com",
      "userEmail": "user@example.com",
      "integrationType": "jobadder",
      "integrationName": "JobAdder",
      "status": "in_progress",
      "submittedAt": "2024-01-15T10:30:00",
      "updatedAt": "2024-01-15T10:30:00",
      "data": {
        "apiKey": "xxx",
        "companyId": "123"
      },
      "savedToServer": true
    }
  ],
  "count": 1
}
```

### 3. Update Integration

**PUT** `/api/integrations/{integration_id}/update`

Update an integration submission. Users can only update their own integrations, admins can update any.

**Request Body:**
```json
{
  "data": {
    "apiKey": "updated_key",
    "companyId": "123"
  },
  "status": "completed"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Integration updated successfully"
}
```

### 4. Send Email Notification

**POST** `/api/integrations/send-email`

Send email notification for integration submission.

**Request Body:**
```json
{
  "integrationData": {
    "integrationName": "JobAdder",
    "userEmail": "user@example.com"
  },
  "recipientEmail": "nishant@adeptaipro.com",
  "subject": "New Integration Request",
  "template": "integration_request"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Email sent successfully"
}
```

## Database Model

### IntegrationSubmission

- `id`: Primary key
- `user_id`: Foreign key to users table
- `user_email`: User's email address
- `integration_type`: Type of integration (e.g., 'jobadder', 'workday')
- `integration_name`: Display name (e.g., 'JobAdder', 'Workday Recruiting')
- `status`: Status ('in_progress', 'completed', 'failed', 'pending')
- `data`: JSON string containing integration credentials/data
- `callback_url`: OAuth callback URL if applicable
- `source`: Source of submission (e.g., 'integration_overview')
- `saved_to_server`: Boolean indicating if saved to backend
- `submitted_at`: Timestamp of submission
- `updated_at`: Timestamp of last update

## Security Notes

1. **Authentication**: All endpoints require JWT authentication
2. **Authorization**: Admin endpoints require admin role
3. **Data Storage**: Sensitive credentials are stored in the `data` field as JSON
4. **Encryption**: Consider encrypting sensitive fields in production

## Testing

Test the endpoints using curl or Postman:

```bash
# Submit integration
curl -X POST http://localhost:8000/api/integrations/submit \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "integrationType": "jobadder",
    "integrationName": "JobAdder",
    "userEmail": "user@example.com",
    "data": {"apiKey": "test"}
  }'

# Get all integrations (admin)
curl -X GET http://localhost:8000/api/integrations/all \
  -H "Authorization: Bearer ADMIN_JWT_TOKEN"
```

