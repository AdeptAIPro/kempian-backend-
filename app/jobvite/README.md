# Jobvite Integration Module

Complete integration with Jobvite API v2 and Onboarding API.

## Components

### Models (`models.py`)
- `JobviteSettings` - Integration configuration per tenant
- `JobviteJob` - Synced jobs
- `JobviteCandidate` - Synced candidates
- `JobviteCandidateDocument` - Candidate documents
- `JobviteOnboardingProcess` - Onboarding processes
- `JobviteOnboardingTask` - Onboarding tasks
- `JobviteWebhookLog` - Webhook event logs

### Utilities
- `utils.py` - Environment normalization, base URL helpers
- `crypto.py` - Encryption/decryption (at-rest, RSA+AES for Onboarding API)

### Clients
- `client_v2.py` - Jobvite API v2 client (HMAC-SHA256 auth)
- `client_onboarding.py` - Onboarding API client (RSA+AES encryption)

### Routes
- `routes.py` - Configuration and sync endpoints
- `webhooks.py` - Webhook handlers (tenant resolution via companyId)

### Sync
- `sync.py` - Background sync jobs for jobs, candidates, onboarding

## Setup

1. **Install dependencies:**
   ```bash
   pip install cryptography>=41.0.0
   ```

2. **Run migrations:**
   ```bash
   python backend/migrations/create_jobvite_tables.py
   ```

3. **Environment variables:**
   - `JOBVITE_ENCRYPTION_KEY` - Key for at-rest encryption (or uses JWT_SECRET_KEY as fallback)
   - `JOBVITE_V2_BASE_URL_STAGE` - Stage API URL (default: https://api-stage.jobvite.com/v2)
   - `JOBVITE_V2_BASE_URL_PROD` - Prod API URL (default: https://api.jobvite.com/v2)
   - `JOBVITE_ONBOARDING_BASE_URL_STAGE` - Stage Onboarding URL
   - `JOBVITE_ONBOARDING_BASE_URL_PROD` - Prod Onboarding URL

## API Endpoints

### Configuration
- `POST /api/integrations/jobvite/config` - Save configuration
- `GET /api/integrations/jobvite/config` - Get configuration
- `POST /api/integrations/jobvite/test-connection` - Test connection

### Sync
- `POST /api/integrations/jobvite/sync/jobs` - Trigger job sync
- `POST /api/integrations/jobvite/sync/candidates` - Trigger candidate sync
- `POST /api/integrations/jobvite/sync/onboarding` - Trigger onboarding sync
- `POST /api/integrations/jobvite/sync/full` - Full sync

### Webhooks
- `POST /webhooks/jobvite/<company_id>/candidate` - Candidate webhook
- `POST /webhooks/jobvite/<company_id>/job` - Job webhook

### Monitoring
- `GET /api/integrations/jobvite/webhooks` - Webhook logs

## Security

- **At-rest encryption:** All secrets encrypted with AES-256-CBC
- **Webhook signatures:** HMAC-SHA256 verification
- **Onboarding API:** RSA 2048 + AES-256-ECB encryption
- **Multi-tenancy:** Tenant resolution via companyId in webhook URLs

## Data Flow

1. **Configuration:** Client saves credentials → encrypted and stored
2. **Sync:** Background jobs fetch data → upsert to Jobvite tables
3. **Webhooks:** Jobvite sends events → signature verified → data fetched → upsert
4. **Frontend:** Reads from Jobvite tables via REST API

## Notes

- Environment values: Frontend uses "Stage"/"Production", backend stores "stage"/"prod"
- Company ID is globally unique (one Jobvite company = one Kempian tenant)
- Webhook tenant resolution uses companyId from URL path
- All sensitive data encrypted at rest using application encryption key

