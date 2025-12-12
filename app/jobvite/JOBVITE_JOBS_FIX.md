# JobVite Jobs Not Showing - Fix Applied

## Problem
Jobs exist in JobVite account but are not appearing in the application.

## Root Cause
The JobVite API response structure uses `requisitions` key instead of `jobs` key:
```json
{
  "total": 10,
  "requisitions": [...],  // Jobs are here, not under "jobs"
  "status": "success"
}
```

The code was only checking for `jobs`, `items`, or `data` keys, but not `requisitions`.

## Fix Applied
Updated `backend/app/jobvite/client_v2.py` to handle `requisitions` key:

```python
if 'requisitions' in data:
    # JobVite API sometimes returns jobs under 'requisitions' key
    requisitions = data['requisitions']
    if isinstance(requisitions, list):
        data['jobs'] = requisitions
    elif isinstance(requisitions, dict) and 'items' in requisitions:
        data['jobs'] = requisitions['items']
    else:
        data['jobs'] = [requisitions] if requisitions else []
    logger.debug(f"Found jobs under 'requisitions' key: {len(data['jobs'])} jobs")
```

## Next Steps to See Jobs

### Option 1: Manual Sync (Recommended for Testing)
1. **Trigger a job sync** via API:
   ```bash
   POST /api/integrations/jobvite/sync/jobs
   ```
   This will:
   - Fetch all jobs from JobVite API
   - Store them in the database
   - Return a job ID to track progress

2. **Check sync status**:
   ```bash
   GET /api/integrations/jobvite/jobs/<job_id>/status
   ```

3. **View synced jobs**:
   ```bash
   GET /api/integrations/jobvite/jobs?page=1&pageSize=25
   ```

### Option 2: Automatic Sync (If Configured)
If `syncJobs` is enabled in the sync configuration, jobs should sync automatically. Check:
- JobVite settings: `sync_config.syncJobs = true`
- Background sync jobs are running (Celery workers)

### Option 3: Via Frontend
1. Go to JobVite Integration page
2. Click "Sync Jobs" button
3. Wait for sync to complete
4. Jobs should appear in the jobs list

## Verification

After syncing, you should see:
- Jobs in `/api/integrations/jobvite/jobs` endpoint
- No more "Response missing 'jobs' key" warnings in logs
- Debug log: "Found jobs under 'requisitions' key: X jobs"

## Testing

1. **Test API directly**:
   ```python
   from app.jobvite.client_v2 import JobviteV2Client
   
   client = JobviteV2Client(api_key, api_secret, company_id, base_url)
   result = client.get_job(start=0, count=10)
   print(f"Found {len(result.get('jobs', []))} jobs")
   ```

2. **Check logs** for:
   - `Found jobs under 'requisitions' key: X jobs` (success)
   - `Response missing 'jobs' key` (should not appear anymore)

## Additional Notes

- The redirect warning (`Jobvite API redirect detected`) is normal - it's just the API adding pagination parameters
- Jobs are stored in `JobviteJob` table in the database
- The `/api/integrations/jobvite/jobs` endpoint returns jobs from the database, not directly from the API
- Sync must be run to populate the database with jobs from JobVite

