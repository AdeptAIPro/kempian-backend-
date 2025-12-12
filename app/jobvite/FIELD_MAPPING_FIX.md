# JobVite Field Mapping Fix

## Problem Identified

The debug output revealed that JobVite API returns requisitions with different field names than expected:

### Issues Found:
1. **ID Field**: API returns `requisitionId` (e.g., "2025014K"), but code was looking for `id` or `jobId`
2. **Status Field**: API returns `jobState`, but code was looking for `status`
3. **Location Field**: API returns `jobLocations` (array) or `location` (string), but code was looking for `locations`
4. **Recruiter Email**: API may return `primaryRecruiterEmail` directly (string) or nested in `primaryRecruiter` object

## Fixes Applied

### 1. ID Field Mapping
**Before:**
```python
jobvite_job_id = job_data.get('id') or job_data.get('jobId')
```

**After:**
```python
requisition_id = job_data.get('requisitionId')
jobvite_job_id = job_data.get('id') or job_data.get('jobId') or requisition_id
# Use requisitionId as primary identifier
lookup_id = requisition_id or jobvite_job_id
```

### 2. Status Field Mapping
**Before:**
```python
status = job_data.get('status')
```

**After:**
```python
# JobVite API uses 'jobState' not 'status' for requisitions endpoint
status = job_data.get('status') or job_data.get('jobState')
```

### 3. Location Field Mapping
**Before:**
```python
locations = job_data.get('locations', [])
location_main = locations[0].get('city', '') if locations else None
```

**After:**
```python
# JobVite API uses 'jobLocations' (array) or 'location' (string) for requisitions
job_locations = job_data.get('jobLocations', [])
location_string = job_data.get('location', '')
location_city = job_data.get('locationCity', '')
location_state = job_data.get('locationState', '')
location_country = job_data.get('locationCountry', '')

if job_locations and isinstance(job_locations, list) and len(job_locations) > 0:
    first_loc = job_locations[0]
    if isinstance(first_loc, dict):
        location_main = first_loc.get('city') or first_loc.get('location') or location_string
    else:
        location_main = str(first_loc) if first_loc else location_string
else:
    # Build location string from components
    location_parts = [p for p in [location_city, location_state, location_country] if p]
    location_main = location_string or (', '.join(location_parts) if location_parts else '')
```

### 4. Recruiter Email Mapping
**Before:**
```python
primary_recruiter = job_data.get('primaryRecruiter') or {}
primary_recruiter_email = primary_recruiter.get('email')
```

**After:**
```python
primary_recruiter = job_data.get('primaryRecruiter') or {}
primary_recruiter_email = primary_recruiter.get('email') or job_data.get('primaryRecruiterEmail')
```

## API Response Structure (from debug output)

```json
{
  "total": 12,
  "requisitions": [
    {
      "requisitionId": "2025014K",  // Primary ID
      "title": "Head of Machine Learning & AI Engineer",
      "jobState": "...",  // Status field
      "location": "Remote in USA only",
      "locationCity": "...",
      "locationState": "...",
      "locationCountry": "...",
      "jobLocations": [...],  // Array of location objects
      "primaryRecruiterEmail": "...",  // Direct email field
      "primaryRecruiter": {  // Or nested object
        "email": "..."
      },
      ...
    }
  ],
  "status": "success"
}
```

## Next Steps

1. **Run a sync** to populate the database:
   ```bash
   POST /api/integrations/jobvite/sync/jobs
   ```

2. **Verify jobs appear**:
   ```bash
   GET /api/integrations/jobvite/jobs?page=1&pageSize=25
   ```

3. **Check logs** for any remaining field mapping issues

## Expected Result

After this fix:
- ✅ Jobs will be identified by `requisitionId`
- ✅ Status will be extracted from `jobState`
- ✅ Locations will be properly parsed from `jobLocations` or location fields
- ✅ Recruiter emails will be extracted correctly
- ✅ All 12 jobs should sync successfully

