# JobVite API Debugging Instructions

## Quick Debug Steps

### Step 1: Run the Debug Script

```bash
# Set your credentials
export JOBVITE_API_KEY="your_api_key"
export JOBVITE_API_SECRET="your_api_secret"
export JOBVITE_COMPANY_ID="your_company_id"
export JOBVITE_BASE_URL="https://api.jvistg2.com/api/v2"

# Run the debug script
cd backend
python debug_jobvite_api.py
```

**Paste the full output here** - especially:
- Status code
- Top-level keys
- First 5 items structure
- Full response preview

### Step 2: Check Application Logs

The client now logs raw API responses. Check your logs for:
```
JobVite API RAW RESPONSE (first 4000 chars)
```

This shows exactly what JobVite returns before parsing.

### Step 3: Manual cURL Test (Alternative)

If you prefer cURL:

```bash
# Legacy auth
curl -s -i -X GET "https://api.jvistg2.com/api/v2/job?start=0&count=100" \
  -H "x-jvi-api: YOUR_API_KEY" \
  -H "x-jvi-sc: YOUR_API_SECRET" \
  | head -100

# HMAC auth (if needed)
API_KEY="YOUR_API_KEY"
API_SECRET="YOUR_API_SECRET"
EPOCH=$(date +%s)
TO_HASH="${API_KEY}|${EPOCH}"
SIGN=$(printf "%s" "$TO_HASH" | openssl dgst -sha256 -hmac "$API_SECRET" -binary | openssl base64)

curl -s -i -X GET "https://api.jvistg2.com/api/v2/job?start=0&count=100" \
  -H "X-JVI-API: $API_KEY" \
  -H "X-JVI-SIGN: $SIGN" \
  -H "X-JVI-EPOCH: $EPOCH" \
  | head -100
```

**Paste the JSON body (first 400-800 chars)**

## What to Look For

### ✅ Good Response
```json
{
  "total": 50,
  "requisitions": [
    {
      "id": "123",
      "title": "Software Engineer",
      "status": "Published"
    }
  ]
}
```

### ❌ Empty Response
```json
{
  "total": 0,
  "requisitions": []
}
```

### ⚠️ Unexpected Structure
```json
{
  "data": {
    "items": [...]
  }
}
```

## Common Issues & Fixes

### Issue 1: Empty `requisitions` array
**Possible causes:**
- Jobs are draft/private (API user lacks permission)
- Wrong Company ID
- Jobs filtered out by default filters

**Fix:** Check JobVite UI - are jobs "Published" and visible to API user?

### Issue 2: Response has different key structure
**Fix:** The code now handles `jobs`, `requisitions`, `items`, `data` - but if you see a different structure, we'll add support.

### Issue 3: Jobs exist but fields are minimal
**Fix:** May need to fetch full job details via `/job/{id}` endpoint.

## Next Steps After Getting Output

1. **If response is empty:** Check permissions and job status in JobVite UI
2. **If response has different structure:** Share the structure and we'll update the parser
3. **If response has jobs but they're not showing:** Check sync process and database

## Enhanced Logging

The client now automatically logs:
- Raw API response (first 4000 chars)
- Final URL after redirects
- Response structure
- Parsed jobs count

Check your application logs for these debug messages.

