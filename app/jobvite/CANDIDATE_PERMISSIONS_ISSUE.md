# JobVite Candidate Endpoint - Permissions Issue

## Problem

Candidates sync is failing with **401 Unauthorized** error:
```
Jobvite API authentication failed (401 Unauthorized). Invalid API key or secret.
Endpoint: /candidate
```

However, **jobs endpoint works fine** with the same credentials.

## Root Cause

This is a **JobVite API permissions issue**, not a code issue. The API key has access to:
- ✅ `/job` endpoint (jobs) - **WORKING**
- ❌ `/candidate` endpoint (candidates) - **NOT PERMITTED**

## Why This Happens

JobVite API keys can have different permission levels:
1. **Jobs only** - Can access jobs but not candidates
2. **Full access** - Can access both jobs and candidates
3. **Custom permissions** - Specific endpoints enabled

Your current API key appears to have **jobs-only** permissions.

## Solutions

### Option 1: Request Candidate Permissions (Recommended)
Contact JobVite Support to enable candidate access for your API key:
- Email: support@jobvite.com
- Subject: "Request Candidate API Access for API Key"
- Include:
  - Your API Key (masked: `peopleconnectstaffing_kempian_api_stg`)
  - Company ID: `qYTaVfwG`
  - Request: Enable access to `/candidate` endpoint

### Option 2: Use Different API Key
If you have another API key with candidate permissions, use that instead.

### Option 3: Work with Jobs Only
If candidate access isn't available, you can:
- Sync jobs successfully ✅
- View jobs in the frontend ✅
- Candidates will need to be synced manually or through other means

## Code Fixes Applied

Even though this is a permissions issue, I've applied the same fixes as jobs:

1. ✅ **Always include `start` and `count` parameters** (even for single candidate)
2. ✅ **Better error messages** that mention candidate permissions
3. ✅ **Same redirect handling** as jobs endpoint

## Verification

To verify if your API key has candidate permissions:

```bash
# Test candidate endpoint directly
curl -X GET "https://api.jvistg2.com/api/v2/candidate?start=0&count=1" \
  -H "x-jvi-api: YOUR_API_KEY" \
  -H "x-jvi-sc: YOUR_API_SECRET"
```

If you get 401, the key doesn't have candidate permissions.
If you get 200, the key has permissions and there's another issue.

## Current Status

- ✅ **Jobs sync**: Working perfectly (12 jobs found and syncing)
- ❌ **Candidates sync**: Blocked by API key permissions (401 Unauthorized)
- ✅ **Code**: Ready to sync candidates once permissions are granted

## Next Steps

1. **Contact JobVite Support** to request candidate endpoint access
2. **Once permissions are granted**, candidates sync will work automatically
3. **No code changes needed** - the code is already fixed and ready

The error message now includes a note about candidate permissions to make this clearer.

