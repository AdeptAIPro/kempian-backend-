# JobAdder OAuth Compliance - Email vs Implementation

## Email Request Summary
The email to JobAdder support requested:
- **Scopes**: `read write offline_access` ✅
- **Client ID**: `qlph67cdt34u5eb65bnohlblbu`
- **Redirect URI**: `https://api.kempian.ai/integrations/jobadder/oauth/callback`
- **Authorization URL shown in email**: `https://id.jobadder.com/oauth2/authorize?...`

## Current Implementation Status

### ✅ Scopes - CORRECT
- **Default scope**: `read write offline_access` (matches email request)
- **Location**: 
  - `routes.py` line 185: Uses `read write offline_access`
  - `auth.py` line 37: Uses `read write offline_access`
  - `client.py` line 36: **FIXED** - Now uses `read write offline_access` (was `jobadder.api offline_access`)

### ⚠️ Endpoint Discrepancy - NEEDS CLARIFICATION
- **Official Documentation**: Uses `/connect/authorize` and `/connect/token`
- **Email Example**: Shows `/oauth2/authorize`
- **Current Implementation**: Uses `/connect/authorize` and `/connect/token` (per official docs)

**Decision**: We're following the official documentation which specifies `/connect/authorize`. However, if JobAdder support confirms that `/oauth2/authorize` is the correct endpoint for your specific client_id, we can override it via environment variable:

```bash
JOBADDER_AUTHORIZE_URL=https://id.jobadder.com/oauth2/authorize
JOBADDER_TOKEN_URL=https://id.jobadder.com/oauth2/token
```

### ✅ Redirect URI - CORRECT
- **Email**: `https://api.kempian.ai/integrations/jobadder/oauth/callback`
- **Implementation**: Uses `JOBADDER_OAUTH_REDIRECT_URI` environment variable or constructs from `BACKEND_URL`/`API_URL`
- **Must match exactly** (no trailing slash, case-sensitive)

### ✅ Authorization URL Format - CORRECT
The implementation generates URLs in the correct format:
```
https://id.jobadder.com/connect/authorize?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=read%20write%20offline_access&state={STATE}
```

## Action Items

1. **Verify Endpoint**: Confirm with JobAdder support whether `/oauth2/authorize` or `/connect/authorize` should be used for client_id `qlph67cdt34u5eb65bnohlblbu`
   - If `/oauth2/authorize` is required, set environment variables:
     ```bash
     JOBADDER_AUTHORIZE_URL=https://id.jobadder.com/oauth2/authorize
     JOBADDER_TOKEN_URL=https://id.jobadder.com/oauth2/token
     ```

2. **Scope Enablement**: Wait for JobAdder support to enable `read write offline_access` scopes for the client_id

3. **Test After Scope Enablement**: Once scopes are enabled, test with:
   ```
   https://id.jobadder.com/connect/authorize?response_type=code&client_id=qlph67cdt34u5eb65bnohlblbu&redirect_uri=https%3A%2F%2Fapi.kempian.ai%2Fintegrations%2Fjobadder%2Foauth%2Fcallback&scope=read%20write%20offline_access&state=STATE_123
   ```

## Code Changes Made

1. ✅ Fixed `client.py` to use `read write offline_access` instead of `jobadder.api offline_access`
2. ✅ All files now consistently use `read write offline_access` as default scope
3. ✅ Endpoints use `/connect/authorize` and `/connect/token` per official documentation

## Environment Variables to Set

For production:
```bash
JOBADDER_OAUTH_REDIRECT_URI=https://api.kempian.ai/integrations/jobadder/oauth/callback
JOBADDER_DEFAULT_SCOPE=read write offline_access
# If /oauth2/authorize is required instead of /connect/authorize:
# JOBADDER_AUTHORIZE_URL=https://id.jobadder.com/oauth2/authorize
# JOBADDER_TOKEN_URL=https://id.jobadder.com/oauth2/token
```

