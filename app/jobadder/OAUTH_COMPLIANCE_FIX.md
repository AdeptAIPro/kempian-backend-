# JobAdder OAuth2 Compliance Fix

## Issue Found
The implementation was using incorrect OAuth2 endpoints that don't match the official JobAdder documentation.

## Official Documentation Requirements
According to [JobAdder OAuth2 Authentication Documentation](https://jobadderapi.zendesk.com/hc/en-us/articles/360022196774-OAuth2-Authentication):

1. **Authorization URL**: `https://id.jobadder.com/connect/authorize`
2. **Token URL**: `https://id.jobadder.com/connect/token`

## What Was Wrong
The code was using:
- ❌ `/oauth2/authorize` instead of `/connect/authorize`
- ❌ `/oauth2/token` instead of `/connect/token`

## Changes Made

### 1. `backend/app/jobadder/auth.py`
- **Line 29-32**: Updated default endpoints from `/oauth2/authorize` and `/oauth2/token` to `/connect/authorize` and `/connect/token`
- Updated comment to reflect correct endpoints per official documentation

### 2. `backend/app/jobadder/routes.py`
- **Line 1937**: Updated default token URL in logging from `https://id.jobadder.com/oauth2/token` to `https://id.jobadder.com/connect/token`

### 3. `backend/app/jobadder/README.md`
- **Line 5**: Corrected documentation to state that endpoints use `/connect/authorize` and `/connect/token` (not `/oauth2/*`)

## Verification

### OAuth2 Flow Compliance
✅ **Authorization URL**: Now correctly uses `/connect/authorize`
✅ **Token Exchange**: Now correctly uses `/connect/token`
✅ **Token Refresh**: Uses same `/connect/token` endpoint with `grant_type=refresh_token`
✅ **Request Format**: POST with `application/x-www-form-urlencoded` (correct)
✅ **Scopes**: Default `read write offline_access` (matches documentation)
✅ **Redirect URI**: Properly URL-encoded and validated

### Other Implementation Details (Already Correct)
- ✅ Authorization code exchange includes all required parameters
- ✅ Refresh token flow properly implemented
- ✅ Token expiry handling (60 minutes) correctly implemented
- ✅ Error handling for OAuth errors
- ✅ State parameter for CSRF protection

## Testing Recommendations

1. **Test Authorization Flow**:
   - Verify that the authorization URL redirects correctly to JobAdder login
   - Confirm the redirect URI matches exactly what's registered in JobAdder Developer Portal

2. **Test Token Exchange**:
   - Verify that authorization code is successfully exchanged for access and refresh tokens
   - Confirm tokens are stored correctly in the database

3. **Test Token Refresh**:
   - Verify that refresh token is used to obtain new access tokens before expiry
   - Confirm refresh token is updated if a new one is provided

## Environment Variables
The following environment variables can still override the defaults if needed:
- `JOBADDER_AUTHORIZE_URL`: Override authorization URL (default: `https://id.jobadder.com/connect/authorize`)
- `JOBADDER_TOKEN_URL`: Override token URL (default: `https://id.jobadder.com/connect/token`)
- `JOBADDER_ENVIRONMENT`: Set to `sandbox` for sandbox environment (uses `id-sandbox.jobadder.com`)

## Notes
- Sandbox environment automatically uses `https://id-sandbox.jobadder.com/connect/authorize` and `https://id-sandbox.jobadder.com/connect/token`
- All changes maintain backward compatibility through environment variable overrides
- The implementation now fully complies with JobAdder's official OAuth2 documentation

