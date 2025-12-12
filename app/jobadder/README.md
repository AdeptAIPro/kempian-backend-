# JobAdder OAuth Integration - Complete Fix Guide

## ✅ Fixed Issues

1. **OAuth Endpoints**: Using `/oauth2/authorize` and `/oauth2/token` endpoints
2. **Environment Support**: Added sandbox vs production environment detection
3. **Health Check**: Added `/integrations/jobadder/health` endpoint for testing
4. **Token Refresh**: Added explicit `/integrations/jobadder/oauth/refresh` endpoint
5. **Redirect URI**: Fixed to ensure exact match (no trailing slashes, proper encoding)
6. **Error Handling**: Improved logging and error messages throughout OAuth flow

## 🔧 Environment Variables

### Required
- `JOBADDER_OAUTH_REDIRECT_URI`: The exact redirect URI registered in JobAdder Developer Portal
  - Example: `https://api.kempian.ai/integrations/jobadder/oauth/callback`
  - **Must match exactly** (case-sensitive, no trailing slash)

### Optional
- `JOBADDER_ENVIRONMENT`: Set to `sandbox` or `production` (default: `production`)
- `JOBADDER_AUTHORIZE_URL`: Override authorize URL (default: auto-detected based on environment)
- `JOBADDER_TOKEN_URL`: Override token URL (default: auto-detected based on environment)
- `JOBADDER_API_BASE_URL`: Override API base URL (default: auto-detected based on environment)
- `JOBADDER_DEFAULT_SCOPE`: OAuth scopes (default: `read write offline_access`)
- `FRONTEND_URL`: Frontend URL for OAuth callback redirect (default: `http://localhost:5173`)
- `JOBADDER_OAUTH_CALLBACK_REDIRECT`: Frontend path after OAuth (default: `/integrations/jobadder/connect`)

## 📋 JobAdder Developer Portal Setup

1. Go to https://developer.jobadder.com → **My Applications**
2. Create or edit your application:
   - **Grant Type**: Must be **Authorization Code** (not Client Credentials)
   - **Scopes**: Add `read`, `write`, and `offline_access` (or `jobadder.api` and `offline_access` if your app requires it)
   - **Redirect URI**: Must be exactly `https://api.kempian.ai/integrations/jobadder/oauth/callback`
     - No trailing slash
     - Case-sensitive
     - Must use HTTPS in production
3. Note your **Client ID** and **Client Secret**
4. Check if your app is in **Sandbox** or **Production** environment

## 🧪 Testing Endpoints

### Health Check
```bash
curl https://api.kempian.ai/integrations/jobadder/health
```
Expected: `{"ok": true, "service": "jobadder", ...}`

### Connect (Initiate OAuth)
```bash
POST /integrations/jobadder/connect
{
  "clientId": "your_client_id",
  "clientSecret": "your_client_secret"
}
```
Returns: `{"authUrl": "...", "redirectUri": "..."}`

### Status Check
```bash
GET /integrations/jobadder/status
```
Returns: Connection status and account info

### Manual Token Refresh
```bash
POST /integrations/jobadder/oauth/refresh
```
Refreshes the access token using the stored refresh token

## 🔍 Troubleshooting

### 404 on Authorize Page
- **Cause**: Wrong environment (sandbox client on production URL or vice versa)
- **Fix**: Set `JOBADDER_ENVIRONMENT=sandbox` if using sandbox, or ensure client is in production

### 404 on Callback
- **Cause**: Route not accessible or Nginx/proxy issue
- **Fix**: 
  1. Test: `curl https://api.kempian.ai/integrations/jobadder/health`
  2. Check Nginx config has proper `proxy_pass` with trailing slash
  3. Verify route is registered in Flask app

### "invalid_grant" Error
- **Cause**: Redirect URI mismatch
- **Fix**: 
  1. Copy exact redirect URI from JobAdder portal
  2. Set `JOBADDER_OAUTH_REDIRECT_URI` to that exact value
  3. Ensure no trailing slash, correct case, HTTPS

### Token Expired
- **Cause**: Access token expired and refresh failed
- **Fix**: 
  1. Check refresh token exists in database
  2. Try manual refresh: `POST /integrations/jobadder/oauth/refresh`
  3. If fails, user must reconnect

## 📝 OAuth Flow

1. User calls `POST /integrations/jobadder/connect` with client credentials
2. Backend stores credentials and returns `authUrl`
3. User visits `authUrl` → JobAdder login/consent page
4. JobAdder redirects to `/integrations/jobadder/oauth/callback?code=...&state=...`
5. Backend exchanges code for tokens
6. Backend fetches account info and stores tokens
7. Backend redirects user to frontend with success/error status

## 🔄 Token Refresh

Tokens are automatically refreshed when:
- Access token is missing
- Access token expires within 5 minutes
- API call returns 401 (unauthorized)

Manual refresh available via `POST /integrations/jobadder/oauth/refresh`

