# API Errors Fix Summary

## Issues Identified

### 1. **404 NOT FOUND - `/api/hr/payroll-runs`**
   - **Problem**: Frontend was calling `/api/hr/payroll-runs` but backend route is `/api/hr/payruns`
   - **Status**: ✅ FIXED - Updated frontend to use correct endpoint

### 2. **401 UNAUTHORIZED Errors**
   - **Problem**: Authentication is failing for multiple endpoints:
     - `/api/hr/employees`
     - `/api/user/bank-account`
     - `/api/hr/employees/bank-accounts`
   - **Possible Causes**:
     1. Token expired or invalid
     2. Token not being sent in Authorization header
     3. Token format doesn't match backend expectations
     4. Backend running on port 8000 but frontend calling port 8081

## Root Causes

### Authentication Flow
1. Frontend stores token in `localStorage` as `auth_token`, `access_token`, or `id_token`
2. Frontend sends token in `Authorization: Bearer <token>` header
3. Backend expects JWT token with `email` field in payload
4. Backend uses `get_current_user_flexible()` which tries:
   - First: Decode as Cognito JWT
   - Fallback: Decode as base64 custom token

### Port Mismatch Issue
- Backend is running on port **8000**
- Frontend is calling **localhost:8081**
- This suggests a proxy configuration issue

## Solutions

### ✅ Fixed: URL Mismatch
- Changed `/api/hr/payroll-runs` → `/api/hr/payruns` in `PayrollPaymentPage.tsx`

### 🔧 To Fix: Authentication Issues

#### Option 1: Check Token Validity
```javascript
// In browser console, check if token exists:
localStorage.getItem('auth_token')
localStorage.getItem('access_token')
localStorage.getItem('id_token')

// Check if token is expired (decode JWT):
const token = localStorage.getItem('auth_token');
const payload = JSON.parse(atob(token.split('.')[1]));
console.log('Token expires:', new Date(payload.exp * 1000));
console.log('Current time:', new Date());
```

#### Option 2: Verify Backend Port
- Ensure backend is running on the port frontend expects (8081)
- OR update frontend proxy to point to port 8000
- Check `vite.config.js` or proxy configuration

#### Option 3: Re-authenticate
- User may need to log out and log back in to get a fresh token
- Check if token refresh mechanism is working

### 🔧 To Fix: Port Configuration

#### Check Vite Proxy (if using Vite)
```javascript
// vite.config.js
export default {
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',  // Backend port
        changeOrigin: true,
      }
    }
  }
}
```

#### Or Update Backend to Run on Port 8081
```bash
# In backend, check how port is configured
# Usually in app.run() or environment variable
```

## Testing Steps

1. **Verify Backend is Running**
   ```bash
   # Check if backend is accessible
   curl http://localhost:8000/health
   # or
   curl http://localhost:8081/api/health
   ```

2. **Test Authentication**
   ```bash
   # Get token from browser localStorage
   TOKEN="your-token-here"
   
   # Test endpoint with token
   curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/hr/employees
   ```

3. **Check Backend Logs**
   - Look for authentication errors in backend console
   - Check if `get_current_user_flexible()` is receiving the token

## Next Steps

1. ✅ Fixed URL mismatch for payroll-runs endpoint
2. ⏳ Verify token is being sent correctly
3. ⏳ Check port configuration (8000 vs 8081)
4. ⏳ Test authentication flow end-to-end
5. ⏳ Add better error logging for 401 errors

## Related Files

- `src/pages/PayrollPaymentPage.tsx` - Fixed URL
- `src/services/api.ts` - Token interceptor
- `backend/app/auth_utils.py` - Authentication logic
- `backend/app/hr/payruns.py` - Pay runs endpoint
- `backend/app/hr/employees.py` - Employees endpoint

