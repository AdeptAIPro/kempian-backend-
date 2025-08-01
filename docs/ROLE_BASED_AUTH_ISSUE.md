# Role-Based Authentication Issue Analysis

## 🚨 **Problem Identified**

**Issue**: When a user signs up as an `admin` role but tries to login as a `job_seeker` role, they get logged in successfully but their role remains the same (admin).

## 🔍 **Root Cause Analysis**

### **1. Backend Login Flow (`/auth/login`)**

The regular login endpoint **does NOT** check or validate the requested role:

```python
# backend/app/auth/routes.py:172-220
@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    # ❌ NO ROLE VALIDATION - Only email/password checked
    
    try:
        tokens = cognito_login(email, password)
        # Fetch user attributes from Cognito
        user_info = cognito_client.admin_get_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email
        )
        attrs = {attr['Name']: attr['Value'] for attr in user_info['UserAttributes']}
        
        # ❌ ROLE IS TAKEN DIRECTLY FROM COGNITO - No validation
        user = {
            "role": attrs.get("custom:role", ""),  # ❌ Always returns stored role
            "userType": attrs.get("custom:user_type", attrs.get("custom:role", ""))
        }
```

### **2. Social Login Flow (`/auth/cognito-social-login`)**

The social login endpoint **DOES** handle role changes:

```python
# backend/app/auth/routes.py:249-410
@auth_bp.route('/cognito-social-login', methods=['POST'])
def cognito_social_login():
    # ✅ Role validation and updating logic exists
    role_from_state = None
    if state:
        try:
            state_data = json.loads(base64.b64decode(state).decode('utf-8'))
            role_from_state = state_data.get('role')
        except Exception as e:
            logger.warning(f"Could not decode state parameter: {e}")
    
    # ✅ Role decision logic
    role = role_from_state or role_fallback or claims.get('custom:role', 'job_seeker')
    
    # ✅ Update Cognito if role changed
    if role_from_state and role_from_state != claims.get('custom:role'):
        try:
            cognito_admin_update_user_attributes(email, {
                "custom:role": role_from_state,
                "custom:user_type": role_from_state
            })
            role = role_from_state
            user_type = role_from_state
        except Exception as e:
            logger.error(f"Failed to update Cognito role for {email}: {e}")
```

### **3. Frontend Role Selection**

The frontend allows role selection but doesn't pass it to the regular login:

```typescript
// src/hooks/use-auth.ts:130-160
login: async (email, password) => {
  // ❌ NO ROLE PARAMETER PASSED TO LOGIN
  const res = await apiLogin(email, password);
  // Role is determined by backend from Cognito attributes
}
```

## 🛠️ **Solution**

### **Option 1: Add Role Validation to Regular Login (Recommended)**

Modify the `/auth/login` endpoint to accept and validate the requested role:

```python
@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    requested_role = data.get('role')  # ✅ Add role parameter
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    try:
        tokens = cognito_login(email, password)
        user_info = cognito_client.admin_get_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email
        )
        attrs = {attr['Name']: attr['Value'] for attr in user_info['UserAttributes']}
        
        stored_role = attrs.get("custom:role", "")
        
        # ✅ Validate role change
        if requested_role and requested_role != stored_role:
            # Check if user is allowed to change to this role
            allowed_role_changes = {
                'admin': ['job_seeker', 'employee', 'recruiter', 'employer'],
                'employer': ['job_seeker', 'employee', 'recruiter'],
                'recruiter': ['job_seeker', 'employee'],
                'employee': ['job_seeker'],
                'job_seeker': []  # Job seekers can't change to other roles
            }
            
            if stored_role in allowed_role_changes and requested_role in allowed_role_changes[stored_role]:
                # Update Cognito with new role
                try:
                    cognito_admin_update_user_attributes(email, {
                        "custom:role": requested_role,
                        "custom:user_type": requested_role
                    })
                    stored_role = requested_role
                except Exception as e:
                    logger.error(f"Failed to update Cognito role for {email}: {e}")
            else:
                return jsonify({'error': f'Role change from {stored_role} to {requested_role} not allowed'}), 403
        
        user = {
            "id": attrs.get("sub"),
            "email": attrs.get("email"),
            "firstName": attrs.get("given_name", ""),
            "lastName": attrs.get("family_name", ""),
            "role": stored_role,
            "userType": attrs.get("custom:user_type", stored_role)
        }
        
        return jsonify({
            "access_token": tokens.get("AccessToken"),
            "id_token": tokens.get("IdToken"),
            "refresh_token": tokens.get("RefreshToken"),
            "user": user
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /auth/login: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 401
```

### **Option 2: Frontend Role Validation**

Add role validation on the frontend before login:

```typescript
// src/hooks/use-auth.ts
login: async (email, password, requestedRole?: string) => {
  set({ isLoading: true, error: null });
  try {
    // ✅ Pass role to backend
    const res = await apiLogin(email, password, requestedRole);
    // ... rest of login logic
  } catch (error: any) {
    // ... error handling
  }
}
```

### **Option 3: Role-Based Login Endpoints**

Create separate login endpoints for different roles:

```python
@auth_bp.route('/login/job-seeker', methods=['POST'])
def login_job_seeker():
    # Only allow job_seeker role login
    
@auth_bp.route('/login/admin', methods=['POST'])
def login_admin():
    # Only allow admin role login
```

## 🔒 **Security Considerations**

### **1. Role Change Permissions**
- **Admin**: Can change to any role
- **Employer**: Can change to job_seeker, employee, recruiter
- **Recruiter**: Can change to job_seeker, employee
- **Employee**: Can change to job_seeker
- **Job Seeker**: Cannot change to other roles

### **2. Audit Trail**
Add logging for role changes:

```python
logger.info(f"Role change requested: {email} from {stored_role} to {requested_role}")
```

### **3. Rate Limiting**
Add rate limiting for role change attempts:

```python
@limiter.limit("5 role changes per hour")
@auth_bp.route('/login', methods=['POST'])
def login():
    # ... role change logic
```

## 📋 **Implementation Steps**

### **Step 1: Update Backend Login Endpoint**
1. Add role parameter to login request
2. Implement role validation logic
3. Add role change functionality
4. Add proper error handling

### **Step 2: Update Frontend**
1. Pass selected role to login API
2. Handle role change errors
3. Update UI to show role restrictions

### **Step 3: Add Role Change API**
1. Create dedicated endpoint for role changes
2. Add proper validation and permissions
3. Implement audit logging

### **Step 4: Testing**
1. Test role change scenarios
2. Test permission restrictions
3. Test error handling
4. Test audit logging

## 🎯 **Recommended Solution**

**Use Option 1** - Add role validation to the regular login endpoint because:

1. ✅ **Consistent**: Same endpoint handles all login scenarios
2. ✅ **Secure**: Role changes happen server-side with proper validation
3. ✅ **User-friendly**: Users can change roles during login
4. ✅ **Maintainable**: Single point of authentication logic

## 🚨 **Immediate Fix**

The quickest fix is to modify the login endpoint to accept a role parameter and validate it against the stored role in Cognito, similar to how the social login endpoint works. 