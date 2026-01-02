// Test Authentication Debug Script
// Run this in the browser console to test authentication flow

console.log('🧪 Testing Authentication Debug Messages...');

// Test 1: Check current authentication state
console.log('=== Current Authentication State ===');
console.log('localStorage auth_token:', localStorage.getItem('auth_token') ? 'Present' : 'Not found');
console.log('localStorage user:', localStorage.getItem('user') ? 'Present' : 'Not found');
console.log('sessionStorage access_token:', sessionStorage.getItem('access_token') ? 'Present' : 'Not found');

// Test 2: Parse current user data if available
const userStr = localStorage.getItem('user');
if (userStr) {
  try {
    const user = JSON.parse(userStr);
    console.log('Current user data:', {
      email: user.email,
      role: user.role,
      userType: user.userType,
      id: user.id
    });
  } catch (e) {
    console.error('Error parsing user data:', e);
  }
}

// Test 3: Parse JWT token if available
const token = localStorage.getItem('auth_token');
if (token) {
  try {
    const payload = JSON.parse(atob(token.split('.')[1] || ''));
    console.log('JWT token payload:', {
      email: payload.email,
      sub: payload.sub,
      role: payload.custom_role || payload.role,
      userType: payload.custom_user_type || payload.user_type
    });
  } catch (e) {
    console.error('Error parsing JWT token:', e);
  }
}

// Test 4: Clear session function
window.clearAuthSession = function() {
  console.log('🧹 Clearing authentication session...');
  localStorage.removeItem('auth_token');
  localStorage.removeItem('user');
  localStorage.removeItem('social_login_role');
  sessionStorage.removeItem('access_token');
  sessionStorage.removeItem('id_token');
  sessionStorage.removeItem('refresh_token');
  sessionStorage.removeItem('user');
  console.log('✅ Session cleared. Please refresh the page.');
};

console.log('=== Test Complete ===');
console.log('To clear session, run: clearAuthSession()');
console.log('To test login, go to the login page and watch the console for debug messages.');
