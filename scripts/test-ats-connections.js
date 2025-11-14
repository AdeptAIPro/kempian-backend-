// Test script to check ATS connection status
console.log('=== ATS Connection Test ===');

// Check localStorage for integration data
console.log('Checking localStorage for integration data...');
const integrationKeys = [];
for (let i = 0; i < localStorage.length; i++) {
  const key = localStorage.key(i);
  if (key && key.startsWith('integration_')) {
    integrationKeys.push(key);
    const value = localStorage.getItem(key);
    console.log(`${key}:`, value);
  }
}

console.log('Found integration keys:', integrationKeys);

// Check for specific ATS connections
const atsKeys = ['workday', 'bamboohr', 'greenhouse', 'lever', 'smartrecruiters', 'icims', 'ceipal', 'taleo', 'stafferlink'];
console.log('\nChecking specific ATS connections...');
atsKeys.forEach(atsKey => {
  const integrationKey = `integration_${atsKey}`;
  const storedData = localStorage.getItem(integrationKey);
  if (storedData) {
    try {
      const data = JSON.parse(storedData);
      console.log(`${atsKey}: ${data.connected ? 'CONNECTED' : 'NOT CONNECTED'}`);
    } catch (error) {
      console.log(`${atsKey}: Error parsing data - ${error.message}`);
    }
  } else {
    console.log(`${atsKey}: No data found`);
  }
});

// Check for any other connection patterns
console.log('\nChecking for other connection patterns...');
const allKeys = [];
for (let i = 0; i < localStorage.length; i++) {
  allKeys.push(localStorage.key(i));
}
console.log('All localStorage keys:', allKeys);

// Look for any keys that might contain connection data
const connectionKeys = allKeys.filter(key => 
  key && (
    key.includes('connection') || 
    key.includes('connected') || 
    key.includes('ats') || 
    key.includes('integration')
  )
);
console.log('Potential connection keys:', connectionKeys);
