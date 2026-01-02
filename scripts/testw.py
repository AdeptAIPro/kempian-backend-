# """
# Test script to check if local Redis is running and accessible
# """
# import redis
# import sys
# import os

# def test_local_redis():
#     """Test connection to local Redis"""
#     print("=" * 60)
#     print("Testing Local Redis Connection")
#     print("=" * 60)
    
#     # Test 1: Check if Redis is running on default port
#     print("\n1. Testing connection to localhost:6379...")
#     try:
#         client = redis.Redis(
#             host='localhost',
#             port=6379,
#             decode_responses=True,
#             socket_connect_timeout=5,
#             socket_timeout=5
#         )
#         response = client.ping()
#         if response:
#             print("   ✓ SUCCESS: Local Redis is running and accessible!")
#             print(f"   ✓ PING response: {response}")
            
#             # Get Redis info
#             info = client.info()
#             print(f"\n   Redis Version: {info.get('redis_version', 'Unknown')}")
#             print(f"   Used Memory: {info.get('used_memory_human', 'Unknown')}")
#             print(f"   Connected Clients: {info.get('connected_clients', 0)}")
            
#             # Test write/read
#             print("\n2. Testing write/read operations...")
#             test_key = "test:local_redis_check"
#             test_value = "Redis is working!"
#             client.set(test_key, test_value, ex=10)  # Expires in 10 seconds
#             retrieved = client.get(test_key)
#             if retrieved == test_value:
#                 print("   ✓ SUCCESS: Write and read operations work!")
#                 client.delete(test_key)
#             else:
#                 print(f"   ✗ FAILED: Expected '{test_value}', got '{retrieved}'")
#                 return False
            
#             print("\n" + "=" * 60)
#             print("✓ Local Redis is fully functional!")
#             print("=" * 60)
#             return True
            
#     except redis.ConnectionError as e:
#         print(f"   ✗ FAILED: Cannot connect to Redis on localhost:6379")
#         print(f"   Error: {e}")
#         print("\n   Possible reasons:")
#         print("   - Redis is not installed")
#         print("   - Redis is not running")
#         print("   - Redis is running on a different port")
#         return False
#     except Exception as e:
#         print(f"   ✗ FAILED: Unexpected error: {e}")
#         return False

# def check_environment_variables():
#     """Check current environment variables"""
#     print("\n" + "=" * 60)
#     print("Current Environment Variables")
#     print("=" * 60)
    
#     elasticache_endpoint = os.environ.get('ELASTICACHE_ENDPOINT', '')
#     elasticache_port = os.environ.get('ELASTICACHE_PORT', '')
#     elasticache_auth = os.environ.get('ELASTICACHE_AUTH_TOKEN', '')
#     redis_url = os.environ.get('REDIS_URL', '')
#     disable_redis = os.environ.get('DISABLE_REDIS', '')
#     redis_ssl = os.environ.get('REDIS_SSL', '')
    
#     print(f"\nELASTICACHE_ENDPOINT: {elasticache_endpoint if elasticache_endpoint else '(not set)'}")
#     print(f"ELASTICACHE_PORT: {elasticache_port if elasticache_port else '(not set)'}")
#     print(f"ELASTICACHE_AUTH_TOKEN: {'***SET***' if elasticache_auth else '(not set)'}")
#     print(f"REDIS_URL: {redis_url if redis_url else '(not set - will use default: redis://localhost:6379/0)'}")
#     print(f"DISABLE_REDIS: {disable_redis if disable_redis else '(not set)'}")
#     print(f"REDIS_SSL: {redis_ssl if redis_ssl else '(not set)'}")
    
#     if elasticache_endpoint:
#         print("\n⚠ WARNING: ELASTICACHE_ENDPOINT is set!")
#         print("   The application will try to connect to ElastiCache first.")
#         print("   To use local Redis, you need to:")
#         print("   1. Unset ELASTICACHE_ENDPOINT (or set it to empty string)")
#         print("   2. Optionally set REDIS_URL=redis://localhost:6379/0")
#     else:
#         print("\n✓ No ElastiCache endpoint set - will use local Redis")

# def show_configuration_instructions():
#     """Show how to configure for local Redis"""
#     print("\n" + "=" * 60)
#     print("Configuration Instructions for Local Redis")
#     print("=" * 60)
    
#     print("\nTo use LOCAL Redis, set these environment variables:")
#     print("\nOption 1: Using .env file (recommended)")
#     print("  Create or edit backend/.env file with:")
#     print("  ---")
#     print("  # Disable ElastiCache")
#     print("  ELASTICACHE_ENDPOINT=")
#     print("  ELASTICACHE_PORT=")
#     print("  ELASTICACHE_AUTH_TOKEN=")
#     print("  ")
#     print("  # Use local Redis")
#     print("  REDIS_URL=redis://localhost:6379/0")
#     print("  REDIS_SSL=false")
#     print("  DISABLE_REDIS=false")
#     print("  ---")
    
#     print("\nOption 2: Using command line (Windows)")
#     print("  set ELASTICACHE_ENDPOINT=")
#     print("  set REDIS_URL=redis://localhost:6379/0")
#     print("  python main.py")
    
#     print("\nOption 3: Using command line (Linux/Mac)")
#     print("  export ELASTICACHE_ENDPOINT=")
#     print("  export REDIS_URL=redis://localhost:6379/0")
#     print("  python main.py")

# if __name__ == "__main__":
#     # Check environment variables first
#     check_environment_variables()
    
#     # Test local Redis
#     redis_working = test_local_redis()
    
#     # Show configuration instructions
#     show_configuration_instructions()
    
#     if not redis_working:
#         print("\n" + "=" * 60)
#         print("Next Steps:")
#         print("=" * 60)
#         print("\n1. Install Redis (if not installed):")
#         print("   Windows: Download from https://github.com/microsoftarchive/redis/releases")
#         print("   Or use Docker: docker run -d -p 6379:6379 redis:latest")
#         print("\n2. Start Redis:")
#         print("   Windows: redis-server.exe")
#         print("   Linux/Mac: redis-server")
#         print("   Docker: docker-compose -f docker-compose.local.yml up -d")
#         print("\n3. Run this test again: python test_local_redis.py")
#         sys.exit(1)
#     else:
#         print("\n✓ You're ready to use local Redis!")
#         print("  Make sure to configure your environment variables as shown above.")
#         sys.exit(0)


import time
import hmac
import base64
import requests
from hashlib import sha256

API_KEY = "peopleconnect_kempian_api_stg"
API_SECRET = "92ad77d0d6a919fcecdfa1ac5981412b"

# Jobvite Stage Base URL
BASE_URL = "https://api.jvistg2.com/api/v2/job?count=1"

def generate_hmac_headers(api_key, api_secret):
    epoch = int(time.time())
    to_hash = f"{api_key}|{epoch}"
    signature = hmac.new(
        api_secret.encode("utf-8"),
        to_hash.encode("utf-8"),
        sha256
    ).digest()

    signature_b64 = base64.b64encode(signature).decode("utf-8")

    return {
        "Content-Type": "application/json",
        "X-JVI-API": api_key,
        "X-JVI-SIGN": signature_b64,
        "X-JVI-EPOCH": str(epoch),
    }

def test_jobvite_credentials():
    headers = generate_hmac_headers(API_KEY, API_SECRET)

    print("=== HEADERS USED ===")
    for k, v in headers.items():
        if k == "X-JVI-SIGN":
            print(f"{k}: {v[:12]}... (masked)")
        else:
            print(f"{k}: {v}")

    print("\n=== SENDING REQUEST ===")
    try:
        r = requests.get(BASE_URL, headers=headers, timeout=20)
        print("Status Code:", r.status_code)
        print("Response Snippet:", r.text[:300])
    except Exception as e:
        print("Network Error:", str(e))

if __name__ == "__main__":
    test_jobvite_credentials()
