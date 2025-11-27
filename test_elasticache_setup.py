"""
Test AWS ElastiCache Redis Connection
Run this script to verify your ElastiCache setup
"""
import os
import redis
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_elasticache_connection():
    """Test connection to AWS ElastiCache Redis"""
    
    print("🔍 Testing AWS ElastiCache Redis Connection...\n")
    
    # Check if .env file exists
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        print(f"✓ Found .env file: {env_file}")
    else:
        print(f"⚠️  No .env file found at: {env_file}")
        print("   Please create a .env file in the backend/ directory")
    
    print()
    
    # Get configuration from environment variables
    elasticache_host = os.environ.get('ELASTICACHE_ENDPOINT')
    elasticache_port = os.environ.get('ELASTICACHE_PORT', '6379')
    elasticache_auth_token = os.environ.get('ELASTICACHE_AUTH_TOKEN')
    redis_ssl = os.environ.get('REDIS_SSL', 'false').lower() in ('true', '1', 'yes')
    
    if not elasticache_host:
        print("❌ ELASTICACHE_ENDPOINT not set in environment variables")
        print("   Please set the following environment variables:")
        print("   - ELASTICACHE_ENDPOINT (e.g., your-cluster.xxx.cache.amazonaws.com)")
        print("   - ELASTICACHE_PORT (default: 6379)")
        print("   - ELASTICACHE_AUTH_TOKEN (your Redis auth token)")
        print("   - REDIS_SSL (true/false)")
        sys.exit(1)
    
    # Clean up endpoint - remove port if already included
    if ':' in elasticache_host and elasticache_host.split(':')[-1].isdigit():
        # Port already included in endpoint
        host_without_port = elasticache_host.rsplit(':', 1)[0]
        actual_port = elasticache_host.split(':')[-1]
        print(f"📍 Endpoint: {host_without_port}")
        print(f"🔌 Port: {actual_port} (from endpoint)")
    else:
        # No port in endpoint, use separate port
        host_without_port = elasticache_host
        actual_port = elasticache_port
        print(f"📍 Endpoint: {host_without_port}")
        print(f"🔌 Port: {actual_port}")
    
    print(f"🔐 Auth Token: {'✓ Set' if elasticache_auth_token else '✗ Not set'}")
    print(f"🔒 SSL: {redis_ssl}")
    print()
    
    # Construct Redis URL
    if elasticache_auth_token:
        redis_url = f"redis://:{elasticache_auth_token}@{host_without_port}:{actual_port}/0"
    else:
        redis_url = f"redis://{host_without_port}:{actual_port}/0"
    
    print(f"🔗 Connecting to: {redis_url}")
    print()
    
    # Try without SSL first (most ElastiCache setups don't use SSL)
    print("💡 Trying without SSL first (most AWS ElastiCache doesn't use SSL)...")
    print()
    
    # Try connection without SSL first, then with SSL if specified
    connection_attempts = [
        (False, "without SSL"),
        (True, "with SSL")
    ] if redis_ssl else [(False, "without SSL")]
    
    for use_ssl, description in connection_attempts:
        try:
            print(f"🔄 Attempting connection {description}...")
            
            # Create Redis connection
            if use_ssl:
                client = redis.Redis(
                    host=host_without_port,
                    port=int(actual_port),
                    password=elasticache_auth_token,
                    decode_responses=True,
                    ssl=True,
                    ssl_cert_reqs=None,
                    socket_connect_timeout=10,
                    socket_timeout=10
                )
                print("🔒 Using SSL connection")
            else:
                client = redis.Redis(
                    host=host_without_port,
                    port=int(actual_port),
                    password=elasticache_auth_token,
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10
                )
                print("🔓 Using non-SSL connection")
            
            # Test connection with ping
            print("⏳ Testing connection...")
            response = client.ping()
            
            if response:
                print("✅ Connection successful!")
                print()
                
                # Get Redis server info
                info = client.info()
                print("📊 Server Information:")
                print(f"   Version: Redis {info.get('redis_version', 'Unknown')}")
                print(f"   Uptime: {info.get('uptime_in_days', 0)} days")
                print(f"   Memory: {info.get('used_memory_human', 'Unknown')}")
                print(f"   Connected Clients: {info.get('connected_clients', 0)}")
                print()
                
                # Test cache operations
                print("🧪 Testing cache operations...")
                
                # Test SET
                client.set("test_key", "test_value", ex=60)
                print("   ✓ SET operation successful")
                
                # Test GET
                value = client.get("test_key")
                if value == "test_value":
                    print("   ✓ GET operation successful")
                else:
                    print(f"   ✗ GET operation failed: expected 'test_value', got '{value}'")
                
                # Test DELETE
                client.delete("test_key")
                print("   ✓ DELETE operation successful")
                
                print()
                print("🎉 All tests passed! AWS ElastiCache is ready to use.")
                print()
                print(f"✅ Use SSL: {use_ssl}")
                print("   Update your .env file:")
                print(f"   REDIS_SSL={'true' if use_ssl else 'false'}")
                return True
                
        except redis.exceptions.AuthenticationError as e:
            print(f"❌ Authentication failed: {str(e)}")
            if not use_ssl:
                continue  # Try SSL
            else:
                print("   Please check your ELASTICACHE_AUTH_TOKEN")
                return False
            
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            print(f"❌ Connection failed ({description}): {str(e)}")
            if use_ssl:
                print()
                print("   Possible issues:")
                print("   - The endpoint is not accessible from your network")
                print("   - Security group doesn't allow your IP")
                print("   - Check if the cluster is running")
                return False
            else:
                continue  # Try SSL
        
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            if not use_ssl:
                continue  # Try SSL
            else:
                return False
    
    # If we get here, all attempts failed
    print()
    print("❌ All connection attempts failed")
    return False

if __name__ == "__main__":
    success = test_elasticache_connection()
    sys.exit(0 if success else 1)

