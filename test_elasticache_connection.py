"""
Test script for AWS ElastiCache connection
Run this to verify your ElastiCache setup is working correctly
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def print_test(test_name):
    """Print test name"""
    print(f"\n{'─'*60}")
    print(f" TEST: {test_name}")
    print(f"{'─'*60}")

def test_environment_variables():
    """Test 1: Check environment variables"""
    print_test("Environment Variables")
    
    redis_url = os.environ.get('REDIS_URL')
    
    if not redis_url:
        print("❌ REDIS_URL not found in environment variables")
        print("\nPlease set REDIS_URL in your .env file or environment")
        print("Example: REDIS_URL=redis://your-endpoint.cache.amazonaws.com:6379/0")
        return False
    
    print(f"✅ REDIS_URL found")
    print(f"   Value: {redis_url}")
    
    # Check if it's AWS ElastiCache (has .cache.amazonaws.com)
    if '.cache.amazonaws.com' in redis_url:
        print("✅ Using AWS ElastiCache")
    elif 'localhost' in redis_url or '127.0.0.1' in redis_url:
        print("⚠️  Using local Redis (not AWS ElastiCache)")
    else:
        print("⚠️  Unknown Redis host")
    
    return True

def test_redis_connection():
    """Test 2: Test Redis connection"""
    print_test("Redis Connection")
    
    try:
        import redis
    except ImportError:
        print("❌ redis package not installed")
        print("Install with: pip install redis")
        return False
    
    redis_url = os.environ.get('REDIS_URL')
    
    try:
        # Parse REDIS_URL to extract host, port, etc.
        r = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5)
        
        # Test ping
        result = r.ping()
        print(f"✅ Connection successful")
        print(f"   Ping result: {result}")
        
        return r
        
    except redis.exceptions.AuthenticationError as e:
        print("❌ Authentication failed")
        print(f"   Error: {e}")
        print("\n   Solutions:")
        print("   1. Check if AUTH token is correct")
        print("   2. Verify REDIS_URL format:")
        print('      REDIS_URL=redis://:your-token@endpoint:6379/0')
        return None
        
    except redis.exceptions.ConnectionError as e:
        print("❌ Connection failed")
        print(f"   Error: {e}")
        print("\n   Solutions:")
        print("   1. Check security group allows port 6379")
        print("   2. Verify VPC configuration")
        print("   3. Check endpoint URL is correct")
        print("   4. Ensure app and cache are in same VPC")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def test_basic_operations(r):
    """Test 3: Test basic Redis operations"""
    print_test("Basic Operations")
    
    if not r:
        print("⚠️  Skipping - connection not available")
        return False
    
    try:
        # Test SET
        r.set('test:key', 'test_value')
        print("✅ SET operation successful")
        
        # Test GET
        value = r.get('test:key')
        if value == 'test_value':
            print("✅ GET operation successful")
            print(f"   Retrieved: {value}")
        else:
            print(f"❌ GET returned unexpected value: {value}")
            return False
        
        # Test DELETE
        result = r.delete('test:key')
        if result == 1:
            print("✅ DELETE operation successful")
        else:
            print(f"❌ DELETE failed: {result}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic operations: {e}")
        return False

def test_server_info(r):
    """Test 4: Get server information"""
    print_test("Server Information")
    
    if not r:
        print("⚠️  Skipping - connection not available")
        return False
    
    try:
        info = r.info('server')
        
        print("✅ Server info retrieved:")
        print(f"   Redis Version: {info.get('redis_version', 'Unknown')}")
        print(f"   Server OS: {info.get('os', 'Unknown')}")
        print(f"   TCP Port: {info.get('tcp_port', 'Unknown')}")
        print(f"   Server Uptime: {info.get('uptime_in_seconds', 0)} seconds")
        
        # Check if it's AWS ElastiCache
        if 'ElastiCache' in str(info):
            print("✅ AWS ElastiCache confirmed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting server info: {e}")
        return False

def test_cache_manager():
    """Test 5: Test Kempian Cache Manager"""
    print_test("Kempian Cache Manager")
    
    try:
        # Import cache manager
        from app.cache import cache_manager
        
        print("✅ Cache Manager imported successfully")
        
        # Test cache operations through cache manager
        test_data = {
            'status': 'working',
            'timestamp': datetime.now().isoformat(),
            'test': True
        }
        
        # Set
        success = cache_manager.set('test:cache_manager', test_data, ttl=60)
        if success:
            print("✅ Cache Manager SET successful")
        else:
            print("❌ Cache Manager SET failed")
            return False
        
        # Get
        retrieved = cache_manager.get('test:cache_manager')
        if retrieved and retrieved.get('test') is True:
            print("✅ Cache Manager GET successful")
            print(f"   Retrieved data: {retrieved}")
        else:
            print("❌ Cache Manager GET returned unexpected data")
            return False
        
        # Test Redis availability
        if hasattr(cache_manager, 'redis_available'):
            if cache_manager.redis_available:
                print("✅ Redis available and working")
            else:
                print("⚠️  Redis not available, using in-memory cache only")
        
        # Get stats
        try:
            stats = cache_manager.get_stats()
            if stats:
                print("✅ Cache stats:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
        except:
            print("⚠️  Could not retrieve detailed stats (might be using in-memory cache)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import cache manager: {e}")
        print("   Make sure you're in the backend directory")
        return False
        
    except Exception as e:
        print(f"❌ Error testing cache manager: {e}")
        return False

def test_performance(r):
    """Test 6: Test performance"""
    print_test("Performance Test")
    
    if not r:
        print("⚠️  Skipping - connection not available")
        return False
    
    try:
        import time
        
        # Test write performance
        start = time.time()
        for i in range(100):
            r.set(f'test:perf:{i}', f'value_{i}')
        write_time = time.time() - start
        
        print(f"✅ Wrote 100 keys in {write_time:.3f}s")
        print(f"   Avg: {write_time/100*1000:.2f}ms per operation")
        
        # Test read performance
        start = time.time()
        for i in range(100):
            _ = r.get(f'test:perf:{i}')
        read_time = time.time() - start
        
        print(f"✅ Read 100 keys in {read_time:.3f}s")
        print(f"   Avg: {read_time/100*1000:.2f}ms per operation")
        
        # Cleanup
        for i in range(100):
            r.delete(f'test:perf:{i}')
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def main():
    """Run all tests"""
    print_header("AWS ElastiCache Connection Test")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: Environment variables
    if not test_environment_variables():
        print("\n❌ FAILED: Environment not properly configured")
        return
    
    # Test 2: Redis connection
    r = test_redis_connection()
    results.append(("Redis Connection", r is not None))
    
    if not r:
        print("\n⚠️  Cannot continue with other tests - connection failed")
        print("\nPlease fix the connection issue and try again")
        return
    
    # Test 3: Basic operations
    results.append(("Basic Operations", test_basic_operations(r)))
    
    # Test 4: Server info
    results.append(("Server Info", test_server_info(r)))
    
    # Test 5: Cache Manager
    results.append(("Cache Manager", test_cache_manager()))
    
    # Test 6: Performance
    results.append(("Performance", test_performance(r)))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your AWS ElastiCache setup is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

