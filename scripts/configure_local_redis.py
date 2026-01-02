"""
Quick script to configure the app to use local Redis
This sets REDIS_URL in the environment, which takes priority over ELASTICACHE_ENDPOINT
"""
import os
from pathlib import Path

def configure_local_redis():
    """Configure environment to use local Redis"""
    backend_dir = Path(__file__).parent
    env_file = backend_dir / '.env'
    
    print("=" * 60)
    print("Configuring Local Redis")
    print("=" * 60)
    
    # Check if .env file exists
    if env_file.exists():
        print(f"\n✓ Found .env file: {env_file}")
        
        # Read current .env
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Check if REDIS_URL is already set
        redis_url_set = False
        elasticache_commented = False
        
        new_lines = []
        for line in lines:
            if line.strip().startswith('REDIS_URL='):
                # Update existing REDIS_URL
                new_lines.append('REDIS_URL=redis://localhost:6379/0\n')
                redis_url_set = True
                print("  ✓ Updated REDIS_URL to use local Redis")
            elif line.strip().startswith('#REDIS_URL='):
                # Uncomment and set
                new_lines.append('REDIS_URL=redis://localhost:6379/0\n')
                redis_url_set = True
                print("  ✓ Uncommented and set REDIS_URL to use local Redis")
            elif line.strip().startswith('ELASTICACHE_ENDPOINT='):
                # Comment out ElastiCache endpoint
                new_lines.append('#ELASTICACHE_ENDPOINT=' + line.split('=', 1)[1])
                elasticache_commented = True
                print("  ✓ Commented out ELASTICACHE_ENDPOINT")
            else:
                new_lines.append(line)
        
        # Add REDIS_URL if not found
        if not redis_url_set:
            new_lines.append('\n# Local Redis Configuration\n')
            new_lines.append('REDIS_URL=redis://localhost:6379/0\n')
            print("  ✓ Added REDIS_URL to use local Redis")
        
        # Write back
        with open(env_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"\n✓ Configuration updated in {env_file}")
        print("\nNext steps:")
        print("  1. Restart your backend: python main.py")
        print("  2. Check logs for: 'Using REDIS_URL: redis://localhost:6379/0'")
        print("  3. Verify connection: python test_local_redis.py")
        
    else:
        print(f"\n⚠ No .env file found at {env_file}")
        print("Creating new .env file with local Redis configuration...")
        
        # Create new .env file
        env_content = """# Local Redis Configuration
REDIS_URL=redis://localhost:6379/0

# ElastiCache Configuration (commented out - using local Redis)
# ELASTICACHE_ENDPOINT=master.redis-cache-server.wwhabv.aps1.cache.amazonaws.com:6379
# ELASTICACHE_PORT=6379
# ELASTICACHE_AUTH_TOKEN=AdeptAiPro_2025_Redis
# REDIS_SSL=false

# Disable Redis (set to 'true' to use in-memory cache only)
DISABLE_REDIS=false
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✓ Created .env file at {env_file}")
        print("\nNext steps:")
        print("  1. Restart your backend: python main.py")
        print("  2. Check logs for: 'Using REDIS_URL: redis://localhost:6379/0'")
        print("  3. Verify connection: python test_local_redis.py")

if __name__ == "__main__":
    configure_local_redis()

