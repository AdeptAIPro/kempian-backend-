# Redis Setup Guide for Kempian Backend

## Why Redis is Needed
- **Caching**: Stores search results, user profiles, and analytics data
- **Performance**: Reduces database load and improves response times
- **Sessions**: Manages user sessions and authentication tokens
- **Background Jobs**: Handles async tasks and job queues

## Local Development Setup

### Option 1: Docker (Recommended - Easiest)
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
# Then run Redis container:
docker run -d -p 6379:6379 --name kempian-redis redis:latest

# To stop Redis:
docker stop kempian-redis

# To start Redis again:
docker start kempian-redis
```

### Option 2: Windows with Chocolatey
```powershell
# Install Chocolatey (if not already installed)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Redis
choco install redis-64

# Start Redis server
redis-server

# In another terminal, test Redis:
redis-cli ping
# Should return: PONG
```

### Option 3: WSL (Windows Subsystem for Linux)
```bash
# Install WSL
wsl --install

# In WSL terminal:
sudo apt update
sudo apt install redis-server
sudo service redis-server start

# Test Redis:
redis-cli ping
```

### Option 4: Download Redis for Windows
1. Download from: https://github.com/microsoftarchive/redis/releases
2. Extract the zip file
3. Run `redis-server.exe` from the extracted folder
4. Keep the terminal open while using the app

## Production Setup

### Option 1: AWS ElastiCache (Recommended for AWS)

**📚 Complete Setup Guide: [AWS_ELASTICACHE_COMPLETE_SETUP.md](./AWS_ELASTICACHE_COMPLETE_SETUP.md)**  
*Step-by-step instructions for setting up AWS ElastiCache in AWS Console*

**📖 Migration Guide: [AWS_ELASTICACHE_MIGRATION.md](./AWS_ELASTICACHE_MIGRATION.md)**  
*Quick overview and migration strategy*

```bash
# Create Redis cluster
aws elasticache create-cache-cluster \
    --cache-cluster-id kempian-redis-prod \
    --cache-node-type cache.t3.micro \
    --engine redis \
    --num-cache-nodes 1 \
    --region ap-south-1

# Get endpoint:
aws elasticache describe-cache-clusters \
    --cache-cluster-id kempian-redis-prod \
    --show-cache-node-info
```

**Quick Start with AWS:**
1. Go to AWS Console → ElastiCache
2. Create Redis cluster (takes ~15 minutes)
3. Get endpoint and update `REDIS_URL`
4. No code changes needed! ✅

### Option 2: Redis Cloud (Managed Service)
1. Sign up at https://redis.com/redis-enterprise-cloud/
2. Create a free database (30MB)
3. Get connection details from dashboard
4. Use the provided URL in your environment variables

### Option 3: Self-hosted on Linux Server
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
# Set: bind 0.0.0.0 (for external access)
# Set: requirepass your_strong_password
sudo systemctl restart redis-server

# CentOS/RHEL
sudo yum install redis
sudo systemctl start redis
sudo systemctl enable redis
```

## Environment Configuration

### For Local Development
Create a `.env` file in the backend directory:
```env
# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Other required variables
SECRET_KEY=your-secret-key-here
DATABASE_URL=mysql+pymysql://username:password@localhost:3306/kempianDB
AWS_REGION=ap-south-1
```

### For Production
```env
# Redis Configuration (use your actual Redis endpoint)
REDIS_URL=redis://your-redis-endpoint:6379/0
# Or with password:
REDIS_URL=redis://:password@your-redis-endpoint:6379/0

# AWS ElastiCache example:
REDIS_URL=redis://kempian-redis-prod.xxxxx.cache.amazonaws.com:6379/0

# Redis Cloud example:
REDIS_URL=redis://username:password@redis-12345.c1.us-east1-2.gce.cloud.redislabs.com:12345
```

## Testing Redis Connection

### Test from Python
```python
import redis
import os

# Test connection
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
r = redis.from_url(redis_url, decode_responses=True)

try:
    r.ping()
    print("✅ Redis connection successful!")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
```

### Test from Command Line
```bash
# If Redis is running locally
redis-cli ping
# Should return: PONG

# Test with specific host/port
redis-cli -h localhost -p 6379 ping
```

## Troubleshooting

### Common Issues:

1. **Connection Refused (Error 10061)**
   - Redis is not running
   - Wrong port (should be 6379)
   - Firewall blocking connection

2. **Authentication Failed**
   - Check if Redis requires password
   - Update REDIS_URL with password

3. **Memory Issues**
   - Redis running out of memory
   - Check Redis memory usage: `redis-cli info memory`

### Redis Commands for Debugging:
```bash
# Check if Redis is running
redis-cli ping

# Check Redis info
redis-cli info

# Check memory usage
redis-cli info memory

# List all keys
redis-cli keys "*"

# Clear all cache
redis-cli flushall

# Monitor Redis commands
redis-cli monitor
```

## Performance Optimization

### Redis Configuration for Production:
```conf
# In redis.conf
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Application Configuration:
```python
# In your app config
REDIS_CONNECTION_POOL = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=20,
    retry_on_timeout=True
)
```

## Monitoring

### Redis Monitoring Tools:
1. **RedisInsight** - Official Redis GUI
2. **Redis Commander** - Web-based admin
3. **Redis Desktop Manager** - Desktop client

### Health Checks:
```python
def check_redis_health():
    try:
        r = redis.from_url(os.environ.get('REDIS_URL'))
        r.ping()
        return True
    except:
        return False
```
