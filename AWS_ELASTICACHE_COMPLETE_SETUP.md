# Complete AWS ElastiCache Setup Guide for Kempian

This is a comprehensive step-by-step guide to set up AWS ElastiCache for your Kempian application.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [AWS Console Setup (Step-by-Step)](#aws-console-setup)
3. [Network Configuration](#network-configuration)
4. [Security Configuration](#security-configuration)
5. [Application Configuration](#application-configuration)
6. [Testing & Verification](#testing--verification)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### What You Need
- AWS Account with administrative access
- VPC already set up (where your application runs)
- Security group configured
- Basic knowledge of AWS Console

### Cost Estimate
- **Development**: `cache.t3.micro` (~$13/month)
- **Production**: `cache.t3.small` (~$26/month) or `cache.t3.medium` (~$52/month)
- **With Multi-AZ + backups**: Add ~20% to base cost

### Network Requirements
- Your app server and ElastiCache must be in the **same VPC** or VPCs connected via VPC Peering
- Security group must allow port **6379** (Redis default port)

---

## AWS Console Setup

### Step 1: Access ElastiCache Service

1. **Log in to AWS Console**  
   Go to: https://console.aws.amazon.com/

2. **Navigate to ElastiCache**  
   - Click **Services** (top left)
   - Type "ElastiCache" in search
   - Click **ElastiCache** service

3. **Choose Redis Cluster Type**  
   - You'll see options for "Redis" and "Memcached"
   - Click **Redis** (select "Create cluster")

---

### Step 2: Create Your Redis Cluster

#### A. Cluster Configuration

**Location**: Click "Redis" → "Create cluster" button (or use "Get started now")

**Cluster type selection**:
- Choose **Cache (Redis)** 
- Click **Create**

**Fill in the configuration:**

1. **Cluster settings**:
   ```
   Name: kempian-cache-prod
   Description: Kempian application Redis cache
   ```

2. **Engine version**:
   ```
   Engine version compatibility: Redis 7.1 (or latest available)
   ✅ Uncheck "Auto minor version upgrade" (for production stability)
   ```

3. **Node type selection**:
   ```
   Development:
   - Type: cache.t3.micro (0.6 GB, 2 vCPUs) ~$13/month
   
   Production:
   - Type: cache.t3.small (1.4 GB, 2 vCPUs) ~$26/month
   OR
   - Type: cache.t3.medium (3.1 GB, 2 vCPUs) ~$52/month
   
   Click on your selected type
   ```

4. **Number of replicas** (for high availability):
   ```
   Development: 0 replicas (1 node only)
   Production: 1-2 replicas (recommended for HA)
   
   For 1 replica: Select "1" (total: 1 primary + 1 replica)
   ```

5. **Advanced Redis settings**:
   ```
   ✅ Enable automatic failover (only if replicas > 0)
   ✅ Enable Multi-AZ (for high availability)
   ✅ Enable snapshots for backup
   ```

6. **Subnet and security**:
   ```
   Subnet group: Select your VPC subnet group
   
   Security groups:
   - Choose "None" initially (we'll configure manually)
   OR
   - Create new: "kempian-cache-sg"
   ```

#### B. Security Configuration (scroll down)

1. **Encryption** (for production):
   ```
   ✅ Enable encryption in-transit
   ✅ Enable encryption at-rest
   ```

2. **Authentication** (optional but recommended):
   ```
   ✅ Enable AUTH token
   Auto-generate token: ✅
   
   Save this token! You'll need it for REDIS_URL
   ```

3. **Log delivery**:
   ```
   Log delivery configuration: 
   ✅ Enable slow log
   Log type: Slow log
   Log format: JSON
   ```

#### C. Maintenance and backup

1. **Backup configuration**:
   ```
   ✅ Enable automatic backups
   Backup retention: 7 days (adjust as needed)
   Backup window: Select low-traffic hours
   ```

2. **Maintenance window**:
   ```
   Maintenance window: Select your preferred time
   Day: Sunday (recommended)
   Time: 3:00 AM (low traffic)
   ```

3. **Notification** (optional):
   ```
   SNS topic: Create or select topic for notifications
   ```

#### D. Tags (optional)

Add tags for organization:
```
Key: Project     Value: Kempian
Key: Environment Value: Production
Key: ManagedBy    Value: DevOps
```

#### E. Review and Create

1. **Review all settings**
2. **Click "Create cluster"**
3. **Wait 15-20 minutes** for creation

✅ **Note the cluster endpoint** (you'll need this for REDIS_URL)

---

### Step 3: Get Your Cluster Endpoint

After cluster creation:

1. **In ElastiCache dashboard**, click on your cluster name
2. **Find "Primary endpoint"** section:
   ```
   Primary endpoint: kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379
   Reader endpoint: kempian-cache-prod-ro.xxxxx.aps1.cache.amazonaws.com:6379
   ```
3. **Copy the Primary endpoint** - this is your `REDIS_URL` host

---

## Network Configuration

### Step 4: VPC and Subnet Setup

#### A. Check Your Current Setup

1. **Go to EC2** → **Instances**
2. **Find your application server**
3. **Click on it** → Click "Security" tab
4. **Note these details**:
   - VPC ID: `vpc-xxxxxxxxxx`
   - Security Group ID: `sg-xxxxxxxxxx`
   - Subnet: `subnet-xxxxxxxxxx`

#### B. Create Subnet Group (if not exists)

1. **In ElastiCache console** → Click **Subnet groups** (left sidebar)
2. **Click "Create subnet group"**
3. **Fill in**:
   ```
   Name: kempian-cache-subnet-group
   Description: Subnets for Kempian cache
   VPC ID: <select your application's VPC>
   ```
4. **Add subnets**:
   ```
   Availability Zone 1: 
   - Subnet: <select your app's subnet>
   
   Availability Zone 2 (if multi-AZ):
   - Subnet: <select another subnet in different AZ>
   ```
5. **Click "Create subnet group"**

---

### Step 5: Security Group Configuration

You need to allow your application to connect to ElastiCache.

#### Option A: Use Existing Security Group

1. **In EC2** → **Security Groups**
2. **Find your application's security group**
3. **Go to "Inbound rules"** → **Edit**
4. **Add rule**:
   ```
   Type: Custom TCP
   Port: 6379
   Source: 
   - If ElastiCache and app in SAME security group: sg-xxxxxxxxxx (this SG itself)
   - If DIFFERENT security groups: The SG ID of where your app is running
   - Or for testing: 0.0.0.0/0 (all IPs - NOT for production!)
   Description: Allow Redis access
   ```

#### Option B: Create Dedicated Security Group for ElastiCache

1. **Create new security group**:
   ```
   Name: kempian-cache-sg
   Description: Security group for Kempian Redis cache
   VPC: <select your app's VPC>
   ```

2. **Add inbound rule**:
   ```
   Type: Custom TCP
   Port: 6379
   Source: <your app server's security group ID>
   Description: Allow access from Kempian app
   ```

3. **Save**

4. **Apply to ElastiCache**:
   - Go back to ElastiCache cluster
   - Click **Modify**
   - Under **Security groups**, select `kempian-cache-sg`
   - Click **Modify**

---

## Security Configuration

### Step 6: Configure Authentication (Recommended)

If you enabled AUTH token earlier:

1. **Get the auth token** from ElastiCache cluster details:
   - Click on your cluster
   - Look for **"AUTH token"** section
   - Copy the token (looks like: `xyz123...`)

2. **Update your REDIS_URL** to include auth:
   ```env
   REDIS_URL=redis://:YOUR_AUTH_TOKEN@kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379/0
   ```

### Step 7: Enable CloudWatch Monitoring

1. **In ElastiCache console** → Click your cluster
2. **Go to "Monitoring" tab**
3. **CloudWatch metrics are automatically enabled**
4. **Set up alerts** (optional):
   - Go to **CloudWatch** → **Alarms** → **Create alarm**
   - Select metric: `CPUUtilization`
   - Threshold: > 80%
   - SNS topic: Your notification topic

---

## Application Configuration

### Step 8: Update Environment Variables

#### A. Update Your `.env` File

**For Local Development** (still using localhost):
```env
# Local development
REDIS_URL=redis://localhost:6379/0
```

**For Staging/Production**:
```env
# AWS ElastiCache
REDIS_URL=redis://kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379/0

# Or with authentication:
REDIS_URL=redis://:your-auth-token@kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379/0

# Other AWS settings
AWS_REGION=ap-south-1
```

#### B. Update AWS Systems Manager Parameter Store (If Used)

```bash
# Add parameters
aws ssm put-parameter \
    --name "/kempian/prod/redis-url" \
    --value "redis://kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379/0" \
    --type "SecureString" \
    --overwrite
```

#### C. Update Environment in Your Application Host

**For EC2/ECS**:
```bash
# SSH into your server
ssh user@your-server-ip

# Edit environment file
sudo nano /etc/environment
# OR
sudo nano /opt/kempian/.env

# Add:
REDIS_URL=redis://kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379/0

# Reload environment
source /etc/environment

# Restart application
sudo systemctl restart kempian
```

**For Elastic Beanstalk**:
```bash
# Use EB console or CLI
eb setenv REDIS_URL=redis://kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379/0
```

**For Docker/ECS**:
```yaml
# In your task definition
environment:
  - name: REDIS_URL
    value: redis://kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379/0
```

---

### Step 9: Update Application Code (If Needed)

Your code in `backend/app/cache.py` already supports this! Just ensure:

```python
# Your existing code already handles this:
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
redis_client = redis.from_url(redis_url, decode_responses=True)
```

✅ **No code changes needed!**

---

## Testing & Verification

### Step 10: Test the Connection

#### A. Test from Local Machine

```bash
# Install redis-cli (if not installed)
# Windows: Download from https://github.com/microsoftarchive/redis/releases
# Or use WSL: sudo apt install redis-tools

# Test connection
redis-cli -h kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com -p 6379 ping
# Should return: PONG

# If using AUTH:
redis-cli -h kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com -p 6379 -a your-auth-token ping
```

#### B. Test from Python

Create a test file: `test_elasticache.py`

```python
import os
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Redis URL
redis_url = os.environ.get('REDIS_URL')

if not redis_url:
    print("❌ REDIS_URL not found in environment")
    exit(1)

print(f"Connecting to: {redis_url}")

try:
    # Connect to Redis
    r = redis.from_url(redis_url, decode_responses=True)
    
    # Test ping
    result = r.ping()
    print(f"✅ Ping successful: {result}")
    
    # Test set/get
    r.set('test_key', 'test_value_from_elasticache')
    value = r.get('test_key')
    print(f"✅ Set/Get test: {value}")
    
    # Test delete
    r.delete('test_key')
    print("✅ Delete test: Success")
    
    # Get server info
    info = r.info('server')
    print(f"✅ Server: Redis {info.get('redis_version')}")
    print(f"✅ ElastiCache connection successful!")
    
except redis.exceptions.AuthenticationError:
    print("❌ Authentication failed - check your AUTH token")
except redis.exceptions.ConnectionError as e:
    print(f"❌ Connection failed: {e}")
    print("   - Check security group allows port 6379")
    print("   - Check VPC configuration")
    print("   - Verify endpoint URL")
except Exception as e:
    print(f"❌ Error: {e}")

# Test from your Kempian app
print("\n" + "="*50)
print("Testing through Kempian Cache Manager")
print("="*50)

try:
    from app.cache import cache_manager
    
    # Test cache operations
    cache_manager.set('test:kempian', {'status': 'working', 'timestamp': 'now'}, ttl=60)
    value = cache_manager.get('test:kempian')
    print(f"✅ Cache Manager test: {value}")
    
    # Get stats
    stats = cache_manager.get_stats()
    print(f"✅ Cache stats: {stats}")
    
except Exception as e:
    print(f"❌ Cache Manager error: {e}")
```

Run it:
```bash
cd backend
python test_elasticache.py
```

#### C. Test from Your Application

Start your Flask app and check logs:

```bash
# Start your application
python main_production.py

# Watch the logs
# Look for: "Redis cache connected successfully"
```

---

## Monitoring & Maintenance

### Step 11: Set Up CloudWatch Monitoring

#### A. Key Metrics to Monitor

Go to **CloudWatch** → **Metrics** → **ElastiCache**

Monitor these metrics:

1. **CPUUtilization**
   - Target: < 70% average
   - Alert if: > 80% for 5 minutes

2. **NetworkBytesIn/NetworkBytesOut**
   - Monitor throughput
   - Alert if: Sudden spike or drop

3. **CacheHitRate** (important!)
   - Target: > 70%
   - Alert if: < 50% (cache not working well)

4. **CurrConnections**
   - Watch for connection limits
   - Alert if approaching max

5. **ReplicationLag** (if using replicas)
   - Target: < 1 second
   - Alert if: > 5 seconds

#### B. Create CloudWatch Alarms

1. **Go to CloudWatch** → **Alarms** → **Create alarm**
2. **Metric selection**:
   ```
   Namespace: AWS/ElastiCache
   Metric name: CPUUtilization
   Dimensions: CacheClusterId = kempian-cache-prod
   ```
3. **Conditions**:
   ```
   Threshold: Static
   Whenever CPUUtilization is > 80
   Period: 5 minutes
   ```
4. **Configure actions**:
   ```
   SNS topic: Select or create notification topic
   Email: your-email@example.com
   ```
5. **Create alarm**

Repeat for other critical metrics.

---

### Step 12: Backup and Recovery

#### Backup Setup (Already configured earlier)

Your backups are already set up if you enabled them during cluster creation.

To verify:
1. **Go to ElastiCache** → Your cluster
2. **Click "Backups" tab**
3. You should see automatic backups listed

#### Manual Backup

Create a manual snapshot:

```bash
# Via AWS Console
ElastiCache → Snapshots → Create snapshot
Fill in:
- Snapshot name: kempian-backup-2024-01-15
- Cluster: kempian-cache-prod

# Via CLI
aws elasticache create-snapshot \
    --replication-group-id kempian-cache-prod \
    --snapshot-name kempian-backup-manual-$(date +%Y%m%d)
```

#### Restore from Backup

1. **Go to ElastiCache** → **Snapshots**
2. **Select your snapshot**
3. **Click "Restore"**
4. **Create new cluster** from snapshot

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "Connection refused" Error

**Problem**: App can't connect to ElastiCache

**Solutions**:
```bash
# Check security group
1. Go to EC2 → Security Groups
2. Find ElastiCache security group
3. Check inbound rules allow port 6379
4. Verify source is your app's security group

# Test connectivity
# From your app server:
telnet kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com 6379
# Should connect (Ctrl+] then q to quit)
```

#### 2. "AUTH failed" Error

**Problem**: Authentication token mismatch

**Solution**:
```bash
# Verify AUTH token in ElastiCache console
# Update REDIS_URL with correct token
REDIS_URL=redis://:CORRECT_TOKEN@endpoint:6379/0
```

#### 3. High Latency

**Problem**: Slow cache responses

**Solutions**:
```bash
# Check CloudWatch metrics
1. CPUUtilization - if > 70%, scale up
2. NetworkBytesIn/Out - check bandwidth
3. CurrConnections - too many connections

# Upgrade node type
ElastiCache → Modify → Change node type → Apply
```

#### 4. "Out of memory" Errors

**Problem**: Cache running out of memory

**Solution**:
```bash
# Scale up node type
# Or check cache eviction policies in parameter group
```

#### 5. Cache Not Working in App

**Debug steps**:
```bash
# 1. Check environment variable
echo $REDIS_URL

# 2. Test connection
redis-cli -h <endpoint> -p 6379 ping

# 3. Check app logs
tail -f /var/log/kempian/app.log | grep -i redis

# 4. Verify code is using cache
# Check that cache_manager.redis_available is True
```

---

## Production Checklist

Before going live, verify:

- [ ] ElastiCache cluster created with appropriate node type
- [ ] Multi-AZ enabled (for production)
- [ ] Automatic failover enabled
- [ ] Backups enabled and tested
- [ ] Security group configured correctly
- [ ] Encryption enabled (at rest and in transit)
- [ ] AUTH token configured and tested
- [ ] CloudWatch alarms configured
- [ ] Application environment variables updated
- [ ] Connection tested from application
- [ ] Cache hit rate monitored (target: > 70%)
- [ ] Disaster recovery plan documented
- [ ] Rollback plan ready (can revert to local Redis)

---

## Cost Optimization Tips

1. **Right-size your instance**
   - Start small, monitor, scale up if needed
   - Use CloudWatch metrics to guide decisions

2. **Optimize cache hit rate**
   - If hit rate < 50%, review TTLs
   - Consider caching more data

3. **Use reserved instances** (for production)
   - Save up to 40% with 1-year or 3-year commitment

4. **Monitor unused resources**
   - Delete test clusters
   - Don't leave clusters running 24/7 if not needed

5. **Enable cost alerts**
   - CloudWatch → Billing → Set up alerts
   - Get notified if costs exceed threshold

---

## Migration Strategy

### Phase 1: Development
1. Create dev ElastiCache cluster
2. Test with dev environment
3. Verify all functionality

### Phase 2: Staging
1. Create staging cluster
2. Run integration tests
3. Performance testing

### Phase 3: Production
1. Create production cluster with:
   - Multi-AZ
   - Replicas
   - Backups
   - Monitoring
2. Update REDIS_URL in production
3. Monitor closely for 48 hours
4. Compare performance vs old setup

### Rollback Plan
If issues occur:
1. Revert REDIS_URL to old Redis server
2. Restart application
3. Investigate issue
4. Fix and retry

---

## Final Configuration Summary

Your final production setup:

```env
# Environment variables
REDIS_URL=redis://kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379/0
AWS_REGION=ap-south-1

# Application will automatically use ElastiCache
# No code changes needed! ✅
```

**Cluster details**:
```
Name: kempian-cache-prod
Type: cache.t3.small (or cache.t3.medium for production)
Replicas: 1-2
Multi-AZ: Enabled
Backups: 7 days retention
Encryption: Enabled
Auth: Enabled
Monitoring: CloudWatch
```

---

## Next Steps

1. ✅ Create ElastiCache cluster (this guide)
2. ✅ Update environment variables
3. ✅ Test connection
4. ✅ Deploy to production
5. ✅ Monitor for 48 hours
6. ✅ Document learnings

---

## Support & Resources

- AWS ElastiCache Documentation: https://docs.aws.amazon.com/elasticache/
- AWS Support: https://console.aws.amazon.com/support/
- Kempian Cache Manager: `backend/app/cache.py`
- Test script: `backend/test_elasticache.py` (create this)

---

## Quick Reference Commands

```bash
# Check cluster status
aws elasticache describe-cache-clusters \
    --cache-cluster-id kempian-cache-prod

# Get endpoint
aws elasticache describe-cache-clusters \
    --cache-cluster-id kempian-cache-prod \
    --show-cache-node-info \
    --query 'CacheClusters[0].CacheNodes[0].Endpoint'

# List all clusters
aws elasticache describe-cache-clusters

# Delete cluster (careful!)
aws elasticache delete-cache-cluster --cache-cluster-id kempian-cache-prod
```

---

**Congratulations! Your AWS ElastiCache setup is complete! 🎉**

Your Kempian application is now using fully managed, highly available AWS caching.

