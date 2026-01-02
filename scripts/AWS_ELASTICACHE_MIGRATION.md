# AWS ElastiCache Migration Guide

## Overview

Yes, you can absolutely use AWS caching instead of Redis! Your code is already compatible with **AWS ElastiCache for Redis**, which is a fully managed service that's 100% compatible with Redis. No code changes are needed - just update your configuration.

## AWS Caching Options

### 1. **AWS ElastiCache for Redis** ⭐ RECOMMENDED
- **100% compatible with your existing code**
- No code changes required
- Just change the REDIS_URL
- Fully managed (AWS handles scaling, patches, backups)
- Supports Redis OSS (up to version 7.1)
- Automatic failover and high availability
- Optional persistence for durability

### 2. **AWS ElastiCache for Valkey**
- New open-source fork of Redis (fully BSD licensed)
- Managed by Linux Foundation
- Similar features to Redis
- Future-proof alternative

### 3. **AWS ElastiCache for Memcached**
- Requires code changes (not recommended for your use case)
- Simpler but lacks Redis features

### 4. **Amazon MemoryDB**
- Redis-compatible with automatic persistence
- Good for when you need data durability
- More expensive than ElastiCache
- Better for databases than caching

## Migration Steps

### Step 1: Create ElastiCache Cluster (via AWS Console)

1. **Go to AWS Console** → **ElastiCache**

2. **Click "Create cluster"**

3. **Configuration**:
   - **Cluster type**: Redis OSS
   - **Name**: `kempian-cache`
   - **Engine version**: Redis 7.1 (latest)
   - **Node type**: 
     - **Development**: `cache.t3.micro` (~$13/month)
     - **Production**: `cache.t3.small` (~$26/month) or `cache.t3.medium` (~$52/month)
   - **Number of replicas**: 1 (for high availability)
   - **Availability zones**: Multi-AZ enabled
   - **Subnet group**: Select your VPC subnet group
   - **Security group**: Allow inbound on port 6379 from your EC2/app server

4. **Optional Settings**:
   - **Parameter group**: `default` (or create custom)
   - **Backup**: Enable for production
   - **Maintenance window**: Select convenient time

5. **Click "Create"**

### Step 2: Update Environment Variables

Once your cluster is created, you'll get an endpoint like:
```
kempian-cache-001.xxxxxx.cache.amazonaws.com:6379
```

Update your environment variables:

**For Production (.env or AWS Systems Manager Parameter Store):**
```env
REDIS_URL=redis://kempian-cache-001.xxxxxx.cache.amazonaws.com:6379/0

# Or if you enabled authentication:
REDIS_URL=redis://:your-auth-token@kempian-cache-001.xxxxxx.cache.amazonaws.com:6379/0
```

**For AWS Lambda or serverless:**
```env
REDIS_URL=redis://kempian-cache-001.xxxxxx.cache.amazonaws.com:6379/0
```

### Step 3: Test Connection

```python
# Test your connection
import redis
import os

redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
r = redis.from_url(redis_url, decode_responses=True)

try:
    r.ping()
    print("✅ AWS ElastiCache connection successful!")
    r.set('test_key', 'test_value')
    print(f"Test value: {r.get('test_key')}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### Step 4: Update Security Group

Make sure your application server can access ElastiCache:

1. Go to **EC2** → **Security Groups**
2. Find your ElastiCache security group
3. Add inbound rule:
   - Type: Custom TCP
   - Port: 6379
   - Source: Your application server's security group or IP

## Cost Comparison

### Current Setup (Self-hosted Redis)
- EC2 instance: ~$10-50/month (depending on size)
- Maintenance: Manual updates, monitoring, backups
- Time: DevOps overhead

### AWS ElastiCache
- **cache.t3.micro** (600MB): ~$13/month
- **cache.t3.small** (1.4GB): ~$26/month  
- **cache.t3.medium** (3.1GB): ~$52/month

**Benefits:**
- Automatic backups
- High availability (multi-AZ)
- Automatic failover
- Managed updates and patches
- CloudWatch monitoring
- No DevOps overhead

## Features You Get with AWS ElastiCache

✅ **High Availability**: Automatic failover to replica  
✅ **Auto Scaling**: Add/remove nodes as needed  
✅ **Automatic Backups**: Point-in-time recovery  
✅ **Monitoring**: CloudWatch metrics out of the box  
✅ **Security**: Encryption at rest and in transit  
✅ **VPC Integration**: Isolated network in your VPC  
✅ **Authentication**: Optional AUTH token support  

## Code Compatibility

Your existing code in `backend/app/cache.py` is **100% compatible**. The `redis` Python client library works identically with AWS ElastiCache:

```python
# This code works with both local Redis and AWS ElastiCache
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
redis_client = redis.from_url(redis_url, decode_responses=True)
redis_client.ping()
```

No code changes needed! 🎉

## Migration Strategy

### Phase 1: Development (Parallel Setup)
1. Keep local Redis for development
2. Create ElastiCache cluster in dev AWS account
3. Test with dev environment
4. Verify all functionality works

### Phase 2: Staging
1. Create ElastiCache in staging
2. Update staging environment variables
3. Run integration tests
4. Monitor performance

### Phase 3: Production
1. Create production ElastiCache with:
   - Multiple replicas
   - Automatic backups
   - Multi-AZ
   - Snapshot retention
2. Update production environment variables
3. Monitor for 24-48 hours
4. Decommission old Redis server

## Monitoring

After migration, monitor these CloudWatch metrics:
- `CPUUtilization`: Should stay < 80%
- `NetworkBytesIn/Out`: Network throughput
- `CacheHitRate`: Cache efficiency
- `CurrConnections`: Active connections
- `ReplicationLag`: Replica sync status

## Rollback Plan

If issues occur:
1. Revert environment variable to old Redis URL
2. Restart application
3. Investigate issue with ElastiCache
4. Fix and retry migration

## Benefits Summary

| Feature | Local Redis | AWS ElastiCache |
|---------|------------|----------------|
| Setup Time | Hours | Minutes |
| High Availability | Manual | Automatic |
| Backups | Manual | Automatic |
| Monitoring | DIY | CloudWatch |
| Scaling | Manual | On-demand |
| Updates | Manual | AWS managed |
| Dev Time | You maintain | AWS maintains |

## Next Steps

1. **Create ElastiCache cluster** in AWS Console
2. **Update REDIS_URL** in your environment
3. **Test the connection** (use the test script above)
4. **Monitor for 24 hours** and verify cache performance
5. **Scale as needed** based on your cache hit rate

## AWS CLI Commands (Optional)

```bash
# Create cluster via CLI
aws elasticache create-replication-group \
    --replication-group-id kempian-cache-prod \
    --description "Kempian production cache" \
    --engine redis \
    --engine-version 7.1 \
    --cache-node-type cache.t3.small \
    --num-cache-clusters 2 \
    --automatic-failover-enabled \
    --snapshot-retention-limit 5 \
    --preferred-cache-cluster-azs us-east-1a us-east-1b

# Get endpoint
aws elasticache describe-replication-groups \
    --replication-group-id kempian-cache-prod
```

## Conclusion

✅ **YES, you can use AWS caching!**  
✅ **No code changes needed** - just configuration  
✅ **AWS ElastiCache for Redis** is recommended  
✅ **Fully managed** - less maintenance for you  
✅ **Higher availability** and automatic failover  

Your current code will work as-is with AWS ElastiCache. Just change the `REDIS_URL` environment variable and you're done!

