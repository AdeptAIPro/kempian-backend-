# AWS ElastiCache for Kempian - Quick Start

## 📚 Documentation Files

1. **[AWS_ELASTICACHE_COMPLETE_SETUP.md](./AWS_ELASTICACHE_COMPLETE_SETUP.md)** ⭐ START HERE
   - Complete step-by-step setup guide
   - AWS Console configuration
   - Network and security setup
   - Testing and monitoring
   - Troubleshooting

2. **[AWS_ELASTICACHE_MIGRATION.md](./AWS_ELASTICACHE_MIGRATION.md)**
   - Migration strategy overview
   - Cost comparison
   - Benefits and features

3. **[test_elasticache_connection.py](./test_elasticache_connection.py)**
   - Automated test script
   - Verifies configuration
   - Performance testing

## 🚀 Quick Setup (5 Minutes)

### Step 1: Create ElastiCache in AWS Console

1. Go to **AWS Console** → **ElastiCache**
2. Click **"Create"** → Select **"Redis"**
3. Fill in:
   - **Name**: `kempian-cache-prod`
   - **Node type**: `cache.t3.small`
   - **Engine version**: Redis 7.1
   - **Subnet group**: Your VPC subnet
   - ✅ Enable Multi-AZ
   - ✅ Enable backups
4. Click **"Create cluster"** (takes ~15 minutes)

### Step 2: Configure Security

1. Go to **EC2** → **Security Groups**
2. Find your app's security group
3. Add inbound rule:
   - Port: `6379`
   - Source: Your ElastiCache security group ID

### Step 3: Get Endpoint

From ElastiCache console → Your cluster:
```
Primary endpoint: kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379
```

### Step 4: Update Environment

Add to your `.env` file:
```env
REDIS_URL=redis://kempian-cache-prod.xxxxx.aps1.cache.amazonaws.com:6379/0
```

### Step 5: Test Connection

```bash
cd backend
python test_elasticache_connection.py
```

That's it! ✅ Your app now uses AWS ElastiCache.

---

## ✨ Key Benefits

- **Fully Managed**: AWS handles everything
- **High Availability**: Auto-failover with Multi-AZ
- **Automatic Backups**: Point-in-time recovery
- **CloudWatch Monitoring**: Built-in metrics
- **Scalable**: Upgrade instance type anytime
- **Secure**: Encryption at rest and in transit
- **No Code Changes**: Your existing code works as-is!

---

## 💰 Cost Estimate

| Instance Type | Memory | Cost/Month |
|--------------|--------|------------|
| cache.t3.micro | 0.6 GB | ~$13 |
| cache.t3.small | 1.4 GB | ~$26 |
| cache.t3.medium | 3.1 GB | ~$52 |

*Plus ~20% for Multi-AZ and backups*

---

## 🧪 Testing

Run the test script:
```bash
python test_elasticache_connection.py
```

Expected output:
```
✅ Environment Variables
✅ Redis Connection  
✅ Basic Operations
✅ Server Information
✅ Cache Manager
✅ Performance Tests
🎉 All tests passed!
```

---

## 📊 Monitoring

Monitor these CloudWatch metrics:
- **CPUUtilization** → Should be < 70%
- **CacheHitRate** → Should be > 70%
- **CurrConnections** → Watch for connection limits
- **NetworkBytesIn/Out** → Monitor throughput

---

## 🔧 Current Configuration

Your Kempian app already supports AWS ElastiCache via:
- `backend/app/cache.py` - Cache manager with Redis support
- Automatic fallback to in-memory cache if Redis unavailable
- Multi-level caching (L1: in-memory, L2: Redis)

**No code changes needed!** Just update the `REDIS_URL` environment variable.

---

## 🆘 Troubleshooting

### Connection Refused?
1. Check security group allows port 6379
2. Verify VPC configuration
3. Test with: `redis-cli -h <endpoint> -p 6379 ping`

### High Latency?
1. Check CloudWatch for CPU or network issues
2. Consider scaling up node type
3. Review cache hit rate

### Authentication Failed?
1. Check AUTH token in ElastiCache console
2. Update REDIS_URL format:
   ```
   redis://:token@endpoint:6379/0
   ```

---

## 📖 For Detailed Instructions

See **[AWS_ELASTICACHE_COMPLETE_SETUP.md](./AWS_ELASTICACHE_COMPLETE_SETUP.md)** for complete setup guide with all details.

---

## 🎯 Summary

✅ **Yes, you can use AWS caching instead of Redis!**  
✅ **100% compatible** - no code changes  
✅ **Fully managed** - AWS handles everything  
✅ **Highly available** - Multi-AZ auto-failover  
✅ **Easy to set up** - 5 steps, 15 minutes  

Start with the [Complete Setup Guide](./AWS_ELASTICACHE_COMPLETE_SETUP.md) to get started!

