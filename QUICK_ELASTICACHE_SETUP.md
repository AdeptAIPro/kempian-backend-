# Quick ElastiCache Setup Guide

## Your ElastiCache Configuration

Based on the details you provided:

- **Primary Endpoint**: `master.redis-cache-server.wwhabv.aps1.cache.amazonaws.com:6379`
- **Port**: `6379`
- **Auth Token**: `AdeptAiPro_2025_Redis`
- **Region**: `ap-south-1` (Asia Pacific - Mumbai)

---

## Step 1: Add to Your .env File

Add these lines to your `.env` file in the `backend/` directory:

```bash
# AWS ElastiCache (Redis) Configuration
ELASTICACHE_ENDPOINT=master.redis-cache-server.wwhabv.aps1.cache.amazonaws.com:6379
ELASTICACHE_PORT=6379
ELASTICACHE_AUTH_TOKEN=AdeptAiPro_2025_Redis
REDIS_SSL=false
DISABLE_REDIS=false

# Search Configuration
FORCE_FULL_LOAD=true
MAX_WORKERS=16
PARALLEL_MATCHING_THRESHOLD=10000
```

**Note**: If you don't have a `.env` file, create one in the `backend/` directory.

---

## Step 2: Verify Security Group

**IMPORTANT**: Ensure your ElastiCache security group allows inbound traffic on port `6379` from your application server.

**To check/update**:
1. Go to AWS Console → EC2 → Security Groups
2. Find your ElastiCache security group
3. Edit inbound rules
4. Add rule:
   - **Type**: Custom TCP
   - **Port**: 6379
   - **Source**: Your application server's security group or IP address

---

## Step 3: Restart Your Application

After adding the environment variables, restart your Flask application:

```bash
# For local development
cd backend
python main.py

# For production (Gunicorn)
gunicorn --bind 0.0.0.0:8000 wsgi:app
```

---

## Step 4: Check Connection Logs

Look for these messages in your application logs:

```
Connecting to AWS ElastiCache: master.redis-cache-server.wwhabv.aps1.cache.amazonaws.com:6379
Port detected in endpoint, using host: master.redis-cache-server.wwhabv.aps1.cache.amazonaws.com, port: 6379
Auth token configured: Yes
SSL enabled: false
Attempting non-SSL connection to ElastiCache...
Non-SSL connection attempt successful
✓ Redis cache connected successfully
```

---

## Step 5: Test Connection

Test the connection using the API endpoint:

```bash
curl http://localhost:8000/search/cache/stats
```

**Expected response**:
```json
{
  "redis_available": true,
  "cached_candidates": 0,
  "batch_count": 0,
  "last_cached_at": null,
  "compression_enabled": true,
  "cache_ttl_days": 7
}
```

---

## Troubleshooting

### Connection Timeout

If you see: `Redis not available: Timeout connecting to server`

**Fix**: Check your ElastiCache security group - ensure port 6379 is open from your application server.

### Authentication Error

If you see: `NOAUTH Authentication required`

**Fix**: Verify `ELASTICACHE_AUTH_TOKEN` is exactly `AdeptAiPro_2025_Redis` (case-sensitive).

### Connection Refused

If you see: `Connection refused`

**Fix**: 
1. Verify the endpoint is correct
2. Check if your application can reach the ElastiCache endpoint (same VPC or VPN)

---

## What Happens Next?

After successful connection:

1. **First Load**: The application will load all candidates from DynamoDB (5-10 minutes)
2. **Caching**: All candidates will be cached in ElastiCache (30-60 seconds)
3. **Subsequent Loads**: Future loads will be ultra-fast (10-30 seconds) from ElastiCache

---

## Need More Help?

- See `ELASTICACHE_CONNECTION_SETUP.md` for detailed setup
- See `SETUP_GUIDE.md` for complete setup guide
- See `REDIS_CACHE_SETUP.md` for AWS ElastiCache setup details

