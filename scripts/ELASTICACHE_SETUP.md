# AWS ElastiCache Redis Setup Guide

This guide will help you configure AWS ElastiCache Redis for your Kempian backend.

## Quick Setup

### Step 1: Set Environment Variables

You have two options:

#### Option A: Using .env file (Recommended)

Create or update your `.env` file in the `backend/` directory:

```bash
# AWS ElastiCache Configuration
ELASTICACHE_ENDPOINT=your-cluster-endpoint.xxx.cache.amazonaws.com
ELASTICACHE_PORT=6379
ELASTICACHE_AUTH_TOKEN=your-auth-token-here
REDIS_SSL=true
```

Replace:
- `your-cluster-endpoint` with your AWS ElastiCache primary endpoint
- `your-auth-token-here` with your auth token

#### Option B: Set in your system environment

```bash
# Windows (Command Prompt)
set ELASTICACHE_ENDPOINT=your-cluster-endpoint.xxx.cache.amazonaws.com
set ELASTICACHE_PORT=6379
set ELASTICACHE_AUTH_TOKEN=your-auth-token-here
set REDIS_SSL=true

# Windows (PowerShell)
$env:ELASTICACHE_ENDPOINT="your-cluster-endpoint.xxx.cache.amazonaws.com"
$env:ELASTICACHE_PORT="6379"
$env:ELASTICACHE_AUTH_TOKEN="your-auth-token-here"
$env:REDIS_SSL="true"

# Linux/Mac
export ELASTICACHE_ENDPOINT=your-cluster-endpoint.xxx.cache.amazonaws.com
export ELASTICACHE_PORT=6379
export ELASTICACHE_AUTH_TOKEN=your-auth-token-here
export REDIS_SSL=true
```

### Step 2: Test the Connection

Run the test script:

```bash
cd backend
python test_elasticache_setup.py
```

You should see:
```
✅ Connection successful!
🎉 All tests passed! AWS ElastiCache is ready to use.
```

### Step 3: Start the Backend

Once the connection test passes, start your backend server:

```bash
python start_server.py
# or
python main.py
```

You should see in the logs:
```
✓ Redis cache connected successfully
```

## Troubleshooting

### Connection Refused

**Problem**: "Connection refused" or "target machine actively refused it"

**Solutions**:
1. Check if `ELASTICACHE_ENDPOINT` is set correctly
2. Verify the endpoint is the **primary endpoint** (not read endpoint)
3. Ensure your security group allows inbound traffic on port 6379 from your IP
4. Check if the ElastiCache cluster is running

### Authentication Failed

**Problem**: "Authentication failed" or "NOAUTH Authentication required"

**Solutions**:
1. Check if `ELASTICACHE_AUTH_TOKEN` is set correctly
2. Verify you're using the correct auth token from AWS
3. Make sure the auth token has no extra spaces or quotes

### SSL Connection Issues

**Problem**: SSL/TLS errors

**Solutions**:
1. Set `REDIS_SSL=true` if your ElastiCache uses SSL
2. Set `REDIS_SSL=false` if your ElastiCache doesn't use SSL
3. If using SSL, ensure you're using TLS connections (Redis 6.0+)

## Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ELASTICACHE_ENDPOINT` | Yes | Primary endpoint from AWS ElastiCache | `kempian-cache.xxx.cache.amazonaws.com` |
| `ELASTICACHE_PORT` | No | Redis port (default: 6379) | `6379` |
| `ELASTICACHE_AUTH_TOKEN` | Yes | Auth token from AWS | `your-token-here` |
| `REDIS_SSL` | No | Enable SSL (default: false) | `true` or `false` |

## Getting Your AWS ElastiCache Details

### 1. AWS Console
1. Log into AWS Console
2. Go to **ElastiCache**
3. Click on your Redis cluster
4. Copy the **Primary endpoint** and **Port**
5. Copy the **Auth token** (if enabled)

### 2. AWS CLI
```bash
# List all ElastiCache clusters
aws elasticache describe-cache-clusters --show-cache-node-info

# Get cluster details
aws elasticache describe-cache-clusters --cache-cluster-id your-cluster-id
```

## Security Notes

⚠️ **Important**: Never commit your `.env` file with real credentials to version control!

- The `.env` file should be in `.gitignore`
- Use environment variables or AWS Secrets Manager in production
- Rotate auth tokens periodically

## Need Help?

If you're still having issues:
1. Check the logs: `backend/logs/app.log`
2. Run the test script: `python test_elasticache_setup.py`
3. Verify your AWS ElastiCache cluster status in AWS Console

