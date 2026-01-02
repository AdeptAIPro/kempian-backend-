# AWS ElastiCache Connection Troubleshooting

## Problem: Connection Timeout

If you're seeing "Timeout connecting to server", this is a **network/security issue**, not a code issue.

## Root Causes

AWS ElastiCache Redis is designed to only be accessible from within your VPC (Virtual Private Cloud) for security reasons.

### 1. VPC Access Only
- ElastiCache is inside a private VPC
- It's NOT directly accessible from your local machine
- It can only be accessed from other AWS resources in the same VPC

### 2. Solutions

#### Option A: Use an EC2 Instance (Recommended for Production)
Deploy your backend on an EC2 instance in the same VPC as your ElastiCache cluster.

1. Create an EC2 instance in the same VPC
2. Install your backend on the EC2 instance
3. The EC2 instance can connect to ElastiCache

#### Option B: SSH Tunnel (For Development)
Create an SSH tunnel through an EC2 instance to access ElastiCache from your local machine.

```bash
# SSH tunnel command
ssh -L 6379:your-elasticache-endpoint:6379 ec2-user@your-ec2-ip

# Then in your .env, use:
ELASTICACHE_ENDPOINT=localhost
ELASTICACHE_PORT=6379
```

#### Option C: VPN (For Development)
If you have a VPN that connects to your AWS VPC:
1. Connect to the VPN
2. Update your .env file to use the ElastiCache endpoint
3. Test the connection

#### Option D: Temporary Workaround - Use In-Memory Cache
The backend automatically falls back to in-memory cache if Redis isn't available.

```bash
# In your .env file, comment out ElastiCache settings
# ELASTICACHE_ENDPOINT=...
# ELASTICACHE_PORT=...
# ELASTICACHE_AUTH_TOKEN=...
# REDIS_SSL=...

# Or keep them and let it fall back to in-memory cache
```

## Quick Test

Run this to check connectivity:

```bash
# Try pinging the ElastiCache endpoint (this will likely fail from your local machine)
telnet your-elasticache-endpoint 6379

# Or with redis-cli (if installed)
redis-cli -h your-elasticache-endpoint -p 6379 -a your-auth-token ping
```

## Verify Your Setup

### 1. Check ElastiCache in AWS Console
- Go to AWS Console → ElastiCache
- Check if your cluster is **Available**
- Note the **Primary endpoint** and **Security group**

### 2. Check Security Groups
The security group attached to your ElastiCache cluster must allow inbound traffic:
- **Port**: 6379
- **Source**: Should be your EC2 instance's security group (for production) or a VPN IP range

### 3. Check Subnet Groups
Ensure your ElastiCache subnet group includes subnets that allow the necessary traffic.

## Current Status

Your backend will work fine without Redis - it will use in-memory caching instead. The ElastiCache setup is for production optimization, but the application works without it.

### Your Options:
1. **Continue development with in-memory cache** (works perfectly, just restarts when you restart the backend)
2. **Set up an EC2 instance** and deploy your backend there
3. **Wait until you deploy to production** where network access is configured properly

## Next Steps

Based on your setup, I recommend:

1. **For now**: Let the backend use in-memory cache (it's working fine)
2. **For testing**: Set up an EC2 instance in the same VPC
3. **For production**: Deploy to EC2/Elastic Beanstalk with proper VPC configuration

The timeout you're seeing is **expected behavior** when trying to access VPC resources from outside the VPC. Your code is working correctly!

