# Redis Installation Guide - Local & EC2

## 🔧 Local Installation (Windows)

### Option 1: Using Docker (Easiest) ⭐ Recommended

```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop

# After Docker is installed, run:
docker run -d -p 6379:6379 --name redis-local redis:latest

# Verify Redis is running:
docker ps

# Test connection:
docker exec -it redis-local redis-cli ping
# Should return: PONG
```

### Option 2: Using WSL (Windows Subsystem for Linux)

```bash
# 1. Install WSL (in PowerShell as Administrator)
wsl --install

# 2. Restart your computer

# 3. Open WSL terminal and run:
sudo apt update
sudo apt install redis-server

# 4. Start Redis:
sudo service redis-server start

# 5. Test:
redis-cli ping
# Should return: PONG
```

### Option 3: Windows Port by Memurai

```bash
# Download from: https://www.memurai.com/get-memurai

# Install the .msi file

# Start Memurai (it runs on port 6379)

# Test:
memurai-cli ping
# Should return: PONG
```

---

## 🚀 EC2 Installation

### Step 1: Connect to Your EC2 Instance

```bash
# Replace with your EC2 details
ssh -i your-key.pem ec2-user@your-ec2-ip

# Or if using Ubuntu:
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### Step 2: Install Redis on EC2

```bash
# Update system
sudo yum update -y   # Amazon Linux
# or
sudo apt update      # Ubuntu

# Install Redis
sudo yum install redis -y   # Amazon Linux
# or
sudo apt install redis-server -y   # Ubuntu
```

### Step 3: Configure Redis for Remote Access

#### For Amazon Linux:

```bash
# 1. Edit Redis config
sudo nano /etc/redis.conf

# 2. Find and change these lines:
# From:
bind 127.0.0.1
protected-mode yes

# To:
bind 0.0.0.0
protected-mode no

# 3. Save and exit (Ctrl+X, then Y, then Enter)

# 4. Start Redis
sudo service redis start
sudo chkconfig redis on  # Make it start on boot
```

#### For Ubuntu:

```bash
# 1. Edit Redis config
sudo nano /etc/redis/redis.conf

# 2. Find and change these lines:
# From:
bind 127.0.0.1
protected-mode yes

# To:
bind 0.0.0.0
protected-mode no

# 3. Save and exit

# 4. Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
sudo systemctl status redis-server  # Verify it's running
```

### Step 4: Configure EC2 Security Group

**In AWS Console:**
1. Go to **EC2** → **Security Groups**
2. Select your EC2 instance's security group
3. Click **Edit inbound rules**
4. Add rule:
   - **Type**: Custom TCP
   - **Port**: 6379
   - **Source**: Your IP address (for testing) or 0.0.0.0/0 (not recommended for production)
5. Save

### Step 5: Test Redis on EC2

```bash
# On EC2, test locally:
redis-cli ping
# Should return: PONG

# Test with IP binding:
redis-cli -h 0.0.0.0 ping
# Should return: PONG
```

---

## 📝 Update Your Backend .env File

### For Local Development:

```bash
# In backend/.env, comment out ElastiCache settings and use local Redis:
# ELASTICACHE_ENDPOINT=...
# ELASTICACHE_PORT=...
# ELASTICACHE_AUTH_TOKEN=...
# REDIS_SSL=...

# Use local Redis instead:
REDIS_URL=redis://localhost:6379/0
```

### For EC2 Deployment:

```bash
# Keep your ElastiCache settings (for production)
ELASTICACHE_ENDPOINT=master.redis-cache-server.wwhabv.aps1.cache.amazonaws.com
ELASTICACHE_PORT=6379
ELASTICACHE_AUTH_TOKEN=AdeptAiPro_2025_Redis
REDIS_SSL=false

# OR use local Redis on EC2 (if needed):
# REDIS_URL=redis://localhost:6379/0
```

---

## ✅ Verification Steps

### 1. Test Local Redis Connection

```bash
# Run from your backend directory:
python test_elasticache_setup.py

# Or create a quick test:
python -c "import redis; r=redis.Redis(host='localhost', port=6379); print(r.ping())"
# Should print: True
```

### 2. Test EC2 Redis Connection

```bash
# From your local machine:
redis-cli -h your-ec2-ip -p 6379 ping

# Should return: PONG
```

### 3. Start Your Backend

```bash
# In backend directory:
python main.py

# Look for in logs:
# "✓ Redis cache connected successfully"
```

---

## 🔐 Security Considerations

### For Production (EC2):

1. **Use ElastiCache** (recommended)
   - Managed by AWS
   - Auto-scaling
   - Backups included

2. **If using Redis on EC2:**
   - Add password authentication
   - Only allow connections from your application
   - Use SSL/TLS if exposed to internet

### Add Password to Redis (Optional but Recommended):

```bash
# On EC2:
redis-cli

# In Redis CLI:
CONFIG SET requirepass your-strong-password

# Then in your .env:
REDIS_URL=redis://:your-strong-password@localhost:6379/0
```

---

## 📊 Summary

| Location | Method | Pros |
|----------|--------|------|
| **Local (Development)** | Docker | Easy, isolated, cross-platform |
| **Local (WSL)** | Native Linux | Full features, good performance |
| **EC2** | ElastiCache | Managed, auto-scaling, best for production |
| **EC2** | Self-hosted Redis | Full control, cost-effective |

### Recommended Setup:
- **Local Development**: Use Docker Redis
- **Production**: Use AWS ElastiCache (what you have configured)

---

## 🐛 Troubleshooting

### Redis not starting on EC2:
```bash
# Check status:
sudo systemctl status redis-server

# View logs:
sudo journalctl -u redis-server

# Restart:
sudo systemctl restart redis-server
```

### Can't connect to Redis:
1. Check security group allows port 6379
2. Verify Redis is running: `redis-cli ping`
3. Check firewall: `sudo iptables -L -n`
4. Test from EC2 itself first

### Windows Docker issues:
```bash
# Check if Docker is running:
docker info

# Check Redis container:
docker logs redis-local

# Restart container:
docker restart redis-local
```

