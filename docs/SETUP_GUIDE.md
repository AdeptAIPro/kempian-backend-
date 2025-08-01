# Backend Setup Guide

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
cd backend
pip install -r requirements.txt
```

### **2. Environment Variables**
Create a `.env` file in the backend directory:

```bash
# Database Configuration
DATABASE_URL=mysql+pymysql://username:password@localhost:3307/kempianDB

# Flask Configuration
SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# AWS Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
COGNITO_USER_POOL_ID=your-cognito-user-pool-id
COGNITO_CLIENT_ID=your-cognito-client-id
COGNITO_REGION=us-east-1
CLIENT_SECRET=your-cognito-client-secret

# Stripe Configuration
STRIPE_SECRET_KEY=sk_test_your-stripe-secret-key
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret

# Email Configuration (AWS SES)
SES_REGION=us-east-1
SES_FROM_EMAIL=noreply@yourdomain.com

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key

# Ceipal Configuration (Optional)
CEIPAL_EMAIL=your-ceipal-email
CEIPAL_PASSWORD=your-ceipal-password
CEIPAL_API_KEY=your-ceipal-api-key

# Frontend URL
FRONTEND_URL=http://localhost:3000
```

### **3. Database Setup**
```bash
# Create database tables
python create_all_tables.py

# Seed initial data
python create_tables_and_seed_plans.py
```

### **4. Run the Server**
```bash
python main.py
```

## 🔒 **Security Checklist**

### **Critical Security Fixes**

#### **1. Database Credentials**
✅ **FIXED**: Removed hardcoded credentials from `config.py`
- Use `DATABASE_URL` environment variable
- Never commit credentials to version control

#### **2. JWT Token Validation**
⚠️ **NEEDS FIXING**: Current implementation doesn't verify signatures
```python
# Current (INSECURE)
payload = jwt.decode(token, options={"verify_signature": False})

# Should be (SECURE)
payload = jwt.decode(token, secret_key, algorithms=['HS256'])
```

#### **3. Debug Statements**
⚠️ **NEEDS CLEANUP**: Remove all `print()` statements
```bash
# Find all debug statements
grep -r "print(" backend/
```

#### **4. Rate Limiting**
⚠️ **MISSING**: Add rate limiting to prevent abuse
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

@limiter.limit("5 per minute")
@app.route('/api/endpoint')
def protected_endpoint():
    pass
```

## 🛠️ **Production Deployment**

### **1. WSGI Server**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

### **2. Environment Variables**
- Use production values for all environment variables
- Never use development defaults in production
- Use a secrets management service

### **3. Database**
- Use a managed database service
- Enable SSL connections
- Set up proper backups
- Configure connection pooling

### **4. Monitoring**
```python
# Add health check endpoint
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.utcnow()}
```

## 📊 **Performance Optimization**

### **1. Database Indexes**
```sql
-- Add indexes for frequently queried fields
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_search_logs_user_id ON jd_search_logs(user_id);
CREATE INDEX idx_tenants_status ON tenants(status);
```

### **2. Caching**
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@cache.memoize(timeout=300)
def expensive_operation():
    pass
```

### **3. Pagination**
```python
# Add pagination to list endpoints
@app.route('/api/items')
def get_items():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    items = Item.query.paginate(page=page, per_page=per_page)
    return jsonify({
        'items': [item.to_dict() for item in items.items],
        'total': items.total,
        'pages': items.pages,
        'current_page': page
    })
```

## 🧪 **Testing**

### **1. Unit Tests**
```bash
pip install pytest
pytest tests/
```

### **2. Integration Tests**
```bash
# Test API endpoints
python -m pytest tests/integration/
```

### **3. Load Testing**
```bash
pip install locust
locust -f tests/load_test.py
```

## 📋 **API Documentation**

### **1. Add Swagger Documentation**
```bash
pip install flask-restx
```

```python
from flask_restx import Api, Resource

api = Api(app, title='Kempian API', version='1.0')

@api.route('/users')
class UserList(Resource):
    def get(self):
        """List all users"""
        pass
```

## 🔍 **Monitoring & Logging**

### **1. Structured Logging**
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/kempian.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Kempian startup')
```

### **2. Error Tracking**
```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)
```

## 🚨 **Emergency Procedures**

### **1. Database Backup**
```bash
mysqldump -u username -p kempianDB > backup.sql
```

### **2. Rollback Procedure**
```bash
# Restore from backup
mysql -u username -p kempianDB < backup.sql
```

### **3. Security Incident Response**
1. Identify the breach
2. Contain the threat
3. Assess the damage
4. Notify stakeholders
5. Implement fixes
6. Monitor for recurrence

## 📞 **Support**

For issues or questions:
1. Check the logs: `tail -f logs/kempian.log`
2. Review the troubleshooting guide
3. Contact the development team 