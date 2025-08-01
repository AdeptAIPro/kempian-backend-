# 🚀 Backend Setup & Hosting Guide

This guide will help you set up and host the complete backend with all AdeptAI-Master algorithm functionality.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- AWS credentials (for DynamoDB access)
- At least 4GB RAM (for ML models)

## 🛠️ Quick Setup

### 1. Install Dependencies

```bash
cd backend
python setup_backend.py
```

This script will:
- Install all required packages
- Download NLTK data
- Create necessary directories
- Test all components

### 2. Configure Environment

Edit the `.env` file with your AWS credentials:

```env
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=ap-south-1
```

### 3. Test Everything

```bash
python test_complete_backend.py
```

This will test all functionality and ensure everything works.

### 4. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## 🔧 Manual Setup (if needed)

### Install Requirements

```bash
pip install -r requirements.txt
```

### Install NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## 📊 What's Included

### ✅ Advanced AdeptAI-Master Algorithm
- **EnhancedRecruitmentSearchSystem**: Advanced semantic search with FAISS
- **EnhancedCandidateMatchingSystem**: Intelligent candidate scoring
- **AdvancedJobQueryParser**: Sophisticated job requirement parsing
- **MemoryOptimizedEmbeddingSystem**: Efficient text embeddings

### ✅ Fallback System
- Basic keyword-based search
- Graceful degradation when advanced system unavailable
- Same API interface for both systems

### ✅ Performance Features
- Caching for repeated searches
- Memory optimization
- Performance monitoring
- Statistics tracking

## 🧪 Testing

### Run Complete Test Suite

```bash
python test_complete_backend.py
```

### Test Individual Components

```bash
# Test basic functionality
python simple_test.py

# Test advanced integration
python test_advanced_integration.py
```

## 🌐 API Endpoints

### Health Check
```
GET /health
```

### Search Candidates
```
POST /api/search
Content-Type: application/json

{
  "job_description": "Python developer with React experience",
  "top_k": 10
}
```

### Semantic Match
```
POST /api/semantic-match
Content-Type: application/json

{
  "job_description": "Senior Software Engineer"
}
```

## 📁 Project Structure

```
backend/
├── app/
│   ├── search/
│   │   ├── adeptai_components/
│   │   │   ├── enhanced_recruitment_search.py
│   │   │   ├── enhanced_candidate_matcher.py
│   │   │   └── advanced_query_parser.py
│   │   ├── service.py
│   │   └── routes.py
│   └── ...
├── requirements.txt
├── setup_backend.py
├── test_complete_backend.py
└── README_BACKEND_SETUP.md
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   python setup_backend.py
   ```

2. **Memory Issues**
   - Ensure you have at least 4GB RAM
   - Close other applications
   - Use CPU-only versions of PyTorch/FAISS

3. **AWS Connection Issues**
   - Check your AWS credentials
   - Ensure DynamoDB tables exist
   - Verify region settings

4. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

### Performance Optimization

1. **Use CPU-only versions** (already configured):
   ```bash
   pip install faiss-cpu
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Enable caching** (already implemented)

3. **Monitor memory usage**:
   ```python
   import psutil
   print(f"Memory usage: {psutil.virtual_memory().percent}%")
   ```

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using uWSGI
pip install uwsgi
uwsgi --http :5000 --wsgi-file app.py --callable app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 📈 Monitoring

The system includes built-in performance monitoring:

- Search response times
- Cache hit rates
- System usage statistics
- Error tracking

Access via:
```python
from app.search.service import adept_ai
stats = adept_ai.performance_stats
print(stats)
```

## 🔄 Updates

To update the system:

1. Pull latest changes
2. Run setup script:
   ```bash
   python setup_backend.py
   ```
3. Test functionality:
   ```bash
   python test_complete_backend.py
   ```

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section
2. Run the test suite
3. Check logs for detailed error messages
4. Ensure all dependencies are installed

## 🎉 Success Indicators

Your backend is working correctly when:

- ✅ All tests pass
- ✅ Server starts without errors
- ✅ API endpoints respond
- ✅ Search returns results
- ✅ No import errors in logs

---

**Ready to deploy! 🚀** 