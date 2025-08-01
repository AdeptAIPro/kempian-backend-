# AdeptAI Masters Algorithm Documentation

## Overview

The AdeptAI Masters Algorithm is an advanced AI-powered candidate matching system that combines multiple techniques to provide highly accurate job-candidate matching. It replaces the previous simple keyword-based approach with a sophisticated multi-layered algorithm.

## Key Features

### 1. **Neural Learning with Reinforcement and Genetic Algorithm (NLRGA)**
- **Adaptive Scoring**: The algorithm learns from user feedback to improve future matches
- **Feedback Integration**: Positive/negative feedback affects candidate scores in subsequent searches
- **Genetic Optimization**: Continuously evolves matching criteria based on successful matches

### 2. **Advanced Keyword Extraction**
- **Domain-Specific Keywords**: Separate keyword sets for software and healthcare domains
- **Smart Filtering**: Removes common stop words and irrelevant terms
- **Frequency Analysis**: Considers keyword frequency for better relevance

### 3. **Semantic Similarity Matching**
- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` model for semantic understanding
- **Context Awareness**: Understands meaning beyond exact keyword matches
- **Caching**: Optimizes performance with semantic similarity caching

### 4. **GPT-4 Reranking**
- **Intelligent Reranking**: Uses GPT-4 to provide reasoning-based candidate reranking
- **Detailed Explanations**: Provides explanations for why candidates are ranked as they are
- **Fallback Mechanism**: Gracefully falls back to semantic ranking if GPT-4 is unavailable

### 5. **Domain Detection**
- **Automatic Classification**: Automatically detects if a job is software or healthcare related
- **Specialized Matching**: Applies domain-specific matching criteria
- **Cross-Domain Support**: Can handle mixed-domain requirements

## Architecture

### Core Classes

#### `AdeptAIMastersAlgorithm`
The main algorithm class that orchestrates all matching functionality.

**Key Methods:**
- `extract_keywords(text)`: Extracts meaningful keywords from text
- `detect_domain(keywords)`: Determines the domain (software/healthcare)
- `nlrga_score(matched_keywords, total_keywords, candidate_id)`: Calculates adaptive scores
- `semantic_similarity(text1, text2)`: Computes semantic similarity between texts
- `rerank_with_gpt4(job_description, candidates)`: Uses GPT-4 for intelligent reranking
- `keyword_search(job_description, top_k)`: Performs advanced keyword-based search
- `semantic_match(job_description)`: Complete semantic matching pipeline

#### `FeedbackManager`
Manages candidate feedback for adaptive learning.

**Key Methods:**
- `load_feedback()`: Loads feedback from DynamoDB or local file
- `save_feedback(candidate_id, positive)`: Saves feedback to both DynamoDB and local storage

## Algorithm Flow

### 1. **Input Processing**
```
Job Description → Keyword Extraction → Domain Detection
```

### 2. **Candidate Matching**
```
Candidate Database → Keyword Matching → Domain Overlap → Initial Scoring
```

### 3. **Advanced Scoring**
```
Initial Score + Semantic Similarity + Feedback Learning = Final Score
```

### 4. **Reranking (Optional)**
```
Top Candidates → GPT-4 Analysis → Reasoning → Final Ranking
```

## API Endpoints

### Search Endpoints

#### `POST /search`
Main search endpoint that uses the AdeptAI Masters algorithm.

**Request Body:**
```json
{
  "job_description": "Senior Python Developer with Django experience..."
}
```

**Response:**
```json
{
  "results": [
    {
      "FullName": "John Doe",
      "email": "john@example.com",
      "Score": 85.5,
      "Grade": "A",
      "MatchPercent": 78.5,
      "SemanticScore": 82.3,
      "Skills": ["Python", "Django", "React"],
      "LLM_Reasoning": "Strong match due to extensive Python experience..."
    }
  ],
  "summary": "Top 10 candidates found using AdeptAI Masters algorithm."
}
```

#### `POST /search/feedback`
Submit feedback for candidates to improve future matching.

**Request Body:**
```json
{
  "candidate_id": "john@example.com",
  "feedback": "good"
}
```

### Quota Management

#### `GET /search/quota`
Get current search quota information.

**Response:**
```json
{
  "quota": 100,
  "used": 45,
  "remaining": 55
}
```

## Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# CEIPAL Integration (Optional)
CEIPAL_EMAIL=your_ceipal_email
CEIPAL_PASSWORD=your_ceipal_password
CEIPAL_API_KEY=your_ceipal_api_key
```

### Dependencies

The algorithm requires the following Python packages:

```txt
nltk>=3.8
sentence-transformers>=2.2
openai>=0.27
numpy>=1.21
requests>=2.28
scikit-learn>=1.1
torch>=1.12
transformers>=4.21
```

## Performance Optimization

### 1. **Caching**
- Semantic similarity results are cached to avoid recomputation
- Feedback data is cached in memory with periodic persistence

### 2. **Batch Processing**
- Multiple candidates are processed in batches for efficiency
- DynamoDB queries are optimized for minimal latency

### 3. **Model Loading**
- Sentence transformer model is loaded once and reused
- Graceful fallback if model loading fails

## Testing

Run the comprehensive test suite:

```bash
cd backend
python test_adeptai_algorithm.py
```

The test suite covers:
- Keyword extraction
- Domain detection
- Scoring algorithms
- Semantic similarity
- Feedback system
- Complete search pipeline

## Monitoring and Logging

### Log Levels
- **INFO**: General algorithm operations
- **DEBUG**: Detailed matching information
- **WARNING**: Non-critical issues (e.g., model loading failures)
- **ERROR**: Critical failures

### Key Metrics
- Search response times
- Candidate match quality scores
- Feedback submission rates
- GPT-4 reranking success rates

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check internet connection for model download
   - Verify sufficient disk space
   - Check Python environment compatibility

2. **OpenAI API Errors**
   - Verify API key is valid
   - Check API quota limits
   - Ensure proper error handling

3. **DynamoDB Connection Issues**
   - Verify AWS credentials
   - Check network connectivity
   - Ensure table permissions

### Debug Mode

Enable debug logging by setting the log level:

```python
import logging
logging.getLogger('app.search.service').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Multi-Language Support**
   - Support for non-English job descriptions
   - Language detection and translation

2. **Advanced ML Models**
   - Custom fine-tuned models for specific domains
   - Ensemble methods for improved accuracy

3. **Real-time Learning**
   - Continuous model updates based on feedback
   - A/B testing for algorithm improvements

4. **Enhanced Analytics**
   - Detailed matching analytics dashboard
   - Performance metrics and insights

## Security Considerations

1. **Data Privacy**
   - All candidate data is encrypted at rest
   - Feedback data is anonymized where possible

2. **API Security**
   - JWT-based authentication
   - Rate limiting on search endpoints
   - Input validation and sanitization

3. **Model Security**
   - Secure model loading and caching
   - Protection against model poisoning attacks

## Support

For technical support or questions about the AdeptAI Masters algorithm:

1. Check the test suite for common issues
2. Review the logging output for error details
3. Verify all dependencies are properly installed
4. Ensure environment variables are correctly set

---

*This documentation covers the complete implementation of the AdeptAI Masters algorithm. For specific implementation details, refer to the source code in `backend/app/search/service.py`.* 