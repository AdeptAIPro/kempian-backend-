# AdeptAI - AI-Powered Recruitment System

AdeptAI is a comprehensive, multi-layered recruitment search system that combines multiple AI/ML technologies to provide accurate, fast, and fair candidate matching. The system integrates 15+ advanced components including instant search, dense retrieval, behavioral analysis, bias prevention, and market intelligence.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Modules](#modules)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)

## Overview

AdeptAI provides intelligent candidate search and matching using:

- **ML-Based Domain Classification**: Automatically categorizes job descriptions and candidate profiles into domains (technology, healthcare, finance, etc.)
- **LLM Query Enhancement**: Uses Large Language Models to intelligently expand search queries
- **Dense Retrieval**: Semantic search using embeddings for better candidate matching
- **Learning to Rank**: Advanced ranking algorithms to prioritize the best candidates
- **Behavioral Analysis**: Analyzes candidate behavior and career patterns
- **Bias Prevention**: Ensures fair and unbiased candidate evaluation
- **Market Intelligence**: Provides insights into market trends and compensation

## Features

### Core Features

- **Intelligent Search**: Multi-layered search system combining keyword, semantic, and ML-based matching
- **Domain Classification**: Automatic categorization of jobs and candidates
- **Query Enhancement**: LLM-powered query expansion with synonyms, related skills, and job variations
- **Candidate Ranking**: Learning-to-rank algorithms for optimal candidate ordering
- **Bias Prevention**: Built-in bias detection and mitigation
- **Explainable AI**: Transparent scoring and ranking explanations

### Advanced Features

- **Behavioral Analysis**: Career pattern analysis and behavioral scoring
- **Market Intelligence**: Real-time market trends and compensation insights
- **Multi-Modal Engine**: Combines text, structured data, and behavioral signals
- **Caching**: Intelligent caching for improved performance
- **Async Processing**: Asynchronous operations for better scalability

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AdeptAI Application                       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Search     │  │   Ranking   │  │   Domain     │    │
│  │   System     │  │   System     │  │   Classifier │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   LLM Query  │  │   Behavioral│  │   Bias       │    │
│  │   Enhancer   │  │   Analysis  │  │   Prevention │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Dense      │  │   Market    │  │   Explainable│    │
│  │   Retrieval  │  │   Intel     │  │   AI         │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- AWS credentials (for SageMaker integration, optional)
- API keys for LLM providers (OpenAI, Anthropic, Hugging Face - optional)

### Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific components
pip install scikit-learn transformers torch  # For ML models
pip install openai anthropic  # For LLM providers
pip install boto3  # For AWS SageMaker integration
```

### Environment Setup

1. Copy the environment template:
```bash
cp env.template .env
```

2. Configure environment variables in `.env`:
```bash
# AWS Configuration (optional)
AWS_REGION=us-east-1
AWS_DEFAULT_REGION=us-east-1

# LLM Provider API Keys (optional)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
HUGGINGFACE_TOKEN=your-huggingface-token

# Application Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=False
```

## Configuration

### Application Settings

Configuration is managed through `app/config.py`. Key settings include:

- **Search System**: Configure search algorithms and ranking parameters
- **ML Models**: Set model paths and parameters
- **LLM Providers**: Configure LLM provider preferences
- **Caching**: Configure cache sizes and TTLs
- **Logging**: Set log levels and output destinations

### Model Configuration

Models can be configured to use:
- **Local execution**: Models run locally (default)
- **SageMaker endpoints**: Models deployed on AWS SageMaker (see `Sagemaker/` folder)

## Usage

### Basic Usage

```python
from adeptai.ml_domain_classifier import get_ml_domain_classifier
from adeptai.llm_query_enhancer import get_llm_query_enhancer

# Initialize domain classifier
classifier = get_ml_domain_classifier()
domain, confidence = classifier.classify_domain("Python developer with 5 years experience")
print(f"Domain: {domain}, Confidence: {confidence:.2f}")

# Initialize query enhancer
enhancer = get_llm_query_enhancer(provider="openai", model="gpt-4")
result = enhancer.enhance_query("Python developer")
print(f"Expanded terms: {result['expanded_terms']}")
```

### Running the Application

```bash
# Using the main entry point
python main.py

# Or using the start script
python scripts/start.py

# Or using the run script
python scripts/run.py
```

The application will start on `http://localhost:5000` by default.

### API Endpoints

Once running, the application provides REST API endpoints:

- `GET /health` - Health check endpoint
- `POST /api/search` - Search for candidates
- `POST /api/classify` - Classify domain
- `POST /api/enhance` - Enhance query

See `app/blueprints/` for detailed API documentation.

## Modules

### Core Modules

- **`ml_domain_classifier.py`**: ML-based domain classification using RandomForest
- **`llm_query_enhancer.py`**: LLM-powered query enhancement
- **`search_system.py`**: Main search system integration
- **`learning_to_rank.py`**: Learning-to-rank algorithms
- **`job_fit_predictor.py`**: Job fit prediction models

### Application Modules

- **`app/`**: Flask application and API endpoints
  - `app/blueprints/`: API route blueprints
  - `app/services.py`: Service layer initialization
  - `app/config.py`: Configuration management
  - `app/schemas/`: Request/response schemas

### Advanced Modules

- **`behavioural_analysis/`**: Behavioral analysis and career pattern recognition
- **`bias_prevention/`**: Bias detection and mitigation
- **`explainable_ai/`**: Explainable AI for transparent scoring
- **`market_intelligence/`**: Market trends and compensation insights
- **`search/`**: Search engines and caching

## API Documentation

### Domain Classification

```python
from adeptai.ml_domain_classifier import get_ml_domain_classifier

classifier = get_ml_domain_classifier()

# Classify domain
domain, confidence = classifier.classify_domain("Software engineer with Python experience")
# Returns: ('technology', 0.85)

# Train model
classifier.train(texts, labels)
classifier.save_model("model/domain_classifier.pkl")
```

### Query Enhancement

```python
from adeptai.llm_query_enhancer import get_llm_query_enhancer

enhancer = get_llm_query_enhancer(provider="openai", model="gpt-4")

# Enhance query
result = enhancer.enhance_query("Python developer")
print(result['synonyms'])  # ['programmer', 'coder', 'engineer', ...]
print(result['related_skills'])  # ['django', 'flask', 'numpy', ...]
print(result['expanded_terms'])  # Combined list of all terms
```

## Development

### Project Structure

```
adeptai/
├── app/                    # Flask application
│   ├── blueprints/         # API route blueprints
│   ├── schemas/           # Request/response schemas
│   └── services.py        # Service layer
├── behavioural_analysis/   # Behavioral analysis modules
├── bias_prevention/       # Bias prevention modules
├── explainable_ai/        # Explainable AI modules
├── market_intelligence/   # Market intelligence modules
├── search/                # Search engines
├── ml_domain_classifier.py # Domain classifier
├── llm_query_enhancer.py  # Query enhancer
├── main.py                # Main entry point
└── requirements.txt       # Dependencies
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Add docstrings to all classes and functions
- Keep functions focused and single-purpose

### Adding New Features

1. Create module in appropriate directory
2. Add comprehensive docstrings
3. Add type hints
4. Write tests in `tests/` directory
5. Update this README

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_services.py

# Run with coverage
pytest --cov=adeptai tests/
```

## Deployment

### Local Deployment

```bash
python main.py
```

### Production Deployment

1. Set environment variables
2. Configure production settings
3. Use production WSGI server (e.g., Gunicorn)
4. Set up monitoring and logging

### SageMaker Deployment

See `../Sagemaker/README.md` for deploying models to AWS SageMaker.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Model Not Found**: Check model paths in configuration
3. **API Key Errors**: Verify API keys in environment variables
4. **SageMaker Connection**: Check AWS credentials and region settings

### Getting Help

- Check logs in `logs/` directory
- Review error messages in console output
- Consult individual module documentation

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]


