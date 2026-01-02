# AdeptAI Search Algorithm Architecture

## Overview

The AdeptAI Search Algorithm is a comprehensive, multi-layered recruitment search system that combines multiple AI/ML technologies to provide accurate, fast, and fair candidate matching. The system integrates 15+ advanced components including instant search, dense retrieval, behavioral analysis, bias prevention, and market intelligence.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Search Flow](#search-flow)
4. [Integration Layers](#integration-layers)
5. [Configuration](#configuration)
6. [Performance Optimizations](#performance-optimizations)
7. [Usage Examples](#usage-examples)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OptimizedSearchSystem                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Caching    │  │   Search      │  │  Ranking    │    │
│  │   Layer      │  │   Engines     │  │  & Scoring  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Enrichment │  │   Bias       │  │  Explainable│    │
│  │   Services   │  │   Prevention │  │  AI         │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

```
OptimizedSearchSystem
├── Caching Layer
│   ├── OptimizedCandidateCache (Primary)
│   └── Regular Search Cache (Secondary)
│
├── Search Engines (Priority Order)
│   ├── Instant Search Engine
│   ├── Dense Retrieval Matcher (FAISS)
│   └── Hybrid Search (Keyword + Semantic)
│
├── Query Enhancement
│   ├── LLM Query Enhancer (GPT-4/Claude)
│   ├── Custom Query Enhancer
│   └── Intent Classification
│
├── Matching & Scoring
│   ├── Enhanced Matcher (XGBoost/Ensemble)
│   ├── Multi-Model Embedding Service
│   ├── Custom Cross-Encoder
│   └── Keyword + Semantic Hybrid
│
├── Ranking & Reranking
│   ├── Learning-to-Rank (LightGBM)
│   ├── RL Ranking Agent (Actor-Critic)
│   └── Cross-Encoder Reranking
│
├── Enrichment Services
│   ├── Behavioral Analysis Pipeline
│   ├── Market Intelligence
│   └── Domain Classification
│
├── Bias Prevention
│   ├── Query Sanitization
│   ├── Resume Sanitization
│   └── Diversity Monitoring
│
└── Explanation & Transparency
    └── Explainable AI (Feature Contributions)
```

---

## Core Components

### 1. Caching Layer

#### OptimizedCandidateCache
- **Purpose**: Ultra-fast candidate retrieval with sub-millisecond response times
- **Features**:
  - Memory-efficient data structures
  - Compressed cache files (gzip)
  - Smart indexing (skills, names, experience, domain)
  - Async processing support
  - Automatic cache warming
- **Location**: `search/optimized_cache.py`
- **Usage**: First cache check for all queries

#### Regular Search Cache
- **Purpose**: LRU cache for recent search results
- **Features**: Simple hash-based caching with size limits
- **Usage**: Secondary cache layer

### 2. Search Engines

#### Instant Search Engine
- **Purpose**: Pre-loaded candidate index for instant results
- **Features**:
  - Pre-processed candidate data at startup
  - Optimized indexes (skill_index, word_index, experience_index)
  - Sub-millisecond query response
  - Automatic cache persistence
- **Location**: `search/instant_search.py`
- **Priority**: 1 (First search method)

#### Dense Retrieval Matcher (FAISS)
- **Purpose**: High-performance semantic search using vector embeddings
- **Features**:
  - FAISS-based similarity search
  - SentenceTransformer embeddings
  - Automatic index building
  - ProductionMatcher with reranking
- **Location**: `enhanced_models/dense_retrieval.py`
- **Priority**: 2 (Second search method)

#### Hybrid Search
- **Purpose**: Combines keyword and semantic matching
- **Features**:
  - Keyword scoring (TF-IDF based)
  - Semantic vector retrieval
  - Weighted combination (60% keyword, 40% semantic)
  - Domain-aware filtering
- **Priority**: 3 (Fallback search method)

### 3. Query Enhancement

#### LLM Query Enhancer
- **Purpose**: Context-aware query expansion using Large Language Models
- **Features**:
  - GPT-4/Claude integration
  - Synonym generation
  - Skill inference
  - Intent analysis
  - Contextual expansion
- **Location**: `llm_query_enhancer.py`
- **Priority**: 1 (Primary enhancer)

#### Custom Query Enhancer
- **Purpose**: Rule-based query expansion
- **Features**: Pattern matching, domain-specific expansion
- **Priority**: 2 (Fallback)

### 4. Matching & Scoring

#### Enhanced Matcher
- **Purpose**: ML-based candidate-job matching
- **Features**:
  - XGBoost/Ensemble models
  - Feature engineering (13+ features)
  - Model stability evaluation
  - Continuous learning from feedback
- **Location**: `semantic_function/matcher/enhanced_matcher.py`
- **Usage**: Additional scoring layer (30% weight)

#### Multi-Model Embedding Service
- **Purpose**: Multiple embedding models for semantic similarity
- **Features**:
  - General model (all-mpnet-base-v2)
  - Fast model (all-MiniLM-L6-v2)
  - Embedding caching
  - Batch processing
- **Location**: `utils/enhanced_embeddings.py`
- **Usage**: Semantic refinement (20% weight)

### 5. Ranking & Reranking

#### Learning-to-Rank (LTR)
- **Purpose**: Optimal feature weighting for ranking
- **Features**:
  - LightGBM Ranker (XGBoost fallback)
  - Feature extraction from query-candidate pairs
  - Trained on historical data
- **Location**: `learning_to_rank.py`
- **Usage**: Reranking after initial search

#### RL Ranking Agent
- **Purpose**: Adaptive, personalized ranking based on user interactions
- **Features**:
  - Actor-Critic architecture
  - PyTorch implementation
  - Continuous learning from feedback
  - Personalized ranking
- **Location**: `rl_ranking_agent.py`
- **Usage**: Adaptive reranking

#### Cross-Encoder Reranking
- **Purpose**: Deep semantic relevance scoring
- **Features**:
  - Neural cross-encoder models
  - Query-candidate pair scoring
  - High accuracy but slower
- **Usage**: Final reranking pass

### 6. Enrichment Services

#### Behavioral Analysis Pipeline
- **Purpose**: Comprehensive candidate profiling
- **Features**:
  - Leadership assessment
  - Collaboration analysis
  - Innovation scoring
  - Adaptability metrics
  - Emotional intelligence
  - Technical depth analysis
  - Cultural alignment
  - Career trajectory prediction
- **Location**: `behavioural_analysis/pipeline.py`
- **Usage**: Adds behavioral scores to results (30% weight)

#### Market Intelligence
- **Purpose**: Market data enrichment for candidates
- **Features**:
  - Talent availability analysis
  - Competitive intelligence
  - Skill demand forecasting
  - Compensation benchmarking
  - Market insights generation
- **Location**: `market_intelligence/`
- **Usage**: Adds market context to results

#### Domain Classification
- **Purpose**: Domain-aware candidate filtering
- **Features**:
  - ML-based classifier (RandomForest + TF-IDF)
  - Custom LLM classifier
  - Pattern-based fallback
- **Location**: `ml_domain_classifier.py`, `search_system.py`
- **Usage**: Filters irrelevant candidates

### 7. Bias Prevention

#### Query Sanitization
- **Purpose**: Remove bias-related terms from queries
- **Features**:
  - Protected characteristic detection
  - Term removal and replacement
  - Sanitization reporting
- **Location**: `bias_prevention/sanitizer.py`

#### Resume Sanitization
- **Purpose**: Remove bias signals from candidate data
- **Features**: PII removal, demographic data filtering

#### Diversity Monitoring
- **Purpose**: Assess and improve diversity in results
- **Features**:
  - Diversity scoring
  - Representation balance analysis
  - Bias flag detection
  - Compliance reporting
- **Location**: `bias_prevention/monitor.py`

### 8. Explainable AI

#### ExplainableRecruitmentAI
- **Purpose**: Generate human-readable explanations for candidate selection
- **Features**:
  - Feature contribution analysis
  - Confidence levels
  - Risk factors identification
  - Strength areas highlighting
  - Recommendations
- **Location**: `explainable_ai/models/recruitment_ai.py`
- **Usage**: Generates explanations for all results

---

## Search Flow

### Complete Search Pipeline

```
1. Query Input
   ↓
2. Query Sanitization (Bias Prevention)
   ↓
3. Cache Check (Optimized Cache → Regular Cache)
   ↓
4. Query Enhancement (LLM → Custom → Fallback)
   ↓
5. Search Execution (Priority Order)
   ├─ Instant Search Engine
   ├─ Dense Retrieval Matcher (FAISS)
   └─ Hybrid Search (Keyword + Semantic)
   ↓
6. Matching & Scoring
   ├─ Enhanced Matcher (XGBoost)
   ├─ Multi-Model Embedding Service
   └─ Score Combination
   ↓
7. Reranking
   ├─ Cross-Encoder Reranking
   ├─ Learning-to-Rank
   └─ RL Ranking Agent
   ↓
8. Enrichment
   ├─ Behavioral Analysis
   ├─ Market Intelligence
   └─ Domain Classification
   ↓
9. Bias Prevention Assessment
   ├─ Diversity Monitoring
   └─ Compliance Reporting
   ↓
10. Explanation Generation
    └─ Explainable AI
    ↓
11. Final Results
```

### Detailed Step-by-Step Flow

#### Step 1: Query Input & Sanitization
```python
query = "senior python developer with 5+ years experience"
# Sanitization removes bias terms if present
sanitized_query = sanitizer.sanitize_query(query)
```

#### Step 2: Cache Check
```python
# Check optimized cache first
if optimized_cache.has_results(query):
    return optimized_cache.get_results(query)

# Check regular cache
if regular_cache.has_results(query):
    return regular_cache.get_results(query)
```

#### Step 3: Query Enhancement
```python
# LLM enhancement (GPT-4/Claude)
enhanced_query = {
    'original_query': query,
    'expanded_terms': ['python', 'developer', 'senior', 'software engineer', ...],
    'synonyms': ['programmer', 'coder', 'engineer', ...],
    'skills_inferred': ['Python', 'Django', 'Flask', 'FastAPI', ...],
    'intent': 'technical_role'
}
```

#### Step 4: Search Execution
```python
# Priority 1: Instant Search
if instant_search_available:
    results = instant_search.search(query, limit=top_k * 2)
    
# Priority 2: Dense Retrieval
elif dense_retrieval_available:
    results = production_matcher.find_matches(query, top_k=top_k * 2)
    
# Priority 3: Hybrid Search
else:
    results = hybrid_search(query, enhanced_query, top_k)
```

#### Step 5: Matching & Scoring
```python
for candidate in results:
    # Enhanced Matcher scoring
    enhanced_score = enhanced_matcher.calculate_match_score(query, candidate)
    
    # Multi-Model Embedding similarity
    embedding_similarity = embedding_service.similarity(query, candidate)
    
    # Combined score
    final_score = 0.7 * original_score + 0.3 * enhanced_score + 0.2 * embedding_similarity
```

#### Step 6: Reranking
```python
# Cross-encoder reranking
results = cross_encoder.rerank(query, results)

# Learning-to-Rank
results = ltr_model.predict(query, results)

# RL Ranking Agent
results = rl_agent.rank(query, results, user_context)
```

#### Step 7: Enrichment
```python
for candidate in results:
    # Behavioral analysis
    behavioral_profile = behavioral_pipeline.analyze(candidate)
    candidate['behavioral_analysis'] = behavioral_profile
    
    # Market intelligence
    market_data = market_intelligence.get_data(candidate)
    candidate['market_intelligence'] = market_data
```

#### Step 8: Bias Prevention
```python
# Diversity assessment
diversity_assessment = bias_monitor.assess_diversity(query, results)
for candidate in results:
    candidate['diversity_info'] = diversity_assessment
```

#### Step 9: Explanation Generation
```python
for candidate in results:
    explanation = explainable_ai.explain_candidate_selection(
        candidate_profile=candidate,
        job_query=query,
        match_scores=scores
    )
    candidate['ai_explanation'] = explanation
```

---

## Integration Layers

### Layer 1: Caching (Fastest)
- **Components**: Optimized Cache, Regular Cache
- **Purpose**: Immediate response for repeated queries
- **Performance**: Sub-millisecond

### Layer 2: Search Engines
- **Components**: Instant Search, Dense Retrieval, Hybrid Search
- **Purpose**: Initial candidate retrieval
- **Performance**: 10-100ms

### Layer 3: Query Enhancement
- **Components**: LLM Enhancer, Custom Enhancer
- **Purpose**: Improve query understanding
- **Performance**: 100-500ms (LLM), 10-50ms (Custom)

### Layer 4: Matching & Scoring
- **Components**: Enhanced Matcher, Multi-Model Embeddings
- **Purpose**: Accurate candidate scoring
- **Performance**: 50-200ms

### Layer 5: Reranking
- **Components**: Cross-Encoder, LTR, RL Agent
- **Purpose**: Optimal candidate ordering
- **Performance**: 100-500ms

### Layer 6: Enrichment
- **Components**: Behavioral Analysis, Market Intelligence
- **Purpose**: Additional candidate insights
- **Performance**: 200-1000ms

### Layer 7: Bias Prevention
- **Components**: Sanitization, Diversity Monitoring
- **Purpose**: Fair and compliant results
- **Performance**: 10-50ms

### Layer 8: Explanation
- **Components**: Explainable AI
- **Purpose**: Transparent decision-making
- **Performance**: 50-200ms

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LLM_PROVIDER=openai  # or 'anthropic'
LLM_MODEL=gpt-4  # or 'claude-3-opus'

# Behavioral Analysis
BEHAVIOURAL_ANALYSIS_CONFIG=lightweight  # or 'comprehensive', 'minimal'

# LinkedIn Integration (for Enhanced Matcher)
LINKEDIN_USERNAME=your_username
LINKEDIN_PASSWORD=your_password

# Database
DYNAMODB_TABLE_NAME=user-resume-metadata
AWS_REGION=ap-south-1
```

### Search Parameters

```python
results = search_system.search(
    query="python developer",
    top_k=10,
    
    # Feature toggles
    use_optimized_cache=True,      # Use optimized cache
    use_instant_search=True,       # Use instant search
    use_dense_retrieval=True,      # Use dense retrieval
    use_enhanced_matcher=True,     # Use enhanced matcher
    use_multi_model_embeddings=True, # Use multi-model embeddings
    include_behavioural_analysis=True, # Include behavioral analysis
    include_market_intelligence=True, # Include market intelligence
    enable_bias_prevention=True    # Enable bias prevention
)
```

### Component Initialization

```python
search_system = OptimizedSearchSystem(
    background_init=False,  # Background preprocessing for large datasets
    use_custom_llm=True     # Use custom LLM components
)
```

---

## Performance Optimizations

### 1. Caching Strategy
- **Optimized Cache**: First-level cache with compressed storage
- **Regular Cache**: LRU cache for recent queries
- **Embedding Cache**: Cached embeddings to avoid recomputation

### 2. Lazy Loading
- Components initialized only when needed
- Heavy dependencies loaded on first use
- Background preprocessing for large datasets

### 3. Batch Processing
- Embedding generation in batches
- Candidate processing in chunks
- Parallel processing where possible

### 4. Index Optimization
- FAISS indexes for fast similarity search
- Pre-built indexes for instant search
- Optimized data structures

### 5. Smart Fallbacks
- Graceful degradation if components unavailable
- Multiple search methods with priority ordering
- Fallback to simpler methods if advanced ones fail

---

## Usage Examples

### Basic Search
```python
from search_system import OptimizedSearchSystem

# Initialize system
search_system = OptimizedSearchSystem()

# Simple search
results = search_system.search("python developer", top_k=10)

# Access results
for result in results:
    print(f"{result['full_name']}: {result['final_score']:.2f}")
    print(f"Explanation: {result['ai_explanation']}")
```

### Advanced Search with Configuration
```python
# Search with specific features enabled
results = search_system.search(
    query="senior data scientist with ML experience",
    top_k=20,
    use_instant_search=True,
    use_dense_retrieval=True,
    include_behavioural_analysis=True,
    include_market_intelligence=True,
    enable_bias_prevention=True
)

# Access enriched results
for result in results:
    print(f"Name: {result['full_name']}")
    print(f"Score: {result['final_score']:.2f}")
    print(f"Behavioral Analysis: {result.get('behavioural_analysis', {})}")
    print(f"Market Intelligence: {result.get('market_intelligence', {})}")
    print(f"Diversity Info: {result.get('diversity_info', {})}")
```

### Performance Monitoring
```python
# Get search statistics
stats = search_system.get_stats()
print(f"Total searches: {stats['total_searches']}")
print(f"Average response time: {stats['avg_response_time']:.2f}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}%")
```

---

## Algorithm Details

### Score Calculation

The final score combines multiple factors:

```python
final_score = (
    0.40 * keyword_score +           # Keyword matching
    0.30 * semantic_score +          # Semantic similarity
    0.15 * enhanced_matcher_score +  # ML-based matching
    0.10 * embedding_similarity +     # Multi-model embeddings
    0.05 * behavioral_score          # Behavioral analysis
)

# After reranking
final_score = (
    0.60 * ltr_score +               # Learning-to-Rank
    0.30 * cross_encoder_score +     # Cross-encoder
    0.10 * rl_ranking_score          # RL ranking
)
```

### Ranking Pipeline

1. **Initial Search**: Retrieve candidates using search engines
2. **Scoring**: Calculate match scores using multiple methods
3. **Reranking**: Apply LTR, cross-encoder, and RL ranking
4. **Enrichment**: Add behavioral analysis and market intelligence
5. **Final Sort**: Sort by final_score (descending)

### Domain Filtering

```python
# Classify query domain
query_domain, query_confidence = domain_classifier.classify_domain(query)

# Filter candidates by domain
for candidate in candidates:
    candidate_domain, candidate_confidence = domain_classifier.classify_domain(candidate)
    
    # Filter if domain mismatch
    if should_filter_candidate(candidate_domain, query_domain, candidate_confidence, query_confidence):
        continue  # Skip candidate
```

---

## Integration Status

### Fully Integrated Systems

✅ **Instant Search** - Fully integrated, priority 1 search method  
✅ **Dense Retrieval Matcher** - Fully integrated, priority 2 search method  
✅ **Enhanced Matcher** - Fully integrated, additional scoring layer  
✅ **Multi-Model Embedding Service** - Fully integrated, semantic refinement  
✅ **Optimized Cache** - Fully integrated, first-level caching  
✅ **Behavioral Analysis** - Fully integrated, automatic enrichment  
✅ **Bias Prevention** - Fully integrated, automatic activation  
✅ **Market Intelligence** - Fully integrated, automatic enrichment  
✅ **Explainable AI** - Fully integrated, automatic explanation generation  
✅ **Learning-to-Rank** - Fully integrated, automatic reranking  
✅ **RL Ranking Agent** - Fully integrated, adaptive ranking  
✅ **LLM Query Enhancer** - Fully integrated, priority query enhancer  
✅ **Domain Classification** - Fully integrated, automatic filtering  

---

## Future Enhancements

### Planned Integrations
- Job Fit Predictor (for ranking integration)
- NER Skill Extractor (for better skill matching)
- Multi-Armed Bandit (for strategy selection)
- Skill Demand Forecaster (for market insights)
- Candidate Clustering (for similar candidate discovery)

### Performance Improvements
- Distributed caching
- GPU acceleration for embeddings
- Async processing for all components
- Query result streaming

---

## Conclusion

The AdeptAI Search Algorithm is a comprehensive, multi-layered system that combines state-of-the-art AI/ML technologies to provide accurate, fast, and fair candidate matching. With 15+ integrated components, intelligent caching, and multiple search strategies, the system delivers high-quality results while maintaining transparency and fairness.

For questions or issues, please refer to the individual component documentation or contact the development team.

