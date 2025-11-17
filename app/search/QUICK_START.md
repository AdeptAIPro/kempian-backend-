# Quick Start Guide: Scalable Search System
## Getting Started with 1M+ Candidate Search

---

## Overview

This guide provides a quick overview of the scalable search architecture and how to get started with implementation.

### Key Goals
- ✅ Handle 10+ lakhs (1M+) candidates
- ✅ Sub-second search response times (< 500ms)
- ✅ 90-95% accuracy
- ✅ Exact field matching (experience, location, education, skills)
- ✅ Self-learning from user feedback
- ✅ Global search capability

---

## Architecture Summary

### Three-Tier Search System

```
┌─────────────────────────────────────────┐
│         Query Input                      │
│    (Natural Language + Filters)          │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌──────▼──────────┐
│ Exact Match │  │ Semantic Search │
│(Elasticsearch)│  │   (FAISS HNSW)  │
└──────┬──────┘  └──────┬───────────┘
       │                │
       └───────┬────────┘
               │
       ┌───────▼──────────┐
       │  Hybrid Scoring  │
       │  + Learning Model │
       └───────┬──────────┘
               │
       ┌───────▼──────────┐
       │  Ranked Results  │
       └──────────────────┘
```

### Core Components

1. **FAISS HNSW Index** - Fast approximate nearest neighbor search
2. **Elasticsearch** - Exact field matching and filtering
3. **Hybrid Engine** - Combines exact + semantic matching
4. **Learning System** - Learns from user feedback
5. **Query Parser** - Extracts structured filters from natural language

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Set up scalable indexing infrastructure

**Tasks**:
1. Replace FAISS IndexFlatIP with IndexHNSWFlat
2. Set up Elasticsearch cluster
3. Design sharding strategy (geographic + domain-based)
4. Optimize PostgreSQL schema with indexes

**Key Files**:
- `ultra_fast_parallel_search.py` - Upgrade to HNSW
- `elasticsearch_service.py` - New Elasticsearch integration
- Database migration scripts

**Expected Outcome**:
- Index can handle 1M+ candidates
- Search time < 100ms for FAISS queries
- Elasticsearch filtering < 50ms

---

### Phase 2: Exact Matching (Weeks 5-6)
**Goal**: Implement exact field matching

**Tasks**:
1. Build query parser for structured filters
2. Implement hybrid search engine
3. Add exact match scoring

**Key Files**:
- `query_parser.py` - Parse natural language to filters
- `hybrid_search_engine.py` - Combine exact + semantic

**Expected Outcome**:
- Can filter by exact skills, location, experience, education
- Hybrid scoring improves relevance

---

### Phase 3: Self-Learning (Weeks 7-10)
**Goal**: Implement feedback-based learning

**Tasks**:
1. Build feedback collection system
2. Implement learning models (XGBoost)
3. Set up continuous learning pipeline

**Key Files**:
- `feedback_system.py` - Collect user feedback
- `learning_model.py` - Train relevance models

**Expected Outcome**:
- System learns from user interactions
- Accuracy improves over time
- Models retrain automatically

---

### Phase 4: Performance (Weeks 11-12)
**Goal**: Optimize for sub-second response

**Tasks**:
1. Implement multi-level caching
2. Optimize parallel processing
3. Set up incremental index updates

**Expected Outcome**:
- Search time < 500ms for 1M candidates
- 1000+ queries/second throughput

---

### Phase 5: Accuracy (Weeks 13-16)
**Goal**: Achieve 90-95% accuracy

**Tasks**:
1. Implement ensemble matching
2. Improve skill/location/experience matching
3. Fine-tune scoring weights

**Expected Outcome**:
- Precision@10 > 0.90
- Recall@100 > 0.85
- User satisfaction > 4.5/5.0

---

### Phase 6: Global Infrastructure (Weeks 17-20)
**Goal**: Deploy globally scalable system

**Tasks**:
1. Set up distributed architecture
2. Implement load balancing
3. Set up multi-region replication

**Expected Outcome**:
- Global search capability
- 99.9% uptime
- Multi-region deployment

---

## Quick Start: Phase 1 Implementation

### Step 1: Install Dependencies

```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install elasticsearch
pip install xgboost
pip install redis
pip install sentence-transformers
```

### Step 2: Upgrade FAISS Index

```python
# In ultra_fast_parallel_search.py
# Replace:
# index = faiss.IndexFlatIP(dimension)

# With:
index = faiss.IndexHNSWFlat(dimension, 32)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64
```

### Step 3: Set Up Elasticsearch

```bash
# Install Elasticsearch
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

### Step 4: Initialize Services

```python
from app.search.elasticsearch_service import ElasticsearchService
from app.search.ultra_fast_parallel_search import ScalableFAISSEngine

# Initialize Elasticsearch
es_service = ElasticsearchService(hosts=['localhost:9200'])

# Initialize FAISS
faiss_engine = ScalableFAISSEngine(dimension=384)
```

### Step 5: Index Candidates

```python
# Load candidates
candidates = load_candidates_from_database()  # Your function

# Generate embeddings
embeddings = []
candidate_ids = []
for candidate in candidates:
    text = f"{candidate['name']} {candidate['skills']} {candidate['experience']}"
    embedding = embedding_service.encode_single(text)
    embeddings.append(embedding)
    candidate_ids.append(candidate['id'])

embeddings = np.array(embeddings)

# Build FAISS index
faiss_engine.build_hnsw_index(embeddings)
faiss_engine.candidate_ids = candidate_ids
faiss_engine.candidate_data = {c['id']: c for c in candidates}

# Index in Elasticsearch
es_service.bulk_index_candidates(candidates)
```

### Step 6: Perform Search

```python
# Simple search
query = "Python developer with 5+ years experience in San Francisco"
results = faiss_engine.search(
    embedding_service.encode_single(query),
    top_k=20
)

# Hybrid search (exact + semantic)
from app.search.hybrid_search_engine import HybridSearchEngine

hybrid_engine = HybridSearchEngine(
    faiss_engine,
    es_service,
    embedding_service
)

filters = {
    'skills': ['Python', 'Django'],
    'location': 'San Francisco',
    'experience_range': (5, 10)
}

results = hybrid_engine.search(query, filters, top_k=20)
```

---

## Performance Benchmarks

### Current System (100K candidates)
- Search time: ~2-5 seconds
- Accuracy: ~70-80%
- Throughput: ~10 queries/second

### Target System (1M+ candidates)
- Search time: < 500ms
- Accuracy: 90-95%
- Throughput: 1000+ queries/second

### Optimization Techniques

1. **FAISS HNSW**: 1000x faster than flat index
2. **Elasticsearch Filtering**: Pre-filter before semantic search
3. **Caching**: Multi-level cache (memory → Redis → DB)
4. **Parallel Processing**: Search shards in parallel
5. **Incremental Updates**: Update index without full rebuild

---

## Monitoring & Metrics

### Key Metrics to Track

1. **Performance**
   - Search latency (p50, p95, p99)
   - Throughput (queries/second)
   - Index size and memory usage

2. **Accuracy**
   - Precision@10, Precision@20
   - Recall@100
   - NDCG@20
   - User satisfaction scores

3. **System Health**
   - FAISS index health
   - Elasticsearch cluster status
   - Cache hit rates
   - Error rates

### Monitoring Setup

```python
# Example monitoring
import time
from prometheus_client import Counter, Histogram

search_latency = Histogram('search_latency_seconds', 'Search latency')
search_count = Counter('search_total', 'Total searches')

def monitored_search(query):
    start = time.time()
    results = search(query)
    duration = time.time() - start
    
    search_latency.observe(duration)
    search_count.inc()
    
    return results
```

---

## Testing Strategy

### Unit Tests
- Query parser accuracy
- Exact matching logic
- Hybrid scoring

### Integration Tests
- End-to-end search flow
- Feedback collection
- Learning pipeline

### Load Tests
- 1M candidate index
- 1000 concurrent queries
- Stress testing

### Accuracy Tests
- Precision/Recall metrics
- User satisfaction surveys
- A/B testing

---

## Troubleshooting

### Common Issues

**Issue**: Search too slow
- **Solution**: Check FAISS index type (use HNSW), enable caching, optimize Elasticsearch queries

**Issue**: Low accuracy
- **Solution**: Improve query parsing, tune hybrid weights, collect more feedback for learning

**Issue**: Index too large
- **Solution**: Implement sharding, use memory-mapped files, compress embeddings

**Issue**: Memory issues
- **Solution**: Use memory-mapped FAISS, implement pagination, optimize batch sizes

---

## Next Steps

1. **Review Architecture Plan** (`SCALABLE_SEARCH_ARCHITECTURE_PLAN.md`)
2. **Follow Implementation Guide** (`IMPLEMENTATION_GUIDE.md`)
3. **Start with Phase 1** (Foundation)
4. **Test incrementally** after each phase
5. **Monitor metrics** and optimize

---

## Resources

- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **Elasticsearch Guide**: https://www.elastic.co/guide/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Architecture Plan**: `SCALABLE_SEARCH_ARCHITECTURE_PLAN.md`
- **Implementation Guide**: `IMPLEMENTATION_GUIDE.md`

---

## Support

For questions or issues:
1. Check the detailed architecture plan
2. Review implementation guide code examples
3. Test with small dataset first
4. Monitor metrics and adjust

---

**Good luck with your implementation! 🚀**

