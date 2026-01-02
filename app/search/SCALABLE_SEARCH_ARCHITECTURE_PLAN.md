# Global Scalable Search Architecture Plan
## Self-Learning System for 1M+ Candidates with 90-95% Accuracy

### Executive Summary
This document outlines a comprehensive plan to transform the current search system into a self-learning, globally scalable platform capable of handling 10+ lakhs (1+ million) candidates with sub-second response times and 90-95% accuracy.

---

## Current State Analysis

### Existing Capabilities
- ✅ Parallel processing with ThreadPoolExecutor
- ✅ FAISS vector search integration
- ✅ Redis caching layer
- ✅ TF-IDF and sentence transformer embeddings
- ✅ Domain detection (Healthcare, IT/Tech)
- ✅ DynamoDB metadata storage
- ✅ Current capacity: ~100K candidates

### Limitations
- ❌ Not optimized for 1M+ candidates
- ❌ Limited exact field matching
- ❌ No self-learning/feedback loop
- ❌ Single-node architecture
- ❌ No distributed indexing
- ❌ Limited accuracy optimization

---

## Target Architecture

### Phase 1: Foundation - Distributed Indexing & Storage (Weeks 1-4)

#### 1.1 Multi-Tier Indexing Strategy

**Primary Index: FAISS with HNSW (Hierarchical Navigable Small World)**
```python
# Replace IndexFlatIP with IndexHNSWFlat for 1000x faster search
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node
index.hnsw.efConstruction = 200  # Build-time search width
index.hnsw.efSearch = 64  # Query-time search width
```

**Benefits:**
- O(log n) search complexity vs O(n) for flat index
- Handles 1M+ vectors efficiently
- Sub-10ms search time for top-K queries

**Secondary Index: Elasticsearch for Exact Matching**
```python
# Elasticsearch schema for structured queries
{
  "mappings": {
    "properties": {
      "skills": {"type": "keyword", "fields": {"text": {"type": "text"}}},
      "location": {"type": "keyword", "fields": {"geo": {"type": "geo_point"}}},
      "experience_years": {"type": "integer_range"},
      "education_level": {"type": "keyword"},
      "certifications": {"type": "keyword"},
      "availability": {"type": "keyword"},
      "salary_range": {"type": "integer_range"},
      "domain": {"type": "keyword"}
    }
  }
}
```

**Tertiary Index: Redis for Hot Data**
- Cache top 10K most frequently accessed candidates
- LRU eviction policy
- Sub-millisecond access

#### 1.2 Sharding Strategy

**Geographic Sharding**
```python
# Shard by region for global search
SHARDS = {
    'us-east': ['New York', 'Boston', 'Washington'],
    'us-west': ['San Francisco', 'Seattle', 'Los Angeles'],
    'europe': ['London', 'Berlin', 'Paris'],
    'asia': ['Mumbai', 'Bangalore', 'Singapore'],
    'global': []  # Fallback for unmatched locations
}
```

**Domain-Based Sharding**
```python
# Separate indices for different domains
DOMAIN_SHARDS = {
    'healthcare': faiss.IndexHNSWFlat(384, 32),
    'it-tech': faiss.IndexHNSWFlat(384, 32),
    'general': faiss.IndexHNSWFlat(384, 32)
}
```

**Benefits:**
- Parallel search across shards
- Reduced index size per shard
- Faster query processing

#### 1.3 Database Architecture

**PostgreSQL for Structured Data**
```sql
-- Optimized candidate table with indexes
CREATE TABLE candidates (
    id VARCHAR(128) PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    location VARCHAR(255),
    experience_years INTEGER,
    education_level VARCHAR(100),
    domain VARCHAR(50),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    embedding_id INTEGER,  -- Reference to FAISS index
    -- GIN indexes for fast text search
    skills GIN,
    certifications GIN
);

-- Composite indexes for common queries
CREATE INDEX idx_location_domain ON candidates(location, domain);
CREATE INDEX idx_experience_domain ON candidates(experience_years, domain);
CREATE INDEX idx_skills_gin ON candidates USING GIN(skills);
```

**Vector Store: FAISS with Memory-Mapped Files**
```python
# Memory-mapped FAISS index for 1M+ vectors
faiss.write_index(index, "/data/faiss_indexes/candidates_1M.index")
index = faiss.read_index("/data/faiss_indexes/candidates_1M.index", faiss.IO_FLAG_MMAP)
```

---

### Phase 2: Exact Field Matching Engine (Weeks 5-6)

#### 2.1 Structured Query Parser

```python
class ExactMatchQueryParser:
    """Parse queries into structured filters"""
    
    def parse(self, query: str) -> Dict:
        return {
            'text_query': self._extract_text(query),
            'skills': self._extract_skills(query),
            'location': self._extract_location(query),
            'experience_range': self._extract_experience(query),
            'education': self._extract_education(query),
            'certifications': self._extract_certifications(query),
            'salary_range': self._extract_salary(query),
            'availability': self._extract_availability(query)
        }
```

#### 2.2 Hybrid Search Strategy

**Step 1: Exact Filtering (Elasticsearch)**
```python
# Filter candidates by exact criteria
exact_filters = {
    "bool": {
        "must": [
            {"term": {"skills": "Python"}},
            {"range": {"experience_years": {"gte": 5, "lte": 10}}},
            {"term": {"location": "San Francisco"}},
            {"term": {"education_level": "Master's"}}
        ]
    }
}
filtered_ids = elasticsearch.search(filter=exact_filters, size=10000)
```

**Step 2: Semantic Ranking (FAISS)**
```python
# Rank filtered candidates by semantic similarity
query_embedding = embedding_service.encode(query)
candidate_embeddings = faiss_index.reconstruct_batch(filtered_ids)
scores = cosine_similarity(query_embedding, candidate_embeddings)
```

**Step 3: Hybrid Scoring**
```python
final_score = (
    0.4 * exact_match_score +  # Exact field matches
    0.4 * semantic_score +      # Semantic similarity
    0.1 * experience_score +    # Experience relevance
    0.1 * location_score        # Location proximity
)
```

---

### Phase 3: Self-Learning System (Weeks 7-10)

#### 3.1 Feedback Collection System

```python
class FeedbackCollector:
    """Collect and store user feedback"""
    
    def record_feedback(self, search_id: str, candidate_id: str, 
                       action: str, relevance_score: float):
        """
        Actions: 'viewed', 'saved', 'contacted', 'hired', 'rejected'
        relevance_score: 0.0 to 1.0
        """
        feedback = {
            'search_id': search_id,
            'candidate_id': candidate_id,
            'action': action,
            'relevance_score': relevance_score,
            'timestamp': datetime.utcnow(),
            'query_features': self._extract_query_features(search_id)
        }
        self.feedback_db.insert(feedback)
```

#### 3.2 Learning Models

**A. Query-Candidate Relevance Model**
```python
class RelevanceLearner:
    """Learn from user interactions"""
    
    def train(self, feedback_data: List[Dict]):
        """
        Train a model to predict candidate relevance
        Features:
        - Query embedding
        - Candidate embedding
        - Exact match features (skills, location, etc.)
        - Historical success rate
        - User interaction patterns
        """
        X = self._extract_features(feedback_data)
        y = [f['relevance_score'] for f in feedback_data]
        
        # Use XGBoost for fast, accurate predictions
        model = xgboost.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        model.fit(X, y)
        return model
```

**B. Query Understanding Model**
```python
class QueryUnderstandingModel:
    """Learn to better understand user queries"""
    
    def improve_parsing(self, query: str, feedback: Dict):
        """
        Learn from corrections:
        - If user filters by location after search, learn location importance
        - If user saves candidates with specific skills, learn skill weights
        """
        # Update query parser weights based on feedback
        self.parser.update_weights(feedback)
```

**C. Ranking Model**
```python
class RankingModel:
    """Learn optimal ranking from user behavior"""
    
    def learn_ranking(self, search_results: List, user_actions: List):
        """
        Learn from:
        - Which candidates were viewed first
        - Which candidates were saved
        - Which candidates led to hires
        """
        # Use learning-to-rank algorithm (LambdaMART)
        ranker = LambdaMART()
        ranker.train(search_results, user_actions)
        return ranker
```

#### 3.3 Continuous Learning Pipeline

```python
class ContinuousLearningPipeline:
    """Automated learning from feedback"""
    
    def __init__(self):
        self.feedback_queue = Queue()
        self.model_updater = ModelUpdater()
        
    def process_feedback(self):
        """Process feedback in batches"""
        while True:
            batch = self.feedback_queue.get_batch(size=1000, timeout=60)
            if batch:
                # Retrain models with new feedback
                self.model_updater.update_models(batch)
                
    def schedule_retraining(self):
        """Retrain models periodically"""
        # Daily retraining with all feedback
        schedule.every().day.at("02:00").do(self.full_retrain)
        
        # Incremental updates every hour
        schedule.every().hour.do(self.incremental_update)
```

---

### Phase 4: Performance Optimization (Weeks 11-12)

#### 4.1 Caching Strategy

**Multi-Level Cache**
```python
class MultiLevelCache:
    """L1: In-memory, L2: Redis, L3: Database"""
    
    def get(self, key: str):
        # L1: In-memory (fastest, ~1ms)
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # L2: Redis (fast, ~5ms)
        if self.redis_client.exists(key):
            value = self.redis_client.get(key)
            self.memory_cache[key] = value  # Promote to L1
            return value
        
        # L3: Database (slower, ~50ms)
        value = self.database.get(key)
        self.redis_client.setex(key, 3600, value)  # Cache in L2
        return value
```

**Query Result Cache**
```python
# Cache search results for common queries
cache_key = hashlib.md5(f"{query}_{filters}_{top_k}".encode()).hexdigest()
if cache_key in cache:
    return cache[cache_key]
```

#### 4.2 Parallel Processing

**Async Search Pipeline**
```python
async def parallel_search(query: str, filters: Dict):
    """Search across multiple indices in parallel"""
    
    # Run searches in parallel
    tasks = [
        self.faiss_search(query),
        self.elasticsearch_filter(filters),
        self.redis_hot_data_search(query)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Merge and rank results
    return self.merge_results(results)
```

**Batch Embedding Generation**
```python
# Generate embeddings in batches of 1000
def generate_embeddings_batch(candidates: List, batch_size=1000):
    with ThreadPoolExecutor(max_workers=8) as executor:
        batches = [candidates[i:i+batch_size] 
                  for i in range(0, len(candidates), batch_size)]
        futures = [executor.submit(embedding_service.encode_batch, batch) 
                  for batch in batches]
        embeddings = [f.result() for f in futures]
    return np.vstack(embeddings)
```

#### 4.3 Index Optimization

**Incremental Index Updates**
```python
class IncrementalIndexUpdater:
    """Update FAISS index incrementally"""
    
    def add_candidates(self, new_candidates: List):
        """Add new candidates without rebuilding entire index"""
        new_embeddings = self.generate_embeddings(new_candidates)
        self.faiss_index.add(new_embeddings.astype('float32'))
        
    def remove_candidates(self, candidate_ids: List):
        """Mark candidates as deleted (lazy deletion)"""
        for candidate_id in candidate_ids:
            self.deleted_ids.add(candidate_id)
            
    def rebuild_if_needed(self):
        """Rebuild index if too many deletions"""
        if len(self.deleted_ids) > self.faiss_index.ntotal * 0.1:
            self.rebuild_index()
```

---

### Phase 5: Accuracy Enhancement (Weeks 13-16)

#### 5.1 Multi-Model Ensemble

```python
class EnsembleMatcher:
    """Combine multiple models for better accuracy"""
    
    def match(self, query: str, candidate: Dict) -> float:
        scores = {
            'semantic': self.semantic_model.score(query, candidate),
            'exact': self.exact_matcher.score(query, candidate),
            'skill': self.skill_matcher.score(query, candidate),
            'experience': self.experience_matcher.score(query, candidate),
            'location': self.location_matcher.score(query, candidate)
        }
        
        # Weighted ensemble
        final_score = (
            0.30 * scores['semantic'] +
            0.25 * scores['exact'] +
            0.20 * scores['skill'] +
            0.15 * scores['experience'] +
            0.10 * scores['location']
        )
        
        return final_score
```

#### 5.2 Skill Normalization & Matching

```python
class SkillNormalizer:
    """Normalize and match skills intelligently"""
    
    def __init__(self):
        self.skill_synonyms = {
            'python': ['python3', 'python 3', 'py'],
            'javascript': ['js', 'ecmascript', 'node.js'],
            'react': ['reactjs', 'react.js']
        }
        
    def normalize_skill(self, skill: str) -> str:
        """Normalize skill name"""
        skill_lower = skill.lower().strip()
        for canonical, variants in self.skill_synonyms.items():
            if skill_lower in variants or skill_lower == canonical:
                return canonical
        return skill_lower
        
    def match_skills(self, query_skills: List[str], 
                    candidate_skills: List[str]) -> float:
        """Fuzzy skill matching"""
        normalized_query = [self.normalize_skill(s) for s in query_skills]
        normalized_candidate = [self.normalize_skill(s) for s in candidate_skills]
        
        matches = len(set(normalized_query) & set(normalized_candidate))
        return matches / len(normalized_query) if normalized_query else 0.0
```

#### 5.3 Experience Matching

```python
class ExperienceMatcher:
    """Intelligent experience matching"""
    
    def match_experience(self, query_exp: str, candidate_exp: int) -> float:
        """
        Parse queries like:
        - "5+ years experience"
        - "Senior level (8-10 years)"
        - "Mid-level (3-5 years)"
        """
        query_range = self._parse_experience_range(query_exp)
        
        if query_range['min'] <= candidate_exp <= query_range['max']:
            return 1.0
        elif candidate_exp < query_range['min']:
            # Penalize under-qualified
            return max(0.0, 1.0 - (query_range['min'] - candidate_exp) * 0.1)
        else:
            # Slight penalty for over-qualified
            return max(0.7, 1.0 - (candidate_exp - query_range['max']) * 0.05)
```

#### 5.4 Location Matching

```python
class LocationMatcher:
    """Intelligent location matching with geocoding"""
    
    def __init__(self):
        self.geocoder = Geocoder()
        self.location_hierarchy = {
            'San Francisco': ['SF', 'Bay Area', 'Silicon Valley'],
            'New York': ['NYC', 'NY', 'Manhattan', 'Brooklyn']
        }
        
    def match_location(self, query_location: str, 
                      candidate_location: str) -> float:
        """Match locations with fuzzy matching"""
        # Exact match
        if query_location.lower() == candidate_location.lower():
            return 1.0
        
        # Hierarchical match (city -> region -> country)
        query_coords = self.geocoder.geocode(query_location)
        candidate_coords = self.geocoder.geocode(candidate_location)
        
        if query_coords and candidate_coords:
            distance_km = self._calculate_distance(query_coords, candidate_coords)
            
            # Score based on distance
            if distance_km < 50:
                return 1.0
            elif distance_km < 100:
                return 0.8
            elif distance_km < 500:
                return 0.5
            else:
                return 0.2
        
        return 0.0
```

---

### Phase 6: Global Search Infrastructure (Weeks 17-20)

#### 6.1 Distributed Architecture

**Microservices Architecture**
```
┌─────────────────┐
│  API Gateway     │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│Search │ │Index  │
│Service│ │Service│
└───┬───┘ └──┬────┘
    │        │
┌───▼────────▼───┐
│  Data Layer     │
│  (FAISS/ES/DB)  │
└────────────────┘
```

**Load Balancing**
```python
class SearchLoadBalancer:
    """Distribute search requests across nodes"""
    
    def route_search(self, query: str) -> str:
        """Route to least loaded search node"""
        nodes = self.get_available_nodes()
        node = min(nodes, key=lambda n: n.current_load)
        return node.search(query)
```

#### 6.2 Data Replication

**Multi-Region Replication**
```python
class ReplicationManager:
    """Replicate indices across regions"""
    
    def replicate_to_region(self, region: str):
        """Replicate FAISS index to region"""
        # Copy index file
        # Sync candidate data
        # Update routing table
        pass
```

#### 6.3 CDN for Static Data

```python
# Serve candidate profiles via CDN
# Cache frequently accessed profiles
# Reduce database load
```

---

## Implementation Roadmap

### Week 1-4: Foundation
- [ ] Implement FAISS HNSW index
- [ ] Set up Elasticsearch for exact matching
- [ ] Design sharding strategy
- [ ] Optimize database schema with indexes

### Week 5-6: Exact Matching
- [ ] Build query parser
- [ ] Implement hybrid search
- [ ] Test exact field matching

### Week 7-10: Self-Learning
- [ ] Build feedback collection system
- [ ] Implement learning models
- [ ] Set up continuous learning pipeline
- [ ] Test learning effectiveness

### Week 11-12: Performance
- [ ] Implement multi-level caching
- [ ] Optimize parallel processing
- [ ] Set up incremental index updates
- [ ] Performance testing (1M candidates)

### Week 13-16: Accuracy
- [ ] Implement ensemble matching
- [ ] Build skill normalization
- [ ] Improve experience/location matching
- [ ] Accuracy testing (target 90-95%)

### Week 17-20: Global Infrastructure
- [ ] Set up distributed architecture
- [ ] Implement load balancing
- [ ] Set up replication
- [ ] Global deployment

---

## Performance Targets

### Response Time
- **Target**: < 500ms for 1M candidates
- **Breakdown**:
  - Exact filtering (Elasticsearch): < 50ms
  - Semantic search (FAISS): < 100ms
  - Ranking & scoring: < 200ms
  - Result formatting: < 150ms

### Accuracy
- **Target**: 90-95% relevance
- **Metrics**:
  - Precision@10: > 0.90
  - Recall@100: > 0.85
  - NDCG@20: > 0.92

### Scalability
- **Target**: 1M+ candidates
- **Capacity**:
  - FAISS HNSW: 10M+ vectors
  - Elasticsearch: 100M+ documents
  - Throughput: 1000+ queries/second

---

## Technology Stack

### Core Technologies
- **Vector Search**: FAISS (HNSW index)
- **Exact Search**: Elasticsearch 8.x
- **Database**: PostgreSQL 15+ with pgvector
- **Cache**: Redis 7+ (cluster mode)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)

### ML/AI
- **Learning**: XGBoost, LightGBM
- **Ranking**: LambdaMART
- **NLP**: spaCy, NLTK

### Infrastructure
- **Containerization**: Docker, Kubernetes
- **Message Queue**: RabbitMQ / Apache Kafka
- **Monitoring**: Prometheus, Grafana
- **Logging**: ELK Stack

---

## Cost Estimation

### Infrastructure (Monthly)
- **Compute**: $2,000-5,000 (8-16 nodes)
- **Storage**: $500-1,000 (FAISS indices + DB)
- **Elasticsearch**: $1,000-2,000 (3-node cluster)
- **Redis**: $500-1,000 (cluster)
- **Total**: ~$4,000-9,000/month

### Development
- **Team**: 3-4 engineers
- **Timeline**: 20 weeks
- **Cost**: $200K-300K

---

## Success Metrics

### Performance
- ✅ Search time < 500ms for 1M candidates
- ✅ 1000+ queries/second throughput
- ✅ 99.9% uptime

### Accuracy
- ✅ 90-95% relevance score
- ✅ Precision@10 > 0.90
- ✅ User satisfaction > 4.5/5.0

### Business
- ✅ 50% reduction in time-to-hire
- ✅ 30% increase in candidate quality
- ✅ 40% improvement in recruiter productivity

---

## Risk Mitigation

### Technical Risks
1. **Index Size**: Use memory-mapped files, sharding
2. **Query Latency**: Aggressive caching, parallel processing
3. **Accuracy**: Ensemble models, continuous learning
4. **Scalability**: Horizontal scaling, load balancing

### Operational Risks
1. **Data Quality**: Validation pipeline, data cleaning
2. **Model Drift**: Continuous monitoring, retraining
3. **Infrastructure**: Multi-region deployment, backups

---

## Next Steps

1. **Immediate (Week 1)**
   - Set up development environment
   - Create FAISS HNSW index prototype
   - Design database schema changes

2. **Short-term (Month 1)**
   - Implement exact matching engine
   - Set up Elasticsearch cluster
   - Build query parser

3. **Medium-term (Months 2-3)**
   - Implement self-learning system
   - Optimize performance
   - Improve accuracy

4. **Long-term (Months 4-5)**
   - Deploy global infrastructure
   - Scale to 1M+ candidates
   - Monitor and optimize

---

## Conclusion

This architecture provides a scalable, self-learning search system capable of handling 1M+ candidates with sub-second response times and 90-95% accuracy. The phased approach allows for incremental development and testing, minimizing risk while maximizing value delivery.

