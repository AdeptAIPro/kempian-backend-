# Accuracy Improvement: 70-80% → 90%+ Implementation Tickets

## Priority 1: Critical Foundation (Weeks 0-2) - **START HERE**

### Ticket 1.1: Skill Canonicalization System
**Priority**: P0 - BLOCKER  
**Effort**: 3 days  
**Dependencies**: None  
**Impact**: +15-20% accuracy improvement

**Description**: Build normalized skill ontology to map synonyms, abbreviations, and variations to canonical skill IDs.

**Acceptance Criteria**:
- [ ] Skill normalization database with 500+ common tech/healthcare skills
- [ ] Fuzzy matching + embedding similarity for skill mapping
- [ ] Human verification interface for high-volume mappings
- [ ] Update candidate ingestion to store `skill_ids` alongside raw skills
- [ ] Update search to match against skill IDs, not raw text
- [ ] Unit tests with 95%+ accuracy on skill matching

**Files to Create**:
- `backend/app/search/skill_canonicalizer.py`
- `backend/app/search/data/skill_ontology.json`
- `backend/app/search/migrations/add_skill_ids_to_candidates.py`

---

### Ticket 1.2: Geocoding & Location Standardization
**Priority**: P0 - BLOCKER  
**Effort**: 2 days  
**Dependencies**: None  
**Impact**: +10-15% accuracy improvement

**Description**: Geocode all candidate and job locations, store lat/lon, standardized location_id, and geohash for fast filtering.

**Acceptance Criteria**:
- [ ] Geocode all existing candidate locations (batch job)
- [ ] Store lat/lon, location_id (country/state/city), geohash in candidate records
- [ ] Geohash-based pre-filtering for location queries
- [ ] Haversine distance calculation for final scoring
- [ ] Handle relocation willingness and remote flags
- [ ] Migration script for existing data

**Files to Create**:
- `backend/app/search/geocoding_service.py`
- `backend/app/search/location_matcher.py`
- `backend/app/search/migrations/add_geocoding_to_candidates.py`

---

### Ticket 1.3: Structured Experience Parsing
**Priority**: P0 - BLOCKER  
**Effort**: 4 days  
**Dependencies**: None  
**Impact**: +8-12% accuracy improvement

**Description**: Parse resumes into normalized experience rows with structured fields.

**Acceptance Criteria**:
- [ ] Extract structured experience: {company, role_normalized, start_date, end_date, location, skills[]}
- [ ] Flag ambiguous parses for human review
- [ ] Store normalized experience in candidate records
- [ ] Use structured experience for matching (not just text)
- [ ] Validation rules to catch common parsing errors

**Files to Create**:
- `backend/app/search/experience_parser.py`
- `backend/app/search/migrations/add_structured_experience.py`

---

### Ticket 1.4: Exact Pre-Filters for Must-Have Constraints
**Priority**: P0 - BLOCKER  
**Effort**: 2 days  
**Dependencies**: Ticket 1.1, 1.2  
**Impact**: +5-8% accuracy improvement

**Description**: Add exact matching pre-filters for certifications, visa status, clearances before semantic search.

**Acceptance Criteria**:
- [ ] Pre-filter on certifications (exact match)
- [ ] Pre-filter on visa/clearance requirements
- [ ] Pre-filter on remote eligibility
- [ ] Integration with existing search pipeline
- [ ] Performance: <50ms for pre-filtering

**Files to Modify**:
- `backend/app/search/service.py` (add pre-filter step)
- `backend/app/search/exact_filter.py` (new)

---

## Priority 2: Hybrid Search & Ranking (Weeks 2-6)

### Ticket 2.1: FAISS Bi-Encoder + Cross-Encoder Pipeline
**Priority**: P1 - HIGH  
**Effort**: 5 days  
**Dependencies**: Ticket 1.1, 1.2  
**Impact**: +10-15% accuracy improvement

**Description**: Implement two-tier embedding system: fast bi-encoder for retrieval, accurate cross-encoder for re-ranking.

**Acceptance Criteria**:
- [ ] Bi-encoder (sentence-transformers) for FAISS retrieval
- [ ] Cross-encoder for re-ranking top 200 candidates
- [ ] Domain-specific embedding models (Healthcare, IT/Tech, General)
- [ ] Integration with existing FAISS index
- [ ] Latency: <300ms for bi-encoder, <200ms for cross-encoder re-rank

**Files to Create**:
- `backend/app/search/hybrid_embedding_service.py`
- `backend/app/search/cross_encoder_reranker.py`
- `backend/app/search/domain_embeddings.py`

---

### Ticket 2.2: XGBoost Feature Extractor & Training Pipeline
**Priority**: P1 - HIGH  
**Effort**: 6 days  
**Dependencies**: Ticket 2.1, 1.1, 1.2, 1.3  
**Impact**: +15-20% accuracy improvement

**Description**: Build feature extraction pipeline and train XGBoost/LightGBM ranking model on historical hire data.

**Acceptance Criteria**:
- [ ] Feature extractor with all 15+ features (dense similarity, cross-encoder score, TF-IDF, skill matches, distance, etc.)
- [ ] Training data collection from historical hires/interviews
- [ ] XGBoost rank:pairwise model training
- [ ] Model evaluation: nDCG@10, Precision@5 metrics
- [ ] Model serving integration
- [ ] A/B testing framework

**Files to Create**:
- `backend/app/search/ranking_feature_extractor.py`
- `backend/app/search/ranking_model_trainer.py`
- `backend/app/search/ranking_model_serving.py`
- `backend/app/search/training_data_collector.py`

---

### Ticket 2.3: Weighted Skill Matching with Hierarchy
**Priority**: P1 - HIGH  
**Effort**: 3 days  
**Dependencies**: Ticket 1.1  
**Impact**: +8-12% accuracy improvement

**Description**: Implement skill hierarchy (parent-child relationships) and weighted matching based on skill importance.

**Acceptance Criteria**:
- [ ] Skill hierarchy graph (e.g., react → javascript)
- [ ] Partial credit for parent-child skill matches
- [ ] Weighted skill matching (required vs preferred)
- [ ] Skill-level detection (junior/mid/senior per skill)
- [ ] Integration with ranking features

**Files to Create**:
- `backend/app/search/skill_hierarchy.py`
- `backend/app/search/weighted_skill_matcher.py`

---

## Priority 3: Advanced Features (Weeks 6-10)

### Ticket 3.1: Distance Scoring with Geohash Pre-filtering
**Priority**: P2 - MEDIUM  
**Effort**: 3 days  
**Dependencies**: Ticket 1.2  
**Impact**: +5-8% accuracy improvement

**Description**: Implement geohash-based pre-filtering and weighted distance scoring.

**Acceptance Criteria**:
- [ ] Geohash prefix indexing for candidates
- [ ] Radius-based pre-filtering (fetch geohash neighbors)
- [ ] Weighted distance scoring: exp(-distance_km / sigma)
- [ ] Timezone compatibility checks
- [ ] Remote/relocation flag integration

**Files to Modify**:
- `backend/app/search/location_matcher.py` (enhance)
- `backend/app/search/service.py` (integrate)

---

### Ticket 3.2: Domain-Specific Fine-Tuning
**Priority**: P2 - MEDIUM  
**Effort**: 5 days  
**Dependencies**: Ticket 2.1  
**Impact**: +8-12% accuracy improvement

**Description**: Fine-tune sentence-transformers on domain-specific job-resume pairs.

**Acceptance Criteria**:
- [ ] Fine-tune models for Healthcare domain
- [ ] Fine-tune models for IT/Tech domain
- [ ] Fine-tune models for General domain
- [ ] Model evaluation and comparison
- [ ] Model deployment and A/B testing

**Files to Create**:
- `backend/app/search/domain_model_finetuner.py`
- `backend/app/search/training_scripts/finetune_healthcare.py`
- `backend/app/search/training_scripts/finetune_tech.py`

---

### Ticket 3.3: Continuous Learning & Feedback Loop
**Priority**: P2 - MEDIUM  
**Effort**: 4 days  
**Dependencies**: Ticket 2.2  
**Impact**: +5-10% accuracy improvement (over time)

**Description**: Build feedback collection system and automated retraining pipeline.

**Acceptance Criteria**:
- [ ] Recruiter feedback collection (structured labels)
- [ ] Implicit signal capture (click-to-contact, message sent)
- [ ] Weekly/bi-weekly model retraining
- [ ] Validation holdout from live traffic
- [ ] Model versioning and rollback capability

**Files to Create**:
- `backend/app/search/feedback_collector.py`
- `backend/app/search/continuous_learning_pipeline.py`

---

## Priority 4: Performance & Scale (Weeks 10-12)

### Ticket 4.1: FAISS HNSW Index Upgrade
**Priority**: P3 - LOW (but needed for scale)  
**Effort**: 3 days  
**Dependencies**: Ticket 2.1  
**Impact**: Performance improvement (10-100x faster)

**Description**: Upgrade from IndexFlatIP to IndexHNSWFlat for 1M+ candidate support.

**Acceptance Criteria**:
- [ ] HNSW index implementation (M=32, efConstruction=200, efSearch=200)
- [ ] Index migration from flat to HNSW
- [ ] Performance benchmarks (<10ms search time)
- [ ] Memory-mapped index support
- [ ] Incremental update capability

**Files to Modify**:
- `backend/app/search/ultra_fast_parallel_search.py`

---

### Ticket 4.2: Telemetry & Evaluation Framework
**Priority**: P3 - LOW  
**Effort**: 3 days  
**Dependencies**: Ticket 2.2  
**Impact**: Monitoring and optimization

**Description**: Build comprehensive metrics tracking and evaluation framework.

**Acceptance Criteria**:
- [ ] Precision@1, @5, @10 tracking
- [ ] nDCG@10, MRR metrics
- [ ] Time-to-contact and interview-rate tracking
- [ ] Model drift detection
- [ ] A/B test framework
- [ ] Dashboard for metrics visualization

**Files to Create**:
- `backend/app/search/metrics_tracker.py`
- `backend/app/search/evaluation_framework.py`
- `backend/app/search/ab_testing.py`

---

## Success Metrics

### Target Improvements (A/B Test):
- **Precision@5**: +15-25% improvement
- **nDCG@10**: Significant increase and stabilization
- **Interview-to-hire ratio**: Improvement or time-to-hire reduction
- **Latency**: Maintain <500ms tail latency

### Monitoring:
- Track metrics daily/weekly
- Auto-rollback if precision drops by >5% vs baseline
- Weekly model retraining with fresh labels

---

## Implementation Order

**Week 0-2 (Foundation)**:
1. Ticket 1.1: Skill Canonicalization
2. Ticket 1.2: Geocoding & Location
3. Ticket 1.3: Structured Experience
4. Ticket 1.4: Exact Pre-Filters

**Week 2-4 (Core Ranking)**:
5. Ticket 2.1: Hybrid Embeddings
6. Ticket 2.2: XGBoost Ranking Model
7. Ticket 2.3: Weighted Skill Matching

**Week 4-6 (Integration & Testing)**:
8. Ticket 3.1: Distance Scoring
9. Ticket 3.2: Domain Fine-Tuning
10. Ticket 3.3: Continuous Learning

**Week 6-8 (Scale & Monitor)**:
11. Ticket 4.1: FAISS HNSW Upgrade
12. Ticket 4.2: Telemetry Framework

---

## Notes

- **Start with Tickets 1.1-1.4** - these provide the foundation for everything else
- **Don't skip data quality** - bad labels = garbage model
- **Human-in-the-loop** - manual corrections accelerate learning
- **A/B test everything** - measure real impact, not just metrics

