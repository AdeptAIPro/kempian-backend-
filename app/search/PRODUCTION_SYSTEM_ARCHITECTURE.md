# Production-Grade Talent Search Engine Architecture
## 90-95% Accuracy System Design

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Query Input                              │
│         (Job Description + Filters + Requirements)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌─────────▼──────────┐
│  Stage 1:      │          │  Stage 2:          │
│  Pre-Filter    │          │  FAISS Retrieval   │
│  (Exact Match) │          │  (Bi-Encoder)      │
│  < 10ms        │          │  < 10ms            │
└───────┬────────┘          └─────────┬──────────┘
        │                             │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────┐
        │  Stage 3:                    │
        │  Cross-Encoder + Ranker      │
        │  (Deep Scoring)              │
        │  < 200ms                     │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────┐
        │  Final Ranking               │
        │  (XGBoost Model)              │
        │  < 50ms                      │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────┐
        │  Ranked Results              │
        │  (Top 20 Candidates)         │
        └──────────────────────────────┘
```

---

## Data Models

### Candidate Schema
```python
{
    "candidate_id": str,
    "name": str,
    "email": str,
    "phone": str,
    
    # Structured Experience
    "experiences": [
        {
            "company": str,
            "title_normalized": str,
            "title_original": str,
            "start_date": date,
            "end_date": date,
            "is_current": bool,
            "duration_months": int,
            "skills": List[str],
            "achievements": List[str],
            "location": str,
            "impact_metrics": List[Dict],
            "seniority_level": str,
            "confidence": float
        }
    ],
    
    # Canonical Skills
    "skill_ids": List[str],  # Canonical skill IDs
    "skills_raw": List[str],  # Original skill text
    "skill_seniorities": Dict[str, str],  # skill_id -> seniority
    
    # Location
    "location_data": {
        "location_id": str,
        "latitude": float,
        "longitude": float,
        "geohash": str,
        "timezone": str,
        "timezone_offset": int,
        "is_remote": bool,
        "willing_to_relocate": bool,
        "relocation_radius_km": float
    },
    
    # Embeddings
    "bi_encoder_embedding": np.ndarray,  # 384-dim
    "experience_embeddings": List[np.ndarray],
    "skill_embeddings": List[np.ndarray],
    
    # Metadata
    "domain": str,  # 'healthcare', 'it/tech', 'general'
    "resume_text": str,
    "education": str,
    "certifications": List[str],
    "updated_at": datetime,
    "source": str,
    "source_reliability": float
}
```

### Job Schema
```python
{
    "job_id": str,
    "title": str,
    "description": str,
    "domain": str,
    
    # Requirements
    "required_skills": List[str],  # Raw skill names
    "required_skill_ids": List[str],  # Canonical IDs
    "preferred_skills": List[str],
    "preferred_skill_ids": List[str],
    "skill_weights": Dict[str, float],  # skill_id -> weight
    
    "required_experience_years": int,
    "required_certifications": List[str],
    "required_education_level": str,
    "required_clearance": Optional[str],
    "required_visa": Optional[str],
    
    # Location
    "location_data": EnhancedLocationData,
    "remote_eligible": bool,
    "timezone_requirement": Optional[int],  # Max offset hours
    
    # Seniority
    "seniority_level": str,  # 'junior', 'mid', 'senior', 'expert'
    
    # Embeddings
    "bi_encoder_embedding": np.ndarray,
    "description_embedding": np.ndarray
}
```

---

## Three-Stage Retrieval Pipeline

### Stage 1: Pre-Filter (Exact Match) - < 10ms

**Purpose**: Filter candidates using strict must-have constraints

**Filters Applied**:
1. **Certifications**: Exact match on required certifications
2. **Visa/Clearance**: Exact match on security clearance or visa status
3. **Remote Eligibility**: Match remote flags
4. **Skill Pre-Filter**: At least 30% of required skills must match
5. **Location Pre-Filter**: Geohash-based radius filtering

**Algorithm**:
```python
def pre_filter_candidates(candidates, job_requirements):
    filtered = []
    
    for candidate in candidates:
        # Certification filter
        if job_requirements.required_certifications:
            candidate_certs = set(candidate.get('certifications', []))
            required_certs = set(job_requirements.required_certifications)
            if not required_certs.issubset(candidate_certs):
                continue
        
        # Visa/Clearance filter
        if job_requirements.required_clearance:
            if candidate.get('clearance') != job_requirements.required_clearance:
                continue
        
        # Remote filter
        if not job_requirements.remote_eligible:
            if candidate['location_data'].is_remote:
                continue
        
        # Skill pre-filter (30% threshold)
        required_skill_ids = set(job_requirements.required_skill_ids)
        candidate_skill_ids = set(candidate.get('skill_ids', []))
        skill_match_ratio = len(required_skill_ids & candidate_skill_ids) / len(required_skill_ids)
        if skill_match_ratio < 0.3:
            continue
        
        # Location pre-filter (geohash)
        if not candidate['location_data'].is_remote:
            job_geohash = job_requirements.location_data.geohash
            candidate_geohash = candidate['location_data'].geohash
            if not self._geohash_in_radius(candidate_geohash, job_geohash, radius_km=100):
                continue
        
        filtered.append(candidate)
    
    return filtered
```

**Latency Budget**: < 10ms for 10K candidates

---

### Stage 2: FAISS Retrieval (Bi-Encoder) - < 10ms

**Purpose**: Fast semantic retrieval using dense embeddings

**Algorithm**:
```python
def faiss_retrieval(query_embedding, candidate_embeddings, top_k=200):
    # Normalize embeddings
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    candidate_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # FAISS search (HNSW index)
    scores, indices = faiss_index.search(query_norm.reshape(1, -1), top_k)
    
    # Return top candidates with scores
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
```

**Latency Budget**: < 10ms for 1M candidates

---

### Stage 3: Cross-Encoder + Ranker - < 200ms

**Purpose**: Deep relevance scoring and final ranking

**Algorithm**:
```python
def cross_encoder_rerank(query, candidate_texts, top_k=200):
    # Create query-candidate pairs
    pairs = [[query, text] for text in candidate_texts]
    
    # Cross-encoder scoring
    cross_scores = cross_encoder.predict(pairs, batch_size=32)
    
    # Sort by score
    scored = [(i, float(score)) for i, score in enumerate(cross_scores)]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return scored[:top_k]
```

**Latency Budget**: < 200ms for 200 candidates

---

## Complete Feature Set (20+ Features)

### Feature Schema
```python
FEATURE_SCHEMA = {
    # Embedding Similarities (3 features)
    'dense_similarity': float,           # Bi-encoder cosine similarity
    'cross_encoder_similarity': float,   # Cross-encoder score
    'tfidf_cosine': float,               # TF-IDF cosine similarity
    
    # Skill Matching (4 features)
    'exact_skill_count': int,            # Number of exact skill matches
    'weighted_skill_match': float,      # Weighted skill match score
    'skill_match_ratio': float,         # Ratio of matched skills
    'preferred_skill_match': float,     # Preferred skills match score
    
    # Experience (3 features)
    'candidate_experience_years': float,
    'job_experience_required': float,
    'experience_match': float,           # Experience match score
    'experience_gap': float,             # Years gap
    
    # Seniority (2 features)
    'seniority_match_distance': float,  # Distance in seniority levels
    'seniority_match': float,          # Seniority match score
    
    # Location (4 features)
    'location_distance_km': float,
    'location_distance_score': float,  # exp(-distance/sigma)
    'timezone_compatibility': float,   # Timezone compatibility score
    'remote_eligible_alignment': float, # Remote flag alignment
    
    # Certifications (1 feature)
    'certification_match': float,
    'certification_match_count': int,
    
    # Education (1 feature)
    'education_match': float,
    
    # Domain (1 feature)
    'domain_match': float,
    
    # Recency (2 features)
    'days_since_resume_update': float,
    'resume_recency_score': float,
    
    # Data Quality (2 features)
    'data_completeness': float,
    'has_resume_text': float,
    
    # Interaction (2 features)
    'candidate_response_rate': float,
    'recruiter_interaction_score': float,
    
    # Source (1 feature)
    'source_reliability': float,
    
    # Skill Diversity (1 feature)
    'skill_diversity': float,
    
    # Achievement Impact (1 feature)
    'achievement_impact_score': float   # Based on quantifiable metrics
}
```

**Total**: 28 features

---

## XGBoost Ranking Model

### Training Configuration
```python
XGBOOST_PARAMS = {
    'objective': 'rank:pairwise',
    'eta': 0.1,
    'max_depth': 8,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'ndcg@10',
    'tree_method': 'hist',
    'device': 'cuda'  # GPU if available
}

TRAINING_CONFIG = {
    'num_boost_round': 500,
    'early_stopping_rounds': 50,
    'train_test_split': 0.2,
    'validation_split': 0.1
}
```

### Feature Importance Weights (Learned)
```python
EXPECTED_FEATURE_IMPORTANCE = {
    'cross_encoder_similarity': 0.25,    # Highest importance
    'dense_similarity': 0.15,
    'weighted_skill_match': 0.15,
    'experience_match': 0.10,
    'location_distance_score': 0.08,
    'seniority_match': 0.07,
    'certification_match': 0.05,
    'domain_match': 0.05,
    'tfidf_cosine': 0.03,
    'resume_recency_score': 0.02,
    # ... other features
}
```

---

## Scoring Formulas

### 1. Distance Score
```
distance_score = exp(-distance_km / sigma)
where sigma = 50.0 (tunable per job seniority)
```

### 2. Skill Match Score
```
skill_match = (
    0.6 * (required_matches / required_total) +
    0.3 * (preferred_matches / preferred_total) +
    0.1 * (hierarchy_matches / total_skills)
)
```

### 3. Experience Match Score
```
if candidate_exp >= job_exp:
    experience_match = 1.0
else:
    gap = job_exp - candidate_exp
    experience_match = max(0.0, 1.0 - (gap * 0.1))
```

### 4. Timezone Compatibility Score
```
timezone_score = max(0.0, 1.0 - (offset_diff / 12.0))
where offset_diff = |candidate_timezone - job_timezone|
```

### 5. Final Ranking Score
```
final_score = XGBoost.predict(features)
where features = [dense_sim, cross_sim, skill_match, ...]
```

---

## Evaluation Metrics

### Target Thresholds
```python
TARGET_METRICS = {
    'precision@5': 0.90,      # 90% of top 5 are relevant
    'precision@10': 0.85,     # 85% of top 10 are relevant
    'ndcg@10': 0.88,          # Normalized DCG at 10
    'mrr': 0.82,              # Mean Reciprocal Rank
    'recall@100': 0.75,       # 75% recall in top 100
    'latency_p50': 0.35,      # 50th percentile < 350ms
    'latency_p95': 0.50,      # 95th percentile < 500ms
    'latency_p99': 0.70       # 99th percentile < 700ms
}
```

### A/B Testing Plan
```python
AB_TEST_CONFIG = {
    'test_duration_days': 14,
    'traffic_split': 0.5,  # 50% control, 50% treatment
    'primary_metric': 'precision@5',
    'secondary_metrics': ['ndcg@10', 'interview_rate', 'hire_rate'],
    'min_sample_size': 1000,
    'significance_level': 0.05,
    'rollback_trigger': {
        'precision_drop': 0.05,  # 5% drop triggers rollback
        'latency_increase': 0.20  # 20% latency increase
    }
}
```

---

## Continuous Learning Pipeline

### Feedback Collection
```python
FEEDBACK_SIGNALS = {
    'positive': [
        'candidate_viewed',
        'candidate_shortlisted',
        'interview_scheduled',
        'interview_completed',
        'offer_extended',
        'offer_accepted',
        'hired'
    ],
    'negative': [
        'candidate_rejected',
        'interview_cancelled',
        'offer_declined',
        'not_selected'
    ],
    'neutral': [
        'candidate_contacted',
        'no_response'
    ]
}
```

### Retraining Schedule
- **Weekly**: Incremental update with new feedback
- **Bi-weekly**: Full retrain with all historical data
- **Monthly**: Model evaluation and comparison
- **Quarterly**: Architecture review and optimization

### Model Versioning
```python
MODEL_VERSION = {
    'version_id': str,
    'trained_at': datetime,
    'training_data_size': int,
    'metrics': Dict[str, float],
    'feature_importance': Dict[str, float],
    'rollback_available': bool
}
```

---

## Latency Budgets

```
Total Pipeline: < 450ms
├── Pre-Filter: < 10ms
├── FAISS Retrieval: < 10ms
├── Cross-Encoder: < 200ms
├── Feature Extraction: < 50ms
├── XGBoost Ranking: < 50ms
└── Result Formatting: < 30ms
```

---

## Implementation Files

1. `production_skill_ontology.py` - Complete skill ontology
2. `experience_parser.py` - Structured experience parsing
3. `enhanced_geocoding.py` - Geocoding with timezone
4. `hybrid_embedding_service.py` - Bi-encoder + Cross-encoder
5. `ranking_feature_extractor.py` - 28-feature extractor
6. `three_stage_retrieval.py` - Complete retrieval pipeline
7. `xgboost_ranking_model.py` - Training and serving
8. `feedback_collector.py` - Feedback collection
9. `continuous_learning.py` - Retraining pipeline
10. `evaluation_metrics.py` - Metrics and A/B testing

---

## Next Steps

1. ✅ Review architecture
2. ⏳ Implement three-stage retrieval
3. ⏳ Build XGBoost training pipeline
4. ⏳ Set up feedback collection
5. ⏳ Deploy A/B testing framework
6. ⏳ Monitor and iterate

