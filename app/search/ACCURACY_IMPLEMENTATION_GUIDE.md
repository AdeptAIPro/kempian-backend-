# Accuracy Improvement Implementation Guide
## Step-by-Step Integration Instructions

This guide shows how to integrate the new accuracy improvement components into your existing search system.

---

## Quick Start: Integration Checklist

### Step 1: Install Dependencies

```bash
pip install sentence-transformers
pip install geopy
pip install pygeohash  # or geohash2
pip install xgboost  # or lightgbm
pip install scikit-learn
```

### Step 2: Initialize Components in Your Service

Add to `backend/app/search/service.py`:

```python
# At the top with other imports
from app.search.skill_canonicalizer import get_skill_canonicalizer
from app.search.geocoding_service import get_geocoding_service
from app.search.hybrid_embedding_service import get_hybrid_embedding_service
from app.search.ranking_feature_extractor import get_feature_extractor

# In your AdeptAIMastersAlgorithm.__init__ or semantic_match method:
class AdeptAIMastersAlgorithm:
    def __init__(self):
        # ... existing code ...
        
        # Initialize new components
        self.skill_canonicalizer = get_skill_canonicalizer()
        self.geocoding_service = get_geocoding_service()
        self.embedding_service = get_hybrid_embedding_service()
        self.feature_extractor = get_feature_extractor()
```

### Step 3: Update Candidate Ingestion to Store Canonical Skills

In your candidate loading/ingestion code:

```python
from app.search.skill_canonicalizer import canonicalize_skills

def process_candidate(candidate_data: Dict) -> Dict:
    """Process and canonicalize candidate skills"""
    raw_skills = candidate_data.get('skills', [])
    
    # Canonicalize skills
    canonicalized = canonicalize_skills(raw_skills, threshold=0.85)
    
    # Store both raw and canonical
    candidate_data['skills_raw'] = raw_skills
    candidate_data['skill_ids'] = [s[0] for s in canonicalized if s[0]]  # skill_id
    candidate_data['skills_canonical'] = [s[1] for s in canonicalized if s[0]]  # canonical_name
    
    return candidate_data
```

### Step 4: Geocode Candidate Locations

```python
from app.search.geocoding_service import geocode_location

def process_candidate_location(candidate_data: Dict) -> Dict:
    """Geocode and store location data"""
    location_str = candidate_data.get('location', '')
    is_remote = candidate_data.get('is_remote', False)
    
    # Geocode location
    location_data = geocode_location(location_str, is_remote)
    
    if location_data:
        candidate_data['location_data'] = {
            'location_id': location_data.location_id,
            'latitude': location_data.latitude,
            'longitude': location_data.longitude,
            'geohash': location_data.geohash,
            'standardized_name': location_data.standardized_name,
            'is_remote': location_data.is_remote
        }
    
    return candidate_data
```

### Step 5: Integrate Hybrid Search in semantic_match

Update your `semantic_match` method:

```python
def semantic_match(self, job_description: str, use_gpt4_reranking=True):
    """Enhanced semantic match with hybrid embeddings"""
    
    # ... existing candidate loading code ...
    
    # Step 1: Extract job requirements
    job_location = self._extract_location(job_description)
    job_required_skills = self._extract_required_skills(job_description)
    
    # Step 2: Pre-filter candidates (exact matching)
    filtered_candidates = self._pre_filter_candidates(
        candidates, job_required_skills, job_location
    )
    
    # Step 3: Encode query
    job_domain = self._detect_domain(job_description)
    query_embedding = self.embedding_service.encode_query(job_description, job_domain)
    
    # Step 4: Get candidate embeddings (from FAISS or compute)
    candidate_texts = [self._candidate_to_text(c) for c in filtered_candidates]
    candidate_embeddings = self.embedding_service.encode_candidates_batch(
        candidate_texts, job_domain
    )
    
    # Step 5: Hybrid search (FAISS + cross-encoder)
    hybrid_results = self.embedding_service.hybrid_search(
        query=job_description,
        candidate_texts=candidate_texts,
        candidate_embeddings=candidate_embeddings,
        query_embedding=query_embedding,
        top_k_retrieval=200,
        top_k_final=20,
        domain=job_domain
    )
    
    # Step 6: Extract features for ranking
    ranked_results = []
    for idx, bi_score, cross_score in hybrid_results:
        candidate = filtered_candidates[idx]
        
        # Extract comprehensive features
        features = self.feature_extractor.extract_features(
            job_description=job_description,
            candidate=candidate,
            job_location=job_location,
            job_required_skills=job_required_skills,
            dense_similarity=bi_score,
            cross_encoder_score=cross_score
        )
        
        # Calculate final score (can use XGBoost model here)
        final_score = self._calculate_final_score(features, bi_score, cross_score)
        
        # Format result
        result = self._format_result(candidate, final_score, features)
        ranked_results.append(result)
    
    # Sort by final score
    ranked_results.sort(key=lambda x: x.get('matchScore', 0), reverse=True)
    
    return {
        'results': ranked_results[:20],
        'total_candidates': len(filtered_candidates),
        'algorithm_used': 'hybrid-embedding-enhanced'
    }
```

### Step 6: Add Pre-Filtering Method

```python
def _pre_filter_candidates(
    self,
    candidates: List[Dict],
    job_required_skills: List[str],
    job_location: Optional[str]
) -> List[Dict]:
    """Pre-filter candidates using exact matching"""
    filtered = []
    
    # Canonicalize job skills
    job_skill_ids = []
    for skill in job_required_skills:
        result = self.skill_canonicalizer.canonicalize_skill(skill)
        if result and result[0]:
            job_skill_ids.append(result[0])
    
    # Geocode job location
    job_location_data = None
    if job_location:
        job_location_data = self.geocoding_service.geocode_location(job_location)
    
    for candidate in candidates:
        # Skill pre-filter
        candidate_skill_ids = candidate.get('skill_ids', [])
        if job_skill_ids:
            skill_match, _ = self.skill_canonicalizer.calculate_skill_match_score(
                job_skill_ids, candidate_skill_ids
            )
            if skill_match < 0.3:  # Require at least 30% skill match
                continue
        
        # Location pre-filter
        if job_location_data and not job_location_data.is_remote:
            candidate_location_data = candidate.get('location_data')
            if candidate_location_data:
                distance_km = self.geocoding_service.calculate_distance_km(
                    job_location_data,
                    LocationData(**candidate_location_data)
                )
                if distance_km > 100.0:  # Filter candidates >100km away
                    continue
        
        filtered.append(candidate)
    
    return filtered
```

### Step 7: Calculate Final Score (Before XGBoost Model)

```python
def _calculate_final_score(
    self,
    features: Dict[str, float],
    bi_score: float,
    cross_score: float
) -> float:
    """Calculate final ranking score"""
    
    # Weighted combination (tune these weights)
    final_score = (
        0.35 * features.get('cross_encoder_score', 0) +
        0.25 * features.get('dense_similarity', 0) +
        0.15 * features.get('weighted_skill_match', 0) +
        0.10 * features.get('experience_match', 0) +
        0.05 * features.get('location_distance_score', 0) +
        0.05 * features.get('certification_match', 0) +
        0.05 * features.get('education_match', 0)
    )
    
    # Normalize to 0-100
    return min(100.0, max(0.0, final_score * 100))
```

---

## Training XGBoost Ranking Model

### Step 1: Collect Training Data

```python
from app.search.ranking_feature_extractor import get_feature_extractor
import pandas as pd

def collect_training_data():
    """Collect features and labels from historical hires"""
    
    # Load historical data: job_id, candidate_id, outcome (1=hired, 0=not)
    historical_data = load_historical_hires()  # Your function
    
    feature_extractor = get_feature_extractor()
    
    training_data = []
    for record in historical_data:
        job = load_job(record['job_id'])
        candidate = load_candidate(record['candidate_id'])
        
        # Extract features
        features = feature_extractor.extract_features(
            job_description=job['description'],
            candidate=candidate,
            job_location=job.get('location'),
            job_required_skills=job.get('required_skills', [])
        )
        
        # Add label
        features['label'] = 1 if record['outcome'] == 'hired' else 0
        features['job_id'] = record['job_id']
        features['candidate_id'] = record['candidate_id']
        
        training_data.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    
    return df
```

### Step 2: Train XGBoost Model

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

def train_ranking_model(training_df):
    """Train XGBoost ranking model"""
    
    # Separate features and labels
    feature_cols = [col for col in training_df.columns 
                    if col not in ['label', 'job_id', 'candidate_id']]
    
    X = training_df[feature_cols].values
    y = training_df['label'].values
    
    # Group by job_id for pairwise ranking
    groups = training_df.groupby('job_id').size().values
    
    # Split data
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42
    )
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(groups_train)
    
    dtest = xgb.DMatrix(X_test, label=y_test)
    dtest.set_group(groups_test)
    
    # Train model
    params = {
        'objective': 'rank:pairwise',
        'eta': 0.1,
        'max_depth': 6,
        'eval_metric': 'ndcg@10'
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=20
    )
    
    # Save model
    model.save_model('ranking_model.json')
    
    return model
```

### Step 3: Use Model for Ranking

```python
import xgboost as xgb

class RankingModel:
    def __init__(self, model_path='ranking_model.json'):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
    
    def predict_score(self, features: Dict[str, float]) -> float:
        """Predict ranking score from features"""
        # Convert features to array in correct order
        feature_array = np.array([[
            features.get('dense_similarity', 0),
            features.get('cross_encoder_score', 0),
            features.get('tfidf_score', 0),
            features.get('weighted_skill_match', 0),
            # ... all other features in order
        ]])
        
        dmatrix = xgb.DMatrix(feature_array)
        score = self.model.predict(dmatrix)[0]
        
        return float(score)
```

---

## Migration Scripts

### Migrate Existing Candidates

```python
# backend/app/search/migrations/migrate_candidates_to_canonical_skills.py

from app.search.skill_canonicalizer import get_skill_canonicalizer
from app.search.geocoding_service import get_geocoding_service

def migrate_candidates():
    """Migrate existing candidates to use canonical skills and geocoding"""
    
    canonicalizer = get_skill_canonicalizer()
    geocoding = get_geocoding_service()
    
    # Load all candidates from your database
    candidates = load_all_candidates()  # Your function
    
    updated = 0
    for candidate in candidates:
        # Canonicalize skills
        raw_skills = candidate.get('skills', [])
        canonicalized = canonicalizer.canonicalize_skill_list(raw_skills)
        candidate['skill_ids'] = [s[0] for s in canonicalized if s[0]]
        
        # Geocode location
        location_str = candidate.get('location', '')
        location_data = geocoding.geocode_location(location_str)
        if location_data:
            candidate['location_data'] = {
                'location_id': location_data.location_id,
                'latitude': location_data.latitude,
                'longitude': location_data.longitude,
                'geohash': location_data.geohash
            }
        
        # Save updated candidate
        save_candidate(candidate)  # Your function
        updated += 1
        
        if updated % 100 == 0:
            print(f"Migrated {updated} candidates...")
    
    print(f"Migration complete: {updated} candidates updated")
```

---

## Testing

### Unit Tests

```python
# test_skill_canonicalizer.py

def test_skill_canonicalization():
    from app.search.skill_canonicalizer import get_skill_canonicalizer
    
    canonicalizer = get_skill_canonicalizer()
    
    # Test exact match
    result = canonicalizer.canonicalize_skill("React")
    assert result[0] == "react"
    assert result[1] == "React"
    assert result[2] >= 0.9
    
    # Test alias
    result = canonicalizer.canonicalize_skill("reactjs")
    assert result[0] == "react"
    
    # Test fuzzy match
    result = canonicalizer.canonicalize_skill("React.js")
    assert result[0] == "react"

def test_geocoding():
    from app.search.geocoding_service import get_geocoding_service
    
    service = get_geocoding_service()
    
    # Test geocoding
    location = service.geocode_location("San Francisco, CA, USA")
    assert location is not None
    assert location.latitude != 0.0
    assert location.longitude != 0.0
    assert location.geohash is not None
```

---

## Performance Optimization

1. **Cache embeddings**: Pre-compute and cache candidate embeddings
2. **Batch geocoding**: Geocode locations in batches to avoid rate limits
3. **Lazy loading**: Load models only when needed
4. **Async processing**: Use async for non-blocking operations

---

## Next Steps

1. ✅ Implement skill canonicalization
2. ✅ Implement geocoding
3. ✅ Integrate hybrid embeddings
4. ⏳ Collect training data
5. ⏳ Train XGBoost model
6. ⏳ A/B test new ranking
7. ⏳ Monitor metrics and iterate

---

## Support

For questions or issues:
- Check the ticket list: `ACCURACY_IMPROVEMENT_TICKETS.md`
- Review code comments in each module
- Test with small dataset first
- Monitor metrics and adjust

