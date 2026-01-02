# Implementation Guide: Scalable Search System
## Step-by-Step Implementation with Code Examples

---

## Phase 1: FAISS HNSW Index Implementation

### 1.1 Upgrade to HNSW Index

**File: `backend/app/search/ultra_fast_parallel_search.py`**

```python
import faiss
import numpy as np
from typing import List, Dict, Any

class ScalableFAISSEngine:
    """FAISS engine with HNSW for 1M+ candidates"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.candidate_ids = []
        self.candidate_data = {}
        
    def build_hnsw_index(self, embeddings: np.ndarray):
        """Build HNSW index for fast approximate search"""
        # HNSW parameters optimized for 1M+ vectors
        M = 32  # Number of connections per node (higher = more accurate, slower)
        ef_construction = 200  # Build-time search width
        ef_search = 64  # Query-time search width
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(self.dimension, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"Built HNSW index with {self.index.ntotal} vectors")
        
    def save_index(self, filepath: str):
        """Save index to disk with memory mapping"""
        faiss.write_index(self.index, filepath)
        
    def load_index(self, filepath: str):
        """Load index from disk with memory mapping"""
        self.index = faiss.read_index(filepath, faiss.IO_FLAG_MMAP)
        
    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Dict]:
        """Search with sub-10ms latency"""
        if self.index is None:
            raise ValueError("Index not initialized")
            
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.candidate_ids):
                candidate_id = self.candidate_ids[idx]
                results.append({
                    'candidate_id': candidate_id,
                    'score': float(score),
                    'candidate_data': self.candidate_data.get(candidate_id, {})
                })
        
        return results
```

### 1.2 Sharded Index Implementation

```python
class ShardedFAISSEngine:
    """Sharded FAISS for distributed search"""
    
    def __init__(self, shard_config: Dict[str, List[str]]):
        self.shards = {}
        self.shard_config = shard_config
        
    def add_shard(self, shard_name: str, dimension: int = 384):
        """Create a new shard"""
        self.shards[shard_name] = {
            'index': faiss.IndexHNSWFlat(dimension, 32),
            'candidate_ids': [],
            'candidate_data': {}
        }
        
    def add_to_shard(self, shard_name: str, candidate_id: str, 
                     embedding: np.ndarray, data: Dict):
        """Add candidate to specific shard"""
        shard = self.shards[shard_name]
        idx = len(shard['candidate_ids'])
        shard['candidate_ids'].append(candidate_id)
        shard['candidate_data'][candidate_id] = data
        
        # Normalize and add embedding
        embedding = embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(embedding)
        shard['index'].add(embedding)
        
    def search_all_shards(self, query_embedding: np.ndarray, 
                          top_k: int = 20) -> List[Dict]:
        """Search across all shards in parallel"""
        from concurrent.futures import ThreadPoolExecutor
        
        def search_shard(shard_name: str):
            shard = self.shards[shard_name]
            query = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query)
            
            scores, indices = shard['index'].search(query, top_k)
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < len(shard['candidate_ids']):
                    candidate_id = shard['candidate_ids'][idx]
                    results.append({
                        'candidate_id': candidate_id,
                        'score': float(score),
                        'shard': shard_name,
                        'candidate_data': shard['candidate_data'].get(candidate_id, {})
                    })
            return results
        
        # Search all shards in parallel
        with ThreadPoolExecutor(max_workers=len(self.shards)) as executor:
            all_results = executor.map(search_shard, self.shards.keys())
        
        # Merge and sort results
        merged_results = []
        for results in all_results:
            merged_results.extend(results)
        
        # Sort by score and return top_k
        merged_results.sort(key=lambda x: x['score'], reverse=True)
        return merged_results[:top_k]
```

---

## Phase 2: Exact Matching with Elasticsearch

### 2.1 Elasticsearch Setup

**File: `backend/app/search/elasticsearch_service.py`**

```python
from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional

class ElasticsearchService:
    """Elasticsearch service for exact field matching"""
    
    def __init__(self, hosts: List[str] = ['localhost:9200']):
        self.client = Elasticsearch(hosts)
        self.index_name = 'candidates'
        self._create_index_if_not_exists()
        
    def _create_index_if_not_exists(self):
        """Create index with optimized mapping"""
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "name": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "email": {"type": "keyword"},
                        "phone": {"type": "keyword"},
                        "location": {
                            "type": "keyword",
                            "fields": {
                                "text": {"type": "text"},
                                "geo": {"type": "geo_point"}
                            }
                        },
                        "skills": {
                            "type": "keyword",
                            "fields": {"text": {"type": "text"}}
                        },
                        "experience_years": {"type": "integer"},
                        "education_level": {"type": "keyword"},
                        "certifications": {"type": "keyword"},
                        "availability": {"type": "keyword"},
                        "salary_range": {
                            "type": "integer_range",
                            "properties": {
                                "gte": {"type": "integer"},
                                "lte": {"type": "integer"}
                            }
                        },
                        "domain": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"}
                    }
                },
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 1,
                    "index": {
                        "max_result_window": 50000
                    }
                }
            }
            
            self.client.indices.create(index=self.index_name, body=mapping)
            
    def index_candidate(self, candidate: Dict[str, Any]):
        """Index a candidate document"""
        doc = {
            "id": candidate.get('id'),
            "name": candidate.get('name'),
            "email": candidate.get('email'),
            "phone": candidate.get('phone'),
            "location": candidate.get('location'),
            "skills": candidate.get('skills', []),
            "experience_years": candidate.get('experience_years', 0),
            "education_level": candidate.get('education_level'),
            "certifications": candidate.get('certifications', []),
            "availability": candidate.get('availability'),
            "salary_range": candidate.get('salary_range'),
            "domain": candidate.get('domain', 'general')
        }
        
        self.client.index(
            index=self.index_name,
            id=candidate.get('id'),
            body=doc
        )
        
    def bulk_index_candidates(self, candidates: List[Dict[str, Any]]):
        """Bulk index candidates for performance"""
        from elasticsearch.helpers import bulk
        
        actions = []
        for candidate in candidates:
            doc = {
                "_index": self.index_name,
                "_id": candidate.get('id'),
                "_source": {
                    "id": candidate.get('id'),
                    "name": candidate.get('name'),
                    "email": candidate.get('email'),
                    "phone": candidate.get('phone'),
                    "location": candidate.get('location'),
                    "skills": candidate.get('skills', []),
                    "experience_years": candidate.get('experience_years', 0),
                    "education_level": candidate.get('education_level'),
                    "certifications": candidate.get('certifications', []),
                    "availability": candidate.get('availability'),
                    "salary_range": candidate.get('salary_range'),
                    "domain": candidate.get('domain', 'general')
                }
            }
            actions.append(doc)
        
        success, failed = bulk(self.client, actions, chunk_size=1000)
        return success, failed
        
    def exact_search(self, filters: Dict[str, Any], 
                    size: int = 10000) -> List[str]:
        """Perform exact field matching"""
        query = {"bool": {"must": []}}
        
        # Skills filter
        if filters.get('skills'):
            query["bool"]["must"].append({
                "terms": {"skills": filters['skills']}
            })
        
        # Location filter
        if filters.get('location'):
            query["bool"]["must"].append({
                "term": {"location.keyword": filters['location']}
            })
        
        # Experience range
        if filters.get('experience_range'):
            min_exp, max_exp = filters['experience_range']
            query["bool"]["must"].append({
                "range": {
                    "experience_years": {
                        "gte": min_exp,
                        "lte": max_exp
                    }
                }
            })
        
        # Education level
        if filters.get('education_level'):
            query["bool"]["must"].append({
                "terms": {"education_level": filters['education_level']}
            })
        
        # Certifications
        if filters.get('certifications'):
            query["bool"]["must"].append({
                "terms": {"certifications": filters['certifications']}
            })
        
        # Domain
        if filters.get('domain'):
            query["bool"]["must"].append({
                "term": {"domain": filters['domain']}
            })
        
        # Execute search
        response = self.client.search(
            index=self.index_name,
            body={"query": query, "size": size},
            scroll='2m'
        )
        
        # Extract candidate IDs
        candidate_ids = [hit['_source']['id'] for hit in response['hits']['hits']]
        
        # Handle scroll for large result sets
        scroll_id = response.get('_scroll_id')
        while scroll_id and len(candidate_ids) < size:
            scroll_response = self.client.scroll(
                scroll_id=scroll_id,
                scroll='2m'
            )
            hits = scroll_response['hits']['hits']
            if not hits:
                break
            candidate_ids.extend([hit['_source']['id'] for hit in hits])
            scroll_id = scroll_response.get('_scroll_id')
        
        return candidate_ids[:size]
```

---

## Phase 3: Hybrid Search Implementation

### 3.1 Hybrid Search Engine

**File: `backend/app/search/hybrid_search_engine.py`**

```python
import numpy as np
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import time

class HybridSearchEngine:
    """Combines exact matching (Elasticsearch) with semantic search (FAISS)"""
    
    def __init__(self, faiss_engine, elasticsearch_service, embedding_service):
        self.faiss_engine = faiss_engine
        self.elasticsearch_service = elasticsearch_service
        self.embedding_service = embedding_service
        
    def search(self, query: str, filters: Dict[str, Any] = None, 
               top_k: int = 20) -> List[Dict]:
        """Perform hybrid search"""
        start_time = time.time()
        
        # Step 1: Parse query and extract filters
        query_parser = QueryParser()
        parsed = query_parser.parse(query)
        
        # Merge with explicit filters
        if filters:
            parsed.update(filters)
        
        # Step 2: Exact filtering (Elasticsearch) - parallel
        exact_filter_time = time.time()
        filtered_ids = self.elasticsearch_service.exact_search(parsed, size=10000)
        exact_filter_time = time.time() - exact_filter_time
        
        # Step 3: Semantic ranking (FAISS) - only on filtered candidates
        semantic_time = time.time()
        query_embedding = self.embedding_service.encode_single(query)
        
        # Get embeddings for filtered candidates
        candidate_embeddings = []
        valid_ids = []
        for candidate_id in filtered_ids[:1000]:  # Limit to top 1000 for performance
            embedding = self.faiss_engine.get_embedding(candidate_id)
            if embedding is not None:
                candidate_embeddings.append(embedding)
                valid_ids.append(candidate_id)
        
        if not candidate_embeddings:
            return []
        
        candidate_embeddings = np.array(candidate_embeddings)
        
        # Calculate semantic similarity
        query_emb = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)
        faiss.normalize_L2(candidate_embeddings)
        
        semantic_scores = np.dot(candidate_embeddings, query_emb.T).flatten()
        semantic_time = time.time() - semantic_time
        
        # Step 4: Calculate exact match scores
        exact_scores = self._calculate_exact_scores(parsed, valid_ids)
        
        # Step 5: Hybrid scoring
        results = []
        for i, candidate_id in enumerate(valid_ids):
            hybrid_score = self._calculate_hybrid_score(
                semantic_scores[i],
                exact_scores.get(candidate_id, 0.0),
                parsed
            )
            
            results.append({
                'candidate_id': candidate_id,
                'score': hybrid_score,
                'semantic_score': float(semantic_scores[i]),
                'exact_score': exact_scores.get(candidate_id, 0.0)
            })
        
        # Sort by hybrid score
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_k]
        
        # Add candidate data
        for result in results:
            result['candidate_data'] = self.faiss_engine.get_candidate_data(
                result['candidate_id']
            )
        
        total_time = time.time() - start_time
        print(f"Hybrid search completed in {total_time:.2f}s "
              f"(exact: {exact_filter_time:.2f}s, semantic: {semantic_time:.2f}s)")
        
        return results
    
    def _calculate_exact_scores(self, filters: Dict, candidate_ids: List[str]) -> Dict[str, float]:
        """Calculate exact match scores"""
        scores = {}
        
        for candidate_id in candidate_ids:
            candidate = self.faiss_engine.get_candidate_data(candidate_id)
            if not candidate:
                continue
            
            score = 0.0
            matches = 0
            total = 0
            
            # Skills match
            if filters.get('skills'):
                total += len(filters['skills'])
                candidate_skills = set(candidate.get('skills', []))
                query_skills = set(filters['skills'])
                matches += len(candidate_skills & query_skills)
            
            # Location match
            if filters.get('location'):
                total += 1
                if candidate.get('location', '').lower() == filters['location'].lower():
                    matches += 1
            
            # Experience match
            if filters.get('experience_range'):
                total += 1
                exp_years = candidate.get('experience_years', 0)
                min_exp, max_exp = filters['experience_range']
                if min_exp <= exp_years <= max_exp:
                    matches += 1
            
            # Education match
            if filters.get('education_level'):
                total += 1
                if candidate.get('education_level') in filters['education_level']:
                    matches += 1
            
            if total > 0:
                score = matches / total
            
            scores[candidate_id] = score
        
        return scores
    
    def _calculate_hybrid_score(self, semantic_score: float, exact_score: float,
                               filters: Dict) -> float:
        """Calculate final hybrid score"""
        # Weighted combination
        # Adjust weights based on whether exact filters are present
        has_exact_filters = any([
            filters.get('skills'),
            filters.get('location'),
            filters.get('experience_range'),
            filters.get('education_level')
        ])
        
        if has_exact_filters:
            # More weight on exact matching when filters are present
            return 0.4 * semantic_score + 0.6 * exact_score
        else:
            # More weight on semantic when no filters
            return 0.7 * semantic_score + 0.3 * exact_score
```

---

## Phase 4: Self-Learning System

### 4.1 Feedback Collection

**File: `backend/app/search/feedback_system.py`**

```python
from datetime import datetime
from typing import List, Dict, Any
import json

class FeedbackCollector:
    """Collect and store user feedback"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        
    def record_feedback(self, search_id: str, candidate_id: str,
                       action: str, relevance_score: float = None,
                       query: str = None, filters: Dict = None):
        """Record user feedback"""
        feedback = {
            'search_id': search_id,
            'candidate_id': candidate_id,
            'action': action,  # 'viewed', 'saved', 'contacted', 'hired', 'rejected'
            'relevance_score': relevance_score or self._infer_score(action),
            'query': query,
            'filters': json.dumps(filters) if filters else None,
            'timestamp': datetime.utcnow()
        }
        
        # Store in database
        self.db.insert('feedback', feedback)
        
    def _infer_score(self, action: str) -> float:
        """Infer relevance score from action"""
        scores = {
            'viewed': 0.3,
            'saved': 0.7,
            'contacted': 0.8,
            'hired': 1.0,
            'rejected': 0.1
        }
        return scores.get(action, 0.5)
    
    def get_feedback_batch(self, limit: int = 1000) -> List[Dict]:
        """Get batch of feedback for training"""
        return self.db.query(
            "SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
```

### 4.2 Learning Model

**File: `backend/app/search/learning_model.py`**

```python
import xgboost as xgb
import numpy as np
from typing import List, Dict, Any
import pickle

class RelevanceLearner:
    """Learn candidate relevance from feedback"""
    
    def __init__(self):
        self.model = None
        self.feature_extractor = FeatureExtractor()
        
    def train(self, feedback_data: List[Dict], embeddings_service):
        """Train relevance model"""
        X = []
        y = []
        
        for feedback in feedback_data:
            # Extract features
            features = self.feature_extractor.extract(
                feedback, embeddings_service
            )
            X.append(features)
            y.append(feedback['relevance_score'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror'
        )
        
        self.model.fit(X, y)
        
    def predict(self, query: str, candidate: Dict, 
                embeddings_service) -> float:
        """Predict relevance score"""
        if self.model is None:
            return 0.5  # Default score
        
        features = self.feature_extractor.extract_for_prediction(
            query, candidate, embeddings_service
        )
        
        score = self.model.predict([features])[0]
        return float(score)
    
    def save(self, filepath: str):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class FeatureExtractor:
    """Extract features for learning"""
    
    def extract(self, feedback: Dict, embeddings_service) -> List[float]:
        """Extract features from feedback"""
        query = feedback.get('query', '')
        candidate = feedback.get('candidate', {})
        
        return self.extract_for_prediction(query, candidate, embeddings_service)
    
    def extract_for_prediction(self, query: str, candidate: Dict,
                              embeddings_service) -> List[float]:
        """Extract features for prediction"""
        features = []
        
        # Semantic similarity
        query_emb = embeddings_service.encode_single(query)
        candidate_text = self._candidate_to_text(candidate)
        candidate_emb = embeddings_service.encode_single(candidate_text)
        semantic_sim = np.dot(query_emb, candidate_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(candidate_emb)
        )
        features.append(semantic_sim)
        
        # Skill match ratio
        query_skills = self._extract_skills(query)
        candidate_skills = candidate.get('skills', [])
        skill_match = len(set(query_skills) & set(candidate_skills))
        skill_total = len(query_skills) if query_skills else 1
        features.append(skill_match / skill_total)
        
        # Experience match
        exp_years = candidate.get('experience_years', 0)
        features.append(min(exp_years / 20.0, 1.0))  # Normalize to 0-1
        
        # Education level (encoded)
        edu_level = candidate.get('education_level', '')
        edu_encoded = self._encode_education(edu_level)
        features.extend(edu_encoded)
        
        # Domain match
        domain = candidate.get('domain', 'general')
        features.append(1.0 if domain != 'general' else 0.0)
        
        return features
    
    def _candidate_to_text(self, candidate: Dict) -> str:
        """Convert candidate to searchable text"""
        parts = [
            candidate.get('name', ''),
            candidate.get('title', ''),
            ' '.join(candidate.get('skills', [])),
            candidate.get('experience', ''),
            candidate.get('education', '')
        ]
        return ' '.join(parts)
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text (simplified)"""
        # In production, use NLP to extract skills
        common_skills = ['python', 'java', 'javascript', 'react', 'node']
        text_lower = text.lower()
        return [skill for skill in common_skills if skill in text_lower]
    
    def _encode_education(self, level: str) -> List[float]:
        """One-hot encode education level"""
        levels = ['high_school', 'bachelor', 'master', 'phd']
        encoded = [0.0] * len(levels)
        if level:
            level_lower = level.lower()
            for i, lvl in enumerate(levels):
                if lvl in level_lower:
                    encoded[i] = 1.0
                    break
        return encoded
```

---

## Phase 5: Query Parser for Exact Matching

### 5.1 Advanced Query Parser

**File: `backend/app/search/query_parser.py`**

```python
import re
from typing import Dict, List, Any, Tuple

class QueryParser:
    """Parse natural language queries into structured filters"""
    
    def __init__(self):
        self.skill_keywords = self._load_skill_keywords()
        self.location_patterns = self._load_location_patterns()
        
    def parse(self, query: str) -> Dict[str, Any]:
        """Parse query into structured format"""
        query_lower = query.lower()
        
        parsed = {
            'text_query': query,
            'skills': self._extract_skills(query_lower),
            'location': self._extract_location(query_lower),
            'experience_range': self._extract_experience(query_lower),
            'education': self._extract_education(query_lower),
            'certifications': self._extract_certifications(query_lower),
            'salary_range': self._extract_salary(query_lower),
            'availability': self._extract_availability(query_lower)
        }
        
        return parsed
    
    def _extract_skills(self, query: str) -> List[str]:
        """Extract skills from query"""
        skills = []
        
        # Common tech skills
        tech_skills = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'node.js', 'django', 'flask', 'spring', 'express',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'sql', 'mongodb', 'postgresql', 'redis'
        ]
        
        for skill in tech_skills:
            if skill in query:
                skills.append(skill)
        
        # Healthcare skills
        healthcare_skills = [
            'nursing', 'rn', 'lpn', 'cna', 'bcls', 'acls',
            'patient care', 'med-surg', 'icu', 'er'
        ]
        
        for skill in healthcare_skills:
            if skill in query:
                skills.append(skill)
        
        return skills
    
    def _extract_location(self, query: str) -> str:
        """Extract location from query"""
        # Common locations
        locations = [
            'san francisco', 'new york', 'los angeles', 'chicago',
            'boston', 'seattle', 'austin', 'denver',
            'mumbai', 'bangalore', 'delhi', 'hyderabad',
            'london', 'berlin', 'paris', 'amsterdam'
        ]
        
        for location in locations:
            if location in query:
                return location.title()
        
        return None
    
    def _extract_experience(self, query: str) -> Tuple[int, int]:
        """Extract experience range"""
        # Patterns: "5+ years", "3-5 years", "senior (8-10 years)"
        patterns = [
            r'(\d+)\+?\s*years?',
            r'(\d+)\s*-\s*(\d+)\s*years?',
            r'senior.*?(\d+)\s*-\s*(\d+)',
            r'mid.*?(\d+)\s*-\s*(\d+)',
            r'junior.*?(\d+)\s*-\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    min_exp = int(groups[0])
                    return (min_exp, min_exp + 5)
                elif len(groups) == 2:
                    return (int(groups[0]), int(groups[1]))
        
        return (0, 20)  # Default range
    
    def _extract_education(self, query: str) -> List[str]:
        """Extract education requirements"""
        education = []
        
        if 'bachelor' in query or "bachelor's" in query:
            education.append("Bachelor's")
        if 'master' in query or "master's" in query:
            education.append("Master's")
        if 'phd' in query or 'ph.d' in query:
            education.append("PhD")
        
        return education
    
    def _extract_certifications(self, query: str) -> List[str]:
        """Extract certification requirements"""
        certs = []
        
        # Tech certifications
        tech_certs = ['aws certified', 'azure certified', 'google cloud',
                     'pmp', 'scrum master', 'cisco']
        
        # Healthcare certifications
        healthcare_certs = ['rn license', 'bcls', 'acls', 'cpr']
        
        all_certs = tech_certs + healthcare_certs
        
        for cert in all_certs:
            if cert in query:
                certs.append(cert.title())
        
        return certs
    
    def _extract_salary(self, query: str) -> Tuple[int, int]:
        """Extract salary range"""
        # Patterns: "$100k-$150k", "100k-150k", "$100,000"
        patterns = [
            r'\$?(\d+)k?\s*-\s*\$?(\d+)k?',
            r'\$?(\d+),?(\d+)?k?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                groups = match.groups()
                if len(groups) >= 2 and groups[1]:
                    min_sal = int(groups[0]) * 1000
                    max_sal = int(groups[1]) * 1000
                    return (min_sal, max_sal)
                elif len(groups) >= 1:
                    salary = int(groups[0]) * 1000
                    return (salary, salary + 50000)
        
        return (0, 200000)  # Default range
    
    def _extract_availability(self, query: str) -> str:
        """Extract availability"""
        if 'immediate' in query or 'available now' in query:
            return 'immediate'
        if '2 weeks' in query or 'two weeks' in query:
            return '2 weeks'
        if '1 month' in query or 'one month' in query:
            return '1 month'
        
        return None
    
    def _load_skill_keywords(self) -> Dict[str, List[str]]:
        """Load skill keywords database"""
        # In production, load from database or file
        return {}
    
    def _load_location_patterns(self) -> List[str]:
        """Load location patterns"""
        # In production, load from database or file
        return []
```

---

## Integration Example

### Complete Search Service

**File: `backend/app/search/scalable_search_service.py`**

```python
from typing import List, Dict, Any
import time

class ScalableSearchService:
    """Complete scalable search service"""
    
    def __init__(self):
        # Initialize components
        self.faiss_engine = ScalableFAISSEngine()
        self.elasticsearch_service = ElasticsearchService()
        self.embedding_service = EmbeddingService()
        self.hybrid_engine = HybridSearchEngine(
            self.faiss_engine,
            self.elasticsearch_service,
            self.embedding_service
        )
        self.query_parser = QueryParser()
        self.feedback_collector = FeedbackCollector(db)
        self.learner = RelevanceLearner()
        
    def search(self, query: str, filters: Dict = None, 
               top_k: int = 20) -> Dict[str, Any]:
        """Main search method"""
        start_time = time.time()
        
        # Perform hybrid search
        results = self.hybrid_engine.search(query, filters, top_k)
        
        # Apply learned relevance scores
        for result in results:
            learned_score = self.learner.predict(
                query,
                result['candidate_data'],
                self.embedding_service
            )
            # Combine with hybrid score
            result['final_score'] = (
                0.7 * result['score'] + 0.3 * learned_score
            )
        
        # Re-sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        search_time = time.time() - start_time
        
        return {
            'results': results[:top_k],
            'total_time': search_time,
            'count': len(results)
        }
    
    def record_feedback(self, search_id: str, candidate_id: str,
                       action: str, query: str = None):
        """Record user feedback for learning"""
        self.feedback_collector.record_feedback(
            search_id, candidate_id, action, query=query
        )
        
        # Trigger incremental learning if enough feedback
        if self.feedback_collector.get_feedback_count() % 100 == 0:
            self._incremental_learn()
    
    def _incremental_learn(self):
        """Incremental model update"""
        feedback_batch = self.feedback_collector.get_feedback_batch(1000)
        if len(feedback_batch) >= 100:
            self.learner.train(feedback_batch, self.embedding_service)
```

---

## Deployment Checklist

- [ ] Set up FAISS HNSW index
- [ ] Configure Elasticsearch cluster
- [ ] Set up Redis cluster
- [ ] Implement sharding strategy
- [ ] Deploy feedback collection
- [ ] Set up learning pipeline
- [ ] Configure monitoring
- [ ] Load test with 1M candidates
- [ ] Optimize based on metrics

---

This implementation guide provides the foundation for building a scalable, self-learning search system. Each component can be developed and tested independently before integration.

