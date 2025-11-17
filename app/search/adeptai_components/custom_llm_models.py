# custom_llm_models.py - Custom LLM Components
"""
Lightweight custom LLM models for search enhancement without external APIs
"""

import os
import json
import re
import time
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
import hashlib
from dataclasses import dataclass

# Optional FAISS for vector indexing
try:
    import faiss  # type: ignore
    _faiss_available = True
except Exception:
    _faiss_available = False

# Optional advanced NLP/transformers
try:
    import torch  # type: ignore
    _torch_available = True
except Exception:
    _torch_available = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    from sentence_transformers import InputExample, losses  # type: ignore
    _st_available = True
except Exception:
    _st_available = False

try:
    import spacy  # type: ignore
    _spacy_available = True
except Exception:
    _spacy_available = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline  # type: ignore
    _hf_available = True
except Exception:
    _hf_available = False

# Custom tokenizer for lightweight processing
class CustomTokenizer:
    """Lightweight tokenizer optimized for resume/candidate text"""
    
    def __init__(self):
        # Common technical terms that should be preserved
        self.technical_terms = {
            'ai', 'ml', 'js', 'ui', 'ux', 'it', 'qa', 'hr', 'rn', 'db', 'os', 'api', 'sql', 
            'aws', 'gcp', 'ios', 'android', 'vr', 'ar', 'iot', 'crm', 'erp', 'sap', 'oracle',
            'mysql', 'postgres', 'redis', 'mongo', 'docker', 'kubernetes', 'k8s', 'ci', 'cd',
            'devops', 'agile', 'scrum', 'kanban', 'jira', 'confluence', 'slack', 'zoom', 'teams',
            'git', 'github', 'gitlab', 'bitbucket', 'jenkins', 'travis', 'circleci', 'azure',
            'heroku', 'netlify', 'vercel', 'firebase', 'supabase', 'stripe', 'paypal', 'twilio',
            'sendgrid', 'mailchimp', 'hubspot', 'salesforce', 'zendesk', 'intercom', 'mixpanel',
            'amplitude', 'segment', 'datadog', 'newrelic', 'sentry', 'rollbar', 'bugsnag',
            'honeybadger', 'airbrake', 'raygun', 'logrocket', 'fullstory', 'hotjar', 'crazyegg',
            'optimizely', 'vwo', 'ab', 'testing', 'a/b', 'test', 'qa', 'qc', 'uat', 'staging',
            'prod', 'production', 'dev', 'development', 'preprod', 'sandbox', 'localhost'
        }
        
        # Stop words for filtering
        self.stop_words = {
            'the', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'to', 'of', 'a', 'an', 'is',
            'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        # Compile regex patterns for faster processing
        self.word_pattern = re.compile(r'\b\w+\b')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text while preserving technical terms"""
        if not text:
            return []
        
        # Extract words using regex
        words = self.word_pattern.findall(text.lower())
        
        # Filter words but preserve technical terms
        filtered_words = []
        for word in words:
            if len(word) > 2 and (word not in self.stop_words or word in self.technical_terms):
                filtered_words.append(word)
        
        return filtered_words

    def generate_ngrams(self, tokens: List[str], min_n: int = 2, max_n: int = 3) -> List[str]:
        """Generate n-grams from tokens"""
        if not tokens:
            return []
        ngrams: List[str] = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i+n]))
        return ngrams
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            'emails': self.email_pattern.findall(text),
            'phones': self.phone_pattern.findall(text),
            'skills': self._extract_skills(text),
            'years': self._extract_years(text)
        }
        return entities
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using pattern matching"""
        skills = []
        text_lower = text.lower()
        
        # Common skill patterns
        skill_patterns = [
            r'\b(?:python|java|javascript|react|node|vue|angular)\b',
            r'\b(?:aws|azure|gcp|docker|kubernetes)\b',
            r'\b(?:sql|mysql|postgres|mongodb|redis)\b',
            r'\b(?:machine learning|ai|ml|data science)\b',
            r'\b(?:agile|scrum|kanban|devops)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower)
            skills.extend(matches)
        
        return list(set(skills))
    
    def _extract_years(self, text: str) -> List[str]:
        """Extract years of experience from text"""
        year_pattern = r'\b(?:[0-9]+(?:\+)?\s*(?:years?|yrs?|y))\b'
        return re.findall(year_pattern, text.lower())

# Custom embedding model using TF-IDF and semantic features
class CustomEmbeddingModel:
    """Lightweight embedding model using TF-IDF, semantic features, and n-grams"""
    
    def __init__(self, max_features: int = 10000, use_ngrams: bool = True, ngram_range: Tuple[int, int] = (1, 3)):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_scores = {}
        self.skill_weights = {}
        self.experience_weights = {}
        self.domain_weights = {}
        self._is_fitted = False
        self.use_ngrams = use_ngrams
        self.ngram_range = ngram_range
        # Vector index fields
        self._index = None
        self._embeddings_matrix: Optional[np.ndarray] = None
        
        # Initialize domain-specific weights
        self._init_domain_weights()
        
    def _init_domain_weights(self):
        """Initialize domain-specific feature weights"""
        self.domain_weights = {
            'technology': {
                'python': 0.9, 'java': 0.9, 'javascript': 0.9, 'react': 0.8, 'node': 0.8,
                'aws': 0.8, 'docker': 0.7, 'kubernetes': 0.7, 'sql': 0.6, 'machine learning': 0.9,
                'ai': 0.9, 'ml': 0.9, 'data science': 0.8, 'backend': 0.7, 'frontend': 0.7,
                'devops': 0.7, 'cloud': 0.6, 'agile': 0.5, 'scrum': 0.5
            },
            'healthcare': {
                'nurse': 0.9, 'doctor': 0.9, 'physician': 0.9, 'medical': 0.8, 'healthcare': 0.8,
                'hospital': 0.7, 'clinic': 0.6, 'patient': 0.7, 'rn': 0.8, 'md': 0.8,
                'icu': 0.7, 'emergency': 0.6, 'surgery': 0.6, 'pediatric': 0.6, 'cardiology': 0.6
            },
            'finance': {
                'finance': 0.9, 'banking': 0.8, 'investment': 0.8, 'accounting': 0.8,
                'financial': 0.7, 'analyst': 0.7, 'advisor': 0.6, 'trading': 0.6,
                'portfolio': 0.6, 'risk': 0.6, 'compliance': 0.6, 'fintech': 0.7
            }
        }
        
        # Experience level weights
        self.experience_weights = {
            'senior': 0.8, 'lead': 0.8, 'principal': 0.9, 'architect': 0.9,
            'manager': 0.7, 'director': 0.8, 'junior': 0.4, 'entry': 0.3,
            'graduate': 0.3, 'intern': 0.2, 'associate': 0.5
        }
    
    def fit(self, documents: List[str]):
        """Fit the embedding model to documents"""
        print("🔧 Training custom embedding model...")
        start_time = time.time()
        
        tokenizer = CustomTokenizer()
        
        # Build vocabulary and compute TF-IDF
        doc_tokens = []
        word_counts = Counter()
        
        for doc in documents:
            tokens = tokenizer.tokenize(doc)
            # Add n-grams
            if self.use_ngrams and self.ngram_range[1] > 1:
                ngrams = []
                min_n, max_n = self.ngram_range
                if min_n < 1:
                    min_n = 1
                if min_n <= 1:
                    # include unigrams already in tokens
                    pass
                ngrams.extend(tokenizer.generate_ngrams(tokens, max(2, min_n), max_n))
                tokens = tokens + ngrams
            doc_tokens.append(tokens)
            word_counts.update(tokens)
        
        # Build vocabulary with frequency threshold
        min_freq = max(1, len(documents) // 1000)  # Dynamic threshold
        vocab_items = [(word, count) for word, count in word_counts.items() 
                      if count >= min_freq and len(word) > 2]
        
        # Sort by frequency and take top features
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(vocab_items[:self.max_features])}
        
        # Compute IDF scores
        doc_count = len(documents)
        for word in self.vocabulary:
            docs_with_word = sum(1 for tokens in doc_tokens if word in tokens)
            self.idf_scores[word] = np.log(doc_count / (docs_with_word + 1))
        
        # Compute skill weights based on domain importance
        for word in self.vocabulary:
            max_weight = 0.0
            for domain, weights in self.domain_weights.items():
                if word in weights:
                    max_weight = max(max_weight, weights[word])
            self.skill_weights[word] = max_weight if max_weight > 0 else 0.1
        
        self._is_fitted = True
        fit_time = time.time() - start_time
        print(f"✅ Custom embedding model trained in {fit_time:.2f}s with {len(self.vocabulary)} features")
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text into embedding vector"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before encoding")
        
        tokenizer = CustomTokenizer()
        tokens = tokenizer.tokenize(text)
        if self.use_ngrams and self.ngram_range[1] > 1:
            ngrams = tokenizer.generate_ngrams(tokens, max(2, self.ngram_range[0]), self.ngram_range[1])
            tokens = tokens + ngrams
        
        # Initialize embedding vector
        embedding = np.zeros(len(self.vocabulary))
        
        # Compute TF-IDF with domain weights
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        for token, count in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf = count / total_tokens
                idf = self.idf_scores[token]
                skill_weight = self.skill_weights[token]
                
                # Combined score: TF-IDF * skill weight
                embedding[idx] = tf * idf * (1 + skill_weight)
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into an embedding matrix (rows normalized)"""
        if not texts:
            return np.zeros((0, len(self.vocabulary)))
        matrix = np.vstack([self.encode(t) for t in texts])
        # Normalize rows for cosine via inner product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def build_index(self, embeddings: np.ndarray):
        """Build a vector index from normalized embeddings (rows)"""
        if embeddings is None or len(embeddings) == 0:
            self._index = None
            self._embeddings_matrix = None
            return
        # Ensure embeddings are float32 and normalized
        emb = embeddings.astype('float32')
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        self._embeddings_matrix = emb
        if _faiss_available:
            dim = emb.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(emb)
            self._index = index
            print(f"✅ FAISS index built: {emb.shape[0]} vectors, dim={dim}")
        else:
            self._index = None
            print(f"⚠️ FAISS not available; using NumPy fallback for semantic search ({emb.shape[0]} vectors)")

    def search_index(self, query_vector: np.ndarray, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index with a query vector and return (indices, scores)"""
        if self._embeddings_matrix is None:
            return np.array([], dtype=int), np.array([])
        # Normalize query vector
        q = query_vector.astype('float32')
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return np.array([], dtype=int), np.array([])
        q = q / q_norm
        if self._index is not None and _faiss_available:
            D, I = self._index.search(q.reshape(1, -1), top_k)
            return I[0], D[0]
        # NumPy fallback: cosine via dot product (embeddings are normalized)
        scores = np.dot(self._embeddings_matrix, q)
        top_k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

    def is_index_ready(self) -> bool:
        return self._embeddings_matrix is not None

# ===================== Advanced Neural Components =====================

class AdvancedEmbeddingModel:
    """Neural embedding model with optional fine-tuning (SentenceTransformers)."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if not (_st_available and _torch_available):
            raise ImportError('SentenceTransformers or Torch not available')
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self._index = None
        self._embeddings_matrix: Optional[np.ndarray] = None

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device if hasattr(self, 'device') else None,
        )

    def fine_tune(self, train_pairs: List[Tuple[str, str, float]]):
        from torch.utils.data import DataLoader  # type: ignore
        train_examples = [InputExample(texts=[a, b], label=score) for a, b, score in train_pairs]
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.model)
        self.model.fit(train_objectives=[(train_loader, train_loss)], epochs=1, warmup_steps=50)

    # Vector index utilities (same API as CustomEmbeddingModel)
    def build_index(self, embeddings: np.ndarray):
        if embeddings is None or len(embeddings) == 0:
            self._index = None
            self._embeddings_matrix = None
            return
        emb = embeddings.astype('float32')
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        self._embeddings_matrix = emb
        if _faiss_available:
            dim = emb.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(emb)
            self._index = index
        else:
            self._index = None

    def search_index(self, query_vector: np.ndarray, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        if self._embeddings_matrix is None:
            return np.array([], dtype=int), np.array([])
        q = query_vector.astype('float32')
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return np.array([], dtype=int), np.array([])
        q = q / q_norm
        if self._index is not None and _faiss_available:
            D, I = self._index.search(q.reshape(1, -1), top_k)
            return I[0], D[0]
        scores = np.dot(self._embeddings_matrix, q)
        top_k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

    def is_index_ready(self) -> bool:
        return self._embeddings_matrix is not None


class AdvancedEntityExtractor:
    """NER-based entity extraction using spaCy with optional patterns."""
    def __init__(self):
        if not _spacy_available:
            raise ImportError('spaCy not available')
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except Exception:
            # If model not present, raise cleaner error
            raise ImportError('spaCy model en_core_web_sm not installed')
        try:
            ruler = self.nlp.add_pipe('entity_ruler', before='ner')
            patterns = [
                {"label": "SKILL", "pattern": "Python"},
                {"label": "SKILL", "pattern": "Machine Learning"},
                {"label": "SKILL", "pattern": "AWS"},
            ]
            ruler.add_patterns(patterns)
        except Exception:
            pass

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        return {
            'persons': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
            'orgs': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
            'skills': [ent.text for ent in doc.ents if ent.label_ == 'SKILL'],
            'dates': [ent.text for ent in doc.ents if ent.label_ == 'DATE'],
        }


class NeuralCrossEncoder:
    """Transformer cross-encoder for pair scoring."""
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        if not (_hf_available and _torch_available):
            raise ImportError('Transformers or Torch not available')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(self, queries: List[str], candidates: List[str]) -> np.ndarray:
        pairs = [[q, c] for q, c in zip(queries, candidates)]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            scores = torch.sigmoid(outputs.logits).detach().cpu().numpy()
        return scores.flatten()


# Lightweight attention and ranking modules (optional training usage)
try:
    import torch.nn as nn  # type: ignore
    _nn_available = True
except Exception:
    _nn_available = False

if _nn_available:
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
            super().__init__()
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )

        def forward(self, query_emb, candidate_emb):
            attn_out, attn_weights = self.attention(query_emb, candidate_emb, candidate_emb)
            x = self.layer_norm(attn_out + query_emb)
            ffn_out = self.ffn(x)
            output = self.layer_norm(ffn_out + x)
            return output, attn_weights

    class LearningToRankModel(nn.Module):
        def __init__(self, input_dim: int = 768, hidden_dims: List[int] = [512, 256, 128]):
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, features):
            return self.network(features).squeeze(-1)

        def pairwise_loss(self, pos_scores, neg_scores, margin: float = 1.0):
            return torch.mean(torch.clamp(margin - (pos_scores - neg_scores), min=0))

    class MultiTaskModel(nn.Module):
        def __init__(self, input_dim: int = 768):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU()
            )
            self.relevance_head = nn.Linear(256, 1)
            self.domain_head = nn.Linear(256, 5)
            self.salary_head = nn.Linear(256, 1)

        def forward(self, features):
            shared = self.shared(features)
            return {
                'relevance': torch.sigmoid(self.relevance_head(shared)),
                'domain': torch.softmax(self.domain_head(shared), dim=-1),
                'salary': self.salary_head(shared),
            }


class QueryIntentClassifier:
    def __init__(self):
        self.intents = {
            'skill_search': ['python', 'java', 'skills', 'experience with'],
            'role_search': ['developer', 'engineer', 'manager', 'looking for'],
            'location_search': ['remote', 'location', 'based in'],
            'experience_search': ['years', 'senior', 'junior', 'experience'],
            'education_search': ['degree', 'phd', 'masters', 'university'],
        }
        try:
            from sklearn.naive_bayes import MultinomialNB  # type: ignore
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            self.vectorizer = TfidfVectorizer(max_features=1000)
            self.classifier = MultinomialNB()
            self._ml_available = True
        except Exception:
            self._ml_available = False

    def classify(self, query: str) -> Tuple[str, float]:
        scores: Dict[str, int] = {}
        for intent, keywords in self.intents.items():
            score = sum(1 for kw in keywords if kw in query.lower())
            if score > 0:
                scores[intent] = score
        if scores:
            max_intent = max(scores, key=scores.get)
            confidence = scores[max_intent] / sum(scores.values())
            return max_intent, float(confidence)
        return 'general', 0.5


class ContextualQueryExpander:
    def __init__(self):
        if not _hf_available:
            raise ImportError('Transformers pipeline not available')
        device = 0 if (_torch_available and torch.cuda.is_available()) else -1
        self.generator = pipeline('text-generation', model='gpt2', device=device)

    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        prompt = (
            f'Given the job search query: "{query}"\n'
            f'Generate {num_expansions} related search queries that capture the same intent:\n1.'
        )
        outputs = self.generator(prompt, max_length=100, num_return_sequences=1, temperature=0.7)
        text = outputs[0].get('generated_text', '')
        lines = [l.strip('- ').strip() for l in text.split('\n') if l.strip()]
        expansions: List[str] = []
        for line in lines:
            if len(expansions) >= num_expansions:
                break
            if line and line != query and line not in expansions:
                expansions.append(line)
        return [query] + expansions


class FeedbackLearner:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.feedback_buffer: List[Dict[str, Any]] = []
        self.alpha = 0.01

    def add_feedback(self, query: str, candidate_text: str, ranking_score: float, user_action: str):
        reward = {'click': 0.5, 'hire': 1.0, 'reject': -0.5}.get(user_action, 0)
        self.feedback_buffer.append({'query': query, 'candidate': candidate_text, 'score': ranking_score, 'reward': reward})

    def update_model(self):
        if len(self.feedback_buffer) < 10:
            return
        batch = self.feedback_buffer[-10:]
        # Simplified update towards positive examples
        for fb in batch:
            if fb['reward'] > 0:
                try:
                    q_emb = self.embedding_model.encode([fb['query']])[0]
                    c_emb = self.embedding_model.encode([fb['candidate']])[0]
                    direction = c_emb - q_emb
                    q_emb += self.alpha * fb['reward'] * direction
                except Exception:
                    continue


class ModelExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        try:
            import shap  # type: ignore
            self._shap_available = True
            self._shap = shap
        except Exception:
            self._shap_available = False

    def explain_prediction(self, query_features: np.ndarray, candidate_features: np.ndarray) -> str:
        combined = np.concatenate([query_features, candidate_features])
        if not self._shap_available:
            return 'Explanation unavailable (SHAP not installed)'
        if self.explainer is None:
            self.explainer = self._shap.KernelExplainer(self.model.predict, combined.reshape(1, -1))
        shap_values = self.explainer.shap_values(combined)
        try:
            important_idx = np.argsort(np.abs(shap_values))[-5:]
        except Exception:
            return 'Explanation unavailable'
        explanation = []
        for idx in important_idx:
            fname = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'
            explanation.append(f"{fname}: {float(shap_values[idx]):.3f}")
        return "\n".join(explanation)


# Availability flags
ADVANCED_EMBEDDINGS_AVAILABLE = _st_available and _torch_available
NEURAL_CROSS_ENCODER_AVAILABLE = _hf_available and _torch_available
NER_AVAILABLE = _spacy_available

# Custom cross-encoder for reranking
class CustomCrossEncoder:
    """Lightweight cross-encoder for candidate reranking"""
    
    def __init__(self):
        self.feature_weights = {
            'exact_match': 0.25,
            'semantic_similarity': 0.30,
            'skill_overlap': 0.20,
            'experience_match': 0.15,
            'domain_match': 0.10
        }
        
        # Experience level mappings
        self.experience_levels = {
            'senior': 5, 'lead': 6, 'principal': 8, 'architect': 8,
            'manager': 7, 'director': 8, 'junior': 2, 'entry': 1,
            'graduate': 1, 'intern': 0, 'associate': 3
        }
    
    def encode_pair(self, query: str, candidate: Dict[str, Any]) -> float:
        """Encode query-candidate pair and return relevance score"""
        # Extract candidate features
        skills = candidate.get('skills', [])
        resume_text = candidate.get('resume_text', '')
        experience_years = candidate.get('total_experience_years', 0)
        
        # Combine candidate text
        candidate_text = f"{' '.join(skills)} {resume_text}"
        
        # Compute feature scores
        scores = {}
        
        # 1. Exact match score
        scores['exact_match'] = self._compute_exact_match(query, candidate_text)
        
        # 2. Semantic similarity (using simple word overlap + synonyms)
        scores['semantic_similarity'] = self._compute_semantic_similarity(query, candidate_text)
        
        # 3. Skill overlap
        scores['skill_overlap'] = self._compute_skill_overlap(query, skills)
        
        # 4. Experience match
        scores['experience_match'] = self._compute_experience_match(query, experience_years)
        
        # 5. Domain match
        scores['domain_match'] = self._compute_domain_match(query, candidate_text)
        
        # Weighted combination
        final_score = sum(scores[feature] * weight 
                         for feature, weight in self.feature_weights.items())
        
        return min(final_score, 1.0)
    
    def _compute_exact_match(self, query: str, candidate_text: str) -> float:
        """Compute exact word match score"""
        query_words = set(query.lower().split())
        candidate_words = set(candidate_text.lower().split())
        
        if not query_words:
            return 0.0
        
        matches = len(query_words.intersection(candidate_words))
        return matches / len(query_words)
    
    def _compute_semantic_similarity(self, query: str, candidate_text: str) -> float:
        """Compute semantic similarity using word overlap and synonyms"""
        query_words = set(query.lower().split())
        candidate_words = set(candidate_text.lower().split())
        
        # Basic word overlap
        exact_matches = len(query_words.intersection(candidate_words))
        
        # Synonym matching (simplified)
        synonym_score = self._compute_synonym_score(query_words, candidate_words)
        
        total_score = exact_matches + synonym_score
        max_possible = len(query_words)
        
        return min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0
    
    def _compute_synonym_score(self, query_words: set, candidate_words: set) -> float:
        """Compute synonym-based similarity score"""
        # Simple synonym mappings
        synonyms = {
            'developer': ['programmer', 'coder', 'engineer'],
            'engineer': ['developer', 'programmer', 'architect'],
            'manager': ['lead', 'supervisor', 'director'],
            'senior': ['lead', 'principal', 'experienced'],
            'junior': ['entry', 'associate', 'graduate'],
            'python': ['py', 'python3'],
            'javascript': ['js', 'node'],
            'react': ['reactjs', 'react.js'],
            'aws': ['amazon web services', 'amazon'],
            'docker': ['containerization'],
            'kubernetes': ['k8s', 'container orchestration']
        }
        
        synonym_score = 0.0
        for query_word in query_words:
            if query_word in synonyms:
                for synonym in synonyms[query_word]:
                    if synonym in candidate_words:
                        synonym_score += 0.5
                        break
        
        return synonym_score
    
    def _compute_skill_overlap(self, query: str, skills: List[str]) -> float:
        """Compute skill overlap score"""
        query_lower = query.lower()
        skill_matches = 0
        
        for skill in skills:
            if skill.lower() in query_lower:
                skill_matches += 1
        
        return min(skill_matches / 5.0, 1.0)  # Normalize by max 5 skills
    
    def _compute_experience_match(self, query: str, experience_years: int) -> float:
        """Compute experience level match score"""
        query_lower = query.lower()
        
        # Check for experience level keywords
        for level, min_years in self.experience_levels.items():
            if level in query_lower:
                if level in ['senior', 'lead', 'principal', 'architect', 'manager', 'director']:
                    return 1.0 if experience_years >= min_years else 0.3
                else:  # junior, entry, etc.
                    return 1.0 if experience_years <= min_years else 0.3
        
        # Default: prefer 3+ years experience
        if experience_years >= 3:
            return 0.7
        elif experience_years >= 1:
            return 0.5
        else:
            return 0.2
    
    def _compute_domain_match(self, query: str, candidate_text: str) -> float:
        """Compute domain match score"""
        query_lower = query.lower()
        candidate_lower = candidate_text.lower()
        
        # Domain keywords
        domains = {
            'technology': ['python', 'java', 'javascript', 'react', 'aws', 'docker', 'kubernetes', 'sql', 'ai', 'ml'],
            'healthcare': ['nurse', 'doctor', 'physician', 'medical', 'healthcare', 'hospital', 'patient', 'rn', 'md'],
            'finance': ['finance', 'banking', 'investment', 'accounting', 'financial', 'analyst', 'trading'],
            'education': ['teacher', 'professor', 'education', 'teaching', 'academic', 'university'],
            'marketing': ['marketing', 'advertising', 'brand', 'digital', 'social media', 'content', 'seo']
        }
        
        query_domain_scores = {}
        candidate_domain_scores = {}
        
        # Score query domains
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            query_domain_scores[domain] = score
        
        # Score candidate domains
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in candidate_lower)
            candidate_domain_scores[domain] = score
        
        # Find best matching domain
        best_match = 0.0
        for domain in domains:
            query_score = query_domain_scores.get(domain, 0)
            candidate_score = candidate_domain_scores.get(domain, 0)
            
            if query_score > 0 and candidate_score > 0:
                match_score = min(query_score, candidate_score) / max(query_score, candidate_score)
                best_match = max(best_match, match_score)
        
        return best_match

# Custom explanation generator
class CustomExplanationGenerator:
    """Generate human-readable explanations for search results"""
    
    def __init__(self):
        self.explanation_templates = {
            'skill_match': "Strong match in {skills} skills",
            'experience_match': "Experience level aligns with requirements ({years} years)",
            'domain_match': "Relevant background in {domain} domain",
            'exact_match': "Direct match with query terms",
            'semantic_match': "Semantic similarity with query intent"
        }
        
        # Skill importance mapping
        self.skill_importance = {
            'python': 'high', 'java': 'high', 'javascript': 'high', 'react': 'high',
            'aws': 'high', 'docker': 'medium', 'kubernetes': 'medium', 'sql': 'medium',
            'machine learning': 'high', 'ai': 'high', 'ml': 'high', 'data science': 'high',
            'nurse': 'high', 'doctor': 'high', 'physician': 'high', 'medical': 'high',
            'finance': 'high', 'banking': 'high', 'investment': 'high', 'accounting': 'high'
        }
    
    def generate_explanation(self, query: str, candidate: Dict[str, Any], 
                           similarity_score: float, cross_encoder_score: float) -> str:
        """Generate explanation for why candidate matches query"""
        explanations = []
        
        # Extract candidate info
        skills = candidate.get('skills', [])
        experience_years = candidate.get('total_experience_years', 0)
        resume_text = candidate.get('resume_text', '')
        
        # 1. Skill matches
        matching_skills = self._find_matching_skills(query, skills)
        if matching_skills:
            high_importance_skills = [s for s in matching_skills 
                                    if self.skill_importance.get(s.lower(), 'low') == 'high']
            if high_importance_skills:
                explanations.append(f"Strong match in key skills: {', '.join(high_importance_skills[:3])}")
            else:
                explanations.append(f"Relevant skills: {', '.join(matching_skills[:3])}")
        
        # 2. Experience match
        if experience_years > 0:
            exp_level = self._determine_experience_level(query, experience_years)
            if exp_level:
                explanations.append(f"Experience level matches: {exp_level} ({experience_years} years)")
        
        # 3. Domain relevance
        domain = self._identify_domain(query, resume_text)
        if domain:
            explanations.append(f"Relevant {domain} background")
        
        # 4. Overall match quality
        if similarity_score > 0.7:
            explanations.append("High semantic similarity with query")
        elif similarity_score > 0.5:
            explanations.append("Good semantic match")
        elif similarity_score > 0.3:
            explanations.append("Moderate relevance")
        
        # 5. Add highlight snippet
        snippet = self._highlight_snippet(query, resume_text)
        if snippet:
            explanations.append(f"Snippet: {snippet}")
        
        # Combine explanations
        if explanations:
            return ". ".join(explanations[:3]) + "."
        else:
            return f"General relevance score: {similarity_score:.2f}"

    def _highlight_snippet(self, query: str, text: str, window: int = 60) -> Optional[str]:
        """Return a small highlighted snippet showing matches"""
        if not text:
            return None
        q_words = [w for w in query.lower().split() if len(w) > 2]
        if not q_words:
            return None
        text_lower = text.lower()
        best_pos = -1
        best_word = None
        for w in q_words:
            pos = text_lower.find(w)
            if pos != -1:
                best_pos = pos
                best_word = w
                break
        if best_pos == -1:
            return None
        start = max(0, best_pos - window)
        end = min(len(text), best_pos + window)
        snippet = text[start:end].strip()
        return snippet
    
    def _find_matching_skills(self, query: str, skills: List[str]) -> List[str]:
        """Find skills that match query terms"""
        query_lower = query.lower()
        matching_skills = []
        
        for skill in skills:
            if skill.lower() in query_lower:
                matching_skills.append(skill)
        
        return matching_skills
    
    def _determine_experience_level(self, query: str, experience_years: int) -> Optional[str]:
        """Determine experience level based on query and years"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['senior', 'lead', 'principal', 'architect']):
            return 'Senior' if experience_years >= 5 else None
        elif any(word in query_lower for word in ['junior', 'entry', 'graduate', 'intern']):
            return 'Junior' if experience_years <= 3 else None
        elif experience_years >= 5:
            return 'Experienced'
        elif experience_years >= 2:
            return 'Mid-level'
        else:
            return 'Entry-level'
    
    def _identify_domain(self, query: str, resume_text: str) -> Optional[str]:
        """Identify domain from query and resume text"""
        combined_text = f"{query} {resume_text}".lower()
        
        domains = {
            'technology': ['python', 'java', 'javascript', 'react', 'aws', 'docker', 'kubernetes', 'sql', 'ai', 'ml'],
            'healthcare': ['nurse', 'doctor', 'physician', 'medical', 'healthcare', 'hospital', 'patient', 'rn', 'md'],
            'finance': ['finance', 'banking', 'investment', 'accounting', 'financial', 'analyst', 'trading'],
            'education': ['teacher', 'professor', 'education', 'teaching', 'academic', 'university'],
            'marketing': ['marketing', 'advertising', 'brand', 'digital', 'social media', 'content', 'seo']
        }
        
        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            return best_domain if domain_scores[best_domain] > 0 else None
        
        return None

# Custom query enhancement system
class CustomQueryEnhancer:
    """Enhance search queries with synonyms and variations"""
    
    def __init__(self):
        # Synonym mappings
        self.synonyms = {
            'developer': ['programmer', 'coder', 'engineer', 'software engineer'],
            'engineer': ['developer', 'programmer', 'architect', 'software engineer'],
            'manager': ['lead', 'supervisor', 'director', 'head'],
            'senior': ['lead', 'principal', 'experienced', 'expert'],
            'junior': ['entry', 'associate', 'graduate', 'trainee'],
            'python': ['py', 'python3', 'python programming'],
            'javascript': ['js', 'node', 'nodejs', 'javascript programming'],
            'react': ['reactjs', 'react.js', 'react development'],
            'aws': ['amazon web services', 'amazon', 'cloud computing'],
            'docker': ['containerization', 'containers', 'docker containers'],
            'kubernetes': ['k8s', 'container orchestration', 'k8s orchestration'],
            'machine learning': ['ml', 'ai', 'artificial intelligence', 'data science'],
            'data science': ['data scientist', 'data analysis', 'machine learning'],
            'nurse': ['rn', 'registered nurse', 'nursing'],
            'doctor': ['physician', 'md', 'medical doctor'],
            'finance': ['financial', 'banking', 'investment'],
            'accounting': ['accountant', 'financial accounting', 'bookkeeping']
        }
        
        # Intent patterns
        self.intent_patterns = {
            'experience_level': r'\b(senior|junior|lead|principal|architect|manager|director|entry|graduate|intern|associate)\b',
            'skill_requirement': r'\b(python|java|javascript|react|aws|docker|kubernetes|sql|machine learning|ai|ml|nurse|doctor|finance|accounting)\b',
            'domain_focus': r'\b(technology|healthcare|finance|education|marketing|it|medical|banking)\b'
        }
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """Enhance query with synonyms, variations, and intent analysis"""
        enhanced = {
            'original_query': query,
            'expanded_terms': [],
            'synonyms': [],
            'intent': {},
            'variations': [],
            'must_terms': [],
            'should_terms': [],
            'must_not_terms': [],
            'phrases': []
        }
        
        query_lower = query.lower()
        words = query_lower.split()
        # Parse boolean operators and phrases in quotes
        enhanced.update(self._parse_boolean_and_phrases(query))
        
        # Extract synonyms
        for word in words:
            if word in self.synonyms:
                enhanced['synonyms'].extend(self.synonyms[word])
        
        # Create expanded terms
        enhanced['expanded_terms'] = list(set(words + enhanced['synonyms']))
        
        # Analyze intent
        enhanced['intent'] = self._analyze_intent(query)
        
        # Generate variations
        enhanced['variations'] = self._generate_variations(query)
        
        return enhanced

    def _parse_boolean_and_phrases(self, query: str) -> Dict[str, Any]:
        data = {
            'must_terms': [],
            'should_terms': [],
            'must_not_terms': [],
            'phrases': []
        }
        # Extract phrases in quotes
        phrase_matches = re.findall(r'"([^"]+)"', query)
        data['phrases'] = phrase_matches
        # Remove phrases for term parsing
        q = re.sub(r'"[^"]+"', ' ', query)
        tokens = q.split()
        current_op = 'SHOULD'
        for tok in tokens:
            t = tok.lower()
            if t in ('and', '&&'):
                current_op = 'AND'
                continue
            if t in ('or', '||'):
                current_op = 'OR'
                continue
            if t in ('not', '!-', '-'):
                current_op = 'NOT'
                continue
            if current_op == 'AND':
                data['must_terms'].append(t)
            elif current_op == 'NOT':
                data['must_not_terms'].append(t)
            else:
                data['should_terms'].append(t)
        return data
    
    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent"""
        intent = {
            'experience_level': None,
            'required_skills': [],
            'domain': None,
            'urgency': 'normal'
        }
        
        query_lower = query.lower()
        
        # Extract experience level
        for pattern_name, pattern in self.intent_patterns.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                if pattern_name == 'experience_level':
                    intent['experience_level'] = matches[0]
                elif pattern_name == 'skill_requirement':
                    intent['required_skills'].extend(matches)
                elif pattern_name == 'domain_focus':
                    intent['domain'] = matches[0]
        
        # Detect urgency
        urgency_indicators = ['urgent', 'asap', 'immediately', 'quickly', 'fast']
        if any(indicator in query_lower for indicator in urgency_indicators):
            intent['urgency'] = 'high'
        
        return intent
    
    def _generate_variations(self, query: str) -> List[str]:
        """Generate query variations"""
        variations = [query]
        
        # Add synonym variations
        words = query.lower().split()
        for i, word in enumerate(words):
            if word in self.synonyms:
                for synonym in self.synonyms[word][:2]:  # Limit to 2 synonyms per word
                    variation_words = words.copy()
                    variation_words[i] = synonym
                    variations.append(' '.join(variation_words))
        
        # Add skill-focused variations
        if any(skill in query.lower() for skill in ['python', 'java', 'javascript', 'react']):
            variations.append(f"{query} developer")
            variations.append(f"{query} engineer")
        
        return list(set(variations))[:5]  # Limit to 5 variations

# Custom domain classifier
class CustomDomainClassifier:
    """Enhanced domain classifier using custom LLM features"""
    
    def __init__(self):
        # Domain-specific feature weights
        self.domain_features = {
            'technology': {
                'keywords': ['python', 'java', 'javascript', 'react', 'aws', 'docker', 'kubernetes', 'sql', 'ai', 'ml', 'data science', 'backend', 'frontend', 'devops', 'cloud'],
                'roles': ['developer', 'engineer', 'programmer', 'architect', 'data scientist', 'devops engineer'],
                'weight': 1.0
            },
            'healthcare': {
                'keywords': ['nurse', 'doctor', 'physician', 'medical', 'healthcare', 'hospital', 'clinic', 'patient', 'rn', 'md', 'icu', 'emergency', 'surgery', 'pediatric'],
                'roles': ['nurse', 'doctor', 'physician', 'therapist', 'pharmacist', 'medical assistant'],
                'weight': 1.0
            },
            'finance': {
                'keywords': ['finance', 'banking', 'investment', 'accounting', 'financial', 'analyst', 'advisor', 'trading', 'portfolio', 'risk', 'compliance'],
                'roles': ['financial analyst', 'accountant', 'banker', 'investment advisor', 'risk manager'],
                'weight': 1.0
            },
            'education': {
                'keywords': ['teacher', 'professor', 'education', 'teaching', 'academic', 'university', 'college', 'school', 'curriculum', 'pedagogy'],
                'roles': ['teacher', 'professor', 'instructor', 'educator', 'academic advisor'],
                'weight': 1.0
            },
            'marketing': {
                'keywords': ['marketing', 'advertising', 'brand', 'digital', 'social media', 'content', 'seo', 'sem', 'campaign', 'strategy', 'analytics'],
                'roles': ['marketer', 'advertiser', 'brand manager', 'content creator', 'social media manager'],
                'weight': 1.0
            }
        }
        
        # Compile patterns for faster matching
        self.compiled_patterns = {}
        for domain, features in self.domain_features.items():
            patterns = []
            for keyword in features['keywords']:
                patterns.append(re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE))
            for role in features['roles']:
                patterns.append(re.compile(rf'\b{re.escape(role)}\b', re.IGNORECASE))
            self.compiled_patterns[domain] = patterns
    
    def classify_domain(self, text: str) -> Tuple[str, float]:
        """Classify domain with enhanced confidence scoring"""
        if not text:
            return 'unknown', 0.0
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, features in self.domain_features.items():
            score = 0.0
            matches = 0
            
            # Count keyword matches
            for pattern in self.compiled_patterns[domain]:
                pattern_matches = len(pattern.findall(text_lower))
                if pattern_matches > 0:
                    matches += pattern_matches
                    score += min(pattern_matches * 0.1, 0.5)  # Cap at 0.5 per pattern
            
            # Apply domain weight
            score *= features['weight']
            
            # Normalize score
            if matches > 0:
                domain_scores[domain] = min(score, 1.0)
        
        if not domain_scores:
            return 'unknown', 0.0
        
        # Normalize confidence across domains (softmax-like)
        vals = np.array(list(domain_scores.values()), dtype=float)
        if vals.size > 0:
            exp_vals = np.exp(vals)
            probs = exp_vals / np.sum(exp_vals)
            best_idx = int(np.argmax(probs))
            best_domain = list(domain_scores.keys())[best_idx]
            confidence = float(probs[best_idx])
        else:
            best_domain = max(domain_scores, key=domain_scores.get)
            confidence = domain_scores[best_domain]
        
        # Boost confidence for very specific matches
        if confidence > 0.7:
            confidence = min(confidence * 1.2, 1.0)
        
        return best_domain, confidence
    
    def should_filter_candidate(self, candidate_domain: str, query_domain: str, 
                              candidate_confidence: float, query_confidence: float) -> bool:
        """Enhanced filtering logic"""
        # If domains are unknown, don't filter
        if candidate_domain == 'unknown' or query_domain == 'unknown':
            return False
        
        # If domains match, don't filter
        if candidate_domain == query_domain:
            return False
        
        # Enhanced filtering logic
        if candidate_confidence > 0.8 and query_confidence > 0.8 and candidate_domain != query_domain:
            return True
        
        # Filter if candidate has very high confidence in a different domain
        if candidate_confidence > 0.9 and candidate_domain != query_domain:
            return True
        
        # Don't filter if both have low confidence
        if candidate_confidence < 0.5 and query_confidence < 0.5:
            return False
        
        return False

if __name__ == "__main__":
    # Test the custom LLM components
    print("🧪 Testing Custom LLM Components...")
    
    # Test tokenizer
    tokenizer = CustomTokenizer()
    test_text = "Senior Python developer with 5 years AWS experience"
    tokens = tokenizer.tokenize(test_text)
    entities = tokenizer.extract_entities(test_text)
    print(f"Tokenizer test: {tokens}")
    print(f"Entities: {entities}")
    
    # Test embedding model
    embedding_model = CustomEmbeddingModel()
    test_docs = [
        "Senior Python developer with AWS experience",
        "Nurse with ICU experience",
        "Financial analyst with banking background"
    ]
    embedding_model.fit(test_docs)
    
    test_query = "Python developer"
    test_candidate = "Senior Python developer with machine learning experience"
    similarity = embedding_model.similarity(test_query, test_candidate)
    print(f"Embedding similarity: {similarity:.3f}")
    
    # Test cross-encoder
    cross_encoder = CustomCrossEncoder()
    candidate = {
        'skills': ['Python', 'AWS', 'Machine Learning'],
        'resume_text': 'Senior Python developer with 5 years experience',
        'total_experience_years': 5
    }
    score = cross_encoder.encode_pair(test_query, candidate)
    print(f"Cross-encoder score: {score:.3f}")
    
    # Test explanation generator
    explanation_gen = CustomExplanationGenerator()
    explanation = explanation_gen.generate_explanation(test_query, candidate, similarity, score)
    print(f"Explanation: {explanation}")
    
    # Test query enhancer
    query_enhancer = CustomQueryEnhancer()
    enhanced = query_enhancer.enhance_query(test_query)
    print(f"Enhanced query: {enhanced}")
    
    # Test domain classifier
    domain_classifier = CustomDomainClassifier()
    domain, confidence = domain_classifier.classify_domain(test_candidate)
    print(f"Domain classification: {domain} (confidence: {confidence:.3f})")
    
    print("✅ All tests completed successfully!")
