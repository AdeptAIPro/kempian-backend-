import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class DenseRetrievalMatcher:
    """
    High-performance dense retrieval system using FAISS
    for efficient semantic search at scale
    """
    
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # FAISS index for fast similarity search
        self.index = None
        self.candidate_metadata = []
        
    def build_index(self, candidates: List[Dict], index_type='IVF'):
        """Build FAISS index for efficient search"""
        
        logger.info(f"Building index for {len(candidates)} candidates...")
        
        # Extract embeddings
        texts = []
        for candidate in candidates:
            resume_text = candidate.get('resume_text', '')
            skills_text = ' '.join(candidate.get('skills', []))
            combined_text = f"{resume_text} {skills_text}"
            texts.append(combined_text)
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts, 
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Build index based on size
        if len(candidates) < 10000:
            # Use flat index for small datasets
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine sim
        else:
            # Use IVF index for large datasets
            nlist = min(int(np.sqrt(len(candidates))), 1000)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        self.candidate_metadata = candidates
        
        logger.info(f"Index built successfully with {self.index.ntotal} vectors")
        
    def search(self, query: str, top_k: int = 10, domain: str = None) -> List[Tuple[Dict, float]]:
        """Search for top matching candidates"""
        
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return results with metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                candidate = self.candidate_metadata[idx]
                results.append((candidate, float(score)))
        
        return results
    
    def save_index(self, filepath: str):
        """Save the FAISS index and metadata"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index, f"{filepath}.index")
        with open(f"{filepath}.metadata", 'wb') as f:
            pickle.dump(self.candidate_metadata, f)
    
    def load_index(self, filepath: str):
        """Load the FAISS index and metadata"""
        self.index = faiss.read_index(f"{filepath}.index")
        with open(f"{filepath}.metadata", 'rb') as f:
            self.candidate_metadata = pickle.load(f)


class ReRanker:
    """
    Re-rank top candidates using cross-encoder
    """
    
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        try:
            self.cross_encoder = CrossEncoder(model_name)
            self.available = True
        except Exception as e:
            logger.warning(f"Cross-encoder not available: {e}")
            self.cross_encoder = None
            self.available = False
        
    def rerank(self, query: str, candidates: List[Tuple[Dict, float]], 
               top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Re-rank top candidates using cross-encoder"""
        
        if not self.available or len(candidates) <= top_k:
            return candidates[:top_k]
        
        try:
            # Prepare pairs for cross-encoder
            pairs = []
            for candidate, _ in candidates:
                resume_text = candidate.get('resume_text', '')
                skills = ' '.join(candidate.get('skills', []))
                combined_text = f"{resume_text} {skills}"
                pairs.append([query, combined_text])
            
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Combine with original scores and re-rank
            reranked = []
            for i, (candidate, original_score) in enumerate(candidates):
                combined_score = 0.6 * cross_scores[i] + 0.4 * original_score
                reranked.append((candidate, combined_score))
            
            # Sort and return top_k
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates[:top_k]


class ProductionMatcher:
    """
    Production-ready matcher combining dense retrieval with re-ranking
    """
    
    def __init__(self, index_path: str = None):
        self.dense_retriever = DenseRetrievalMatcher()
        self.reranker = ReRanker()
        
        if index_path and os.path.exists(f"{index_path}.index"):
            try:
                self.dense_retriever.load_index(index_path)
                logger.info("Loaded existing index")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
    
    def index_candidates(self, candidates: List[Dict], save_path: str = None):
        """Index candidates for fast retrieval"""
        self.dense_retriever.build_index(candidates)
        
        if save_path:
            self.dense_retriever.save_index(save_path)
    
    def find_matches(self, job_description: str, top_k: int = 10, 
                    rerank_top: int = 20, domain: str = None) -> List[Dict]:
        """
        Find and rank candidates
        
        Args:
            job_description: Job posting text
            top_k: Final number of candidates to return
            rerank_top: Number of candidates to retrieve for re-ranking
            domain: Domain hint for better model selection
        """
        
        # Step 1: Dense retrieval
        candidates = self.dense_retriever.search(
            job_description, 
            top_k=rerank_top, 
            domain=domain
        )
        
        # Step 2: Re-ranking
        reranked = self.reranker.rerank(
            job_description, 
            candidates, 
            top_k=top_k
        )
        
        # Step 3: Format results
        results = []
        for candidate, score in reranked:
            result = {
                **candidate,
                'match_score': float(score),
                'match_percentage': min(int(score * 100), 100)
            }
            results.append(result)
        
        return results
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index"""
        if self.dense_retriever.index is None:
            return {'status': 'No index built'}
        
        return {
            'total_candidates': self.dense_retriever.index.ntotal,
            'index_type': type(self.dense_retriever.index).__name__,
            'dimension': self.dense_retriever.dimension,
            'model_name': self.dense_retriever.model._modules['0'].auto_model.name_or_path
        }


# Integration function for existing DynamoDB system
def create_production_matcher_from_dynamodb(table, index_path: str = "indexes/resume_index"):
    """
    Create a production matcher from DynamoDB data
    """
    matcher = ProductionMatcher(index_path)
    
    # If index doesn't exist, build it from DynamoDB
    if matcher.dense_retriever.index is None:
        logger.info("Building new index from DynamoDB...")
        
        # Scan DynamoDB table
        response = table.scan()
        items = response.get('Items', [])
        
        # Convert DynamoDB items to standard format
        candidates = []
        for item in items:
            candidate = {
                'email': item.get('email', ''),
                'full_name': item.get('full_name') or item.get('FullName', ''),
                'phone': item.get('phone', ''),
                'skills': item.get('skills') or item.get('Skills', []),
                'resume_text': item.get('resume_text') or item.get('ResumeText', ''),
                'total_experience_years': item.get('total_experience_years') or item.get('Experience', 0),
                'sourceURL': item.get('sourceURL') or item.get('SourceURL', '')
            }
            candidates.append(candidate)
        
        # Build and save index
        matcher.index_candidates(candidates, index_path)
        logger.info(f"Index built with {len(candidates)} candidates")
    
    return matcher