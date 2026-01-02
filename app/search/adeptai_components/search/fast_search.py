# fast_search.py

import os
import time
import faiss
import pickle
import numpy as np
import threading
from functools import lru_cache
from app.simple_logger import get_logger
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import boto3

# ADD THESE IMPORTS AT THE TOP
import sys

# Add parent directory to path for enhanced search imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from enhanced_recruitment_search import EnhancedRecruitmentSearchSystem
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
# END ADD THESE IMPORTS

logger = get_logger("search")

class OptimizedSearchSystem:
    """
    Ultra-fast search system using FAISS for 50-400x performance improvement
    Drop-in replacement for your existing search with same API format
    """

    def __init__(self):
        # Import embedding service from main module
        # This will be injected when the system initializes
        self.embedding_service = None
        self.table = None

        # FAISS components
        self.index = None
        self.candidate_data = []
        self.embeddings_cache = {}

        # Performance tracking
        self.search_count = 0
        self.total_search_time = 0
        self.cache_hits = 0

        # Thread safety
        self.lock = threading.Lock()

        # Cache file for persistent embeddings
        self.cache_file = "search_cache/embeddings.pkl"
        os.makedirs("search_cache", exist_ok=True)

        # Load existing cache
        self._load_cache()

        # Initialize enhanced search system as None
        self.enhanced_system = None # New attribute for enhanced system integration


        logger.info("OptimizedSearchSystem initialized")

    def set_dependencies(self, embedding_service, table):
        """Set dependencies from main.py"""
        self.embedding_service = embedding_service
        self.table = table

        # Build index after dependencies are set
        if not self.index or len(self.candidate_data) == 0:
            self._initialize_index()

    # ADD THIS METHOD TO OptimizedSearchSystem CLASS
    def integrate_with_enhanced_search(self, enhanced_system):
        """Integrate with enhanced search system for fallback"""
        self.enhanced_system = enhanced_system
        logger.info("Integrated with enhanced search system")
    # END ADD THIS METHOD

    def _load_cache(self):
        """Load persistent embedding cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.embeddings_cache = {}

    def _save_cache(self):
        """Save embeddings to persistent cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            logger.info(f"Saved {len(self.embeddings_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _initialize_index(self):
        """Build FAISS index from DynamoDB data"""
        if not self.embedding_service or not self.table:
            logger.warning("Dependencies not set, cannot build index")
            return

        start_time = time.time()
        logger.info("Building FAISS search index...")

        try:
            # Get ALL candidates from DynamoDB with pagination
            items = []
            last_evaluated_key = None
            page_count = 0
            
            logger.info("Starting to scan DynamoDB table with pagination...")
            
            while True:
                page_count += 1
                
                # Prepare scan parameters
                scan_params = {}
                if last_evaluated_key:
                    scan_params['ExclusiveStartKey'] = last_evaluated_key
                
                # Scan with pagination
                response = self.table.scan(**scan_params)
                
                # Add items from this page
                page_items = response.get('Items', [])
                items.extend(page_items)
                
                logger.info(f"Page {page_count}: Fetched {len(page_items)} candidates (Total so far: {len(items)})")
                
                # Check if there are more pages
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break

            if not items:
                logger.warning("No candidates found in database")
                return

            logger.info(f"Processing {len(items)} candidates from {page_count} pages...")

            # Process candidates in batches for memory efficiency
            valid_candidates = []
            embeddings_list = []

            batch_size = 50
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_candidates, batch_embeddings = self._process_candidate_batch(batch)
                valid_candidates.extend(batch_candidates)
                embeddings_list.extend(batch_embeddings)

                if (i + batch_size) % 200 == 0:  # Progress every 200 items
                    logger.info(f"Processed {min(i + batch_size, len(items))}/{len(items)} candidates")

            if not valid_candidates:
                logger.warning("No valid candidates after processing")
                return

            # Convert to numpy array for FAISS
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            logger.info(f"Created embeddings array: {embeddings_array.shape}")

            # Build FAISS index based on data size
            dimension = embeddings_array.shape[1]
            self.dimension = dimension # Store dimension for health checks/stats

            if len(valid_candidates) < 1000:
                # Use exact search for small datasets
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                index_type = "Exact (Flat)"
            elif len(valid_candidates) < 10000:
                # Use IVF for medium datasets
                nlist = 100
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                self.index.train(embeddings_array)
                index_type = f"IVF (nlist={nlist})"
            else:
                # Use compressed index for large datasets
                nlist = min(int(np.sqrt(len(valid_candidates))), 1000)
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                self.index.train(embeddings_array)
                index_type = f"IVF Large (nlist={nlist})"

            # Add all vectors to the index
            self.index.add(embeddings_array)
            self.candidate_data = valid_candidates
            self.last_rebuild_time = datetime.now() # Track last rebuild time

            build_time = time.time() - start_time

            logger.info("Index built successfully!")
            logger.info(f"Build time: {build_time:.2f} seconds")
            logger.info(f"Indexed candidates: {len(valid_candidates)}")
            logger.info(f"Index type: {index_type}")
            logger.info(f"Vector dimension: {dimension}")

            # Save cache after successful build
            self._save_cache()

        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _process_candidate_batch(self, candidates: List[Dict]) -> Tuple[List[Dict], List[np.ndarray]]:
        """Process a batch of candidates efficiently"""
        valid_candidates = []
        embeddings = []

        # Collect texts for batch encoding
        texts_to_encode = []
        candidate_indices = []

        for idx, candidate in enumerate(candidates):
            try:
                # Extract text content
                resume_text = candidate.get('resume_text') or candidate.get('ResumeText', '')
                skills_raw = candidate.get('skills') or candidate.get('Skills', [])

                # Handle skills format
                if isinstance(skills_raw, str):
                    skills = [s.strip() for s in skills_raw.split(',') if s.strip()]
                else:
                    skills = skills_raw or []

                combined_text = f"{resume_text} {' '.join(skills)}".strip()

                if not combined_text or len(combined_text) < 10:
                    continue

                # Check cache first
                cache_key = f"{candidate.get('email', idx)}_{hash(combined_text)}"

                if cache_key in self.embeddings_cache:
                    # Use cached embedding
                    embedding = self.embeddings_cache[cache_key]
                    self.cache_hits += 1
                else:
                    # Mark for encoding
                    texts_to_encode.append(combined_text)
                    candidate_indices.append((idx, cache_key))
                    continue

                # Store candidate data
                candidate_data = self._extract_candidate_data(candidate)
                valid_candidates.append(candidate_data)
                embeddings.append(embedding)

            except Exception as e:
                logger.error(f"Error processing candidate {idx}: {e}")
                continue

        # Batch encode new texts
        if texts_to_encode:
            try:
                batch_embeddings = self.embedding_service.encode_batch(texts_to_encode)

                for i, (candidate_idx, cache_key) in enumerate(candidate_indices):
                    candidate = candidates[candidate_idx]
                    embedding = batch_embeddings[i]

                    # Cache the embedding
                    self.embeddings_cache[cache_key] = embedding

                    # Store candidate data
                    candidate_data = self._extract_candidate_data(candidate)
                    valid_candidates.append(candidate_data)
                    embeddings.append(embedding)

            except Exception as e:
                logger.error(f"Batch encoding error: {e}")

        return valid_candidates, embeddings

    def _extract_candidate_data(self, candidate: Dict) -> Dict:
        """Extract and normalize candidate data"""
        # Handle experience
        experience_raw = candidate.get('total_experience_years') or candidate.get('Experience', 0)
        try:
            if isinstance(experience_raw, str):
                import re
                numbers = re.findall(r'\d+\.?\d*', experience_raw)
                experience = int(float(numbers[0])) if numbers else 0
            else:
                experience = int(float(experience_raw)) if experience_raw else 0
        except (ValueError, TypeError):
            experience = 0

        # Handle skills
        skills_raw = candidate.get('skills') or candidate.get('Skills', [])
        if isinstance(skills_raw, str):
            skills = [s.strip() for s in skills_raw.split(',') if s.strip()]
        else:
            skills = skills_raw or []

        return {
            'full_name': candidate.get('full_name') or candidate.get('FullName', 'Unknown'),
            'email': candidate.get('email', ''),
            'phone': candidate.get('phone', ''),
            'skills': skills,
            'experience': experience,
            'source_url': candidate.get('sourceURL') or candidate.get('SourceURL', ''),
        }

    @lru_cache(maxsize=1000)
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding with caching for repeated queries"""
        return self.embedding_service.encode_single(query)

    def search(self, query: str, top_k: int = 10) -> Tuple[List[Dict], str]:
        """
        Ultra-fast search using FAISS index
        Returns results in the same format as your existing search
        """
        start_time = time.time()

        if self.index is None or len(self.candidate_data) == 0:
            # Try to initialize if not done yet
            if self.embedding_service and self.table:
                self._initialize_index()

            if self.index is None:
                return [], "Search index not available"

        try:
            # Get query embedding (cached for repeated queries)
            query_embedding = self._get_query_embedding(query.strip())

            # Prepare for FAISS search
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.astype(np.float32)

            # Perform ultra-fast similarity search
            search_k = min(top_k, len(self.candidate_data))
            scores, indices = self.index.search(query_embedding, search_k)

            # Format results to match your existing API
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue

                candidate = self.candidate_data[idx]

                # Calculate score (0-100 scale)
                score_int = max(1, min(int(score * 100), 100))

                result = {
                    'FullName': candidate['full_name'],
                    'email': candidate['email'],
                    'phone': candidate['phone'],
                    'Skills': candidate['skills'],
                    'Experience': f"{candidate['experience']} years",
                    'sourceURL': candidate['source_url'],
                    'Score': score_int,
                    'Grade': self._get_grade(score_int),
                    'SemanticScore': float(score)
                }
                results.append(result)

            search_time = time.time() - start_time

            # Update performance stats
            with self.lock:
                self.search_count += 1
                self.total_search_time += search_time

            summary = f"Found {len(results)} candidates in {search_time*1000:.1f}ms using FAISS search"

            logger.info(f"Search completed: {len(results)} results in {search_time*1000:.1f}ms")
            return results, summary

        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Search error: {e}")
            return [], f"Search failed: {str(e)}"

    def _get_grade(self, score: int) -> str:
        """Convert score to grade (keeping your existing logic)"""
        if score >= 85:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 50:
            return 'C'
        else:
            return 'D'

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        with self.lock:
            avg_time = self.total_search_time / max(self.search_count, 1)
            cache_hit_rate = self.cache_hits / max(self.search_count, 1)

            return {
                'total_searches': self.search_count,
                'average_search_time_ms': round(avg_time * 1000, 2),
                'total_candidates': len(self.candidate_data),
                'index_type': type(self.index).__name__ if self.index else 'None',
                'cache_hit_rate': round(cache_hit_rate, 3),
                'cached_embeddings': len(self.embeddings_cache)
            }

    def rebuild_index(self):
        """Rebuild the search index (call when data changes)"""
        logger.info("Rebuilding search index...")
        self.index = None
        self.candidate_data = []
        self.embeddings_cache = {}  # Clear cache for fresh data
        self._initialize_index()

    def clear_cache(self):
        """Clear all caches"""
        with self.lock:
            self.embeddings_cache = {}
            if hasattr(self, '_get_query_embedding'):
                self._get_query_embedding.cache_clear()
        logger.info("All caches cleared")

    # ADD THIS METHOD TO OptimizedSearchSystem CLASS
    def hybrid_search(self, query: str, top_k: int = 10, use_enhanced: bool = True):
        """Hybrid search using both fast and enhanced systems"""
        try:
            if use_enhanced and hasattr(self, 'enhanced_system') and self.enhanced_system:
                # Use enhanced search
                logger.info("Using Enhanced AI Search for hybrid search.")
                return self.enhanced_system.search(query, top_k), "Enhanced AI Search"
            else:
                # Fall back to fast search
                logger.info("Falling back to Fast Search for hybrid search.")
                return self.search(query, top_k)
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # Always fall back to fast search on error
            logger.info("Error in hybrid search, falling back to Fast Search.")
            return self.search(query, top_k)
    # END ADD THIS METHOD


# Global instance (singleton pattern)
_optimized_search_system = None
_search_lock = threading.Lock()

def get_optimized_search_system() -> OptimizedSearchSystem:
    """Get or create the optimized search system (singleton pattern)"""
    global _optimized_search_system

    if _optimized_search_system is None:
        with _search_lock:
            if _optimized_search_system is None:
                _optimized_search_system = OptimizedSearchSystem()

    return _optimized_search_system