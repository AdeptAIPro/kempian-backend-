"""
Candidate Clustering

Clusters candidates into similar groups for segmentation and targeting.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import sklearn
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")


class CandidateSegmenter:
    """
    Candidate Segmenter
    
    Clusters candidates into similar groups using ML algorithms
    """
    
    def __init__(self, n_clusters: int = 10, algorithm: str = 'gmm'):
        """
        Initialize Candidate Segmenter
        
        Args:
            n_clusters: Number of clusters/segments
            algorithm: Clustering algorithm ('gmm', 'kmeans', 'dbscan')
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")
        
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        
        # Initialize model
        if algorithm == 'gmm':
            self.model = GaussianMixture(n_components=n_clusters, random_state=42)
        elif algorithm == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features from candidates
        
        Args:
            candidates: List of candidate dictionaries
            
        Returns:
            Feature matrix [n_candidates, n_features]
        """
        features = []
        
        for candidate in candidates:
            feature_vec = []
            
            # Skill count
            skills = candidate.get('skills', [])
            feature_vec.append(len(skills))
            
            # Experience years
            experience = candidate.get('total_experience_years', 0)
            feature_vec.append(experience)
            
            # Domain encoding (simple)
            domain = candidate.get('domain', 'unknown')
            domain_map = {
                'technology': 1.0,
                'healthcare': 0.8,
                'finance': 0.6,
                'education': 0.4,
                'marketing': 0.2,
                'unknown': 0.0
            }
            feature_vec.append(domain_map.get(domain, 0.0))
            
            # Skill diversity (unique skill categories)
            skill_categories = self._categorize_skills(skills)
            feature_vec.append(len(skill_categories))
            
            # Resume length (proxy for detail level)
            resume_text = candidate.get('resume_text', '')
            feature_vec.append(len(resume_text.split()))
            
            # Behavioral score (if available)
            behavioral_score = candidate.get('behavioral_score', 0.5)
            feature_vec.append(behavioral_score)
            
            features.append(feature_vec)
        
        return np.array(features, dtype=np.float32)
    
    def _categorize_skills(self, skills: List[str]) -> set:
        """Categorize skills into groups"""
        categories = set()
        
        programming = {'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust'}
        web = {'react', 'vue', 'angular', 'node', 'django', 'flask'}
        cloud = {'aws', 'azure', 'gcp', 'docker', 'kubernetes'}
        database = {'sql', 'mysql', 'postgresql', 'mongodb', 'redis'}
        ml_ai = {'tensorflow', 'pytorch', 'machine learning', 'deep learning', 'ai'}
        medical = {'nursing', 'patient care', 'medical', 'surgery', 'pharmacy'}
        finance = {'accounting', 'financial', 'cfa', 'cpa', 'trading'}
        
        skill_lower = {s.lower() for s in skills}
        
        if skill_lower & programming:
            categories.add('programming')
        if skill_lower & web:
            categories.add('web')
        if skill_lower & cloud:
            categories.add('cloud')
        if skill_lower & database:
            categories.add('database')
        if skill_lower & ml_ai:
            categories.add('ml_ai')
        if skill_lower & medical:
            categories.add('medical')
        if skill_lower & finance:
            categories.add('finance')
        
        return categories
    
    def segment_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Segment candidates into clusters
        
        Args:
            candidates: List of candidate dictionaries
            
        Returns:
            Dictionary with segment information
        """
        if not candidates:
            return {}
        
        # Extract features
        features = self.extract_features(candidates)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Cluster
        if self.algorithm == 'dbscan':
            clusters = self.model.fit_predict(features_scaled)
            # DBSCAN may produce -1 (noise) clusters
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        else:
            clusters = self.model.fit_predict(features_scaled)
            n_clusters = self.n_clusters
        
        self.is_trained = True
        
        # Analyze clusters
        segments = {}
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Noise cluster in DBSCAN
                continue
            
            cluster_candidates = [c for c, cl in zip(candidates, clusters) if cl == cluster_id]
            
            # Analyze cluster characteristics
            characteristics = self._analyze_cluster(cluster_candidates)
            
            segments[cluster_id] = {
                'candidates': cluster_candidates,
                'characteristics': characteristics,
                'size': len(cluster_candidates),
                'cluster_id': int(cluster_id)
            }
        
        return {
            'segments': segments,
            'n_segments': len(segments),
            'algorithm': self.algorithm,
            'cluster_assignments': clusters.tolist()
        }
    
    def _analyze_cluster(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of a cluster"""
        if not candidates:
            return {}
        
        # Average experience
        avg_experience = np.mean([c.get('total_experience_years', 0) for c in candidates])
        
        # Common skills
        all_skills = []
        for candidate in candidates:
            all_skills.extend(candidate.get('skills', []))
        
        from collections import Counter
        skill_counts = Counter(all_skills)
        top_skills = [skill for skill, count in skill_counts.most_common(5)]
        
        # Common domain
        domains = [c.get('domain', 'unknown') for c in candidates]
        from collections import Counter
        domain_counts = Counter(domains)
        common_domain = domain_counts.most_common(1)[0][0] if domain_counts else 'unknown'
        
        return {
            'avg_experience': float(avg_experience),
            'top_skills': top_skills,
            'common_domain': common_domain,
            'size': len(candidates)
        }
    
    def predict_cluster(self, candidate: Dict[str, Any]) -> int:
        """
        Predict cluster for a single candidate
        
        Args:
            candidate: Candidate dictionary
            
        Returns:
            Cluster ID
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call segment_candidates first.")
        
        features = self.extract_features([candidate])
        features_scaled = self.scaler.transform(features)
        
        if self.algorithm == 'dbscan':
            # DBSCAN doesn't support prediction, use nearest cluster
            cluster = self.model.fit_predict(features_scaled)[0]
        else:
            cluster = self.model.predict(features_scaled)[0]
        
        return int(cluster)


# Global instance
_segmenter = None


def get_segmenter(n_clusters: int = 10, algorithm: str = 'gmm') -> CandidateSegmenter:
    """Get or create global segmenter instance"""
    global _segmenter
    if _segmenter is None:
        _segmenter = CandidateSegmenter(n_clusters=n_clusters, algorithm=algorithm)
    return _segmenter

