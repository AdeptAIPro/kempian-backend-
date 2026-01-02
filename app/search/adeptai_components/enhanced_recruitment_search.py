# enhanced_recruitment_search.py - COMPLETE FIXED VERSION WITH ADVANCED MATCHING

import os
import faiss
import numpy as np
import pickle
import logging
import time
import re
import math
import json
import hashlib
from collections import Counter, defaultdict
from app.simple_logger import get_logger
from typing import List, Dict, Any, Optional, Tuple

import boto3
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import nltk

# Import our enhanced matching system
from .enhanced_candidate_matcher import EnhancedCandidateMatchingSystem, MatchScore

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

logger = get_logger("search")

# Data Classes
class CandidateProfile:
    """Represents a standardized candidate profile."""
    def __init__(self,
                 email: str,
                 full_name: str,
                 phone: str,
                 resume_text: str,
                 skills: List[str],
                 experience_years: int,
                 education: str,
                 certifications: List[str],
                 previous_roles: List[str],
                 industries: List[str],
                 location: str,
                 source_url: str,
                 skill_proficiency: Dict[str, float],
                 seniority_level: str
                ):
        self.email = email
        self.full_name = full_name
        self.phone = phone
        self.resume_text = resume_text
        self.skills = skills
        self.experience_years = experience_years
        self.education = education
        self.certifications = certifications
        self.previous_roles = previous_roles
        self.industries = industries
        self.location = location
        self.source_url = source_url
        self.skill_proficiency = skill_proficiency
        self.seniority_level = seniority_level
        self.combined_text = self._generate_combined_text()

    def _safe_list_to_string(self, item, default: str = "") -> str:
        """Safely convert any item to string, handling lists properly"""
        if item is None:
            return default
        elif isinstance(item, list):
            clean_items = [str(x) for x in item if x is not None]
            return ', '.join(clean_items) if clean_items else default
        elif isinstance(item, str):
            return item
        else:
            return str(item)

    def _generate_combined_text(self) -> str:
        """Generates a comprehensive text representation of the candidate - FIXED VERSION"""
        try:
            # Safely convert all fields to strings
            name = self._safe_list_to_string(self.full_name, "Unknown")
            resume = self._safe_list_to_string(self.resume_text, "")
            skills_str = self._safe_list_to_string(self.skills, "")
            education_str = self._safe_list_to_string(self.education, "")
            certifications_str = self._safe_list_to_string(self.certifications, "")
            roles_str = self._safe_list_to_string(self.previous_roles, "")
            industries_str = self._safe_list_to_string(self.industries, "")
            location_str = self._safe_list_to_string(self.location, "")
            seniority_str = self._safe_list_to_string(self.seniority_level, "")
            
            # Build text parts with safe string concatenation
            text_parts = []
            
            if name:
                text_parts.append(f"Name: {name}")
            if resume:
                text_parts.append(f"Resume: {resume}")
            if skills_str:
                text_parts.append(f"Skills: {skills_str}")
            if self.experience_years and self.experience_years > 0:
                text_parts.append(f"Experience: {self.experience_years} years")
            if education_str:
                text_parts.append(f"Education: {education_str}")
            if certifications_str:
                text_parts.append(f"Certifications: {certifications_str}")
            if roles_str:
                text_parts.append(f"Previous Roles: {roles_str}")
            if industries_str:
                text_parts.append(f"Industries: {industries_str}")
            if location_str:
                text_parts.append(f"Location: {location_str}")
            if seniority_str:
                text_parts.append(f"Seniority: {seniority_str}")
            
            # Join with periods and ensure we have valid text
            combined = ". ".join(text_parts)
            
            # Fallback if somehow we get empty text
            if not combined.strip():
                combined = f"Candidate with {self.experience_years} years experience"
            
            return combined
            
        except Exception as e:
            logger.error(f"Error generating combined text for {self.email}: {e}")
            return f"Candidate: {self.email}, Experience: {self.experience_years} years, Skills: {len(self.skills) if isinstance(self.skills, list) else 0}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to a dictionary for storage/JSON output."""
        return {
            "email": self.email,
            "full_name": self.full_name,
            "phone": self.phone,
            "resume_text": self.resume_text,
            "skills": self.skills,
            "experience_years": self.experience_years,
            "education": self.education,
            "certifications": self.certifications,
            "previous_roles": self.previous_roles,
            "industries": self.industries,
            "location": self.location,
            "source_url": self.source_url,
            "skill_proficiency": self.skill_proficiency,
            "seniority_level": self.seniority_level,
            "combined_text": self.combined_text
        }

class SkillExtractor:
    """Extracts skills and their proficiency from text."""
    def __init__(self):
        self.known_skills = {
            "python": ["django", "flask", "numpy", "pandas", "scikit-learn", "tensorflow", "pytorch"],
            "java": ["spring", "hibernate", "maven", "gradle"],
            "javascript": ["react", "angular", "vue", "node.js", "express.js"],
            "aws": ["s3", "ec2", "lambda", "dynamodb"],
            "sql": ["mysql", "postgresql", "oracle"],
            "docker": [], "kubernetes": [], "git": [], "agile": [], "scrum": [],
            "machine learning": [], "deep learning": [], "nlp": [], "data science": []
        }
        self.tokenizer = RegexpTokenizer(r'\w+')
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        logger.info("SkillExtractor initialized.")

    def extract_skills_with_context(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extracts skills and infers proficiency based on context."""
        found_skills = {}
        text_lower = text.lower()

        for main_skill, sub_skills in self.known_skills.items():
            pattern = r'\b' + re.escape(main_skill) + r'\b'
            if re.search(pattern, text_lower):
                proficiency = self._infer_proficiency(text_lower, main_skill)
                found_skills[main_skill] = {"proficiency": proficiency, "mentions": text_lower.count(main_skill)}

            for sub_skill in sub_skills:
                sub_pattern = r'\b' + re.escape(sub_skill) + r'\b'
                if re.search(sub_pattern, text_lower) and sub_skill not in found_skills:
                    proficiency = self._infer_proficiency(text_lower, sub_skill)
                    found_skills[sub_skill] = {"proficiency": proficiency, "mentions": text_lower.count(sub_skill)}
        return found_skills

    def _infer_proficiency(self, text: str, skill: str) -> float:
        """Infer proficiency based on keywords like 'expert', 'proficient', 'experience'."""
        score = 0.5  # Default to intermediate

        # Keywords indicating higher proficiency
        if re.search(r'\b(expert|master|advanced|lead|senior)\s*(' + re.escape(skill) + r'|developer|engineer)\b', text):
            score = 0.9
        elif re.search(r'\b(proficient|strong|deep knowledge|extensive experience)\s*in\s*(' + re.escape(skill) + r')\b', text):
            score = 0.8
        elif re.search(r'\b(\d+\s+years?\s+experience\s*in\s*|developed|implemented|architected)\s*(' + re.escape(skill) + r')\b', text):
            score = max(score, 0.7)
        elif re.search(r'\b(familiar with|basic knowledge|interested in)\s*(' + re.escape(skill) + r')\b', text):
            score = 0.3

        return score

# Memory Optimized Embedding System
class MemoryOptimizedEmbeddingSystem:
    """Memory-optimized embedding system"""

    def __init__(self):
        self.models = {}
        self.cross_encoders = {}
        self.cache = {}
        self.cache_hits = 0
        self.cache_total = 0

        self._initialize_models_safely()

    def _initialize_models_safely(self):
        """Initialize models with memory management"""
        try:
            logger.info("Loading embedding models with memory optimization...")

            # Start with smaller, faster model
            self.models["general"] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded lightweight general model")

            # Try to load better model if memory allows
            try:
                import psutil
                memory = psutil.virtual_memory()
                available_gb = memory.available / (1024**3)

                if available_gb > 3:
                    self.models["technical"] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    logger.info("Loaded enhanced technical model")
                else:
                    logger.info("Limited memory, using lightweight model for technical tasks")
                    self.models["technical"] = self.models["general"]

            except Exception as e:
                logger.warning(f"Could not load enhanced model: {e}")
                self.models["technical"] = self.models["general"]

            # Cross-encoder (optional)
            try:
                if available_gb > 2:
                    self.cross_encoders["rerank"] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                    logger.info("Cross-encoder loaded")
                else:
                    logger.info("Skipping cross-encoder due to memory constraints")
                    self.cross_encoders["rerank"] = None
            except Exception as e:
                logger.warning(f"Cross-encoder not available: {e}")
                self.cross_encoders["rerank"] = None

            logger.info("Memory-optimized embedding system initialized")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Fallback to basic model
            self.models["general"] = SentenceTransformer('all-MiniLM-L6-v2')
            self.models["technical"] = self.models["general"]
            self.cross_encoders["rerank"] = None

    def encode(self, text: str, model_type: str = "general") -> np.ndarray:
        """Encode text to embedding with error handling and fallback."""
        if not text or not text.strip():
            # Return zero embedding for empty text
            model = self.models.get(model_type, self.models["general"])
            dimension = model.get_sentence_embedding_dimension()
            return np.zeros(dimension, dtype=np.float32)
        
        # Check cache first
        cache_key = f"{text[:100]}_{model_type}"
        if cache_key in self.cache:
            self.cache_hits += 1
            self.cache_total += 1
            return self.cache[cache_key]

        self.cache_total += 1
        
        try:
            model = self.models.get(model_type, self.models["general"])
            
            # Truncate text if it's too long to prevent memory issues
            max_length = 512  # Safe length for most models
            if len(text) > max_length:
                text = text[:max_length]
                logger.warning(f"Text truncated to {max_length} characters for embedding")
            
            embedding = model.encode(text, convert_to_numpy=True)
            
            # Validate embedding
            if embedding is None or embedding.size == 0:
                raise ValueError("Generated embedding is empty")
            
            # Manage cache size
            if len(self.cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.cache.keys())[:100]
                for key in oldest_keys:
                    del self.cache[key]
            
            self.cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for text (first 50 chars: '{text[:50]}...'): {e}")
            
            # Fallback: return a random embedding with proper dimension
            try:
                model = self.models.get(model_type, self.models["general"])
                dimension = model.get_sentence_embedding_dimension()
                # Generate a simple embedding based on text length and content
                np.random.seed(hash(text) % 2**32)  # Deterministic based on text
                fallback_embedding = np.random.normal(0, 0.1, dimension).astype(np.float32)
                # Normalize the embedding
                norm = np.linalg.norm(fallback_embedding)
                if norm > 0:
                    fallback_embedding = fallback_embedding / norm
                
                logger.info(f"Using fallback embedding for text: {text[:50]}...")
                return fallback_embedding
                
            except Exception as fallback_error:
                logger.error(f"Fallback embedding also failed: {fallback_error}")
                # Last resort: return zero embedding
                model = self.models.get(model_type, self.models["general"])
                dimension = model.get_sentence_embedding_dimension()
                return np.zeros(dimension, dtype=np.float32)

    def encode_batch(self, texts: List[str], model_type: str = "general") -> np.ndarray:
        """Encode batch of texts to embeddings - MUCH FASTER than individual encoding."""
        if not texts:
            return np.array([])
        
        try:
            model = self.models.get(model_type, self.models["general"])
            
            # Process texts in batches to avoid memory issues
            # For large datasets (200K+), use larger batches for better performance
            batch_size = 200 if len(texts) > 10000 else 100  # Larger batches for big datasets
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Clean and truncate texts
                processed_texts = []
                for text in batch_texts:
                    if not text or not text.strip():
                        continue
                    
                    # Truncate if needed
                    max_length = 512
                    if len(text) > max_length:
                        text = text[:max_length]
                    
                    processed_texts.append(text)
                
                if not processed_texts:
                    continue
                
                # Batch encode
                embeddings = model.encode(
                    processed_texts, 
                    convert_to_numpy=True, 
                    batch_size=32,
                    show_progress_bar=False
                )
                
                all_embeddings.append(embeddings)
            
            if not all_embeddings:
                return np.array([])
            
            # Combine all embeddings
            result = np.vstack(all_embeddings)
            return result
                
        except Exception as e:
            logger.error(f"Error in batch encoding: {e}")
            return np.array([])

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """Reranks texts based on query using cross-encoder."""
        if not self.cross_encoders.get("rerank"):
            logger.warning("Reranking requested but cross-encoder is not available. Returning default scores.")
            return [1.0 / (i + 1) for i in range(len(texts))]

        pairs = [[query, text] for text in texts]
        scores = self.cross_encoders["rerank"].predict(pairs).tolist()
        return scores

    def get_cache_hit_rate(self) -> float:
        """Returns the cache hit rate."""
        return (self.cache_hits / self.cache_total) * 100 if self.cache_total > 0 else 0.0

# Enhanced Recruitment Search System
class EnhancedRecruitmentSearchSystem:
    """
    An enhanced recruitment search system leveraging FAISS, semantic embeddings,
    and advanced candidate profiling.
    """
    def __init__(self, index_path: str = "enhanced_search_index"):
        self.index_path = index_path
        self.embedding_service = MemoryOptimizedEmbeddingSystem()
        self.skill_extractor = SkillExtractor()
        self.enhanced_matcher = EnhancedCandidateMatchingSystem()  # NEW: Advanced matching system
        self.index = None
        self.candidates: Dict[str, CandidateProfile] = {}
        self.candidate_embeddings: Optional[np.ndarray] = None
        self.dimension = self.embedding_service.models["general"].get_sentence_embedding_dimension()
        self._load_index()
        self.total_searches = 0
        self.total_search_time = 0
        self.cache_hits = 0
        self.latest_bias_assessment = None
        # Performance optimization: Prevent frequent rebuilds
        self.last_rebuild_time = 0
        self.rebuild_cooldown = 60  # Don't rebuild more than once per minute
        self.rebuild_threshold = 0.1  # Only rebuild if mismatch is >10%
        logger.info(f"EnhancedRecruitmentSearchSystem initialized with advanced matching and dimension {self.dimension}.")

    def _load_index(self):
        """Load FAISS index and candidate data if they exist."""
        faiss_path = f"{self.index_path}.faiss"
        candidates_path = f"{self.index_path}_candidates.pkl"
        embeddings_path = f"{self.index_path}_embeddings.npy"

        if os.path.exists(faiss_path) and os.path.exists(candidates_path) and os.path.exists(embeddings_path):
            try:
                self.index = faiss.read_index(faiss_path)
                with open(candidates_path, 'rb') as f:
                    self.candidates = pickle.load(f)
                self.candidate_embeddings = np.load(embeddings_path)
                logger.info(f"Loaded FAISS index from {faiss_path} with {self.index.ntotal} vectors.")
                logger.info(f"Loaded {len(self.candidates)} candidates and embeddings.")
            except Exception as e:
                logger.error(f"Failed to load index components: {e}. Starting fresh.")
                self.index = None
                self.candidates = {}
                self.candidate_embeddings = None
        else:
            logger.info("No existing FAISS index or data found. Will create new ones.")

    def _save_index(self):
        """Save FAISS index and candidate data - FIXED VERSION"""
        try:
            # Create directory properly
            index_dir = os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else "."
            if index_dir != "." and not os.path.exists(index_dir):
                os.makedirs(index_dir, exist_ok=True)
                logger.info(f"Created directory: {index_dir}")

            # Save FAISS index
            faiss_path = f"{self.index_path}.faiss"
            faiss.write_index(self.index, faiss_path)
            logger.info(f"FAISS index saved to {faiss_path}")

            # Save candidate data
            candidates_path = f"{self.index_path}_candidates.pkl"
            with open(candidates_path, 'wb') as f:
                pickle.dump(self.candidates, f)
            logger.info(f"Candidates saved to {candidates_path}")

            # Save embeddings
            embeddings_path = f"{self.index_path}_embeddings.npy"
            np.save(embeddings_path, self.candidate_embeddings)
            logger.info(f"Embeddings saved to {embeddings_path}")

            logger.info("Index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def cleanup_candidate_data(self, raw_candidates: List[Dict]) -> List[Dict]:
        """Clean up candidate data to prevent processing errors"""
        cleaned_candidates = []
        
        for i, candidate in enumerate(raw_candidates):
            try:
                cleaned = {}
                
                # Clean each field
                for key, value in candidate.items():
                    if value is None:
                        cleaned[key] = ""
                    elif isinstance(value, list):
                        # Convert list elements to strings
                        cleaned[key] = [str(item) for item in value if item is not None]
                    elif isinstance(value, dict):
                        # Convert dict to string representation
                        cleaned[key] = str(value)
                    else:
                        cleaned[key] = value
                
                cleaned_candidates.append(cleaned)
                
            except Exception as e:
                logger.error(f"Error cleaning candidate data at index {i}: {e}")
                continue
        
        logger.info(f"Cleaned {len(cleaned_candidates)} out of {len(raw_candidates)} candidates")
        return cleaned_candidates

    def _create_candidate_profile(self, raw_candidate: Dict) -> CandidateProfile:
        """Create enhanced candidate profile from raw data - FIXED VERSION"""
        
        def safe_str_conversion(value, default: str = "") -> str:
            """Safely convert any value to string"""
            if value is None:
                return default
            elif isinstance(value, list):
                clean_items = [str(x) for x in value if x is not None]
                return ', '.join(clean_items) if clean_items else default
            elif isinstance(value, str):
                return value
            else:
                return str(value)
        
        def safe_list_conversion(value, default: List[str] = None) -> List[str]:
            """Safely convert any value to list of strings"""
            if default is None:
                default = []
            
            if value is None:
                return default
            elif isinstance(value, list):
                return [str(x) for x in value if x is not None]
            elif isinstance(value, str):
                if ',' in value:
                    return [s.strip() for s in value.split(',') if s.strip()]
                elif value.strip():
                    return [value.strip()]
                else:
                    return default
            else:
                return [str(value)] if value else default
        
        try:
            # Extract and safely convert resume text
            resume_text = safe_str_conversion(
                raw_candidate.get('resume_text') or raw_candidate.get('ResumeText', '')
            )

            contact_sources: List[Dict[str, Any]] = []
            for key in ('contact', 'contactInfo', 'contact_info', 'contactDetails', 'contact_details'):
                value = raw_candidate.get(key)
                if isinstance(value, dict):
                    contact_sources.append(value)

            contact_bundle: Dict[str, Any] = {}
            for source in contact_sources:
                for key, value in source.items():
                    if value is not None and key not in contact_bundle:
                        contact_bundle[key] = value
            
            # Extract and safely convert skills
            skills_raw = raw_candidate.get('skills') or raw_candidate.get('Skills', [])
            skills = safe_list_conversion(skills_raw)
            
            # Extract detailed skills with context if we have resume text
            skill_details = {}
            skill_proficiency = {}
            if resume_text:
                try:
                    skill_details = self.skill_extractor.extract_skills_with_context(resume_text)
                    skill_proficiency = {skill: details["proficiency"] for skill, details in skill_details.items()}
                except Exception as e:
                    logger.warning(f"Skill extraction failed for candidate: {e}")
            
            # Safely convert experience
            experience_raw = raw_candidate.get('total_experience_years') or raw_candidate.get('Experience', 0)
            experience_years = self._safe_convert_experience(experience_raw)
            
            # Determine seniority
            seniority = self._determine_seniority_from_text(resume_text, experience_years)
            
            # Extract other fields safely
            education = safe_str_conversion(
                raw_candidate.get('education') or raw_candidate.get('Education', '')
            )
            
            certifications = safe_list_conversion(
                raw_candidate.get('certifications') or raw_candidate.get('Certifications', [])
            )
            
            # Extract industries and roles from resume text
            industries = self._extract_industries(resume_text)
            previous_roles = self._extract_roles(resume_text)
            
            # Handle email - ensure we have one
            email = safe_str_conversion(raw_candidate.get('email') or raw_candidate.get('Email'))
            if not email:
                email_candidates = [
                    raw_candidate.get('primary_email'),
                    raw_candidate.get('primaryEmail'),
                    raw_candidate.get('contact_email'),
                    raw_candidate.get('contactEmail'),
                    contact_bundle.get('email'),
                    contact_bundle.get('primaryEmail'),
                    contact_bundle.get('emailAddress'),
                ]
                for candidate_email in email_candidates:
                    email = safe_str_conversion(candidate_email)
                    if email:
                        break
            if not email:
                for key in ('emails', 'emailAddresses', 'email_addresses'):
                    email_list = raw_candidate.get(key) or contact_bundle.get(key)
                    if isinstance(email_list, list):
                        for entry in email_list:
                            email = safe_str_conversion(entry)
                            if email:
                                break
                    if email:
                        break
            if not email:
                candidate_id_hash = hashlib.sha256(str(raw_candidate).encode()).hexdigest()[:8]
                email = f"candidate_{candidate_id_hash}@unknown.com"
            
            # Handle other contact info
            full_name = safe_str_conversion(
                raw_candidate.get('full_name') or raw_candidate.get('FullName'), 
                'Unknown Candidate'
            )
            
            phone = safe_str_conversion(raw_candidate.get('phone') or raw_candidate.get('Phone'))
            if not phone:
                phone_candidates = [
                    contact_bundle.get('phone'),
                    contact_bundle.get('mobile'),
                    contact_bundle.get('contactNumber'),
                ]
                for candidate_phone in phone_candidates:
                    phone = safe_str_conversion(candidate_phone)
                    if phone:
                        break
            if not phone:
                phone = ''
            location = safe_str_conversion(
                raw_candidate.get('location') or raw_candidate.get('Location', '')
            )
            source_url = safe_str_conversion(
                raw_candidate.get('sourceURL') or raw_candidate.get('SourceURL', '')
            )
            
            # Create and return the profile
            return CandidateProfile(
                email=email,
                full_name=full_name,
                phone=phone,
                resume_text=resume_text,
                skills=skills,
                experience_years=experience_years,
                education=education,
                certifications=certifications,
                previous_roles=previous_roles,
                industries=industries,
                location=location,
                source_url=source_url,
                skill_proficiency=skill_proficiency,
                seniority_level=seniority
            )
            
        except Exception as e:
            logger.error(f"Error creating candidate profile for {raw_candidate.get('email', 'unknown')}: {e}")
            
            # Create a minimal fallback profile
            email = raw_candidate.get('email') or raw_candidate.get('Email') or f"error_candidate_{int(time.time())}@temp.com"
            return CandidateProfile(
                email=email,
                full_name=safe_str_conversion(raw_candidate.get('full_name') or raw_candidate.get('FullName'), 'Error Processing Candidate'),
                phone='',
                resume_text='Error processing resume data',
                skills=[],
                experience_years=0,
                education='',
                certifications=[],
                previous_roles=[],
                industries=[],
                location='',
                source_url='',
                skill_proficiency={},
                seniority_level='Unknown'
            )

    def _safe_convert_experience(self, experience_raw: Any) -> int:
        """Safely convert experience to integer"""
        try:
            if experience_raw is None:
                return 0

            if isinstance(experience_raw, (int, float)):
                return int(experience_raw)

            if isinstance(experience_raw, str):
                if not experience_raw.strip():
                    return 0
                numbers = re.findall(r'\d+\.?\d*', str(experience_raw))
                if numbers:
                    return int(float(numbers[0]))
                else:
                    return 0

            return int(float(str(experience_raw)))
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Could not convert experience '{experience_raw}': {e}. Returning 0.")
            return 0

    def _determine_seniority_from_text(self, text: str, experience_years: int) -> str:
        """Determines seniority level based on text and experience."""
        text_lower = text.lower()
        if "senior" in text_lower or "lead" in text_lower or "principal" in text_lower or experience_years >= 7:
            return "Senior"
        elif "mid-level" in text_lower or "experienced" in text_lower or (experience_years >= 3 and experience_years < 7):
            return "Mid-Level"
        elif "junior" in text_lower or "entry-level" in text_lower or experience_years < 3:
            return "Junior"
        return "Not Specified"

    def _extract_industries(self, text: str) -> List[str]:
        """Extract industries from resume text."""
        industries = []
        text_lower = text.lower()
        if "healthcare" in text_lower or "medical" in text_lower:
            industries.append("Healthcare")
        if "finance" in text_lower or "banking" in text_lower:
            industries.append("Finance")
        if "software" in text_lower or "tech" in text_lower:
            industries.append("Software/IT")
        return industries[:3]

    def _extract_roles(self, text: str) -> List[str]:
        """Extract previous roles/titles from text."""
        roles = []
        matches = re.findall(r'(software engineer|data scientist|project manager|product manager|devops engineer)', text, re.IGNORECASE)
        roles.extend(list(set(matches)))
        return roles[:5]

    def index_candidates(self, raw_candidates: List[Dict]):
        """Process raw candidate data and build/update the FAISS index."""
        if not raw_candidates:
            logger.warning("No candidates provided for indexing.")
            return

        start_time = time.time()
        
        # Clean up the data first
        cleaned_candidates = self.cleanup_candidate_data(raw_candidates)
        
        new_candidates = {}
        combined_texts = []
        valid_indices = []

        # Step 1: Create profiles and collect texts for progressive batch encoding
        # For very large datasets, process in chunks to avoid memory issues
        large_dataset = len(cleaned_candidates) > 10000
        chunk_size = 5000 if large_dataset else len(cleaned_candidates)
        
        logger.info(f"Processing {len(cleaned_candidates)} candidates in chunks of {chunk_size}...")
        
        all_embeddings_chunks = []
        processed_candidates = {}
        candidate_to_embedding_idx = {}  # Track which embedding index corresponds to which candidate
        total_processed = 0
        embedding_global_idx = 0
        
        for chunk_start in range(0, len(cleaned_candidates), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(cleaned_candidates))
            chunk = cleaned_candidates[chunk_start:chunk_end]
            
            chunk_texts = []
            chunk_profiles = {}
            chunk_embedding_indices = []
            
            for idx, raw_candidate in enumerate(chunk):
                try:
                    profile = self._create_candidate_profile(raw_candidate)
                    email = profile.email
                    
                    # Only process if candidate is new (prevent duplicates)
                    if email not in self.candidates:
                        chunk_profiles[email] = profile
                        chunk_texts.append(profile.combined_text)
                        chunk_embedding_indices.append(embedding_global_idx)
                        candidate_to_embedding_idx[email] = embedding_global_idx
                        embedding_global_idx += 1
                    else:
                        # Update existing candidate but don't add to index
                        self.candidates[email] = profile
                except Exception as e:
                    logger.error(f"Error processing candidate {raw_candidate.get('email', 'unknown')}: {e}")
                    continue
            
            if not chunk_texts:
                continue
            
            # Batch encode this chunk
            logger.info(f"Encoding chunk {chunk_start//chunk_size + 1}/{(len(cleaned_candidates)-1)//chunk_size + 1} ({len(chunk_texts)} new candidates)...")
            chunk_embeddings = self.embedding_service.encode_batch(chunk_texts, model_type="technical")
            
            if chunk_embeddings.shape[0] > 0:
                all_embeddings_chunks.append(chunk_embeddings)
                processed_candidates.update(chunk_profiles)
                total_processed += len(chunk_profiles)
        
        if not all_embeddings_chunks:
            logger.warning("No valid candidate profiles created.")
            return
        
        # Combine all embeddings - these are already filtered for new candidates only
        logger.info("Combining embeddings from all chunks...")
        new_embeddings_array = np.vstack(all_embeddings_chunks)
        new_candidates = processed_candidates
        
        if new_embeddings_array.shape[0] == 0:
            logger.warning("No valid candidate embeddings generated. Index not built/updated.")
            return

        # Initialize FAISS index if it doesn't exist
        if self.index is None:
            self.dimension = new_embeddings_array.shape[1]
            
            # Use HNSW index for large datasets (200K+ candidates) - MUCH FASTER!
            total_candidates = len(new_candidates) + len(self.candidates) if self.candidates else len(new_candidates)
            
            if total_candidates > 50000:
                # Use HNSW for large datasets - approximate search but MUCH faster
                logger.info(f"Large dataset detected ({total_candidates} candidates). Using HNSW index for fast approximate search.")
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors for good accuracy
                self.index.hnsw.efConstruction = 200  # High quality
                self.index.hnsw.efSearch = 128  # Good balance between speed and accuracy
                logger.info(f"HNSW FAISS index created with dimension {self.dimension} (optimized for 200K+ candidates).")
            else:
                # Use flat index for smaller datasets - exact search
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"FAISS index created with dimension {self.dimension} (flat index for small datasets).")

        # Add vectors to the index in batches for large datasets
        add_batch_size = 10000  # Add 10K at a time to avoid memory issues
        total_vectors = new_embeddings_array.shape[0]
        
        if total_vectors > add_batch_size and hasattr(self.index, 'add'):
            logger.info(f"Adding {total_vectors} new vectors in batches of {add_batch_size}...")
            for i in range(0, total_vectors, add_batch_size):
                batch = new_embeddings_array[i:i+add_batch_size]
                self.index.add(batch)
                logger.info(f"Added batch {i//add_batch_size + 1}/{(total_vectors-1)//add_batch_size + 1}")
        else:
            self.index.add(new_embeddings_array)
        
        duplicate_count = len(cleaned_candidates) - len(new_candidates)
        if duplicate_count > 0:
            logger.info(f"Added {new_embeddings_array.shape[0]} new embeddings to FAISS index (skipped {duplicate_count} duplicates).")
        else:
            logger.info(f"Added {new_embeddings_array.shape[0]} new embeddings to FAISS index.")

        # Update candidate data map - only add new candidates
        self.candidates.update(new_candidates)
        
        # Update candidate embeddings array
        if self.candidate_embeddings is None:
            self.candidate_embeddings = new_embeddings_array
        else:
            self.candidate_embeddings = np.vstack([self.candidate_embeddings, new_embeddings_array])

        logger.info(f"Total candidates in system: {len(self.candidates)}")
        logger.info(f"Total vectors in FAISS index: {self.index.ntotal}")
        
        # Warn if mismatch detected
        if self.index.ntotal != len(self.candidates):
            mismatch = abs(self.index.ntotal - len(self.candidates))
            logger.warning(
                f"Index mismatch detected: {self.index.ntotal} vectors vs {len(self.candidates)} candidates "
                f"(difference: {mismatch}). This may indicate duplicate vectors in the index."
            )

        self._save_index()
        end_time = time.time()
        logger.info(f"Indexing completed in {end_time - start_time:.2f} seconds.")
        logger.info(f"Successfully processed {total_processed} out of {len(raw_candidates)} candidates.")

    def _rebuild_index_safely(self):
        """Rebuild the FAISS index to ensure consistency with candidate data"""
        try:
            rebuild_start = time.time()
            logger.info("Rebuilding FAISS index for consistency...")
            
            # Create new embeddings for all candidates using BATCH ENCODING
            candidate_emails = list(self.candidates.keys())
            combined_texts = []
            
            for email in candidate_emails:
                candidate = self.candidates[email]
                combined_texts.append(candidate.combined_text)
            
            if not combined_texts:
                logger.warning("No embeddings to rebuild index with")
                return
            
            # Batch encode all candidates at once - MUCH FASTER!
            logger.info(f"Batch encoding {len(combined_texts)} candidates for index rebuild...")
            new_embeddings_array = self.embedding_service.encode_batch(combined_texts, model_type="technical")
            
            if new_embeddings_array.shape[0] == 0:
                logger.warning("No embeddings generated during index rebuild")
                return
            
            new_embeddings_array = new_embeddings_array.astype('float32')
            self.dimension = new_embeddings_array.shape[1]
            
            # Use appropriate index type based on dataset size
            total_candidates = len(candidate_emails)
            if total_candidates > 50000:
                # Use HNSW for large datasets
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 128
                logger.info("Using HNSW index for large dataset during rebuild")
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info("Using flat index for small dataset during rebuild")
            
            # Add in batches for large datasets
            batch_size = 10000
            if new_embeddings_array.shape[0] > batch_size:
                for i in range(0, new_embeddings_array.shape[0], batch_size):
                    batch = new_embeddings_array[i:i+batch_size]
                    self.index.add(batch)
            else:
                self.index.add(new_embeddings_array)
            
            self.candidate_embeddings = new_embeddings_array
            
            rebuild_time = time.time() - rebuild_start
            logger.info(
                f"Index rebuilt successfully with {self.index.ntotal} vectors for {len(candidate_emails)} candidates "
                f"in {rebuild_time:.2f} seconds"
            )
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            # Fallback: create empty index
            self.index = None
            self.candidate_embeddings = None

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Performs an ADVANCED semantic search for candidates based on a complex job query.
        Returns enhanced results with detailed scoring using the new matching system.
        """
        start_time = time.time()
        self.total_searches += 1

        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is not built or empty. Cannot perform search.")
            return []

        try:
            logger.info(f"Starting ADVANCED search for: '{query[:100]}...'")
            
            # Step 1: Encode query for semantic search
            query_embedding = self.embedding_service.encode(query, model_type="general")
            query_embedding = query_embedding.reshape(1, -1).astype('float32')

            # Step 2: Perform FAISS search (optimized for large datasets)
            # For large datasets, get more candidates for better ranking
            total_candidates = self.index.ntotal
            if total_candidates > 50000:
                search_k = min(total_candidates, 2000)  # More candidates for large datasets
            else:
                search_k = min(total_candidates, 1000)  # Standard for smaller datasets
            
            D, I = self.index.search(query_embedding, search_k)

            # Step 3: Collect candidates for advanced matching
            candidates_for_matching = []
            candidate_emails = list(self.candidates.keys())
            
            # Ensure FAISS index and candidates are in sync (with performance optimization)
            current_time = time.time()
            index_count = self.index.ntotal if self.index else 0
            candidate_count = len(candidate_emails)
            
            # Handle index mismatch gracefully - don't block search
            if index_count > 0 and candidate_count > 0:
                mismatch_ratio = abs(index_count - candidate_count) / max(index_count, candidate_count)
                time_since_rebuild = current_time - self.last_rebuild_time
                
                # Only rebuild if significant mismatch AND cooldown passed
                should_rebuild = (
                    mismatch_ratio > self.rebuild_threshold and  # Significant mismatch
                    time_since_rebuild > self.rebuild_cooldown  # Cooldown period passed
                )
                
                if should_rebuild:
                    logger.warning(
                        f"FAISS index mismatch detected: {index_count} vectors vs {candidate_count} candidates "
                        f"(mismatch: {mismatch_ratio:.1%}). Scheduling rebuild in background..."
                    )
                    # Schedule rebuild in background - don't block current search
                    try:
                        import threading
                        def background_rebuild():
                            try:
                                self._rebuild_index_safely()
                                self.last_rebuild_time = time.time()
                            except Exception as e:
                                logger.error(f"Background rebuild failed: {e}")
                        
                        rebuild_thread = threading.Thread(target=background_rebuild, daemon=True)
                        rebuild_thread.start()
                        self.last_rebuild_time = current_time  # Mark as scheduled
                    except Exception as e:
                        logger.error(f"Failed to start background rebuild: {e}")
                        # Fallback: continue with existing index
                
                # Use existing index even if there's a mismatch (handle bounds in loop)
                # Adjust search_k to account for potential mismatch
                if index_count > candidate_count:
                    # Index has more vectors - limit search to candidate count
                    search_k = min(search_k, candidate_count)
                    logger.debug(f"Limiting search to {search_k} results due to index mismatch")
                elif mismatch_ratio > 0.01:  # Small mismatch, log but don't rebuild
                    logger.debug(
                        f"FAISS index has minor mismatch: {index_count} vectors vs {candidate_count} candidates "
                        f"(mismatch: {mismatch_ratio:.1%}). Using existing index (cooldown: {max(0, self.rebuild_cooldown - time_since_rebuild):.1f}s remaining)"
                    )
            
            for i, (i_idx, distance) in enumerate(zip(I[0], D[0])):
                if i_idx == -1:
                    continue
                
                # Fix: Handle index bounds properly - skip if index is out of bounds
                if i_idx >= len(candidate_emails):
                    # Index has more vectors than candidates - skip this result
                    continue

                try:
                    candidate_email = candidate_emails[i_idx]
                    candidate_profile = self.candidates.get(candidate_email)

                    if candidate_profile:
                        # Convert candidate profile to dict for matching system
                        candidate_dict = candidate_profile.to_dict()
                        candidates_for_matching.append((candidate_profile, candidate_dict, distance))
                        logger.debug(f"Added candidate {candidate_email} for matching (distance: {distance:.3f})")
                    else:
                        logger.warning(f"Candidate profile not found for email: {candidate_email}")
                        
                except Exception as e:
                    logger.error(f"Error processing candidate at index {i_idx}: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    continue

            if not candidates_for_matching:
                logger.warning(f"No valid candidates found for matching. FAISS returned {len(I[0])} indices, candidate_emails has {len(candidate_emails)} entries")
                logger.warning(f"FAISS indices: {I[0][:10]}...")  # Show first 10 indices
                logger.warning(f"FAISS distances: {D[0][:10]}...")  # Show first 10 distances
                
                # Try fallback: return some candidates anyway for debugging
                if len(self.candidates) > 0:
                    logger.info("Attempting fallback with first few candidates...")
                    fallback_candidates = list(self.candidates.values())[:min(5, len(self.candidates))]
                    for candidate_profile in fallback_candidates:
                        candidate_dict = candidate_profile.to_dict()
                        candidates_for_matching.append((candidate_profile, candidate_dict, 0.5))  # Default distance
                
                if not candidates_for_matching:
                    return []

            logger.info(f"Processing {len(candidates_for_matching)} candidates with advanced matching")

            # Step 4: Apply ADVANCED MATCHING SYSTEM
            enhanced_results = []
            for candidate_profile, candidate_dict, faiss_distance in candidates_for_matching:
                try:
                    # Use the enhanced matching system
                    match_score = self.enhanced_matcher.match_candidate_to_job(candidate_dict, query)
                    
                    # Combine FAISS semantic similarity with advanced matching
                    faiss_normalized = ((faiss_distance + 1) / 2) * 100  # Convert to 0-100
                    
                    # Weight the scores: 60% advanced matching, 40% semantic similarity
                    final_score = (match_score.overall_score * 0.6) + (faiss_normalized * 0.4)
                    
                    # Create enhanced result with detailed scoring
                    enhanced_result = {
                        'email': candidate_profile.email,
                        'full_name': candidate_profile.full_name,
                        'phone': candidate_profile.phone,
                        'skills': candidate_profile.skills,
                        'experience_years': candidate_profile.experience_years,
                        'education': candidate_profile.education,
                        'certifications': candidate_profile.certifications,
                        'location': candidate_profile.location,
                        'source_url': candidate_profile.source_url,
                        'seniority_level': candidate_profile.seniority_level,
                        
                        # ENHANCED SCORING DETAILS
                        'overall_score': final_score,
                        'advanced_match_score': match_score.overall_score,
                        'semantic_similarity_score': faiss_normalized,
                        'skill_match_score': match_score.technical_skills_score,
                        'experience_relevance': match_score.experience_score,
                        'seniority_match': match_score.seniority_score,
                        'education_match': match_score.education_score,
                        'soft_skills_match': match_score.soft_skills_score,
                        'location_match': match_score.location_score,
                        'confidence': match_score.confidence,
                        'grade': self._get_grade(int(final_score)),
                        'match_explanation': match_score.match_explanation,
                        'missing_requirements': match_score.missing_requirements,
                        'strength_areas': match_score.strength_areas,
                        
                        # Additional metadata
                        'processing_timestamp': time.time(),
                        'matching_algorithm': 'advanced_v2.0'
                    }
                    
                    enhanced_results.append(enhanced_result)
                    
                except Exception as e:
                    logger.error(f"Error in advanced matching for {candidate_profile.email}: {e}")
                    # Fallback to basic result
                    fallback_result = self._format_enhanced_result(candidate_profile, faiss_distance, query, len(enhanced_results) + 1)
                    enhanced_results.append(fallback_result)
                    continue

            # Step 5: Sort by final score and return top results
            enhanced_results.sort(key=lambda x: x['overall_score'], reverse=True)
            final_results = enhanced_results[:top_k]

            end_time = time.time()
            duration = end_time - start_time
            self.total_search_time += duration
            
            logger.info(f"ADVANCED search completed in {duration:.4f} seconds")
            logger.info(f"Found {len(final_results)} highly matched candidates")
            
            # Log top result details for debugging
            if final_results:
                top_result = final_results[0]
                logger.info(f"Top match: {top_result['full_name']} - Score: {top_result['overall_score']:.1f}% - Skills: {top_result['skill_match_score']:.1f}%")

            return final_results
            
        except Exception as e:
            logger.error(f"Advanced search error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def _format_enhanced_result(self, candidate: CandidateProfile, score: float, query: str, rank: int) -> Dict[str, Any]:
        """Format candidate result with enhanced scoring details."""
        # Calculate detailed scores
        skill_match_score = self._calculate_skill_match_score(candidate.skills, query)
        experience_relevance = self._calculate_experience_relevance(candidate.experience_years, query)
        education_match = self._calculate_education_match(candidate.education, query)
        
        # Calculate overall score based on multiple factors
        overall_score = (
            skill_match_score * 0.4 +  # Skills are most important
            experience_relevance * 0.3 +  # Experience is second
            education_match * 0.2 +  # Education is third
            score * 0.1  # FAISS similarity is least important
        )
        
        # Ensure score is within reasonable bounds and round to 1 decimal place
        overall_score = max(0, min(100, overall_score))
        overall_score = round(overall_score, 1)
        
        # Overall confidence calculation
        confidence = min(95, max(60, overall_score + (rank * -1)))  # Decrease confidence by rank
        
        # Generate match explanation
        match_explanation = self._generate_match_explanation(candidate, query, skill_match_score)
        
        # Determine grade
        grade = self._get_grade(int(overall_score))
        
        # Determine category based on candidate data
        category = self._detect_candidate_category(candidate)

        return {
            'email': candidate.email,
            'full_name': candidate.full_name,
            'phone': candidate.phone,
            'skills': candidate.skills,
            'experience_years': candidate.experience_years,
            'education': candidate.education,
            'certifications': candidate.certifications,
            'location': candidate.location,
            'source_url': candidate.source_url,
            'seniority_level': candidate.seniority_level,
            
            # Enhanced scoring
            'overall_score': overall_score,
            'skill_match_score': skill_match_score,
            'experience_relevance': experience_relevance,
            'education_match': education_match,
            'confidence': confidence,
            'grade': grade,
            'match_explanation': match_explanation,
            'rank': rank,
            'category': category  # Add category field
        }
    
    def _detect_candidate_category(self, candidate: CandidateProfile) -> str:
        """Detect candidate category based on their profile data."""
        # Combine all text data for analysis
        combined_text = f"{candidate.education} {' '.join(candidate.skills)} {' '.join(candidate.certifications)} {candidate.resume_text}".lower()
        
        # Healthcare indicators (more specific, check first)
        healthcare_indicators = [
            'lpn', 'rn', 'nursing', 'nurse', 'patient care', 'medical', 'healthcare', 'clinical', 
            'hospital', 'licensed practical nurse', 'registered nurse', 'practical nursing',
            'patient', 'treatment', 'medication', 'vitals', 'charting', 'wound care',
            'infection control', 'home healthcare', 'visiting nurse', 'nursing school',
            'clinical rotation', 'patient assessment', 'care plan', 'medical records',
            'healthcare provider', 'medical assistant', 'phlebotomy', 'radiology',
            'laboratory', 'pharmacy', 'therapist', 'physician', 'doctor'
        ]
        
        # IT/Tech indicators
        tech_indicators = [
            'python', 'java', 'javascript', 'react', 'angular', 'node', 'sql', 'aws', 'docker',
            'kubernetes', 'devops', 'git', 'api', 'frontend', 'backend', 'fullstack', 'database',
            'cloud', 'machine learning', 'ai', 'data science', 'cybersecurity', 'network',
            'system administrator', 'software engineer', 'developer', 'programmer', 'coding',
            'programming', 'software', 'web development', 'mobile development'
        ]
        
        # Count matches
        healthcare_count = sum(1 for indicator in healthcare_indicators if indicator in combined_text)
        tech_count = sum(1 for indicator in tech_indicators if indicator in combined_text)
        
        # Determine category
        if healthcare_count > tech_count:
            return "Healthcare"
        elif tech_count > healthcare_count:
            return "IT/Tech"
        else:
            return "General"

    def _calculate_skill_match_score(self, candidate_skills: List[str], query: str) -> float:
        """Calculate skill matching score."""
        if not candidate_skills:
            return 0.0
        
        query_lower = query.lower()
        query_skills = re.findall(r'\b(?:python|java|javascript|react|node|aws|sql|machine learning|data science)\b', query_lower)
        
        if not query_skills:
            return 50.0  # Default if no specific skills detected
        
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        matches = sum(1 for skill in query_skills if any(skill in cs for cs in candidate_skills_lower))
        
        return min(95, (matches / len(query_skills)) * 100)

    def _calculate_experience_relevance(self, experience_years: int, query: str) -> float:
        """Calculate experience relevance score."""
        # Extract experience requirements from query
        exp_patterns = re.findall(r'(\d+)\+?\s*years?', query.lower())
        
        if not exp_patterns:
            return 75.0  # Default if no experience mentioned
        
        required_exp = int(exp_patterns[0])
        
        if experience_years >= required_exp:
            return min(95, 80 + (experience_years - required_exp) * 2)
        else:
            # Penalty for insufficient experience
            return max(20, 80 - (required_exp - experience_years) * 10)

    def _calculate_education_match(self, education: str, query: str) -> float:
        """Calculate education matching score."""
        if not education:
            return 60.0  # Default for missing education
        
        education_lower = education.lower()
        query_lower = query.lower()
        
        # Check for education keywords in query
        if any(keyword in query_lower for keyword in ['degree', 'bachelor', 'master', 'phd', 'education']):
            if any(keyword in education_lower for keyword in ['bachelor', 'master', 'phd', 'degree']):
                return 85.0
            else:
                return 40.0
        
        return 70.0  # Default when education not specified in query

    def _generate_match_explanation(self, candidate: CandidateProfile, query: str, skill_score: float) -> str:
        """Generate human-readable match explanation."""
        explanations = []
        
        if skill_score > 80:
            explanations.append(f"Strong skill alignment with {len(candidate.skills)} relevant skills")
        elif skill_score > 60:
            explanations.append("Good skill match with some relevant expertise")
        else:
            explanations.append("Basic skill compatibility")
        
        if candidate.experience_years > 5:
            explanations.append(f"Experienced professional with {candidate.experience_years} years")
        elif candidate.experience_years > 2:
            explanations.append(f"Mid-level candidate with {candidate.experience_years} years experience")
        else:
            explanations.append("Entry-level or junior candidate")
        
        if candidate.seniority_level != "Not Specified":
            explanations.append(f"{candidate.seniority_level} level position fit")
        
        return ". ".join(explanations) + "."

    def _get_grade(self, score: int) -> str:
        """Convert score to grade."""
        if score >= 85:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 50:
            return 'C'
        else:
            return 'D'

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_search_time = self.total_search_time / max(self.total_searches, 1)
        cache_hit_rate = self.embedding_service.get_cache_hit_rate()
        
        return {
            'total_searches': self.total_searches,
            'average_search_time_ms': round(avg_search_time * 1000, 2),
            'total_candidates': len(self.candidates),
            'index_type': type(self.index).__name__ if self.index else 'None',
            'cache_hit_rate': cache_hit_rate,
            'embedding_dimension': self.dimension
        }

    def record_feedback(self, candidate_email: str, query: str, is_relevant: bool):
        """Record feedback for continuous learning."""
        try:
            feedback_data = {
                'candidate_email': candidate_email,
                'query': query,
                'is_relevant': is_relevant,
                'timestamp': time.time()
            }
            
            # Store feedback for future model improvements
            feedback_file = os.path.join(os.path.dirname(self.index_path) or ".", 'feedback.json')
            
            try:
                if os.path.exists(feedback_file):
                    with open(feedback_file, 'r') as f:
                        existing_feedback = json.load(f)
                else:
                    existing_feedback = []
            except Exception as e:
                logger.warning(f"Could not load existing feedback: {e}")
                existing_feedback = []
            
            existing_feedback.append(feedback_data)
            
            # Keep only the last 1000 feedback entries to prevent file from growing too large
            if len(existing_feedback) > 1000:
                existing_feedback = existing_feedback[-1000:]
            
            try:
                with open(feedback_file, 'w') as f:
                    json.dump(existing_feedback, f, indent=2)
                    
                logger.info(f"Recorded feedback for {candidate_email}: {'positive' if is_relevant else 'negative'}")
            except Exception as e:
                logger.error(f"Failed to save feedback to file: {e}")
                
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")


# Utility function for main.py integration
def create_ultra_fast_search_system(dynamodb_table) -> Tuple[Any, 'EnhancedRecruitmentSearchSystem']:
    """
    Creates and initializes the EnhancedRecruitmentSearchSystem.
    This is designed to be called once at application startup.
    """
    try:
        search_system = EnhancedRecruitmentSearchSystem(index_path="enhanced_search_index")

        # Get ALL candidates from DynamoDB with pagination
        try:
            raw_candidates = []
            last_evaluated_key = None
            page_count = 0
            
            logger.info("Starting to fetch candidates from DynamoDB with pagination...")
            
            while True:
                page_count += 1
                
                # Prepare scan parameters
                scan_params = {}
                if last_evaluated_key:
                    scan_params['ExclusiveStartKey'] = last_evaluated_key
                
                # Scan with pagination
                response = dynamodb_table.scan(**scan_params)
                
                # Add items from this page
                page_items = response.get('Items', [])
                raw_candidates.extend(page_items)
                
                logger.info(f"Page {page_count}: Fetched {len(page_items)} candidates (Total so far: {len(raw_candidates)})")
                
                # Check if there are more pages
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            logger.info(f"Successfully fetched {len(raw_candidates)} raw candidates from DynamoDB across {page_count} pages.")
        except Exception as e:
            logger.error(f"Failed to fetch candidates from DynamoDB: {e}")
            return None, None

        if not raw_candidates:
            logger.warning("No candidates found in DynamoDB. Search system will be empty.")
            return None, search_system

        # Preprocess candidates before indexing
        processed_candidates = preprocess_candidates_for_indexing(raw_candidates)

        if processed_candidates:
            search_system.index_candidates(processed_candidates)
            logger.info(f"Indexed {len(processed_candidates)} candidates into Enhanced Search System.")
        else:
            logger.warning("No valid candidates after preprocessing. Index not built.")

        return True, search_system

    except Exception as e:
        logger.error(f"Failed to create enhanced search system: {e}")
        return None, None


def preprocess_candidates_for_indexing(raw_candidates: List[Dict]) -> List[Dict]:
    """Preprocess candidates to fix common data issues - FIXED VERSION"""
    processed_candidates = []

    for i, candidate in enumerate(raw_candidates):
        try:
            # Create a copy to avoid modifying original
            processed_candidate = candidate.copy()

            # Fix email field
            if not processed_candidate.get('email') and not processed_candidate.get('Email'):
                candidate_id_hash = hashlib.sha256(str(candidate).encode()).hexdigest()[:8]
                processed_candidate['email'] = f"candidate_{i}_{candidate_id_hash}@temp.com"

            # Fix experience field
            experience_raw = processed_candidate.get('total_experience_years') or processed_candidate.get('Experience')
            processed_candidate['experience_years'] = _safe_convert_experience_static(experience_raw)

            # Ensure skills is a list
            skills = processed_candidate.get('skills') or processed_candidate.get('Skills', [])
            if isinstance(skills, str):
                processed_candidate['skills'] = [s.strip() for s in skills.split(',') if s.strip()]
            elif not isinstance(skills, list):
                processed_candidate['skills'] = []

            # Ensure resume text exists
            if not (processed_candidate.get('resume_text') or processed_candidate.get('ResumeText')):
                processed_candidate['resume_text'] = f"Skills: {', '.join(processed_candidate['skills'])}. Experience: {processed_candidate['experience_years']} years."

            processed_candidates.append(processed_candidate)

        except Exception as e:
            logger.warning(f"Error preprocessing candidate (index {i}): {e}. Skipping this candidate.")
            continue
    
    logger.info(f"Preprocessed {len(processed_candidates)}/{len(raw_candidates)} candidates.")
    return processed_candidates


def _safe_convert_experience_static(experience_raw):
    """Static version of experience conversion for preprocessing"""
    try:
        if experience_raw is None:
            return 0

        if isinstance(experience_raw, (int, float)):
            return int(experience_raw)

        if isinstance(experience_raw, str):
            if not experience_raw.strip():
                return 0
            numbers = re.findall(r'\d+\.?\d*', str(experience_raw))
            if numbers:
                return int(float(numbers[0]))
            else:
                return 0

        return int(float(str(experience_raw)))
    except (ValueError, TypeError, AttributeError):
        return 0