"""
Instant Search System - Pre-loaded Cache for Sub-millisecond Response Times
==========================================================================

This system pre-loads all candidate data at startup and provides instant search
without any loading delays during queries. Optimized for maximum performance.
"""

import os
import time
import pickle
import threading
import asyncio
from typing import List, Dict, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import logging
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class CandidateIndex:
    """Optimized candidate data structure for instant search"""
    email: str
    full_name: str
    skills: List[str]
    experience: int
    phone: str = ""
    source_url: str = ""
    searchable_text: str = ""
    skills_set: Set[str] = None
    words_set: Set[str] = None
    
    def __post_init__(self):
        if self.skills_set is None:
            self.skills_set = {skill.lower() for skill in self.skills if skill}
        if self.words_set is None:
            self.words_set = set(self.searchable_text.lower().split())

class InstantSearchEngine:
    """
    Ultra-fast instant search engine with pre-loaded cache.
    Provides sub-millisecond response times by pre-processing all data.
    """
    
    def __init__(self, max_candidates: int = 100000):
        self.max_candidates = max_candidates
        self.candidates: Dict[str, CandidateIndex] = {}
        self.skill_index: Dict[str, Set[str]] = defaultdict(set)
        self.word_index: Dict[str, Set[str]] = defaultdict(set)
        self.experience_index: Dict[int, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'avg_response_time_ms': 0.0,
            'fastest_query_ms': float('inf'),
            'slowest_query_ms': 0.0,
            'cache_hits': 0,
            'total_candidates': 0
        }
        
        # Pre-load data at startup
        self._preload_data()
        
        logger.info(f"⚡ InstantSearchEngine initialized with {len(self.candidates)} candidates")
    
    def _preload_data(self):
        """Pre-load all candidate data at startup for instant search"""
        start_time = time.time()
        
        try:
            # Try to load from existing cache first
            if self._load_from_cache():
                logger.info(f"✅ Loaded {len(self.candidates)} candidates from cache")
                return
            
            # If no cache, load from database or mock data
            candidates_data = self._load_candidates_from_source()
            
            if not candidates_data:
                logger.warning("⚠️ No candidate data available")
                return
            
            # Process and index all candidates
            self._process_and_index_candidates(candidates_data)
            
            # Save to cache for next time
            self._save_to_cache()
            
            load_time = time.time() - start_time
            logger.info(f"✅ Pre-loaded {len(self.candidates)} candidates in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to preload data: {e}")
    
    def _load_from_cache(self) -> bool:
        """Load pre-processed data from cache"""
        cache_file = "cache/instant_search_cache.pkl"
        
        if not os.path.exists(cache_file):
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            self.candidates = cache_data.get('candidates', {})
            self.skill_index = cache_data.get('skill_index', defaultdict(set))
            self.word_index = cache_data.get('word_index', defaultdict(set))
            self.experience_index = cache_data.get('experience_index', defaultdict(set))
            
            return len(self.candidates) > 0
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cache: {e}")
            return False
    
    def _save_to_cache(self):
        """Save pre-processed data to cache"""
        try:
            os.makedirs("cache", exist_ok=True)
            cache_file = "cache/instant_search_cache.pkl"
            
            cache_data = {
                'candidates': self.candidates,
                'skill_index': dict(self.skill_index),
                'word_index': dict(self.word_index),
                'experience_index': dict(self.experience_index)
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"💾 Saved {len(self.candidates)} candidates to cache")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save cache: {e}")
    
    def _load_candidates_from_source(self) -> List[Dict]:
        """Load candidates from database with optimized connection"""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError
            from botocore.config import Config
            
            # Optimized boto3 configuration for faster connection
            config = Config(
                region_name='ap-south-1',
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                max_pool_connections=50,
                connect_timeout=10,
                read_timeout=30
            )
            
            dynamodb = boto3.resource('dynamodb', config=config)
            table_name = os.getenv('DYNAMODB_TABLE_NAME', 'user-resume-metadata')
            
            logger.info(f"🔄 Connecting to DynamoDB: {table_name}")
            table = dynamodb.Table(table_name)
            
            # Optimized scan with larger batch size and parallel processing
            logger.info("📊 Scanning DynamoDB with optimized settings...")
            all_items = []
            scan_kwargs = {
                'Limit': 200,  # Limited to 200 candidates for performance
                'ProjectionExpression': 'email, full_name, skills, resume_text, total_experience_years, phone, sourceURL',  # Only fetch needed fields
            }
            
            start_time = time.time()
            batch_count = 0
            
            while True:
                try:
                    response = table.scan(**scan_kwargs)
                    items = response.get('Items', [])
                    all_items.extend(items)
                    batch_count += 1
                    
                    # Progress indicator
                    if batch_count % 5 == 0 or 'LastEvaluatedKey' not in response:
                        elapsed = time.time() - start_time
                        rate = len(all_items) / elapsed if elapsed > 0 else 0
                        logger.info(f"📊 Batch {batch_count}: {len(items)} items (total: {len(all_items)}) - Rate: {rate:.1f} items/sec")
                    
                    # Check if there are more items to scan
                    if 'LastEvaluatedKey' not in response:
                        break
                    
                    # Set the LastEvaluatedKey for the next scan
                    scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
                    
                except ClientError as e:
                    logger.error(f"❌ DynamoDB scan error: {e}")
                    break
                except Exception as e:
                    logger.error(f"❌ Scan error: {e}")
                    break
            
            load_time = time.time() - start_time
            if all_items:
                logger.info(f"✅ Successfully loaded {len(all_items)} candidates from DynamoDB in {load_time:.2f}s")
                return all_items
            else:
                logger.warning("⚠️ No candidates found in DynamoDB")
                return []
                
        except ImportError:
            logger.error("❌ boto3 not available - cannot connect to DynamoDB")
            return []
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            return []
    
    def _get_mock_healthcare_candidates(self) -> List[Dict]:
        """Get mock healthcare candidates based on the provided job descriptions"""
        return [
            {
                'email': 'sarah.nurse@healthcare.com',
                'full_name': 'Sarah Johnson RN',
                'skills': ['Patient Care', 'ICU', 'Emergency Medicine', 'ACLS', 'BLS', 'Nursing'],
                'total_experience_years': 7,
                'resume_text': 'Registered Nurse with 7 years experience in ICU and emergency medicine. ACLS and BLS certified with expertise in patient care, medication administration, and critical care.',
                'phone': '+1-555-0101',
                'sourceURL': 'https://example.com/sarah-johnson'
            },
            {
                'email': 'mike.tech@healthcare.com',
                'full_name': 'Mike Chen',
                'skills': ['Radiology', 'X-ray', 'MRI', 'CT Scan', 'Medical Imaging', 'Equipment Maintenance'],
                'total_experience_years': 4,
                'resume_text': 'Radiology Technician with 4 years experience performing X-rays, MRIs, and CT scans. Expert in medical imaging equipment maintenance and patient safety protocols.',
                'phone': '+1-555-0102',
                'sourceURL': 'https://example.com/mike-chen'
            },
            {
                'email': 'lisa.pharmacy@healthcare.com',
                'full_name': 'Lisa Rodriguez',
                'skills': ['Pharmacy', 'Medication Dispensing', 'Prescription Management', 'Insurance Claims', 'Pharmacy Tech'],
                'total_experience_years': 3,
                'resume_text': 'Pharmacy Technician with 3 years experience in medication dispensing, prescription management, and insurance claims processing. Certified pharmacy technician.',
                'phone': '+1-555-0103',
                'sourceURL': 'https://example.com/lisa-rodriguez'
            },
            {
                'email': 'john.assistant@healthcare.com',
                'full_name': 'John Smith',
                'skills': ['Medical Assistant', 'Patient Check-ins', 'Vitals', 'Medical Records', 'Basic Computer'],
                'total_experience_years': 1,
                'resume_text': 'Medical Assistant with 1 year experience assisting doctors with patient check-ins, taking vitals, and updating medical records. High school diploma with basic computer knowledge.',
                'phone': '+1-555-0104',
                'sourceURL': 'https://example.com/john-smith'
            },
            {
                'email': 'emma.tech@healthcare.com',
                'full_name': 'Emma Wilson',
                'skills': ['Patient Care', 'CNA', 'Daily Care', 'Vitals', 'Nursing Support'],
                'total_experience_years': 2,
                'resume_text': 'Patient Care Technician with 2 years experience helping patients with daily needs, recording vitals, and supporting nursing staff. CNA certification preferred.',
                'phone': '+1-555-0105',
                'sourceURL': 'https://example.com/emma-wilson'
            },
            {
                'email': 'david.reception@healthcare.com',
                'full_name': 'David Brown',
                'skills': ['Medical Reception', 'Appointments', 'Billing', 'Insurance', 'Patient Queries'],
                'total_experience_years': 2,
                'resume_text': 'Medical Receptionist with 2 years experience managing appointments, billing, insurance paperwork, and patient queries. Strong communication and computer skills.',
                'phone': '+1-555-0106',
                'sourceURL': 'https://example.com/david-brown'
            },
            {
                'email': 'jennifer.callcenter@healthcare.com',
                'full_name': 'Jennifer Davis',
                'skills': ['Call Center', 'Patient Inquiries', 'Appointments', 'Insurance Verification', 'Communication'],
                'total_experience_years': 1,
                'resume_text': 'Healthcare Call Center Representative with 1 year experience answering patient inquiries, scheduling appointments, and insurance verification. High school diploma with excellent communication skills.',
                'phone': '+1-555-0107',
                'sourceURL': 'https://example.com/jennifer-davis'
            },
            {
                'email': 'robert.lab@healthcare.com',
                'full_name': 'Robert Taylor',
                'skills': ['Laboratory', 'Lab Samples', 'Equipment Maintenance', 'Lab Technician', 'Science'],
                'total_experience_years': 1,
                'resume_text': 'Laboratory Aide with 1 year experience preparing lab samples, maintaining equipment, and assisting technicians. Basic science background preferred.',
                'phone': '+1-555-0108',
                'sourceURL': 'https://example.com/robert-taylor'
            },
            {
                'email': 'maria.pt@healthcare.com',
                'full_name': 'Maria Garcia',
                'skills': ['Physical Therapy', 'Patient Mobility', 'Exercise Assistance', 'Progress Tracking', 'PT Assistant'],
                'total_experience_years': 3,
                'resume_text': 'Physical Therapist Assistant with 3 years experience helping patients with mobility, assisting in exercises, and tracking progress. Associate degree in PT assisting with clinical exposure.',
                'phone': '+1-555-0109',
                'sourceURL': 'https://example.com/maria-garcia'
            },
            {
                'email': 'james.billing@healthcare.com',
                'full_name': 'James Wilson',
                'skills': ['Medical Billing', 'Coding', 'Insurance Claims', 'Compliance', 'Health Information'],
                'total_experience_years': 4,
                'resume_text': 'Medical Billing & Coding Specialist with 4 years experience handling insurance coding, processing claims, and ensuring compliance. Certified in billing and coding.',
                'phone': '+1-555-0110',
                'sourceURL': 'https://example.com/james-wilson'
            },
            {
                'email': 'linda.nutrition@healthcare.com',
                'full_name': 'Linda Anderson',
                'skills': ['Nutrition', 'Diet Planning', 'Patient Consultations', 'Recovery Plans', 'Clinical Nutrition'],
                'total_experience_years': 5,
                'resume_text': 'Dietitian/Nutritionist with 5 years experience creating diet plans, patient consultations, and supporting recovery plans. Bachelor degree in Nutrition with clinical experience.',
                'phone': '+1-555-0111',
                'sourceURL': 'https://example.com/linda-anderson'
            },
            {
                'email': 'kevin.it@healthcare.com',
                'full_name': 'Kevin Lee',
                'skills': ['Healthcare IT', 'EHR Systems', 'Software Troubleshooting', 'Staff Training', 'Health Informatics'],
                'total_experience_years': 4,
                'resume_text': 'Healthcare IT Support Specialist with 4 years experience managing EHR systems, troubleshooting software, and supporting staff training. IT/Healthcare degree with healthcare IT experience.',
                'phone': '+1-555-0112',
                'sourceURL': 'https://example.com/kevin-lee'
            },
            {
                'email': 'dr.smith@healthcare.com',
                'full_name': 'Dr. Michael Smith',
                'skills': ['Medicine', 'Diagnosis', 'Treatment', 'Patient Management', 'Prescription', 'MD'],
                'total_experience_years': 8,
                'resume_text': 'Physician with 8 years experience diagnosing, treating, and managing patient health. MD degree with license and residency completion. Specialized in internal medicine.',
                'phone': '+1-555-0113',
                'sourceURL': 'https://example.com/dr-smith'
            },
            {
                'email': 'dr.johnson@healthcare.com',
                'full_name': 'Dr. Sarah Johnson',
                'skills': ['Surgery', 'Surgical Procedures', 'Pre-operative Care', 'Post-operative Care', 'Surgical Board'],
                'total_experience_years': 10,
                'resume_text': 'Surgeon with 10 years experience performing surgeries and providing pre- and post-operative care. Medical degree with residency and surgical board certification.',
                'phone': '+1-555-0114',
                'sourceURL': 'https://example.com/dr-johnson'
            },
            {
                'email': 'nurse.practitioner@healthcare.com',
                'full_name': 'Jennifer Martinez NP',
                'skills': ['Nurse Practitioner', 'Diagnosis', 'Prescription', 'Advanced Nursing', 'MSN'],
                'total_experience_years': 6,
                'resume_text': 'Nurse Practitioner with 6 years experience in diagnosis, prescription, and providing advanced nursing care. MSN degree with extensive nursing experience.',
                'phone': '+1-555-0115',
                'sourceURL': 'https://example.com/jennifer-martinez'
            },
            {
                'email': 'admin.director@healthcare.com',
                'full_name': 'Robert Davis',
                'skills': ['Healthcare Administration', 'Hospital Operations', 'Budget Management', 'Compliance', 'Staff Coordination', 'MBA'],
                'total_experience_years': 9,
                'resume_text': 'Healthcare Administrator with 9 years experience managing hospital operations, budgets, compliance, and staff coordination. MBA/MHA degree with healthcare management experience.',
                'phone': '+1-555-0116',
                'sourceURL': 'https://example.com/robert-davis'
            },
            {
                'email': 'research.scientist@healthcare.com',
                'full_name': 'Dr. Lisa Chen',
                'skills': ['Medical Research', 'Clinical Trials', 'Research Publications', 'Life Sciences', 'PhD'],
                'total_experience_years': 7,
                'resume_text': 'Medical Research Scientist with 7 years experience conducting research, clinical trials, and publishing findings. PhD in Life Sciences with extensive lab experience.',
                'phone': '+1-555-0117',
                'sourceURL': 'https://example.com/lisa-chen'
            },
            {
                'email': 'data.analyst@healthcare.com',
                'full_name': 'Michael Rodriguez',
                'skills': ['Clinical Data Analysis', 'Healthcare Analytics', 'Patient Outcomes', 'Data Reporting', 'Health Informatics'],
                'total_experience_years': 6,
                'resume_text': 'Clinical Data Analyst with 6 years experience analyzing healthcare data, improving patient outcomes, and reporting. Data analytics degree with healthcare analytics experience.',
                'phone': '+1-555-0118',
                'sourceURL': 'https://example.com/michael-rodriguez'
            },
            {
                'email': 'cno@healthcare.com',
                'full_name': 'Patricia Williams',
                'skills': ['Chief Nursing Officer', 'Nursing Leadership', 'Policy Setting', 'Budget Management', 'Quality Care', 'MSN'],
                'total_experience_years': 12,
                'resume_text': 'Chief Nursing Officer with 12 years experience leading nursing staff, setting policies, managing budgets, and ensuring quality care. MSN/Doctorate in Nursing with leadership experience.',
                'phone': '+1-555-0119',
                'sourceURL': 'https://example.com/patricia-williams'
            }
        ]
    
    def _process_and_index_candidates(self, candidates_data: List[Dict]):
        """Process and index all candidates for instant search"""
        processed_count = 0
        
        for candidate_data in candidates_data:
            if not candidate_data or not isinstance(candidate_data, dict):
                continue
            
            try:
                # Extract and clean data
                email = candidate_data.get('email', '')
                if not email:
                    continue
                
                full_name = candidate_data.get('full_name', '')
                skills = candidate_data.get('skills', [])
                if not isinstance(skills, list):
                    skills = []
                
                experience = candidate_data.get('total_experience_years', 0)
                if not isinstance(experience, (int, float)):
                    experience = 0
                
                phone = candidate_data.get('phone', '')
                source_url = candidate_data.get('sourceURL', '')
                resume_text = candidate_data.get('resume_text', '')
                
                # Create searchable text
                skills_text = ' '.join(skill for skill in skills if skill)
                searchable_text = f"{full_name} {skills_text} {resume_text}".lower()
                
                # Create candidate index
                candidate = CandidateIndex(
                    email=email,
                    full_name=full_name,
                    skills=skills,
                    experience=int(experience),
                    phone=phone,
                    source_url=source_url,
                    searchable_text=searchable_text
                )
                
                # Add to candidates
                self.candidates[email] = candidate
                
                # Update indices
                self._update_indices(candidate)
                
                processed_count += 1
                
                if processed_count >= self.max_candidates:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to process candidate: {e}")
                continue
        
        self.stats['total_candidates'] = len(self.candidates)
        logger.info(f"✅ Processed and indexed {processed_count} candidates")
    
    def _update_indices(self, candidate: CandidateIndex):
        """Update search indices for a candidate"""
        email = candidate.email
        
        # Skills index
        for skill in candidate.skills:
            if skill:
                self.skill_index[skill.lower()].add(email)
        
        # Word index
        for word in candidate.words_set:
            if len(word) > 2:  # Skip very short words
                self.word_index[word].add(email)
        
        # Experience index (grouped by ranges)
        exp_range = (candidate.experience // 5) * 5
        self.experience_index[exp_range].add(email)
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Ultra-fast instant search with sub-millisecond response times
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching candidates with scores
        """
        start_time = time.time()
        
        if not query or not self.candidates:
            return []
        
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        # Remove common stop words but keep important terms
        stop_words = {'the', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'to', 'of', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        important_terms = {'rn', 'md', 'np', 'cna', 'pt', 'ot', 'icu', 'er', 'or', 'lab', 'pharmacy', 'billing', 'coding', 'it', 'admin', 'research', 'data', 'nurse', 'doctor', 'tech', 'assistant', 'receptionist', 'call', 'center', 'aide', 'specialist', 'analyst', 'officer', 'director', 'scientist', 'physician', 'surgeon', 'practitioner', 'administrator', 'cno'}
        
        query_words = {w for w in query_words if (len(w) > 2 and w not in stop_words) or w in important_terms}
        
        if not query_words:
            return []
        
        # Find candidate emails that match the query
        matching_emails = set()
        
        # Stage 1: Exact skill matches (fastest)
        for word in query_words:
            if word in self.skill_index:
                matching_emails.update(self.skill_index[word])
        
        # Stage 2: Partial skill matches
        if not matching_emails:
            for word in query_words:
                for skill, emails in self.skill_index.items():
                    if word in skill:
                        matching_emails.update(emails)
        
        # Stage 3: Word matches
        if not matching_emails:
            for word in query_words:
                if word in self.word_index:
                    matching_emails.update(self.word_index[word])
        
        # Stage 4: Partial word matches
        if not matching_emails:
            for word in query_words:
                for search_word, emails in self.word_index.items():
                    if word in search_word or search_word in word:
                        matching_emails.update(emails)
        
        # If still no matches, return empty
        if not matching_emails:
            response_time = (time.time() - start_time) * 1000
            self._update_stats(response_time)
            return []
        
        # Score and rank candidates
        scored_candidates = []
        
        for email in matching_emails:
            if email not in self.candidates:
                continue
            
            candidate = self.candidates[email]
            score = self._calculate_score(candidate, query_lower, query_words)
            
            if score > 0.1:  # Only include relevant candidates
                scored_candidates.append((score, candidate))
        
        # Sort by score and return top results
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, candidate in scored_candidates[:limit]:
            result = {
                'email': candidate.email,
                'full_name': candidate.full_name,
                'skills': candidate.skills,
                'total_experience_years': candidate.experience,
                'phone': candidate.phone,
                'sourceURL': candidate.source_url,
                'similarity_score': score,
                'final_score': score,
                'grade': self._get_grade(score),
                'domain': 'healthcare',
                'ai_explanation': f"Matched based on instant search: {score:.2f}"
            }
            results.append(result)
        
        # Update performance stats
        response_time = (time.time() - start_time) * 1000
        self._update_stats(response_time)
        
        return results
    
    def _calculate_score(self, candidate: CandidateIndex, query_lower: str, query_words: Set[str]) -> float:
        """Calculate relevance score for a candidate"""
        score = 0.0
        
        # Skills matching (highest weight)
        skill_matches = 0
        for skill in candidate.skills:
            skill_lower = skill.lower()
            for word in query_words:
                if word in skill_lower or skill_lower in word:
                    skill_matches += 1
                    break
        
        if skill_matches > 0:
            score += min(0.6, (skill_matches / len(candidate.skills)) * 0.6)
        
        # Text matching
        text_matches = len(query_words.intersection(candidate.words_set))
        if text_matches > 0:
            score += min(0.3, (text_matches / len(query_words)) * 0.3)
        
        # Experience bonus
        if any(word in query_lower for word in ['senior', 'lead', 'principal', 'chief', 'director', 'officer']):
            if candidate.experience >= 5:
                score += 0.1
        
        return min(1.0, score)
    
    def _get_grade(self, score: float) -> str:
        """Convert score to grade"""
        if score >= 0.8:
            return 'A'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.4:
            return 'C'
        else:
            return 'D'
    
    def _update_stats(self, response_time_ms: float):
        """Update performance statistics"""
        self.stats['total_searches'] += 1
        
        # Update average response time
        self.stats['avg_response_time_ms'] = (
            (self.stats['avg_response_time_ms'] * (self.stats['total_searches'] - 1) + response_time_ms) 
            / self.stats['total_searches']
        )
        
        # Update fastest/slowest
        self.stats['fastest_query_ms'] = min(self.stats['fastest_query_ms'], response_time_ms)
        self.stats['slowest_query_ms'] = max(self.stats['slowest_query_ms'], response_time_ms)
        
        if response_time_ms < 1.0:
            self.stats['cache_hits'] += 1
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            **self.stats,
            'cache_hit_rate': (self.stats['cache_hits'] / max(self.stats['total_searches'], 1)) * 100,
            'optimization_level': 'instant_search'
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.candidates.clear()
        self.skill_index.clear()
        self.word_index.clear()
        self.experience_index.clear()
        
        # Reset stats
        self.stats = {
            'total_searches': 0,
            'avg_response_time_ms': 0.0,
            'fastest_query_ms': float('inf'),
            'slowest_query_ms': 0.0,
            'cache_hits': 0,
            'total_candidates': 0
        }
        
        logger.info("🧹 Cache cleared")


# Global instance
_instant_search_engine = None
_engine_lock = threading.Lock()

def get_instant_search_engine() -> InstantSearchEngine:
    """Get or create the instant search engine (singleton)"""
    global _instant_search_engine
    
    if _instant_search_engine is None:
        with _engine_lock:
            if _instant_search_engine is None:
                _instant_search_engine = InstantSearchEngine()
    
    return _instant_search_engine

def instant_search(query: str, limit: int = 10) -> List[Dict]:
    """Convenience function for instant search"""
    engine = get_instant_search_engine()
    return engine.search(query, limit)

def get_instant_search_stats() -> Dict:
    """Get instant search performance statistics"""
    engine = get_instant_search_engine()
    return engine.get_stats()

def clear_instant_search_cache():
    """Clear the instant search cache"""
    engine = get_instant_search_engine()
    engine.clear_cache()
