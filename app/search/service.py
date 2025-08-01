
import os
import sys
import boto3
import logging
import time
import json
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

# Add adeptai components to path for imports
adeptai_components_path = os.path.join(os.path.dirname(__file__), 'adeptai_components')
if os.path.exists(adeptai_components_path):
    sys.path.append(adeptai_components_path)

# Setup logging
logger = logging.getLogger(__name__)

# Setup AWS
REGION = 'ap-south-1'

# Initialize DynamoDB only if credentials are available
try:
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        dynamodb = boto3.resource('dynamodb', region_name=REGION,
                                  aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                  aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
        table = dynamodb.Table('user-resume-metadata')
        feedback_table = dynamodb.Table('resume_feedback')
    else:
        dynamodb = None
        table = None
        feedback_table = None
except Exception as e:
    logger.warning(f"Could not initialize DynamoDB: {e}")
    dynamodb = None
    table = None
    feedback_table = None

# Try to import advanced adeptai-master components
ADVANCED_SYSTEM_AVAILABLE = False
try:
    # Check if required dependencies are available
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
   
    # Try to import the advanced system
    from enhanced_recruitment_search import EnhancedRecruitmentSearchSystem
    from enhanced_candidate_matcher import EnhancedCandidateMatchingSystem, MatchScore
    ADVANCED_SYSTEM_AVAILABLE = True
    logger.info("✅ Advanced adeptai-master system imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Advanced adeptai-master system not available: {e}")
    logger.info("🔄 Will use fallback search system")
    ADVANCED_SYSTEM_AVAILABLE = False
except Exception as e:
    logger.warning(f"⚠️ Error initializing advanced system: {e}")
    logger.info("🔄 Will use fallback search system")
    ADVANCED_SYSTEM_AVAILABLE = False

# Domain-specific keywords for better matching
DOMAIN_KEYWORDS = {
    "software": {
        'developer', 'engineer', 'backend', 'frontend', 'fullstack', 'programmer', 'software',
        'ai', 'ml', 'machine learning', 'data', 'api', 'apis', 'rest', 'graphql',
        'javascript', 'typescript', 'react', 'node', 'python', 'java', 'flask', 'django',
        'spring', 'springboot', 'hibernate', 'express', 'fastapi', 'nextjs', 'angular',
        'aws', 'gcp', 'azure', 'cloud', 'devops', 'microservices', 'docker', 'kubernetes',
        'lambda', 'serverless', 'terraform', 'ansible', 'jenkins', 'ci', 'cd',
        'linux', 'bash', 'shell', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
        'security', 'cybersecurity', 'firewall', 'penetration', 'siem', 'compliance',
        'iso', 'soc2', 'ceh', 'comptia', 'agile', 'scrum', 'jira', 'git', 'github', 'bitbucket',
        'unit testing', 'integration testing', 'automation', 'selenium', 'pytest', 'cypress',
        'nlp', 'cv', 'transformer', 'bert', 'gpt', 'llm', 'huggingface', 'pytorch', 'tensorflow',
        'pandas', 'numpy', 'matplotlib', 'scikit-learn', 'kafka', 'redis', 'elasticsearch',
        'firebase', 'sentry', 'newrelic', 'logstash', 'prometheus', 'grafana',
    },
    "healthcare": {
        'nurse', 'nursing', 'rn', 'bsc nursing', 'm.sc nursing', 'icu', 'surgery',
        'healthcare', 'patient', 'hospital', 'medical', 'clinical', 'ward', 'gynae',
        'cardiology', 'therapeutic', 'anesthesia', 'nursys', 'registered nurse',
        'cna', 'care', 'charting', 'vitals', 'mobility', 'therapy', 'rehab',
        'phlebotomy', 'pediatrics', 'geriatrics', 'ophthalmology', 'dermatology',
        'radiology', 'oncology', 'pharmacy', 'diagnosis', 'prescription', 'labs',
        'first aid', 'emergency', 'triage', 'bcls', 'acls', 'infection control',
        'patient care', 'clinical documentation', 'medication', 'wound care',
        'telemedicine', 'public health', 'mental health', 'physician', 'assistant',
        'doctor', 'dentist', 'midwife', 'vaccination', 'epidemiology', 'biomedical',
        'health record', 'ehr', 'emr', 'insurance', 'hipaa', 'claims', 'billing',
        'lab technician', 'radiographer', 'ultrasound', 'x-ray', 'immunization',
        'hematology', 'pathology', 'microbiology', 'clinical trials', 'vaccine',
        'occupational therapy', 'speech therapy', 'physical therapy', 'audiology',
        'home health', 'ambulatory care', 'long-term care', 'geriatrics nurse',
        'palliative care', 'end of life care', 'hospice', 'dementia', 'alzheimers',
        'behavioral health', 'psychology', 'psychiatry', 'mental illness', 'counseling',
        'blood pressure', 'temperature monitoring', 'surgical tech', 'scrub nurse',
        'health informatics', 'clinical informatics', 'medical coding', 'icd-10',
        'cpt coding', 'hl7', 'fhir', 'pacs', 'ris', 'health it', 'medical records',
        'case manager', 'insurance claims', 'utilization review', 'care coordinator',
        'revenue cycle', 'medical scribe', 'compliance', 'regulatory', 'audit',
        'cms', 'medicare', 'medicaid', 'prior authorization', 'medical transcription',
        'ehr implementation', 'healthcare analytics', 'population health', 'care quality',
        'patient satisfaction', 'value-based care', 'telehealth',
        'remote monitoring', 'patient portal', 'healthcare provider'
    }
}

@dataclass
class MatchScore:
    """Detailed matching score breakdown"""
    overall_score: float
    technical_skills_score: float
    experience_score: float
    seniority_score: float
    education_score: float
    soft_skills_score: float
    location_score: float
    confidence: float
    match_explanation: str
    missing_requirements: List[str]
    strength_areas: List[str]

class AdeptAIMastersAlgorithm:
    """Advanced AI-powered candidate matching algorithm with adeptai-master integration"""
   
    def __init__(self):
        self.performance_stats = {
            'total_searches': 0,
            'avg_response_time': 0,
            'advanced_system_used': 0,
            'fallback_used': 0
        }
       
        # Initialize advanced system if available
        self.advanced_system = None
        if ADVANCED_SYSTEM_AVAILABLE:
            try:
                # Try to initialize the advanced system
                self.advanced_system = EnhancedRecruitmentSearchSystem()
                logger.info("✅ Advanced adeptai-master system initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize advanced system: {e}")
                self.advanced_system = None
   
    def get_grade(self, score):
        """Convert score to letter grade - EXACT SAME AS ORIGINAL ADEPTAI MASTERS"""
        if score >= 85:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 50:
            return 'C'
        else:
            return 'D'

    def extract_keywords(self, text):
        """Extract keywords from text - EXACT SAME AS ORIGINAL ADEPTAI MASTERS"""
        if not text:
            return []
       
        try:
            # Simple keyword extraction
            words = re.findall(r'\b\w+\b', str(text).lower())
            # Filter out common words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'}
            words = [w for w in words if w not in stop_words and len(w) > 2]
           
            # Count frequency
            word_freq = Counter(words)
            return [word for word, freq in word_freq.items() if freq >= 1]
        except Exception as e:
            logger.warning(f"Error extracting keywords: {e}")
            return []
   
    def detect_domain(self, keywords):
        """Detect the domain (software/healthcare) based on keywords"""
        sw = sum(1 for w in keywords if w in DOMAIN_KEYWORDS['software'])
        hw = sum(1 for w in keywords if w in DOMAIN_KEYWORDS['healthcare'])
        return 'software' if sw >= hw else 'healthcare'
   
    def semantic_similarity(self, text1, text2):
        """Basic similarity function as fallback - EXACT SAME AS ORIGINAL ADEPTAI MASTERS"""
        words1 = set(self.extract_keywords(text1))
        words2 = set(self.extract_keywords(text2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
   
    def _load_candidates_from_dynamodb(self):
        """Load candidates from DynamoDB for advanced system indexing"""
        if not table:
            return []
       
        try:
            logger.info("📊 Loading candidates from DynamoDB for advanced indexing...")
            response = table.scan()
            items = response.get('Items', [])
            logger.info(f"✅ Loaded {len(items)} candidates from DynamoDB")
            return items
        except Exception as e:
            logger.error(f"❌ Error loading candidates from DynamoDB: {e}")
            return []
   
    def _initialize_advanced_system_if_needed(self):
        """Initialize advanced system with candidates if not already done"""
        if not self.advanced_system:
            return False
       
        try:
            # Check if system is already initialized
            if hasattr(self.advanced_system, 'candidates') and self.advanced_system.candidates:
                return True
           
            # Load candidates and initialize
            candidates = self._load_candidates_from_dynamodb()
            if candidates:
                logger.info("🔧 Initializing advanced system with candidates...")
                self.advanced_system.index_candidates(candidates)
                logger.info("✅ Advanced system initialized with candidates")
                return True
            else:
                logger.warning("⚠️ No candidates available for advanced system initialization")
                return False
        except Exception as e:
            logger.error(f"❌ Error initializing advanced system: {e}")
            return False
   
    def _format_advanced_result_for_frontend(self, advanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format advanced system result to match frontend expectations"""
        try:
            return {
                'FullName': advanced_result.get('full_name', 'Unknown'),
                'email': advanced_result.get('email', ''),
                'phone': advanced_result.get('phone', 'Not provided'),
                'Skills': advanced_result.get('skills', []),
                'Experience': f"{advanced_result.get('experience_years', 0)} years",
                'sourceURL': advanced_result.get('source_url', 'Not available'),
                'Score': int(advanced_result.get('overall_score', 0)),
                'Grade': advanced_result.get('grade', 'C'),
                'SemanticScore': float(advanced_result.get('overall_score', 0)) / 100,
                'ProcessingTimestamp': datetime.now().isoformat(),
                # Additional advanced fields
                'Education': advanced_result.get('education', ''),
                'Certifications': advanced_result.get('certifications', []),
                'SeniorityLevel': advanced_result.get('seniority_level', ''),
                'MatchExplanation': advanced_result.get('match_explanation', ''),
                'TechnicalScore': advanced_result.get('technical_skills_score', 0),
                'ExperienceScore': advanced_result.get('experience_score', 0),
                'EducationScore': advanced_result.get('education_score', 0),
                'Confidence': advanced_result.get('confidence', 0.5)
            }
        except Exception as e:
            logger.error(f"Error formatting advanced result: {e}")
            return self._create_fallback_result(advanced_result.get('full_name', 'Unknown'))
   
    def _create_fallback_result(self, name: str) -> Dict[str, Any]:
        """Create a fallback result when formatting fails"""
        return {
            'FullName': name,
            'email': 'error@example.com',
            'phone': 'Not provided',
            'Skills': [],
            'Experience': '0 years',
            'sourceURL': 'Not available',
            'Score': 50,
            'Grade': 'C',
            'SemanticScore': 0.5,
            'ProcessingTimestamp': datetime.now().isoformat()
        }
   
    def keyword_search(self, job_description, top_k=10):
        """Advanced search using adeptai-master system with fallback"""
        start_time = time.time()
        self.performance_stats['total_searches'] += 1
       
        # Handle None job_description
        if job_description is None:
            job_description = ""
       
        logger.info(f"🔍 Starting search for: '{job_description[:100]}...'")
       
        # Try advanced system first
        if self.advanced_system and self._initialize_advanced_system_if_needed():
            try:
                logger.info("🚀 Using ADVANCED adeptai-master system...")
                advanced_results = self.advanced_system.search(job_description, top_k=top_k)
               
                if advanced_results:
                    # Format results for frontend
                    formatted_results = []
                    for result in advanced_results:
                        formatted_result = self._format_advanced_result_for_frontend(result)
                        formatted_results.append(formatted_result)
                   
                    self.performance_stats['advanced_system_used'] += 1
                    response_time = time.time() - start_time
                    self.performance_stats['avg_response_time'] = (
                        (self.performance_stats['avg_response_time'] * (self.performance_stats['total_searches'] - 1) + response_time) /
                        self.performance_stats['total_searches']
                    )
                   
                    summary = f"Found {len(formatted_results)} candidates using ADVANCED adeptai-master system"
                    logger.info(f"✅ Advanced search completed: {len(formatted_results)} results in {response_time:.2f}s")
                   
                    return formatted_results, summary
                else:
                    logger.warning("⚠️ Advanced system returned no results, falling back to basic search")
            except Exception as e:
                logger.error(f"❌ Advanced system error: {e}")
                logger.info("🔄 Falling back to basic search...")
       
        # Fallback to basic search (original adeptai-master fallback)
        logger.info("🔄 Using FALLBACK search (basic keyword matching)...")
        self.performance_stats['fallback_used'] += 1
       
        return self._fallback_keyword_search(job_description, top_k)
   
    def _fallback_keyword_search(self, job_description, top_k=10):
        """DIMENSION-SAFE fallback search function - EXACT SAME AS ORIGINAL ADEPTAI MASTERS"""
        if not table:
            return [], "Database connection not available"
       
        try:
            logger.info("📊 Starting DynamoDB scan...")
            response = table.scan()
            items = response.get('Items', [])
            logger.info(f"✅ Retrieved {len(items)} items from DynamoDB")
            if not items:
                return [], "No candidates found in database"
        except Exception as e:
            logger.error(f"❌ DynamoDB error: {e}")
            return [], f"Database error: {str(e)}"
       
        valid_candidates = []
        texts_to_encode = []
       
        for item in items:
            try:
                resume_text = item.get('resume_text') or item.get('ResumeText') or ''
                skills_raw = item.get('skills') or item.get('Skills') or []
               
                if isinstance(skills_raw, str):
                    skills = [s.strip() for s in skills_raw.split(',') if s.strip()]
                else:
                    skills = skills_raw or []
               
                combined_text = f"{resume_text} {' '.join(skills)}".strip()
               
                if combined_text:
                    valid_candidates.append(item)
                    texts_to_encode.append(combined_text)
            except Exception as e:
                logger.error(f"❌ Error processing candidate: {e}")
                continue
       
        if not valid_candidates:
            return [], "No valid candidates after processing"
       
        logger.info(f"📈 Processing {len(valid_candidates)} valid candidates")
       
        similarities = []
        for text in texts_to_encode:
            sim = self.semantic_similarity(job_description, text)
            similarities.append(sim)
       
        scored_documents = []
       
        for i, (item, similarity) in enumerate(zip(valid_candidates, similarities)):
            try:
                full_name = item.get('full_name') or item.get('FullName') or f'Candidate_{i+1}'
                email = item.get('email') or f'candidate{i+1}@example.com'
                phone = item.get('phone') or 'Not provided'
                skills_raw = item.get('skills') or item.get('Skills') or []
               
                if isinstance(skills_raw, str):
                    skills = [s.strip() for s in skills_raw.split(',')]
                else:
                    skills = skills_raw
               
                exp_raw = item.get('total_experience_years') or item.get('Experience') or 0
                try:
                    experience = int(float(str(exp_raw))) if exp_raw else 0
                except:
                    experience = 0
               
                score_int = max(1, min(int(float(similarity) * 100), 100))
               
                candidate_result = {
                    'FullName': full_name,
                    'email': email,
                    'phone': phone,
                    'Skills': skills,
                    'Experience': f"{experience} years",
                    'sourceURL': item.get('sourceURL') or item.get('SourceURL') or 'Not available',
                    'Score': score_int,
                    'Grade': self.get_grade(score_int),
                    'SemanticScore': float(similarity),
                    'ProcessingTimestamp': datetime.now().isoformat()
                }
                scored_documents.append(candidate_result)
            except Exception as e:
                logger.error(f"❌ Error processing result {i}: {e}")
            continue
       
        scored_documents.sort(key=lambda x: x['Score'], reverse=True)
        results = scored_documents[:top_k]
       
        summary = f"Found {len(results)} candidates using fallback search"
        logger.info(f"✅ Fallback search completed: {len(results)} results")
       
        return results, summary
   
    def semantic_match(self, job_description, use_gpt4_reranking=True):
        """Complete semantic matching with enhanced scoring"""
        # Use the advanced keyword search
        results, summary = self.keyword_search(job_description, top_k=15)
       
        return {
            'results': results,
            'summary': summary
        }

# Initialize the algorithm
adept_ai = AdeptAIMastersAlgorithm()

def semantic_match(job_description):
    """Main entry point for semantic matching - maintains backward compatibility"""
    return adept_ai.semantic_match(job_description)

def keyword_search(job_description, top_k=10):
    """Main entry point for keyword search - maintains backward compatibility"""
    return adept_ai.keyword_search(job_description, top_k)

def register_feedback(candidate_id, positive=True):
    """Register feedback for a candidate"""
    try:
        if feedback_table:
            # Update DynamoDB
            feedback_table.put_item(Item={
                'candidate_id': candidate_id,
                'positive': 1 if positive else 0,
                'negative': 0 if positive else 1,
                'last_updated': datetime.utcnow().isoformat()
            })
        logger.info(f"Feedback registered for candidate {candidate_id}")
    except Exception as e:
        logger.error(f"Error registering feedback: {e}")