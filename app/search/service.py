import os
import sys
import boto3
import logging
import time
import json
import numpy as np
import re
from datetime import datetime
from app.simple_logger import get_logger
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
from app.cache import search_cache, cache_manager

# Add adeptai components to path for imports
adeptai_components_path = os.path.join(os.path.dirname(__file__), 'adeptai_components')
if os.path.exists(adeptai_components_path):
    sys.path.append(adeptai_components_path)

# Setup logging
logger = get_logger("search")

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

# Import the ORIGINAL AdeptAI algorithm components (from adeptbackend/)
try:
    # Import the main enhanced recruitment search system
    from .adeptai_components.enhanced_recruitment_search import (
        EnhancedRecruitmentSearchSystem, 
        CandidateProfile, 
        SkillExtractor, 
        MemoryOptimizedEmbeddingSystem
    )
    
    # Import the enhanced candidate matcher
    from .adeptai_components.enhanced_candidate_matcher import (
        EnhancedCandidateMatchingSystem, 
        MatchScore
    )
    
    # Import the advanced query parser
    from .adeptai_components.advanced_query_parser import (
        AdvancedJobQueryParser, 
        ParsedJobQuery, 
        JobRequirement
    )
    
    # Import search components
    from .adeptai_components.search.fast_search import OptimizedSearchSystem
    from .adeptai_components.search.ultra_fast_search import FastSearchSystem
    from .adeptai_components.search.performance import PerformanceMonitor
    from .adeptai_components.search.cache import EmbeddingCache
    
    # Import utils components
    from .adeptai_components.utils.batch_processor import BatchProcessor
    from .adeptai_components.utils.caching import EmbeddingCache as UtilsEmbeddingCache
    from .adeptai_components.utils.precision_scorer import AdvancedSkillMatcher
    from .adeptai_components.utils.query_parser import NaturalLanguageQueryParser
    from .adeptai_components.utils.enhanced_embeddings import MultiModelEmbeddingService
    
    # Import additional components (if they exist)
    try:
        from .adeptai_components.domain_integration import DomainIntegration
    except ImportError:
        DomainIntegration = None
        
    try:
        from .adeptai_components.bias_prevention import BiasPrevention
    except ImportError:
        BiasPrevention = None
    
    ORIGINAL_ALGORITHM_AVAILABLE = True
    logger.info("Original AdeptAI algorithm imported successfully")
    
except ImportError as e:
    logger.warning(f"Original AdeptAI algorithm not available: {e}")
    ORIGINAL_ALGORITHM_AVAILABLE = False

# Fallback algorithm (simplified version)
@dataclass
class FallbackMatchScore:
    """Fallback matching score"""
    overall_score: float
    confidence: float
    match_explanation: str

class FallbackAlgorithm:
    """Fallback algorithm when original is not available"""
    
    def __init__(self):
        self.candidates = []
        logger.info("Fallback algorithm initialized")
    
    def keyword_search(self, job_description, top_k=20):
        """Simple keyword-based search"""
        if not job_description:
            return [], "No job description provided"
        
        # Simple keyword matching
        keywords = self.extract_keywords(job_description.lower())
        results = []
        
        for candidate in self.candidates[:top_k]:
            score = self.calculate_simple_score(candidate, keywords)
            results.append({
                'email': candidate.get('email', 'unknown'),
                'full_name': candidate.get('full_name', 'Unknown'),
                'phone': candidate.get('phone', 'Not provided'),
                'match_percentage': score,
                'grade': self.get_grade(score),
                'category': 'IT/Tech',
                'top_skills': candidate.get('skills', [])[:3],
                'experience_years': candidate.get('experience_years', 0),
                'location': candidate.get('location', 'Unknown'),
                'source_url': candidate.get('source_url', ''),
                'resume_text': candidate.get('resume_text', ''),
                'match_explanation': f"Basic keyword match: {score}%"
            })
        
        return results, f"Found {len(results)} candidates using fallback algorithm"
    
    def extract_keywords(self, text):
        """Extract keywords from text"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def calculate_simple_score(self, candidate, keywords):
        """Calculate simple matching score"""
        candidate_text = f"{candidate.get('full_name', '')} {candidate.get('resume_text', '')} {' '.join(candidate.get('skills', []))}"
        candidate_text = candidate_text.lower()
        
        matches = sum(1 for keyword in keywords if keyword in candidate_text)
        return min(100, (matches / len(keywords)) * 100) if keywords else 0
    
    def get_grade(self, score):
        """Get grade based on score"""
        if score >= 80:
            return "Grade A"
        elif score >= 60:
            return "Grade B"
        elif score >= 40:
            return "Grade C"
        else:
            return "Grade D"

    def semantic_match(self, job_description, use_gpt4_reranking=True):
        """Semantic match method for fallback algorithm"""
        try:
            logger.info(f"Fallback algorithm: Performing semantic match for query: {job_description[:100]}...")
            
            # Use keyword search as fallback
            results, summary = self.keyword_search(job_description, top_k=20)
            
            return {
                'results': results,
                'summary': summary,
                'total_candidates': len(results),
                'search_query': job_description,
                'algorithm_used': 'Fallback Algorithm',
                'fallback': True
            }
            
        except Exception as e:
            logger.error(f"Fallback algorithm error: {e}", exc_info=True)
            return {
                'results': [],
                'summary': f"Fallback search failed: {str(e)}",
                'total_candidates': 0,
                'search_query': job_description,
                'algorithm_used': 'Fallback Algorithm',
                'fallback': True,
                'error': True
            }

class AdeptAIMastersAlgorithm:
    """Main algorithm class that uses the original AdeptAI implementation"""
    
    def __init__(self):
        self.performance_stats = {
            'total_searches': 0,
            'avg_response_time': 0,
            'original_algorithm_used': 0,
            'fallback_used': 0
        }
        
        # Initialize the original algorithm if available
        if ORIGINAL_ALGORITHM_AVAILABLE:
            try:
                # Initialize the enhanced recruitment search system
                self.enhanced_system = EnhancedRecruitmentSearchSystem()
                
                # Initialize performance monitoring
                self.performance_monitor = PerformanceMonitor()
                
                # Initialize caching system
                self.embedding_cache = EmbeddingCache()
                
                # Initialize batch processor
                self.batch_processor = BatchProcessor()
                
                # Initialize advanced utils
                self.skill_matcher = AdvancedSkillMatcher()
                self.query_parser = NaturalLanguageQueryParser()
                self.multi_model_embeddings = MultiModelEmbeddingService()
                self.utils_cache = UtilsEmbeddingCache()
                
                # Initialize additional components (if available)
                self.domain_integration = DomainIntegration() if DomainIntegration else None
                self.bias_prevention = BiasPrevention() if BiasPrevention else None
                
                logger.info("✅ Original AdeptAI algorithm initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Error initializing original algorithm: {e}")
                self.enhanced_system = None
        else:
            self.enhanced_system = None
            logger.warning("⚠️ Original algorithm not available, using fallback")
        
        # Initialize fallback algorithm
        self.fallback_algorithm = FallbackAlgorithm()
        
        # Load candidates from DynamoDB if available
        self._load_candidates_from_dynamodb()
    
    def _load_candidates_from_dynamodb(self):
        """Load candidates from DynamoDB"""
        if table:
            try:
                response = table.scan()
                candidates = response.get('Items', [])
                logger.info(f"Loaded {len(candidates)} candidates from DynamoDB")
                
                # Store candidates in fallback algorithm
                self.fallback_algorithm.candidates = candidates
                
                # If original algorithm is available, index the candidates
                if self.enhanced_system and candidates:
                    try:
                        self.enhanced_system.index_candidates(candidates)
                        logger.info("✅ Candidates indexed in original algorithm")
                    except Exception as e:
                        logger.error(f"❌ Error indexing candidates: {e}")
                        
            except Exception as e:
                logger.error(f"❌ Error loading candidates from DynamoDB: {e}")
        else:
            logger.warning("⚠️ DynamoDB not available, using empty candidate list")
    
    def get_grade(self, score):
        """Get grade based on score"""
        if score >= 80:
            return "Grade A"
        elif score >= 60:
            return "Grade B"
        elif score >= 40:
            return "Grade C"
        else:
            return "Grade D"
    
    def extract_keywords(self, text):
        """Extract keywords from text using original algorithm"""
        if not text:
            return []
        
        try:
            # Use the original algorithm's keyword extraction
            if self.enhanced_system:
                # Extract keywords using the enhanced system
                keywords = re.findall(r'\b\w+\b', text.lower())
                # Remove common stop words
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                return [word for word in keywords if word not in stop_words and len(word) > 2]
            else:
                # Fallback keyword extraction
                return self.fallback_algorithm.extract_keywords(text)
        except Exception as e:
            logger.error(f"❌ Error extracting keywords: {e}")
            return []
    
    def detect_domain(self, keywords):
        """Detect domain based on keywords"""
        if not keywords:
            return "General"
        
        # Enhanced domain detection logic with more comprehensive keywords
        tech_keywords = {
    'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node', 'express',
    'spring', 'hibernate', 'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'nosql', 'firebase',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'ci/cd',
    'git', 'github', 'gitlab', 'bitbucket', 'api', 'rest', 'graphql', 'json', 'xml',
    'frontend', 'backend', 'fullstack', 'devops', 'sre', 'networking', 'tcp/ip', 'dns', 'http',
    'https', 'cloud', 'serverless', 'lambda', 'ec2', 's3', 'vpc', 'rds', 'cloudfront', 'elastic beanstalk',
    'machine learning', 'deep learning', 'nlp', 'computer vision', 'neural networks',
    'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy', 'data science',
    'data engineering', 'etl', 'big data', 'spark', 'hadoop', 'kafka', 'airflow', 'mlops',
    'cybersecurity', 'penetration testing', 'vulnerability assessment', 'firewall', 'ids', 'ips',
    'ethical hacking', 'encryption', 'security analyst', 'information security', 'siem',
    'system administrator', 'linux', 'bash', 'powershell', 'windows server', 'active directory',
    'virtualization', 'vmware', 'hyper-v', 'containers', 'microservices', 'service mesh',
    'api gateway', 'load balancer', 'reverse proxy', 'ssl', 'tls', 'certificates',
    'observability', 'monitoring', 'logging', 'prometheus', 'grafana', 'elasticsearch', 'splunk',
    'software engineer', 'developer', 'programmer', 'software architect', 'technical lead',
    'qa', 'quality assurance', 'test automation', 'selenium', 'cypress', 'junit', 'testng',
    'scrum', 'agile', 'kanban', 'jira', 'confluence', 'project management', 'product management',
    'ux', 'ui', 'design systems', 'responsive design', 'accessibility', 'wireframes',
    'figma', 'adobe xd', 'version control', 'containerization', 'release engineering',
    'refactoring', 'code review', 'clean code', 'oop', 'functional programming',
    'data structures', 'algorithms', 'system design', 'design patterns', 'low-level design',
    'high-level design', 'load testing', 'performance tuning', 'scalability', 'reliability',
    'availability', 'fault tolerance', 'disaster recovery', 'backup', 'integration testing',
    'unit testing', 'acceptance testing', 'regression testing', 'a/b testing',
    'command line', 'cli tools', 'dev environment', 'ide', 'intellij', 'vscode', 'eclipse',
    'visual studio', 'notepad++', 'debugging', 'profiling', 'deployment', 'release',
    'continuous integration', 'continuous deployment', 'infrastructure as code', 'cloud-native',
    'edge computing', 'blockchain', 'web3', 'solidity', 'smart contracts', 'crypto', 'bitcoin',
    'ethereum', 'iot', 'robotics', 'embedded systems', 'firmware', 'raspberry pi', 'arduino',
    'low-code', 'no-code', 'platform engineering', 'data lake', 'data warehouse',
    'business intelligence', 'tableau', 'power bi', 'lookml', 'metabase', 'snowflake',
    'dbt', 'redshift', 'athena', 'clickhouse', 'databricks', 'elastic stack', 'logstash',
    'kibana', 'telemetry', 'incident management', 'on-call', 'sla', 'slo', 'slis'
}

        healthcare_keywords = {
    'nursing', 'rn', 'lpn', 'nurse practitioner', 'registered nurse', 'licensed practical nurse',
    'medical', 'clinical', 'hospital', 'patient', 'care', 'healthcare', 'doctor', 'physician',
    'surgeon', 'therapist', 'occupational therapist', 'physical therapist', 'speech therapist',
    'pharmacist', 'pharmacy', 'technician', 'lab tech', 'x-ray tech', 'radiologic technologist',
    'ultrasound technician', 'sonographer', 'radiology', 'mri', 'ct scan', 'imaging', 'diagnostics',
    'treatment', 'medication', 'medication administration', 'dosage', 'prescription', 'drug',
    'charting', 'electronic health record', 'emr', 'ehr', 'epic', 'cerner', 'meditech',
    'vitals', 'blood pressure', 'pulse', 'respiration', 'temperature', 'oxygen saturation',
    'icu', 'ccu', 'ward', 'inpatient', 'outpatient', 'emergency', 'er', 'triage',
    'ambulance', 'paramedic', 'emt', 'first responder', 'bcls', 'acls', 'cpr',
    'infection control', 'aseptic technique', 'sterilization', 'hand hygiene',
    'wound care', 'dressing change', 'surgical wound', 'incision', 'suturing',
    'patient care', 'patient safety', 'discharge planning', 'care coordination',
    'home healthcare', 'visiting nurse', 'hospice', 'palliative care',
    'nursing home', 'long-term care', 'assisted living', 'geriatrics',
    'pediatrics', 'neonatal', 'nicu', 'labor and delivery', 'ob/gyn', 'midwife',
    'mental health', 'psychiatry', 'psychology', 'behavioral health',
    'substance abuse', 'addiction treatment', 'detox', 'rehab', 'counseling',
    'social work', 'case management', 'insurance', 'medicare', 'medicaid',
    'claims', 'billing', 'medical coding', 'icd-10', 'cpt', 'hipaa',
    'compliance', 'regulatory', 'quality assurance', 'joint commission',
    'clinical documentation', 'progress notes', 'care plan', 'assessment',
    'evaluation', 'diagnosis', 'disease management', 'chronic care',
    'diabetes management', 'hypertension', 'cardiology', 'oncology',
    'nephrology', 'pulmonology', 'neurology', 'gastroenterology',
    'hepatology', 'endocrinology', 'orthopedics', 'pain management',
    'anesthesiology', 'surgery', 'pre-op', 'post-op', 'perioperative',
    'scrub nurse', 'circulating nurse', 'anesthetist', 'surgical tech',
    'occupational health', 'industrial medicine', 'telemedicine', 'telehealth',
    'virtual care', 'remote monitoring', 'health informatics', 'biomedical',
    'clinical trials', 'research nurse', 'principal investigator',
    'institutional review board', 'data collection', 'public health',
    'epidemiology', 'vaccination', 'immunization', 'screening', 'prevention',
    'wellness', 'health education', 'nutrition', 'dietician', 'fitness',
    'rehabilitation', 'physical medicine', 'mobility', 'assistive devices',
    'wheelchair', 'prosthetics', 'orthotics', 'speech pathology',
    'medical assistant', 'certified nursing assistant', 'cna',
    'medical receptionist', 'healthcare administrator', 'medical records',
    'healthcare provider', 'healthcare professional', 'continuing education',
    'board certification', 'state license', 'clinical rotation',
    'nursing school', 'preceptorship', 'internship', 'residency',
    'fellowship', 'grand rounds', 'rounding', 'consultation', 'handoff',
    'multidisciplinary team', 'interprofessional', 'patient satisfaction',
    'patient rights', 'informed consent', 'advanced directive', 'dnr',
    'code blue', 'rapid response', 'falls risk', 'pressure ulcer', 'restraints','cna','word','wound' 'care','charting'
}

        # Convert keywords to lowercase for comparison
        keywords_lower = [kw.lower() for kw in keywords]
        
        tech_count = sum(1 for keyword in keywords_lower if any(tech_kw in keyword for tech_kw in tech_keywords))
        healthcare_count = sum(1 for keyword in keywords_lower if any(health_kw in keyword for health_kw in healthcare_keywords))
        
        if tech_count > healthcare_count:
            return "IT/Tech"
        elif healthcare_count > tech_count:
            return "Healthcare"
        else:
            return "General"
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        try:
            # Use the original algorithm's semantic similarity
            if self.enhanced_system:
                # This would use the enhanced system's embedding model
                # For now, use a simple similarity calculation
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union) if union else 0.0
            else:
                # Fallback similarity calculation
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union) if union else 0.0
        except Exception as e:
            logger.error(f"❌ Error calculating semantic similarity: {e}")
            return 0.0
    
    def keyword_search(self, job_description, top_k=20):
        """Advanced search using the original AdeptAI algorithm with caching"""
        start_time = time.time()
        self.performance_stats['total_searches'] += 1
        
        # Record search start in performance monitor
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            self.performance_monitor.record_search(0, job_description, False)
        
        # Handle None job_description
        if job_description is None:
            job_description = ""
        
        logger.info(f"🔍 Starting ORIGINAL AdeptAI search for: '{job_description[:100]}...'")
        
        # Check cache first
        try:
            cached_results = search_cache.get_search_results(job_description, {'top_k': top_k})
            if cached_results:
                logger.info("✅ Cache hit for search query")
                response_time = time.time() - start_time
                self.performance_stats['avg_response_time'] = (
                    (self.performance_stats['avg_response_time'] * (self.performance_stats['total_searches'] - 1) + response_time) /
                    self.performance_stats['total_searches']
                )
                return cached_results, f"Found {len(cached_results)} candidates (cached)"
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        # Try original algorithm first
        if self.enhanced_system:
            try:
                logger.info("🚀 Using ORIGINAL AdeptAI algorithm...")
                
                # Use batch processing if available
                if hasattr(self, 'batch_processor') and self.batch_processor:
                    logger.info("📦 Using batch processing for enhanced performance")
                
                # Use the original algorithm's search method
                original_results = self.enhanced_system.search(job_description, top_k=top_k)
                
                if original_results:
                    # Format results for frontend
                    formatted_results = []
                    for result in original_results:
                        formatted_result = self._format_original_result_for_frontend(result)
                        
                        # Enhance with skill matching if available
                        if hasattr(self, 'skill_matcher') and self.skill_matcher:
                            try:
                                # Extract skills from the result
                                candidate_skills = result.get('skills', [])
                                if isinstance(candidate_skills, str):
                                    candidate_skills = [candidate_skills]
                                
                                # Calculate enhanced skill score
                                skill_score = self.skill_matcher.calculate_skill_match_score(
                                    candidate_skills, 
                                    job_description
                                )
                                formatted_result['enhanced_skill_score'] = skill_score
                            except Exception as e:
                                logger.warning(f"⚠️ Skill matching failed: {e}")
                        
                        formatted_results.append(formatted_result)
                    
                    self.performance_stats['original_algorithm_used'] += 1
                    response_time = time.time() - start_time
                    self.performance_stats['avg_response_time'] = (
                        (self.performance_stats['avg_response_time'] * (self.performance_stats['total_searches'] - 1) + response_time) /
                        self.performance_stats['total_searches']
                    )
                    
                    # Record performance metrics
                    if hasattr(self, 'performance_monitor') and self.performance_monitor:
                        self.performance_monitor.record_search(response_time, job_description, True)
                    
                    summary = f"Found {len(formatted_results)} candidates using ORIGINAL AdeptAI algorithm"
                    logger.info(f"✅ Original algorithm search completed: {len(formatted_results)} results in {response_time:.2f}s")
                    
                    # Cache the results
                    try:
                        search_cache.set_search_results(job_description, formatted_results, {'top_k': top_k})
                        logger.debug("✅ Search results cached successfully")
                    except Exception as e:
                        logger.warning(f"Failed to cache search results: {e}")
                    
                    return formatted_results, summary
                else:
                    logger.warning("⚠️ Original algorithm returned no results, falling back to fallback search")
            except Exception as e:
                logger.error(f"❌ Original algorithm error: {e}")
                logger.info("🔄 Falling back to fallback search...")
        
        # Fallback to simple search
        logger.info("🔄 Using FALLBACK search (simple keyword matching)...")
        self.performance_stats['fallback_used'] += 1
        
        # Record fallback usage
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            self.performance_monitor.record_search(0, job_description, False)
        
        return self.fallback_algorithm.keyword_search(job_description, top_k)
    
    def _format_original_result_for_frontend(self, original_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format original algorithm result for frontend compatibility"""
        try:
            # Extract basic information
            email = original_result.get('email', 'unknown')
            full_name = original_result.get('full_name', 'Unknown')
            phone = original_result.get('phone', 'Not provided')
            location = original_result.get('location', 'Unknown')
            source_url = original_result.get('source_url', '')
            resume_text = original_result.get('resume_text', '')
            
            # Extract skills
            skills = original_result.get('skills', [])
            if isinstance(skills, str):
                skills = [skills]
            elif not isinstance(skills, list):
                skills = []
            
            # Calculate match percentage - check multiple possible field names
            match_percentage = 0
            if 'overall_score' in original_result:
                match_percentage = original_result.get('overall_score', 0)
            elif 'match_percentage' in original_result:
                match_percentage = original_result.get('match_percentage', 0)
            elif 'Score' in original_result:
                match_percentage = original_result.get('Score', 0)
            
            # Ensure it's a number and round to 1 decimal place
            if isinstance(match_percentage, str):
                try:
                    match_percentage = float(match_percentage.replace('%', ''))
                except:
                    match_percentage = 0
            elif not isinstance(match_percentage, (int, float)):
                match_percentage = 0
            
            # Round the percentage to 1 decimal place
            match_percentage = round(match_percentage, 1)
            
            # Get experience years
            experience_years = original_result.get('experience_years', 0)
            if isinstance(experience_years, str):
                try:
                    experience_years = int(experience_years)
                except:
                    experience_years = 0
            
            # Determine category - check if already set, otherwise detect from skills and education
            category = original_result.get('category', '')
            if not category:
                # Extract keywords from skills, education, and experience for category detection
                skills_text = ' '.join(skills) if skills else ''
                education_text = original_result.get('education', '')
                experience_text = original_result.get('experience', '')
                combined_text = f"{skills_text} {education_text} {experience_text}".lower()
                
                # Check for healthcare keywords first (more specific)
                healthcare_indicators = ['lpn', 'rn', 'nursing', 'nurse', 'patient care', 'medical', 'healthcare', 'clinical', 'hospital', 'licensed practical nurse', 'registered nurse', 'practical nursing']
                if any(indicator in combined_text for indicator in healthcare_indicators):
                    category = "Healthcare"
                # Check for IT/Tech keywords
                elif any(tech_word in combined_text for tech_word in ['python', 'java', 'javascript', 'developer', 'programming', 'software', 'database', 'aws', 'cloud']):
                    category = "IT/Tech"
                else:
                    category = "General"
            
            # Get grade
            grade = original_result.get('grade', self.get_grade(match_percentage))
            
            # Get match explanation
            match_explanation = original_result.get('match_explanation', f"Match score: {match_percentage}%")
            
            return {
                'email': email,
                'FullName': full_name,  # Frontend expects FullName
                'phone': phone,
                'Score': match_percentage,  # Frontend expects Score (rounded)
                'Grade': grade,  # Frontend expects Grade
                'category': category,  # Frontend expects category
                'skills': skills,  # Frontend expects skills array
                'Skills': skills,  # Also provide Skills for compatibility
                'Experience': str(experience_years),  # Frontend expects Experience as string
                'experience': str(experience_years),  # Also provide experience for compatibility
                'location': location,
                'sourceUrl': source_url,  # Frontend expects sourceUrl
                'sourceURL': source_url,  # Also provide sourceURL for compatibility
                'resumeText': resume_text,  # Frontend expects resumeText
                'match_explanation': match_explanation,  # Backend field name
                'MatchExplanation': match_explanation,  # Frontend expects MatchExplanation
                'seniority_level': original_result.get('seniority_level', 'Mid-level'),
                'Education': original_result.get('education', ''),  # Frontend expects Education
                'Certifications': original_result.get('certifications', []),  # Frontend expects Certifications
                'previous_roles': original_result.get('previous_roles', []),
                'industries': original_result.get('industries', []),
                # Add additional fields that frontend might expect
                'name': full_name,  # Alternative name field
                'contactInfo': {
                    'email': email,
                    'phone': phone
                },
                'Confidence': 75,  # Default confidence score
                'source': 'AI Matching'
            }
        except Exception as e:
            logger.error(f"❌ Error formatting original result: {e}")
            return self._create_fallback_result(full_name or "Unknown")
    
    def _create_fallback_result(self, name: str) -> Dict[str, Any]:
        """Create a fallback result when formatting fails"""
        return {
            'email': 'unknown',
            'FullName': name,  # Frontend expects FullName
            'name': name,  # Alternative name field
            'phone': 'Not provided',
            'Score': 0,  # Frontend expects Score
            'Grade': 'D',  # Frontend expects Grade
            'category': 'IT/Tech',
            'skills': [],  # Frontend expects skills array
            'Skills': [],  # Also provide Skills for compatibility
            'Experience': '0',  # Frontend expects Experience as string
            'experience': '0',  # Also provide experience for compatibility
            'location': 'Unknown',
            'sourceUrl': '',  # Frontend expects sourceUrl
            'sourceURL': '',  # Also provide sourceURL for compatibility
            'resumeText': '',  # Frontend expects resumeText
            'match_explanation': 'Error processing result',
            'MatchExplanation': 'Error processing result',
            'seniority_level': 'Unknown',
            'Education': '',  # Frontend expects Education
            'Certifications': [],  # Frontend expects Certifications
            'previous_roles': [],
            'industries': [],
            'contactInfo': {
                'email': 'unknown',
                'phone': 'Not provided'
            },
            'Confidence': 0,  # Default confidence score
            'source': 'AI Matching'
        }
    
    def get_enhanced_performance_stats(self):
        """Get comprehensive performance statistics"""
        stats = {
            'total_searches': self.performance_stats['total_searches'],
            'avg_response_time': round(self.performance_stats['avg_response_time'], 2),
            'original_algorithm_usage': self.performance_stats['original_algorithm_used'],
            'fallback_usage': self.performance_stats['fallback_used'],
            'original_algorithm_available': ORIGINAL_ALGORITHM_AVAILABLE
        }
        
        # Add performance monitor stats if available
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            try:
                monitor_stats = self.performance_monitor.get_current_status()
                stats.update({
                    'performance_monitor_stats': monitor_stats,
                    'cache_hit_rate': self.performance_monitor.get_cache_hit_rate(),
                    'total_indexing_operations': self.performance_monitor.get_total_indexing_operations()
                })
            except Exception as e:
                logger.warning(f"⚠️ Could not get performance monitor stats: {e}")
        
        # Add cache stats if available
        if hasattr(self, 'embedding_cache') and self.embedding_cache:
            try:
                cache_stats = self.embedding_cache.get_stats()
                stats['embedding_cache_stats'] = cache_stats
            except Exception as e:
                logger.warning(f"⚠️ Could not get cache stats: {e}")
        
        # Add batch processor stats if available
        if hasattr(self, 'batch_processor') and self.batch_processor:
            try:
                batch_stats = self.batch_processor.get_stats()
                stats['batch_processor_stats'] = batch_stats
            except Exception as e:
                logger.warning(f"⚠️ Could not get batch processor stats: {e}")
        
        return stats
    
    def semantic_match(self, job_description, use_gpt4_reranking=True):
        """Semantic matching using the original algorithm"""
        try:
            # Use the original algorithm's semantic matching
            if self.enhanced_system:
                logger.info("🔍 Using original algorithm for semantic matching")
                results, summary = self.keyword_search(job_description, top_k=20)
                return {'results': results, 'summary': summary}
            else:
                logger.info("🔍 Using fallback algorithm for semantic matching")
                fallback_results = self.fallback_algorithm.semantic_match(job_description)
                return {'results': fallback_results['results'], 'summary': fallback_results['summary']}
        except Exception as e:
            logger.error(f"❌ Error in semantic matching: {e}")
            return {'results': [], 'summary': f"Error in semantic matching: {str(e)}"}

# Global instances
_algorithm_instance = None

def get_algorithm_instance():
    """Get or create the algorithm instance with error handling"""
    global _algorithm_instance
    try:
        if _algorithm_instance is None:
            logger.info("🔧 Initializing new algorithm instance...")
            _algorithm_instance = AdeptAIMastersAlgorithm()
            logger.info("✅ Algorithm instance initialized successfully")
        return _algorithm_instance
    except Exception as e:
        logger.error(f"❌ Failed to initialize algorithm instance: {e}", exc_info=True)
        # Return a minimal fallback instance
        if _algorithm_instance is None:
            logger.warning("⚠️ Creating minimal fallback algorithm instance")
            _algorithm_instance = FallbackAlgorithm()
    return _algorithm_instance

def semantic_match(job_description):
    """Semantic matching function with error handling and fallback"""
    try:
        logger.info(f"Starting semantic match for query: {job_description[:100]}...")
        
        # Try to get the algorithm instance
        try:
            algorithm = get_algorithm_instance()
            logger.info("✅ Algorithm instance obtained successfully")
        except Exception as e:
            logger.error(f"❌ Failed to get algorithm instance: {e}")
            # Return fallback results instead of failing
            return {
                'results': [],
                'summary': f"Search temporarily unavailable. Please try again later. Error: {str(e)}",
                'error': True,
                'fallback': True
            }
        
        # Try to perform the semantic match
        try:
            result = algorithm.semantic_match(job_description)
            logger.info(f"✅ Semantic match completed successfully: {result.get('summary', 'No summary')}")
            return result
        except Exception as e:
            logger.error(f"❌ Semantic match failed: {e}", exc_info=True)
            # Return fallback results instead of failing
            return {
                'results': [],
                'summary': f"Search failed. Please try again later. Error: {str(e)}",
                'error': True,
                'fallback': True
            }
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in semantic_match: {e}", exc_info=True)
        # Return fallback results instead of failing
        return {
            'results': [],
            'summary': f"Search temporarily unavailable. Please try again later. Error: {str(e)}",
            'error': True,
            'fallback': True
        }

def keyword_search(job_description, top_k=20):
    """Keyword search function"""
    algorithm = get_algorithm_instance()
    return algorithm.keyword_search(job_description, top_k)

def register_feedback(candidate_id, positive=True):
    """Register feedback for a candidate"""
    try:
        if feedback_table:
            feedback_table.put_item(Item={
                'candidate_id': candidate_id,
                'feedback': 'positive' if positive else 'negative',
                'timestamp': datetime.utcnow().isoformat()
            })
            logger.info(f"Feedback registered for candidate {candidate_id}")
        else:
            logger.warning("Feedback table not available")
    except Exception as e:

        logger.error(f"Error registering feedback: {e}")
