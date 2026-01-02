import re
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TalentMatcher:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )

    def calculate_match_score(self, job_description: str, resume: str) -> Dict[str, float]:
        """Calculate real match score based on semantic similarity and skill overlap"""
        try:
            # Clean and preprocess text
            job_clean = self._clean_text(job_description)
            resume_clean = self._clean_text(resume)
            
            # Extract skills from both texts
            job_skills = self._extract_skills(job_clean)
            resume_skills = self._extract_skills(resume_clean)
            
            # Calculate skill overlap
            skill_overlap = self._calculate_skill_overlap(job_skills, resume_skills)
            
            # Calculate semantic similarity using TF-IDF
            semantic_similarity = self._calculate_semantic_similarity(job_clean, resume_clean)
            
            # Calculate experience relevance
            experience_relevance = self._calculate_experience_relevance(job_clean, resume_clean)
            
            # Calculate education relevance
            education_relevance = self._calculate_education_relevance(job_clean, resume_clean)
            
            # Calculate location relevance
            location_relevance = self._calculate_location_relevance(job_clean, resume_clean)
            
            # Weighted overall score
            overall_score = (
                skill_overlap * 0.35 +
                semantic_similarity * 0.30 +
                experience_relevance * 0.20 +
                education_relevance * 0.10 +
                location_relevance * 0.05
            )
            
            # Ensure score is between 50 and 100 (addressing user requirement)
            overall_score = max(50, min(100, overall_score * 100))
            
            return {
                "overall_match_percentage": overall_score,
                "skill_overlap": skill_overlap * 100,
                "semantic_similarity": semantic_similarity * 100,
                "experience_relevance": experience_relevance * 100,
                "education_relevance": education_relevance * 100,
                "location_relevance": location_relevance * 100
            }
            
        except Exception as e:
            print(f"Error calculating match score: {e}")
            return {
                "overall_match_percentage": 50.0,  # Fallback score
                "skill_overlap": 50.0,
                "semantic_similarity": 50.0,
                "experience_relevance": 50.0,
                "education_relevance": 50.0,
                "location_relevance": 50.0
            }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove special characters, normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using common skill patterns"""
        # Common technical skills
        skill_patterns = [
            r'\b(python|java|c\+\+|javascript|html|css|sql|react|angular|vue|node\.js|docker|kubernetes|aws|azure|gcp)\b',
            r'\b(machine learning|ai|artificial intelligence|data science|analytics|statistics)\b',
            r'\b(project management|agile|scrum|kanban|lean|six sigma)\b',
            r'\b(leadership|team management|mentoring|coaching|communication)\b',
            r'\b(design|ux|ui|user experience|user interface|wireframing|prototyping)\b'
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.update(matches)
        
        return list(skills)

    def _calculate_skill_overlap(self, job_skills: List[str], resume_skills: List[str]) -> float:
        """Calculate skill overlap between job requirements and resume"""
        if not job_skills:
            return 0.5  # Default if no skills found
        
        if not resume_skills:
            return 0.1  # Very low if no skills in resume
        
        # Calculate Jaccard similarity
        intersection = set(job_skills) & set(resume_skills)
        union = set(job_skills) | set(resume_skills)
        
        if not union:
            return 0.5
        
        return len(intersection) / len(union)

    def _calculate_semantic_similarity(self, job_text: str, resume_text: str) -> float:
        """Calculate semantic similarity using TF-IDF and cosine similarity"""
        try:
            if not job_text or not resume_text:
                return 0.5
            
            # Vectorize texts
            vectors = self.vectorizer.fit_transform([job_text, resume_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return max(0, min(1, similarity))
        except Exception:
            return 0.5

    def _calculate_experience_relevance(self, job_text: str, resume_text: str) -> float:
        """Calculate experience relevance based on years and seniority"""
        try:
            # Extract years of experience patterns
            job_exp_pattern = r'\b(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)\b'
            resume_exp_pattern = r'\b(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)\b'
            
            job_exp = re.findall(job_exp_pattern, job_text, re.IGNORECASE)
            resume_exp = re.findall(resume_exp_pattern, resume_text, re.IGNORECASE)
            
            if not job_exp or not resume_exp:
                return 0.5
            
            job_years = int(job_exp[0])
            resume_years = int(resume_exp[0])
            
            # Calculate relevance (closer years = higher relevance)
            if resume_years >= job_years:
                return 0.8  # Overqualified or perfect match
            elif resume_years >= job_years * 0.7:
                return 0.6  # Close match
            elif resume_years >= job_years * 0.5:
                return 0.4  # Some experience
            else:
                return 0.2  # Underqualified
                
        except Exception:
            return 0.5

    def _calculate_education_relevance(self, job_text: str, resume_text: str) -> float:
        """Calculate education relevance"""
        try:
            # Education keywords
            education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college']
            
            job_edu = any(keyword in job_text for keyword in education_keywords)
            resume_edu = any(keyword in resume_text for keyword in education_keywords)
            
            if job_edu and resume_edu:
                return 0.8  # Both mention education
            elif job_edu or resume_edu:
                return 0.5  # One mentions education
            else:
                return 0.3  # Neither mentions education
                
        except Exception:
            return 0.5

    def _calculate_location_relevance(self, job_text: str, resume_text: str) -> float:
        """Calculate location relevance"""
        try:
            # Location keywords
            location_keywords = ['remote', 'onsite', 'hybrid', 'location', 'city', 'state']
            
            job_loc = any(keyword in job_text for keyword in location_keywords)
            resume_loc = any(keyword in resume_text for keyword in location_keywords)
            
            if job_loc and resume_loc:
                return 0.7  # Both mention location
            elif job_loc or resume_loc:
                return 0.5  # One mentions location
            else:
                return 0.4  # Neither mentions location
                
        except Exception:
            return 0.5
