"""
Complete Production Integration Example
Shows how to use all components together for 90%+ accuracy search.
"""

import logging
from typing import Dict, List, Any
import time

# Import all production components
from app.search.production_skill_ontology import get_production_ontology
from app.search.experience_parser import get_experience_parser
from app.search.enhanced_geocoding import get_enhanced_geocoding
from app.search.hybrid_embedding_service import get_hybrid_embedding_service
from app.search.ranking_feature_extractor import get_feature_extractor
from app.search.three_stage_retrieval import get_retrieval_pipeline, JobRequirements
from app.search.xgboost_ranking_model import get_ranking_model
from app.search.feedback_collector import get_feedback_collector, FeedbackAction
from app.search.evaluation_metrics import get_metrics_evaluator

logger = logging.getLogger(__name__)


def complete_production_search(
    job_description: str,
    job_location: str,
    required_skills: List[str],
    preferred_skills: List[str],
    candidates: List[Dict[str, Any]],
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Complete production search pipeline
    
    This function demonstrates the full integration of all components.
    """
    start_time = time.time()
    
    # Initialize all services
    skill_ontology = get_production_ontology()
    experience_parser = get_experience_parser()
    geocoding = get_enhanced_geocoding()
    embedding_service = get_hybrid_embedding_service()
    feature_extractor = get_feature_extractor()
    retrieval_pipeline = get_retrieval_pipeline()
    ranking_model = get_ranking_model('models/ranking_model.json')  # Load trained model
    
    # Step 1: Process job requirements
    logger.info("Step 1: Processing job requirements...")
    
    # Canonicalize job skills
    required_skill_ids = []
    for skill in required_skills:
        result = skill_ontology.canonicalize_skill(skill)
        if result and result[0]:
            required_skill_ids.append(result[0])
    
    preferred_skill_ids = []
    for skill in preferred_skills or []:
        result = skill_ontology.canonicalize_skill(skill)
        if result and result[0]:
            preferred_skill_ids.append(result[0])
    
    # Geocode job location
    job_location_data = geocoding.geocode_location(job_location, is_remote=False)
    
    # Detect domain
    domain = _detect_domain(job_description)
    
    # Extract experience requirement
    required_experience = _extract_experience_requirement(job_description)
    
    # Create job requirements
    job_requirements = JobRequirements(
        job_id=f"job_{int(time.time())}",
        description=job_description,
        domain=domain,
        required_skill_ids=required_skill_ids,
        preferred_skill_ids=preferred_skill_ids,
        skill_weights={skill_id: 1.0 for skill_id in required_skill_ids},
        required_experience_years=required_experience,
        required_certifications=[],
        required_clearance=None,
        required_visa=None,
        location_data=job_location_data,
        remote_eligible=False,
        timezone_requirement=3,
        seniority_level='mid'
    )
    
    # Step 2: Process candidates (canonicalize skills, geocode, parse experience)
    logger.info(f"Step 2: Processing {len(candidates)} candidates...")
    
    processed_candidates = []
    for candidate in candidates:
        # Canonicalize skills
        raw_skills = candidate.get('skills', [])
        canonicalized = skill_ontology.canonicalize_skill_list(raw_skills)
        candidate['skill_ids'] = [s[0] for s in canonicalized if s[0]]
        candidate['skills_canonical'] = [s[1] for s in canonicalized if s[0]]
        
        # Geocode location
        location_str = candidate.get('location', '')
        is_remote = candidate.get('is_remote', False)
        location_data = geocoding.geocode_location(location_str, is_remote)
        candidate['location_data'] = location_data
        
        # Parse experience
        experience_text = candidate.get('resume_text', '') or candidate.get('experience', '')
        if experience_text:
            structured_experiences = experience_parser.parse_experience(
                experience_text,
                resume_context=candidate
            )
            candidate['experiences'] = [
                {
                    'company': exp.company,
                    'title_normalized': exp.title_normalized,
                    'start_date': exp.start_date.isoformat() if exp.start_date else None,
                    'end_date': exp.end_date.isoformat() if exp.end_date else None,
                    'duration_months': exp.duration_months,
                    'skills': exp.skills,
                    'achievements': exp.achievements,
                    'impact_metrics': [
                        {
                            'type': m['type'],
                            'value': m['value'],
                            'unit': m['unit']
                        } for m in exp.impact_metrics
                    ],
                    'seniority_level': exp.seniority_level
                }
                for exp in structured_experiences
            ]
        
        # Generate embeddings (if not already present)
        if 'bi_encoder_embedding' not in candidate:
            candidate_text = _candidate_to_text(candidate)
            embedding = embedding_service.encode_query(candidate_text, domain)
            if embedding is not None:
                candidate['bi_encoder_embedding'] = embedding.tolist()
        
        processed_candidates.append(candidate)
    
    # Step 3: Three-stage retrieval
    logger.info("Step 3: Running three-stage retrieval pipeline...")
    
    search_results = retrieval_pipeline.search(
        job_requirements=job_requirements,
        candidates=processed_candidates,
        top_k=top_k,
        strict_required_skills=True
    )
    
    # Step 4: Apply XGBoost ranking model (if available)
    if ranking_model and ranking_model.model:
        logger.info("Step 4: Applying XGBoost ranking model...")
        
        for result in search_results['results']:
            features = result.get('features', {})
            if features:
                xgboost_score = ranking_model.predict_score(features)
                # Combine with existing score
                result['matchScore'] = (result['matchScore'] * 0.7 + xgboost_score * 0.3)
                result['xgboost_score'] = xgboost_score
        
        # Re-sort by final score
        search_results['results'].sort(key=lambda x: x.get('matchScore', 0), reverse=True)
    
    # Step 5: Format final results
    final_results = []
    for result in search_results['results']:
        candidate = result['candidate']
        final_results.append({
            'candidate_id': candidate.get('id') or candidate.get('candidate_id'),
            'name': candidate.get('name'),
            'title': candidate.get('title'),
            'matchScore': result['matchScore'],
            'Score': result['matchScore'],
            'dense_similarity': result.get('dense_similarity', 0),
            'cross_encoder_score': result.get('cross_encoder_score', 0),
            'xgboost_score': result.get('xgboost_score', 0),
            'skills': candidate.get('skills_canonical', []),
            'location': candidate.get('location_data', {}).get('standardized_name', ''),
            'experience_years': _calculate_total_experience(candidate.get('experiences', [])),
            'features': result.get('features', {})
        })
    
    total_time = time.time() - start_time
    
    return {
        'results': final_results,
        'total_candidates': len(processed_candidates),
        'candidates_per_stage': search_results.get('candidates_per_stage', {}),
        'stage_timings': search_results.get('stage_timings', {}),
        'total_time': total_time,
        'algorithm_used': 'production-three-stage-hybrid-xgboost'
    }


def record_feedback_and_retrain(
    job_id: str,
    candidate_id: str,
    action: str,
    recruiter_id: Optional[str] = None
):
    """Record feedback and trigger retraining if needed"""
    from app.search.continuous_learning import get_continuous_learning
    
    feedback_collector = get_feedback_collector()
    continuous_learning = get_continuous_learning()
    
    # Record feedback
    feedback = feedback_collector.record_feedback(
        job_id=job_id,
        candidate_id=candidate_id,
        action=action,
        recruiter_id=recruiter_id
    )
    
    # Check if we have enough feedback for retraining
    feedback_count = len(feedback_collector.feedback_cache)
    
    if feedback_count >= 1000 and feedback_count % 100 == 0:
        # Trigger incremental update
        logger.info("Triggering incremental model update...")
        update_results = continuous_learning.incremental_update(days_back=7)
        
        if update_results.get('status') == 'success':
            logger.info(f"Model updated: {update_results['version']}")
            logger.info(f"New metrics: {update_results['metrics']}")
    
    return feedback


# Helper functions

def _detect_domain(text: str) -> str:
    """Detect domain from text"""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ['healthcare', 'medical', 'nursing', 'hospital']):
        return 'healthcare'
    if any(kw in text_lower for kw in ['software', 'developer', 'programming', 'tech']):
        return 'it/tech'
    return 'general'

def _extract_experience_requirement(text: str) -> int:
    """Extract experience requirement"""
    import re
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'(\d+)\+?\s*years?\s*(?:in|with)',
        r'minimum\s*(?:of\s*)?(\d+)\s*years?'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0

def _candidate_to_text(candidate: Dict) -> str:
    """Convert candidate to searchable text"""
    parts = []
    if candidate.get('resume_text'):
        parts.append(candidate['resume_text'])
    if candidate.get('experiences'):
        for exp in candidate['experiences']:
            parts.append(f"{exp.get('title_normalized', '')} at {exp.get('company', '')}")
    if candidate.get('skills_canonical'):
        parts.append(', '.join(candidate['skills_canonical'][:10]))
    return ' '.join(parts)

def _calculate_total_experience(experiences: List[Dict]) -> float:
    """Calculate total experience in years"""
    total_months = sum(exp.get('duration_months', 0) for exp in experiences)
    return round(total_months / 12.0, 1)


# Example usage
if __name__ == "__main__":
    # Example job
    job_description = """
    We are looking for a Senior Software Engineer with 5+ years of experience in React and Node.js.
    Must have experience with AWS and Docker. Preferred: Kubernetes, TypeScript.
    Location: San Francisco, CA. Remote eligible.
    """
    
    # Example candidates (would come from your database)
    candidates = [
        {
            'id': 'candidate_1',
            'name': 'John Doe',
            'skills': ['React', 'JavaScript', 'Node.js', 'AWS'],
            'location': 'San Francisco, CA',
            'resume_text': 'Senior Software Engineer at Tech Corp (2019-present)...',
            'experience': '6 years'
        }
        # ... more candidates
    ]
    
    # Run search
    results = complete_production_search(
        job_description=job_description,
        job_location='San Francisco, CA',
        required_skills=['React', 'Node.js', 'AWS'],
        preferred_skills=['Kubernetes', 'TypeScript'],
        candidates=candidates,
        top_k=20
    )
    
    print(f"Found {len(results['results'])} candidates in {results['total_time']:.2f}s")

