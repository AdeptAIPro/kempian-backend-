"""
API routes for the matchmaking system.
Provides endpoints that integrate with the frontend.
"""

import logging
from flask import Blueprint, request, jsonify
from typing import List, Dict, Any, Optional

from app.simple_logger import get_logger
from app.models import User, db
import jwt

from .pipelines.matcher import match_candidates, CandidateJobMatcher
from .collectors.resume_parser import ResumeParser
from .collectors.job_parser import JobParser

logger = get_logger("matchmaking")

matchmaking_bp = Blueprint('matchmaking', __name__)


def get_jwt_payload():
    """Extract JWT payload from request."""
    auth = request.headers.get('Authorization', None)
    if not auth:
        return None
    
    if not auth.startswith('Bearer '):
        return None
    
    try:
        token = auth.split(' ')[1]
        if not token or len(token.split('.')) != 3:
            return None
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload
    except Exception as e:
        logger.error(f"Error decoding JWT: {e}")
        return None


def transform_match_result_to_frontend_format(
    match_result: Dict[str, Any],
    original_candidate: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Transform matchmaking result to frontend-expected format.
    Comprehensively extracts and maps all candidate fields from database.
    
    Args:
        match_result: Result from matchmaking system
        original_candidate: Original candidate data from database (can be empty dict)
    
    Returns:
        Transformed candidate dictionary with all fields properly mapped
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # CRITICAL: Handle empty original_candidate - try to extract from match_result
    if not original_candidate or len(original_candidate) < 3:
        logger.warning(f"Original candidate is empty or minimal (has {len(original_candidate) if original_candidate else 0} fields). Attempting to extract from match_result")
        # Try to get candidate data from match_result if it exists (this is stored in MatchResult.candidate_data)
        if 'candidate_data' in match_result and match_result['candidate_data']:
            original_candidate = match_result['candidate_data']
            logger.info(f"Found candidate_data in match_result with {len(original_candidate)} fields")
        elif 'candidate' in match_result and match_result['candidate']:
            original_candidate = match_result['candidate']
            logger.info(f"Found candidate in match_result with {len(original_candidate)} fields")
        # If match_result itself has candidate fields, use those
        elif any(key in match_result for key in ['email', 'FullName', 'full_name', 'name', 'skills']):
            original_candidate = match_result
            logger.info(f"Using match_result directly as it contains candidate fields")
        else:
            logger.error(f"Could not find candidate data in match_result. Available keys: {list(match_result.keys())}")
    # Extract match data
    score = match_result.get('score', 0.0)
    matched_skills = match_result.get('matched_skills', [])
    missing_skills = match_result.get('missing_skills', [])
    explanation = match_result.get('explanation', '')
    details = match_result.get('details', {})
    
    # Calculate match percentage (0-100)
    match_percentage = int(score * 100)
    
    # Determine grade based on score
    if match_percentage >= 80:
        grade = 'A'
    elif match_percentage >= 70:
        grade = 'B'
    elif match_percentage >= 60:
        grade = 'C'
    elif match_percentage >= 50:
        grade = 'D'
    else:
        grade = 'F'
    
    # Helper function to get value from multiple possible field names
    # Also checks match_result as fallback
    def get_field(*field_names, default=None):
        # First try original_candidate
        for field in field_names:
            value = original_candidate.get(field) if original_candidate else None
            if value is not None:
                # Handle different types
                if isinstance(value, (list, dict)):
                    if value:  # Non-empty list/dict
                        return value
                elif isinstance(value, (int, float)):
                    return value
                elif isinstance(value, str):
                    stripped = value.strip()
                    if stripped and stripped.lower() not in ('unknown', 'n/a', 'not provided', 'none', 'null', ''):
                        return stripped
                else:
                    return value
        
        # Fallback: try match_result if original_candidate didn't have it
        if not original_candidate or len(original_candidate) < 3:
            for field in field_names:
                value = match_result.get(field)
                if value is not None:
                    if isinstance(value, (list, dict)):
                        if value:
                            return value
                    elif isinstance(value, (int, float)):
                        return value
                    elif isinstance(value, str):
                        stripped = value.strip()
                        if stripped and stripped.lower() not in ('unknown', 'n/a', 'not provided', 'none', 'null', ''):
                            return stripped
                    else:
                        return value
        
        return default
    
    # Extract all candidate fields comprehensively
    email = get_field('email', 'Email', 'email_address', 'e_mail', 
                     default=original_candidate.get('contactInfo', {}).get('email') if isinstance(original_candidate.get('contactInfo'), dict) else None)
    
    phone = get_field('phone', 'Phone', 'phone_number', 'phoneNumber', 'mobile', 'Mobile',
                     default=original_candidate.get('contactInfo', {}).get('phone') if isinstance(original_candidate.get('contactInfo'), dict) else None)
    
    full_name = get_field('full_name', 'FullName', 'fullName', 'name', 'candidate_name', 'candidateName', default='')
    
    location = get_field('location', 'Location', 'current_location', 'city', 'address', 'Address', default='')
    
    # Extract skills - handle both array and string formats
    skills = original_candidate.get('skills') or original_candidate.get('Skills') or []
    if isinstance(skills, str):
        import re
        skills = [s.strip() for s in re.split(r'[,;\n]', skills) if s.strip()]
    elif not isinstance(skills, list):
        skills = []
    
    # Extract experience
    experience = get_field('experience', 'Experience', 'experience_years', 'total_experience', 'years_of_experience', default='')
    experience_years = original_candidate.get('experience_years') or original_candidate.get('total_experience_years')
    if experience_years is None and experience:
        # Try to parse from experience string
        import re
        exp_match = re.search(r'(\d+(?:\.\d+)?)', str(experience))
        if exp_match:
            try:
                experience_years = float(exp_match.group(1))
            except:
                experience_years = None
    
    # Extract education
    education = get_field('education', 'Education', 'educational_background', 'degree', 'highest_education', default='')
    
    # Extract certifications
    certifications = original_candidate.get('certifications') or original_candidate.get('Certifications') or []
    if isinstance(certifications, str):
        import re
        certifications = [c.strip() for c in re.split(r'[,;\n]', certifications) if c.strip()]
    elif not isinstance(certifications, list):
        certifications = []
    
    # Extract resume text
    resume_text = get_field('resume_text', 'resumeText', 'ResumeText', 'resume', 'Resume', 
                           'summary', 'Summary', 'profile_summary', 'bio', 'description', default='')
    
    # Extract source URL
    source_url = get_field('sourceUrl', 'sourceURL', 'source_url', 'url', 'linkedin_url', 'linkedInUrl', default='')
    
    # Extract other fields
    category = get_field('category', 'domain_tag', 'domain', 'Category', 'Domain', default='')
    
    # Build comprehensive transformed object
    transformed = {
        # Preserve ALL original candidate fields first
        **original_candidate,
        
        # Match scoring fields (frontend expects these)
        'Score': match_percentage,
        'score': match_percentage,
        'matchScore': match_percentage,
        'match_percentage': match_percentage,
        'Grade': grade,
        'grade': grade,
        
        # Match explanation
        'MatchExplanation': explanation,
        'match_explanation': explanation,
        'LLM_Reasoning': explanation,
        
        # Name fields (all variations)
        'FullName': full_name,
        'full_name': full_name,
        'name': full_name,
        
        # Contact fields (all variations)
        'email': email or '',
        'Email': email or '',
        'phone': phone or '',
        'Phone': phone or '',
        'contactInfo': {
            'email': email or '',
            'phone': phone or ''
        },
        
        # Location fields
        'location': location or '',
        'Location': location or '',
        
        # Skills (all variations)
        'skills': skills,
        'Skills': skills,
        
        # Experience fields
        'experience': experience or '',
        'Experience': experience or '',
        'experience_years': experience_years,
        'total_experience_years': experience_years,
        
        # Education fields
        'education': education or '',
        'Education': education or '',
        
        # Certifications
        'certifications': certifications,
        'Certifications': certifications,
        
        # Resume text
        'resumeText': resume_text,
        'resume_text': resume_text,
        
        # Source fields
        'sourceUrl': source_url,
        'sourceURL': source_url,
        'source_url': source_url,
        
        # Category/Domain fields
        'category': category or '',
        'domain_tag': category or '',
        'domain': category or '',
        'Category': category or '',
        'Domain': category or '',
        
        # Match-specific fields
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'similarity_breakdown': {
            'matching_skills': matched_skills,
            'missing_skills': missing_skills,
            'total_job_skills': len(matched_skills) + len(missing_skills),
            'candidate_skill_count': len(skills),
            'match_count': len(matched_skills)
        },
        
        # Score breakdown
        'score_breakdown': {
            'skill_score': round(details.get('skill_score', 0.0) * 100, 2),
            'experience_score': round(details.get('experience_score', 0.0) * 100, 2),
            'semantic_score': round(details.get('semantic_score', 0.0) * 100, 2),
            'additional_score': round(details.get('additional_score', 0.0) * 100, 2),
            'overall_score': match_percentage
        },
        
        # Additional fields that frontend might use
        'id': original_candidate.get('id') or original_candidate.get('candidate_id') or original_candidate.get('_id') or '',
        'candidate_id': original_candidate.get('candidate_id') or original_candidate.get('id') or original_candidate.get('_id') or '',
    }
    
    return transformed


@matchmaking_bp.route('/match', methods=['POST'])
def match():
    """
    Match candidates to a job description using the new matchmaking system.
    
    Request body:
    {
        "job_description": "Job description text",
        "top_k": 20  # Optional, default 20
    }
    
    Response format matches the existing /search endpoint for frontend compatibility.
    """
    try:
        # Authentication
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Get request data
        data = request.get_json() or {}
        job_description = data.get('job_description') or data.get('query')
        
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        top_k = data.get('top_k', 20)
        if not isinstance(top_k, int) or top_k < 1:
            top_k = 20
        
        logger.info(f"Matchmaking request: top_k={top_k}")
        
        # Load candidates from request or use existing search system to load from database
        candidates = data.get('candidates')
        if not candidates:
            # If no candidates provided, use existing search system to load candidates
            try:
                from app.search.service import semantic_match
                # Get candidates using existing system (handles DynamoDB, caching, etc.)
                search_results = semantic_match(job_description, top_k=100)
                candidates = search_results.get('results', [])
                if not candidates:
                    return jsonify({
                        'error': 'No candidates found in database',
                        'results': [],
                        'total_candidates': 0
                    }), 404
                logger.info(f"Loaded {len(candidates)} candidates from database")
            except Exception as e:
                logger.error(f"Failed to load candidates from database: {e}")
                return jsonify({
                    'error': 'Failed to load candidates. Please provide candidates array in request.',
                    'results': [],
                    'total_candidates': 0
                }), 500
        
        # Run matchmaking
        match_results = match_candidates(job_description, candidates, top_k=top_k)
        
        # Transform results to frontend format
        transformed_results = []
        for match_result in match_results:
            # Find original candidate data
            candidate_id = match_result['candidate_id']
            original_candidate = next(
                (c for c in candidates if str(c.get('candidate_id') or c.get('id') or c.get('_id')) == candidate_id),
                {}
            )
            
            # Transform to frontend format
            transformed = transform_match_result_to_frontend_format(match_result, original_candidate)
            transformed_results.append(transformed)
        
        # Return in same format as /search endpoint
        return jsonify({
            'results': transformed_results,
            'total_candidates': len(transformed_results),
            'algorithm_used': 'matchmaking_system_v1'
        })
        
    except Exception as e:
        logger.error(f"Error in matchmaking endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Matchmaking failed. Please try again.'}), 500


@matchmaking_bp.route('/match-csv', methods=['POST'])
def match_csv():
    """
    Match CSV-imported candidates to a job description.
    Compatible with existing /search/csv-match endpoint.
    """
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json() or {}
        job_description = data.get('job_description') or data.get('query')
        
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        candidates = data.get('candidates')
        if not isinstance(candidates, list) or not candidates:
            return jsonify({'error': 'Candidates array is required'}), 400
        
        # Run matchmaking
        match_results = match_candidates(job_description, candidates)
        
        # Transform results
        transformed_results = []
        for match_result in match_results:
            candidate_id = match_result['candidate_id']
            original_candidate = next(
                (c for c in candidates if str(c.get('candidate_id') or c.get('id') or c.get('_id')) == candidate_id),
                {}
            )
            transformed = transform_match_result_to_frontend_format(match_result, original_candidate)
            transformed_results.append(transformed)
        
        return jsonify({
            'results': transformed_results,
            'total_candidates': len(transformed_results)
        })
        
    except Exception as e:
        logger.error(f"Error in CSV matchmaking: {e}", exc_info=True)
        return jsonify({'error': 'Matchmaking failed'}), 500

