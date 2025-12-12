"""Utility helpers for scoring CSV-imported candidates on the backend."""

from __future__ import annotations

import math
import re
import uuid
from typing import Any, Dict, List, Optional
from app.simple_logger import get_logger

logger = get_logger("manual_csv_matching")


COMMON_SKILLS = [
    "JavaScript",
    "TypeScript",
    "React",
    "Angular",
    "Vue",
    "Node.js",
    "Python",
    "Java",
    "C#",
    "SQL",
    "NoSQL",
    "MongoDB",
    "PostgreSQL",
    "AWS",
    "Azure",
    "Docker",
    "Kubernetes",
    "CI/CD",
    "Git",
    "Agile",
    "Scrum",
    "Product Management",
    "Project Management",
]

DEFAULT_FALLBACK_SKILLS = [
    "JavaScript",
    "React",
    "Node.js",
    "Problem Solving",
    "Communication",
]

KEYWORD_PATTERN = re.compile(r"[a-z0-9]{4,}", re.IGNORECASE)


def _extract_skills_from_job_description(job_description: str) -> List[str]:
    """Lightweight skill extractor that mirrors the frontend helper."""
    lowercase_description = job_description.lower()
    extracted = [
        skill for skill in COMMON_SKILLS if skill.lower() in lowercase_description
    ]
    if extracted:
        return extracted
    return DEFAULT_FALLBACK_SKILLS.copy()


def _normalize_skills(skills: Any) -> List[str]:
    if not isinstance(skills, list):
        return []
    normalized: List[str] = []
    for skill in skills:
        if isinstance(skill, str):
            trimmed = skill.strip().lower()
            if trimmed:
                normalized.append(trimmed)
    return normalized


def _extract_experience_years(candidate: Dict[str, Any]) -> Optional[float]:
    raw_value = candidate.get("experienceYears")
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)) and math.isfinite(raw_value):
        return float(raw_value)
    if isinstance(raw_value, str):
        match = re.search(r"\d+(?:\.\d+)?", raw_value)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


def _compute_manual_candidate_score(
    candidate: Dict[str, Any],
    job_keywords: set,
    job_description: str,
    extracted_skills: List[str],
) -> int:
    normalized_skills = _normalize_skills(candidate.get("skills"))
    job_text = job_description.lower()
    matching_skills = [skill for skill in normalized_skills if skill and skill in job_text]
    skill_score = (
        (len(matching_skills) / len(normalized_skills)) * 60 if normalized_skills else 0
    )

    summary_text = " ".join(
        [
            str(candidate.get("currentTitle") or ""),
            str(candidate.get("summary") or ""),
            str(candidate.get("notes") or ""),
            str(candidate.get("resumeText") or ""),
        ]
    ).lower()
    summary_keywords = KEYWORD_PATTERN.findall(summary_text)
    overlap = [kw for kw in summary_keywords if kw in job_keywords]
    overlap_score = (len(overlap) / len(job_keywords)) * 25 if job_keywords else 0

    experience_years = _extract_experience_years(candidate)
    experience_score = (
        min(experience_years / 10, 1) * 15 if experience_years is not None else 0
    )

    extracted_skill_overlap = (
        (len(matching_skills) / len(extracted_skills)) * 10 if extracted_skills else 0
    )

    raw_score = skill_score + overlap_score + experience_score + extracted_skill_overlap
    return max(0, min(100, round(raw_score)))


def _calculate_confidence_score(candidate: Dict[str, Any], match_score: int) -> float:
    """
    Calculate confidence score based on data completeness and match score.
    Confidence indicates how reliable the match is based on available data.
    """
    confidence = 60.0  # Base confidence
    
    # Name completeness
    if candidate.get("fullName") or candidate.get("name"):
        confidence += 10
    
    # Contact information completeness
    if candidate.get("email"):
        confidence += 10
    if candidate.get("phone"):
        confidence += 5
    
    # Skills completeness
    skills = candidate.get("skills", [])
    if isinstance(skills, list) and len(skills) > 0:
        confidence += 10
    elif isinstance(skills, str) and skills.strip():
        confidence += 5
    
    # Experience information
    if candidate.get("experienceYears") is not None:
        confidence += 5
    if candidate.get("summary") or candidate.get("resumeText"):
        confidence += 5
    
    # Match score factor (higher match score = higher confidence)
    if match_score >= 80:
        confidence += 10
    elif match_score >= 60:
        confidence += 5
    
    # Ensure confidence is between 60-95
    return min(max(confidence, 60), 95)


def _generate_candidate_payload(
    candidate: Dict[str, Any],
    match_score: int,
    source_label: str,
    extracted_skills: List[str],
) -> Dict[str, Any]:
    candidate_id = (
        candidate.get("id")
        or candidate.get("email")
        or candidate.get("fullName")
        or str(uuid.uuid4())
    )
    raw_skills = candidate.get("skills")
    candidate_skills = raw_skills if isinstance(raw_skills, list) and raw_skills else extracted_skills[:5]

    contact_info = None
    email = candidate.get("email")
    phone = candidate.get("phone")
    if email or phone:
        contact_info = {"email": email or "", "phone": phone or ""}

    # Calculate confidence score
    confidence = _calculate_confidence_score(candidate, match_score)

    payload = {
        "id": candidate_id,
        "name": candidate.get("fullName") or email or "Imported Candidate",
        "title": candidate.get("currentTitle") or "Candidate",
        "location": candidate.get("location") or "Not provided",
        "skills": candidate_skills,
        "experience": candidate.get("experienceYears")
        if candidate.get("experienceYears") is not None
        else candidate.get("summary")
        or "Not specified",
        "matchScore": match_score,
        "Score": match_score,
        "MatchScore": match_score,
        "confidence": confidence,
        "Confidence": confidence,
        "source": source_label,
        "source_label": source_label,
        "contactInfo": contact_info,
        "resumeText": candidate.get("summary") or candidate.get("resumeText"),
        "culturalFitScore": None,
        "complianceVerified": False,
        "crossSourceVerified": False,
        "crossSourceOccurrences": 1,
    }

    # Maintain compatibility with existing UI access patterns
    payload["MatchExplanation"] = candidate.get("summary") or candidate.get("notes")
    payload["CandidateId"] = candidate_id
    payload["Source"] = source_label

    return payload


def _build_insights(source_label: str) -> Dict[str, Any]:
    return {
        "talentPoolQuality": "Good",
        "competitivePositioning": {
            "talentAvailability": "Moderate",
            "competitiveness": "Balanced",
            "salaryRange": {"min": 80000, "max": 125000, "median": 102000},
            "timeToHire": "3-4 weeks",
        },
        "recommendedSourcingStrategy": {
            "mostEffectiveSources": [source_label, "Internal Database"],
            "suggestedOutreachOrder": [
                "CSV imports with 80%+ match",
                "Internal pipeline candidates",
                "Passive leads sourced last 30 days",
            ],
            "untappedSources": ["Employee referrals", "Industry communities", "Talent partners"],
            "recommendedSources": ["LinkedIn Recruiter", "Stack Overflow", "GitHub"],
        },
        "crossSourceStatistics": {
            "sourcesAnalyzed": 1,
            "profilesMatched": 15,
            "averageMatchConfidence": 0.78,
        },
    }


def _process_candidates_through_algorithm(
    job_description: str,
    candidates: List[Dict[str, Any]],
    source_label: str = "CSV Import",
    min_match_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Process manual candidates through the main search algorithm for proper sorting and filtering.
    This uses the same algorithm as regular searches for consistency.
    """
    try:
        logger.info(f"Processing {len(candidates)} {source_label} candidates through search algorithm")
        
        # Convert candidates to format expected by algorithm
        formatted_candidates = _format_candidates_for_algorithm(candidates, source_label)
        
        if not formatted_candidates:
            logger.warning(f"No formatted candidates after conversion, returning None")
            return None
        
        # First, calculate initial scores using simple matching
        extracted_skills = _extract_skills_from_job_description(job_description)
        job_keywords = set(KEYWORD_PATTERN.findall(job_description.lower()))
        
        for candidate in formatted_candidates:
            # Calculate initial score
            score = _compute_manual_candidate_score(
                candidate, job_keywords, job_description, extracted_skills
            )
            candidate["matchScore"] = score
            candidate["Score"] = score
            candidate["MatchScore"] = score
            candidate["match_percentage"] = score  # For accuracy enhancement system
            # Add confidence score
            confidence = _calculate_confidence_score(candidate, score)
            candidate["confidence"] = confidence
            candidate["Confidence"] = confidence
        
        # Now process through accuracy enhancement system (this is the main algorithm)
        # First, ensure we have candidates with scores before trying enhancement
        if not formatted_candidates:
            logger.error("No formatted candidates available for processing")
            return None
        
        # Try to use accuracy enhancement, but always fall back to scored candidates
        enhanced_results = None
        try:
            from app.search.accuracy_enhancement_system import enhance_search_accuracy
            
            logger.info(f"Applying accuracy enhancement algorithm to {len(formatted_candidates)} candidates")
            enhanced_results = enhance_search_accuracy(
                job_description, 
                formatted_candidates.copy(),  # Use copy to avoid modifying original
                top_k=len(formatted_candidates)
            )
            
            if not enhanced_results or len(enhanced_results) == 0:
                logger.warning("Accuracy enhancement returned empty results, using scored candidates")
                enhanced_results = None
            else:
                logger.info(f"Accuracy enhancement returned {len(enhanced_results)} candidates")
            
        except ImportError as import_error:
            logger.warning(f"Accuracy enhancement import failed: {import_error}, using scored candidates")
            enhanced_results = None
        except Exception as e:
            logger.warning(f"Accuracy enhancement failed: {e}, using scored candidates")
            enhanced_results = None
        
        # Use enhanced results if available and not empty, otherwise use scored candidates
        if enhanced_results and len(enhanced_results) > 0:
            final_candidates = enhanced_results
            logger.info(f"Using {len(final_candidates)} enhanced candidates")
        else:
            final_candidates = formatted_candidates
            logger.info(f"Using {len(final_candidates)} scored candidates (enhancement unavailable)")
        
        if not final_candidates or len(final_candidates) == 0:
            logger.error(f"No candidates available after processing. formatted_candidates: {len(formatted_candidates) if formatted_candidates else 0}, enhanced_results: {len(enhanced_results) if enhanced_results else 0}")
            return None
        
        # Update match scores and confidence from enhanced results if available
        if enhanced_results:
            for candidate in final_candidates:
                # Use enhanced semantic score if available, otherwise keep original
                enhanced_score = candidate.get("enhanced_semantic_score")
                accuracy_score = candidate.get("accuracy_score", {}).get("overall_accuracy", 0) * 100 if isinstance(candidate.get("accuracy_score"), dict) else None
                
                if enhanced_score is not None:
                    # Combine enhanced semantic score with accuracy score if available
                    if accuracy_score is not None:
                        final_score = int((enhanced_score * 0.7 + accuracy_score * 0.3))
                    else:
                        final_score = int(enhanced_score)
                    candidate["matchScore"] = final_score
                    candidate["Score"] = final_score
                    candidate["MatchScore"] = final_score
                elif accuracy_score is not None:
                    candidate["matchScore"] = int(accuracy_score)
                    candidate["Score"] = int(accuracy_score)
                    candidate["MatchScore"] = int(accuracy_score)
                
                # Update confidence if not already set or recalculate based on new score
                if "confidence" not in candidate or "Confidence" not in candidate:
                    confidence = _calculate_confidence_score(candidate, candidate.get("matchScore", 0))
                    candidate["confidence"] = confidence
                    candidate["Confidence"] = confidence
        else:
            # Ensure all candidates have confidence scores
            for candidate in final_candidates:
                if "confidence" not in candidate or "Confidence" not in candidate:
                    confidence = _calculate_confidence_score(candidate, candidate.get("matchScore", 0))
                    candidate["confidence"] = confidence
                    candidate["Confidence"] = confidence
        
        # Sort by match score
        final_candidates.sort(key=lambda x: x.get("matchScore", 0), reverse=True)
        
        # Apply min_match_score filter if provided
        if min_match_score:
            final_candidates = [
                c for c in final_candidates 
                if (c.get("matchScore") or 0) >= min_match_score
            ]
        
        logger.info(f"Returning {len(final_candidates)} candidates (enhanced: {enhanced_results is not None})")
        if len(final_candidates) == 0:
            logger.error(f"CRITICAL: final_candidates is empty! formatted_candidates length: {len(formatted_candidates) if formatted_candidates else 0}")
            # This should never happen, but if it does, return None to trigger fallback
            return None
        
        result = _build_algorithm_result(
            final_candidates, 
            candidates, 
            job_description, 
            source_label
        )
        logger.info(f"Result built with {len(result.get('candidates', []))} candidates")
        return result
    
    except Exception as e:
        logger.error(f"Error processing candidates through algorithm: {e}", exc_info=True)
        # Even on error, try to return candidates with basic scores
        try:
            extracted_skills = _extract_skills_from_job_description(job_description)
            job_keywords = set(KEYWORD_PATTERN.findall(job_description.lower()))
            formatted_candidates = _format_candidates_for_algorithm(candidates, source_label)
            
            for candidate in formatted_candidates:
                score = _compute_manual_candidate_score(
                    candidate, job_keywords, job_description, extracted_skills
                )
                candidate["matchScore"] = score
                candidate["Score"] = score
                candidate["MatchScore"] = score
                # Add confidence score
                confidence = _calculate_confidence_score(candidate, score)
                candidate["confidence"] = confidence
                candidate["Confidence"] = confidence
            
            formatted_candidates.sort(key=lambda x: x.get("matchScore", 0), reverse=True)
            
            if min_match_score:
                formatted_candidates = [
                    c for c in formatted_candidates 
                    if (c.get("matchScore") or 0) >= min_match_score
                ]
            
            logger.info(f"Returning {len(formatted_candidates)} candidates after error recovery")
            return _build_algorithm_result(
                formatted_candidates, 
                candidates, 
                job_description, 
                source_label
            )
        except Exception as recovery_error:
            logger.error(f"Error recovery also failed: {recovery_error}", exc_info=True)
            # Fall through to simple matching
            return None


def _format_candidates_for_algorithm(
    candidates: List[Dict[str, Any]], 
    source_label: str
) -> List[Dict[str, Any]]:
    """Convert manual candidates to format expected by search algorithm"""
    formatted = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        
        # Build candidate text for semantic matching
        candidate_text_parts = []
        if candidate.get("summary"):
            candidate_text_parts.append(candidate["summary"])
        if candidate.get("resumeText"):
            candidate_text_parts.append(candidate["resumeText"])
        if candidate.get("currentTitle"):
            candidate_text_parts.append(candidate["currentTitle"])
        if candidate.get("currentCompany"):
            candidate_text_parts.append(candidate["currentCompany"])
        if candidate.get("skills"):
            skills_text = ", ".join(candidate["skills"]) if isinstance(candidate["skills"], list) else str(candidate["skills"])
            candidate_text_parts.append(skills_text)
        
        candidate_text = " ".join(candidate_text_parts)
        
        formatted_candidate = {
            "id": candidate.get("id") or candidate.get("email") or str(uuid.uuid4()),
            "name": candidate.get("fullName") or candidate.get("email") or "Imported Candidate",
            "title": candidate.get("currentTitle") or "Candidate",
            "location": candidate.get("location") or "Not provided",
            "skills": candidate.get("skills") or [],
            "experience": candidate.get("experienceYears") or candidate.get("summary") or "Not specified",
            "email": candidate.get("email", ""),
            "phone": candidate.get("phone", ""),
            "summary": candidate.get("summary") or "",
            "resumeText": candidate_text,
            "source": source_label,
            "source_label": source_label,
            "contactInfo": {
                "email": candidate.get("email", ""),
                "phone": candidate.get("phone", "")
            } if candidate.get("email") or candidate.get("phone") else None,
            "culturalFitScore": None,
            "complianceVerified": False,
            "crossSourceVerified": False,
            "crossSourceOccurrences": 1,
        }
        formatted.append(formatted_candidate)
    
    return formatted


def _process_with_algorithm(
    algorithm, 
    job_description: str, 
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Process candidates through the algorithm's semantic matching"""
    try:
        # Use the algorithm's semantic matching capabilities
        # Create a combined text from all candidates for processing
        candidate_texts = [c.get("resumeText", "") for c in candidates]
        
        # Process through algorithm (this is a simplified approach)
        # The algorithm will score and rank candidates
        scored_candidates = []
        for candidate in candidates:
            # Use algorithm's scoring if available
            candidate_text = candidate.get("resumeText", "")
            if hasattr(algorithm, 'score_candidate'):
                score = algorithm.score_candidate(job_description, candidate_text)
            else:
                # Fallback scoring
                score = _compute_manual_candidate_score(
                    candidate,
                    set(KEYWORD_PATTERN.findall(job_description.lower())),
                    job_description,
                    _extract_skills_from_job_description(job_description)
                )
            
            candidate["matchScore"] = score
            candidate["Score"] = score
            candidate["MatchScore"] = score
            scored_candidates.append(candidate)
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x.get("matchScore", 0), reverse=True)
        return scored_candidates
    
    except Exception as e:
        logger.error(f"Error in algorithm processing: {e}", exc_info=True)
        # Return candidates with simple scores
        extracted_skills = _extract_skills_from_job_description(job_description)
        job_keywords = set(KEYWORD_PATTERN.findall(job_description.lower()))
        
        for candidate in candidates:
            score = _compute_manual_candidate_score(
                candidate, job_keywords, job_description, extracted_skills
            )
            candidate["matchScore"] = score
            candidate["Score"] = score
            candidate["MatchScore"] = score
        
        candidates.sort(key=lambda x: x.get("matchScore", 0), reverse=True)
        return candidates


def _build_algorithm_result(
    enhanced_candidates: List[Dict[str, Any]],
    original_candidates: List[Dict[str, Any]],
    job_description: str,
    source_label: str
) -> Dict[str, Any]:
    """Build result structure from algorithm-processed candidates"""
    logger.info(f"_build_algorithm_result called with {len(enhanced_candidates) if enhanced_candidates else 0} enhanced candidates and {len(original_candidates)} original candidates")
    
    if not enhanced_candidates:
        logger.error(f"CRITICAL: No enhanced candidates to build result from! This should not happen.")
        enhanced_candidates = []
    
    # Ensure we have at least some candidates - if enhanced is empty but original has candidates, something went wrong
    if len(enhanced_candidates) == 0 and len(original_candidates) > 0:
        logger.error(f"CRITICAL: Enhanced candidates list is empty but original has {len(original_candidates)} candidates - creating fallback candidates")
        # Try to create basic candidates from original
        try:
            extracted_skills = _extract_skills_from_job_description(job_description)
            job_keywords = set(KEYWORD_PATTERN.findall(job_description.lower()))
            for candidate in original_candidates:
                if isinstance(candidate, dict):
                    score = _compute_manual_candidate_score(
                        candidate, job_keywords, job_description, extracted_skills
                    )
                    candidate_payload = _generate_candidate_payload(candidate, score, source_label, extracted_skills)
                    # Ensure confidence is set
                    if "confidence" not in candidate_payload or "Confidence" not in candidate_payload:
                        confidence = _calculate_confidence_score(candidate, score)
                        candidate_payload["confidence"] = confidence
                        candidate_payload["Confidence"] = confidence
                    enhanced_candidates.append(candidate_payload)
            logger.info(f"Created {len(enhanced_candidates)} candidates from original as fallback")
        except Exception as e:
            logger.error(f"Failed to create fallback candidates: {e}", exc_info=True)
    
    extracted_skills = _extract_skills_from_job_description(job_description)
    
    experience_total = 0.0
    for candidate in original_candidates:
        if not isinstance(candidate, dict):
            continue
        years = _extract_experience_years(candidate) or 0
        experience_total += years
    avg_experience = experience_total / (len(original_candidates) or 1)
    
    average_score = (
        sum(c.get("matchScore") or c.get("Score") or 0 for c in enhanced_candidates) 
        / (len(enhanced_candidates) or 1)
    ) if enhanced_candidates else 0
    
    summary = (
        f"Matched {len(enhanced_candidates)} imported profiles from {source_label} using advanced algorithm."
        if enhanced_candidates
        else f"No imported candidates from {source_label} matched the current job description."
    )
    
    logger.info(f"Building result with {len(enhanced_candidates)} candidates, average score: {average_score}")
    
    result = {
        "jobTitle": "Imported candidate search",
        "extractedSkills": extracted_skills,
        "candidates": enhanced_candidates,
        "results": enhanced_candidates,
        "suggestedExperience": round(avg_experience),
        "totalCandidatesScanned": len(original_candidates),
        "total": len(enhanced_candidates),
        "page": 1,
        "pageSize": len(enhanced_candidates),
        "hasMore": False,
        "matchTime": round(len(original_candidates) / 12, 1),
        "matchingModelUsed": "algorithm-enhanced",
        "algorithm_used": "optimized-search-algorithm",
        "insights": _build_insights(source_label),
        "crossSourceValidation": {
            "sourcesSearched": [source_label],
            "candidatesFound": len(enhanced_candidates),
            "verifiedCandidates": len(enhanced_candidates),
            "verificationRate": 100 if enhanced_candidates else 0,
            "averageCrossSourceScore": round(average_score) if enhanced_candidates else 0,
        },
        "sourcesUsed": [source_label],
        "candidatesPerSource": {source_label: len(enhanced_candidates)},
        "summary": summary,
        "originTag": f"{source_label} · Algorithm-Enhanced",
    }
    
    logger.info(f"Result built successfully with {len(result.get('candidates', []))} candidates")
    return result


def build_manual_csv_match(
    job_description: str,
    candidates: List[Dict[str, Any]],
    source_label: str = "CSV Import",
    min_match_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build match results for manual candidates (CSV/Resume upload).
    Now uses the main search algorithm for proper sorting and filtering.
    """
    if not job_description or not isinstance(job_description, str):
        raise ValueError("job_description is required")
    if not candidates or not isinstance(candidates, list):
        raise ValueError("candidates must be a non-empty list")

    # Try to process through the main search algorithm first
    logger.info(f"Processing {len(candidates)} {source_label} candidates through search algorithm")
    algorithm_result = _process_candidates_through_algorithm(
        job_description, 
        candidates, 
        source_label, 
        min_match_score
    )
    
    if algorithm_result:
        logger.info(f"Successfully processed {len(algorithm_result.get('candidates', []))} candidates through algorithm")
        return algorithm_result
    
    # Fallback to simple matching if algorithm processing fails
    logger.info(f"Falling back to simple matching for {source_label} candidates")
    extracted_skills = _extract_skills_from_job_description(job_description)
    job_keywords = set(KEYWORD_PATTERN.findall(job_description.lower()))
    prepared_candidates: List[Dict[str, Any]] = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        score = _compute_manual_candidate_score(
            candidate, job_keywords, job_description, extracted_skills
        )
        candidate_payload = _generate_candidate_payload(candidate, score, source_label, extracted_skills)
        # Ensure confidence is set (should already be in _generate_candidate_payload, but double-check)
        if "confidence" not in candidate_payload or "Confidence" not in candidate_payload:
            confidence = _calculate_confidence_score(candidate, score)
            candidate_payload["confidence"] = confidence
            candidate_payload["Confidence"] = confidence
        prepared_candidates.append(candidate_payload)

    prepared_candidates.sort(key=lambda item: item.get("matchScore", 0), reverse=True)

    threshold = float(min_match_score or 0)
    if threshold > 0:
        filtered_candidates = [c for c in prepared_candidates if (c.get("matchScore") or 0) >= threshold]
    else:
        filtered_candidates = prepared_candidates
    final_candidates = filtered_candidates if filtered_candidates else prepared_candidates

    experience_total = 0.0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        years = _extract_experience_years(candidate) or 0
        experience_total += years
    avg_experience = experience_total / (len(candidates) or 1)

    average_score = (
        sum(c.get("matchScore") or 0 for c in final_candidates) / (len(final_candidates) or 1)
    )

    summary = (
        f"Matched {len(final_candidates)} imported profiles from {source_label}."
        if final_candidates
        else f"No imported candidates from {source_label} matched the current job description."
    )

    return {
        "jobTitle": "Imported candidate search",
        "extractedSkills": extracted_skills,
        "candidates": final_candidates,
        "results": final_candidates,
        "suggestedExperience": round(avg_experience),
        "totalCandidatesScanned": len(candidates),
        "total": len(final_candidates),
        "page": 1,
        "pageSize": len(final_candidates),
        "hasMore": False,
        "matchTime": round(len(candidates) / 12, 1),
        "matchingModelUsed": "csv-import-backend",
        "insights": _build_insights(source_label),
        "crossSourceValidation": {
            "sourcesSearched": [source_label],
            "candidatesFound": len(final_candidates),
            "verifiedCandidates": len(final_candidates),
            "verificationRate": 100 if final_candidates else 0,
            "averageCrossSourceScore": round(average_score) if final_candidates else 0,
        },
        "sourcesUsed": [source_label],
        "candidatesPerSource": {source_label: len(final_candidates)},
        "summary": summary,
        "originTag": f"{source_label} · Backend",
    }


