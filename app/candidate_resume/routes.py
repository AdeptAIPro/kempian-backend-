"""
Candidate Resume Parsing Routes
Handles parsing of resume files and converting them to ManualCandidateRecord format
"""
import os
import re
import tempfile
import uuid
from datetime import datetime

import boto3
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from app.simple_logger import get_logger
from app.services.resume_parser import ResumeParser
from app.talent.routes import s3_client, S3_BUCKET

logger = get_logger("candidate_resume")

candidate_resume_bp = Blueprint('candidate_resume', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'rtf'}
MAX_FILES_PER_UPLOAD = int(os.getenv('MAX_FILES_PER_UPLOAD', 200))  # Default: 200 files

resume_parser = ResumeParser()

# DynamoDB configuration for storing parsed candidates
DYNAMODB_REGION = os.getenv('AWS_REGION', 'ap-south-1')
DYNAMODB_TABLE_NAME = os.getenv('DYNAMODB_TABLE_NAME', 'resume_metadata')

try:
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=DYNAMODB_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        resume_table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        logger.info(f"[DYNAMODB] Connected to table '{DYNAMODB_TABLE_NAME}' for storing parsed resumes")
    else:
        dynamodb = None
        resume_table = None
        logger.warning("[DYNAMODB] AWS credentials not found; parsed resumes will not be persisted to DynamoDB")
except Exception as e:
    dynamodb = None
    resume_table = None
    logger.warning(f"[DYNAMODB] Could not initialize DynamoDB in candidate_resume routes: {e}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_parsed_data_to_manual_candidate(parsed_data: dict, filename: str = "") -> dict:
    """
    Convert parsed resume data to ManualCandidateRecord format
    """
    logger.info(f"[DEBUG convert_parsed_data] Starting conversion for file: {filename}")
    logger.info(f"[DEBUG convert_parsed_data] Input parsed_data keys: {list(parsed_data.keys()) if parsed_data else 'None'}")
    
    # Extract current title and company from work experience
    current_title = None
    current_company = None
    
    work_experience = parsed_data.get('work_experience', '')
    logger.info(f"[DEBUG convert_parsed_data] work_experience type: {type(work_experience)}, length: {len(str(work_experience))}")
    if work_experience:
        # Try to extract the most recent position
        lines = work_experience.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) > 5:
                # Look for common patterns like "Title at Company" or "Title - Company"
                if ' at ' in line.lower() or ' - ' in line or ' | ' in line:
                    parts = re.split(r'\s+(?:at|@|-|\|)\s+', line, flags=re.IGNORECASE)
                    if len(parts) >= 2:
                        current_title = parts[0].strip()
                        current_company = parts[1].strip()
                        break
                elif not current_title and len(line) > 3:
                    # If no pattern found, use first substantial line as title
                    current_title = line[:100]  # Limit length
    
    # Extract skills array
    skills = parsed_data.get('skills', [])
    logger.info(f"[DEBUG convert_parsed_data] skills type: {type(skills)}, value: {skills}")
    if isinstance(skills, str):
        # If skills is a string, split it
        skills = [s.strip() for s in skills.split(',') if s.strip()]
        logger.info(f"[DEBUG convert_parsed_data] Converted skills string to list: {len(skills)} skills")
    elif not isinstance(skills, list):
        logger.warning(f"[DEBUG convert_parsed_data] skills is not a list or string, type: {type(skills)}, converting to empty list")
        skills = []
    else:
        logger.info(f"[DEBUG convert_parsed_data] skills is already a list with {len(skills)} items")
    
    # Extract experience years
    experience_years = parsed_data.get('experience_years')
    if experience_years:
        try:
            experience_years = int(experience_years)
        except (ValueError, TypeError):
            experience_years = None
    
    # Build summary from available data
    summary_parts = []
    if parsed_data.get('summary'):
        summary_parts.append(parsed_data['summary'])
    if work_experience:
        summary_parts.append(f"Experience: {work_experience[:200]}")
    if parsed_data.get('education'):
        summary_parts.append(f"Education: {parsed_data['education'][:200]}")
    
    summary = ' | '.join(summary_parts) if summary_parts else None
    
    # Build resume text from raw text or work experience
    resume_text = parsed_data.get('raw_text') or parsed_data.get('work_experience') or summary
    logger.info(f"[DEBUG convert_parsed_data] resume_text length: {len(str(resume_text))}")
    
    # Create ManualCandidateRecord
    candidate = {
        'id': str(uuid.uuid4()),
        'fullName': parsed_data.get('full_name') or parsed_data.get('name') or 'Unknown Candidate',
        'email': parsed_data.get('email') or None,
        'phone': parsed_data.get('phone') or None,
        'location': parsed_data.get('location') or None,
        'currentTitle': current_title or None,
        'currentCompany': current_company or None,
        'skills': skills,
        'experienceYears': experience_years,
        'summary': summary,
        'resumeText': resume_text,
        'source': 'Resume Upload',
        'raw': {
            'filename': filename,
            'education': parsed_data.get('education'),
            'work_experience': work_experience,
            'projects': parsed_data.get('projects'),
            'certifications': parsed_data.get('certifications'),
        }
    }
    
    logger.info(f"[DEBUG convert_parsed_data] Final candidate keys: {list(candidate.keys())}")
    logger.info(f"[DEBUG convert_parsed_data] Candidate has data: name={bool(candidate.get('fullName'))}, email={bool(candidate.get('email'))}, skills={len(candidate.get('skills', []))}")
    
    return candidate


def _map_manual_candidate_to_dynamodb_item(candidate: dict) -> dict:
    """
    Map ManualCandidateRecord structure to the generic DynamoDB candidate schema
    used by resume_metadata. This ensures new uploads participate in search.
    """
    now_iso = datetime.utcnow().isoformat()

    raw = candidate.get('raw', {}) or {}

    full_name = candidate.get('fullName') or raw.get('full_name') or 'Unknown Candidate'
    email = candidate.get('email') or raw.get('email')
    phone = candidate.get('phone') or raw.get('phone')
    location = candidate.get('location') or raw.get('location')

    skills = candidate.get('skills') or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',') if s.strip()]

    experience_years = candidate.get('experienceYears')

    resume_text = candidate.get('resumeText') or candidate.get('summary') or raw.get('resume_text')

    # Resume storage metadata (optional, especially for S3-based uploads)
    resume_s3_key = raw.get('resume_s3_key') or raw.get('s3_key')
    resume_url = raw.get('resume_url') or raw.get('s3_url')

    # As a fallback, if we know the S3 key but not the URL, construct an s3:// URL
    if not resume_url and resume_s3_key and S3_BUCKET:
        resume_url = f"s3://{S3_BUCKET}/{resume_s3_key}"

    item = {
        # Primary identifier
        'candidate_id': candidate.get('id') or str(uuid.uuid4()),
        # Core identity fields
        'full_name': full_name,
        'FullName': full_name,  # Some parts of the system expect this casing
        'email': email,
        'phone': phone,
        'location': location,
        # Experience & skills
        'skills': skills,
        'total_experience_years': experience_years,
        'experience_years': experience_years,
        # Summary / resume text
        'summary': candidate.get('summary'),
        'resume_text': resume_text,
        # Current role
        'current_position': candidate.get('currentTitle'),
        'current_company': candidate.get('currentCompany'),
        # File & source metadata
        'filename': raw.get('filename'),
        'source': candidate.get('source', 'Resume Upload'),
        'resume_s3_key': resume_s3_key,
        'resume_url': resume_url,
        # Rich fields for downstream use
        'education': raw.get('education'),
        'work_experience': raw.get('work_experience'),
        'projects': raw.get('projects'),
        'certifications': raw.get('certifications'),
        # Audit
        'created_at': now_iso,
        'updated_at': now_iso,
    }

    # Remove keys with value None to keep items compact
    return {k: v for k, v in item.items() if v is not None}


def _store_candidates_in_dynamodb(candidates: list) -> None:
    """Persist parsed candidates into DynamoDB so they participate in search."""
    if not candidates:
        return

    if not resume_table:
        logger.warning("[DYNAMODB] resume_table is not configured; skipping persistence of parsed candidates")
        return

    stored = 0
    errors = 0

    for candidate in candidates:
        try:
            item = _map_manual_candidate_to_dynamodb_item(candidate)
            resume_table.put_item(Item=item)
            stored += 1
        except Exception as e:
            errors += 1
            logger.error(f"[DYNAMODB] Failed to store parsed candidate '{candidate.get('fullName')}' in DynamoDB: {e}", exc_info=True)

    logger.info(f"[DYNAMODB] Stored {stored} parsed candidate(s) in DynamoDB (errors: {errors})")

@candidate_resume_bp.route('/api/candidate-resume/parse', methods=['POST'])
def parse_resumes():
    """
    Parse resume files and return candidates in ManualCandidateRecord format
    Accepts multiple files in a single request
    """
    try:
        # Check if files are present
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # Check file count limit
        if len(files) > MAX_FILES_PER_UPLOAD:
            return jsonify({
                'success': False,
                'error': f'Too many files. Maximum {MAX_FILES_PER_UPLOAD} files allowed per upload. You selected {len(files)} files.'
            }), 400
        
        # Process each file
        candidates = []
        failed_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                failed_files.append({
                    'filename': file.filename,
                    'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
                })
                continue
            
            try:
                # Save file to temporary location
                file_extension = file.filename.rsplit('.', 1)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                    file.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                try:
                    # Parse the resume
                    logger.info(f"Parsing resume: {file.filename}")
                    parsed_data = resume_parser.parse_file(tmp_path)
                    
                    # DEBUG: Log what was parsed
                    logger.info(f"[DEBUG] Parsed data type: {type(parsed_data)}, is None: {parsed_data is None}, is empty dict: {parsed_data == {}}")
                    logger.info(f"[DEBUG] Parsed data keys: {list(parsed_data.keys()) if parsed_data else 'None'}")
                    
                    if parsed_data:
                        logger.info(f"[DEBUG] Parsed data summary:")
                        logger.info(f"  - full_name: {parsed_data.get('full_name', 'NOT FOUND')}")
                        logger.info(f"  - email: {parsed_data.get('email', 'NOT FOUND')}")
                        logger.info(f"  - phone: {parsed_data.get('phone', 'NOT FOUND')}")
                        logger.info(f"  - location: {parsed_data.get('location', 'NOT FOUND')}")
                        logger.info(f"  - skills: {len(parsed_data.get('skills', []))} skills found")
                        logger.info(f"  - experience_years: {parsed_data.get('experience_years', 'NOT FOUND')}")
                        logger.info(f"  - education: {'FOUND' if parsed_data.get('education') else 'NOT FOUND'} ({len(str(parsed_data.get('education', '')))} chars)")
                        logger.info(f"  - work_experience: {'FOUND' if parsed_data.get('work_experience') else 'NOT FOUND'} ({len(str(parsed_data.get('work_experience', '')))} chars)")
                        logger.info(f"  - projects: {'FOUND' if parsed_data.get('projects') else 'NOT FOUND'} ({len(str(parsed_data.get('projects', '')))} chars)")
                        logger.info(f"  - certifications: {'FOUND' if parsed_data.get('certifications') else 'NOT FOUND'} ({len(str(parsed_data.get('certifications', '')))} chars)")
                        logger.info(f"  - summary: {'FOUND' if parsed_data.get('summary') else 'NOT FOUND'} ({len(str(parsed_data.get('summary', '')))} chars)")
                        logger.info(f"  - raw_text: {'FOUND' if parsed_data.get('raw_text') else 'NOT FOUND'} ({len(str(parsed_data.get('raw_text', '')))} chars)")
                    
                    # Check if parsed_data is None or empty dict
                    if not parsed_data or parsed_data == {}:
                        logger.warning(f"[DEBUG] parsed_data is empty or None for {file.filename}")
                        failed_files.append({
                            'filename': file.filename,
                            'error': 'Failed to extract data from resume - parsed_data is empty'
                        })
                        continue
                    
                    # Check if we have at least some meaningful data
                    has_meaningful_data = any([
                        parsed_data.get('full_name'),
                        parsed_data.get('email'),
                        parsed_data.get('phone'),
                        parsed_data.get('location'),
                        len(parsed_data.get('skills', [])) > 0,
                        parsed_data.get('education'),
                        parsed_data.get('work_experience'),
                        parsed_data.get('projects'),
                        parsed_data.get('certifications'),
                        parsed_data.get('raw_text')
                    ])
                    
                    if not has_meaningful_data:
                        logger.warning(f"[DEBUG] parsed_data has no meaningful data for {file.filename}")
                        failed_files.append({
                            'filename': file.filename,
                            'error': 'Failed to extract meaningful data from resume - all fields are empty'
                        })
                        continue
                    
                    # Convert to ManualCandidateRecord format
                    logger.info(f"[DEBUG] Converting parsed data to candidate format for {file.filename}")
                    candidate = convert_parsed_data_to_manual_candidate(parsed_data, file.filename)
                    
                    # DEBUG: Log what was converted
                    logger.info(f"[DEBUG] Converted candidate keys: {list(candidate.keys())}")
                    logger.info(f"[DEBUG] Converted candidate summary:")
                    logger.info(f"  - fullName: {candidate.get('fullName', 'NOT FOUND')}")
                    logger.info(f"  - email: {candidate.get('email', 'NOT FOUND')}")
                    logger.info(f"  - phone: {candidate.get('phone', 'NOT FOUND')}")
                    logger.info(f"  - location: {candidate.get('location', 'NOT FOUND')}")
                    logger.info(f"  - currentTitle: {candidate.get('currentTitle', 'NOT FOUND')}")
                    logger.info(f"  - currentCompany: {candidate.get('currentCompany', 'NOT FOUND')}")
                    logger.info(f"  - skills: {len(candidate.get('skills', []))} skills")
                    logger.info(f"  - experienceYears: {candidate.get('experienceYears', 'NOT FOUND')}")
                    logger.info(f"  - summary: {'FOUND' if candidate.get('summary') else 'NOT FOUND'} ({len(str(candidate.get('summary', '')))} chars)")
                    logger.info(f"  - resumeText: {'FOUND' if candidate.get('resumeText') else 'NOT FOUND'} ({len(str(candidate.get('resumeText', '')))} chars)")
                    logger.info(f"  - raw data keys: {list(candidate.get('raw', {}).keys())}")
                    
                    # Validate that we have at least a name
                    if not candidate.get('fullName') or candidate['fullName'] == 'Unknown Candidate':
                        # Try to use filename as fallback
                        name_from_filename = os.path.splitext(secure_filename(file.filename))[0]
                        if name_from_filename and len(name_from_filename) > 2:
                            candidate['fullName'] = name_from_filename.replace('_', ' ').replace('-', ' ')
                    
                    # Final validation - ensure candidate has at least some data
                    candidate_has_data = any([
                        candidate.get('fullName') and candidate['fullName'] != 'Unknown Candidate',
                        candidate.get('email'),
                        candidate.get('phone'),
                        len(candidate.get('skills', [])) > 0,
                        candidate.get('summary'),
                        candidate.get('resumeText'),
                        candidate.get('raw', {}).get('education'),
                        candidate.get('raw', {}).get('work_experience')
                    ])
                    
                    if not candidate_has_data:
                        logger.warning(f"[DEBUG] Candidate has no meaningful data after conversion for {file.filename}")
                        failed_files.append({
                            'filename': file.filename,
                            'error': 'Failed to extract meaningful data from resume after conversion'
                        })
                        continue
                    
                    candidates.append(candidate)
                    logger.info(f"Successfully parsed resume: {file.filename} -> {candidate['fullName']}")
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {tmp_path}: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
                failed_files.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        # Persist parsed candidates into DynamoDB so they are available for search
        try:
            _store_candidates_in_dynamodb(candidates)
        except Exception as e:
            # Do not fail the request if persistence fails; just log it
            logger.error(f"[DYNAMODB] Error while storing parsed candidates: {e}", exc_info=True)
        
        # Prepare response
        response_data = {
            'success': len(failed_files) == 0,
            'total_files': len(files),
            'candidates': candidates,
            'candidates_count': len(candidates),
            'failed_files': failed_files,
            'failed_count': len(failed_files)
        }
        
        # DEBUG: Log response data
        logger.info(f"[DEBUG] Response data summary:")
        logger.info(f"  - success: {response_data['success']}")
        logger.info(f"  - total_files: {response_data['total_files']}")
        logger.info(f"  - candidates_count: {response_data['candidates_count']}")
        logger.info(f"  - failed_count: {response_data['failed_count']}")
        if candidates:
            logger.info(f"[DEBUG] First candidate sample:")
            first_candidate = candidates[0]
            logger.info(f"  - Keys: {list(first_candidate.keys())}")
            logger.info(f"  - fullName: {first_candidate.get('fullName', 'MISSING')}")
            logger.info(f"  - email: {first_candidate.get('email', 'MISSING')}")
            logger.info(f"  - skills count: {len(first_candidate.get('skills', []))}")
            logger.info(f"  - Has summary: {bool(first_candidate.get('summary'))}")
            logger.info(f"  - Has resumeText: {bool(first_candidate.get('resumeText'))}")
        
        if len(candidates) > 0:
            status_code = 200 if len(failed_files) == 0 else 207  # 207 Multi-Status
            logger.info(f"[DEBUG] Returning {status_code} with {len(candidates)} candidates")
            return jsonify(response_data), status_code
        else:
            logger.warning(f"[DEBUG] No candidates found, returning 400")
            return jsonify(response_data), 400
            
    except Exception as e:
        logger.error(f"Error in resume parsing endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@candidate_resume_bp.route('/api/candidate-resume/parse-from-s3', methods=['POST'])
def parse_resumes_from_s3():
    """
    Parse resumes that already exist in S3 (by s3_key) and return ManualCandidateRecord candidates.
    This avoids browser-side S3 access and CORS issues.
    Request body: { "s3_keys": ["career_resume/user_x/...pdf", ...] }
    """
    try:
        if not s3_client or not S3_BUCKET:
            return jsonify({
                'success': False,
                'error': 'S3 is not configured on the server.'
            }), 500

        data = request.get_json(silent=True) or {}
        s3_keys = data.get('s3_keys') or data.get('keys') or []

        if isinstance(s3_keys, str):
            s3_keys = [s3_keys]

        if not isinstance(s3_keys, list) or not s3_keys:
            return jsonify({
                'success': False,
                'error': 's3_keys must be a non-empty list.'
            }), 400

        candidates = []
        failed_files = []

        for s3_key in s3_keys:
            if not isinstance(s3_key, str) or not s3_key.strip():
                continue

            filename = os.path.basename(s3_key)
            try:
                # Download object from S3 to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
                    tmp_path = tmp_file.name
                    s3_client.download_fileobj(S3_BUCKET, s3_key, tmp_file)

                try:
                    parsed_data = resume_parser.parse_file(tmp_path)

                    if not parsed_data or parsed_data == {}:
                        failed_files.append({
                            'filename': filename,
                            's3_key': s3_key,
                            'error': 'Failed to extract data from resume - parsed_data is empty'
                        })
                        continue

                    candidate = convert_parsed_data_to_manual_candidate(parsed_data, filename)

                    # Attach S3 metadata so it can be stored with the candidate in DynamoDB
                    raw_meta = candidate.get('raw') or {}
                    raw_meta['filename'] = filename
                    raw_meta['s3_key'] = s3_key
                    raw_meta['resume_s3_key'] = s3_key

                    # Build a short-lived HTTPS URL for this resume; fall back to s3:// URL if needed
                    resume_url = None
                    try:
                        resume_url = s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
                            ExpiresIn=3600,
                        )
                    except Exception as presign_error:
                        logger.error(f"Failed to generate presigned URL for {s3_key}: {presign_error}")
                        resume_url = f"s3://{S3_BUCKET}/{s3_key}" if S3_BUCKET else None

                    if resume_url:
                        raw_meta['resume_url'] = resume_url
                        raw_meta['s3_url'] = resume_url

                    candidate['raw'] = raw_meta
                    candidate = convert_parsed_data_to_manual_candidate(parsed_data, filename)

                    # Attach S3 metadata so it can be stored with the candidate in DynamoDB
                    raw_meta = candidate.get('raw') or {}
                    raw_meta['filename'] = filename
                    raw_meta['s3_key'] = s3_key
                    raw_meta['resume_s3_key'] = s3_key

                    # Build a short-lived HTTPS URL for this resume; fall back to s3:// URL if needed
                    resume_url = None
                    try:
                        resume_url = s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
                            ExpiresIn=3600,
                        )
                    except Exception as presign_error:
                        logger.error(f"Failed to generate presigned URL for {s3_key}: {presign_error}")
                        resume_url = f"s3://{S3_BUCKET}/{s3_key}" if S3_BUCKET else None

                    if resume_url:
                        raw_meta['resume_url'] = resume_url
                        raw_meta['s3_url'] = resume_url

                    candidate['raw'] = raw_meta

                    if not candidate.get('fullName') or candidate['fullName'] == 'Unknown Candidate':
                        name_from_filename = os.path.splitext(secure_filename(filename))[0]
                        if name_from_filename and len(name_from_filename) > 2:
                            candidate['fullName'] = name_from_filename.replace('_', ' ').replace('-', ' ')

                    candidate_has_data = any([
                        candidate.get('fullName') and candidate['fullName'] != 'Unknown Candidate',
                        candidate.get('email'),
                        candidate.get('phone'),
                        len(candidate.get('skills', [])) > 0,
                        candidate.get('summary'),
                        candidate.get('resumeText'),
                        candidate.get('raw', {}).get('education'),
                        candidate.get('raw', {}).get('work_experience')
                    ])

                    if not candidate_has_data:
                        failed_files.append({
                            'filename': filename,
                            's3_key': s3_key,
                            'error': 'Failed to extract meaningful data from resume after conversion'
                        })
                        continue

                    candidates.append(candidate)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {tmp_path}: {e}")

            except Exception as e:
                logger.error(f"Error processing S3 file {s3_key}: {str(e)}", exc_info=True)
                failed_files.append({
                    'filename': filename,
                    's3_key': s3_key,
                    'error': str(e)
                })

        # Persist parsed candidates into DynamoDB
        try:
            _store_candidates_in_dynamodb(candidates)
        except Exception as e:
            logger.error(f"[DYNAMODB] Error while storing parsed S3 candidates: {e}", exc_info=True)

        response_data = {
            'success': len(failed_files) == 0 and len(candidates) > 0,
            'total_files': len(s3_keys),
            'candidates': candidates,
            'candidates_count': len(candidates),
            'failed_files': failed_files,
            'failed_count': len(failed_files),
        }

        status_code = 200 if len(candidates) > 0 else 400
        return jsonify(response_data), status_code

    except Exception as e:
        logger.error(f"Error in S3 resume parsing endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@candidate_resume_bp.route('/api/candidate-resume/status', methods=['GET'])
def get_parse_status():
    """
    Get status of resume parsing functionality
    """
    try:
        return jsonify({
            'success': True,
            'allowed_extensions': list(ALLOWED_EXTENSIONS),
            'max_files_per_upload': MAX_FILES_PER_UPLOAD
        }), 200
        
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

