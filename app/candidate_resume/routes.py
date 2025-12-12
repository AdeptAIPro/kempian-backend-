"""
Candidate Resume Parsing Routes
Handles parsing of resume files and converting them to ManualCandidateRecord format
"""
import os
import re
import tempfile
import uuid
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.simple_logger import get_logger
from app.services.resume_parser import ResumeParser

logger = get_logger("candidate_resume")

candidate_resume_bp = Blueprint('candidate_resume', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'rtf'}
MAX_FILES_PER_UPLOAD = int(os.getenv('MAX_FILES_PER_UPLOAD', 200))  # Default: 200 files

resume_parser = ResumeParser()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_parsed_data_to_manual_candidate(parsed_data: dict, filename: str = "") -> dict:
    """
    Convert parsed resume data to ManualCandidateRecord format
    """
    # Extract current title and company from work experience
    current_title = None
    current_company = None
    
    work_experience = parsed_data.get('work_experience', '')
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
    if isinstance(skills, str):
        # If skills is a string, split it
        skills = [s.strip() for s in skills.split(',') if s.strip()]
    elif not isinstance(skills, list):
        skills = []
    
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
    
    return candidate

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
                    
                    if not parsed_data:
                        failed_files.append({
                            'filename': file.filename,
                            'error': 'Failed to extract data from resume'
                        })
                        continue
                    
                    # Convert to ManualCandidateRecord format
                    candidate = convert_parsed_data_to_manual_candidate(parsed_data, file.filename)
                    
                    # Validate that we have at least a name
                    if not candidate.get('fullName') or candidate['fullName'] == 'Unknown Candidate':
                        # Try to use filename as fallback
                        name_from_filename = os.path.splitext(secure_filename(file.filename))[0]
                        if name_from_filename and len(name_from_filename) > 2:
                            candidate['fullName'] = name_from_filename.replace('_', ' ').replace('-', ' ')
                    
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
        
        # Prepare response
        response_data = {
            'success': len(failed_files) == 0,
            'total_files': len(files),
            'candidates': candidates,
            'candidates_count': len(candidates),
            'failed_files': failed_files,
            'failed_count': len(failed_files)
        }
        
        if len(candidates) > 0:
            status_code = 200 if len(failed_files) == 0 else 207  # 207 Multi-Status
            return jsonify(response_data), status_code
        else:
            return jsonify(response_data), 400
            
    except Exception as e:
        logger.error(f"Error in resume parsing endpoint: {str(e)}", exc_info=True)
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

