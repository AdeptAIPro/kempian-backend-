import os
import uuid
import boto3
import json
from datetime import datetime, date, timedelta
from app.simple_logger import get_logger
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from sqlalchemy.exc import IntegrityError
from app.models import db, User, CandidateProfile, CandidateSkill, CandidateEducation, CandidateExperience, CandidateCertification, CandidateProject, UserSocialLinks, UserKPIs, UserSkillGap, UserLearningPath, UserAchievement, UserGoal, UserSchedule, LearningModule, LearningCourse, JDSearchLog, UserTrial, CeipalIntegration, UserImage
from app.search.routes import get_user_from_jwt, get_jwt_payload

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-south-1')
)

S3_BUCKET = os.getenv('RESUME_BUCKET') or os.getenv('S3_BUCKET') or "resume-bucket-adept-ai-pro"
RESUME_PREFIX = os.getenv('RESUME_PREFIX', 'career_resume/')

talent_bp = Blueprint('talent', __name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}  # Added txt for testing
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------
# Sanitization/Normalization
# -----------------------------
def _strip_personal_lines(text: str) -> str:
    """Remove lines likely to contain personal details from a block of text.
    Helps prevent personal details contaminating project/certification sections.
    """
    try:
        import re
        if not text:
            return text
        personal_patterns = [
            r'\b(email|e-mail|mail id)\b',
            r'\b(phone|mobile|contact|whatsapp)\b',
            r'\baddress\b',
            r'@[\w.-]+',  # email @domain
            r'\b\+?\d[\d\s().-]{6,}\b',  # phone-like numbers
            r'linkedin\.com/\w+',
            r'github\.com/\w+',
        ]
        lines = [ln for ln in text.split('\n') if ln.strip()]
        cleaned = []
        for ln in lines:
            lower_ln = ln.lower()
            if any(re.search(pat, lower_ln, re.IGNORECASE) for pat in personal_patterns):
                continue
            cleaned.append(ln)
        return '\n'.join(cleaned)
    except Exception:
        return text

def _extract_certifications_from_text(text: str) -> list:
    """Heuristically extract certification names from raw text when parser misses them."""
    try:
        import re
        if not text:
            return []
        patterns = [
            r'\b(AWS\s+Certified[^\n,;]*)',
            r'\b(Azure\s+Certified[^\n,;]*)',
            r'\b(Google\s+Cloud\s+Certified[^\n,;]*)',
            r'\b(PMP|PRINCE2|ITIL\s*\b[^\n,;]*)',
            r'\b(Oracle\s+Certified[^\n,;]*)',
            r'\b(CEH|CISSP|CompTIA\s+[^\n,;]*)',
            r'\b(Scrum\s*(Master|Product Owner)[^\n,;]*)',
            r'\b(Professional|Associate|Expert)\s+Certificate[^\n,;]*',
        ]
        found = []
        for pat in patterns:
            for m in re.findall(pat, text, re.IGNORECASE):
                if isinstance(m, tuple):
                    m = m[0]
                val = m.strip()
                if val and val.lower() not in {x.lower() for x in found}:
                    found.append(val)
        return found[:10]
    except Exception:
        return []

def _normalize_experience_text(text: str) -> str:
    """Keep experience lines with useful content/date ranges and drop objective-like sentences."""
    try:
        import re
        if not text:
            return text
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        useful = []
        objective_cues = [
            'objective',
            'seeking',
            'to achieve a challenging position',
            'looking for a position',
        ]
        date_regex = re.compile(r'(\d{4})\s*[-to]+\s*(\d{4}|present|current)', re.IGNORECASE)
        for ln in lines:
            low = ln.lower()
            if any(cue in low for cue in objective_cues):
                continue
            if len(ln) < 20 and not date_regex.search(ln):
                continue
            useful.append(ln)
        return '\n'.join(useful)
    except Exception:
        return text

def sanitize_parsed_resume(parsed: dict) -> dict:
    """Sanitize and enrich parser output to improve accuracy for projects, certifications, and experience."""
    if not isinstance(parsed, dict):
        return parsed or {}
    sanitized = dict(parsed)
    # Clean projects block from personal information contamination
    if 'projects' in sanitized and isinstance(sanitized['projects'], str):
        sanitized['projects'] = _strip_personal_lines(sanitized['projects'])
    # Improve experience block
    if 'work_experience' in sanitized and isinstance(sanitized['work_experience'], str):
        sanitized['work_experience'] = _normalize_experience_text(sanitized['work_experience'])
    # Fill certifications if missing
    certs = sanitized.get('certifications')
    if not certs or (isinstance(certs, str) and len(certs.strip()) < 6):
        raw_source = ''
        try:
            raw_source = (sanitized.get('raw_text') or '') + '\n' + (sanitized.get('projects') or '')
        except Exception:
            raw_source = sanitized.get('raw_text') or ''
        extracted = _extract_certifications_from_text(raw_source)
        if extracted:
            # Store as newline-separated string to keep compatibility with downstream code
            sanitized['certifications'] = '\n'.join(extracted)
    return sanitized

def _is_likely_project_line(line: str) -> bool:
    """Heuristic to determine if a line looks like a real project entry."""
    try:
        import re
        if not line:
            return False
        text = line.strip()
        if len(text) < 12:
            return False
        low = text.lower()
        # Explicitly reject personal-info sections frequently misclassified as projects
        reject_terms = [
            'personal detail', 'father', 'mother', 'parent', 'gender', 'marital', 'nationality',
            'dob', 'date of birth', 'address', 'languages', 'hobbies', 'interests', 'strength',
            'religion', 'caste'
        ]
        if any(term in low for term in reject_terms):
            return False
        # Require some project/tech/action signal
        positive_terms = [
            'project', 'develop', 'built', 'implemented', 'designed', 'created', 'deployed',
            'application', 'app', 'system', 'module', 'platform', 'website', 'api', 'service',
            'react', 'node', 'django', 'flask', 'python', 'java', 'spring', 'angular', 'vue',
            'aws', 'azure', 'gcp', 'sql', 'postgres', 'mysql', 'mongodb', 'docker', 'kubernetes'
        ]
        if not any(term in low for term in positive_terms):
            # Allow lines with a colon indicating name:description
            if ':' not in text:
                return False
        # Avoid lines that look like one or two words only
        if len(text.split()) < 3:
            return False
        # Avoid lines that are all lowercase/uppercase noise
        letters = re.sub(r'[^A-Za-z]', '', text)
        if letters and (letters.islower() or letters.isupper()) and len(text.split()) <= 3:
            return False
        return True
    except Exception:
        return False

def _filter_experience_entry(text: str) -> bool:
    """Filter out objective-like or non-experience lines."""
    try:
        import re
        if not text or len(text.strip()) < 20:
            return False
        low = text.strip().lower()
        if low.startswith('to '):
            return False
        reject_cues = [
            'objective', 'seeking', 'aim to', 'desire to', 'career goal',
        ]
        if any(cue in low for cue in reject_cues):
            return False
        # Prefer entries mentioning company/role cues or dates
        company_cues = ['inc', 'corp', 'llc', 'ltd', 'company', 'technologies', 'systems']
        role_cues = ['engineer', 'developer', 'manager', 'scientist', 'analyst', 'designer', 'architect']
        has_company_or_role = any(c in low for c in company_cues + role_cues)
        date_present = re.search(r'(\d{4})\s*[-to]+\s*(\d{4}|present|current)', low)
        return has_company_or_role or bool(date_present)
    except Exception:
        return False

@talent_bp.route('/upload-resume-enhanced', methods=['POST'])
def upload_resume_enhanced():
    """Enhanced resume upload with comprehensive parsing"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF, DOC, DOCX allowed'}), 400

        # Generate unique filename for S3
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        s3_key = f"{RESUME_PREFIX}{unique_filename}"

        # Read file content into memory to avoid file pointer issues
        try:
            file.seek(0)
            file_content = file.read()
            if not file_content:
                return jsonify({'error': 'File is empty or corrupted'}), 400
        except Exception as e:
            current_app.logger.error(f"File read error: {str(e)}")
            return jsonify({'error': 'Failed to read file content'}), 400

        # Upload file to S3
        try:
            from io import BytesIO
            file_obj = BytesIO(file_content)
            file_obj.seek(0)
            s3_client.upload_fileobj(
                file_obj,
                S3_BUCKET,
                s3_key,
                ExtraArgs={
                    'ContentType': file.content_type,
                    'ACL': 'private'
                }
            )
        except Exception as e:
            current_app.logger.error(f"S3 upload failed: {str(e)}")
            return jsonify({'error': 'Failed to upload file to cloud storage'}), 500

        # Parse resume using comprehensive parser
        from app.services.resume_parser import resume_parser
        import tempfile
        import os
        temp_file_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            current_app.logger.info(f"Starting resume parsing for file: {temp_file_path}")
            resume_data = resume_parser.parse_file(temp_file_path)
            current_app.logger.info("Resume parsing completed successfully")
            current_app.logger.info(f"Parsed data keys: {list(resume_data.keys()) if resume_data else 'None'}")
            current_app.logger.info(f"Skills found: {resume_data.get('skills', []) if resume_data else 'None'}")
            current_app.logger.info(f"Education found: {resume_data.get('education', '') if resume_data else 'None'}")
            current_app.logger.info(f"All resume data: {resume_data}")
        finally:
            # Clean up temporary file
            try:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception as cleanup_error:
                current_app.logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

        # Sanitize and normalize parsed data to improve accuracy
        resume_data = sanitize_parsed_resume(resume_data or {})

        # Validate parsed data
        if not resume_data:
            current_app.logger.warning("Resume parser returned empty data")
            resume_data = {}

        # Create or update candidate profile with comprehensive data
        profile = CandidateProfile.query.filter_by(user_id=user.id).first()

        # Ensure we have a valid full_name
        profile_full_name = resume_data.get('full_name', '')
        if not profile_full_name or profile_full_name.strip() == '':
            profile_full_name = user.email.split('@')[0]  # Use email prefix as fallback

        # Create or update the base profile first
        is_new_profile = False
        if not profile:
            is_new_profile = True
            profile = CandidateProfile(
                user_id=user.id,
                full_name=profile_full_name,
                phone=resume_data.get('phone', ''),
                location=resume_data.get('location', ''),
                summary=resume_data.get('summary', ''),
                experience_years=resume_data.get('experience_years'),
                resume_s3_key=s3_key,
                resume_filename=file.filename,
                resume_upload_date=datetime.utcnow()
            )
            db.session.add(profile)
            db.session.flush()  # Ensure we have profile.id
        else:
            profile.full_name = profile_full_name if profile_full_name else profile.full_name
            profile.phone = resume_data.get('phone', profile.phone)
            profile.location = resume_data.get('location', profile.location)
            profile.summary = resume_data.get('summary', profile.summary)
            profile.experience_years = resume_data.get('experience_years', profile.experience_years)
            profile.resume_s3_key = s3_key
            profile.resume_filename = file.filename
            profile.resume_upload_date = datetime.utcnow()
            profile.updated_at = datetime.utcnow()
            db.session.flush()

        # Clear previous parsed aggregates if updating existing profile
        if not is_new_profile:
            CandidateSkill.query.filter_by(profile_id=profile.id).delete()
            CandidateEducation.query.filter_by(profile_id=profile.id).delete()
            CandidateProject.query.filter_by(profile_id=profile.id).delete()
            CandidateCertification.query.filter_by(profile_id=profile.id).delete()

        # Add parsed skills
        if resume_data.get('skills'):
            current_app.logger.info(f"Processing {len(resume_data['skills'])} skills for profile {profile.id}")
            for skill_name in resume_data['skills']:
                skill = CandidateSkill(
                    profile_id=profile.id,
                    skill_name=skill_name
                )
                db.session.add(skill)
            current_app.logger.info(f"Added skills to database for profile {profile.id}")
        else:
            current_app.logger.warning(f"No skills found in resume data for profile {profile.id}")

        # Add parsed education
        if resume_data.get('education'):
            current_app.logger.info(f"Processing education data: {resume_data['education']}")
            import re
            education_text = resume_data['education']
            if education_text:
                # Split by double newlines to get education entries
                education_entries = re.split(r'\n\s*\n', education_text)
                education_entries = [entry.strip() for entry in education_entries if entry.strip()]
                # If no double newlines, try splitting by single newlines
                if len(education_entries) <= 1:
                    education_entries = re.split(r'\n', education_text)
                    education_entries = [entry.strip() for entry in education_entries if entry.strip()]

                current_app.logger.info(f"Found {len(education_entries)} education entries")
                for i, edu_entry in enumerate(education_entries[:3]):  # Limit to 3 education entries
                    current_app.logger.info(f"Processing education entry {i+1}: {edu_entry[:100]}...")
                    if not edu_entry or len(edu_entry) < 10:
                        current_app.logger.warning(f"Skipping education entry {i+1} - too short or empty")
                        continue

                    degree = edu_entry
                    institution = "Unknown Institution"
                    field_of_study = None

                    # Degree patterns
                    degree_patterns = [
                        r'(Bachelor\s+of\s+Technology\s*\([^)]+\)|Bachelor\s+of\s+[^,\n]+)',
                        r'(Master\s+of\s+[^,\n]+)',
                        r'(PhD|Ph\.D\.|Doctorate)[^,\n]*',
                        r'(Associate|Certificate|Diploma)[^,\n]*',
                        r'(HSC|SSC|10th|12th)[^,\n]*',
                        r'(B\.S\.|M\.S\.|B\.A\.|M\.A\.|Ph\.D\.)[^,\n]*',
                    ]
                    for pattern in degree_patterns:
                        match = re.search(pattern, edu_entry, re.IGNORECASE)
                        if match:
                            degree = match.group(0).strip()
                            break

                    # Institution patterns
                    institution_patterns = [
                        r'([A-Z][^,\n]*College[^,\n]*)',
                        r'([A-Z][^,\n]*University[^,\n]*)',
                        r'([A-Z][^,\n]*School[^,\n]*)',
                        r'([A-Z][^,\n]*Institute[^,\n]*)',
                        r'(State\s+Board|Central\s+Board)',
                    ]
                    for pattern in institution_patterns:
                        match = re.search(pattern, edu_entry, re.IGNORECASE)
                        if match:
                            institution = match.group(0).strip()
                            break

                    # Field of study
                    if 'in ' in degree.lower():
                        field_match = re.search(r'in\s+([^,\n]+)', degree, re.IGNORECASE)
                        if field_match:
                            field_of_study = field_match.group(1).strip()
                    elif '(' in degree and ')' in degree:
                        paren_match = re.search(r'\(([^)]+)\)', degree)
                        if paren_match:
                            field_of_study = paren_match.group(1).strip()

                    education = CandidateEducation(
                        profile_id=profile.id,
                        institution=institution,
                        degree=degree,
                        field_of_study=field_of_study,
                        description=edu_entry
                    )
                    db.session.add(education)
                    current_app.logger.info(f"Added education entry: {degree} from {institution}")
        else:
            current_app.logger.warning(f"No education data found in resume for profile {profile.id}")

        # Add parsed projects
        if resume_data.get('projects'):
            import re
            # Parse projects from the extracted text
            projects_text = resume_data['projects']
            if projects_text:
                # Split by common project separators
                project_lines = re.split(r'[;\n]', projects_text)
                for project_line in project_lines[:5]:  # Limit to 5 projects
                    project_line = project_line.strip()
                    if project_line and len(project_line) > 10 and _is_likely_project_line(project_line):
                        # Extract project name and description
                        if ':' in project_line:
                            name, description = project_line.split(':', 1)
                            name = name.strip()
                            description = description.strip()
                        else:
                            name = project_line[:50]  # First 50 chars as name
                            description = project_line
                            
                        project = CandidateProject(
                            profile_id=profile.id,
                            name=name,
                            description=description
                        )
                        db.session.add(project)

        # Add parsed certifications
        if resume_data.get('certifications'):
            import re
            # Parse certifications from the extracted text
            certs_text = resume_data['certifications']
            if certs_text:
                # Split by common certification separators
                cert_lines = re.split(r'[;\n]', certs_text)
                for cert_line in cert_lines[:5]:  # Limit to 5 certifications
                    cert_line = cert_line.strip()
                    if cert_line and len(cert_line) > 10:
                        # Extract certification name
                        name = cert_line
                        # Try to extract organization from common patterns
                        organization = None
                        if '(' in cert_line and ')' in cert_line:
                            # Extract text in parentheses as organization
                            org_match = re.search(r'\(([^)]+)\)', cert_line)
                            if org_match:
                                organization = org_match.group(1).strip()
                        
                        certification = CandidateCertification(
                            profile_id=profile.id,
                            name=name,
                            issuing_organization=organization
                        )
                        db.session.add(certification)
        else:
            # If no certifications found in resume data, try to extract from projects or other sections
            if resume_data.get('projects'):
                projects_text = resume_data['projects']
                # Look for certification patterns in projects
                cert_patterns = [
                    r'(AWS|Amazon Web Services)[^,\n]*',
                    r'(Google Cloud|GCP)[^,\n]*',
                    r'(Microsoft Azure|Azure)[^,\n]*',
                    r'(Certified|Certification)[^,\n]*',
                    r'(Professional|Associate|Expert)[^,\n]*',
                ]
                
                found_certs = []
                for pattern in cert_patterns:
                    matches = re.findall(pattern, projects_text, re.IGNORECASE)
                    found_certs.extend(matches)
                
                if found_certs:
                    for cert_name in found_certs[:3]:  # Limit to 3
                        certification = CandidateCertification(
                            profile_id=profile.id,
                            name=cert_name.strip(),
                            issuing_organization=None
                        )
                        db.session.add(certification)

        # Add parsed work experience
        if resume_data.get('work_experience'):
            import re
            experience_text = resume_data['work_experience']
            if experience_text:
                experience_entries = re.split(r'\n\s*\n', experience_text)
                experience_entries = [entry.strip() for entry in experience_entries if entry.strip()]
                if len(experience_entries) <= 1:
                    experience_entries = re.split(r'\n', experience_text)
                    experience_entries = [entry.strip() for entry in experience_entries if entry.strip()]

                for i, exp_entry in enumerate(experience_entries[:5]):  # Limit to 5
                    if not exp_entry or len(exp_entry) < 20 or not _filter_experience_entry(exp_entry):
                        continue

                    company = "Unknown Company"
                    position = exp_entry
                    description = exp_entry

                    # Company
                    company_patterns = [
                        r'([A-Z][^,\n]*Inc[^,\n]*)',
                        r'([A-Z][^,\n]*Corp[^,\n]*)',
                        r'([A-Z][^,\n]*LLC[^,\n]*)',
                        r'([A-Z][^,\n]*Ltd[^,\n]*)',
                        r'([A-Z][^,\n]*Company[^,\n]*)',
                        r'([A-Z][^,\n]*Technologies[^,\n]*)',
                        r'([A-Z][^,\n]*Systems[^,\n]*)',
                    ]
                    for pattern in company_patterns:
                        match = re.search(pattern, exp_entry, re.IGNORECASE)
                        if match:
                            company = match.group(0).strip()
                            break

                    # Position
                    position_patterns = [
                        r'(Software\s+Engineer|Developer|Programmer)',
                        r'(Senior|Lead|Principal|Manager)',
                        r'(Data\s+Scientist|Analyst)',
                        r'(Product\s+Manager|Project\s+Manager)',
                        r'(Designer|Architect)',
                    ]
                    for pattern in position_patterns:
                        match = re.search(pattern, exp_entry, re.IGNORECASE)
                        if match:
                            position = match.group(0).strip()
                            break

                    # Dates
                    from datetime import date as _date
                    start_date = _date.today()
                    end_date = None
                    is_current = False

                    date_patterns = [
                        r'(\d{4})\s*-\s*(\d{4})',
                        r'(\d{4})\s*to\s*(\d{4})',
                        r'(\d{4})\s*-\s*(present|current)',
                        r'(\d{4})\s*to\s*(present|current)',
                    ]
                    for pattern in date_patterns:
                        match = re.search(pattern, exp_entry, re.IGNORECASE)
                        if match:
                            try:
                                start_year = int(match.group(1))
                                start_date = _date(start_year, 1, 1)
                                g2 = match.group(2) if len(match.groups()) > 1 else None
                                if g2 and g2.lower() not in ['present', 'current']:
                                    end_year = int(g2)
                                    end_date = _date(end_year, 12, 31)
                                    is_current = False
                                else:
                                    end_date = None
                                    is_current = True
                                break
                            except (ValueError, IndexError):
                                pass

                    experience = CandidateExperience(
                        profile_id=profile.id,
                        company=company,
                        position=position,
                        description=description,
                        start_date=start_date,
                        end_date=end_date,
                        is_current=is_current
                    )
                    db.session.add(experience)
            
            db.session.commit()

        # Prepare response data
        response_data = {
            'message': 'Resume uploaded and comprehensive profile created successfully',
            'profile_id': profile.id,
            'resume_url': f"s3://{S3_BUCKET}/{s3_key}",
            'profile_created': True,
            'parsed_data': {
                'full_name': profile_full_name,
                'phone': resume_data.get('phone', ''),
                'location': resume_data.get('location', ''),
                'summary': resume_data.get('summary', ''),
                'experience_years': resume_data.get('experience_years', 0),
                'skills': resume_data.get('skills', []),
                'skills_count': len(resume_data.get('skills', [])),
                'education': resume_data.get('education', ''),
                'projects': resume_data.get('projects', ''),
                'certifications': resume_data.get('certifications', ''),
                'work_experience': resume_data.get('work_experience', ''),
                'raw_text': (resume_data.get('raw_text', '')[:500] + '...') if resume_data.get('raw_text') else ''
            },
            'profile_summary': {
                'total_skills': len(resume_data.get('skills', [])),
                'education_entries': len([e for e in (resume_data.get('education') or '').split('\n') if e.strip()]),
                'project_entries': len([p for p in (resume_data.get('projects') or '').split('\n') if p.strip()]),
                'certification_entries': len([c for c in (resume_data.get('certifications') or '').split('\n') if c.strip()]),
                'experience_entries': len([e for e in (resume_data.get('work_experience') or '').split('\n') if e.strip()])
            }
        }
        return jsonify(response_data), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Enhanced resume upload error: {str(e)}")
        error_message = 'Internal server error'
        if 'I/O operation on closed file' in str(e):
            error_message = 'File processing error. Please try uploading the file again.'
        elif 'S3' in str(e):
            error_message = 'File upload failed. Please check your connection and try again.'
        elif 'database' in str(e).lower():
            error_message = 'Database error. Please try again.'
        return jsonify({'error': error_message}), 500

@talent_bp.route('/upload-resume', methods=['POST'])
def upload_resume():
    """Upload resume to S3 and create/update candidate profile"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF, DOC, DOCX allowed'}), 400

        # Get form data
        full_name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        skills = request.form.get('skills', '').strip()
        experience = request.form.get('experience', '').strip()
        linkedin = request.form.get('linkedin', '').strip()
        github = request.form.get('github', '').strip()
        facebook = request.form.get('facebook', '').strip()
        twitter = request.form.get('twitter', '').strip()

        if not full_name or not email:
            return jsonify({'error': 'Name and email are required'}), 400

        # Generate unique filename for S3
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        s3_key = f"{RESUME_PREFIX}{unique_filename}"

        # Upload file to S3
        try:
            s3_client.upload_fileobj(
                file,
                S3_BUCKET,
                s3_key,
                ExtraArgs={
                    'ContentType': file.content_type,
                    'ACL': 'private'
                }
            )
        except Exception as e:
            current_app.logger.error(f"S3 upload failed: {str(e)}")
            return jsonify({'error': 'Failed to upload file to cloud storage'}), 500

        # Create or update candidate profile
        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        
        if not profile:
            profile = CandidateProfile(
                user_id=user.id,
                full_name=full_name,
                phone=phone,
                resume_s3_key=s3_key,
                resume_filename=file.filename,
                resume_upload_date=datetime.utcnow()
            )
            db.session.add(profile)
        else:
            profile.full_name = full_name
            profile.phone = phone
            profile.resume_s3_key = s3_key
            profile.resume_filename = file.filename
            profile.resume_upload_date = datetime.utcnow()
            profile.updated_at = datetime.utcnow()

        db.session.commit()

        # Parse and store skills
        if skills:
            # Clear existing skills
            CandidateSkill.query.filter_by(profile_id=profile.id).delete()
            
            # Parse skills (comma-separated or newline-separated)
            skill_list = [skill.strip() for skill in skills.replace('\n', ',').split(',') if skill.strip()]
            
            for skill_name in skill_list:
                skill = CandidateSkill(
                    profile_id=profile.id,
                    skill_name=skill_name
                )
                db.session.add(skill)

        # Store experience summary
        if experience:
            profile.summary = experience

        # Save social links
        if linkedin or github or facebook or twitter:
            social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
            if social_links:
                social_links.linkedin = linkedin
                social_links.github = github
                social_links.facebook = facebook
                social_links.x = twitter
                social_links.updated_at = datetime.utcnow()
            else:
                social_links = UserSocialLinks(
                    user_id=user.id,
                    linkedin=linkedin,
                    github=github,
                    facebook=facebook,
                    x=twitter
                )
                db.session.add(social_links)

        db.session.commit()

        return jsonify({
            'message': 'Resume uploaded successfully',
            'profile_id': profile.id,
            'resume_url': f"s3://{S3_BUCKET}/{s3_key}"
        }), 200

    except IntegrityError:
        db.session.rollback()
        return jsonify({'error': 'Profile already exists for this user'}), 409
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Resume upload error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/upload-resume-public', methods=['POST'])
def upload_resume_public():
    """Public endpoint for resume upload (no authentication required)"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF, DOC, DOCX allowed'}), 400

        # Get form data
        full_name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        skills = request.form.get('skills', '').strip()
        experience = request.form.get('experience', '').strip()
        linkedin = request.form.get('linkedin', '').strip()
        github = request.form.get('github', '').strip()
        facebook = request.form.get('facebook', '').strip()
        twitter = request.form.get('twitter', '').strip()

        if not full_name or not email:
            return jsonify({'error': 'Name and email are required'}), 400

        # Check if user exists with this email
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found. Please sign up first.'}), 404

        # Generate unique filename for S3
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        s3_key = f"career_resume/{unique_filename}"

        # Upload file to S3
        try:
            s3_client.upload_fileobj(
                file,
                S3_BUCKET,
                s3_key,
                ExtraArgs={
                    'ContentType': file.content_type,
                    'ACL': 'private'
                }
            )
        except Exception as e:
            current_app.logger.error(f"S3 upload failed: {str(e)}")
            return jsonify({'error': 'Failed to upload file to cloud storage'}), 500

        # Create or update candidate profile
        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        
        if not profile:
            profile = CandidateProfile(
                user_id=user.id,
                full_name=full_name,
                phone=phone,
                resume_s3_key=s3_key,
                resume_filename=file.filename,
                resume_upload_date=datetime.utcnow()
            )
            db.session.add(profile)
        else:
            profile.full_name = full_name
            profile.phone = phone
            profile.resume_s3_key = s3_key
            profile.resume_filename = file.filename
            profile.resume_upload_date = datetime.utcnow()
            profile.updated_at = datetime.utcnow()

        db.session.commit()

        # Parse and store skills
        if skills:
            # Clear existing skills
            CandidateSkill.query.filter_by(profile_id=profile.id).delete()
            
            # Parse skills (comma-separated or newline-separated)
            skill_list = [skill.strip() for skill in skills.replace('\n', ',').split(',') if skill.strip()]
            
            for skill_name in skill_list:
                skill = CandidateSkill(
                    profile_id=profile.id,
                    skill_name=skill_name
                )
                db.session.add(skill)

        # Store experience summary
        if experience:
            profile.summary = experience

        # Save social links
        if linkedin or github or facebook or twitter:
            social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
            if social_links:
                social_links.linkedin = linkedin
                social_links.github = github
                social_links.facebook = facebook
                social_links.x = twitter
                social_links.updated_at = datetime.utcnow()
            else:
                social_links = UserSocialLinks(
                    user_id=user.id,
                    linkedin=linkedin,
                    github=github,
                    facebook=facebook,
                    x=twitter
                )
                db.session.add(social_links)

        db.session.commit()

        return jsonify({
            'message': 'Resume uploaded successfully',
            'profile_id': profile.id
        }), 200

    except IntegrityError:
        db.session.rollback()
        return jsonify({'error': 'Profile already exists for this user'}), 409
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Public resume upload error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/profile', methods=['GET'])
def get_profile():
    """Get current user's candidate profile"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            return jsonify({'error': 'Profile not found'}), 404

        return jsonify(profile.to_dict()), 200

    except Exception as e:
        current_app.logger.error(f"Get profile error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/profile', methods=['PUT'])
def update_profile():
    """Update candidate profile"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            return jsonify({'error': 'Profile not found'}), 404

        data = request.get_json()
        
        # Update basic profile fields
        if 'full_name' in data:
            profile.full_name = data['full_name']
        if 'phone' in data:
            profile.phone = data['phone']
        if 'location' in data:
            profile.location = data['location']
        if 'summary' in data:
            profile.summary = data['summary']
        if 'experience_years' in data:
            profile.experience_years = data['experience_years']
        if 'current_salary' in data:
            profile.current_salary = data['current_salary']
        if 'expected_salary' in data:
            profile.expected_salary = data['expected_salary']
        if 'availability' in data:
            profile.availability = data['availability']
        if 'is_public' in data:
            profile.is_public = data['is_public']

        profile.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify(profile.to_dict()), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Update profile error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/profile/skills', methods=['POST'])
def add_skill():
    """Add a skill to the candidate profile"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            return jsonify({'error': 'Profile not found'}), 404

        data = request.get_json()
        skill_name = data.get('skill_name', '').strip()
        proficiency_level = data.get('proficiency_level', '')
        years_experience = data.get('years_experience')

        if not skill_name:
            return jsonify({'error': 'Skill name is required'}), 400

        skill = CandidateSkill(
            profile_id=profile.id,
            skill_name=skill_name,
            proficiency_level=proficiency_level,
            years_experience=years_experience
        )
        db.session.add(skill)
        db.session.commit()

        return jsonify(skill.to_dict()), 201

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Add skill error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/profile/skills/<int:skill_id>', methods=['DELETE'])
def delete_skill(skill_id):
    """Delete a skill from the candidate profile"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            return jsonify({'error': 'Profile not found'}), 404

        skill = CandidateSkill.query.filter_by(id=skill_id, profile_id=profile.id).first()
        if not skill:
            return jsonify({'error': 'Skill not found'}), 404

        db.session.delete(skill)
        db.session.commit()

        return jsonify({'message': 'Skill deleted successfully'}), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Delete skill error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/profile/education', methods=['POST'])
def add_education():
    """Add education to the candidate profile"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            return jsonify({'error': 'Profile not found'}), 404

        data = request.get_json()
        
        # Parse dates
        start_date = None
        end_date = None
        if data.get('start_date'):
            start_date = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
        if data.get('end_date'):
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d').date()

        education = CandidateEducation(
            profile_id=profile.id,
            institution=data.get('institution', ''),
            degree=data.get('degree', ''),
            field_of_study=data.get('field_of_study', ''),
            start_date=start_date,
            end_date=end_date,
            gpa=data.get('gpa'),
            description=data.get('description', '')
        )
        db.session.add(education)
        db.session.commit()

        return jsonify(education.to_dict()), 201

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Add education error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/profile/experience', methods=['POST'])
def add_experience():
    """Add work experience to the candidate profile"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            return jsonify({'error': 'Profile not found'}), 404

        data = request.get_json()
        
        # Parse dates
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
        end_date = None
        if data.get('end_date'):
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d').date()

        experience = CandidateExperience(
            profile_id=profile.id,
            company=data.get('company', ''),
            position=data.get('position', ''),
            start_date=start_date,
            end_date=end_date,
            is_current=data.get('is_current', False),
            description=data.get('description', ''),
            achievements=data.get('achievements', '')
        )
        db.session.add(experience)
        db.session.commit()

        return jsonify(experience.to_dict()), 201

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Add experience error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/profile/certifications', methods=['POST'])
def add_certification():
    """Add certification to the candidate profile"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            return jsonify({'error': 'Profile not found'}), 404

        data = request.get_json()
        
        # Parse dates
        issue_date = None
        expiry_date = None
        if data.get('issue_date'):
            issue_date = datetime.strptime(data['issue_date'], '%Y-%m-%d').date()
        if data.get('expiry_date'):
            expiry_date = datetime.strptime(data['expiry_date'], '%Y-%m-%d').date()

        certification = CandidateCertification(
            profile_id=profile.id,
            name=data.get('name', ''),
            issuing_organization=data.get('issuing_organization', ''),
            issue_date=issue_date,
            expiry_date=expiry_date,
            credential_id=data.get('credential_id', ''),
            credential_url=data.get('credential_url', '')
        )
        db.session.add(certification)
        db.session.commit()

        return jsonify(certification.to_dict()), 201

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Add certification error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/resume/<int:profile_id>', methods=['GET'])
def get_resume_url(profile_id):
    """Get presigned URL for resume download"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        profile = CandidateProfile.query.filter_by(id=profile_id, user_id=user.id).first()
        if not profile or not profile.resume_s3_key:
            return jsonify({'error': 'Resume not found'}), 404

        # Generate download URL (CloudFront preferred, S3 fallback)
        try:
            import os
            download_url = None
            
            # Try CloudFront first if configured
            cf_domain = os.getenv('CLOUDFRONT_DOMAIN')
            if cf_domain:
                try:
                    from app.utils.cloudfront_utils import generate_resume_download_url
                    download_url = generate_resume_download_url(profile.resume_s3_key, ttl_minutes=60)
                    current_app.logger.info(f"Generated CloudFront URL for profile resume {profile.resume_s3_key}")
                except Exception as cf_error:
                    current_app.logger.warning(f"CloudFront URL generation failed, falling back to S3: {str(cf_error)}")
                    download_url = None
            
            # Fallback to S3 presigned URL if CloudFront failed or not configured
            if not download_url:
                download_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': S3_BUCKET, 'Key': profile.resume_s3_key},
                    ExpiresIn=3600
                )
                current_app.logger.info(f"Generated S3 presigned URL for profile resume {profile.resume_s3_key}")
            
            return jsonify({
                'download_url': download_url,
                'filename': profile.resume_filename
            }), 200
            
        except Exception as e:
            current_app.logger.error(f"Failed to generate download URL for profile resume {profile.resume_s3_key}: {str(e)}")
            return jsonify({'error': 'Failed to generate download URL'}), 500

    except Exception as e:
        current_app.logger.error(f"Get resume URL error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/candidates', methods=['GET'])
def get_candidates():
    """Get all public candidate profiles (for employers)"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Only employers, recruiters, admins, or owners can view candidates
        if user.role not in ['employer', 'recruiter', 'admin', 'owner']:
            return jsonify({'error': 'Access denied'}), 403

        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        skills_filter = request.args.get('skills', '')
        location_filter = request.args.get('location', '')

        # Build query
        query = CandidateProfile.query.filter_by(is_public=True)

        if skills_filter:
            skills_list = [skill.strip() for skill in skills_filter.split(',')]
            for skill in skills_list:
                query = query.join(CandidateSkill).filter(
                    CandidateSkill.skill_name.ilike(f'%{skill}%')
                )

        if location_filter:
            query = query.filter(CandidateProfile.location.ilike(f'%{location_filter}%'))

        # Paginate results
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )

        candidates = []
        for profile in pagination.items:
            candidate_data = profile.to_dict()
            # Remove sensitive information for public view
            candidate_data.pop('phone', None)
            candidate_data.pop('current_salary', None)
            candidate_data.pop('expected_salary', None)
            candidates.append(candidate_data)

        return jsonify({
            'candidates': candidates,
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
            'per_page': per_page
        }), 200

    except Exception as e:
        current_app.logger.error(f"Get candidates error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/candidates/<int:candidate_id>', methods=['GET'])
def get_candidate_profile(candidate_id):
    """Get full candidate profile by ID (for employers)"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Only employers, recruiters, admins, or owners can view candidate profiles
        if user.role not in ['employer', 'recruiter', 'admin', 'owner']:
            return jsonify({'error': 'Access denied'}), 403

        # Get candidate profile
        # Support lookup by CandidateProfile.id OR by User.id (applicant_id from applications)
        profile = CandidateProfile.query.filter_by(id=candidate_id, is_public=True).first()
        if not profile:
            # Fallback: treat candidate_id as user_id from applications
            profile = CandidateProfile.query.filter_by(user_id=candidate_id, is_public=True).first()
        if not profile:
            return jsonify({'error': 'Candidate profile not found'}), 404

        # Get full profile data
        candidate_data = profile.to_dict()
        
        # Get additional profile data
        skills = CandidateSkill.query.filter_by(profile_id=profile.id).all()
        education = CandidateEducation.query.filter_by(profile_id=profile.id).all()
        experience = CandidateExperience.query.filter_by(profile_id=profile.id).all()
        certifications = CandidateCertification.query.filter_by(profile_id=profile.id).all()
        projects = CandidateProject.query.filter_by(profile_id=profile.id).all()

        candidate_data['skills'] = [skill.to_dict() for skill in skills]
        candidate_data['education'] = [edu.to_dict() for edu in education]
        candidate_data['experience'] = [exp.to_dict() for exp in experience]
        candidate_data['certifications'] = [cert.to_dict() for cert in certifications]
        candidate_data['projects'] = [proj.to_dict() for proj in projects]

        return jsonify({
            'candidate': candidate_data,
            'success': True
        }), 200

    except Exception as e:
        current_app.logger.error(f"Get candidate profile error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@talent_bp.route('/profile', methods=['DELETE'])
def delete_profile():
    """Delete user profile and all associated data"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get the candidate profile
        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            return jsonify({'error': 'Profile not found'}), 404
        
        # Remove all associated data first (due to foreign key constraints)
        # Remove candidate skills
        CandidateSkill.query.filter_by(profile_id=profile.id).delete()
        
        # Remove candidate education
        CandidateEducation.query.filter_by(profile_id=profile.id).delete()
        
        # Remove candidate experience
        CandidateExperience.query.filter_by(profile_id=profile.id).delete()
        
        # Remove candidate certifications
        CandidateCertification.query.filter_by(profile_id=profile.id).delete()
        
        # Remove KPI data
        UserKPIs.query.filter_by(user_id=user.id).delete()
        UserSkillGap.query.filter_by(user_id=user.id).delete()
        UserAchievement.query.filter_by(user_id=user.id).delete()
        UserGoal.query.filter_by(user_id=user.id).delete()
        UserSchedule.query.filter_by(user_id=user.id).delete()
        
        # Remove learning modules and courses first (due to foreign key constraints)
        learning_paths = UserLearningPath.query.filter_by(user_id=user.id).all()
        for path in learning_paths:
            # First delete all courses associated with modules in this learning path
            modules = LearningModule.query.filter_by(learning_path_id=path.id).all()
            for module in modules:
                LearningCourse.query.filter_by(module_id=module.id).delete()
            # Then delete the modules
            LearningModule.query.filter_by(learning_path_id=path.id).delete()
        # Finally delete the learning paths
        UserLearningPath.query.filter_by(user_id=user.id).delete()
        
        # Remove other user-related data
        JDSearchLog.query.filter_by(user_id=user.id).delete()
        UserSocialLinks.query.filter_by(user_id=user.id).delete()
        UserTrial.query.filter_by(user_id=user.id).delete()
        CeipalIntegration.query.filter_by(user_id=user.id).delete()
        
        # Remove user images
        UserImage.query.filter_by(user_id=user.id).delete()
        
        # Finally, remove the candidate profile
        db.session.delete(profile)
        
        # Commit all changes
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Profile and all associated data removed successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Delete profile error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to remove profile: {str(e)}'
        }), 500

# Saved Candidates API endpoints
@talent_bp.route('/saved-candidates', methods=['POST'])
def save_candidate():
    """Save a candidate for 20-day visibility"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        data = request.get_json()
        candidate_data = data.get('candidate_data')
        job_description = data.get('job_description', '')
        
        if not candidate_data:
            return jsonify({'error': 'Candidate data is required'}), 400

        # Import the new models
        from app.models import SavedCandidate
        
        # Check if candidate is already saved by this user
        existing_save = SavedCandidate.query.filter_by(
            user_id=user.id,
            candidate_data=json.dumps(candidate_data)
        ).first()
        
        if existing_save:
            # Update the existing save with new expiration date
            existing_save.expires_at = datetime.utcnow() + timedelta(days=20)
            existing_save.is_active = True
            existing_save.updated_at = datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Candidate save updated successfully',
                'saved_candidate': existing_save.to_dict()
            }), 200

        # Create new saved candidate
        saved_candidate = SavedCandidate(
            user_id=user.id,
            candidate_data=json.dumps(candidate_data),
            job_description=job_description
        )
        
        db.session.add(saved_candidate)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Candidate saved successfully for 20 days',
            'saved_candidate': saved_candidate.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Save candidate error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to save candidate: {str(e)}'
        }), 500

@talent_bp.route('/saved-candidates', methods=['GET'])
def get_saved_candidates():
    """Get all saved candidates for the user (active and non-expired)"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Import the new models
        from app.models import SavedCandidate
        
        # Get active, non-expired saved candidates
        saved_candidates = SavedCandidate.query.filter_by(
            user_id=user.id,
            is_active=True
        ).filter(
            SavedCandidate.expires_at > datetime.utcnow()
        ).order_by(SavedCandidate.saved_at.desc()).all()
        
        # Clean up expired candidates
        expired_candidates = SavedCandidate.query.filter_by(
            user_id=user.id,
            is_active=True
        ).filter(
            SavedCandidate.expires_at <= datetime.utcnow()
        ).all()
        
        for candidate in expired_candidates:
            candidate.is_active = False
            candidate.updated_at = datetime.utcnow()
        
        if expired_candidates:
            db.session.commit()
        
        return jsonify({
            'success': True,
            'saved_candidates': [candidate.to_dict() for candidate in saved_candidates],
            'total_count': len(saved_candidates)
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Get saved candidates error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get saved candidates: {str(e)}'
        }), 500

@talent_bp.route('/saved-candidates/<int:candidate_id>', methods=['DELETE'])
def delete_saved_candidate(candidate_id):
    """Delete a saved candidate"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Import the new models
        from app.models import SavedCandidate
        
        saved_candidate = SavedCandidate.query.filter_by(
            id=candidate_id,
            user_id=user.id
        ).first()
        
        if not saved_candidate:
            return jsonify({'error': 'Saved candidate not found'}), 404
        
        saved_candidate.is_active = False
        saved_candidate.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Saved candidate deleted successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Delete saved candidate error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to delete saved candidate: {str(e)}'
        }), 500

# Search History API endpoints
@talent_bp.route('/search-history', methods=['POST'])
def save_search_history():
    """Save search history for 5-day retention"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        data = request.get_json()
        search_query = data.get('search_query')
        search_results = data.get('search_results')
        conversation_history = data.get('conversation_history', [])
        search_type = data.get('search_type', 'chatbot')
        
        if not search_query:
            return jsonify({'error': 'Search query is required'}), 400

        # Import the new models
        from app.models import SearchHistory
        
        # Create new search history entry
        search_history = SearchHistory(
            user_id=user.id,
            search_query=search_query,
            search_results=json.dumps(search_results) if search_results else None,
            conversation_history=json.dumps(conversation_history),
            search_type=search_type
        )
        
        db.session.add(search_history)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Search history saved successfully for 5 days',
            'search_history': search_history.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Save search history error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to save search history: {str(e)}'
        }), 500

@talent_bp.route('/search-history', methods=['GET'])
def get_search_history():
    """Get search history for the user (active and non-expired)"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Import the new models
        from app.models import SearchHistory
        
        # Get active, non-expired search history
        search_history = SearchHistory.query.filter_by(
            user_id=user.id,
            is_active=True
        ).filter(
            SearchHistory.expires_at > datetime.utcnow()
        ).order_by(SearchHistory.searched_at.desc()).all()
        
        # Clean up expired search history
        expired_history = SearchHistory.query.filter_by(
            user_id=user.id,
            is_active=True
        ).filter(
            SearchHistory.expires_at <= datetime.utcnow()
        ).all()
        
        for history in expired_history:
            history.is_active = False
            history.updated_at = datetime.utcnow()
        
        if expired_history:
            db.session.commit()
        
        return jsonify({
            'success': True,
            'search_history': [history.to_dict() for history in search_history],
            'total_count': len(search_history)
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Get search history error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get search history: {str(e)}'
        }), 500

@talent_bp.route('/search-history/<int:history_id>', methods=['DELETE'])
def delete_search_history(history_id):
    """Delete a search history entry"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Import the new models
        from app.models import SearchHistory
        
        search_history = SearchHistory.query.filter_by(
            id=history_id,
            user_id=user.id
        ).first()
        
        if not search_history:
            return jsonify({'error': 'Search history not found'}), 404
        
        search_history.is_active = False
        search_history.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Search history deleted successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Delete search history error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to delete search history: {str(e)}'
        }), 500 