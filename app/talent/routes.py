import os
import uuid
import boto3
import json
import re
from datetime import datetime, date, timedelta
from app.simple_logger import get_logger
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from sqlalchemy.exc import IntegrityError
from app.models import db, User, CandidateProfile, CandidateSkill, CandidateEducation, CandidateExperience, CandidateCertification, CandidateProject, UserSocialLinks, UserKPIs, UserSkillGap, UserLearningPath, UserAchievement, UserGoal, UserSchedule, LearningModule, LearningCourse, JDSearchLog, UserTrial, CeipalIntegration, UserImage
from app.search.routes import get_user_from_jwt, get_jwt_payload

logger = get_logger(__name__.split('.')[-1])

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-south-1')
)

def _validate_s3_bucket_name(bucket_name: str) -> bool:
    """Validate S3 bucket name according to AWS naming rules"""
    if not bucket_name:
        return False
    # S3 bucket names must match: ^[a-zA-Z0-9.\-_]{1,255}$
    # Check for invalid characters like <, >, or placeholders
    if '<' in bucket_name or '>' in bucket_name:
        return False
    if not re.match(r'^[a-zA-Z0-9.\-_]{1,255}$', bucket_name):
        return False
    return True

# Get bucket name from environment with validation
_resume_bucket = os.getenv('RESUME_BUCKET')
_s3_bucket = os.getenv('S3_BUCKET')
_default_bucket = "resume-bucket-adept-ai-pro"

# Validate and use the first valid bucket name
S3_BUCKET = None
if _resume_bucket and _validate_s3_bucket_name(_resume_bucket):
    S3_BUCKET = _resume_bucket
elif _s3_bucket and _validate_s3_bucket_name(_s3_bucket):
    S3_BUCKET = _s3_bucket
else:
    S3_BUCKET = _default_bucket

# Log warning if invalid bucket name was provided
if _resume_bucket and not _validate_s3_bucket_name(_resume_bucket):
    logger.warning(f"Invalid RESUME_BUCKET name '{_resume_bucket}', using default '{S3_BUCKET}'")
if _s3_bucket and not _validate_s3_bucket_name(_s3_bucket):
    logger.warning(f"Invalid S3_BUCKET name '{_s3_bucket}', using default '{S3_BUCKET}'")

RESUME_PREFIX = os.getenv('RESUME_PREFIX', 'career_resume/')

talent_bp = Blueprint('talent', __name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}  # Added txt for testing
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_job_description_file(filename):
    """Check if file extension is allowed for job description uploads"""
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'rtf', 'csv', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_folder_name(user, tenant_id):
    """
    Get or generate a unique folder name for user based on existing files or email/tenant_id.
    Checks if user already has files stored and uses the same folder to maintain consistency.
    """
    import re
    
    # First, check if user has existing resume files stored
    # Look for CandidateProfile with resume_s3_key that might indicate a folder structure
    try:
        from app.models import CandidateProfile
        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if profile and profile.resume_s3_key:
            # Extract folder from existing s3_key
            # Example: "career_resume/user_john_123/filename.pdf" -> "user_john_123"
            # Example: "career_resume/filename.pdf" -> None (no folder)
            s3_key = profile.resume_s3_key
            if RESUME_PREFIX in s3_key:
                # Remove the prefix to get the path after career_resume/
                path_after_prefix = s3_key.replace(RESUME_PREFIX, '', 1)
                # Check if there's a folder (contains '/')
                if '/' in path_after_prefix:
                    # Extract folder name (everything before the last '/')
                    existing_folder = path_after_prefix.split('/')[0]
                    if existing_folder and (existing_folder.startswith('user_') or existing_folder.startswith('tenant_')):
                        logger.info(f"[DEBUG] Found existing folder for user {user.id}: {existing_folder}")
                        return existing_folder
    except Exception as e:
        logger.warning(f"Could not check existing folder for user {user.id}: {str(e)}")
    
    # If no existing folder found, generate new one based on email or tenant_id
    if user and hasattr(user, 'email') and user.email:
        # Sanitize email to create folder name (remove special chars, keep alphanumeric and underscore)
        email_base = user.email.split('@')[0]  # Get part before @
        folder_name = re.sub(r'[^a-zA-Z0-9_]', '_', email_base)
        # Limit length to avoid S3 key length issues
        folder_name = folder_name[:50]
        generated_folder = f"user_{folder_name}_{user.id}"
        logger.info(f"[DEBUG] Generated new folder for user {user.id}: {generated_folder}")
        return generated_folder
    elif tenant_id:
        generated_folder = f"tenant_{tenant_id}"
        logger.info(f"[DEBUG] Generated tenant folder: {generated_folder}")
        return generated_folder
    else:
        # Fallback to timestamp-based folder
        from datetime import datetime
        generated_folder = f"uploads_{datetime.utcnow().strftime('%Y%m%d')}"
        logger.info(f"[DEBUG] Generated fallback folder: {generated_folder}")
        return generated_folder

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
        from app.services.ats_analyzer import ats_analyzer
        import tempfile
        import os
        temp_file_path = None
        resume_text = None
        ats_analysis = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            current_app.logger.info(f"Starting resume parsing for file: {temp_file_path}")
            
            # Extract text first for ATS analysis
            resume_text = resume_parser.extract_text_from_file(temp_file_path, file_extension)
            
            # Parse the resume
            resume_data = resume_parser.parse_file(temp_file_path)
            current_app.logger.info("Resume parsing completed successfully")
            current_app.logger.info(f"Parsed data keys: {list(resume_data.keys()) if resume_data else 'None'}")
            current_app.logger.info(f"Skills found: {resume_data.get('skills', []) if resume_data else 'None'}")
            current_app.logger.info(f"Education found: {resume_data.get('education', '') if resume_data else 'None'}")
            current_app.logger.info(f"All resume data: {resume_data}")
            
            # Perform ATS analysis
            if resume_text:
                try:
                    current_app.logger.info("Starting ATS analysis...")
                    ats_analysis_result = ats_analyzer.analyze_resume(resume_text, resume_data)
                    # Convert dataclass to dict for JSON serialization
                    ats_analysis = {
                        'overall_score': ats_analysis_result.overall_score,
                        'ats_compatibility': ats_analysis_result.ats_compatibility,
                        'section_scores': {
                            k: {
                                'section': v.section,
                                'score': v.score,
                                'max_score': v.max_score,
                                'issues': v.issues or [],
                                'strengths': v.strengths or []
                            }
                            for k, v in ats_analysis_result.section_scores.items()
                        },
                        'suggestions': [
                            {
                                'category': s.category,
                                'priority': s.priority,
                                'title': s.title,
                                'description': s.description,
                                'current_state': s.current_state,
                                'recommended_action': s.recommended_action,
                                'impact': s.impact
                            }
                            for s in ats_analysis_result.suggestions
                        ],
                        'critical_issues': ats_analysis_result.critical_issues,
                        'keyword_analysis': ats_analysis_result.keyword_analysis,
                        'formatting_analysis': ats_analysis_result.formatting_analysis,
                        'structure_analysis': ats_analysis_result.structure_analysis,
                        'estimated_improvement_potential': ats_analysis_result.estimated_improvement_potential,
                        'template_suggestions': ats_analyzer.get_improved_template_suggestions(ats_analysis_result)
                    }
                    current_app.logger.info(f"ATS analysis completed. Score: {ats_analysis_result.overall_score}/100")
                except Exception as ats_error:
                    current_app.logger.error(f"ATS analysis failed: {str(ats_error)}", exc_info=True)
                    # Don't fail the upload if ATS analysis fails
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

        # Get additional form data (phone, location, visa_status, social links)
        # These can override or supplement parsed data from resume
        form_phone = request.form.get('phone', '').strip()
        form_location = request.form.get('location', '').strip()
        form_visa_status = request.form.get('visa_status', '').strip()
        form_linkedin = request.form.get('linkedin', '').strip()
        form_facebook = request.form.get('facebook', '').strip()
        form_x = request.form.get('x', '').strip()
        form_github = request.form.get('github', '').strip()
        
        # Use form data if provided, otherwise fall back to parsed data
        final_phone = form_phone or resume_data.get('phone', '')
        final_location = form_location or resume_data.get('location', '')
        
        # Create or update the base profile first
        is_new_profile = False
        if not profile:
            is_new_profile = True
            profile = CandidateProfile(
                user_id=user.id,
                full_name=profile_full_name,
                phone=final_phone,
                location=final_location,
                summary=resume_data.get('summary', ''),
                experience_years=resume_data.get('experience_years'),
                visa_status=form_visa_status if form_visa_status else None,
                resume_s3_key=s3_key,
                resume_filename=file.filename,
                resume_upload_date=datetime.utcnow()
            )
            db.session.add(profile)
            db.session.flush()  # Ensure we have profile.id
            
            # Trigger job recommendations for newly created profile
            try:
                from app.jobs.routes import get_recommended_jobs_for_user
                recommended_jobs = get_recommended_jobs_for_user(user.id, limit=10)
                logger.info(f"Found {len(recommended_jobs)} recommended jobs for new profile (user_id: {user.id})")
            except Exception as e:
                logger.error(f"Error getting recommended jobs after profile creation: {str(e)}")
                # Continue without recommended jobs - not critical
        else:
            profile.full_name = profile_full_name if profile_full_name else profile.full_name
            profile.phone = final_phone if final_phone else profile.phone
            profile.location = final_location if final_location else profile.location
            profile.summary = resume_data.get('summary', profile.summary)
            profile.experience_years = resume_data.get('experience_years', profile.experience_years)
            if form_visa_status:
                profile.visa_status = form_visa_status
            profile.resume_s3_key = s3_key
            profile.resume_filename = file.filename
            profile.resume_upload_date = datetime.utcnow()
            profile.updated_at = datetime.utcnow()
            db.session.flush()
        
        # Handle social links
        if form_linkedin or form_facebook or form_x or form_github:
            from app.models import UserSocialLinks
            social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
            if not social_links:
                social_links = UserSocialLinks(user_id=user.id)
                db.session.add(social_links)
            if form_linkedin:
                social_links.linkedin = form_linkedin
            if form_facebook:
                social_links.facebook = form_facebook
            if form_x:
                social_links.x = form_x
            if form_github:
                social_links.github = form_github
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
            current_app.logger.info(f"Processing education data: {resume_data['education'][:200]}...")
            import re
            education_text = resume_data['education']
            if education_text:
                # First try splitting by the exact separator used by resume parser (\n\n---\n\n)
                # Try multiple patterns to catch variations
                education_entries = []
                # Pattern 1: Exact match \n\n---\n\n
                if '\n\n---\n\n' in education_text:
                    education_entries = education_text.split('\n\n---\n\n')
                    education_entries = [entry.strip() for entry in education_entries if entry.strip()]
                # Pattern 2: Flexible regex with 3+ dashes
                if len(education_entries) <= 1:
                    education_entries = re.split(r'\n\s*\n\s*-{3,}\s*\n\s*\n', education_text)
                    education_entries = [entry.strip() for entry in education_entries if entry.strip()]
                
                # If no separator found, try splitting by double newlines with --- in between
                if len(education_entries) <= 1:
                    education_entries = re.split(r'\n\s*\n\s*-{3,}\s*\n', education_text)
                    education_entries = [entry.strip() for entry in education_entries if entry.strip()]
                
                # If still only one, try splitting by double newlines
                if len(education_entries) <= 1:
                    education_entries = re.split(r'\n\s*\n+', education_text)
                    education_entries = [entry.strip() for entry in education_entries if entry.strip()]
                
                # If still only one, try splitting by single newlines (fallback)
                if len(education_entries) <= 1:
                    education_entries = re.split(r'\n', education_text)
                    education_entries = [entry.strip() for entry in education_entries if entry.strip() and len(entry.strip()) > 10]

                current_app.logger.info(f"Found {len(education_entries)} education entries after splitting")
                for idx, entry in enumerate(education_entries):
                    current_app.logger.info(f"  Entry {idx+1}: {entry[:100]}...")
                for i, edu_entry in enumerate(education_entries[:10]):  # Limit to 10 education entries
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

                    # Field of study - extract more carefully to avoid getting full text
                    if 'in ' in degree.lower():
                        # Extract field of study from "Bachelor in Computer Science" pattern
                        # Stop at "from", "at", or opening parenthesis
                        field_match = re.search(r'in\s+([^,\n(]+?)(?:\s+from|\s+at|\s*\(|$)', degree, re.IGNORECASE)
                        if field_match:
                            field_of_study = field_match.group(1).strip()
                            # Remove institution names that might have been captured
                            field_of_study = re.sub(r'\s+(university|college|institute|school|engineering).*$', '', field_of_study, flags=re.IGNORECASE)
                            field_of_study = field_of_study.strip()
                            # Limit length to avoid capturing too much (should be just the field name)
                            if len(field_of_study) > 100:
                                field_of_study = field_of_study[:100].strip()
                    elif '(' in degree and ')' in degree:
                        paren_match = re.search(r'\(([^)]+)\)', degree)
                        if paren_match:
                            paren_content = paren_match.group(1).strip()
                            # Check if it's a date range (e.g., 2020-2024) - don't use as field of study
                            if not re.match(r'^\d{4}', paren_content):
                                field_of_study = paren_content
                                # Limit length
                                if len(field_of_study) > 100:
                                    field_of_study = field_of_study[:100].strip()
                    
                    # Extract dates from description if available
                    start_date = None
                    end_date = None
                    date_patterns = [
                        r'\((\d{4})\s*-\s*(\d{4})\)',
                        r'(\d{4})\s*-\s*(\d{4})',
                        r'(\d{4})\s*to\s*(\d{4})',
                    ]
                    for pattern in date_patterns:
                        match = re.search(pattern, edu_entry)
                        if match:
                            try:
                                start_year = int(match.group(1))
                                start_date = date(start_year, 1, 1)
                                if len(match.groups()) > 1 and match.group(2):
                                    end_year = int(match.group(2))
                                    end_date = date(end_year, 12, 31)
                            except (ValueError, IndexError):
                                pass
                            break

                    education = CandidateEducation(
                        profile_id=profile.id,
                        institution=institution,
                        degree=degree,
                        field_of_study=field_of_study,
                        start_date=start_date,
                        end_date=end_date,
                        description=edu_entry
                    )
                    db.session.add(education)
                    current_app.logger.info(f"Added education entry {i+1}: {degree} from {institution}, field: {field_of_study}")
        else:
            current_app.logger.warning(f"No education data found in resume for profile {profile.id}")

        # Add parsed projects
        if resume_data.get('projects'):
            import re
            # Parse projects from the extracted text
            projects_text = resume_data['projects']
            if projects_text:
                # First try splitting by the exact separator used by resume parser (\n\n---\n\n)
                # Try multiple patterns to catch variations
                project_entries = []
                # Pattern 1: Exact match \n\n---\n\n
                if '\n\n---\n\n' in projects_text:
                    project_entries = projects_text.split('\n\n---\n\n')
                    project_entries = [entry.strip() for entry in project_entries if entry.strip()]
                # Pattern 2: Flexible regex with 3+ dashes
                if len(project_entries) <= 1:
                    project_entries = re.split(r'\n\s*\n\s*-{3,}\s*\n\s*\n', projects_text)
                    project_entries = [entry.strip() for entry in project_entries if entry.strip()]
                
                # If no separator found, try splitting by double newlines with --- in between
                if len(project_entries) <= 1:
                    project_entries = re.split(r'\n\s*\n\s*-{3,}\s*\n', projects_text)
                    project_entries = [entry.strip() for entry in project_entries if entry.strip()]
                
                # If still only one, try splitting by double newlines
                if len(project_entries) <= 1:
                    project_entries = re.split(r'\n\s*\n+', projects_text)
                    project_entries = [entry.strip() for entry in project_entries if entry.strip()]

                current_app.logger.info(f"Found {len(project_entries)} project entries after splitting by separators")

                # If still only one, try splitting by inline project titles with colons
                # Handles patterns like: "Shoe Store Application: ... Instagram Clone: ... Zomato Clone: ..."
                if len(project_entries) <= 1 and len(projects_text) > 200 and projects_text.count(':') >= 2:
                    title_pattern = re.compile(
                        r'(?=(?:[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+'
                        r'(?:Application|App|Project|Clone|Website|System|Platform)\s*:))'
                    )
                    parts = [p.strip() for p in title_pattern.split(projects_text) if p.strip()]
                    current_app.logger.info(f"Inline title split produced {len(parts)} parts")
                    if len(parts) > 1:
                        recombined = []
                        buffer = ""
                        for part in parts:
                            if title_pattern.match(part) and buffer:
                                if len(buffer.strip()) > 10:
                                    recombined.append(buffer.strip())
                                buffer = part
                            else:
                                buffer = (buffer + " " + part).strip() if buffer else part
                        if buffer and len(buffer.strip()) > 10:
                            recombined.append(buffer.strip())
                        if len(recombined) > 1:
                            current_app.logger.info(f"Inline title recombination produced {len(recombined)} project entries")
                            project_entries = recombined

                # If still only one, try splitting by semicolons or single newlines (fallback)
                if len(project_entries) <= 1:
                    project_entries = re.split(r'[;\n]', projects_text)
                    project_entries = [entry.strip() for entry in project_entries if entry.strip() and len(entry.strip()) > 10]

                current_app.logger.info(f"Final project entry count after all splitting strategies: {len(project_entries)}")
                for i, project_entry in enumerate(project_entries[:10]):  # Limit to 10 projects
                    project_entry = project_entry.strip()
                    if project_entry and len(project_entry) > 10 and _is_likely_project_line(project_entry):
                        # Extract project name and description
                        # Look for common project title patterns
                        name = None
                        description = project_entry
                        
                        # Try to find project name at the start (before colon or first line)
                        lines = project_entry.split('\n')
                        first_line = lines[0].strip() if lines else project_entry
                        
                        if ':' in first_line and len(first_line) < 100:
                            # Split by colon - first part is likely the name
                            name_parts = first_line.split(':', 1)
                            if len(name_parts) == 2:
                                name = name_parts[0].strip()
                                # Remove common prefixes
                                name = re.sub(r'^(project|title|name)\s*:?\s*', '', name, flags=re.IGNORECASE)
                                description = '\n'.join([name_parts[1].strip()] + lines[1:]).strip()
                        elif len(first_line) < 80 and not any(word in first_line.lower() for word in ['technology', 'technologies', 'details', 'description']):
                            # First line might be the project name
                            name = first_line
                            description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else project_entry
                        else:
                            # Use first 50 chars as name
                            name = project_entry[:50].strip()
                            description = project_entry
                        
                        # Clean up name
                        if name:
                            name = name.strip()
                            # Remove trailing punctuation
                            name = re.sub(r'[.,;:]+$', '', name)
                            if len(name) > 100:
                                name = name[:100]
                        
                        if not name or len(name) < 3:
                            name = f"Project {i+1}"
                            
                        project = CandidateProject(
                            profile_id=profile.id,
                            name=name,
                            description=description
                        )
                        db.session.add(project)
                        current_app.logger.info(f"Added project entry {i+1}: {name}")

        # Add parsed certifications
        if resume_data.get('certifications'):
            import re
            # Parse certifications from the extracted text
            certs_text = resume_data['certifications']
            if certs_text:
                # First try splitting by the exact separator used by resume parser (\n\n---\n\n)
                # Try multiple patterns to catch variations
                cert_entries = []
                # Pattern 1: Exact match \n\n---\n\n
                if '\n\n---\n\n' in certs_text:
                    cert_entries = certs_text.split('\n\n---\n\n')
                    cert_entries = [entry.strip() for entry in cert_entries if entry.strip()]
                # Pattern 2: Flexible regex with 3+ dashes
                if len(cert_entries) <= 1:
                    cert_entries = re.split(r'\n\s*\n\s*-{3,}\s*\n\s*\n', certs_text)
                    cert_entries = [entry.strip() for entry in cert_entries if entry.strip()]
                
                # If no separator found, try splitting by double newlines with --- in between
                if len(cert_entries) <= 1:
                    cert_entries = re.split(r'\n\s*\n\s*-{3,}\s*\n', certs_text)
                    cert_entries = [entry.strip() for entry in cert_entries if entry.strip()]
                
                # If still only one, try splitting by double newlines
                if len(cert_entries) <= 1:
                    cert_entries = re.split(r'\n\s*\n+', certs_text)
                    cert_entries = [entry.strip() for entry in cert_entries if entry.strip()]
                
                # If still only one, try splitting by semicolons, commas, or single newlines (fallback)
                if len(cert_entries) <= 1:
                    # Try splitting by comma first (common in certifications like "React.js (Basic) - HackerRank, Node.js (Basic) - HackerRank")
                    # Split by comma followed by space and capital letter (likely new cert)
                    cert_entries = re.split(r',\s+(?=[A-Z][a-z])', certs_text)
                    cert_entries = [entry.strip() for entry in cert_entries if entry.strip() and len(entry.strip()) > 5]
                    
                    # If still only one, try newlines
                    if len(cert_entries) <= 1:
                        cert_entries = re.split(r'[;\n]', certs_text)
                        cert_entries = [entry.strip() for entry in cert_entries if entry.strip() and len(entry.strip()) > 5]
                
                current_app.logger.info(f"Found {len(cert_entries)} certification entries after splitting")
                for idx, entry in enumerate(cert_entries):
                    current_app.logger.info(f"  Cert {idx+1}: {entry[:100]}...")
                for i, cert_entry in enumerate(cert_entries[:10]):  # Limit to 10 certifications
                    cert_entry = cert_entry.strip()
                    if cert_entry and len(cert_entry) > 5:
                        # Extract certification name and organization
                        name = cert_entry
                        organization = None
                        
                        # Try to extract organization from common patterns
                        # Pattern 1: "Cert Name - Organization" or "Cert Name (Organization)"
                        if ' - ' in cert_entry:
                            parts = cert_entry.split(' - ', 1)
                            name = parts[0].strip()
                            organization = parts[1].strip() if len(parts) > 1 else None
                        elif '(' in cert_entry and ')' in cert_entry:
                            # Extract text in parentheses as organization
                            org_match = re.search(r'\(([^)]+)\)', cert_entry)
                            if org_match:
                                organization = org_match.group(1).strip()
                                # Remove organization from name
                                name = re.sub(r'\s*\([^)]+\)', '', cert_entry).strip()
                        
                        # Pattern 2: Look for common cert provider names
                        if not organization:
                            cert_providers = ['AWS', 'Amazon', 'Google', 'Microsoft', 'Oracle', 'Cisco', 'CompTIA', 
                                            'HackerRank', 'Coursera', 'Udemy', 'edX', 'LinkedIn', 'IBM', 'SAP']
                            for provider in cert_providers:
                                if provider.lower() in cert_entry.lower():
                                    # Try to extract the provider name
                                    provider_match = re.search(rf'({provider}[^,\n-]*)', cert_entry, re.IGNORECASE)
                                    if provider_match:
                                        org_text = provider_match.group(1).strip()
                                        # Check if it's part of the cert name or separate
                                        if org_text.lower() not in name.lower()[:50]:
                                            organization = org_text
                                    break
                        
                        # Clean up name - remove common prefixes
                        name = re.sub(r'^(certification|certificate|certified)\s*:?\s*', '', name, flags=re.IGNORECASE)
                        name = name.strip()
                        
                        # If name is too long, truncate it
                        if len(name) > 200:
                            name = name[:200]
                        
                        certification = CandidateCertification(
                            profile_id=profile.id,
                            name=name,
                            issuing_organization=organization
                        )
                        db.session.add(certification)
                        current_app.logger.info(f"Added certification entry {i+1}: {name} from {organization or 'Unknown'}")
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
            
            # Store ATS analysis results in profile
            if ats_analysis:
                # Store ATS analysis in benchmarking_data JSON field
                if not profile.benchmarking_data:
                    profile.benchmarking_data = {}
                profile.benchmarking_data['ats_analysis'] = ats_analysis
                profile.benchmarking_data['ats_analysis_date'] = datetime.utcnow().isoformat()
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
        
        # Commit all changes (skills, education, projects, certifications, experience)
        db.session.commit()
        
        # Get recommended jobs for the profile (after all data is committed)
        try:
            from app.jobs.routes import get_recommended_jobs_for_user
            recommended_jobs = get_recommended_jobs_for_user(user.id, limit=10)
            if recommended_jobs:
                response_data['recommended_jobs'] = recommended_jobs
                logger.info(f"Found {len(recommended_jobs)} recommended jobs for profile (user_id: {user.id})")
        except Exception as e:
            logger.error(f"Error getting recommended jobs after resume upload: {str(e)}")
            # Continue without recommended jobs - not critical
        
        # Add ATS analysis if available
        if ats_analysis:
            response_data['ats_analysis'] = ats_analysis
        
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

@talent_bp.route('/profile/ats-analysis', methods=['GET'])
def get_ats_analysis():
    """Get stored ATS analysis for the current user's profile"""
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
        
        # Retrieve ATS analysis from benchmarking_data
        ats_analysis = None
        analysis_date = None
        if profile.benchmarking_data and isinstance(profile.benchmarking_data, dict):
            ats_analysis = profile.benchmarking_data.get('ats_analysis')
            analysis_date = profile.benchmarking_data.get('ats_analysis_date')
        
        # Return 200 even if no analysis exists (frontend handles this gracefully)
        # This prevents 404 errors when the route exists but data doesn't
        return jsonify({
            'ats_analysis': ats_analysis,
            'analysis_date': analysis_date,
            'has_analysis': ats_analysis is not None
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error retrieving ATS analysis: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to retrieve ATS analysis'}), 500

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
        if 'target_role_for_insights' in data:
            profile.target_role_for_insights = data['target_role_for_insights']

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

@talent_bp.route('/profile/update-resume', methods=['POST'])
def update_resume_from_profile():
    """Update resume and trigger ATS re-analysis based on current profile data"""
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

        # Generate resume text from profile data
        from app.services.resume_parser import resume_parser
        from app.services.ats_analyzer import ats_analyzer
        
        # Build resume text from profile data
        resume_text_parts = []
        
        # Personal Information
        resume_text_parts.append(f"{profile.full_name or 'Name'}")
        if profile.phone:
            resume_text_parts.append(f"Phone: {profile.phone}")
        if profile.location:
            resume_text_parts.append(f"Location: {profile.location}")
        if user.email:
            resume_text_parts.append(f"Email: {user.email}")
        resume_text_parts.append("")
        
        # Summary
        if profile.summary:
            resume_text_parts.append("PROFESSIONAL SUMMARY")
            resume_text_parts.append(profile.summary)
            resume_text_parts.append("")
        
        # Skills
        skills = CandidateSkill.query.filter_by(profile_id=profile.id).all()
        if skills:
            resume_text_parts.append("SKILLS")
            skill_names = [skill.skill_name for skill in skills]
            resume_text_parts.append(", ".join(skill_names))
            resume_text_parts.append("")
        
        # Work Experience
        experiences = CandidateExperience.query.filter_by(profile_id=profile.id).order_by(
            CandidateExperience.start_date.desc()
        ).all()
        if experiences:
            resume_text_parts.append("WORK EXPERIENCE")
            for exp in experiences:
                resume_text_parts.append(f"{exp.position} at {exp.company}")
                if exp.start_date:
                    date_str = exp.start_date.strftime('%Y-%m') if hasattr(exp.start_date, 'strftime') else str(exp.start_date)
                    end_str = "Present" if exp.is_current else (exp.end_date.strftime('%Y-%m') if exp.end_date and hasattr(exp.end_date, 'strftime') else str(exp.end_date) if exp.end_date else "")
                    resume_text_parts.append(f"{date_str} - {end_str}")
                if exp.description:
                    resume_text_parts.append(exp.description)
                if exp.achievements:
                    resume_text_parts.append(f"Achievements: {exp.achievements}")
                resume_text_parts.append("")
        
        # Education
        educations = CandidateEducation.query.filter_by(profile_id=profile.id).order_by(
            CandidateEducation.start_date.desc() if CandidateEducation.start_date else CandidateEducation.id.desc()
        ).all()
        if educations:
            resume_text_parts.append("EDUCATION")
            for edu in educations:
                edu_line = f"{edu.degree}"
                if edu.field_of_study:
                    edu_line += f" in {edu.field_of_study}"
                edu_line += f" from {edu.institution}"
                resume_text_parts.append(edu_line)
                if edu.start_date:
                    start_str = edu.start_date.strftime('%Y') if hasattr(edu.start_date, 'strftime') else str(edu.start_date)
                    end_str = edu.end_date.strftime('%Y') if edu.end_date and hasattr(edu.end_date, 'strftime') else str(edu.end_date) if edu.end_date else "Present"
                    resume_text_parts.append(f"{start_str} - {end_str}")
                if edu.gpa:
                    resume_text_parts.append(f"GPA: {edu.gpa}")
                if edu.description:
                    resume_text_parts.append(edu.description)
                resume_text_parts.append("")
        
        # Certifications
        certifications = CandidateCertification.query.filter_by(profile_id=profile.id).all()
        if certifications:
            resume_text_parts.append("CERTIFICATIONS")
            for cert in certifications:
                cert_line = cert.name
                if cert.issuing_organization:
                    cert_line += f" - {cert.issuing_organization}"
                resume_text_parts.append(cert_line)
                if cert.issue_date:
                    issue_str = cert.issue_date.strftime('%Y-%m') if hasattr(cert.issue_date, 'strftime') else str(cert.issue_date)
                    resume_text_parts.append(f"Issued: {issue_str}")
                resume_text_parts.append("")
        
        # Projects
        projects = CandidateProject.query.filter_by(profile_id=profile.id).all()
        if projects:
            resume_text_parts.append("PROJECTS")
            for proj in projects:
                resume_text_parts.append(proj.name)
                if proj.description:
                    resume_text_parts.append(proj.description)
                if proj.technologies:
                    resume_text_parts.append(f"Technologies: {proj.technologies}")
                resume_text_parts.append("")
        
        resume_text = "\n".join(resume_text_parts)
        
        # Perform ATS analysis on the generated resume text
        ats_analysis = None
        if resume_text:
            try:
                current_app.logger.info("Performing ATS analysis on updated profile resume...")
                
                # Create parsed_data dict from profile
                parsed_data = {
                    'full_name': profile.full_name,
                    'phone': profile.phone,
                    'location': profile.location,
                    'summary': profile.summary,
                    'experience_years': profile.experience_years,
                    'skills': [skill.skill_name for skill in skills],
                    'education': "\n\n---\n\n".join([
                        f"{edu.degree} from {edu.institution}" + 
                        (f" ({edu.field_of_study})" if edu.field_of_study else "") +
                        (f" - {edu.description}" if edu.description else "")
                        for edu in educations
                    ]) if educations else '',
                    'work_experience': "\n\n---\n\n".join([
                        f"{exp.position} at {exp.company}\n{exp.description or ''}\n{exp.achievements or ''}"
                        for exp in experiences
                    ]) if experiences else '',
                    'certifications': "\n\n---\n\n".join([
                        f"{cert.name} - {cert.issuing_organization or 'N/A'}"
                        for cert in certifications
                    ]) if certifications else '',
                    'projects': "\n\n---\n\n".join([
                        f"{proj.name}\n{proj.description or ''}"
                        for proj in projects
                    ]) if projects else ''
                }
                
                ats_analysis_result = ats_analyzer.analyze_resume(resume_text, parsed_data)
                
                # Convert to dict for JSON serialization
                ats_analysis = {
                    'overall_score': ats_analysis_result.overall_score,
                    'ats_compatibility': ats_analysis_result.ats_compatibility,
                    'section_scores': {
                        k: {
                            'section': v.section,
                            'score': v.score,
                            'max_score': v.max_score,
                            'issues': v.issues or [],
                            'strengths': v.strengths or []
                        }
                        for k, v in ats_analysis_result.section_scores.items()
                    },
                    'suggestions': [
                        {
                            'category': s.category,
                            'priority': s.priority,
                            'title': s.title,
                            'description': s.description,
                            'current_state': s.current_state,
                            'recommended_action': s.recommended_action,
                            'impact': s.impact
                        }
                        for s in ats_analysis_result.suggestions
                    ],
                    'critical_issues': ats_analysis_result.critical_issues,
                    'keyword_analysis': ats_analysis_result.keyword_analysis,
                    'formatting_analysis': ats_analysis_result.formatting_analysis,
                    'structure_analysis': ats_analysis_result.structure_analysis,
                    'estimated_improvement_potential': ats_analysis_result.estimated_improvement_potential,
                    'template_suggestions': ats_analyzer.get_improved_template_suggestions(ats_analysis_result)
                }
                
                # Store ATS analysis in profile
                if not profile.benchmarking_data:
                    profile.benchmarking_data = {}
                profile.benchmarking_data['ats_analysis'] = ats_analysis
                profile.benchmarking_data['ats_analysis_date'] = datetime.utcnow().isoformat()
                
                current_app.logger.info(f"ATS analysis completed. Score: {ats_analysis_result.overall_score}/100")
            except Exception as ats_error:
                current_app.logger.error(f"ATS analysis failed: {str(ats_error)}", exc_info=True)
                # Don't fail the update if ATS analysis fails
        
        profile.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Resume updated successfully from profile data',
            'ats_analysis': ats_analysis,
            'resume_text_length': len(resume_text),
            'sections_included': {
                'personal_info': bool(profile.full_name),
                'summary': bool(profile.summary),
                'skills': len(skills) > 0,
                'experience': len(experiences) > 0,
                'education': len(educations) > 0,
                'certifications': len(certifications) > 0,
                'projects': len(projects) > 0
            }
        }), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Update resume from profile error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to update resume from profile'}), 500

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

@talent_bp.route('/contact-candidate', methods=['POST'])
def contact_candidate():
    """Send message to candidate via Email, SMS, or WhatsApp (legacy endpoint - uses new communication system)"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Only employers, recruiters, admins, or owners can contact candidates
        if user.role not in ['employer', 'recruiter', 'admin', 'owner']:
            return jsonify({'error': 'Access denied'}), 403

        data = request.get_json()
        candidate_id = data.get('candidate_id')
        candidate_name = data.get('candidate_name', '')
        candidate_email = data.get('candidate_email', '')
        candidate_phone = data.get('candidate_phone', '')
        job_title = data.get('job_title', '')
        job_description = data.get('job_description', '')
        company_name = data.get('company_name', user.company_name or 'Our Company')
        
        # Get channel (default to email for backward compatibility)
        channel = data.get('channel', 'email')
        email_provider = data.get('email_provider', 'sendgrid')  # 'sendgrid' or 'smtp'
        
        if not candidate_id:
            return jsonify({'error': 'Candidate ID is required'}), 400
        
        # Validate channel-specific requirements
        if channel == 'email' and not candidate_email:
            return jsonify({'error': 'Candidate email is required for email channel'}), 400
        if channel in ['sms', 'whatsapp'] and not candidate_phone:
            return jsonify({'error': f'Candidate phone is required for {channel} channel'}), 400

        # Use new communication service
        from app.communications.service import send_candidate_message
        
        # Prepare variables for template
        variables = {
            'candidate_name': candidate_name,
            'job_title': job_title,
            'company_name': company_name,
            'job_description': job_description,
            'sender_name': user.email.split('@')[0] if user.email else 'Recruiter',
            'contact_email': user.email
        }
        
        # Send message
        result = send_candidate_message(
            user_id=user.id,
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            candidate_phone=candidate_phone,
            channel=channel,
            template_id=data.get('template_id'),  # Optional template
            custom_message=data.get('custom_message'),  # Optional custom message
            subject=data.get('subject'),  # Optional subject for email
            email_provider=email_provider,
            variables=variables
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': f'Message sent successfully via {channel}',
                'communication_id': result.get('communication_id')
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to send message')
            }), 500

    except Exception as e:
        current_app.logger.error(f"Contact candidate error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to contact candidate: {str(e)}'
        }), 500

@talent_bp.route('/upload-job-description-file', methods=['POST'])
def upload_job_description_file():
    """
    Upload job description files (PDF, DOCX, images, CSV) to S3 with user/tenant-specific folders.
    Files are stored in career_resume/{user_folder}/ format and are only visible to that user.
    """
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

        if not allowed_job_description_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PDF, DOC, DOCX, TXT, RTF, CSV, JPG, PNG, GIF, BMP, TIFF'}), 400

        # Get user-specific folder name
        user_folder = get_user_folder_name(user, tenant_id)
        
        # Generate unique filename for S3
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'bin'
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        secure_name = secure_filename(file.filename)
        # Remove extension from secure_name and add it back after unique_id
        name_without_ext = secure_name.rsplit('.', 1)[0] if '.' in secure_name else secure_name
        unique_filename = f"{timestamp}_{unique_id}_{name_without_ext[:50]}.{file_extension}"
        
        # Store in user-specific folder: career_resume/{user_folder}/{filename}
        s3_key = f"{RESUME_PREFIX}{user_folder}/{unique_filename}"
        
        # Debug logging: Show where file is being stored
        current_app.logger.info(f"[DEBUG] File upload details:")
        current_app.logger.info(f"[DEBUG]   - Original filename: {file.filename}")
        current_app.logger.info(f"[DEBUG]   - Secure filename: {secure_name}")
        current_app.logger.info(f"[DEBUG]   - User folder: {user_folder}")
        current_app.logger.info(f"[DEBUG]   - User ID: {user.id}")
        current_app.logger.info(f"[DEBUG]   - Tenant ID: {tenant_id}")
        current_app.logger.info(f"[DEBUG]   - S3 Bucket: {S3_BUCKET}")
        current_app.logger.info(f"[DEBUG]   - S3 Key (full path): {s3_key}")
        current_app.logger.info(f"[DEBUG]   - Full S3 URL: s3://{S3_BUCKET}/{s3_key}")
        current_app.logger.info(f"[DEBUG]   - Stored filename: {unique_filename}")

        # Read file content into memory to avoid file pointer issues
        try:
            file.seek(0)
            file_content = file.read()
            if not file_content:
                return jsonify({'error': 'File is empty or corrupted'}), 400
        except Exception as e:
            current_app.logger.error(f"File read error: {str(e)}")
            return jsonify({'error': 'Failed to read file content'}), 400

        # Update debug log with actual file size
        current_app.logger.info(f"[DEBUG]   - File size: {len(file_content)} bytes")

        # Upload file to S3 with private ACL (only accessible to the user)
        try:
            from io import BytesIO
            file_obj = BytesIO(file_content)
            file_obj.seek(0)
            s3_client.upload_fileobj(
                file_obj,
                S3_BUCKET,
                s3_key,
                ExtraArgs={
                    'ContentType': file.content_type or 'application/octet-stream',
                    'ACL': 'private'
                }
            )
            current_app.logger.info(f"[DEBUG] ✅ Successfully uploaded file to s3://{S3_BUCKET}/{s3_key} for user {user.id}")
        except Exception as e:
            current_app.logger.error(f"S3 upload failed: {str(e)}")
            return jsonify({'error': 'Failed to upload file to cloud storage'}), 500

        # Return success response with S3 key
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            's3_key': s3_key,
            's3_url': f"s3://{S3_BUCKET}/{s3_key}",
            'filename': file.filename,
            'file_size': len(file_content),
            'user_folder': user_folder
        }), 200

    except Exception as e:
        current_app.logger.error(f"Job description file upload error: {str(e)}")
        error_message = 'Internal server error'
        if 'I/O operation on closed file' in str(e):
            error_message = 'File processing error. Please try uploading the file again.'
        elif 'S3' in str(e):
            error_message = 'File upload failed. Please check your connection and try again.'
        return jsonify({'error': error_message}), 500

@talent_bp.route('/upload-job-description-files', methods=['POST'])
def upload_job_description_files():
    """
    Upload multiple job description files (PDF, DOCX, images, CSV) to S3 with user/tenant-specific folders.
    Files are stored in career_resume/{user_folder}/ format and are only visible to that user.
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Check if files are present
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400

        # Limit number of files per upload (max 200 files)
        MAX_FILES = 200
        if len(files) > MAX_FILES:
            return jsonify({'error': f'Too many files. Maximum {MAX_FILES} files allowed per upload.'}), 400

        # Get user-specific folder name
        user_folder = get_user_folder_name(user, tenant_id)
        
        successful_uploads = []
        failed_uploads = []

        for file in files:
            if file.filename == '':
                continue

            if not allowed_job_description_file(file.filename):
                failed_uploads.append({
                    'filename': file.filename,
                    'error': 'Invalid file type'
                })
                continue

            try:
                # Generate unique filename for S3
                file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'bin'
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                unique_id = str(uuid.uuid4())[:8]
                secure_name = secure_filename(file.filename)
                name_without_ext = secure_name.rsplit('.', 1)[0] if '.' in secure_name else secure_name
                unique_filename = f"{timestamp}_{unique_id}_{name_without_ext[:50]}.{file_extension}"
                
                # Store in user-specific folder: career_resume/{user_folder}/{filename}
                s3_key = f"{RESUME_PREFIX}{user_folder}/{unique_filename}"

                # Read file content
                file.seek(0)
                file_content = file.read()
                if not file_content:
                    failed_uploads.append({
                        'filename': file.filename,
                        'error': 'File is empty'
                    })
                    continue

                # Upload file to S3
                from io import BytesIO
                file_obj = BytesIO(file_content)
                file_obj.seek(0)
                s3_client.upload_fileobj(
                    file_obj,
                    S3_BUCKET,
                    s3_key,
                    ExtraArgs={
                        'ContentType': file.content_type or 'application/octet-stream',
                        'ACL': 'private'
                    }
                )
                
                successful_uploads.append({
                    'filename': file.filename,
                    's3_key': s3_key,
                    's3_url': f"s3://{S3_BUCKET}/{s3_key}",
                    'file_size': len(file_content)
                })
                
                current_app.logger.info(f"Uploaded job description file to s3://{S3_BUCKET}/{s3_key} for user {user.id}")
                
            except Exception as e:
                current_app.logger.error(f"Failed to upload {file.filename}: {str(e)}")
                failed_uploads.append({
                    'filename': file.filename,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'message': f'Uploaded {len(successful_uploads)} file(s) successfully',
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'user_folder': user_folder
        }), 200

    except Exception as e:
        current_app.logger.error(f"Job description files upload error: {str(e)}")
        return jsonify({'error': f'Failed to upload files: {str(e)}'}), 500

@talent_bp.route('/list-user-files', methods=['GET'])
def list_user_files():
    """
    List all files uploaded by the user (resumes, CSV, images, etc.) from their S3 folder.
    Returns files stored in career_resume/{user_folder}/ format.
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get user-specific folder name
        user_folder = get_user_folder_name(user, tenant_id)
        
        # List files in the user's folder
        folder_prefix = f"{RESUME_PREFIX}{user_folder}/"
        
        try:
            # List objects in S3 with the user's folder prefix
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=folder_prefix,
                MaxKeys=1000  # Limit to 1000 files per user
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extract filename from full S3 key
                    s3_key = obj['Key']
                    filename = s3_key.replace(folder_prefix, '')
                    
                    # Skip if it's a folder marker (ends with /)
                    if filename.endswith('/'):
                        continue
                    
                    # Build a short-lived pre-signed HTTPS URL so the browser can download the file
                    try:
                        presigned_url = s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
                            ExpiresIn=3600,  # 1 hour
                        )
                    except Exception as presign_error:
                        current_app.logger.error(f"Failed to generate presigned URL for {s3_key}: {presign_error}")
                        presigned_url = None

                    # Get file metadata
                    file_info = {
                        'filename': filename,
                        's3_key': s3_key,
                        # Frontend expects s3_url; keep the key but provide an HTTPS URL instead of s3://
                        's3_url': presigned_url,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat() if 'LastModified' in obj else None,
                        'file_type': filename.split('.')[-1].lower() if '.' in filename else 'unknown',
                        'user_folder': user_folder
                    }
                    files.append(file_info)
            
            # Sort by last modified (newest first)
            files.sort(key=lambda x: x['last_modified'] or '', reverse=True)
            
            current_app.logger.info(f"[DEBUG] Listed {len(files)} files for user {user.id} from folder {user_folder}")
            
            return jsonify({
                'success': True,
                'files': files,
                'total_files': len(files),
                'user_folder': user_folder,
                'folder_path': folder_prefix
            }), 200
            
        except Exception as e:
            current_app.logger.error(f"S3 list error: {str(e)}")
            return jsonify({'error': f'Failed to list files: {str(e)}'}), 500

    except Exception as e:
        current_app.logger.error(f"List user files error: {str(e)}")
        return jsonify({'error': f'Failed to list files: {str(e)}'}), 500

@talent_bp.route('/delete-user-file', methods=['DELETE'])
def delete_user_file():
    """
    Delete a file uploaded by the user from their S3 folder.
    """
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get file S3 key from request
        data = request.get_json()
        if not data or 's3_key' not in data:
            return jsonify({'error': 's3_key is required'}), 400
        
        s3_key = data['s3_key']
        
        # Verify the file belongs to this user
        user_folder = get_user_folder_name(user, tenant_id)
        folder_prefix = f"{RESUME_PREFIX}{user_folder}/"
        
        if not s3_key.startswith(folder_prefix):
            return jsonify({'error': 'File does not belong to this user'}), 403
        
        try:
            # Delete file from S3
            s3_client.delete_object(
                Bucket=S3_BUCKET,
                Key=s3_key
            )
            
            current_app.logger.info(f"[DEBUG] Deleted file {s3_key} for user {user.id}")
            
            return jsonify({
                'success': True,
                'message': 'File deleted successfully',
                's3_key': s3_key
            }), 200
            
        except Exception as e:
            current_app.logger.error(f"S3 delete error: {str(e)}")
            return jsonify({'error': f'Failed to delete file: {str(e)}'}), 500

    except Exception as e:
        current_app.logger.error(f"Delete user file error: {str(e)}")
        return jsonify({'error': f'Failed to delete file: {str(e)}'}), 500

@talent_bp.route('/suggest-candidates', methods=['POST', 'OPTIONS'])
def suggest_candidates():
    """
    Suggest candidates based on job description.
    Accepts job description and returns matching candidates.
    """
    import json
    import re
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get job description from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract job information
        job_description = data.get('job_description', '')
        job_title = data.get('job_title', '')
        company_name = data.get('company_name', '')
        location = data.get('location', '')
        
        # Build comprehensive job description
        job_description_parts = []
        
        if job_title:
            job_description_parts.append(f"Job Title: {job_title}")
        
        if company_name:
            job_description_parts.append(f"Company: {company_name}")
        
        if location:
            job_description_parts.append(f"Location: {location}")
        
        if job_description:
            job_description_parts.append(f"\nJob Description:\n{job_description}")
        
        # Combine into full job description
        full_job_description = "\n".join(job_description_parts)
        
        if not full_job_description.strip():
            return jsonify({'error': 'Job description is required'}), 400
        
        # Get query parameters
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        limit = int(request.args.get('limit', 5))  # Default to 5, but can be customized
        
        # Use semantic matching algorithm from service.py
        try:
            from app.search.service import semantic_match
            
            logger.info(f"Using semantic matching algorithm for job suggestion: {job_title or 'Untitled'}")
            
            # Call semantic match with the job description
            result = semantic_match(full_job_description, top_k=20)  # Fetch 20, then take top N
            
            if not result or not result.get('results'):
                logger.warning(f"Semantic match returned no results for job: {job_title}")
                return jsonify({
                    'candidates': [],
                    'suggested_candidates': [],
                    'message': 'No matching candidates found'
                }), 200
            
            # Format candidates from semantic match results
            formatted_candidates = []
            for candidate in result.get('results', []):
                try:
                    # Extract candidate data (handle different field name variations)
                    candidate_email = candidate.get('email') or candidate.get('Email') or ''
                    candidate_name = (
                        candidate.get('full_name') or 
                        candidate.get('FullName') or 
                        candidate.get('name') or 
                        candidate.get('Name') or 
                        'Unknown'
                    )
                    
                    # Extract skills
                    candidate_skills = candidate.get('skills') or candidate.get('Skills') or []
                    if isinstance(candidate_skills, str):
                        candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                    elif not isinstance(candidate_skills, list):
                        candidate_skills = []
                    
                    # Extract experience years
                    experience_years = 0
                    exp_sources = [
                        candidate.get('experience_years'),
                        candidate.get('total_experience_years'),
                        candidate.get('Experience'),
                        candidate.get('experience')
                    ]
                    for exp_source in exp_sources:
                        if exp_source is not None:
                            try:
                                if isinstance(exp_source, (int, float)):
                                    experience_years = int(exp_source)
                                    break
                                elif isinstance(exp_source, str):
                                    # Try to extract number from string
                                    match = re.search(r'\d+', exp_source)
                                    if match:
                                        experience_years = int(match.group())
                                        break
                            except:
                                pass
                    
                    # Get match score/percentage
                    match_score = (
                        candidate.get('match_percentage') or 
                        candidate.get('Score') or 
                        candidate.get('score') or 
                        0
                    )
                    if isinstance(match_score, str):
                        try:
                            match_score = float(match_score.replace('%', ''))
                        except:
                            match_score = 0
                    
                    # Format candidate data for frontend
                    formatted_candidate = {
                        'id': candidate_email or f"candidate_{len(formatted_candidates)}",
                        'full_name': candidate_name,
                        'name': candidate_name,  # Also include 'name' for compatibility
                        'email': candidate_email,
                        'phone': candidate.get('phone') or candidate.get('Phone') or '',
                        'location': candidate.get('location') or candidate.get('Location') or '',
                        'experience_years': experience_years,
                        'skills': candidate_skills,
                        'summary': (
                            candidate.get('summary') or 
                            candidate.get('description') or 
                            candidate.get('resumeText') or 
                            candidate.get('resume_text') or 
                            ''
                        ),
                        'match_score': match_score / 100.0 if match_score > 1 else match_score,  # Normalize to 0-1
                        'matching_skills_count': len(candidate_skills)
                    }
                    
                    formatted_candidates.append(formatted_candidate)
                    
                except Exception as e:
                    logger.warning(f"Error formatting candidate from semantic match: {str(e)}")
                    continue
            
            # Sort by match score and take only top N
            formatted_candidates.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            top_candidates = formatted_candidates[:limit]
            
            logger.info(f"Found {len(top_candidates)} suggested candidates (top {limit}) using semantic matching")
            
            return jsonify({
                'candidates': top_candidates,
                'suggested_candidates': top_candidates,  # Also include for compatibility
                'total_matched': len(formatted_candidates),
                'algorithm_used': result.get('algorithm_used', 'Semantic Matching Algorithm'),
                'cached': False
            }), 200
            
        except ImportError as e:
            logger.error(f"Failed to import semantic_match from service: {str(e)}")
            return jsonify({
                'candidates': [],
                'suggested_candidates': [],
                'error': 'Semantic matching algorithm not available',
                'message': 'Please ensure the search service is properly configured'
            }), 500
        except Exception as e:
            logger.error(f"Error using semantic matching algorithm: {str(e)}", exc_info=True)
            return jsonify({
                'candidates': [],
                'suggested_candidates': [],
                'error': 'Failed to get suggested candidates using semantic matching',
                'message': str(e)
            }), 500
        
    except Exception as e:
        logger.error(f"Error in suggest_candidates: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get suggested candidates'}), 500
