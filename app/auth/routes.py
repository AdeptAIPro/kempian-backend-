import re
from flask import Blueprint, request, jsonify, send_file, current_app
from app.simple_logger import get_logger
from werkzeug.exceptions import HTTPException
from .cognito import cognito_signup, cognito_confirm_signup, cognito_login, cognito_admin_update_user_attributes
from .cognito import cognito_client, COGNITO_USER_POOL_ID, COGNITO_REGION
import os
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import random
import string
from urllib.parse import quote
from app.utils import get_current_user, get_current_user_flexible
from app.models import db, Tenant, User, Plan, UserSocialLinks, UserImage, UserFunctionalityPreferences, OnboardingSubmission, UserBankAccount, CandidateProfile, EmployeeProfile
from app.search.routes import get_user_from_jwt
from sqlalchemy.exc import SQLAlchemyError
from jose import jwt
import requests
import json
from app.utils.trial_manager import create_user_trial
from app.utils.admin_activity_decorator import log_admin_login_activity
import base64
import io
from PIL import Image
from app.auth.unconfirmed_handler import (
    check_user_status, 
    resend_confirmation_code, 
    initiate_password_reset, 
    confirm_signup_with_reset,
    get_recovery_options
)

logger = get_logger("auth")

auth_bp = Blueprint('auth', __name__)

def get_frontend_url():
    """Get the frontend URL, defaulting to localhost when running locally"""
    # If FRONTEND_URL is explicitly set, use it
    frontend_url = os.getenv('FRONTEND_URL')
    if frontend_url:
        return frontend_url
    
    # Check if we're running in development/local environment
    flask_env = os.getenv('FLASK_ENV', '').lower()
    flask_debug = os.getenv('FLASK_DEBUG', '').lower()
    is_development = (
        flask_env == 'development' or 
        flask_debug == 'true' or 
        flask_debug == '1' or
        os.getenv('ENVIRONMENT', '').lower() == 'development' or
        os.getenv('ENV', '').lower() == 'development'
    )
    
    # Default to localhost for development, production URL for production
    if is_development:
        # Try common local development ports
        local_port = os.getenv('FRONTEND_PORT', '5173')  # Default to Vite port
        return f'http://localhost:{local_port}'
    else:
        return 'https://kempian.ai'

COGNITO_CLIENT_ID = os.getenv('COGNITO_CLIENT_ID')
COGNITO_CLIENT_SECRET = os.getenv('CLIENT_SECRET')

def get_secret_hash(username):
    import hmac, hashlib, base64
    message = username + COGNITO_CLIENT_ID
    dig = hmac.new(
        str(COGNITO_CLIENT_SECRET).encode('utf-8'),
        msg=message.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(dig).decode()

# Helper to get Cognito public keys (cache for production)
COGNITO_KEYS_URL = f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}/.well-known/jwks.json'
_cognito_jwk_cache = None

def get_cognito_jwk():
    global _cognito_jwk_cache
    if _cognito_jwk_cache is None:
        resp = requests.get(COGNITO_KEYS_URL)
        _cognito_jwk_cache = resp.json()
    return _cognito_jwk_cache

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    role = data.get('role', 'job_seeker')  # default to job_seeker
    company_name = data.get('company_name')  # Company name for employers/recruiters
    visa_status = data.get('visa_status')  # Visa status for job seekers

    if not email or not password or not first_name or not last_name:
        return jsonify({'error': 'Email, password, first name, and last name required'}), 400
    
    # Validate company name for employers/recruiters
    if role in ['employer', 'recruiter'] and not company_name:
        return jsonify({'error': 'Company name is required for employers and recruiters'}), 400
    
    full_name = f"{first_name} {last_name}"
    try:
        # For employers/recruiters/admin, they get 'owner' or 'admin' role in tenant system
        if role == 'admin':
            tenant_role = 'admin'
        elif role in ['employer', 'recruiter']:
            tenant_role = 'owner'
        else:
            tenant_role = role
        cognito_signup(email, password, role=tenant_role, user_type=role, full_name=full_name, first_name=first_name, last_name=last_name)

        return jsonify({'message': 'Signup successful. Please check your email for confirmation code.'}), 201
    except Exception as e:
        logger.error(f"Error in /auth/signup: {str(e)}", exc_info=True)
        
        # Check if user exists and is unconfirmed
        error_message = str(e)
        if 'UsernameExistsException' in error_message:
            try:
                user_status = check_user_status(email)
                if user_status['exists'] and user_status['status'] == 'UNCONFIRMED':
                    return jsonify({
                        'error': 'An account with this email already exists but is not confirmed.',
                        'error_type': 'UNCONFIRMED_EXISTS',
                        'recovery_options': ['resend_confirmation', 'reset_account'],
                        'message': 'You can either resend the confirmation code or reset your account to sign up again.'
                    }), 400
            except:
                pass  # Fall back to generic error
        
        return jsonify({'error': error_message}), 400

@auth_bp.route('/signup-jobseeker', methods=['POST'])
def signup_jobseeker():
    """Enhanced signup endpoint for job seekers with resume upload and profile creation"""
    try:
        from app.models import Tenant
        # Get form data
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        phone = request.form.get('phone', '').strip()
        location = request.form.get('location', '').strip()
        visa_status = request.form.get('visa_status', '').strip()
        
        # Validate required fields
        if not all([email, password, first_name, last_name, phone, location]):
            return jsonify({'error': 'Email, password, first name, last name, phone, and location are required'}), 400
        
        # Check if file is present
        if 'resume' not in request.files:
            return jsonify({'error': 'Resume file is required'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No resume file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'pdf', 'doc', 'docx', 'txt'}  # Added txt for testing
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Only PDF, DOC, DOCX, TXT allowed'}), 400
        
        full_name = f"{first_name} {last_name}"
        
        # Create user in Cognito first
        try:
            cognito_signup(email, password, role='job_seeker', user_type='job_seeker', 
                          full_name=full_name, first_name=first_name, last_name=last_name)
        except Exception as e:
            error_msg = str(e)
            if "UsernameExistsException" in error_msg:
                return jsonify({'error': 'User with this email already exists in Cognito'}), 400
            logger.error(f"Cognito signup error: {error_msg}")
            return jsonify({'error': f'Failed to create account: {error_msg}'}), 400
        
        # Check if user already exists in database and clean up if they do (orphaned user)
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            logger.info(f"Found orphaned user {email} in database, cleaning up...")
            try:
                # Delete the orphaned user and their tenant
                if existing_user.tenant_id:
                    tenant = Tenant.query.get(existing_user.tenant_id)
                    if tenant:
                        db.session.delete(tenant)
                db.session.delete(existing_user)
                db.session.commit()
                logger.info(f"Cleaned up orphaned user {email}")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up orphaned user {email}: {cleanup_error}")
                db.session.rollback()
        
        # Create user in database immediately (like LinkedIn signup does)
        # Find Free Trial plan
        trial_plan = Plan.query.filter_by(name="Free Trial").first()
        if not trial_plan:
            return jsonify({'error': 'Free Trial plan not found. Please contact support.'}), 500
        
        # Create tenant
        tenant = Tenant(
            plan_id=trial_plan.id,
            stripe_customer_id="",
            stripe_subscription_id="",
            status="active"
        )
        db.session.add(tenant)
        db.session.commit()

        # Create user
        user = User(
            tenant_id=tenant.id,
            email=email,
            role='job_seeker',
            user_type='job_seeker'
        )
        db.session.add(user)
        db.session.commit()
        
        # Create trial
        try:
            from app.utils.trial_manager import create_user_trial
            create_user_trial(user.id)
        except Exception as e:
            logger.error(f"Error creating user trial: {str(e)}")
            # Continue without trial creation
        
        # Process resume file with enhanced parsing
        resume_data = {}
        try:
            # Save file temporarily for parsing
            import tempfile
            import os
            from app.services.resume_parser import parse_resume_data
            
            # Create temporary file with proper cleanup
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
            temp_file_path = temp_file.name
            temp_file.close()  # Close the file handle
            
            # Save the uploaded file to temp location
            file.save(temp_file_path)
            
            # Parse resume using enhanced parser
            parsed_data = parse_resume_data(temp_file_path)
            resume_data = parsed_data
            
            # Resume parsing completed successfully
            
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Could not delete temp file {temp_file_path}: {cleanup_error}")
                
        except Exception as e:
            logger.error(f"Resume parsing error: {str(e)}")
            # Continue without parsed data
        
        # Upload resume to S3
        import uuid
        from app.talent.routes import s3_client, S3_BUCKET
        
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        s3_key = f"career_resume/{unique_filename}"
        
        try:
            file.seek(0)  # Reset file pointer
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
            logger.error(f"S3 upload failed: {str(e)}")
            return jsonify({'error': 'Failed to upload resume to cloud storage'}), 500
        
        # Create candidate profile with parsed data
        from app.models import CandidateProfile, CandidateSkill, UserSocialLinks
        from datetime import datetime
        
        # Ensure we have a valid full_name
        profile_full_name = resume_data.get('full_name') or full_name
        if not profile_full_name or profile_full_name.strip() == '':
            profile_full_name = f"{first_name} {last_name}"
        
        logger.info(f"Final profile full_name: '{profile_full_name}' (from resume: '{resume_data.get('full_name')}', from form: '{full_name}')")
        
        profile = CandidateProfile(
            user_id=user.id,
            full_name=profile_full_name,
            phone=resume_data.get('phone', phone),
            location=resume_data.get('location', location),
            summary=resume_data.get('summary', ''),
            experience_years=resume_data.get('experience_years'),
            resume_s3_key=s3_key,
            resume_filename=file.filename,
            resume_upload_date=datetime.utcnow(),
            visa_status=visa_status if visa_status else None
        )
        db.session.add(profile)
        db.session.flush()  # Get the profile ID
        
        # Add parsed skills
        if resume_data.get('skills'):
            for skill_name in resume_data['skills']:
                skill = CandidateSkill(
                    profile_id=profile.id,
                    skill_name=skill_name
                )
                db.session.add(skill)
        
        # Add parsed projects
        if resume_data.get('projects'):
            from app.models import CandidateProject
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

                current_app.logger.info(f"[SIGNUP] Initial project entry count after separators: {len(project_entries)}")

                # If still only one, try splitting by inline project titles with colons
                if len(project_entries) <= 1 and len(projects_text) > 200 and projects_text.count(':') >= 2:
                    title_pattern = re.compile(
                        r'(?=(?:[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+'
                        r'(?:Application|App|Project|Clone|Website|System|Platform)\s*:))'
                    )
                    parts = [p.strip() for p in title_pattern.split(projects_text) if p.strip()]
                    current_app.logger.info(f"[SIGNUP] Inline title split produced {len(parts)} parts")
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
                            current_app.logger.info(f"[SIGNUP] Inline title recombination produced {len(recombined)} project entries")
                            project_entries = recombined
                
                # If still only one, try splitting by semicolons or single newlines (fallback)
                if len(project_entries) <= 1:
                    project_entries = re.split(r'[;\n]', projects_text)
                    project_entries = [entry.strip() for entry in project_entries if entry.strip() and len(entry.strip()) > 10]
                
                for i, project_entry in enumerate(project_entries[:10]):  # Limit to 10 projects
                    project_entry = project_entry.strip()
                    if project_entry and len(project_entry) > 10:
                        # Extract project name and description
                        name = None
                        description = project_entry
                        
                        # Try to find project name at the start
                        lines = project_entry.split('\n')
                        first_line = lines[0].strip() if lines else project_entry
                        
                        if ':' in first_line and len(first_line) < 100:
                            name_parts = first_line.split(':', 1)
                            if len(name_parts) == 2:
                                name = name_parts[0].strip()
                                name = re.sub(r'^(project|title|name)\s*:?\s*', '', name, flags=re.IGNORECASE)
                                description = '\n'.join([name_parts[1].strip()] + lines[1:]).strip()
                        elif len(first_line) < 80:
                            name = first_line
                            description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else project_entry
                        else:
                            name = project_entry[:50].strip()
                            description = project_entry
                        
                        if not name or len(name) < 3:
                            name = f"Project {i+1}"
                        
                        project = CandidateProject(
                            profile_id=profile.id,
                            name=name,
                            description=description
                        )
                        db.session.add(project)
        
        # Add parsed education
        if resume_data.get('education'):
            from app.models import CandidateEducation
            import re
            from datetime import datetime
            
            # Parse education from the extracted text
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
                
                for i, edu_entry in enumerate(education_entries[:10]):  # Limit to 10 education entries
                    if not edu_entry or len(edu_entry) < 10:
                        continue
                        
                    # Initialize variables
                    degree = edu_entry
                    institution = "Unknown Institution"
                    field_of_study = None
                    
                    # Look for degree patterns first
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
                    
                    # Look for institution patterns
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
                    
                    # Extract field of study if present
                    if 'in ' in degree.lower():
                        field_match = re.search(r'in\s+([^,\n]+)', degree, re.IGNORECASE)
                        if field_match:
                            field_of_study = field_match.group(1).strip()
                    elif '(' in degree and ')' in degree:
                        # Extract from parentheses
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
        
        # Add parsed certifications
        if resume_data.get('certifications'):
            from app.models import CandidateCertification
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
                
                for i, cert_entry in enumerate(cert_entries[:10]):  # Limit to 10 certifications
                    cert_entry = cert_entry.strip()
                    if cert_entry and len(cert_entry) > 5:
                        # Extract certification name and organization
                        name = cert_entry
                        organization = None
                        
                        # Try to extract organization from common patterns
                        if ' - ' in cert_entry:
                            parts = cert_entry.split(' - ', 1)
                            name = parts[0].strip()
                            organization = parts[1].strip() if len(parts) > 1 else None
                        elif '(' in cert_entry and ')' in cert_entry:
                            org_match = re.search(r'\(([^)]+)\)', cert_entry)
                            if org_match:
                                organization = org_match.group(1).strip()
                                name = re.sub(r'\s*\([^)]+\)', '', cert_entry).strip()
                        
                        # Look for common cert provider names
                        if not organization:
                            cert_providers = ['AWS', 'Amazon', 'Google', 'Microsoft', 'Oracle', 'Cisco', 'CompTIA', 
                                            'HackerRank', 'Coursera', 'Udemy', 'edX', 'LinkedIn', 'IBM', 'SAP']
                            for provider in cert_providers:
                                if provider.lower() in cert_entry.lower():
                                    provider_match = re.search(rf'({provider}[^,\n-]*)', cert_entry, re.IGNORECASE)
                                    if provider_match:
                                        org_text = provider_match.group(1).strip()
                                        if org_text.lower() not in name.lower()[:50]:
                                            organization = org_text
                                    break
                        
                        # Clean up name
                        name = re.sub(r'^(certification|certificate|certified)\s*:?\s*', '', name, flags=re.IGNORECASE)
                        name = name.strip()
                        
                        if len(name) > 200:
                            name = name[:200]
                        
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
        
        db.session.commit()
        
        # Get recommended jobs for the newly created profile
        recommended_jobs = []
        try:
            # Import the function from jobs routes
            from app.jobs.routes import get_recommended_jobs_for_user
            recommended_jobs = get_recommended_jobs_for_user(user.id, limit=10)
            logger.info(f"Found {len(recommended_jobs)} recommended jobs for new user {email}")
        except ImportError:
            # If import fails, try calling the logic directly
            try:
                from app.models import Job
                from datetime import datetime
                import json
                
                # Get all active public jobs
                query = Job.query.filter_by(is_public=True, status='active')
                query = query.filter(
                    db.or_(
                        Job.expires_at.is_(None),
                        Job.expires_at > datetime.utcnow()
                    )
                )
                all_jobs = query.limit(50).all()  # Limit to 50 for performance
                
                if all_jobs and profile:
                    # Calculate match scores
                    scored_jobs = []
                    user_skills = [skill.skill_name.lower() for skill in profile.skills]
                    user_experience_years = profile.experience_years or 0
                    user_location = (profile.location or '').lower()
                    
                    # Import the calculate function
                    from app.jobs.routes import calculate_job_match_score
                    
                    for job in all_jobs:
                        match_score = calculate_job_match_score(
                            job, 
                            user_skills, 
                            user_experience_years, 
                            user_location,
                            profile
                        )
                        
                        if match_score > 0:
                            job_dict = job.to_dict()
                            job_dict['matchScore'] = match_score
                            scored_jobs.append((match_score, job_dict))
                    
                    # Sort and get top 10
                    scored_jobs.sort(key=lambda x: x[0], reverse=True)
                    recommended_jobs = [job_dict for _, job_dict in scored_jobs[:10]]
                    logger.info(f"Found {len(recommended_jobs)} recommended jobs for new user {email}")
            except Exception as e:
                logger.error(f"Error getting recommended jobs after signup: {str(e)}")
                # Continue without recommended jobs - not critical
        except Exception as e:
            logger.error(f"Error getting recommended jobs after signup: {str(e)}")
            # Continue without recommended jobs - not critical
        
        # Count education and certifications for response
        education_count = 0
        certifications_count = 0
        if resume_data.get('education'):
            education_lines = re.split(r'[;\n]', resume_data['education'])
            education_count = len([line for line in education_lines if line.strip() and len(line.strip()) > 10])
        if resume_data.get('certifications'):
            cert_lines = re.split(r'[;\n]', resume_data['certifications'])
            certifications_count = len([line for line in cert_lines if line.strip() and len(line.strip()) > 10])
        
        return jsonify({
            'message': 'Job seeker signup successful. Please check your email for confirmation code.',
            'profile_created': True,
            'resume_parsed': bool(resume_data),
            'recommended_jobs': recommended_jobs,  # Include recommended jobs in response
            'parsed_data': {
                'skills_found': len(resume_data.get('skills', [])),
                'experience_years': resume_data.get('experience_years'),
                'education_found': bool(resume_data.get('education')),
                'education_count': education_count,
                'certifications_found': bool(resume_data.get('certifications')),
                'certifications_count': certifications_count,
                'projects_found': bool(resume_data.get('projects')),
                'projects_count': len(resume_data.get('projects', '').split(';')) if resume_data.get('projects') else 0
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Job seeker signup error: {str(e)}", exc_info=True)
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Internal server error during signup: {str(e)}'}), 500

@auth_bp.route('/confirm', methods=['POST'])
def confirm():
    data = request.get_json()
    email = data.get('email')
    code = data.get('code')
    company_name = data.get('company_name')  # Get company name from confirm request
    visa_status = data.get('visa_status')  # Get visa status from confirm request

    if not email or not code:
        return jsonify({'error': 'Email and code required'}), 400
    try:
        cognito_confirm_signup(email, code)

        # Gather Cognito attributes for notification/context
        attrs = {}
        try:
            user_info = cognito_client.admin_get_user(
                UserPoolId=COGNITO_USER_POOL_ID,
                Username=email
            )
            attrs = {attr['Name']: attr['Value'] for attr in user_info.get('UserAttributes', [])}
        except Exception as attr_err:
            logger.warning(f"Unable to fetch Cognito attributes for {email}: {attr_err}")

        original_user_type = attrs.get("custom:user_type", attrs.get("custom:role", "owner"))
        first_name_attr = (
            attrs.get("given_name")
            or attrs.get("name")
            or attrs.get("custom:first_name")
            or attrs.get("preferred_username")
            or email.split('@')[0]
        )

        # --- Allocate Free Trial Plan if user has no tenant ---
        # Check if user already exists in DB
        user = User.query.filter_by(email=email).first()
        tenant_id = None
        created_user = None

        if not user:
            # Find Free Trial plan
            trial_plan = Plan.query.filter_by(name="Free Trial").first()
            if not trial_plan:
                return jsonify({'error': 'Free Trial plan not found. Please contact support.'}), 500
            # Create tenant
            tenant = Tenant(
                plan_id=trial_plan.id,
                stripe_customer_id="",
                stripe_subscription_id="",
                status="active"
            )
            db.session.add(tenant)
            db.session.commit()

            # Determine tenant role based on user type
            if original_user_type == 'admin':
                tenant_role = 'admin'
            elif original_user_type in ['employer', 'recruiter']:
                tenant_role = 'owner'
            else:
                tenant_role = original_user_type
            
            # Always store the original user type for display, even if system role is owner/admin
            db_user = User(
                tenant_id=tenant.id,
                email=email,
                role=tenant_role,
                user_type=original_user_type,
                company_name=company_name
            )

            db.session.add(db_user)
            db.session.commit()
            created_user = db_user
            
            # Create trial for new user
            trial = create_user_trial(db_user.id)
            if not trial:
                pass  # Trial creation failed, but continue
            
            # Save visa status if provided for job seekers
            if visa_status and original_user_type in ['job_seeker', 'employee']:
                candidate_profile = CandidateProfile.query.filter_by(user_id=db_user.id).first()
                if not candidate_profile:
                    # Create a basic candidate profile if it doesn't exist
                    candidate_profile = CandidateProfile(
                        user_id=db_user.id,
                        full_name=db_user.email  # Use email as fallback since User model doesn't have first_name/last_name
                    )
                    db.session.add(candidate_profile)
                
                candidate_profile.visa_status = visa_status
                db.session.commit()

            tenant_id = tenant.id
        else:
            # Update company name if user exists and it's not set (for employers/recruiters and admins/owners)
            if company_name and not user.company_name and (user.user_type in ['employer', 'recruiter', 'admin'] or user.role in ['admin', 'owner']):
                user.company_name = company_name
                db.session.commit()

            tenant_id = user.tenant_id
            if not original_user_type:
                original_user_type = user.user_type or user.role

        final_user = created_user or user

        # Always update Cognito user with tenant_id
        if tenant_id is not None:
            try:
                cognito_admin_update_user_attributes(email, {"custom:tenant_id": str(tenant_id)})
            except Exception as e:
                logger.error(f"Failed to update Cognito tenant_id for {email}: {e}")

        # Post-confirmation notifications (best-effort)
        try:
            from app.emails.smtp import send_welcome_email_smtp
            from app.utils import notify_admins_new_user

            notification_role = None
            if final_user:
                notification_role = final_user.user_type or final_user.role
            if not notification_role:
                notification_role = original_user_type

            display_name = first_name_attr
            if not display_name and final_user and final_user.candidate_profile and final_user.candidate_profile.full_name:
                display_name = final_user.candidate_profile.full_name.split()[0]
            if not display_name:
                display_name = email.split('@')[0]

            send_welcome_email_smtp(email, display_name, notification_role)
            notify_admins_new_user(email, notification_role, display_name)
        except Exception as _email_err:
            logger.warning(f"Post-confirmation notifications skipped: {_email_err}")

        # --- End allocation ---
        return jsonify({'message': 'Confirmation successful.'}), 200
    except Exception as e:
        logger.error(f"Error in /auth/confirm: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@auth_bp.route('/confirm-temp-password', methods=['POST'])
def confirm_temp_password():
    data = request.get_json()
    email = data.get('email')
    temp_password = data.get('temp_password')
    new_password = data.get('new_password')
    
    if not email or not temp_password or not new_password:
        return jsonify({'error': 'Email, temporary password, and new password required'}), 400
    
    try:
        # First, authenticate with temporary password
        cognito_client = boto3.client('cognito-idp', region_name=COGNITO_REGION)
        
        # Build auth parameters - only include SECRET_HASH if client secret is configured
        auth_params = {
            'USERNAME': email,
            'PASSWORD': temp_password
        }
        
        # Only add SECRET_HASH if client secret is configured
        if COGNITO_CLIENT_SECRET:
            auth_params['SECRET_HASH'] = get_secret_hash(email)
        
        # Initiate auth with temporary password
        auth_response = cognito_client.initiate_auth(
            ClientId=COGNITO_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters=auth_params
        )
        
        # Check if we got a challenge response (expected for temporary passwords)
        if 'ChallengeName' not in auth_response or 'Session' not in auth_response:
            # If no challenge, the password might already be permanent or there's an issue
            if 'AuthenticationResult' in auth_response:
                # User already has a permanent password, no need to change it
                return jsonify({'error': 'Password is already set. Please use the login endpoint.'}), 400
            else:
                # Unexpected response structure
                logger.error(f"Unexpected auth response structure: {auth_response}")
                return jsonify({'error': 'Unexpected authentication response. Please try again.'}), 400
        
        # Verify it's the expected challenge
        if auth_response.get('ChallengeName') != 'NEW_PASSWORD_REQUIRED':
            logger.warning(f"Unexpected challenge name: {auth_response.get('ChallengeName')}")
            return jsonify({'error': 'Unexpected authentication challenge. Please contact support.'}), 400
        
        # Get the session token from the challenge response
        session_token = auth_response['Session']
        
        # If we get here, the temp password is correct
        # Now change the password
        # First, check if user already has a name attribute set
        try:
            user_info = cognito_client.admin_get_user(
                UserPoolId=COGNITO_USER_POOL_ID,
                Username=email
            )
            attrs = {attr['Name']: attr['Value'] for attr in user_info.get('UserAttributes', [])}
            user_name = attrs.get('name') or attrs.get('given_name') or email.split('@')[0]
        except Exception:
            # If we can't get user info, use email prefix as name
            user_name = email.split('@')[0]
        
        # Build challenge responses with required attributes
        # Note: userAttributes must be provided as userAttributes.attributeName format
        challenge_responses = {
            'USERNAME': email,
            'NEW_PASSWORD': new_password,
        }
        
        # Add user attributes - Cognito requires name attribute
        # Format: userAttributes.attributeName = value
        challenge_responses['userAttributes.name'] = user_name
        
        # Only add SECRET_HASH if client secret is configured
        if COGNITO_CLIENT_SECRET:
            challenge_responses['SECRET_HASH'] = get_secret_hash(email)
        
        cognito_client.respond_to_auth_challenge(
            ClientId=COGNITO_CLIENT_ID,
            ChallengeName='NEW_PASSWORD_REQUIRED',
            Session=session_token,
            ChallengeResponses=challenge_responses
        )
        
        return jsonify({'message': 'Password changed successfully. You can now log in.'}), 200
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        error_message = str(e)
        
        # Check if the error is about expired temporary password
        if 'NotAuthorizedException' in error_code or 'expired' in error_message.lower() or 'must be reset by an administrator' in error_message.lower():
            logger.warning(f"Temporary password expired for {email}: {error_message}")
            return jsonify({
                'error': 'Temporary password has expired and must be reset by an administrator.',
                'error_code': 'EXPIRED_TEMP_PASSWORD',
                'message': 'Your invite link has expired. Please request a new invite link from your administrator.'
            }), 400
        
        logger.error(f"Error in /auth/confirm-temp-password: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in /auth/confirm-temp-password: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@auth_bp.route('/regenerate-invite', methods=['POST'])
def regenerate_invite():
    """
    Regenerate expired temporary password and return new invite link
    This endpoint allows users with expired invite links to get a new one
    """
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    
    try:
        # Find user in database to get tenant_id and role
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found. Please contact your administrator.'}), 404
        
        tenant_id = user.tenant_id
        role = user.role or 'subuser'
        
        # Generate a new temporary password
        new_temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12)) + '1!A'
        
        # Reset the user's password in Cognito
        try:
            cognito_client.admin_set_user_password(
                UserPoolId=COGNITO_USER_POOL_ID,
                Username=email,
                Password=new_temp_password,
                Permanent=False  # Keep as temporary so user must change it
            )
            logger.info(f"Regenerated temporary password for {email}")
        except ClientError as cognito_error:
            error_code = cognito_error.response.get('Error', {}).get('Code', '')
            if error_code == 'UserNotFoundException':
                # User doesn't exist in Cognito, try to create them
                try:
                    from .cognito import cognito_admin_create_user
                    new_temp_password, _ = cognito_admin_create_user(email, tenant_id, role=role)
                    logger.info(f"Created new Cognito user for {email}")
                except Exception as create_error:
                    logger.error(f"Failed to create Cognito user for {email}: {str(create_error)}")
                    return jsonify({'error': 'Failed to create user account. Please contact your administrator.'}), 500
            else:
                logger.error(f"Failed to reset password for {email}: {str(cognito_error)}")
                return jsonify({'error': 'Failed to regenerate invite. Please contact your administrator.'}), 500
        
        # Generate new invite link
        frontend_url = get_frontend_url()
        encoded_email = quote(email, safe='')
        encoded_code = quote(new_temp_password, safe='')
        invite_link = (
            f"{frontend_url}/invite?"
            f"email={encoded_email}&username={encoded_email}&code={encoded_code}"
        )
        
        # Optionally send email (default to False to avoid spam)
        send_email = data.get('send_email', False)
        email_sent = False
        if send_email:
            try:
                from app.emails.ses import send_invite_email
                email_sent = send_invite_email(email, invite_link)
            except Exception as email_error:
                logger.warning(f"Failed to send invite email to {email}: {str(email_error)}")
        
        return jsonify({
            'message': 'New invite link generated successfully',
            'invite_link': invite_link,
            'email_sent': email_sent
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /auth/regenerate-invite: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred. Please contact your administrator.'}), 500

@auth_bp.route('/login', methods=['POST'])
@log_admin_login_activity
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    requested_role = data.get('role')  # Add role parameter for role change requests
    
    logger.info(f"🔐 Login attempt started for email: {email} with requested role: {requested_role}")
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    try:
        # First, attempt Cognito login to validate credentials
        login_result = cognito_login(email, password)
        
        # Check if this is a challenge response (e.g., NEW_PASSWORD_REQUIRED for temporary passwords)
        if isinstance(login_result, dict) and 'ChallengeName' in login_result:
            challenge_name = login_result.get('ChallengeName')
            
            # If it's a NEW_PASSWORD_REQUIRED challenge, handle it to allow temp passwords to work
            if challenge_name == 'NEW_PASSWORD_REQUIRED':
                logger.info(f"⚠️ Temporary password detected for {email}, handling challenge to allow access")
                
                try:
                    from .cognito import cognito_client, COGNITO_USER_POOL_ID, COGNITO_CLIENT_ID, get_secret_hash, get_user_by_email
                    
                    # Get user by email to get username
                    user_info = get_user_by_email(email)
                    username = user_info['Username']
                    attrs = {attr['Name']: attr['Value'] for attr in user_info.get('UserAttributes', [])}
                    user_name = attrs.get('name') or attrs.get('given_name') or email.split('@')[0]
                    
                    # Respond to the challenge by setting the same password as permanent
                    # This allows the temporary password to work for all tasks while still allowing users to change it later
                    challenge_responses = {
                        'USERNAME': username,
                        'NEW_PASSWORD': password,  # Use same password to make it work
                        'userAttributes.name': user_name
                    }
                    
                    if COGNITO_CLIENT_SECRET:
                        challenge_responses['SECRET_HASH'] = get_secret_hash(username)
                    
                    # Respond to challenge
                    challenge_response = cognito_client.respond_to_auth_challenge(
                        ClientId=COGNITO_CLIENT_ID,
                        ChallengeName='NEW_PASSWORD_REQUIRED',
                        Session=login_result['Session'],
                        ChallengeResponses=challenge_responses
                    )
                    
                    # Now set the password as permanent so it works for all tasks
                    # Users can still change it later if they want
                    cognito_client.admin_set_user_password(
                        UserPoolId=COGNITO_USER_POOL_ID,
                        Username=username,
                        Password=password,
                        Permanent=True
                    )
                    logger.info(f"✅ Temporary password set as permanent for {email} to allow full access")
                    
                    # Get tokens from challenge response
                    if 'AuthenticationResult' in challenge_response:
                        tokens = challenge_response['AuthenticationResult']
                    else:
                        # Try admin auth as fallback
                        auth_params = {
                            'USERNAME': username,
                            'PASSWORD': password
                        }
                        if COGNITO_CLIENT_SECRET:
                            auth_params['SECRET_HASH'] = get_secret_hash(username)
                        
                        admin_auth_response = cognito_client.admin_initiate_auth(
                            UserPoolId=COGNITO_USER_POOL_ID,
                            ClientId=COGNITO_CLIENT_ID,
                            AuthFlow='ADMIN_NO_SRP_AUTH',
                            AuthParameters=auth_params
                        )
                        tokens = admin_auth_response.get('AuthenticationResult')
                        
                except Exception as challenge_error:
                    logger.error(f"Error handling temporary password challenge: {challenge_error}")
                    return jsonify({
                        'error': 'Temporary password authentication failed. Please use /auth/confirm-temp-password to set a new password.',
                        'error_type': 'TEMP_PASSWORD_REQUIRED',
                        'message': 'Your account uses a temporary password. Please change it to continue.'
                    }), 401
            else:
                # Other challenges - return error
                return jsonify({
                    'error': f'Authentication challenge required: {challenge_name}',
                    'error_type': 'CHALLENGE_REQUIRED',
                    'challenge_name': challenge_name
                }), 401
        else:
            # Normal login successful
            tokens = login_result
            logger.info(f"✅ Cognito login successful for {email}")
        
        # Fetch user attributes from Cognito using email resolution
        from .cognito import get_user_by_email, cognito_admin_update_user_attributes
        logger.info(f"🔍 Fetching user info for email: {email}")
        
        try:
            user_info = get_user_by_email(email)
            attrs = {attr['Name']: attr['Value'] for attr in user_info['UserAttributes']}
            logger.info(f"🔍 User info from Cognito: {attrs}")
        except Exception as cognito_error:
            logger.error(f"❌ Failed to fetch user info from Cognito for {email}: {cognito_error}")
            return jsonify({
                'error': 'User account not found in our system. Please contact support.',
                'error_type': 'USER_NOT_FOUND',
                'message': 'Your credentials are valid but your account could not be located.'
            }), 404
        
        stored_role = attrs.get("custom:role", "")
        
        # STRONG ADMIN ROLE PROTECTION - Never allow admin roles to be changed
        if stored_role == 'admin':
            logger.info(f"Admin login detected: {email} - Role protected from changes")
            # Force admin role and ignore any requested role changes
            stored_role = 'admin'
            requested_role = None  # Prevent any role change attempts for admin
        elif stored_role == 'owner':
            logger.info(f"Owner login detected: {email} - Role protected from changes")
            # Force owner role and ignore any requested role changes
            stored_role = 'owner'
            requested_role = None  # Prevent any role change attempts for owner
        
        # Role validation and change logic - ONLY if explicitly requested AND safe
        if requested_role and requested_role != stored_role and stored_role not in ['admin', 'owner']:
            logger.info(f"Explicit role change requested: {email} from {stored_role} to {requested_role}")
            
            # Define allowed role changes (hierarchical permissions)
            # Note: 'owner' and 'admin' are system roles that can access any user role
            allowed_role_changes = {
                'employer': ['job_seeker', 'employee', 'recruiter'],
                'recruiter': ['job_seeker', 'employee'],
                'employee': ['job_seeker'],
                'job_seeker': []  # Job seekers can't change to other roles
            }
            
            # Check if role change is allowed
            role_change_allowed = False
            if stored_role in allowed_role_changes and requested_role in allowed_role_changes[stored_role]:
                role_change_allowed = True
            
            if role_change_allowed:
                # Update Cognito with new role using email resolution
                try:
                    cognito_admin_update_user_attributes(email, {
                        "custom:role": requested_role,
                        "custom:user_type": requested_role
                    })
                    stored_role = requested_role
                    logger.info(f"Role change successful: {email} changed to {requested_role}")
                except Exception as e:
                    logger.error(f"Failed to update Cognito role for {email}: {e}")
                    return jsonify({'error': 'Failed to update role. Please try again.'}), 500
            else:
                logger.warning(f"Role change denied: {email} from {stored_role} to {requested_role}")
                return jsonify({'error': f'Role change from {stored_role} to {requested_role} not allowed'}), 403
        
        # --- Ensure Starter Plan and Tenant after login with proper locking ---
        try:
            with db.session.begin():
                # Use SELECT FOR UPDATE to prevent race conditions
                db_user = User.query.filter_by(email=email).with_for_update().first()
                tenant_id = None
                
                if not db_user:
                    logger.info(f"[DB] User {email} not found in database, creating new user")
                    
                    # Get the original user type from Cognito FIRST
                    user_info = get_user_by_email(email)
                    attrs = {attr['Name']: attr['Value'] for attr in user_info['UserAttributes']}
                    original_role = attrs.get("custom:role", "owner")
                    original_user_type = attrs.get("custom:user_type", original_role)
                    cognito_tenant_id = attrs.get("custom:tenant_id")
                    
                    logger.info(f"[DB] Creating user {email} with role from Cognito: {original_role}, user_type: {original_user_type}, tenant_id: {cognito_tenant_id}")
                    
                    # If user has a tenant_id in Cognito, use that tenant (for subusers)
                    if cognito_tenant_id:
                        try:
                            tenant_id = int(cognito_tenant_id)
                            tenant = Tenant.query.get(tenant_id)
                            if not tenant:
                                logger.warning(f"[DB] Tenant {tenant_id} from Cognito not found, creating new tenant")
                                # Fall through to create new tenant
                                tenant = None
                            else:
                                logger.info(f"[DB] Using existing tenant {tenant_id} from Cognito")
                        except (ValueError, TypeError):
                            logger.warning(f"[DB] Invalid tenant_id in Cognito: {cognito_tenant_id}")
                            tenant = None
                    else:
                        tenant = None
                    
                    # If no tenant found, create a new one (for new signups)
                    if not tenant:
                        # Find Free Trial plan
                        free_trial_plan = Plan.query.filter_by(name="Free Trial").first()
                        if not free_trial_plan:
                            return jsonify({'error': 'Free Trial plan not found. Please contact support.'}), 500
                        
                        # Create tenant
                        tenant = Tenant(
                            plan_id=free_trial_plan.id,
                            stripe_customer_id="",
                            stripe_subscription_id="",
                            status="active"
                        )
                        db.session.add(tenant)
                        db.session.flush()  # Get tenant ID without committing
                        tenant_id = tenant.id
                        logger.info(f"[DB] Created new tenant {tenant_id} for user {email}")
                    else:
                        tenant_id = tenant.id
                    
                    # PRESERVE the role from Cognito - don't override subuser with owner
                    # Store the original user type for display purposes
                    user_type = original_user_type if original_user_type in ['job_seeker', 'employee', 'recruiter', 'employer', 'subuser'] else None
                    
                    # Use the role from Cognito - preserve subuser, admin, etc.
                    db_role = original_role  # Preserve the role from Cognito
                    
                    db_user = User(tenant_id=tenant_id, email=email, role=db_role, user_type=user_type or db_role)
                    db.session.add(db_user)
                    db.session.flush()  # Get user ID without committing
                    
                    logger.info(f"[DB] Created new user {email} with tenant_id {tenant_id}, role: {db_role}, user_type: {user_type or db_role}")
                else:
                    tenant_id = db_user.tenant_id
                    logger.info(f"[DB] Found existing user {email} with tenant_id {tenant_id}")
                
                # Commit the transaction
                db.session.commit()
                
        except Exception as db_error:
            logger.error(f"[DB] Database error during user creation/lookup: {db_error}")
            db.session.rollback()
            return jsonify({'error': 'Database error occurred. Please try again.'}), 500
        # Always update Cognito user with tenant_id using email resolution
        try:
            cognito_admin_update_user_attributes(email, {"custom:tenant_id": str(tenant_id)})
        except Exception as e:
            logger.error(f"Failed to update Cognito tenant_id for {email}: {e}")
        # --- End ensure Starter plan ---
        
        # Get candidate profile data for job seekers
        candidate_profile = None
        if db_user and (stored_role == 'job_seeker' or attrs.get("custom:user_type") == 'job_seeker'):
            from app.models import CandidateProfile
            candidate_profile = CandidateProfile.query.filter_by(user_id=db_user.id).first()
        
        # Get projects and certifications for job seekers
        projects_data = []
        certifications_data = []
        if candidate_profile:
            from app.models import CandidateProject, CandidateCertification
            projects = CandidateProject.query.filter_by(profile_id=candidate_profile.id).all()
            projects_data = [project.to_dict() for project in projects]
            
            certifications = CandidateCertification.query.filter_by(profile_id=candidate_profile.id).all()
            certifications_data = [cert.to_dict() for cert in certifications]
        
        user = {
            "id": attrs.get("sub"),
            "email": attrs.get("email"),
            "firstName": attrs.get("given_name", ""),
            "lastName": attrs.get("family_name", ""),
            "role": stored_role,
            "userType": attrs.get("custom:user_type", stored_role),
            "companyName": db_user.company_name if db_user else None,
            "phone": candidate_profile.phone if candidate_profile else None,
            "location": candidate_profile.location if candidate_profile else None,
            "bio": candidate_profile.summary if candidate_profile else None,
            "visaStatus": candidate_profile.visa_status if candidate_profile else None,
            "projects": projects_data,
            "certifications": certifications_data
        }
        
        logger.info(f"✅ Login successful! User authenticated: email='{user['email']}', role='{user['role']}', userType='{user['userType']}', id='{user['id']}', requestedRole='{requested_role}', tenantId='{db_user.tenant_id if db_user else None}'")

        
        return jsonify({
            "access_token": tokens.get("AccessToken"),
            "id_token": tokens.get("IdToken"),
            "refresh_token": tokens.get("RefreshToken"),
            "user": user
        }), 200
    except Exception as e:
        logger.error(f"Error in /auth/login: {str(e)}", exc_info=True)
        
        error_message = str(e)
        
        # Handle specific Cognito errors with user-friendly messages
        if 'UserNotConfirmedException' in error_message:
            try:
                user_status = check_user_status(email)
                if user_status['exists'] and user_status['status'] == 'UNCONFIRMED':
                    return jsonify({
                        'error': 'Your account is not confirmed. Please check your email for a confirmation code.',
                        'error_type': 'UNCONFIRMED',
                        'recovery_options': ['resend_confirmation', 'reset_account'],
                        'message': 'You can either resend the confirmation code or reset your account to sign up again.'
                    }), 401
            except:
                pass  # Fall back to generic error
                
        elif 'NotAuthorizedException' in error_message:
            return jsonify({
                'error': 'Invalid email or password. Please check your credentials and try again.',
                'error_type': 'INVALID_CREDENTIALS',
                'message': 'The email or password you entered is incorrect.'
            }), 401
            
        elif 'UserNotFoundException' in error_message:
            return jsonify({
                'error': 'No account found with this email address.',
                'error_type': 'USER_NOT_FOUND',
                'message': 'Please check your email address or create a new account.'
            }), 404
            
        elif 'TooManyRequestsException' in error_message:
            return jsonify({
                'error': 'Too many login attempts. Please try again later.',
                'error_type': 'RATE_LIMITED',
                'message': 'You have exceeded the maximum number of login attempts. Please wait before trying again.'
            }), 429
            
        elif 'User with email' in error_message and 'not found' in error_message:
            return jsonify({
                'error': 'User account not found in our system.',
                'error_type': 'USER_NOT_FOUND',
                'message': 'Your credentials are valid but your account could not be located. Please contact support.'
            }), 404
        
        # Generic error fallback
        return jsonify({
            'error': 'Login failed. Please try again.',
            'error_type': 'LOGIN_ERROR',
            'message': 'An unexpected error occurred during login. Please try again.'
        }), 401

@auth_bp.route('/cognito-social-login', methods=['POST'])
def cognito_social_login():
    try:
        data = request.get_json()
        # Handle both frontend format (idToken, accessToken) and backend format (id_token, access_token)
        id_token = data.get('id_token') or data.get('idToken')
        access_token = data.get('access_token') or data.get('accessToken')
        state = data.get('state')  # Get state parameter from frontend
        role_fallback = data.get('role_fallback') or data.get('role')  # Get role fallback from frontend
        
        if not id_token:
            return jsonify({'error': 'ID token is required'}), 400
        if not access_token:
            return jsonify({'error': 'Access token is required'}), 400
            
        logger.info("🔍 Starting Cognito social login verification...")
        
        # Extract role from state if provided
        role_from_state = None
        if state:
            try:
                import base64
                import json
                state_data = json.loads(base64.b64decode(state).decode('utf-8'))
                role_from_state = state_data.get('role')
                logger.info(f"🔍 State parameter decoded: {state_data}, role_from_state: {role_from_state}")
            except Exception as e:
                logger.warning(f"Could not decode state parameter: {e}")
        
        # Debug logging for role extraction
        logger.info(f"🔍 Role extraction debug:")
        logger.info(f"   state: {state}")
        logger.info(f"   role_fallback: {role_fallback}")
        logger.info(f"   role_from_state: {role_from_state}")
        
        # 1. JWT verification logic
        # logger.info("🔍 Decoding JWT header to get KID...")
        header = jwt.get_unverified_header(id_token)
        kid = header.get('kid')
        # logger.info(f"🔑 JWT KID: {kid}")
        
        if not kid:
            logger.error("No KID found in JWT header")
            return jsonify({'error': 'Invalid token format'}), 401
        
        # Get JWKS
        # logger.info("🔍 Fetching JWKS from Cognito...")
        jwks = get_cognito_jwk()
        
        # Find the matching key
        key = None
        for jwk in jwks.get('keys', []):
            if jwk['kid'] == kid:
                key = jwk
                break
        
        if not key:
            logger.error(f"No matching key found for KID: {kid}")
            return jsonify({'error': 'No matching key found for token'}), 401
        
        # logger.info("✅ Found matching JWK key")
        
        # Verify the token
        # logger.info("🔍 Verifying JWT token...")
        # logger.info(f"🔑 Backend COGNITO_CLIENT_ID: {COGNITO_CLIENT_ID}")
        # logger.info(f"🔑 JWT token audience (from token): {jwt.get_unverified_claims(id_token).get('aud')}")
        
        claims = jwt.decode(
            id_token,
            key,
            algorithms=['RS256'],
            audience=COGNITO_CLIENT_ID,
            issuer=f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}',
            access_token=access_token  # Add access token for at_hash validation
        )
        
        # logger.info(f"✅ JWT verification successful")
        # logger.info(f"📋 Claims: {claims}")
        
        email = claims.get('email')
        sub = claims.get('sub')
        first_name = claims.get('given_name', '')
        last_name = claims.get('family_name', '')
        
        # 2. User creation logic (reuse from /login)
        # logger.info("🔍 Checking if user exists in database...")
        db_user = User.query.filter_by(email=email).first()
        tenant_id = None
        
        # Check if user already exists - if so, ALWAYS use their existing role to prevent role overwrites
        if db_user:
            # User exists - use their existing role and user_type, ignore state/fallback to prevent role changes
            role = db_user.user_type or db_user.role
            user_type = db_user.user_type or db_user.role
            tenant_id = db_user.tenant_id
            
            # Log warning if user tried to login with different role
            attempted_role = role_from_state or role_fallback or claims.get('custom:role', 'job_seeker')
            if attempted_role and attempted_role != role and attempted_role != user_type:
                logger.warning(f"⚠️ User {email} attempted to login with role '{attempted_role}' but existing role '{role}'/'{user_type}' is preserved. Role changes are not allowed.")
            
            logger.info(f"✅ User already exists in database with ID: {db_user.id}, preserving existing role: {role}, user_type: {user_type}")
        else:
            # New user - use role from state, then fallback, then claims, then default
            role = role_from_state or role_fallback or claims.get('custom:role', 'job_seeker')
            user_type = claims.get('custom:user_type', role)
            
            # logger.info("👤 User not found in database, creating new user...")
            free_trial_plan = Plan.query.filter_by(name="Free Trial").first()
            if not free_trial_plan:
                logger.error("Free Trial plan not found")
                return jsonify({'error': 'Free Trial plan not found. Please contact support.'}), 500
            
            # logger.info("🏢 Creating new tenant...")
            tenant = Tenant(
                plan_id=free_trial_plan.id,
                stripe_customer_id="",
                stripe_subscription_id="",
                status="active"
            )
            db.session.add(tenant)
            db.session.commit()
            # logger.info(f"✅ Tenant created with ID: {tenant.id}")
            
            # logger.info("👤 Creating new user...")
            db_user = User(tenant_id=tenant.id, email=email, role=role, user_type=user_type)
            db.session.add(db_user)
            db.session.commit()
            tenant_id = tenant.id
            # logger.info(f"✅ User created successfully with ID: {db_user.id}")
            
            # Update Cognito with the role for new users
            try:
                cognito_admin_update_user_attributes(email, {
                    "custom:role": role,
                    "custom:user_type": user_type
                })
                logger.info(f"Updated Cognito user attributes for new user {email}: role={role}, user_type={user_type}")
            except Exception as e:
                logger.error(f"Failed to update Cognito attributes for new user {email}: {e}")
            
            # Send admin notification for new Google OAuth signup
            try:
                from app.utils import notify_admins_new_user
                from app.emails.smtp import send_welcome_email_smtp
                
                display_name = first_name or email.split('@')[0]
                notification_role = user_type or role
                
                # Send welcome email to user
                send_welcome_email_smtp(email, display_name, notification_role)
                
                # Notify admin about new signup
                notify_admins_new_user(email, notification_role, display_name)
                logger.info(f"✅ Admin notification sent for new Google OAuth user: {email}")
            except Exception as email_err:
                logger.warning(f"Failed to send notifications for new Google OAuth user {email}: {email_err}")
        
        # Debug logging for final role resolution
        logger.info(f"🔍 Final role resolution:")
        logger.info(f"   role_from_state: {role_from_state}")
        logger.info(f"   role_fallback: {role_fallback}")
        logger.info(f"   claims.get('custom:role'): {claims.get('custom:role')}")
        logger.info(f"   db_user exists: {db_user is not None}")
        if db_user:
            logger.info(f"   db_user.role: {db_user.role}")
            logger.info(f"   db_user.user_type: {db_user.user_type}")
        logger.info(f"   Final role: {role}")
        logger.info(f"   Final user_type: {user_type}")
        
        # 3. Update Cognito user attributes ONLY if they don't match existing user's role
        # For existing users, only sync Cognito if it's missing or different from database
        cognito_role = claims.get('custom:role')
        cognito_user_type = claims.get('custom:user_type')
        
        # Only update Cognito if:
        # 1. Cognito role/user_type is missing AND we have a role to set
        # 2. Cognito role/user_type doesn't match database role (sync database to Cognito, not the other way)
        if db_user:
            # For existing users, sync Cognito to match database (not the other way)
            if not cognito_role or not cognito_user_type or cognito_role != role or cognito_user_type != user_type:
                try:
                    logger.info(f"🔄 Syncing Cognito user attributes to match database: role={role}, user_type={user_type}")
                    cognito_admin_update_user_attributes(email, {
                        "custom:role": role,
                        "custom:user_type": user_type
                    })
                except Exception as e:
                    logger.error(f"Failed to sync Cognito role for {email}: {e}")
        else:
            # For new users, Cognito should already be updated above, but ensure it's set
            if not cognito_role or cognito_role != role:
                try:
                    cognito_admin_update_user_attributes(email, {
                        "custom:role": role,
                        "custom:user_type": user_type
                    })
                except Exception as e:
                    logger.error(f"Failed to update Cognito role for {email}: {e}")
        
        # 4. DO NOT update database user record for existing users - roles are immutable once set
        # This prevents role overwrites when users try to login with different roles
            
        # 4. Get candidate profile data for job seekers
        candidate_profile = None
        if db_user and (role == 'job_seeker' or user_type == 'job_seeker'):
            from app.models import CandidateProfile
            candidate_profile = CandidateProfile.query.filter_by(user_id=db_user.id).first()
        
        # 5. Get projects and certifications for job seekers
        projects_data = []
        certifications_data = []
        if candidate_profile:
            from app.models import CandidateProject, CandidateCertification
            projects = CandidateProject.query.filter_by(profile_id=candidate_profile.id).all()
            projects_data = [project.to_dict() for project in projects]
            
            certifications = CandidateCertification.query.filter_by(profile_id=candidate_profile.id).all()
            certifications_data = [cert.to_dict() for cert in certifications]
        
        # 6. Return user info
        user = {
            "id": sub,
            "email": email,
            "firstName": first_name,
            "lastName": last_name,
            "role": role,
            "userType": user_type,
            "phone": candidate_profile.phone if candidate_profile else None,
            "location": candidate_profile.location if candidate_profile else None,
            "bio": candidate_profile.summary if candidate_profile else None,
            "visaStatus": candidate_profile.visa_status if candidate_profile else None,
            "projects": projects_data,
            "certifications": certifications_data
        }
        
        logger.info(f"✅ Social login successful! User authenticated: email='{user['email']}', role='{user['role']}', userType='{user['userType']}', id='{user['id']}', tenantId='{db_user.tenant_id if db_user else None}'")
        
        return jsonify({
            "id_token": id_token,
            "token": id_token,  # Add token field for frontend compatibility
            "user": user
        }), 200
        
    except jwt.JWTError as e:
        logger.error(f"❌ JWT verification error in /auth/cognito-social-login: {str(e)}", exc_info=True)
        return jsonify({'error': f'Invalid token: {str(e)}'}), 401
    except Exception as e:
        logger.error(f"❌ Error in /auth/cognito-social-login: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 401

@auth_bp.route('/social-login', methods=['POST'])
def social_login():
    """Frontend-compatible social login endpoint that redirects to cognito-social-login"""
    return cognito_social_login()

@auth_bp.route('/resend-confirmation', methods=['POST'])
def resend_confirmation():
    data = request.get_json()
    email = data.get('email')
    if not email:
        return jsonify({'error': 'Email required'}), 400
    try:
        result = resend_confirmation_code(email)
        if result['success']:
            return jsonify({'message': result['message']}), 200
        else:
            return jsonify({'error': result['message']}), 400
    except Exception as e:
        logger.error(f"Error in /auth/resend-confirmation: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400 

@auth_bp.route('/check-user-status', methods=['POST'])
def check_user_status_endpoint():
    """Check the status of a user account"""
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email required'}), 400
    
    try:
        result = get_recovery_options(email)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /auth/check-user-status: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/reset-unconfirmed-account', methods=['POST'])
def reset_unconfirmed_account():
    """Reset an unconfirmed account to allow re-signup"""
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email required'}), 400
    
    try:
        result = initiate_password_reset(email)
        if result['success']:
            return jsonify({'message': result['message']}), 200
        else:
            return jsonify({'error': result['message']}), 400
    except Exception as e:
        logger.error(f"Error in /auth/reset-unconfirmed-account: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/confirm-with-fallback', methods=['POST'])
def confirm_with_fallback():
    """Confirm signup with fallback options if confirmation fails"""
    data = request.get_json()
    email = data.get('email')
    code = data.get('code')
    
    if not email or not code:
        return jsonify({'error': 'Email and code required'}), 400
    
    try:
        result = confirm_signup_with_reset(email, code)
        if result['success']:
            return jsonify({'message': result['message']}), 200
        else:
            return jsonify({
                'error': result['message'],
                'can_retry': result.get('can_retry', False),
                'can_reset': result.get('can_reset', False),
                'error_code': result.get('error_code')
            }), 400
    except Exception as e:
        logger.error(f"Error in /auth/confirm-with-fallback: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 

@auth_bp.route('/user/social-links', methods=['POST'])
def save_social_links():
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    data = request.get_json()
    linkedin = data.get('linkedin')
    facebook = data.get('facebook')
    x = data.get('x')
    github = data.get('github')
    social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
    if social_links:
        social_links.linkedin = linkedin
        social_links.facebook = facebook
        social_links.x = x
        social_links.github = github
        social_links.updated_at = datetime.utcnow()
    else:
        social_links = UserSocialLinks(
            user_id=user.id,
            linkedin=linkedin,
            facebook=facebook,
            x=x,
            github=github
        )
        db.session.add(social_links)
    db.session.commit()
    return jsonify({'message': 'Social links saved successfully.'}), 200

@auth_bp.route('/user/social-links', methods=['GET'])
def get_social_links():
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
    if not social_links:
        return jsonify({'linkedin': '', 'facebook': '', 'x': '', 'github': ''}), 200
    return jsonify({
        'linkedin': social_links.linkedin or '',
        'facebook': social_links.facebook or '',
        'x': social_links.x or '',
        'github': social_links.github or ''
    }), 200 

@auth_bp.route('/admin/social-links', methods=['GET'])
def admin_get_all_social_links():
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email') or user_jwt.get('custom:role') != 'admin':
        return jsonify({'error': 'Forbidden'}), 403
    all_links = []
    for social in UserSocialLinks.query.all():
        user = User.query.get(social.user_id)
        all_links.append({
            'user_id': social.user_id,
            'email': user.email if user else '',
            'linkedin': social.linkedin or '',
            'facebook': social.facebook or '',
            'x': social.x or '',
            'github': social.github or ''
        })
    return jsonify({'social_links': all_links}), 200 

@auth_bp.route('/user/<int:user_id>/image', methods=['GET'])
def get_user_image(user_id):
    """Get user profile image"""
    try:
        # Get user image from database
        user_image = UserImage.query.filter_by(user_id=user_id).first()
        if not user_image:
            return jsonify({'error': 'User image not found'}), 404
        
        # Decode base64 image data with proper padding
        try:
            # Add padding if needed
            padded_data = user_image.image_data
            while len(padded_data) % 4 != 0:
                padded_data += '='
            
            image_data = base64.b64decode(padded_data)
        except Exception as decode_error:
            logger.error(f"Base64 decode error for user {user_id}: {decode_error}")
            # Try to fix common base64 issues
            try:
                # Remove any whitespace and try again
                cleaned_data = user_image.image_data.strip()
                image_data = base64.b64decode(cleaned_data)
            except Exception as final_error:
                logger.error(f"Final decode attempt failed for user {user_id}: {final_error}")
                return jsonify({'error': 'Image data corrupted'}), 500
        
        # Create response with proper content type
        response = send_file(
            io.BytesIO(image_data),
            mimetype=user_image.image_type,
            as_attachment=False,
            download_name=user_image.file_name or 'profile_image'
        )
        
        # Add cache headers with no-cache to prevent old images from being displayed
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving user image: {e}")
        return jsonify({'error': 'Failed to retrieve image'}), 500

@auth_bp.route('/user/image/upload', methods=['POST'])
def upload_user_image():
    """Upload user profile image"""
    try:
        # Get current user from JWT token or custom auth token
        current_user = get_current_user_flexible()
        if not current_user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = current_user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found in token'}), 400
        
        # Get user from database
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif']
        if image_file.content_type not in allowed_types:
            return jsonify({'error': 'Invalid file type. Only JPEG, PNG, and GIF are allowed'}), 400
        
        # Validate file size (max 5MB)
        max_size = 5 * 1024 * 1024  # 5MB
        image_file.seek(0, 2)  # Seek to end
        file_size = image_file.tell()
        image_file.seek(0)  # Reset to beginning
        
        if file_size > max_size:
            return jsonify({'error': 'File size too large. Maximum size is 5MB'}), 400
        
        # Read and process image
        image_data = image_file.read()
        
        # Optimize image quality if it's a JPEG
        if image_file.content_type in ['image/jpeg', 'image/jpg']:
            try:
                from PIL import Image
                import io
                
                # Open image with PIL
                img = Image.open(io.BytesIO(image_data))
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Optimize quality
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=95, optimize=True)
                image_data = output.getvalue()
                output.close()
                
            except Exception as e:
                logger.warning(f"Image optimization failed: {e}, using original image")
        
        # Convert to base64 for storage
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Check if user already has an image
        existing_image = UserImage.query.filter_by(user_id=user.id).first()
        
        if existing_image:
            # Update existing image
            existing_image.image_data = image_base64
            existing_image.image_type = image_file.content_type
            existing_image.file_name = image_file.filename
            existing_image.file_size = file_size
            existing_image.updated_at = datetime.utcnow()
            db.session.commit()
            
            logger.info(f"Updated profile image for user {user.email}")
            return jsonify({
                'message': 'Profile image updated successfully',
                'image_info': existing_image.to_dict()
            }), 200
        else:
            # Create new image record
            new_image = UserImage(
                user_id=user.id,
                image_data=image_base64,
                image_type=image_file.content_type,
                file_name=image_file.filename,
                file_size=file_size
            )
            
            db.session.add(new_image)
            db.session.commit()
            
            logger.info(f"Uploaded new profile image for user {user.email}")
            return jsonify({
                'message': 'Profile image uploaded successfully',
                'image_info': new_image.to_dict()
            }), 201
        
    except Exception as e:
        logger.error(f"Error uploading user image: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to upload image'}), 500

@auth_bp.route('/user/image', methods=['DELETE'])
def delete_user_image():
    """Delete user profile image"""
    try:
        # Get current user from JWT token or custom auth token
        current_user = get_current_user_flexible()
        if not current_user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = current_user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found in token'}), 400
        
        # Get user from database
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user has an image
        user_image = UserImage.query.filter_by(user_id=user.id).first()
        if not user_image:
            return jsonify({'error': 'No profile image found'}), 404
        
        # Delete the image
        db.session.delete(user_image)
        db.session.commit()
        
        logger.info(f"Deleted profile image for user {user.email}")
        return jsonify({'message': 'Profile image deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error deleting user image: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete image'}), 500

@auth_bp.route('/user/image/info', methods=['GET'])
def get_user_image_info():
    """Get user profile image information"""
    try:
        # Get current user from JWT token or custom auth token
        current_user = get_current_user_flexible()
        if not current_user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = current_user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found in token'}), 400
        
        # Get user from database
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user has an image
        user_image = UserImage.query.filter_by(user_id=user.id).first()
        if not user_image:
            # Return success with null image info instead of 404 error
            return jsonify({
                'message': 'No profile image found',
                'image_info': None
            }), 200
        
        return jsonify({
            'message': 'Profile image information retrieved successfully',
            'image_info': user_image.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting user image info: {e}")
        return jsonify({'error': 'Failed to get image information'}), 500

@auth_bp.route('/user/visa-status', methods=['POST'])
def update_visa_status():
    """Update visa status for job seekers"""
    try:
        # Get current user from JWT token or custom auth token
        current_user = get_current_user_flexible()
        if not current_user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = current_user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found in token'}), 400
        
        # Get user from database
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Only allow job seekers to update visa status
        if user.user_type not in ['job_seeker', 'employee']:
            return jsonify({'error': 'Visa status can only be updated by job seekers'}), 403
        
        data = request.get_json()
        visa_status = data.get('visa_status')
        
        if not visa_status:
            return jsonify({'error': 'Visa status is required'}), 400
        
        # Get or create candidate profile
        candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not candidate_profile:
            # Create a basic candidate profile if it doesn't exist
            candidate_profile = CandidateProfile(
                user_id=user.id,
                full_name=f"{user.first_name} {user.last_name}" if hasattr(user, 'first_name') else user.email
            )
            db.session.add(candidate_profile)
        
        # Update visa status
        candidate_profile.visa_status = visa_status
        db.session.commit()
        
        logger.info(f"Updated visa status for user {user.email}: {visa_status}")
        return jsonify({
            'message': 'Visa status updated successfully',
            'visa_status': visa_status
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating visa status: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update visa status'}), 500

@auth_bp.route('/user/visa-status', methods=['GET'])
def get_visa_status():
    """Get visa status for job seekers"""
    try:
        # Get current user from JWT token or custom auth token
        current_user = get_current_user_flexible()
        if not current_user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = current_user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found in token'}), 400
        
        # Get user from database
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get candidate profile
        candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        
        return jsonify({
            'message': 'Visa status retrieved successfully',
            'visa_status': candidate_profile.visa_status if candidate_profile else None
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting visa status: {e}")
        return jsonify({'error': 'Failed to get visa status'}), 500

@auth_bp.route('/user/profile', methods=['POST'])
def update_profile():
    """Update user profile information"""
    try:
        # Get user from JWT token using the flexible method
        current_user = get_current_user_flexible()
        if not current_user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = current_user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found in token'}), 400
        
        # Get user from database
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        email = data.get('email')
        company_name = data.get('companyName') or data.get('company_name')
        first_name = data.get('firstName') or data.get('first_name')
        last_name = data.get('lastName') or data.get('last_name')
        phone = data.get('phone')
        location = data.get('location')
        bio = data.get('bio') or data.get('summary')
        
        # Update email if changed
        if email is not None and email != user.email:
            # Check if email is already taken
            existing_user = User.query.filter_by(email=email).first()
            if existing_user and existing_user.id != user.id:
                return jsonify({'error': 'Email already in use'}), 400
            user.email = email
        
        # Update company name for employers/recruiters and admins/owners
        if company_name is not None and (user.user_type in ['employer', 'recruiter', 'admin'] or user.role in ['admin', 'owner']):
            user.company_name = company_name
        
        # Update firstName and lastName in Cognito (these are stored as given_name and family_name)
        if first_name is not None or last_name is not None:
            try:
                cognito_attrs = {}
                if first_name is not None:
                    cognito_attrs['given_name'] = first_name
                if last_name is not None:
                    cognito_attrs['family_name'] = last_name
                
                if cognito_attrs:
                    cognito_admin_update_user_attributes(user.email, cognito_attrs)
                    logger.info(f"Updated Cognito attributes for user {user.email}: {cognito_attrs}")
            except Exception as e:
                logger.warning(f"Failed to update Cognito attributes for {user.email}: {e}")
                # Continue even if Cognito update fails
        
        # Update phone, location, bio based on user type
        if user.user_type in ['job_seeker', 'employee'] or user.role in ['job_seeker', 'employee']:
            # For job seekers, update CandidateProfile
            if user.user_type == 'job_seeker' or user.role == 'job_seeker':
                candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
                if not candidate_profile:
                    # Create candidate profile if it doesn't exist
                    candidate_profile = CandidateProfile(
                        user_id=user.id,
                        full_name=f"{first_name or ''} {last_name or ''}".strip() or user.email
                    )
                    db.session.add(candidate_profile)
                
                if phone is not None:
                    candidate_profile.phone = phone
                if location is not None:
                    candidate_profile.location = location
                if bio is not None:
                    candidate_profile.summary = bio
                
                # Update full_name if first_name or last_name changed
                if first_name is not None or last_name is not None:
                    full_name = f"{first_name or ''} {last_name or ''}".strip()
                    if full_name:
                        candidate_profile.full_name = full_name
            
            # For employees, update EmployeeProfile
            elif user.user_type == 'employee' or user.role == 'employee':
                try:
                    from sqlalchemy.orm import load_only
                    # Query with only essential columns to avoid loading non-existent columns
                    employee_profile = EmployeeProfile.query.options(
                        load_only(EmployeeProfile.id, EmployeeProfile.user_id, 
                                 EmployeeProfile.first_name, EmployeeProfile.last_name,
                                 EmployeeProfile.phone, EmployeeProfile.location)
                    ).filter_by(user_id=user.id).first()
                except Exception as query_error:
                    # If query fails due to missing columns, try without load_only
                    if 'Unknown column' in str(query_error) or 'country_code' in str(query_error):
                        logger.warning(f"EmployeeProfile query failed due to missing columns for user {user.id}: {str(query_error)}. Trying basic query.")
                        try:
                            employee_profile = EmployeeProfile.query.filter_by(user_id=user.id).first()
                        except Exception:
                            employee_profile = None
                    else:
                        raise
                
                if not employee_profile:
                    # Create employee profile if it doesn't exist
                    employee_profile = EmployeeProfile(user_id=user.id)
                    db.session.add(employee_profile)
                
                if first_name is not None:
                    employee_profile.first_name = first_name
                if last_name is not None:
                    employee_profile.last_name = last_name
                if phone is not None:
                    employee_profile.phone = phone
                if location is not None:
                    employee_profile.location = location
        
        # For employers/recruiters, we could store phone/location in a separate model if needed
        # For now, we'll just update company_name which is already handled above
        
        db.session.commit()
        
        logger.info(f"Profile updated successfully for user {user.email}")
        return jsonify({'message': 'Profile updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({'error': 'Failed to update profile'}), 500

@auth_bp.route('/user/functionality-preferences', methods=['GET'])
def get_functionality_preferences():
    """Get user functionality preferences"""
    try:
        current_user = get_current_user_flexible()
        if not current_user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = current_user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found in token'}), 400
        
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Don't show functionality preferences for job seekers
        if user.user_type == 'job_seeker' or user.role == 'job_seeker':
            return jsonify({'functionalities': []}), 200
        
        preferences = UserFunctionalityPreferences.query.filter_by(user_id=user.id).first()
        
        if preferences:
            return jsonify({
                'functionalities': preferences.functionalities if isinstance(preferences.functionalities, list) else []
            }), 200
        else:
            return jsonify({'functionalities': []}), 200
            
    except Exception as e:
        logger.error(f"Error getting functionality preferences: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get functionality preferences'}), 500

@auth_bp.route('/user/functionality-preferences', methods=['POST'])
def save_functionality_preferences():
    """Save user functionality preferences"""
    try:
        current_user = get_current_user_flexible()
        if not current_user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = current_user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found in token'}), 400
        
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Don't allow job seekers to set functionality preferences
        if user.user_type == 'job_seeker' or user.role == 'job_seeker':
            return jsonify({'error': 'Functionality preferences are not available for job seekers'}), 403
        
        data = request.get_json()
        functionalities = data.get('functionalities', [])
        
        if not isinstance(functionalities, list):
            return jsonify({'error': 'Functionalities must be an array'}), 400
        
        # Validate functionality IDs (optional - you can add a list of valid IDs)
        valid_ids = [
            'talent_matchmaker', 'payroll', 'agentic_ai', 'compliance',
            'candidate_search', 'document_management', 'scheduling',
            'analytics', 'communications', 'workflow_automation'
        ]
        
        # Filter out invalid IDs (optional validation)
        validated_functionalities = [f for f in functionalities if f in valid_ids] if valid_ids else functionalities
        
        # Upsert preferences
        preferences = UserFunctionalityPreferences.query.filter_by(user_id=user.id).first()
        
        if preferences:
            preferences.functionalities = validated_functionalities
            preferences.updated_at = datetime.utcnow()
        else:
            preferences = UserFunctionalityPreferences(
                user_id=user.id,
                functionalities=validated_functionalities
            )
            db.session.add(preferences)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Functionality preferences saved successfully',
            'functionalities': validated_functionalities
        }), 200
        
    except Exception as e:
        logger.error(f"Error saving functionality preferences: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({'error': 'Failed to save functionality preferences'}), 500

@auth_bp.route('/user/profile-completion', methods=['GET'])
def get_profile_completion():
    """Get user profile completion percentage based on all profile fields"""
    try:
        current_user = get_current_user_flexible()
        if not current_user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = current_user.get('email')
        if not user_email:
            return jsonify({'error': 'User email not found in token'}), 400
        
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        completed_sections = 0
        # For job seekers, only 3 sections (Personal Info, Business Details, Bank Account)
        # For other roles, 5 sections (add Preferences and Service Requirements)
        is_job_seeker = (user.user_type == 'job_seeker' or user.role == 'job_seeker')
        total_sections = 3 if is_job_seeker else 5
        
        # 1. Personal Information - check firstName, lastName, email, phone
        # Check in appropriate profile based on user type
        first_name = None
        last_name = None
        phone = None
        
        # For job seekers, check CandidateProfile
        if user.user_type == 'job_seeker' or user.role == 'job_seeker':
            candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
            if candidate_profile:
                # Extract first and last name from full_name
                if candidate_profile.full_name:
                    name_parts = candidate_profile.full_name.strip().split(' ', 1)
                    first_name = name_parts[0] if len(name_parts) > 0 else None
                    last_name = name_parts[1] if len(name_parts) > 1 else None
                phone = candidate_profile.phone
        
        # For employees, check EmployeeProfile
        elif user.user_type == 'employee' or user.role == 'employee':
            try:
                from sqlalchemy.orm import load_only
                # Query with only essential columns to avoid loading non-existent columns
                employee_profile = EmployeeProfile.query.options(
                    load_only(EmployeeProfile.id, EmployeeProfile.user_id,
                             EmployeeProfile.first_name, EmployeeProfile.last_name,
                             EmployeeProfile.phone)
                ).filter_by(user_id=user.id).first()
            except Exception as query_error:
                # If query fails due to missing columns, try without load_only or skip
                if 'Unknown column' in str(query_error) or 'country_code' in str(query_error):
                    logger.debug(f"EmployeeProfile query failed due to missing columns for user {user.id}: {str(query_error)}. Trying basic query.")
                    try:
                        employee_profile = EmployeeProfile.query.filter_by(user_id=user.id).first()
                    except Exception:
                        employee_profile = None
                else:
                    employee_profile = None
            
            if employee_profile:
                first_name = employee_profile.first_name
                last_name = employee_profile.last_name
                phone = employee_profile.phone
        
        # For employers/recruiters/admins, try multiple sources
        else:
            # First, check if there's an EmployeeProfile (some employers/recruiters might have one)
            try:
                from sqlalchemy.orm import load_only
                # Query with only essential columns to avoid loading non-existent columns
                employee_profile = EmployeeProfile.query.options(
                    load_only(EmployeeProfile.id, EmployeeProfile.user_id,
                             EmployeeProfile.first_name, EmployeeProfile.last_name,
                             EmployeeProfile.phone)
                ).filter_by(user_id=user.id).first()
            except Exception as query_error:
                # If query fails due to missing columns, try without load_only or skip
                if 'Unknown column' in str(query_error) or 'country_code' in str(query_error):
                    logger.debug(f"EmployeeProfile query failed due to missing columns for user {user.id}: {str(query_error)}. Skipping EmployeeProfile lookup.")
                    employee_profile = None
                else:
                    # For other errors, try basic query
                    try:
                        employee_profile = EmployeeProfile.query.filter_by(user_id=user.id).first()
                    except Exception:
                        employee_profile = None
            
            if employee_profile:
                first_name = employee_profile.first_name
                last_name = employee_profile.last_name
                phone = employee_profile.phone
            
            # If not found in EmployeeProfile, try to get from Cognito attributes
            if not first_name or not last_name:
                try:
                    from .cognito import get_user_by_email
                    cognito_user = get_user_by_email(user.email)
                    if cognito_user:
                        attrs = {attr['Name']: attr['Value'] for attr in cognito_user.get('UserAttributes', [])}
                        if not first_name:
                            first_name = attrs.get('given_name') or attrs.get('custom:first_name') or attrs.get('firstName')
                        if not last_name:
                            last_name = attrs.get('family_name') or attrs.get('custom:last_name') or attrs.get('lastName')
                        if not phone:
                            phone = attrs.get('phone_number') or attrs.get('custom:phone') or attrs.get('phone')
                except Exception as e:
                    logger.debug(f"Could not get Cognito attributes for {user.email}: {e}")
            
            # Try to get from User model if fields exist (fallback)
            if not first_name:
                first_name = getattr(user, 'first_name', None) or getattr(user, 'firstName', None)
            if not last_name:
                last_name = getattr(user, 'last_name', None) or getattr(user, 'lastName', None)
            if not phone:
                phone = getattr(user, 'phone', None) or getattr(user, 'phone_number', None)
        
        # For job seekers and employees, require phone. For others (employers/recruiters/admins), phone is optional
        is_job_seeker_or_employee = (user.user_type in ['job_seeker', 'employee'] or user.role in ['job_seeker', 'employee'])
        
        if is_job_seeker_or_employee:
            # Job seekers and employees must have phone
            has_personal_info = bool(
                first_name and str(first_name).strip() and
                last_name and str(last_name).strip() and
                user.email and str(user.email).strip() and
                phone and str(phone).strip()
            )
        else:
            # For employers/recruiters/admins, phone is optional
            has_personal_info = bool(
                first_name and str(first_name).strip() and
                last_name and str(last_name).strip() and
                user.email and str(user.email).strip()
            )
        if has_personal_info:
            completed_sections += 1
        
        # 2. Business Details - check onboarding submission for companyName or legalBusinessName
        has_business_details = False
        onboarding_submission = OnboardingSubmission.query.filter_by(
            user_id=user.id
        ).order_by(OnboardingSubmission.created_at.desc()).first()
        
        if onboarding_submission and onboarding_submission.data:
            data = onboarding_submission.data
            company_name = data.get('companyName') or data.get('company_name') or data.get('legalBusinessName')
            has_business_details = bool(company_name and str(company_name).strip())
        
        if has_business_details:
            completed_sections += 1
        
        # 3. Bank Account - check if bank account exists with bankName or accountNumber
        has_bank_account = False
        bank_account = UserBankAccount.query.filter_by(user_id=user.id).first()
        if bank_account:
            has_bank_account = bool(
                (bank_account.bank_name and bank_account.bank_name.strip()) or
                (bank_account.account_number and bank_account.account_number.strip())
            )
        
        if has_bank_account:
            completed_sections += 1
        
        # 4. Preferences - check if functionality preferences exist AND have at least one functionality
        # For job seekers, skip this check as they don't have functionality preferences
        has_preferences = False
        preferences = None
        if user.user_type != 'job_seeker' and user.role != 'job_seeker':
            preferences = UserFunctionalityPreferences.query.filter_by(user_id=user.id).first()
            if preferences and preferences.functionalities:
                functionalities = preferences.functionalities if isinstance(preferences.functionalities, list) else []
                has_preferences = len(functionalities) > 0
        
        if has_preferences:
            completed_sections += 1
        
        # 5. Service Requirements - check if functionality preferences have any functionalities
        # This is the same as Preferences for non-job-seekers, but kept separate for UI clarity
        has_service_requirements = False
        if user.user_type != 'job_seeker' and user.role != 'job_seeker':
            if preferences and preferences.functionalities:
                functionalities = preferences.functionalities if isinstance(preferences.functionalities, list) else []
                has_service_requirements = len(functionalities) > 0
        
        if has_service_requirements:
            completed_sections += 1
        
        percentage = round((completed_sections / total_sections) * 100) if total_sections > 0 else 0
        
        return jsonify({
            'percentage': percentage,
            'completedItems': completed_sections,
            'totalItems': total_sections,
            'sections': {
                'personalInfo': has_personal_info,
                'businessDetails': has_business_details,
                'bankAccount': has_bank_account,
                'preferences': has_preferences,
                'serviceRequirements': has_service_requirements
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting profile completion: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get profile completion'}), 500

# LinkedIn OAuth Authentication Routes
@auth_bp.route('/oauth/linkedin/member/callback', methods=['GET'])
def linkedin_member_callback():
    """LinkedIn OAuth callback for individual members"""
    try:
        # Get authorization code from query parameters
        code = request.args.get('code')
        state = request.args.get('state')
        
        if not code:
            return jsonify({'error': 'Authorization code not provided'}), 400
        
        logger.info(f"🔗 LinkedIn member callback received - code: {code[:10]}..., state: {state}")
        
        # Exchange authorization code for access token
        token_response = exchange_linkedin_code_for_token(code, 'member')
        
        if not token_response:
            return jsonify({'error': 'Failed to exchange code for token'}), 400
        
        access_token = token_response.get('access_token')
        
        # Get user profile from LinkedIn
        user_profile = get_linkedin_member_profile(access_token)
        
        if not user_profile:
            return jsonify({'error': 'Failed to get LinkedIn profile'}), 400
        
        # Extract user information
        email = user_profile.get('email-address')
        first_name = user_profile.get('localized-first-name', '')
        last_name = user_profile.get('localized-last-name', '')
        linkedin_id = user_profile.get('id')
        
        if not email:
            return jsonify({'error': 'Email not provided by LinkedIn'}), 400
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        
        if user:
            # User exists, log them in
            logger.info(f"🔗 LinkedIn member login for existing user: {email}")
            return handle_linkedin_existing_user_login(user, access_token, linkedin_id)
        else:
            # New user, create account
            logger.info(f"🔗 LinkedIn member signup for new user: {email}")
            return handle_linkedin_new_user_signup(email, first_name, last_name, access_token, linkedin_id, 'job_seeker')
            
    except Exception as e:
        logger.error(f"❌ Error in LinkedIn member callback: {str(e)}", exc_info=True)
        return jsonify({'error': 'LinkedIn authentication failed'}), 500

@auth_bp.route('/oauth/linkedin/org/callback', methods=['GET'])
def linkedin_org_callback():
    """LinkedIn OAuth callback for organizations"""
    try:
        # Get authorization code from query parameters
        code = request.args.get('code')
        state = request.args.get('state')
        
        if not code:
            return jsonify({'error': 'Authorization code not provided'}), 400
        
        logger.info(f"🔗 LinkedIn org callback received - code: {code[:10]}..., state: {state}")
        
        # Exchange authorization code for access token
        token_response = exchange_linkedin_code_for_token(code, 'organization')
        
        if not token_response:
            return jsonify({'error': 'Failed to exchange code for token'}), 400
        
        access_token = token_response.get('access_token')
        
        # Get organization profile from LinkedIn
        org_profile = get_linkedin_organization_profile(access_token)
        
        if not org_profile:
            return jsonify({'error': 'Failed to get LinkedIn organization profile'}), 400
        
        # Extract organization information
        org_id = org_profile.get('id')
        org_name = org_profile.get('localized-name', '')
        org_email = org_profile.get('email-address')  # Admin email
        
        if not org_email:
            return jsonify({'error': 'Organization email not provided by LinkedIn'}), 400
        
        # Check if user exists
        user = User.query.filter_by(email=org_email).first()
        
        if user:
            # User exists, log them in
            logger.info(f"🔗 LinkedIn org login for existing user: {org_email}")
            return handle_linkedin_existing_user_login(user, access_token, org_id, is_org=True)
        else:
            # New user, create account
            logger.info(f"🔗 LinkedIn org signup for new user: {org_email}")
            return handle_linkedin_new_user_signup(org_email, '', '', access_token, org_id, 'employer', org_name)
            
    except Exception as e:
        logger.error(f"❌ Error in LinkedIn org callback: {str(e)}", exc_info=True)
        return jsonify({'error': 'LinkedIn organization authentication failed'}), 500

def exchange_linkedin_code_for_token(code, scope_type):
    """Exchange LinkedIn authorization code for access token"""
    try:
        # LinkedIn OAuth configuration
        client_id = os.environ.get('LINKEDIN_CLIENT_ID')
        client_secret = os.environ.get('LINKEDIN_CLIENT_SECRET')
        redirect_uri = f"https://kempian.ai/oauth/linkedin/{scope_type}/callback"
        
        if not client_id or not client_secret:
            logger.error("❌ LinkedIn OAuth credentials not configured")
            return None
        
        # Exchange code for token
        token_url = "https://www.linkedin.com/oauth/v2/accessToken"
        token_data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': redirect_uri,
            'client_id': client_id,
            'client_secret': client_secret
        }
        
        response = requests.post(token_url, data=token_data)
        
        if response.status_code == 200:
            token_info = response.json()
            logger.info(f"✅ LinkedIn token exchange successful for {scope_type}")
            return token_info
        else:
            logger.error(f"❌ LinkedIn token exchange failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error exchanging LinkedIn code for token: {str(e)}")
        return None

def get_linkedin_member_profile(access_token):
    """Get LinkedIn member profile"""
    try:
        headers = {
            'Authorization': f'Bearer {access_token}',
            'X-Restli-Protocol-Version': '2.0.0'
        }
        
        # Get basic profile
        profile_url = "https://api.linkedin.com/v2/me"
        response = requests.get(profile_url, headers=headers)
        
        if response.status_code == 200:
            profile = response.json()
            
            # Get email address
            email_url = "https://api.linkedin.com/v2/emailAddress?q=members&projection=(elements*(handle~))"
            email_response = requests.get(email_url, headers=headers)
            
            if email_response.status_code == 200:
                email_data = email_response.json()
                email = email_data.get('elements', [{}])[0].get('handle~', {}).get('emailAddress')
                profile['email-address'] = email
            
            return profile
        else:
            logger.error(f"❌ Failed to get LinkedIn member profile: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error getting LinkedIn member profile: {str(e)}")
        return None

def get_linkedin_organization_profile(access_token):
    """Get LinkedIn organization profile"""
    try:
        headers = {
            'Authorization': f'Bearer {access_token}',
            'X-Restli-Protocol-Version': '2.0.0'
        }
        
        # Get organization profile
        org_url = "https://api.linkedin.com/v2/organizations"
        response = requests.get(org_url, headers=headers)
        
        if response.status_code == 200:
            org_data = response.json()
            orgs = org_data.get('elements', [])
            
            if orgs:
                org = orgs[0]  # Get first organization
                org_id = org.get('id')
                
                # Get detailed organization info
                detail_url = f"https://api.linkedin.com/v2/organizations/{org_id}"
                detail_response = requests.get(detail_url, headers=headers)
                
                if detail_response.status_code == 200:
                    org_detail = detail_response.json()
                    org_detail['email-address'] = os.environ.get('LINKEDIN_ORG_EMAIL')  # Admin email
                    return org_detail
            
            return None
        else:
            logger.error(f"❌ Failed to get LinkedIn organization profile: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error getting LinkedIn organization profile: {str(e)}")
        return None

def handle_linkedin_existing_user_login(user, access_token, linkedin_id, is_org=False):
    """Handle login for existing user via LinkedIn"""
    try:
        # Update user's LinkedIn ID if needed
        if not user.linkedin_id:
            user.linkedin_id = linkedin_id
            db.session.commit()
        
        # Generate JWT token
        token_payload = {
            'user_id': user.id,
            'email': user.email,
            'role': user.role,
            'tenant_id': user.tenant_id
        }
        
        token = jwt.encode(token_payload, current_app.config['SECRET_KEY'], algorithm='HS256')
        
        # Get candidate profile data for job seekers
        candidate_profile = None
        if user.role == 'job_seeker' or user.user_type == 'job_seeker':
            from app.models import CandidateProfile
            candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        
        # Get projects and certifications for job seekers
        projects_data = []
        certifications_data = []
        if candidate_profile:
            from app.models import CandidateProject, CandidateCertification
            projects = CandidateProject.query.filter_by(profile_id=candidate_profile.id).all()
            projects_data = [project.to_dict() for project in projects]
            
            certifications = CandidateCertification.query.filter_by(profile_id=candidate_profile.id).all()
            certifications_data = [cert.to_dict() for cert in certifications]
        
        # Return user data and token
        user_data = {
            "id": str(user.id),
            "email": user.email,
            "firstName": user.first_name or "",
            "lastName": user.last_name or "",
            "role": user.role,
            "userType": user.user_type or user.role,
            "companyName": user.company_name,
            "phone": candidate_profile.phone if candidate_profile else None,
            "location": candidate_profile.location if candidate_profile else None,
            "bio": candidate_profile.summary if candidate_profile else None,
            "visaStatus": candidate_profile.visa_status if candidate_profile else None,
            "projects": projects_data,
            "certifications": certifications_data
        }
        
        return jsonify({
            "access_token": token,
            "id_token": token,
            "refresh_token": "",
            "user": user_data,
            "message": f"LinkedIn {'organization' if is_org else 'member'} login successful"
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error handling LinkedIn existing user login: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

def handle_linkedin_new_user_signup(email, first_name, last_name, access_token, linkedin_id, role, company_name=None):
    """Handle signup for new user via LinkedIn"""
    try:
        # Create tenant
        trial_plan = Plan.query.filter_by(name="Free Trial").first()
        if not trial_plan:
            return jsonify({'error': 'Free Trial plan not found'}), 500
        
        tenant = Tenant(
            plan_id=trial_plan.id,
            stripe_customer_id="",
            stripe_subscription_id="",
            status="active"
        )
        db.session.add(tenant)
        db.session.commit()
        
        # Determine tenant role
        if role == 'admin':
            tenant_role = 'admin'
        elif role in ['employer', 'recruiter']:
            tenant_role = 'owner'
        else:
            tenant_role = role
        
        # Create user
        db_user = User(
            tenant_id=tenant.id,
            email=email,
            role=tenant_role,
            user_type=role,
            first_name=first_name,
            last_name=last_name,
            company_name=company_name,
            linkedin_id=linkedin_id
        )
        db.session.add(db_user)
        db.session.commit()
        
        # Create trial
        create_user_trial(db_user.id)
        
        # Generate JWT token
        token_payload = {
            'user_id': db_user.id,
            'email': db_user.email,
            'role': db_user.role,
            'tenant_id': db_user.tenant_id
        }
        
        token = jwt.encode(token_payload, current_app.config['SECRET_KEY'], algorithm='HS256')
        
        # Return user data and token
        user_data = {
            "id": str(db_user.id),
            "email": db_user.email,
            "firstName": db_user.first_name or "",
            "lastName": db_user.last_name or "",
            "role": db_user.role,
            "userType": db_user.user_type or db_user.role,
            "companyName": db_user.company_name
        }
        
        return jsonify({
            "access_token": token,
            "id_token": token,
            "refresh_token": "",
            "user": user_data,
            "message": f"LinkedIn {'organization' if company_name else 'member'} signup successful"
        }), 201
        
    except Exception as e:
        logger.error(f"❌ Error handling LinkedIn new user signup: {str(e)}")
        return jsonify({'error': 'Signup failed'}), 500

@auth_bp.route('/company-suggestions', methods=['GET'])
def get_company_suggestions():
    """Get company name suggestions based on search query"""
    try:
        logger.info(f"Company suggestions request received: {request.args}")
        query = request.args.get('q', '').strip()
        limit = request.args.get('limit', 10, type=int)
        
        logger.info(f"Searching for companies with query: '{query}', limit: {limit}")
        
        if not query or len(query) < 2:
            logger.info("Query too short, returning empty suggestions")
            return jsonify({'suggestions': []}), 200
        
        # Search for companies that contain the query (case-insensitive)
        companies = User.query.filter(
            User.company_name.isnot(None),
            User.company_name != '',
            User.company_name.ilike(f'%{query}%')
        ).distinct().limit(limit).all()
        
        logger.info(f"Found {len(companies)} companies in database")
        
        # Extract unique company names
        suggestions = list(set([user.company_name for user in companies if user.company_name]))
        
        # Sort by relevance (exact matches first, then alphabetical)
        suggestions.sort(key=lambda x: (
            not x.lower().startswith(query.lower()),  # Exact matches first
            x.lower()  # Then alphabetical
        ))
        
        logger.info(f"Returning {len(suggestions)} suggestions: {suggestions}")
        
        return jsonify({
            'suggestions': suggestions[:limit]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting company suggestions: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to get company suggestions'}), 500 

@auth_bp.route('/profile/check', methods=['GET'])
def check_user_profile():
    """Check if user has a complete profile - for jobseekers and employers/recruiters"""
    try:
        # Get current user from JWT token
        user = get_current_user_flexible()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        # For employers/recruiters, check if they have company name
        if user.user_type in ['employer', 'recruiter']:
            has_company_name = bool(user.company_name and user.company_name.strip())
            return jsonify({
                'has_profile': True,
                'is_complete': has_company_name,
                'message': 'Profile check not required for this user type.' if has_company_name else 'Please enter your company name to complete your profile.',
                'skip_check': True,
                'needs_company_name': not has_company_name,
                'company_name': user.company_name or ''
            }), 200
        
        # Only check profile for jobseekers
        if user.user_type != 'job_seeker' and user.role != 'job_seeker':
            return jsonify({
                'has_profile': True,
                'is_complete': True,
                'message': 'Profile check not required for this user type.',
                'skip_check': True
            }), 200
        
        # Check if user has a candidate profile
        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        
        if not profile:
            return jsonify({
                'has_profile': False,
                'is_complete': False,
                'message': 'No profile found. Please upload your resume to create a profile.'
            }), 200
        
        # Check if profile is complete (has basic required fields)
        is_complete = bool(
            profile.full_name and 
            profile.email and 
            (profile.skills or profile.experience_years is not None)
        )
        
        return jsonify({
            'has_profile': True,
            'is_complete': is_complete,
            'profile_id': profile.id,
            'message': 'Profile found' if is_complete else 'Profile incomplete. Please upload resume to complete it.'
        }), 200
        
    except Exception as e:
        logger.error(f"Error checking user profile: {str(e)}")
        return jsonify({'error': 'Failed to check profile status'}), 500 

