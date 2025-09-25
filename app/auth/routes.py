import re
from flask import Blueprint, request, jsonify, send_file, current_app
from app.simple_logger import get_logger
from werkzeug.exceptions import HTTPException
from .cognito import cognito_signup, cognito_confirm_signup, cognito_login, cognito_admin_update_user_attributes
from .cognito import cognito_client, COGNITO_USER_POOL_ID, COGNITO_REGION
import os
import boto3
from datetime import datetime
from app.utils import get_current_user, get_current_user_flexible
from app.models import db, Tenant, User, Plan, UserSocialLinks, UserImage
from app.search.routes import get_user_from_jwt
from sqlalchemy.exc import SQLAlchemyError
from jose import jwt
import requests
import json
from app.utils.trial_manager import create_user_trial
import base64
import io
from PIL import Image
from app.models import CandidateProfile
from app.auth.unconfirmed_handler import (
    check_user_status, 
    resend_confirmation_code, 
    initiate_password_reset, 
    confirm_signup_with_reset,
    get_recovery_options
)

logger = get_logger("auth")

auth_bp = Blueprint('auth', __name__)

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
                # Split by common project separators
                project_lines = re.split(r'[;\n]', projects_text)
                for project_line in project_lines[:5]:  # Limit to 5 projects
                    project_line = project_line.strip()
                    if project_line and len(project_line) > 10:
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
        
        # Add parsed education
        if resume_data.get('education'):
            from app.models import CandidateEducation
            import re
            from datetime import datetime
            
            # Parse education from the extracted text
            education_text = resume_data['education']
            if education_text:
                # Split by double newlines to get education entries
                education_entries = re.split(r'\n\s*\n', education_text)
                education_entries = [entry.strip() for entry in education_entries if entry.strip()]
                
                # If no double newlines, try splitting by single newlines
                if len(education_entries) <= 1:
                    education_entries = re.split(r'\n', education_text)
                    education_entries = [entry.strip() for entry in education_entries if entry.strip()]
                
                for i, edu_entry in enumerate(education_entries[:3]):  # Limit to 3 education entries
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
        
        db.session.commit()
        
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
        # --- Allocate Free Trial Plan if user has no tenant ---
        # Check if user already exists in DB
        user = User.query.filter_by(email=email).first()
        tenant_id = None
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
            # Create user as owner
            # Get the original user type from Cognito
            cognito_client = boto3.client('cognito-idp', region_name=COGNITO_REGION)
            user_info = cognito_client.admin_get_user(
                UserPoolId=COGNITO_USER_POOL_ID,
                Username=email
            )
            attrs = {attr['Name']: attr['Value'] for attr in user_info['UserAttributes']}
            original_user_type = attrs.get("custom:user_type", attrs.get("custom:role", "owner"))

            
            # Determine tenant role based on user type
            if original_user_type == 'admin':
                tenant_role = 'admin'
            elif original_user_type in ['employer', 'recruiter']:
                tenant_role = 'owner'
            else:
                tenant_role = original_user_type
            
            # Always store the original user type for display, even if system role is owner/admin
            db_user = User(tenant_id=tenant.id, email=email, role=tenant_role, user_type=original_user_type, company_name=company_name)

            db.session.add(db_user)
            db.session.commit()
            
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

            # Update company name if user exists and it's not set
            if company_name and not user.company_name and user.user_type in ['employer', 'recruiter']:
                user.company_name = company_name
                db.session.commit()

            tenant_id = user.tenant_id
        # Always update Cognito user with tenant_id
        try:
            cognito_admin_update_user_attributes(email, {"custom:tenant_id": str(tenant_id)})
        except Exception as e:
            logger.error(f"Failed to update Cognito tenant_id for {email}: {e}")
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
        
        # Initiate auth with temporary password
        auth_response = cognito_client.initiate_auth(
            ClientId=COGNITO_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': email,
                'PASSWORD': temp_password,
                'SECRET_HASH': get_secret_hash(email)
            }
        )
        
        # If we get here, the temp password is correct
        # Now change the password
        cognito_client.respond_to_auth_challenge(
            ClientId=COGNITO_CLIENT_ID,
            ChallengeName='NEW_PASSWORD_REQUIRED',
            Session=auth_response['Session'],
            ChallengeResponses={
                'USERNAME': email,
                'NEW_PASSWORD': new_password,
                'SECRET_HASH': get_secret_hash(email)
            }
        )
        
        return jsonify({'message': 'Password changed successfully. You can now log in.'}), 200
        
    except Exception as e:
        logger.error(f"Error in /auth/confirm-temp-password: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    requested_role = data.get('role')  # Add role parameter for role change requests
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    try:
        tokens = cognito_login(email, password)
        # Fetch user attributes from Cognito using email resolution
        from .cognito import get_user_by_email, cognito_admin_update_user_attributes
        user_info = get_user_by_email(email)
        attrs = {attr['Name']: attr['Value'] for attr in user_info['UserAttributes']}
        
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
        
        # --- Ensure Starter Plan and Tenant after login ---
        db_user = User.query.filter_by(email=email).first()
        tenant_id = None
        if not db_user:
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
            db.session.commit()
            # Create user as owner
            # Get the original user type from Cognito
            user_info = get_user_by_email(email)
            attrs = {attr['Name']: attr['Value'] for attr in user_info['UserAttributes']}
            original_role = attrs.get("custom:role", "owner")
            
            # Store the original user type for display purposes
            user_type = original_role if original_role in ['job_seeker', 'employee', 'recruiter', 'employer'] else None
            
            # Preserve admin role - don't override it with "owner"
            db_role = original_role if original_role == 'admin' else "owner"
            
            db_user = User(tenant_id=tenant.id, email=email, role=db_role, user_type=user_type)
            db.session.add(db_user)
            db.session.commit()
            tenant_id = tenant.id
        else:
            tenant_id = db_user.tenant_id
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
        

        
        return jsonify({
            "access_token": tokens.get("AccessToken"),
            "id_token": tokens.get("IdToken"),
            "refresh_token": tokens.get("RefreshToken"),
            "user": user
        }), 200
    except Exception as e:
        logger.error(f"Error in /auth/login: {str(e)}", exc_info=True)
        
        # Check if user exists and is unconfirmed
        error_message = str(e)
        if 'UserNotConfirmedException' in error_message:
            # Check user status to provide better guidance
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
        
        return jsonify({'error': error_message}), 401

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
        
        # Use role from state, then fallback, then database, then default
        role = role_from_state or role_fallback or claims.get('custom:role', 'job_seeker')
        user_type = claims.get('custom:user_type', role)
        
        # If user exists in database, get their stored role as fallback
        if db_user and not role_from_state and not role_fallback and not claims.get('custom:role'):
            role = db_user.user_type or db_user.role
            user_type = db_user.user_type or db_user.role
        
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
        
        if not db_user:
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
        else:
            # logger.info(f"✅ User already exists in database with ID: {db_user.id}")
            tenant_id = db_user.tenant_id
            
        # 3. Update Cognito user attributes with the role if it's different
        cognito_role = claims.get('custom:role')
        if (role_from_state and role_from_state != cognito_role) or \
           (not cognito_role and role != 'job_seeker') or \
           (db_user and not cognito_role and db_user.user_type):
            try:
                # logger.info(f"🔄 Updating Cognito user role from {cognito_role} to {role}")
                cognito_admin_update_user_attributes(email, {
                    "custom:role": role,
                    "custom:user_type": user_type
                })
            except Exception as e:
                logger.error(f"Failed to update Cognito role for {email}: {e}")
        
        # 4. Update database user record if needed
        if db_user and (db_user.user_type != user_type or db_user.role != role):
            db_user.user_type = user_type
            if role != 'owner':
                db_user.role = role
            db.session.commit()
            
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
        
        # logger.info(f"✅ Returning user data: {user}")
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
        company_name = data.get('companyName')
        
        # Update user fields that exist in the model
        if email is not None and email != user.email:
            # Check if email is already taken
            existing_user = User.query.filter_by(email=email).first()
            if existing_user and existing_user.id != user.id:
                return jsonify({'error': 'Email already in use'}), 400
            user.email = email
        
        # Update company name for employers/recruiters
        if company_name is not None and user.user_type in ['employer', 'recruiter']:
            user.company_name = company_name
        
        db.session.commit()
        
        return jsonify({'message': 'Profile updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to update profile'}), 500

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
    """Check if user has a complete profile - only for jobseekers"""
    try:
        # Get current user from JWT token
        user = get_current_user_flexible()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
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

