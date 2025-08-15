import os
import uuid
import boto3
from datetime import datetime, date
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from sqlalchemy.exc import IntegrityError
from app.models import db, User, CandidateProfile, CandidateSkill, CandidateEducation, CandidateExperience, CandidateCertification, UserSocialLinks
from app.search.routes import get_user_from_jwt, get_jwt_payload

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

S3_BUCKET = "resume-bucket-adept-ai-pro"

talent_bp = Blueprint('talent', __name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        s3_key = f"resumes/{user.id}/{unique_filename}"

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
        s3_key = f"resumes/{user.id}/{unique_filename}"

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

        # Generate presigned URL for download (valid for 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': profile.resume_s3_key},
            ExpiresIn=3600
        )

        return jsonify({
            'download_url': presigned_url,
            'filename': profile.resume_filename
        }), 200

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

        # Only employers and recruiters can view candidates
        if user.role not in ['employer', 'recruiter', 'admin']:
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