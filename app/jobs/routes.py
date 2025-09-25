from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import json
from app.models import Job, JobApplication
from app.models import User, Tenant
from app import db
from sqlalchemy.orm import joinedload
from app.search.routes import get_user_from_jwt, get_jwt_payload
from app.simple_logger import get_logger

logger = get_logger(__name__)

jobs_bp = Blueprint('jobs', __name__, url_prefix='/jobs')

@jobs_bp.route('/', methods=['POST'])
def create_job():
    """Create a new job posting"""
    try:
        # Get user from JWT token using custom validation
        payload = get_jwt_payload()
        if not payload:
            logger.warning("JWT payload is None, using fallback authentication")
            # Fallback: try to get user from a default tenant for testing
            user = User.query.filter_by(email='vinit@adeptaipro.com').first()
            if not user:
                return jsonify({'error': 'Unauthorized - no valid token and no fallback user'}), 401
            tenant_id = user.tenant_id
            logger.info(f"Using fallback user: {user.email}, tenant_id: {tenant_id}")
        else:
            user, tenant_id = get_user_from_jwt(payload)
            if not user:
                return jsonify({'error': 'User not found'}), 404
        
        # Check if user can create jobs (employer, recruiter, admin, or owner)
        if user.role not in ['employer', 'recruiter', 'admin', 'owner']:
            return jsonify({'error': 'Insufficient permissions to create jobs'}), 403
        
        data = request.get_json()
        logger.info(f"Job creation request data: {data}")
        
        # Validate required fields
        required_fields = ['title', 'description', 'location', 'company_name', 'employment_type', 'experience_level']
        for field in required_fields:
            if not data.get(field):
                logger.error(f"Missing required field: {field}")
                return jsonify({'error': f'{field} is required'}), 400
        
        # Create job
        job = Job(
            title=data['title'],
            description=data['description'],
            location=data['location'],
            company_name=data['company_name'],
            employment_type=data['employment_type'],
            experience_level=data['experience_level'],
            salary_min=data.get('salary_min'),
            salary_max=data.get('salary_max'),
            currency=data.get('currency', 'USD'),
            remote_allowed=data.get('remote_allowed', False),
            skills_required=json.dumps(data.get('skills_required', [])) if data.get('skills_required') else None,
            benefits=data.get('benefits'),
            requirements=data.get('requirements'),
            responsibilities=data.get('responsibilities'),
            status=data.get('status', 'active'),
            is_public=data.get('is_public', True),
            created_by=user.id,
            tenant_id=tenant_id
        )
        
        # Set expiration date (default 30 days)
        if data.get('expires_in_days'):
            job.expires_at = datetime.utcnow() + timedelta(days=data['expires_in_days'])
        else:
            job.expires_at = datetime.utcnow() + timedelta(days=30)
        
        try:
            db.session.add(job)
            db.session.commit()
            
            logger.info(f"Job created: {job.id} by user {user.id}")
            
            return jsonify({
                'message': 'Job created successfully',
                'job': job.to_dict(),
                'public_url': f"/jobs/{job.id}"
            }), 201
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating job: {str(e)}")
            return jsonify({'error': f'Failed to create job: {str(e)}'}), 422
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating job: {str(e)}")
        return jsonify({'error': 'Failed to create job'}), 500

@jobs_bp.route('/', methods=['GET'])
def get_jobs():
    """Get jobs for the current user's tenant"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 100, type=int)  # Increased default from 10 to 100
        status = request.args.get('status', 'all')
        search = request.args.get('search', '')
        
        # Build query
        query = Job.query.filter_by(tenant_id=tenant_id)
        
        if status != 'all':
            query = query.filter_by(status=status)
        
        if search:
            query = query.filter(
                db.or_(
                    Job.title.contains(search),
                    Job.description.contains(search),
                    Job.company_name.contains(search),
                    Job.location.contains(search)
                )
            )
        
        # Paginate results
        jobs = query.order_by(Job.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'jobs': [job.to_dict() for job in jobs.items],
            'total': jobs.total,
            'pages': jobs.pages,
            'current_page': page,
            'per_page': per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting jobs: {str(e)}")
        return jsonify({'error': 'Failed to get jobs'}), 500

@jobs_bp.route('/public', methods=['GET'])
def get_public_jobs():
    """Get all public jobs (no authentication required)"""
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 100, type=int)  # Increased default from 10 to 100
        status = request.args.get('status', 'active')
        search = request.args.get('search', '')
        location = request.args.get('location', '')
        company = request.args.get('company', '')
        employment_type = request.args.get('employment_type', '')
        experience_level = request.args.get('experience_level', '')
        
        # Build query for public jobs only
        query = Job.query.filter_by(is_public=True)
        
        if status == 'active':
            query = query.filter(Job.status == 'active')
        elif status != 'all':
            query = query.filter_by(status=status)
        
        if search:
            query = query.filter(
                db.or_(
                    Job.title.contains(search),
                    Job.description.contains(search),
                    Job.company_name.contains(search),
                    Job.location.contains(search)
                )
            )
        
        if location:
            query = query.filter(Job.location.contains(location))
        
        if company:
            query = query.filter(Job.company_name.contains(company))
        
        if employment_type:
            query = query.filter_by(employment_type=employment_type)
        
        if experience_level:
            query = query.filter_by(experience_level=experience_level)
        
        # Paginate results
        jobs = query.order_by(Job.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'jobs': [job.to_dict() for job in jobs.items],
            'total': jobs.total,
            'pages': jobs.pages,
            'current_page': page,
            'per_page': per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting public jobs: {str(e)}")
        return jsonify({'error': 'Failed to get public jobs'}), 500

@jobs_bp.route('/<int:job_id>', methods=['GET'])
def get_job(job_id):
    """Get a specific job (public endpoint)"""
    try:
        job = Job.query.get_or_404(job_id)
        
        # Check if job is public and active
        if not job.is_public or not job.is_active():
            return jsonify({'error': 'Job not found or not available'}), 404
        
        # Increment view count
        job.increment_views()
        
        return jsonify({
            'job': job.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to get job'}), 500

@jobs_bp.route('/<int:job_id>', methods=['PUT'])
def update_job(job_id):
    """Update a job posting"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        job = Job.query.get_or_404(job_id)
        
        # Check permissions
        if job.tenant_id != tenant_id:
            return jsonify({'error': 'Job not found'}), 404
        
        if user.role not in ['employer', 'recruiter', 'admin', 'owner'] and job.created_by != user.id:
            return jsonify({'error': 'Insufficient permissions to update this job'}), 403
        
        data = request.get_json()
        
        # Update job fields
        updatable_fields = [
            'title', 'description', 'location', 'company_name', 'employment_type',
            'experience_level', 'salary_min', 'salary_max', 'currency', 'remote_allowed',
            'benefits', 'requirements', 'responsibilities', 'status', 'is_public'
        ]
        
        for field in updatable_fields:
            if field in data:
                setattr(job, field, data[field])
        
        if 'skills_required' in data:
            job.skills_required = json.dumps(data['skills_required']) if data['skills_required'] else None
        
        if 'expires_in_days' in data:
            job.expires_at = datetime.utcnow() + timedelta(days=data['expires_in_days'])
        
        job.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        logger.info(f"Job updated: {job.id} by user {user.id}")
        
        return jsonify({
            'message': 'Job updated successfully',
            'job': job.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to update job'}), 500

@jobs_bp.route('/<int:job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job posting"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        job = Job.query.get_or_404(job_id)
        
        # Check permissions
        if job.tenant_id != tenant_id:
            return jsonify({'error': 'Job not found'}), 404
        
        if user.role not in ['employer', 'recruiter', 'admin', 'owner'] and job.created_by != user.id:
            return jsonify({'error': 'Insufficient permissions to delete this job'}), 403
        
        db.session.delete(job)
        db.session.commit()
        
        logger.info(f"Job deleted: {job_id} by user {user.id}")
        
        return jsonify({'message': 'Job deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to delete job'}), 500

@jobs_bp.route('/<int:job_id>/applications', methods=['GET'])
def get_job_applications(job_id):
    """Get applications for a specific job"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        job = Job.query.get_or_404(job_id)
        
        # Check permissions
        if job.tenant_id != tenant_id:
            return jsonify({'error': 'Job not found'}), 404
        
        if user.role not in ['employer', 'recruiter', 'admin', 'owner'] and job.created_by != user.id:
            return jsonify({'error': 'Insufficient permissions to view applications'}), 403
        
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 100, type=int)  # Increased default from 10 to 100
        status = request.args.get('status', 'all')
        
        # Build query
        query = JobApplication.query.filter_by(job_id=job_id)
        
        if status != 'all':
            query = query.filter_by(status=status)
        
        # Paginate results
        applications = query.order_by(JobApplication.applied_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'applications': [app.to_dict() for app in applications.items],
            'total': applications.total,
            'pages': applications.pages,
            'current_page': page,
            'per_page': per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting job applications for job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to get job applications'}), 500


@jobs_bp.route('/applications/my', methods=['GET'])
def get_my_applications():
    """Get current user's job applications"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 100, type=int)  # Increased default from 10 to 100
        status = request.args.get('status', 'all')
        
        # Build query with job relationship loaded
        query = JobApplication.query.options(joinedload(JobApplication.job)).filter_by(applicant_id=user.id)
        
        if status != 'all':
            query = query.filter_by(status=status)
        
        # Paginate results
        applications = query.order_by(JobApplication.applied_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'applications': [app.to_dict() for app in applications.items],
            'total': applications.total,
            'pages': applications.pages,
            'current_page': page,
            'per_page': per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting user applications: {str(e)}")
        return jsonify({'error': 'Failed to get applications'}), 500

@jobs_bp.route('/applications/<int:application_id>/status', methods=['PUT'])
def update_application_status(application_id):
    """Update application status (for employers)"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        application = JobApplication.query.get_or_404(application_id)
        job = application.job
        
        # Check permissions
        if job.tenant_id != tenant_id:
            return jsonify({'error': 'Application not found'}), 404
        
        if user.role not in ['employer', 'recruiter', 'admin', 'owner'] and job.created_by != user.id:
            return jsonify({'error': 'Insufficient permissions to update application status'}), 403
        
        data = request.get_json()
        
        # Update application
        if 'status' in data:
            application.status = data['status']
            if data['status'] in ['reviewed', 'shortlisted', 'rejected', 'hired']:
                application.reviewed_at = datetime.utcnow()
        
        if 'notes' in data:
            application.notes = data['notes']
        
        application.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # Send email notification to candidate if status changed to reviewed, shortlisted, rejected, or hired
        if 'status' in data and data['status'] in ['reviewed', 'shortlisted', 'rejected', 'hired']:
            try:
                from app.emails.ses import send_application_status_email
                
                # Get candidate email and name from application
                candidate_email = application.applicant.email if application.applicant else None
                
                # Get candidate name from candidate profile if available, otherwise use email or default
                candidate_name = 'Candidate'
                if application.applicant and application.applicant.candidate_profile:
                    candidate_name = application.applicant.candidate_profile.full_name
                elif application.applicant and application.applicant.email:
                    # Use email prefix as fallback
                    candidate_name = application.applicant.email.split('@')[0]
                
                logger.info(f"[EMAIL] Preparing to send status update email:")
                logger.info(f"   - Application ID: {application_id}")
                logger.info(f"   - Candidate Email: {candidate_email}")
                logger.info(f"   - Candidate Name: {candidate_name}")
                logger.info(f"   - Job Title: {job.title}")
                logger.info(f"   - Company: {job.company_name}")
                logger.info(f"   - New Status: {data['status']}")
                
                if candidate_email:
                    email_sent = send_application_status_email(
                        to_email=candidate_email,
                        candidate_name=candidate_name,
                        job_title=job.title,
                        company_name=job.company_name,
                        job_location=job.location,
                        applied_date=application.applied_at,
                        status=data['status']
                    )
                    
                    if email_sent:
                        logger.info(f"[SUCCESS] Status update email sent successfully to {candidate_email} for application {application_id}")
                    else:
                        logger.warning(f"[FAILED] Failed to send status update email to {candidate_email} for application {application_id}")
                else:
                    logger.warning(f"[SKIP] No candidate email found for application {application_id}, skipping email notification")
                    
            except Exception as email_error:
                logger.error(f"[ERROR] Error sending status update email: {str(email_error)}")
                import traceback
                logger.error(f"[TRACEBACK] Full traceback: {traceback.format_exc()}")
                # Don't fail the request if email sending fails
        
        logger.info(f"Application status updated: {application_id} to {application.status} by user {user.id}")
        
        return jsonify({
            'message': 'Application status updated successfully',
            'application': application.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating application status {application_id}: {str(e)}")
        return jsonify({'error': 'Failed to update application status'}), 500

@jobs_bp.route('/applications/<int:application_id>/schedule-interview', methods=['POST'])
def schedule_interview(application_id):
    """Schedule an interview for a job application"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        application = JobApplication.query.get_or_404(application_id)
        job = application.job
        
        # Check permissions
        if job.tenant_id != tenant_id:
            return jsonify({'error': 'Application not found'}), 404
        
        if user.role not in ['employer', 'recruiter', 'admin', 'owner'] and job.created_by != user.id:
            return jsonify({'error': 'Insufficient permissions to schedule interview'}), 403
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['interview_date', 'interview_time', 'meeting_link', 'meeting_type']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Validate meeting type
        if data['meeting_type'] not in ['zoom', 'google_meet', 'teams', 'other']:
            return jsonify({'error': 'Invalid meeting type. Must be zoom, google_meet, teams, or other'}), 400
        
        # Parse interview date and time
        try:
            from datetime import datetime
            interview_datetime = datetime.strptime(f"{data['interview_date']} {data['interview_time']}", "%Y-%m-%d %H:%M")
        except ValueError:
            return jsonify({'error': 'Invalid date or time format. Use YYYY-MM-DD for date and HH:MM for time'}), 400
        
        # Update application with interview details
        application.interview_scheduled = True
        application.interview_date = interview_datetime
        application.interview_meeting_link = data['meeting_link']
        application.interview_meeting_type = data['meeting_type']
        application.interview_notes = data.get('interview_notes', '')
        application.status = 'interview_scheduled'
        application.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # Send interview invitation email
        try:
            from app.emails.interview import send_interview_invitation_email
            
            # Get candidate details
            candidate_email = application.applicant.email if application.applicant else None
            candidate_name = 'Candidate'
            if application.applicant and application.applicant.candidate_profile:
                candidate_name = application.applicant.candidate_profile.full_name
            elif application.applicant and application.applicant.email:
                candidate_name = application.applicant.email.split('@')[0]
            
            logger.info(f"[EMAIL] Sending interview invitation email to {candidate_email}")
            
            email_sent = send_interview_invitation_email(
                to_email=candidate_email,
                candidate_name=candidate_name,
                job_title=job.title,
                company_name=job.company_name,
                job_location=job.location,
                interview_date=interview_datetime,
                meeting_link=data['meeting_link'],
                meeting_type=data['meeting_type'],
                interviewer_name=user.candidate_profile.full_name if user.candidate_profile else user.email.split('@')[0],
                interview_notes=data.get('interview_notes', '')
            )
            
            if email_sent:
                logger.info(f"[SUCCESS] Interview invitation email sent to {candidate_email}")
            else:
                logger.warning(f"[FAILED] Failed to send interview invitation email to {candidate_email}")
                
        except Exception as email_error:
            logger.error(f"[ERROR] Error sending interview invitation email: {str(email_error)}")
            # Don't fail the interview scheduling if email fails
        
        logger.info(f"Interview scheduled for application {application_id} by user {user.id}")
        
        return jsonify({
            'message': 'Interview scheduled successfully',
            'interview_details': {
                'interview_date': interview_datetime.strftime('%Y-%m-%d'),
                'interview_time': interview_datetime.strftime('%H:%M'),
                'meeting_link': data['meeting_link'],
                'meeting_type': data['meeting_type'],
                'interview_notes': data.get('interview_notes', '')
            }
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error scheduling interview for application {application_id}: {str(e)}")
        return jsonify({'error': 'Failed to schedule interview'}), 500

@jobs_bp.route('/<int:job_id>/apply', methods=['POST'])
def apply_for_job(job_id):
    """Apply for a job"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Check if user is a job seeker
        if user.role != 'job_seeker':
            return jsonify({'error': 'Only job seekers can apply for jobs'}), 403

        # Get job
        job = Job.query.get_or_404(job_id)
        
        # Check if job is active and public
        if not job.is_active() or not job.is_public:
            return jsonify({'error': 'This job is no longer available'}), 400

        # Check if user already applied
        existing_application = JobApplication.query.filter_by(
            job_id=job_id, 
            applicant_id=user.id
        ).first()
        
        if existing_application:
            return jsonify({'error': 'You have already applied for this job'}), 400

        # Get form data
        cover_letter = request.form.get('cover_letter', '')
        additional_info = request.form.get('additional_info', '')
        availability = request.form.get('availability', '')
        expected_salary = request.form.get('expected_salary', '')
        notice_period = request.form.get('notice_period', '')
        portfolio_url = request.form.get('portfolio_url', '')
        linkedin_url = request.form.get('linkedin_url', '')
        github_url = request.form.get('github_url', '')

        # Validate required fields
        if not cover_letter:
            return jsonify({'error': 'Cover letter is required'}), 400

        if not availability:
            return jsonify({'error': 'Availability is required'}), 400

        # Handle resume file upload
        resume_file = request.files.get('resume_file')
        resume_s3_key = None
        
        if resume_file and resume_file.filename:
            try:
                # Upload resume to S3 with consistent key prefix
                from io import BytesIO
                from app.talent.routes import s3_client, S3_BUCKET, RESUME_PREFIX
                from werkzeug.utils import secure_filename
                # Use human-readable name: <userId>_<jobId>_<originalName>
                original_name = secure_filename(resume_file.filename)
                resume_s3_key = f"{RESUME_PREFIX}{user.id}_{job_id}_{original_name}"

                # Ensure stream is at start and upload
                resume_file.stream.seek(0)
                file_bytes = resume_file.read()
                buffer = BytesIO(file_bytes)
                buffer.seek(0)

                s3_client.upload_fileobj(
                    buffer,
                    S3_BUCKET,
                    resume_s3_key,
                    ExtraArgs={
                        'ContentType': resume_file.content_type or 'application/octet-stream',
                        'ACL': 'private'
                    }
                )
                logger.info(f"Uploaded resume to s3://{S3_BUCKET}/{resume_s3_key}")
            except Exception as upload_error:
                logger.error(f"S3 upload failed for application resume: {str(upload_error)}")
                return jsonify({'error': 'Failed to upload resume. Please try again.'}), 500

        # Create application
        application = JobApplication(
            job_id=job_id,
            applicant_id=user.id,
            cover_letter=cover_letter,
            resume_s3_key=resume_s3_key,
            resume_filename=resume_file.filename if resume_file and resume_file.filename else None,
            additional_answers=json.dumps({
                'additional_info': additional_info,
                'availability': availability,
                'expected_salary': expected_salary,
                'notice_period': notice_period,
                'portfolio_url': portfolio_url,
                'linkedin_url': linkedin_url,
                'github_url': github_url
            }) if any([additional_info, availability, expected_salary, notice_period, portfolio_url, linkedin_url, github_url]) else None
        )

        db.session.add(application)
        
        # Increment job application count
        job.applications_count = (job.applications_count or 0) + 1
        
        db.session.commit()

        logger.info(f"User {user.id} applied for job {job_id}")

        # Send application confirmation email
        try:
            from app.emails.ses import send_application_confirmation_email
            
            # Get candidate name from profile if available
            candidate_name = 'Candidate'
            if user.candidate_profile and user.candidate_profile.full_name:
                candidate_name = user.candidate_profile.full_name
            elif user.email:
                # Use email prefix as fallback
                candidate_name = user.email.split('@')[0]
            
            logger.info(f"[EMAIL] Sending application confirmation email to {user.email}")
            
            email_sent = send_application_confirmation_email(
                to_email=user.email,
                candidate_name=candidate_name,
                job_title=job.title,
                company_name=job.company_name,
                job_location=job.location,
                applied_date=application.applied_at
            )
            
            if email_sent:
                logger.info(f"[SUCCESS] Application confirmation email sent to {user.email}")
            else:
                logger.warning(f"[FAILED] Failed to send application confirmation email to {user.email}")
                
        except Exception as email_error:
            logger.error(f"[ERROR] Error sending application confirmation email: {str(email_error)}")
            # Don't fail the application if email fails

        return jsonify({
            'message': 'Application submitted successfully',
            'application_id': application.id
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error applying for job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to submit application'}), 500

@jobs_bp.route('/public/<int:job_id>/check-application', methods=['POST'])
def check_public_application_status(job_id):
    """Check if a user has applied for a specific job (public route)"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get job
        job = Job.query.get_or_404(job_id)
        
        # Check if user already applied
        existing_application = JobApplication.query.filter_by(
            job_id=job_id, 
            applicant_id=user.id
        ).first()
        
        has_applied = existing_application is not None
        
        return jsonify({
            'has_applied': has_applied,
            'application_id': existing_application.id if existing_application else None,
            'application_status': existing_application.status if existing_application else None
        }), 200

    except Exception as e:
        logger.error(f"Error checking application status for job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to check application status'}), 500

@jobs_bp.route('/<int:job_id>/check-application', methods=['POST'])
def check_application_status(job_id):
    """Check if a user has applied for a specific job"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get job
        job = Job.query.get_or_404(job_id)
        
        # Check if user already applied
        existing_application = JobApplication.query.filter_by(
            job_id=job_id, 
            applicant_id=user.id
        ).first()
        
        has_applied = existing_application is not None
        
        return jsonify({
            'has_applied': has_applied,
            'application_id': existing_application.id if existing_application else None,
            'application_status': existing_application.status if existing_application else None
        }), 200

    except Exception as e:
        logger.error(f"Error checking application status for job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to check application status'}), 500

@jobs_bp.route('/applications/<int:application_id>/resume', methods=['GET'])
def download_application_resume(application_id):
    """Download resume for a job application"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get the application
        application = JobApplication.query.filter_by(id=application_id).first()
        if not application:
            return jsonify({'error': 'Application not found'}), 404

        # Check if user has permission to view this application
        # User must be the job creator, the applicant, or have admin/owner role
        job = Job.query.filter_by(id=application.job_id).first()
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        # Allow access if user is the applicant, job creator, or admin/owner
        is_applicant = application.applicant_id == user.id
        is_job_creator = job.created_by == user.id
        is_admin = user.role in ['admin', 'owner']
        
        if not (is_applicant or is_job_creator or is_admin):
            return jsonify({'error': 'Access denied'}), 403

        if not application.resume_s3_key:
            return jsonify({'error': 'Resume not available'}), 404

        # Generate download URL (CloudFront preferred, S3 fallback)
        try:
            import os
            import botocore
            download_url = None
            
            # Try CloudFront first if configured
            cf_domain = os.getenv('CLOUDFRONT_DOMAIN')
            if cf_domain:
                try:
                    from app.utils.cloudfront_utils import generate_resume_download_url
                    download_url = generate_resume_download_url(application.resume_s3_key, ttl_minutes=60)
                    logger.info(f"Generated CloudFront URL for resume {application.resume_s3_key}")
                except Exception as cf_error:
                    logger.warning(f"CloudFront URL generation failed, falling back to S3: {str(cf_error)}")
                    download_url = None
            
            # Fallback to S3 presigned URL if CloudFront failed or not configured
            if not download_url:
                from app.talent.routes import s3_client, S3_BUCKET, RESUME_PREFIX
                key_candidates = [application.resume_s3_key]
                # Fallbacks: try to reconstruct known patterns
                try:
                    # Strip any accidental 'resumes/' segment
                    if '/resumes/' in application.resume_s3_key or application.resume_s3_key.startswith('resumes/'):
                        key_candidates.append(application.resume_s3_key.replace('/resumes/', '/').replace('resumes/', RESUME_PREFIX))
                except Exception:
                    pass

                chosen_key = None
                for candidate_key in key_candidates:
                    try:
                        s3_client.head_object(Bucket=S3_BUCKET, Key=candidate_key)
                        chosen_key = candidate_key
                        break
                    except Exception:
                        continue

                if not chosen_key:
                    chosen_key = application.resume_s3_key

                download_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': S3_BUCKET, 'Key': chosen_key},
                    ExpiresIn=3600
                )
                logger.info(f"Generated S3 presigned URL for resume {chosen_key}")
            
            return jsonify({
                'download_url': download_url,
                'filename': application.resume_filename
            }), 200
            
        except Exception as e:
            logger.error(f"Failed to generate download URL for resume {application.resume_s3_key}: {str(e)}")
            return jsonify({'error': 'Failed to generate download URL'}), 500

    except Exception as e:
        logger.error(f"Error downloading resume for application {application_id}: {str(e)}")
        return jsonify({'error': 'Failed to download resume'}), 500
