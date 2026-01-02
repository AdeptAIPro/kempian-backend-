from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import json
from app.models import Job, JobApplication
from app.models import User, Tenant
from app import db
from sqlalchemy.orm import joinedload
from app.search.routes import get_user_from_jwt, get_jwt_payload
from app.utils import get_current_tenant_id, require_tenant_context
from app.simple_logger import get_logger

logger = get_logger(__name__)

jobs_bp = Blueprint('jobs', __name__, url_prefix='/jobs')

@jobs_bp.route('/', methods=['POST'])
@require_tenant_context
def create_job():
    """Create a new job posting"""
    try:
        # Get tenant context from middleware (already validated by @require_tenant_context)
        from flask import g
        tenant_id = g.tenant_id
        user_email = g.user.get('email') if g.user else None
        
        logger.info(f"Job creation - tenant_id={tenant_id}, user_email={user_email}")
        
        if not user_email:
            logger.error("No user email in JWT payload")
            return jsonify({'error': 'User not found'}), 404
            
        # Get user from database
        user = User.query.filter_by(email=user_email, tenant_id=tenant_id).first()
        logger.info(f"Job creation - user found: {user is not None}")
        if not user:
            logger.error(f"User not found in database: email={user_email}, tenant_id={tenant_id}")
            return jsonify({'error': 'User not found'}), 404
        
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
        
        # Get user from JWT token to check permissions
        payload = get_jwt_payload()
        user = None
        tenant_id = None
        if payload:
            user, tenant_id = get_user_from_jwt(payload)
        
        # Check if user has permission to view this job
        can_view = False
        
        if user:
            # User is the job creator, admin, or owner
            if (job.created_by == user.id or 
                user.role in ['admin', 'owner'] or
                (job.tenant_id == tenant_id and user.role in ['employer', 'recruiter'])):
                can_view = True
        
        # If not authorized user, check if job is public and active
        if not can_view:
            if not job.is_public or not job.is_active():
                return jsonify({'error': 'Job not found or not available'}), 404
        
        # Increment view count only for public access
        if not can_view:
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
        from flask import g
        
        # Try to use tenant context from middleware first
        tenant_id = getattr(g, 'tenant_id', None)
        user_payload = getattr(g, 'user', None)
        
        # If tenant context is not available, try to extract from JWT
        if not tenant_id or not user_payload:
            logger.debug(f"PUT /jobs/{job_id}: Tenant context not available, extracting from JWT")
            payload = get_jwt_payload()
            if not payload:
                logger.error(f"PUT /jobs/{job_id}: No JWT payload found")
                return jsonify({'error': 'Unauthorized'}), 401
            
            user, tenant_id = get_user_from_jwt(payload)
            if not user:
                email = payload.get('email', 'unknown')
                logger.error(f"PUT /jobs/{job_id}: User not found for email={email}, tenant_id={tenant_id}")
                return jsonify({'error': 'User not found. Please ensure you are logged in with a valid account.'}), 404
        else:
            # Use tenant context from middleware, but get User object from database
            email = user_payload.get('email')
            if not email:
                logger.error(f"PUT /jobs/{job_id}: No email in user payload")
                return jsonify({'error': 'Invalid user information'}), 401
            
            user = User.query.filter_by(email=email).first()
            if not user:
                logger.error(f"PUT /jobs/{job_id}: User not found in database for email={email}")
                return jsonify({'error': 'User not found. Please ensure you are logged in with a valid account.'}), 404
            
            # Ensure tenant_id matches user's actual tenant_id
            if user.tenant_id != tenant_id:
                logger.warning(f"PUT /jobs/{job_id}: Tenant mismatch - user.tenant_id={user.tenant_id}, g.tenant_id={tenant_id}")
                tenant_id = user.tenant_id
        
        job = Job.query.get_or_404(job_id)
        
        # Check permissions
        # Admins and owners can update any job regardless of tenant
        if user.role in ['admin', 'owner']:
            # Admin/owner can update any job - skip tenant check
            pass
        else:
            # For non-admin users, check tenant match
            if job.tenant_id != tenant_id:
                logger.warning(f"PUT /jobs/{job_id}: Tenant mismatch - job.tenant_id={job.tenant_id}, user.tenant_id={tenant_id}, user.id={user.id}")
                return jsonify({'error': f'Job not found or access denied. Job belongs to tenant {job.tenant_id}, but your account belongs to tenant {tenant_id}.'}), 404
            
            # Employers and recruiters can only update their own jobs
            if user.role in ['employer', 'recruiter']:
                if job.created_by != user.id:
                    return jsonify({'error': 'Insufficient permissions to update this job'}), 403
            else:
                # Other roles cannot update jobs
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
        from flask import g
        
        # Try to use tenant context from middleware first
        tenant_id = getattr(g, 'tenant_id', None)
        user_payload = getattr(g, 'user', None)
        
        # If tenant context is not available, try to extract from JWT
        if not tenant_id or not user_payload:
            logger.debug(f"DELETE /jobs/{job_id}: Tenant context not available, extracting from JWT")
            payload = get_jwt_payload()
            if not payload:
                logger.error(f"DELETE /jobs/{job_id}: No JWT payload found")
                return jsonify({'error': 'Unauthorized'}), 401
            
            user, tenant_id = get_user_from_jwt(payload)
            if not user:
                email = payload.get('email', 'unknown')
                logger.error(f"DELETE /jobs/{job_id}: User not found for email={email}, tenant_id={tenant_id}")
                return jsonify({'error': 'User not found. Please ensure you are logged in with a valid account.'}), 404
        else:
            # Use tenant context from middleware, but get User object from database
            email = user_payload.get('email')
            if not email:
                logger.error(f"DELETE /jobs/{job_id}: No email in user payload")
                return jsonify({'error': 'Invalid user information'}), 401
            
            user = User.query.filter_by(email=email).first()
            if not user:
                logger.error(f"DELETE /jobs/{job_id}: User not found in database for email={email}")
                return jsonify({'error': 'User not found. Please ensure you are logged in with a valid account.'}), 404
            
            # Ensure tenant_id matches user's actual tenant_id
            if user.tenant_id != tenant_id:
                logger.warning(f"DELETE /jobs/{job_id}: Tenant mismatch - user.tenant_id={user.tenant_id}, g.tenant_id={tenant_id}")
                tenant_id = user.tenant_id
        
        job = Job.query.get_or_404(job_id)
        
        # Check permissions
        # Admins and owners can delete any job regardless of tenant
        if user.role in ['admin', 'owner']:
            # Admin/owner can delete any job - skip tenant check
            pass
        else:
            # For non-admin users, check tenant match
            if job.tenant_id != tenant_id:
                logger.warning(f"DELETE /jobs/{job_id}: Tenant mismatch - job.tenant_id={job.tenant_id}, user.tenant_id={tenant_id}, user.id={user.id}")
                return jsonify({'error': f'Job not found or access denied. Job belongs to tenant {job.tenant_id}, but your account belongs to tenant {tenant_id}.'}), 404
            
            # Employers and recruiters can only delete their own jobs
            if user.role in ['employer', 'recruiter']:
                if job.created_by != user.id:
                    return jsonify({'error': 'Insufficient permissions to delete this job'}), 403
            else:
                # Other roles cannot delete jobs
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
        # For all statuses, only send email if explicitly requested (send_email=true)
        # Default to not sending email unless explicitly requested
        should_send_email = data.get('send_email', False)
        if 'status' in data and data['status'] in ['reviewed', 'shortlisted', 'rejected', 'hired']:
            
            if should_send_email:
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
                        # Get email subject based on status
                        status_subjects = {
                            'reviewed': f'Application Update: {job.title} at {job.company_name}',
                            'shortlisted': f'Congratulations! You\'ve been shortlisted for {job.title} at {job.company_name}',
                            'rejected': f'Application Update: {job.title} at {job.company_name}',
                            'hired': f'Congratulations! You\'re hired for {job.title} at {job.company_name}'
                        }
                        email_subject = status_subjects.get(data['status'], f'Application Update: {job.title} at {job.company_name}')
                        
                        # Create email body content
                        if isinstance(application.applied_at, str):
                            try:
                                applied_date_obj = datetime.fromisoformat(application.applied_at.replace('Z', '+00:00'))
                                formatted_date = applied_date_obj.strftime('%B %d, %Y')
                            except:
                                formatted_date = application.applied_at
                        else:
                            formatted_date = application.applied_at.strftime('%B %d, %Y') if application.applied_at else 'N/A'
                        
                        email_body = f"Dear {candidate_name},\n\n"
                        if data['status'] == 'shortlisted':
                            email_body += f"Congratulations! We are pleased to inform you that you have been shortlisted for the {job.title} position at {job.company_name}.\n\n"
                        elif data['status'] == 'hired':
                            email_body += f"Congratulations! We are thrilled to offer you the {job.title} position at {job.company_name}.\n\n"
                        elif data['status'] == 'reviewed':
                            email_body += f"Thank you for your interest in the {job.title} position at {job.company_name}. We have reviewed your application.\n\n"
                        elif data['status'] == 'rejected':
                            email_body += f"Thank you for your interest in the {job.title} position at {job.company_name}. After careful consideration, we have decided to move forward with other candidates.\n\n"
                        email_body += f"Job: {job.title}\n"
                        email_body += f"Company: {job.company_name}\n"
                        email_body += f"Location: {job.location}\n"
                        email_body += f"Applied Date: {formatted_date}\n"
                        
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
                            
                            # Save to communications history so it appears in OutreachPage
                            try:
                                from app.communications.service import send_candidate_message
                                
                                # Get candidate_id from application
                                candidate_id = str(application.applicant_id) if application.applicant_id else candidate_email
                                
                                # Save communication record
                                comm_result = send_candidate_message(
                                    user_id=user.id,
                                    candidate_id=candidate_id,
                                    candidate_name=candidate_name,
                                    candidate_email=candidate_email,
                                    candidate_phone=None,
                                    channel='email',
                                    template_id=None,
                                    custom_message=email_body,
                                    subject=email_subject,
                                    email_provider='smtp'  # Use the same provider that was used to send
                                )
                                
                                if comm_result.get('success'):
                                    logger.info(f"[COMMUNICATIONS] Email saved to communications history: {comm_result.get('communication_id')}")
                                else:
                                    logger.warning(f"[COMMUNICATIONS] Failed to save email to communications history: {comm_result.get('error')}")
                            except Exception as comm_error:
                                logger.warning(f"[COMMUNICATIONS] Error saving to communications history: {str(comm_error)}")
                                # Don't fail the request if saving to communications fails
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
        logger.error(f"Error scheduling interview for application {application_id}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to schedule interview: {str(e)}'}), 500

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

@jobs_bp.route('/recommended', methods=['GET'])
def get_recommended_jobs():
    """Get recommended jobs for the current user based on their profile"""
    try:
        # Get user from JWT token
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get user's candidate profile
        from app.models import CandidateProfile
        candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        
        if not candidate_profile:
            # If no profile, return empty results
            return jsonify({
                'jobs': [],
                'total': 0,
                'message': 'Please complete your profile to get job recommendations'
            }), 200

        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        # Get all active public jobs
        query = Job.query.filter_by(is_public=True, status='active')
        query = query.filter(
            db.or_(
                Job.expires_at.is_(None),
                Job.expires_at > datetime.utcnow()
            )
        )
        
        # Get all jobs
        all_jobs = query.all()
        
        # Calculate match scores for each job
        scored_jobs = []
        user_skills = [skill.skill_name.lower() for skill in candidate_profile.skills]
        user_experience_years = candidate_profile.experience_years or 0
        user_location = (candidate_profile.location or '').lower()
        
        for job in all_jobs:
            match_score = calculate_job_match_score(
                job, 
                user_skills, 
                user_experience_years, 
                user_location,
                candidate_profile
            )
            
            if match_score > 0:  # Only include jobs with some match
                job_dict = job.to_dict()
                job_dict['matchScore'] = match_score
                scored_jobs.append((match_score, job_dict))
        
        # Sort by match score (descending)
        scored_jobs.sort(key=lambda x: x[0], reverse=True)
        
        # Extract just the job dicts
        recommended_jobs = [job_dict for _, job_dict in scored_jobs]
        
        # Paginate
        total = len(recommended_jobs)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_jobs = recommended_jobs[start:end]
        
        return jsonify({
            'jobs': paginated_jobs,
            'total': total,
            'pages': (total + per_page - 1) // per_page,
            'current_page': page,
            'per_page': per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting recommended jobs: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get recommended jobs'}), 500

def get_recommended_jobs_for_user(user_id, limit=10):
    """
    Get recommended jobs for a user based on their candidate profile.
    This is a helper function that can be called after profile creation.
    
    Args:
        user_id: The user ID
        limit: Maximum number of jobs to return
    
    Returns:
        List of recommended job dictionaries with match scores
    """
    try:
        from app.models import CandidateProfile, User
        from datetime import datetime
        
        # Get user and candidate profile
        user = User.query.get(user_id)
        if not user:
            return []
        
        candidate_profile = CandidateProfile.query.filter_by(user_id=user_id).first()
        if not candidate_profile:
            return []
        
        # Get all active public jobs
        query = Job.query.filter_by(is_public=True, status='active')
        query = query.filter(
            db.or_(
                Job.expires_at.is_(None),
                Job.expires_at > datetime.utcnow()
            )
        )
        
        all_jobs = query.all()
        
        if not all_jobs:
            return []
        
        # Calculate match scores for each job
        scored_jobs = []
        user_skills = [skill.skill_name.lower() for skill in candidate_profile.skills]
        user_experience_years = candidate_profile.experience_years or 0
        user_location = (candidate_profile.location or '').lower()
        
        for job in all_jobs:
            match_score = calculate_job_match_score(
                job, 
                user_skills, 
                user_experience_years, 
                user_location,
                candidate_profile
            )
            
            if match_score > 0:  # Only include jobs with some match
                job_dict = job.to_dict()
                job_dict['matchScore'] = match_score
                scored_jobs.append((match_score, job_dict))
        
        # Sort by match score (descending)
        scored_jobs.sort(key=lambda x: x[0], reverse=True)
        
        # Extract just the job dicts and limit
        recommended_jobs = [job_dict for _, job_dict in scored_jobs[:limit]]
        
        return recommended_jobs
        
    except Exception as e:
        logger.error(f"Error getting recommended jobs for user {user_id}: {str(e)}", exc_info=True)
        return []

def calculate_job_match_score(job, user_skills, user_experience_years, user_location, candidate_profile):
    """
    Calculate a match score (0-100) for a job based on user profile
    
    Scoring breakdown:
    - Skills match: 40 points
    - Experience level match: 25 points
    - Location match: 15 points
    - Salary expectations: 10 points
    - Remote preference: 10 points
    """
    score = 0
    
    # 1. Skills match (40 points)
    job_skills = []
    if job.skills_required:
        try:
            if isinstance(job.skills_required, str):
                job_skills = json.loads(job.skills_required)
            else:
                job_skills = job.skills_required
        except:
            # If not JSON, try splitting by comma
            job_skills = [s.strip().lower() for s in str(job.skills_required).split(',')]
    
    if job_skills:
        job_skills_lower = [s.lower() if isinstance(s, str) else str(s).lower() for s in job_skills]
        matched_skills = sum(1 for skill in user_skills if any(js in skill or skill in js for js in job_skills_lower))
        if len(job_skills) > 0:
            skills_match_ratio = matched_skills / len(job_skills)
            score += min(40, skills_match_ratio * 40)
    else:
        # If job has no skills specified, give partial credit
        score += 20
    
    # 2. Experience level match (25 points)
    job_exp_level = (job.experience_level or '').lower()
    user_exp_years = user_experience_years or 0
    
    # Map experience levels to years
    exp_level_mapping = {
        'entry': (0, 2),
        'entry-level': (0, 2),
        'junior': (0, 3),
        'mid': (2, 5),
        'mid-level': (2, 5),
        'senior': (5, 10),
        'lead': (8, 15),
        'director': (10, 20),
        'executive': (15, 30)
    }
    
    if job_exp_level in exp_level_mapping:
        min_years, max_years = exp_level_mapping[job_exp_level]
        if min_years <= user_exp_years <= max_years:
            score += 25
        elif user_exp_years < min_years:
            # User has less experience, partial credit
            score += max(0, 25 * (user_exp_years / min_years))
        else:
            # User has more experience, still good match
            score += 20
    else:
        # Unknown experience level, give partial credit
        score += 12
    
    # 3. Location match (15 points)
    job_location = (job.location or '').lower()
    if user_location and job_location:
        # Check for city/state/country matches
        user_location_parts = user_location.split(',')
        job_location_parts = job_location.split(',')
        
        # Check if any part matches
        if any(part.strip() in job_location for part in user_location_parts):
            score += 15
        elif any(part.strip() in user_location for part in job_location_parts):
            score += 15
        # Check for remote jobs if user location is far
        elif job.remote_allowed:
            score += 10  # Partial credit for remote jobs
    elif job.remote_allowed:
        # Remote job, give full location credit
        score += 15
    
    # 4. Salary expectations (10 points)
    if candidate_profile.expected_salary:
        try:
            # Try to extract number from expected salary string
            import re
            expected_salary_match = re.search(r'[\d,]+', str(candidate_profile.expected_salary).replace(',', ''))
            if expected_salary_match:
                expected_salary = int(expected_salary_match.group())
                # Convert to thousands if needed (assume if > 1000, it's in actual dollars)
                if expected_salary > 1000:
                    expected_salary = expected_salary / 1000
                
                if job.salary_min and job.salary_max:
                    # Check if expected salary is within range
                    if job.salary_min <= expected_salary <= job.salary_max:
                        score += 10
                    elif expected_salary < job.salary_min:
                        # User expects less, still a match
                        score += 7
                    else:
                        # User expects more, partial match
                        ratio = job.salary_max / expected_salary
                        score += max(0, 10 * ratio)
                elif job.salary_min:
                    if expected_salary >= job.salary_min * 0.8:  # Within 20% of minimum
                        score += 8
        except:
            pass
    
    # 5. Remote preference (10 points)
    if job.remote_allowed:
        # Check if user prefers remote (could be inferred from location or explicitly set)
        # For now, give credit if job is remote
        score += 10
    
    # Normalize score to 0-100
    return min(100, int(score))

@jobs_bp.route('/<int:job_id>/suggested-candidates', methods=['GET'])
@require_tenant_context
def get_suggested_candidates(job_id):
    """Get suggested candidates for a specific job based on job requirements"""
    try:
        from flask import g
        
        # Get tenant context from middleware (already validated by @require_tenant_context)
        tenant_id = g.tenant_id
        user_payload = g.user
        
        # Get user from database
        user_email = user_payload.get('email') if user_payload else None
        if not user_email:
            logger.error("No user email in JWT payload")
            return jsonify({'error': 'User not found'}), 404
        
        user = User.query.filter_by(email=user_email, tenant_id=tenant_id).first()
        if not user:
            logger.error(f"User not found in database: email={user_email}, tenant_id={tenant_id}")
            return jsonify({'error': 'User not found'}), 404
        
        # Get the job
        job = Job.query.get_or_404(job_id)
        
        # Check permissions
        # Admins and owners can view suggested candidates for any job
        if user.role in ['admin', 'owner']:
            # Admin/owner can view any job - skip tenant check
            pass
        else:
            # For non-admin users, check tenant match
            if job.tenant_id != tenant_id:
                return jsonify({'error': 'Job not found or access denied'}), 404
            
            # Employers and recruiters can only view suggested candidates for their own jobs
            if user.role in ['employer', 'recruiter']:
                if job.created_by != user.id:
                    return jsonify({'error': 'Insufficient permissions to view suggested candidates for this job'}), 403
            else:
                # Other roles cannot view suggested candidates
                return jsonify({'error': 'Insufficient permissions to view suggested candidates'}), 403
        
        # Get query parameters
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        limit = 5  # Always return top 5 candidates
        
        # Import JobSuggestedCandidates model
        from app.models import JobSuggestedCandidates
        
        # Check for saved suggestions first (unless refresh is requested)
        if not refresh:
            saved_suggestions = JobSuggestedCandidates.query.filter_by(job_id=job_id).first()
            if saved_suggestions:
                logger.info(f"Returning saved suggested candidates for job {job_id}")
                candidates_data = json.loads(saved_suggestions.candidates_data) if saved_suggestions.candidates_data else []
                return jsonify({
                    'candidates': candidates_data[:limit],  # Ensure only top 5
                    'total_matched': len(candidates_data),
                    'job_id': job_id,
                    'job_title': job.title,
                    'algorithm_used': saved_suggestions.algorithm_used,
                    'generated_at': saved_suggestions.generated_at.isoformat() if saved_suggestions.generated_at else None,
                    'cached': True
                }), 200
        
        # If refresh requested or no saved suggestions, fetch new candidates
        logger.info(f"Fetching new suggested candidates for job {job_id} (refresh={refresh})")
        
        # Build comprehensive job description for semantic matching
        job_description_parts = []
        
        # Add job title
        if job.title:
            job_description_parts.append(f"Job Title: {job.title}")
        
        # Add company name
        if job.company_name:
            job_description_parts.append(f"Company: {job.company_name}")
        
        # Add job description
        if job.description:
            job_description_parts.append(f"Description: {job.description}")
        
        # Add requirements
        if job.requirements:
            job_description_parts.append(f"Requirements: {job.requirements}")
        
        # Add responsibilities
        if job.responsibilities:
            job_description_parts.append(f"Responsibilities: {job.responsibilities}")
        
        # Add skills required
        if job.skills_required:
            try:
                if isinstance(job.skills_required, str):
                    skills = json.loads(job.skills_required)
                else:
                    skills = job.skills_required
                if isinstance(skills, list):
                    job_description_parts.append(f"Required Skills: {', '.join(str(s) for s in skills)}")
                else:
                    job_description_parts.append(f"Required Skills: {job.skills_required}")
            except:
                job_description_parts.append(f"Required Skills: {job.skills_required}")
        
        # Add experience level
        if job.experience_level:
            job_description_parts.append(f"Experience Level: {job.experience_level}")
        
        # Add location
        if job.location:
            job_description_parts.append(f"Location: {job.location}")
        
        # Add employment type
        if job.employment_type:
            job_description_parts.append(f"Employment Type: {job.employment_type}")
        
        # Add remote information
        if job.remote_allowed:
            job_description_parts.append("Remote work allowed")
        
        # Combine into full job description
        full_job_description = "\n".join(job_description_parts)
        
        # Use semantic matching algorithm from service.py
        try:
            from app.search.service import semantic_match
            
            logger.info(f"Using semantic matching algorithm for job {job_id}: {job.title}")
            
            # Call semantic match with the job description (fetch more to ensure we have good top 5)
            result = semantic_match(full_job_description, top_k=20)  # Fetch 20, then take top 5
            
            if not result or not result.get('results'):
                logger.warning(f"Semantic match returned no results for job {job_id}")
                return jsonify({
                    'candidates': [],
                    'message': 'No matching candidates found',
                    'job_id': job_id,
                    'job_title': job.title
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
                                    import re
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
                    
                    # Count matching skills (if job has required skills)
                    matching_skills_count = 0
                    if job.skills_required:
                        try:
                            if isinstance(job.skills_required, str):
                                job_skills = json.loads(job.skills_required)
                            else:
                                job_skills = job.skills_required
                            if not isinstance(job_skills, list):
                                job_skills = [s.strip().lower() for s in str(job_skills).split(',')]
                            
                            job_skills_lower = [s.lower() if isinstance(s, str) else str(s).lower() for s in job_skills]
                            candidate_skills_lower = [s.lower() if isinstance(s, str) else str(s).lower() for s in candidate_skills]
                            
                            matching_skills_count = sum(1 for job_skill in job_skills_lower 
                                                       if any(job_skill in cand_skill or cand_skill in job_skill 
                                                             for cand_skill in candidate_skills_lower))
                        except:
                            pass
                    
                    # Format candidate data for frontend
                    formatted_candidate = {
                        'id': candidate_email or f"candidate_{len(formatted_candidates)}",
                        'full_name': candidate_name,
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
                        'experience': (
                            candidate.get('experience') or 
                            candidate.get('work_history') or 
                            candidate.get('designations_with_experience') or 
                            []
                        ),
                        'education': candidate.get('education') or candidate.get('Education') or [],
                        'certifications': candidate.get('certifications') or candidate.get('Certifications') or [],
                        'current_position': (
                            candidate.get('current_position') or 
                            candidate.get('title') or 
                            candidate.get('Title') or 
                            ''
                        ),
                        'match_score': match_score / 100.0 if match_score > 1 else match_score,  # Normalize to 0-1
                        'matching_skills_count': matching_skills_count
                    }
                    
                    formatted_candidates.append(formatted_candidate)
                    
                except Exception as e:
                    logger.warning(f"Error formatting candidate from semantic match: {str(e)}")
                    continue
            
            # Sort by match score and take only top 5
            formatted_candidates.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            top_candidates = formatted_candidates[:limit]
            
            logger.info(f"Found {len(top_candidates)} suggested candidates (top 5) for job {job_id} using semantic matching")
            
            # Save the top 5 candidates to database
            try:
                saved_suggestions = JobSuggestedCandidates.query.filter_by(job_id=job_id).first()
                if saved_suggestions:
                    # Update existing record
                    saved_suggestions.candidates_data = json.dumps(top_candidates)
                    saved_suggestions.algorithm_used = result.get('algorithm_used', 'Semantic Matching Algorithm')
                    saved_suggestions.updated_at = datetime.utcnow()
                else:
                    # Create new record
                    saved_suggestions = JobSuggestedCandidates(
                        job_id=job_id,
                        candidates_data=json.dumps(top_candidates),
                        algorithm_used=result.get('algorithm_used', 'Semantic Matching Algorithm')
                    )
                    db.session.add(saved_suggestions)
                
                db.session.commit()
                logger.info(f"Saved suggested candidates for job {job_id}")
            except Exception as save_error:
                logger.error(f"Error saving suggested candidates: {str(save_error)}")
                db.session.rollback()
                # Continue even if save fails
            
            return jsonify({
                'candidates': top_candidates,
                'total_matched': len(formatted_candidates),
                'job_id': job_id,
                'job_title': job.title,
                'algorithm_used': result.get('algorithm_used', 'Semantic Matching Algorithm'),
                'cached': False
            }), 200
            
        except ImportError as e:
            logger.error(f"Failed to import semantic_match from service: {str(e)}")
            return jsonify({
                'candidates': [],
                'error': 'Semantic matching algorithm not available',
                'message': 'Please ensure the search service is properly configured'
            }), 500
        except Exception as e:
            logger.error(f"Error using semantic matching algorithm: {str(e)}", exc_info=True)
            return jsonify({
                'candidates': [],
                'error': 'Failed to get suggested candidates using semantic matching',
                'message': str(e)
            }), 500
        
    except Exception as e:
        logger.error(f"Error getting suggested candidates for job {job_id}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get suggested candidates'}), 500

def calculate_candidate_match_score(
    job_skills, job_experience_level, job_location, job_remote_allowed,
    candidate_skills, candidate_experience_years, candidate_location
):
    """
    Calculate a match score (0-100) for a candidate based on job requirements
    
    Scoring breakdown:
    - Skills match: 40 points
    - Experience level match: 25 points
    - Location match: 15 points
    - Remote preference: 10 points
    - Additional factors: 10 points
    """
    score = 0
    
    # 1. Skills match (40 points)
    if job_skills and candidate_skills:
        matched_skills = sum(1 for job_skill in job_skills 
                            if any(job_skill in cand_skill or cand_skill in job_skill 
                                  for cand_skill in candidate_skills))
        if len(job_skills) > 0:
            skills_match_ratio = matched_skills / len(job_skills)
            score += min(40, skills_match_ratio * 40)
    elif not job_skills:
        # If job has no skills specified, give partial credit
        score += 20
    
    # 2. Experience level match (25 points)
    exp_level_mapping = {
        'entry': (0, 2),
        'entry-level': (0, 2),
        'junior': (0, 3),
        'mid': (2, 5),
        'mid-level': (2, 5),
        'senior': (5, 10),
        'lead': (8, 15),
        'director': (10, 20),
        'executive': (15, 30)
    }
    
    if job_experience_level in exp_level_mapping:
        min_years, max_years = exp_level_mapping[job_experience_level]
        if min_years <= candidate_experience_years <= max_years:
            score += 25
        elif candidate_experience_years < min_years:
            # Candidate has less experience, partial credit
            score += max(0, 25 * (candidate_experience_years / max(min_years, 1)))
        else:
            # Candidate has more experience, still good match
            score += 20
    else:
        # Unknown experience level, give partial credit
        score += 12
    
    # 3. Location match (15 points)
    if candidate_location and job_location:
        # Check for city/state/country matches
        candidate_location_parts = candidate_location.split(',')
        job_location_parts = job_location.split(',')
        
        # Check if any part matches
        if any(part.strip() in job_location for part in candidate_location_parts):
            score += 15
        elif any(part.strip() in candidate_location for part in job_location_parts):
            score += 15
        # Check for remote jobs
        elif job_remote_allowed:
            score += 10  # Partial credit for remote jobs
    elif job_remote_allowed:
        # Remote job, give full location credit
        score += 15
    
    # 4. Remote preference (10 points)
    if job_remote_allowed:
        score += 10
    
    # 5. Additional factors (10 points)
    # Give bonus for having skills even if not all match
    if candidate_skills and len(candidate_skills) > 0:
        score += min(10, len(candidate_skills) * 0.5)
    
    # Normalize score to 0-100
    return min(100, int(score))
