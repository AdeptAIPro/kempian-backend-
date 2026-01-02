from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import json
from sqlalchemy import or_
from app.models import Job, JobApplication
from app.models import User
from app import db
from app.simple_logger import get_logger

logger = get_logger(__name__)

public_jobs_bp = Blueprint('public_jobs', __name__)

@public_jobs_bp.route('/jobs', methods=['GET'])
def get_public_jobs():
    """Get all public jobs (no authentication required)"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 12, type=int)
        search = request.args.get('search', '')
        location = request.args.get('location', '')
        employment_type = request.args.get('employment_type', '')
        experience_level = request.args.get('experience_level', '')
        include_expired = request.args.get('include_expired', 'false').lower() in ('true', '1', 'yes')
        
        # Build query - only show active jobs
        query = Job.query.filter(
            Job.is_public == True,
            Job.status == 'active'
        )
        
        # If not including expired jobs, filter by expiration date
        if not include_expired:
            query = query.filter(
                or_(
                    Job.expires_at > datetime.utcnow(),
                    Job.expires_at == None
                )
            )
        
        # Apply filters
        if search:
            query = query.filter(
                Job.title.contains(search) | 
                Job.description.contains(search) |
                Job.company_name.contains(search)
            )
        
        if location:
            query = query.filter(Job.location.contains(location))
        
        if employment_type:
            query = query.filter(Job.employment_type == employment_type)
        
        if experience_level:
            query = query.filter(Job.experience_level == experience_level)
        
        # Order by created date (newest first)
        query = query.order_by(Job.created_at.desc())
        
        # Paginate
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        jobs = []
        for job in pagination.items:
            job_data = job.to_dict()
            # Add application count for public view
            job_data['application_count'] = JobApplication.query.filter_by(job_id=job.id).count()
            jobs.append(job_data)
        
        return jsonify({
            'jobs': jobs,
            'pagination': {
                'page': pagination.page,
                'pages': pagination.pages,
                'per_page': pagination.per_page,
                'total': pagination.total,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching public jobs: {str(e)}")
        return jsonify({'error': 'Failed to fetch jobs'}), 500

@public_jobs_bp.route('/jobs/<int:job_id>', methods=['GET'])
def get_public_job(job_id):
    """Get a job for public viewing (no authentication required)"""
    try:
        job = Job.query.get_or_404(job_id)
        
        # Check if job is public and active
        if not job.is_public or not job.is_active():
            return jsonify({'error': 'Job not found or not available'}), 404
        
        # Increment view count
        job.increment_views()
        
        # Parse skills if they exist
        skills_required = []
        if job.skills_required:
            try:
                skills_required = json.loads(job.skills_required)
            except (json.JSONDecodeError, TypeError):
                skills_required = []
        
        job_data = job.to_dict()
        job_data['skills_required'] = skills_required
        
        return jsonify({
            'job': job_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting public job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to get job'}), 500

@public_jobs_bp.route('/', methods=['GET'])
def get_public_jobs_list():
    """Get all public active jobs (for job board)"""
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '')
        location = request.args.get('location', '')
        employment_type = request.args.get('employment_type', '')
        experience_level = request.args.get('experience_level', '')
        include_expired = request.args.get('include_expired', 'false').lower() in ('true', '1', 'yes')
        
        # Build query for public active jobs
        query = Job.query.filter(
            Job.is_public == True,
            Job.status == 'active'
        )
        
        # If not including expired jobs, filter by expiration date
        if not include_expired:
            query = query.filter(
                or_(
                    Job.expires_at > datetime.utcnow(),
                    Job.expires_at == None
                )
            )
        
        # Apply filters
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
        
        if employment_type:
            query = query.filter(Job.employment_type == employment_type)
        
        if experience_level:
            query = query.filter(Job.experience_level == experience_level)
        
        # Paginate results
        jobs = query.order_by(Job.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        # Process jobs to include parsed skills
        jobs_data = []
        for job in jobs.items:
            job_dict = job.to_dict()
            if job.skills_required:
                try:
                    job_dict['skills_required'] = json.loads(job.skills_required)
                except (json.JSONDecodeError, TypeError):
                    job_dict['skills_required'] = []
            else:
                job_dict['skills_required'] = []
            jobs_data.append(job_dict)
        
        return jsonify({
            'jobs': jobs_data,
            'total': jobs.total,
            'pages': jobs.pages,
            'current_page': page,
            'per_page': per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting public jobs: {str(e)}")
        return jsonify({'error': 'Failed to get jobs'}), 500

@public_jobs_bp.route('/jobs/<int:job_id>/check-application', methods=['POST'])
def check_application_status(job_id):
    """Check if a user has already applied to a job (for logged-in users)"""
    try:
        # This endpoint requires authentication but is public
        # The frontend will handle the authentication check
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'has_applied': False}), 200
        
        # Check if user has applied
        application = JobApplication.query.filter_by(
            job_id=job_id, applicant_id=user_id
        ).first()
        
        return jsonify({
            'has_applied': application is not None,
            'application_id': application.id if application else None,
            'application_status': application.status if application else None
        }), 200
        
    except Exception as e:
        logger.error(f"Error checking application status for job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to check application status'}), 500
