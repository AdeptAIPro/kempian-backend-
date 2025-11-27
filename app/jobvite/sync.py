"""
Background sync jobs for Jobvite data.
Syncs data into dedicated Jobvite tables (jobvite_jobs, jobvite_candidates).
"""

from app.models import (
    JobviteSettings, JobviteJob, JobviteCandidate, JobviteCandidateDocument,
    JobviteOnboardingProcess, JobviteOnboardingTask, db
)
from app.jobvite.client_v2 import JobviteV2Client
from app.jobvite.client_onboarding import JobviteOnboardingClient
from app.jobvite.utils import get_base_urls
from app.jobvite.logging_utils import log_sync_operation
from app.simple_logger import get_logger
from datetime import datetime
from typing import Dict, Any, List
import time
import base64

logger = get_logger("jobvite_sync")

def _upsert_job_from_jobvite_data(tenant_id: int, job_data: Dict[str, Any], db_session=None):
    """
    Helper to upsert a job from Jobvite API response.
    
    Args:
        tenant_id: Tenant ID
        job_data: Job data from Jobvite API
        db_session: Database session (defaults to db.session)
    """
    if db_session is None:
        db_session = db.session
    
    jobvite_job_id = job_data.get('id') or job_data.get('jobId')
    if not jobvite_job_id:
        logger.warning(f"Skipping job with no ID: {job_data}")
        return None
    
    job = JobviteJob.query.filter_by(
        tenant_id=tenant_id,
        jobvite_job_id=str(jobvite_job_id)
    ).first()
    
    # Extract fields from job_data
    title = job_data.get('title', '')
    status = job_data.get('status')
    department = job_data.get('department')
    category = job_data.get('category')
    requisition_id = job_data.get('requisitionId')
    
    # Recruiters/HMs
    primary_recruiter = job_data.get('primaryRecruiter') or {}
    primary_hiring_manager = job_data.get('primaryHiringManager') or {}
    primary_recruiter_email = primary_recruiter.get('email')
    primary_hiring_manager_email = primary_hiring_manager.get('email')
    
    # Location
    locations = job_data.get('locations', [])
    location_main = locations[0].get('city', '') if locations else None
    region = job_data.get('region')
    subsidiary = job_data.get('subsidiary')
    
    # Salary
    salary = job_data.get('salary', {})
    salary_currency = salary.get('currency')
    salary_min = salary.get('min')
    salary_max = salary.get('max')
    salary_frequency = salary.get('frequency')
    
    # Remote
    remote_type = job_data.get('remoteType')
    
    if job:
        # Update existing
        job.title = title or job.title
        job.status = status or job.status
        job.department = department or job.department
        job.category = category or job.category
        job.requisition_id = requisition_id or job.requisition_id
        job.primary_recruiter_email = primary_recruiter_email or job.primary_recruiter_email
        job.primary_hiring_manager_email = primary_hiring_manager_email or job.primary_hiring_manager_email
        job.location_main = location_main or job.location_main
        job.region = region or job.region
        job.subsidiary = subsidiary or job.subsidiary
        job.salary_currency = salary_currency or job.salary_currency
        job.salary_min = salary_min or job.salary_min
        job.salary_max = salary_max or job.salary_max
        job.salary_frequency = salary_frequency or job.salary_frequency
        job.remote_type = remote_type or job.remote_type
        job.raw_json = job_data
        job.updated_at = datetime.utcnow()
    else:
        # Create new
        job = JobviteJob(
            tenant_id=tenant_id,
            jobvite_job_id=str(jobvite_job_id),
            requisition_id=requisition_id,
            title=title,
            status=status,
            department=department,
            category=category,
            primary_recruiter_email=primary_recruiter_email,
            primary_hiring_manager_email=primary_hiring_manager_email,
            location_main=location_main,
            region=region,
            subsidiary=subsidiary,
            salary_currency=salary_currency,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_frequency=salary_frequency,
            remote_type=remote_type,
            raw_json=job_data
        )
        db_session.add(job)
    
    return job

def _upsert_candidate_from_jobvite_data(tenant_id: int, candidate_data: Dict[str, Any], db_session=None):
    """
    Helper to upsert a candidate from Jobvite API response.
    
    Args:
        tenant_id: Tenant ID
        candidate_data: Candidate data from Jobvite API
        db_session: Database session (defaults to db.session)
    """
    if db_session is None:
        db_session = db.session
    
    jobvite_candidate_id = candidate_data.get('id') or candidate_data.get('candidateId')
    if not jobvite_candidate_id:
        logger.warning(f"Skipping candidate with no ID: {candidate_data}")
        return None
    
    candidate = JobviteCandidate.query.filter_by(
        tenant_id=tenant_id,
        jobvite_candidate_id=str(jobvite_candidate_id)
    ).first()
    
    # Extract fields
    email = candidate_data.get('email')
    first_name = candidate_data.get('firstName')
    last_name = candidate_data.get('lastName')
    workflow_state = candidate_data.get('workflowState')
    personal_data_processing_status = candidate_data.get('personalDataProcessingStatus')
    jobvite_application_id = candidate_data.get('applicationId')
    
    # Link to job if available
    job_id = None
    jobvite_job_id = None
    applications = candidate_data.get('applications', [])
    if applications:
        app = applications[0]
        jobvite_job_id = app.get('jobId')
        if jobvite_job_id:
            # Find the job in our DB
            job = JobviteJob.query.filter_by(
                tenant_id=tenant_id,
                jobvite_job_id=str(jobvite_job_id)
            ).first()
            if job:
                job_id = job.id
    
    if candidate:
        # Update existing
        candidate.email = email or candidate.email
        candidate.first_name = first_name or candidate.first_name
        candidate.last_name = last_name or candidate.last_name
        candidate.workflow_state = workflow_state or candidate.workflow_state
        candidate.personal_data_processing_status = personal_data_processing_status or candidate.personal_data_processing_status
        candidate.jobvite_application_id = jobvite_application_id or candidate.jobvite_application_id
        candidate.jobvite_job_id = jobvite_job_id or candidate.jobvite_job_id
        candidate.job_id = job_id or candidate.job_id
        candidate.raw_json = candidate_data
        candidate.updated_at = datetime.utcnow()
    else:
        # Create new
        candidate = JobviteCandidate(
            tenant_id=tenant_id,
            jobvite_candidate_id=str(jobvite_candidate_id),
            jobvite_application_id=jobvite_application_id,
            jobvite_job_id=jobvite_job_id,
            job_id=job_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            workflow_state=workflow_state,
            personal_data_processing_status=personal_data_processing_status,
            raw_json=candidate_data
        )
        db_session.add(candidate)
    
    # Handle documents if available
    artifacts = candidate_data.get('artifacts', [])
    for artifact in artifacts:
        doc_type = artifact.get('type', 'attachment')
        if doc_type not in ['resume', 'cover_letter', 'attachment']:
            doc_type = 'attachment'
        
        # Check if document already exists
        existing_doc = JobviteCandidateDocument.query.filter_by(
            candidate_id=candidate.id,
            doc_type=doc_type,
            filename=artifact.get('filename', 'unknown')
        ).first()
        
        if not existing_doc:
            # Decode base64 content if available
            content_base64 = artifact.get('content')
            storage_path = None
            external_url = None
            
            if content_base64:
                # Upload to S3
                from app.jobvite.storage import upload_document_from_base64
                s3_key, s3_url, public_url = upload_document_from_base64(
                    content_base64=content_base64,
                    filename=artifact.get('filename', 'unknown'),
                    tenant_id=tenant_id,
                    candidate_id=candidate.id,
                    doc_type=doc_type
                )
                
                if s3_key:
                    storage_path = s3_key
                    external_url = public_url
                else:
                    # Fallback: log error but still create document record
                    logger.warning(f"Failed to upload document to S3 for candidate {candidate.id}, filename: {artifact.get('filename')}")
            
            doc = JobviteCandidateDocument(
                candidate_id=candidate.id,
                doc_type=doc_type,
                filename=artifact.get('filename', 'unknown'),
                mime_type=artifact.get('mimeType'),
                storage_path=storage_path,
                external_url=external_url,
                size_bytes=artifact.get('size')
            )
            db_session.add(doc)
    
    return candidate

def sync_jobs_for_tenant(tenant_id: int) -> Dict[str, Any]:
    """
    Sync jobs for a tenant.
    
    Args:
        tenant_id: Tenant ID
    
    Returns:
        Dict with sync results (count, errors, etc.)
    """
    start_time = time.time()
    settings = JobviteSettings.query.filter_by(
        tenant_id=tenant_id,
        is_active=True
    ).first()
    
    if not settings or not settings.sync_config.get('syncJobs'):
        logger.info(f"Job sync skipped for tenant {tenant_id}: not configured")
        return {'success': False, 'reason': 'not_configured'}
    
    try:
        # Get client
        base_urls = get_base_urls(settings.environment)
        client = JobviteV2Client(
            api_key=settings.api_key,
            api_secret=settings.decrypt_secret(),
            company_id=settings.company_id,
            base_url=base_urls['v2']
        )
        
        # Get filters
        job_filters = settings.sync_config.get('jobFilters', {})
        
        # Fetch all jobs with pagination
        all_jobs = client.paginate_all(
            client.get_job,
            filters=job_filters
        )
        
        # Upsert to database
        synced_count = 0
        error_count = 0
        
        for job_data in all_jobs:
            try:
                _upsert_job_from_jobvite_data(tenant_id, job_data)
                synced_count += 1
                
                # Commit in batches to avoid long transactions
                if synced_count % 50 == 0:
                    db.session.commit()
                    time.sleep(0.1)  # Small delay to avoid rate limits
                    
            except Exception as e:
                logger.error(f"Error syncing job {job_data.get('id')}: {e}")
                error_count += 1
                continue
        
        # Final commit
        db.session.commit()
        
        # Update sync status
        settings.last_sync_at = datetime.utcnow()
        if error_count == 0:
            settings.last_sync_status = 'success'
            settings.last_error = None
        elif synced_count > 0:
            settings.last_sync_status = 'partial'
            settings.last_error = f"{error_count} errors during sync"
        else:
            settings.last_sync_status = 'failed'
            settings.last_error = f"All {error_count} jobs failed to sync"
        
        db.session.commit()
        
        duration_seconds = time.time() - start_time
        
        # Determine final status
        if error_count == 0:
            final_status = 'success'
        elif synced_count > 0:
            final_status = 'partial'
        else:
            final_status = 'failed'
        
        # Structured logging
        log_sync_operation(
            tenant_id=tenant_id,
            sync_type='jobs',
            status=final_status,
            synced_count=synced_count,
            error_count=error_count,
            duration_seconds=duration_seconds
        )
        
        logger.info(f"Job sync completed for tenant {tenant_id}: {synced_count} synced, {error_count} errors")
        
        return {
            'success': True,
            'synced_count': synced_count,
            'error_count': error_count
        }
        
    except Exception as e:
        duration_seconds = time.time() - start_time
        error_msg = str(e)
        settings.last_sync_status = 'failed'
        settings.last_error = error_msg
        db.session.commit()
        
        log_sync_operation(
            tenant_id=tenant_id,
            sync_type='jobs',
            status='failed',
            duration_seconds=duration_seconds,
            error=error_msg
        )
        
        logger.error(f"Job sync failed for tenant {tenant_id}: {error_msg}")
        return {'success': False, 'error': error_msg}

def sync_candidates_for_tenant(tenant_id: int) -> Dict[str, Any]:
    """
    Sync candidates for a tenant.
    Similar structure to sync_jobs_for_tenant.
    """
    start_time = time.time()
    settings = JobviteSettings.query.filter_by(
        tenant_id=tenant_id,
        is_active=True
    ).first()
    
    if not settings or not settings.sync_config.get('syncCandidates'):
        return {'success': False, 'reason': 'not_configured'}
    
    try:
        base_urls = get_base_urls(settings.environment)
        client = JobviteV2Client(
            api_key=settings.api_key,
            api_secret=settings.decrypt_secret(),
            company_id=settings.company_id,
            base_url=base_urls['v2']
        )
        
        candidate_filters = settings.sync_config.get('candidateFilters', {})
        
        # Fetch candidates
        all_candidates = client.paginate_all(
            client.get_candidate,
            filters=candidate_filters
        )
        
        synced_count = 0
        error_count = 0
        
        for candidate_data in all_candidates:
            try:
                _upsert_candidate_from_jobvite_data(tenant_id, candidate_data)
                synced_count += 1
                
                if synced_count % 50 == 0:
                    db.session.commit()
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error syncing candidate {candidate_data.get('id')}: {e}")
                error_count += 1
                continue
        
        db.session.commit()
        
        # Update sync status
        settings.last_sync_at = datetime.utcnow()
        if error_count == 0:
            settings.last_sync_status = 'success'
        elif synced_count > 0:
            settings.last_sync_status = 'partial'
        else:
            settings.last_sync_status = 'failed'
        
        settings.last_error = f"{error_count} errors" if error_count > 0 else None
        db.session.commit()
        
        duration_seconds = time.time() - start_time
        
        # Determine final status
        if error_count == 0:
            final_status = 'success'
        elif synced_count > 0:
            final_status = 'partial'
        else:
            final_status = 'failed'
        
        log_sync_operation(
            tenant_id=tenant_id,
            sync_type='candidates',
            status=final_status,
            synced_count=synced_count,
            error_count=error_count,
            duration_seconds=duration_seconds
        )
        
        return {
            'success': True,
            'synced_count': synced_count,
            'error_count': error_count
        }
        
    except Exception as e:
        duration_seconds = time.time() - start_time
        error_msg = str(e)
        settings.last_sync_status = 'failed'
        settings.last_error = error_msg
        db.session.commit()
        
        log_sync_operation(
            tenant_id=tenant_id,
            sync_type='candidates',
            status='failed',
            duration_seconds=duration_seconds,
            error=error_msg
        )
        
        logger.error(f"Candidate sync failed for tenant {tenant_id}: {error_msg}")
        return {'success': False, 'error': error_msg}

def sync_onboarding_for_tenant(tenant_id: int) -> Dict[str, Any]:
    """
    Sync onboarding data for a tenant.
    Uses OnboardingClient with encryption.
    
    Args:
        tenant_id: Tenant ID
    
    Returns:
        Dict with sync results
    """
    settings = JobviteSettings.query.filter_by(
        tenant_id=tenant_id,
        is_active=True
    ).first()
    
    if not settings or not settings.sync_config.get('syncOnboarding'):
        return {'success': False, 'reason': 'not_configured'}
    
    # Validate RSA keys are present
    if not settings.our_private_rsa_key_encrypted or not settings.jobvite_public_rsa_key:
        logger.warning(f"Onboarding sync skipped for tenant {tenant_id}: missing RSA keys")
        return {'success': False, 'reason': 'missing_rsa_keys'}
    
    start_time = time.time()
    try:
        base_urls = get_base_urls(settings.environment)
        
        # Use service account if available, otherwise use API key/secret
        service_account_username = settings.service_account_username
        service_account_password = None
        if settings.service_account_password_encrypted:
            service_account_password = settings.decrypt_service_account_password()
        
        client = JobviteOnboardingClient(
            api_key=settings.api_key,
            api_secret=settings.decrypt_secret(),
            base_url=base_urls['onboarding'],
            our_private_key_pem=settings.decrypt_private_key(),
            jobvite_public_key_pem=settings.jobvite_public_rsa_key,
            service_account_username=service_account_username,
            service_account_password=service_account_password
        )
        
        # Get filters
        onboarding_filters = settings.sync_config.get('onboardingFilters', {})
        
        # Fetch all processes with pagination
        all_processes = []
        start = 0
        count = 50
        
        while True:
            result = client.get_processes(
                filters=onboarding_filters,
                start=start,
                count=count
            )
            
            processes = result.get('processes', [])
            total = result.get('total', 0)
            
            all_processes.extend(processes)
            
            if len(processes) < count or total == 0:
                break
            
            start += count
            time.sleep(0.1)  # Rate limit protection
        
        # Upsert processes
        synced_count = 0
        error_count = 0
        
        for process_data in all_processes:
            try:
                jobvite_process_id = process_data.get('processId') or process_data.get('id')
                if not jobvite_process_id:
                    continue
                
                process = JobviteOnboardingProcess.query.filter_by(
                    tenant_id=tenant_id,
                    jobvite_process_id=str(jobvite_process_id)
                ).first()
                
                # Extract new hire info
                new_hire = process_data.get('newHire', {})
                jobvite_new_hire_id = new_hire.get('id') or new_hire.get('newHireId')
                jobvite_candidate_id = process_data.get('candidateId')
                
                # Extract dates
                hire_date = process_data.get('hireDate')
                kickoff_date = process_data.get('kickoffDate')
                if hire_date:
                    try:
                        hire_date = datetime.fromisoformat(hire_date.replace('Z', '+00:00')).date()
                    except:
                        hire_date = None
                if kickoff_date:
                    try:
                        kickoff_date = datetime.fromisoformat(kickoff_date.replace('Z', '+00:00')).date()
                    except:
                        kickoff_date = None
                
                if process:
                    # Update existing
                    process.status = process_data.get('status', process.status)
                    process.hire_date = hire_date or process.hire_date
                    process.kickoff_date = kickoff_date or process.kickoff_date
                    process.milestone_status_json = process_data.get('milestones', process.milestone_status_json)
                    process.jobvite_new_hire_id = jobvite_new_hire_id or process.jobvite_new_hire_id
                    process.jobvite_candidate_id = jobvite_candidate_id or process.jobvite_candidate_id
                    process.raw_json = process_data
                    process.updated_at = datetime.utcnow()
                else:
                    # Create new
                    process = JobviteOnboardingProcess(
                        tenant_id=tenant_id,
                        jobvite_process_id=str(jobvite_process_id),
                        jobvite_new_hire_id=jobvite_new_hire_id,
                        jobvite_candidate_id=jobvite_candidate_id,
                        status=process_data.get('status'),
                        hire_date=hire_date,
                        kickoff_date=kickoff_date,
                        milestone_status_json=process_data.get('milestones'),
                        raw_json=process_data
                    )
                    db.session.add(process)
                
                synced_count += 1
                
                # Optionally fetch tasks for this process
                if settings.sync_config.get('syncTasks', True):
                    try:
                        tasks_result = client.get_tasks(
                            filters={'processId': jobvite_process_id},
                            return_file_info=False
                        )
                        
                        tasks = tasks_result.get('tasks', [])
                        for task_data in tasks:
                            jobvite_task_id = task_data.get('taskId') or task_data.get('id')
                            if not jobvite_task_id:
                                continue
                            
                            task = JobviteOnboardingTask.query.filter_by(
                                process_id=process.id,
                                jobvite_task_id=str(jobvite_task_id)
                            ).first()
                            
                            # Extract task type
                            task_type = task_data.get('type', 'CUSTOM')
                            if task_type not in ['W4', 'I9', 'DOC', 'FORM', 'CUSTOM']:
                                task_type = 'CUSTOM'
                            
                            # Extract due date
                            due_date = task_data.get('dueDate')
                            if due_date:
                                try:
                                    due_date = datetime.fromisoformat(due_date.replace('Z', '+00:00')).date()
                                except:
                                    due_date = None
                            
                            if task:
                                task.name = task_data.get('name', task.name)
                                task.type = task_type or task.type
                                task.status = task_data.get('status', task.status)
                                task.due_date = due_date or task.due_date
                                task.raw_json = task_data
                                task.updated_at = datetime.utcnow()
                            else:
                                task = JobviteOnboardingTask(
                                    process_id=process.id,
                                    jobvite_task_id=str(jobvite_task_id),
                                    name=task_data.get('name'),
                                    type=task_type,
                                    status=task_data.get('status'),
                                    due_date=due_date,
                                    raw_json=task_data
                                )
                                db.session.add(task)
                    except Exception as e:
                        logger.warning(f"Error syncing tasks for process {jobvite_process_id}: {e}")
                
                if synced_count % 50 == 0:
                    db.session.commit()
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error syncing process {process_data.get('id')}: {e}")
                error_count += 1
                continue
        
        db.session.commit()
        
        # Update sync status
        settings.last_sync_at = datetime.utcnow()
        if error_count == 0:
            settings.last_sync_status = 'success'
        elif synced_count > 0:
            settings.last_sync_status = 'partial'
        else:
            settings.last_sync_status = 'failed'
        
        settings.last_error = f"{error_count} errors" if error_count > 0 else None
        db.session.commit()
        
        return {
            'success': True,
            'synced_count': synced_count,
            'error_count': error_count
        }
        
    except Exception as e:
        duration_seconds = time.time() - start_time
        error_msg = str(e)
        logger.error(f"Onboarding sync failed for tenant {tenant_id}: {error_msg}")
        settings.last_sync_status = 'failed'
        settings.last_error = error_msg
        db.session.commit()
        
        log_sync_operation(
            tenant_id=tenant_id,
            sync_type='onboarding',
            status='failed',
            duration_seconds=duration_seconds,
            error=error_msg
        )
        
        return {'success': False, 'error': error_msg}

def run_sync_for_all_tenants():
    """
    Run sync for all tenants with active Jobvite settings.
    Called by scheduled job (Celery/cron).
    """
    all_settings = JobviteSettings.query.filter_by(is_active=True).all()
    
    results = []
    for settings in all_settings:
        tenant_id = settings.tenant_id
        
        if settings.sync_config.get('syncJobs'):
            results.append(('jobs', tenant_id, sync_jobs_for_tenant(tenant_id)))
        
        if settings.sync_config.get('syncCandidates'):
            results.append(('candidates', tenant_id, sync_candidates_for_tenant(tenant_id)))
        
        if settings.sync_config.get('syncOnboarding'):
            results.append(('onboarding', tenant_id, sync_onboarding_for_tenant(tenant_id)))
    
    return results

