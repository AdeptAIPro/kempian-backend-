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
    
    # JobVite API returns requisitionId as the primary identifier
    # Also check for id/jobId as fallback for different API versions
    requisition_id = job_data.get('requisitionId')
    jobvite_job_id = job_data.get('id') or job_data.get('jobId') or requisition_id
    
    if not jobvite_job_id and not requisition_id:
        logger.warning(f"Skipping job with no ID or requisitionId: {job_data.get('title', 'Unknown')}")
        return None
    
    # Use requisitionId as primary identifier if available, otherwise use jobvite_job_id
    lookup_id = requisition_id or jobvite_job_id
    
    job = JobviteJob.query.filter_by(
        tenant_id=tenant_id,
        jobvite_job_id=str(lookup_id)
    ).first()
    
    # Extract fields from job_data
    title = job_data.get('title', '')
    # JobVite API uses 'jobState' not 'status' for requisitions endpoint
    status = job_data.get('status') or job_data.get('jobState')
    department = job_data.get('department')
    category = job_data.get('category')
    
    # Recruiters/HMs
    primary_recruiter = job_data.get('primaryRecruiter') or {}
    primary_hiring_manager = job_data.get('primaryHiringManager') or {}
    primary_recruiter_email = primary_recruiter.get('email') or job_data.get('primaryRecruiterEmail')
    primary_hiring_manager_email = primary_hiring_manager.get('email') or job_data.get('primaryHiringManagerEmail')
    
    # Location - JobVite API uses 'jobLocations' (array) or 'location' (string) for requisitions
    # Also check locationCity, locationState, locationCountry for direct fields
    job_locations = job_data.get('jobLocations', [])
    location_string = job_data.get('location', '')
    location_city = job_data.get('locationCity', '')
    location_state = job_data.get('locationState', '')
    location_country = job_data.get('locationCountry', '')
    
    if job_locations and isinstance(job_locations, list) and len(job_locations) > 0:
        # If jobLocations is array, extract first location
        first_loc = job_locations[0]
        if isinstance(first_loc, dict):
            location_main = first_loc.get('city') or first_loc.get('location') or location_string
        else:
            location_main = str(first_loc) if first_loc else location_string
    else:
        # Build location string from components if available
        location_parts = [p for p in [location_city, location_state, location_country] if p]
        location_main = location_string or (', '.join(location_parts) if location_parts else '')
    
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
        # Update jobvite_job_id if we have requisitionId and it's different
        if requisition_id and str(job.jobvite_job_id) != str(requisition_id):
            job.jobvite_job_id = str(requisition_id)
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
        # Create new - use requisitionId as primary ID if available
        primary_id = requisition_id or jobvite_job_id
        job = JobviteJob(
            tenant_id=tenant_id,
            jobvite_job_id=str(primary_id),
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
    
    # Jobvite API v2 uses 'eId' as the candidate identifier
    # Also check for 'id' and 'candidateId' for backward compatibility
    jobvite_candidate_id = candidate_data.get('eId') or candidate_data.get('id') or candidate_data.get('candidateId')
    if not jobvite_candidate_id:
        logger.warning(f"Skipping candidate with no ID (checked eId, id, candidateId): {candidate_data}")
        return None
    
    candidate = JobviteCandidate.query.filter_by(
        tenant_id=tenant_id,
        jobvite_candidate_id=str(jobvite_candidate_id)
    ).first()
    
    # Extract application data first - check both 'application' (singular) and 'applications' (plural)
    # Jobvite API v2 uses 'application' (singular) object with 'eId' field
    application_data = candidate_data.get('application') or {}
    
    # Extract fields
    email = candidate_data.get('email')
    first_name = candidate_data.get('firstName')
    last_name = candidate_data.get('lastName')
    
    # Workflow state can be in application object (Jobvite API v2) or at top level
    workflow_state = candidate_data.get('workflowState')
    if not workflow_state and application_data and isinstance(application_data, dict):
        workflow_state = application_data.get('workflowState')
    
    personal_data_processing_status = candidate_data.get('personalDataProcessingStatus')
    
    # Extract application ID from application_data
    if isinstance(application_data, dict):
        jobvite_application_id = application_data.get('eId') or application_data.get('id') or application_data.get('applicationId')
    else:
        jobvite_application_id = candidate_data.get('applicationId')
    
    # If no application data found, try applications array
    if not jobvite_application_id:
        applications = candidate_data.get('applications', [])
        if applications and isinstance(applications, list) and len(applications) > 0:
            app = applications[0]
            if isinstance(app, dict):
                jobvite_application_id = app.get('eId') or app.get('id') or app.get('applicationId')
    
    # Link to job if available - check both 'application' and 'applications'
    job_id = None
    jobvite_job_id = None
    
    # Check application object first (Jobvite API v2 format)
    if application_data and isinstance(application_data, dict):
        job_data = application_data.get('job', {})
        if isinstance(job_data, dict):
            jobvite_job_id = job_data.get('eId') or job_data.get('id') or job_data.get('jobId') or job_data.get('requisitionId')
    
    # Fallback to applications array
    if not jobvite_job_id:
        applications = candidate_data.get('applications', [])
        if applications and isinstance(applications, list) and len(applications) > 0:
            app = applications[0]
            if isinstance(app, dict):
                job_data = app.get('job', {})
                if isinstance(job_data, dict):
                    jobvite_job_id = job_data.get('eId') or job_data.get('id') or job_data.get('jobId') or job_data.get('requisitionId')
                else:
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
        # Flush to ensure candidate.id is available for document handling
        db_session.flush()
    
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
                job_id = job_data.get('requisitionId') or job_data.get('id') or job_data.get('jobId') or 'unknown'
                logger.error(f"Error syncing job {job_id}: {e}")
                error_count += 1
                continue
        
        # Final commit
        db.session.commit()
        
        # Update sync status
        settings.last_sync_at = datetime.utcnow()
        if error_count == 0:
            settings.last_sync_status = 'success'
            settings.last_error = None
            logger.info(f"Job sync completed successfully: {synced_count} jobs synced for tenant {tenant_id}")
        elif synced_count > 0:
            settings.last_sync_status = 'partial'
            settings.last_error = f"{error_count} errors during sync"
            logger.warning(f"Job sync completed with errors: {synced_count} jobs synced, {error_count} errors for tenant {tenant_id}")
        else:
            settings.last_sync_status = 'failed'
            settings.last_error = f"All {error_count} jobs failed to sync"
            logger.error(f"Job sync failed: {error_count} errors, 0 jobs synced for tenant {tenant_id}")
        
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
    # Ensure we get fresh data from database
    db.session.expunge_all()
    settings = JobviteSettings.query.filter_by(
        tenant_id=tenant_id,
        is_active=True
    ).first()
    
    if not settings:
        error_msg = f"Candidate sync failed for tenant {tenant_id}: No Jobvite settings found"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg, 'reason': 'no_settings'}
    
    # Refresh to ensure we have the latest sync_config from database
    db.session.refresh(settings)
    
    # Ensure sync_config exists and is a dict
    if not settings.sync_config:
        settings.sync_config = {}
    
    if not settings.sync_config.get('syncCandidates'):
        error_msg = (
            f"Candidate sync skipped for tenant {tenant_id}: syncCandidates not enabled in config. "
            f"Please enable syncCandidates in the Jobvite integration settings."
        )
        logger.warning(error_msg)
        return {'success': False, 'error': error_msg, 'reason': 'not_configured'}
    
    try:
        base_urls = get_base_urls(settings.environment)
        client = JobviteV2Client(
            api_key=settings.api_key,
            api_secret=settings.decrypt_secret(),
            company_id=settings.company_id,
            base_url=base_urls['v2']
        )
        
        candidate_filters = settings.sync_config.get('candidateFilters', {})
        candidate_limit = settings.sync_config.get('candidateSyncLimit')
        
        # Convert limit to int if it's a string (JSON sometimes stores numbers as strings)
        if candidate_limit is not None:
            try:
                candidate_limit = int(candidate_limit)
                if candidate_limit <= 0:
                    candidate_limit = None
            except (ValueError, TypeError):
                candidate_limit = None
        
        # Apply default safety limit to prevent runaway pagination
        if candidate_limit is None:
            DEFAULT_SAFETY_LIMIT = 200
            candidate_limit = DEFAULT_SAFETY_LIMIT
            logger.info(f"No candidate sync limit set, applying default safety limit of {DEFAULT_SAFETY_LIMIT} candidates")
        
        # Ensure limit is always an integer and positive
        try:
            candidate_limit = int(candidate_limit)
            if candidate_limit <= 0:
                candidate_limit = 200
        except (ValueError, TypeError):
            candidate_limit = 200
        
        # Check how many candidates we already have in the database
        existing_count = JobviteCandidate.query.filter_by(tenant_id=tenant_id).count()
        
        # Get list of existing candidate IDs to track duplicates
        existing_candidate_ids = set(
            JobviteCandidate.query.filter_by(tenant_id=tenant_id)
            .with_entities(JobviteCandidate.jobvite_candidate_id)
            .all()
        )
        existing_candidate_ids = {str(cid[0]) for cid in existing_candidate_ids if cid[0]}
        
        logger.info(f"[SYNC] ========== CANDIDATE SYNC START ==========")
        logger.info(f"[SYNC] Tenant ID: {tenant_id}")
        logger.info(f"[SYNC] Existing candidates in database: {existing_count}")
        logger.info(f"[SYNC] Existing candidate IDs (sample): {list(existing_candidate_ids)[:10] if existing_candidate_ids else 'None'}")
        logger.info(f"[SYNC] Requested sync limit: {candidate_limit}")
        logger.info(f"[SYNC] Filters: {candidate_filters}")
        
        # For incremental sync: start from where we left off
        # NOTE: This assumes Jobvite API returns candidates in stable order
        # If order is not stable, some candidates may be updated instead of inserted as new
        start_position = existing_count
        total_to_fetch = candidate_limit
        
        logger.info(f"[SYNC] Incremental sync strategy:")
        logger.info(f"[SYNC]   - Starting position (start_offset): {start_position}")
        logger.info(f"[SYNC]   - Candidates to fetch in this sync: {total_to_fetch}")
        logger.info(f"[SYNC]   - Expected final count if all new: {existing_count + total_to_fetch}")
        logger.info(f"[SYNC]   - Will fetch candidates from position {start_position} to {start_position + total_to_fetch - 1}")
        logger.info(f"[SYNC]   - NOTE: If API order is not stable, some candidates may be updates instead of new")
        
        # Fetch candidates starting from where we left off
        try:
            logger.info(f"[SYNC] ========== STARTING API FETCH ==========")
            logger.info(f"[SYNC] Calling paginate_all with:")
            logger.info(f"[SYNC]   - start_offset: {start_position}")
            logger.info(f"[SYNC]   - limit: {total_to_fetch}")
            logger.info(f"[SYNC]   - filters: {candidate_filters}")
            logger.info(f"[SYNC] This will fetch candidates from Jobvite API starting at position {start_position}")
            
            all_candidates = client.paginate_all(
                client.get_candidate,
                filters=candidate_filters,
                limit=total_to_fetch,
                start_offset=start_position  # Pass start position for incremental sync
            )
            
            logger.info(f"[SYNC] ========== FETCH COMPLETE ==========")
            logger.info(f"[SYNC] API returned {len(all_candidates)} candidates")
            logger.info(f"[SYNC] Expected: {total_to_fetch}, Actual: {len(all_candidates)}")
            if len(all_candidates) < total_to_fetch:
                logger.info(f"[SYNC] NOTE: Fetched fewer candidates than requested. This may mean:")
                logger.info(f"[SYNC]   - No more candidates available beyond position {start_position + len(all_candidates)}")
                logger.info(f"[SYNC]   - Jobvite API has fewer total candidates than expected")
            elif len(all_candidates) == total_to_fetch:
                logger.info(f"[SYNC] SUCCESS: Fetched exactly {total_to_fetch} candidates as requested")
            
            if len(all_candidates) == 0:
                logger.warning(
                    f"[SYNC] No candidates found in JobVite API for tenant {tenant_id}. "
                    f"This may indicate: 1) No new candidates exist beyond position {start_position}, "
                    f"2) Filters are too restrictive, 3) API key lacks candidate permissions. "
                    f"Filters used: {candidate_filters}"
                )
                logger.info(f"[SYNC] Current database count: {existing_count}, No new candidates to sync")
        except ValueError as auth_error:
            # Check if it's a permissions issue
            error_str = str(auth_error)
            if '401' in error_str or 'Unauthorized' in error_str:
                logger.error(
                    f"Candidate sync failed for tenant {tenant_id}: API KEY PERMISSIONS ISSUE\n"
                    f"Your JobVite API key does not have permissions to access the /candidate endpoint.\n"
                    f"The API key works for jobs but not for candidates.\n"
                    f"Solution: Contact JobVite support to enable candidate endpoint access for your API key.\n"
                    f"API Key: {settings.api_key[:10]}...\n"
                    f"Company ID: {settings.company_id}\n"
                    f"Error: {error_str[:500]}"
                )
            raise
        
        # Process candidates: upsert (insert new or update existing)
        synced_count = 0
        new_count = 0
        updated_count = 0
        error_count = 0
        
        logger.info(f"[SYNC] ========== PROCESSING CANDIDATES ==========")
        logger.info(f"[SYNC] Total candidates to process: {len(all_candidates)}")
        
        for idx, candidate_data in enumerate(all_candidates, 1):
            try:
                # Extract candidate ID for logging
                candidate_id = (
                    candidate_data.get('eId') or 
                    candidate_data.get('id') or 
                    candidate_data.get('candidateId') or 
                    'unknown'
                )
                
                # Debug: Log first few candidates being processed
                if idx <= 3 or idx == len(all_candidates):
                    logger.info(f"[SYNC] Processing candidate {idx}/{len(all_candidates)}: ID={candidate_id}")
                
                # Check if candidate already exists before upsert
                existing_candidate = JobviteCandidate.query.filter_by(
                    tenant_id=tenant_id,
                    jobvite_candidate_id=str(candidate_id)
                ).first()
                
                is_new = existing_candidate is None
                
                # Track if this candidate ID was in our existing set
                was_in_existing_set = str(candidate_id) in existing_candidate_ids
                
                # Log first 5 and last 5 candidates for debugging
                if idx <= 5 or idx > len(all_candidates) - 5:
                    logger.info(f"[SYNC] Processing candidate {idx}/{len(all_candidates)}: ID={candidate_id}")
                    logger.info(f"[SYNC]   - Status: {'NEW' if is_new else 'EXISTING (will update)'}")
                    logger.info(f"[SYNC]   - Was in existing set: {was_in_existing_set}")
                    if existing_candidate:
                        logger.info(f"[SYNC]   - Existing candidate found in DB with ID: {existing_candidate.id}")
                    if not is_new and not was_in_existing_set:
                        logger.warning(f"[SYNC]   - WARNING: Candidate exists in DB but wasn't in initial set (possible race condition)")
                    if is_new and was_in_existing_set:
                        logger.warning(f"[SYNC]   - WARNING: Candidate marked as new but was in existing set (possible query issue)")
                
                # Upsert the candidate
                candidate = _upsert_candidate_from_jobvite_data(tenant_id, candidate_data)
                
                if candidate:
                    synced_count += 1
                    if is_new:
                        new_count += 1
                    else:
                        updated_count += 1
                    
                    # Log progress every 50 candidates
                    if synced_count % 50 == 0:
                        db.session.commit()
                        logger.info(
                            f"[SYNC] Progress: {synced_count}/{len(all_candidates)} candidates processed "
                            f"(New: {new_count}, Updated: {updated_count}, Errors: {error_count})"
                        )
                        time.sleep(0.1)
                    elif idx == len(all_candidates):
                        # Log final candidate
                        logger.info(
                            f"[SYNC] Processing final candidate {idx}/{len(all_candidates)}: "
                            f"ID={candidate_id}, New={is_new}"
                        )
                    
            except Exception as e:
                # Extract candidate ID for error logging - check all possible ID fields
                candidate_id = (
                    candidate_data.get('eId') or 
                    candidate_data.get('id') or 
                    candidate_data.get('candidateId') or 
                    'unknown'
                )
                logger.error(f"[SYNC ERROR] Failed to sync candidate {candidate_id} (position {idx}): {e}")
                import traceback
                logger.error(f"[SYNC ERROR] Traceback: {traceback.format_exc()}")
                error_count += 1
                continue
        
        # Final commit
        db.session.commit()
        
        # Get final count after sync
        final_count = JobviteCandidate.query.filter_by(tenant_id=tenant_id).count()
        
        # Calculate expected vs actual
        expected_final = existing_count + new_count
        actual_change = final_count - existing_count
        
        logger.info(f"[SYNC] ========== SYNC COMPLETE ==========")
        logger.info(f"[SYNC] Processing summary:")
        logger.info(f"[SYNC]   - Total processed: {synced_count}")
        logger.info(f"[SYNC]   - New candidates: {new_count}")
        logger.info(f"[SYNC]   - Updated candidates: {updated_count}")
        logger.info(f"[SYNC]   - Errors: {error_count}")
        logger.info(f"[SYNC] Database counts:")
        logger.info(f"[SYNC]   - Before sync: {existing_count}")
        logger.info(f"[SYNC]   - After sync: {final_count}")
        logger.info(f"[SYNC]   - Net change: {actual_change}")
        logger.info(f"[SYNC]   - Expected final (existing + new): {expected_final}")
        logger.info(f"[SYNC]   - Actual final: {final_count}")
        if final_count != expected_final:
            logger.warning(f"[SYNC] WARNING: Count mismatch! Expected {expected_final} but got {final_count}")
            logger.warning(f"[SYNC] This could indicate candidates were deleted or there's a database issue")
        else:
            logger.info(f"[SYNC] SUCCESS: Count matches expected value")
        
        # Update sync status
        settings.last_sync_at = datetime.utcnow()
        if error_count == 0:
            settings.last_sync_status = 'success'
            logger.info(f"[SYNC] Candidate sync completed successfully: {synced_count} candidates synced for tenant {tenant_id}")
        elif synced_count > 0:
            settings.last_sync_status = 'partial'
            logger.warning(f"[SYNC] Candidate sync completed with errors: {synced_count} candidates synced, {error_count} errors for tenant {tenant_id}")
        else:
            settings.last_sync_status = 'failed'
            logger.error(f"[SYNC] Candidate sync failed: {error_count} errors, 0 candidates synced for tenant {tenant_id}")
        
        settings.last_error = f"{error_count} errors" if error_count > 0 else None
        db.session.commit()
        
        duration_seconds = time.time() - start_time
        
        logger.info(f"[SYNC] Sync duration: {duration_seconds:.2f} seconds")
        logger.info(f"[SYNC] ========================================")
        
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
            'new_count': new_count,
            'updated_count': updated_count,
            'error_count': error_count,
            'existing_count': existing_count,
            'final_count': final_count
        }
        
    except ValueError as e:
        # Authentication/permissions error - most common issue
        duration_seconds = time.time() - start_time
        error_str = str(e)
        
        # Check if it's a permissions issue
        if '401' in error_str or 'Unauthorized' in error_str or 'permissions' in error_str.lower():
            error_msg = (
                f"[ERROR] CANDIDATE SYNC FAILED: API KEY PERMISSIONS ISSUE\n"
                f"{'=' * 70}\n"
                f"Your JobVite API key does NOT have permissions to access candidates.\n"
                f"The API key works for jobs (/job endpoint) but NOT for candidates (/candidate endpoint).\n\n"
                f"SOLUTION:\n"
                f"1. Contact JobVite support to enable candidate endpoint access for your API key\n"
                f"2. Request permissions for: /api/v2/candidate endpoint\n"
                f"3. Verify your API key has 'Candidate Read' permissions enabled\n\n"
                f"Details:\n"
                f"   Tenant ID: {tenant_id}\n"
                f"   API Key: {settings.api_key[:10]}...{settings.api_key[-4:] if len(settings.api_key) > 14 else ''}\n"
                f"   Company ID: {settings.company_id}\n"
                f"   Environment: {settings.environment}\n"
                f"   Error: {error_str[:400]}"
            )
            logger.error(error_msg)
        else:
            error_msg = f"Candidate sync failed (ValueError): {error_str}"
            logger.error(f"Candidate sync failed for tenant {tenant_id}: {error_msg}")
        
        settings.last_sync_status = 'failed'
        settings.last_error = error_msg[:500]  # Truncate for database
        db.session.commit()
        
        log_sync_operation(
            tenant_id=tenant_id,
            sync_type='candidates',
            status='failed',
            duration_seconds=duration_seconds,
            error=error_msg[:500]
        )
        
        return {'success': False, 'error': error_msg, 'reason': 'api_permissions' if 'permissions' in error_msg.lower() else 'api_error'}
    except Exception as e:
        duration_seconds = time.time() - start_time
        error_msg = str(e)
        
        logger.error(
            f"[ERROR] CANDIDATE SYNC FAILED: Unexpected error\n"
            f"{'=' * 70}\n"
            f"Tenant ID: {tenant_id}\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Message: {error_msg}\n"
            f"Duration: {duration_seconds:.2f}s"
        )
        
        settings.last_sync_status = 'failed'
        settings.last_error = error_msg[:500]
        db.session.commit()
        
        log_sync_operation(
            tenant_id=tenant_id,
            sync_type='candidates',
            status='failed',
            duration_seconds=duration_seconds,
            error=error_msg[:500]
        )
        
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

