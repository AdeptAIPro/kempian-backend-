import os
import requests
from flask import current_app
from app.simple_logger import get_logger
from app.db import db
from sqlalchemy import text

logger = get_logger(__name__)

# LinkedIn OAuth Configuration
LINKEDIN_CLIENT_ID = os.getenv('LINKEDIN_CLIENT_ID')
LINKEDIN_CLIENT_SECRET = os.getenv('LINKEDIN_CLIENT_SECRET')
# Get redirect_uri from request if provided, otherwise use default
def get_linkedin_redirect_uri(request_data):
    return request_data.get('redirect_uri', os.getenv('LINKEDIN_REDIRECT_URI', 'http://localhost:8081/linkedin-callback'))

def connect_linkedin(request_data, user):
    """Exchange OAuth code for access token and store it"""
    try:
        code = request_data.get('code')
        # Accept both organizationId and tenantId (they are the same)
        organization_id = request_data.get('organizationId') or request_data.get('tenantId')
        
        if not code:
            return {'error': 'Authorization code required'}, 400
        
        if not organization_id:
            return {'error': 'Organization ID or Tenant ID required'}, 400
        
        logger.info(f"Processing LinkedIn OAuth for organization: {organization_id}")
        logger.info(f"Request data keys: {request_data.keys()}")
        logger.info(f"Request data redirect_uri: {request_data.get('redirect_uri')}")
        
        # Get redirect URI from request data
        redirect_uri = get_linkedin_redirect_uri(request_data)
        logger.info(f"Using LinkedIn redirect URI: {redirect_uri}")
        logger.info(f"Client ID: {LINKEDIN_CLIENT_ID}")
        logger.info(f"Code (first 20 chars): {code[:20] if code else 'None'}...")
        
        # Exchange code for access token
        token_response = requests.post(
            'https://www.linkedin.com/oauth/v2/accessToken',
            data={
                'grant_type': 'authorization_code',
                'code': code,
                'client_id': LINKEDIN_CLIENT_ID,
                'client_secret': LINKEDIN_CLIENT_SECRET,
                'redirect_uri': redirect_uri
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        
        logger.info(f"LinkedIn token response status: {token_response.status_code}")
        
        if not token_response.ok:
            logger.error(f"LinkedIn OAuth error: {token_response.text}")
            logger.error(f"REDIRECT URI USED: {redirect_uri}")
            logger.error(f"Client ID: {LINKEDIN_CLIENT_ID}")
            return {'error': f'Failed to exchange code for token: {token_response.text}'}, 400
        
        token_data = token_response.json()
        access_token = token_data.get('access_token')
        
        if not access_token:
            logger.error("No access token received from LinkedIn")
            return {'error': 'No access token received'}, 400
        
        # Store access token in database
        try:
            with db.engine.connect() as conn:
                # Insert or update LinkedIn integration
                conn.execute(text("""
                    INSERT INTO linkedin_integrations 
                    (organization_id, access_token, created_by) 
                    VALUES (:org_id, :token, :user_id)
                    ON DUPLICATE KEY UPDATE 
                    access_token = VALUES(access_token),
                    updated_at = CURRENT_TIMESTAMP
                """), {
                    'org_id': organization_id,
                    'token': access_token,
                    'user_id': user.get('id') if user else None
                })
                conn.commit()
                
            logger.info(f"LinkedIn connected successfully for organization {organization_id}")
            return {"success": True, "message": "LinkedIn connected successfully"}
            
        except Exception as db_error:
            logger.error(f"Database error storing LinkedIn token: {db_error}")
            return {'error': 'Failed to store authentication data'}, 500
        
    except Exception as error:
        logger.error(f"Error connecting LinkedIn: {error}")
        return {'error': str(error)}, 500


def get_linkedin_jobs(organization_id, user, tenant_id=None):
    """Fetch LinkedIn job postings for the organization"""
    try:
        # Use tenant_id if provided, otherwise use organization_id (they're the same)
        actual_org_id = tenant_id or organization_id
        logger.info(f"Fetching LinkedIn jobs for organization: {actual_org_id}")
        
        # Get access token from database
        with db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT access_token 
                FROM linkedin_integrations 
                WHERE organization_id = :org_id
            """), {'org_id': actual_org_id})
            
            integration = result.fetchone()
        
        if not integration:
            logger.warning(f"No LinkedIn integration found for organization: {organization_id}")
            return {'error': 'LinkedIn not connected'}, 404
        
        access_token = integration[0]  # access_token is the first column
        
        # Get organization ID from LinkedIn
        org_response = requests.get(
            'https://api.linkedin.com/v2/organizationalEntityAcls?q=roleAssignee',
            headers={
                'Authorization': f'Bearer {access_token}',
                'X-Restli-Protocol-Version': '2.0.0'
            }
        )
        
        if not org_response.ok:
            logger.error(f"Failed to fetch LinkedIn org: {org_response.text}")
            return {'error': 'Failed to fetch organization'}, 400
        
        org_data = org_response.json()
        linkedin_org_id = None
        
        if org_data.get('elements') and len(org_data['elements']) > 0:
            linkedin_org_id = org_data['elements'][0].get('organizationalTarget')
        
        if not linkedin_org_id:
            logger.info("No LinkedIn organization found")
            return {"jobs": []}
        
        # Fetch job postings
        jobs_response = requests.get(
            f'https://api.linkedin.com/v2/jobs?q=organization&organization={linkedin_org_id}',
            headers={
                'Authorization': f'Bearer {access_token}',
                'X-Restli-Protocol-Version': '2.0.0'
            }
        )
        
        if not jobs_response.ok:
            logger.error(f"Failed to fetch jobs: {jobs_response.text}")
            return {'error': 'Failed to fetch jobs'}, 400
        
        jobs_data = jobs_response.json()
        
        jobs = []
        for element in jobs_data.get('elements', []):
            jobs.append({
                'id': element.get('id'),
                'title': element.get('title'),
                'applicantCount': element.get('applicantCount', 0),
                'postedDate': element.get('listedAt')
            })
        
        logger.info(f"Found {len(jobs)} LinkedIn jobs for organization {actual_org_id}")
        return {"jobs": jobs}
        
    except Exception as error:
        logger.error(f"Error fetching LinkedIn jobs: {error}")
        return {'error': str(error)}, 500


def get_job_applicants(job_id, organization_id, user, tenant_id=None):
    """Fetch applicants for a specific job posting"""
    try:
        # Use tenant_id if provided, otherwise use organization_id (they're the same)
        actual_org_id = tenant_id or organization_id
        logger.info(f"Fetching applicants for job {job_id} in organization {actual_org_id}")
        
        # Get access token
        with db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT access_token 
                FROM linkedin_integrations 
                WHERE organization_id = :org_id
            """), {'org_id': actual_org_id})
            
            integration = result.fetchone()
        
        if not integration:
            logger.warning(f"No LinkedIn integration found for organization: {actual_org_id}")
            return {'error': 'LinkedIn not connected'}, 404
        
        access_token = integration[0]  # access_token is the first column
        
        # Fetch job applications
        applications_response = requests.get(
            f'https://api.linkedin.com/v2/jobApplications?q=job&job={job_id}',
            headers={
                'Authorization': f'Bearer {access_token}',
                'X-Restli-Protocol-Version': '2.0.0'
            }
        )
        
        if not applications_response.ok:
            logger.error(f"Failed to fetch applications: {applications_response.text}")
            return {'error': 'Failed to fetch applicants'}, 400
        
        applications_data = applications_response.json()
        
        applicants = []
        for element in applications_data.get('elements', []):
            candidate = element.get('candidate', {})
            profile = candidate.get('profile', {})
            
            # Extract candidate information
            applicant = {
                'firstName': profile.get('firstName', {}).get('localized', {}).get('en_US', ''),
                'lastName': profile.get('lastName', {}).get('localized', {}).get('en_US', ''),
                'email': candidate.get('emailAddress', ''),
                'phone': candidate.get('phoneNumber', ''),
                'headline': profile.get('headline', {}).get('localized', {}).get('en_US', ''),
                'location': profile.get('location', {}).get('name', ''),
                'profileUrl': f"https://www.linkedin.com/in/{profile.get('publicIdentifier', '')}",
                'skills': [],
                'experience': 0
            }
            
            # Fetch detailed profile for skills and experience
            if profile.get('id'):
                try:
                    profile_response = requests.get(
                        f'https://api.linkedin.com/v2/people/{profile.get("id")}?projection=(skills,positions)',
                        headers={
                            'Authorization': f'Bearer {access_token}',
                            'X-Restli-Protocol-Version': '2.0.0'
                        }
                    )
                    
                    if profile_response.ok:
                        detailed_profile = profile_response.json()
                        
                        # Extract skills
                        skills = detailed_profile.get('skills', {}).get('elements', [])
                        applicant['skills'] = [skill.get('name') for skill in skills[:10]]
                        
                        # Calculate experience
                        positions = detailed_profile.get('positions', {}).get('elements', [])
                        total_months = 0
                        for position in positions:
                            start = position.get('timePeriod', {}).get('startDate', {})
                            end = position.get('timePeriod', {}).get('endDate', {})
                            
                            if start:
                                start_year = start.get('year', 0)
                                start_month = start.get('month', 1)
                                end_year = end.get('year', 0) if end else 2024
                                end_month = end.get('month', 12) if end else 12
                                
                                months = (end_year - start_year) * 12 + (end_month - start_month)
                                total_months += max(0, months)
                        
                        applicant['experience'] = round(total_months / 12, 1)
                except Exception as profile_error:
                    logger.warning(f"Could not fetch detailed profile for candidate: {profile_error}")
            
            applicants.append(applicant)
        
        logger.info(f"Found {len(applicants)} applicants for job {job_id}")
        return {"applicants": applicants}
        
    except Exception as error:
        logger.error(f"Error fetching job applicants: {error}")
        return {'error': str(error)}, 500


def import_candidates_to_system(candidates_data, organization_id, user, tenant_id=None):
    """Import LinkedIn candidates to the system's candidate database"""
    try:
        # Use tenant_id if provided, otherwise use organization_id (they're the same)
        actual_org_id = tenant_id or organization_id
        logger.info(f"Importing {len(candidates_data)} candidates for organization {actual_org_id}")
        
        imported_count = 0
        with db.engine.connect() as conn:
            for candidate in candidates_data:
                try:
                    # Insert candidate into the candidates table
                    conn.execute(text("""
                        INSERT INTO candidates 
                        (organization_id, candidate_id, first_name, last_name, email, phone, 
                         current_location, experience_years, skills, linkedin_url, source, 
                         status, notes, created_at, updated_at)
                        VALUES (:org_id, :candidate_id, :first_name, :last_name, :email, :phone,
                                :location, :experience, :skills, :linkedin_url, :source,
                                :status, :notes, NOW(), NOW())
                        ON DUPLICATE KEY UPDATE
                        first_name = VALUES(first_name),
                        last_name = VALUES(last_name),
                        email = VALUES(email),
                        phone = VALUES(phone),
                        current_location = VALUES(current_location),
                        experience_years = VALUES(experience_years),
                        skills = VALUES(skills),
                        linkedin_url = VALUES(linkedin_url),
                        notes = VALUES(notes),
                        updated_at = NOW()
                    """), {
                        'org_id': actual_org_id,
                        'candidate_id': f"linkedin_{candidate.get('email', 'unknown')}",
                        'first_name': candidate.get('firstName', ''),
                        'last_name': candidate.get('lastName', ''),
                        'email': candidate.get('email', ''),
                        'phone': candidate.get('phone', ''),
                        'location': candidate.get('location', ''),
                        'experience': candidate.get('experience', 0),
                        'skills': str(candidate.get('skills', [])),
                        'linkedin_url': candidate.get('profileUrl', ''),
                        'source': 'linkedin',
                        'status': 'active',
                        'notes': f"Imported from LinkedIn - {candidate.get('headline', '')}"
                    })
                    imported_count += 1
                except Exception as insert_error:
                    logger.warning(f"Failed to import candidate {candidate.get('email', 'unknown')}: {insert_error}")
                    continue
            
            conn.commit()
        
        logger.info(f"Successfully imported {imported_count} candidates for organization {actual_org_id}")
        return {"success": True, "imported_count": imported_count, "message": f"Imported {imported_count} candidates successfully"}
        
    except Exception as error:
        logger.error(f"Error importing candidates: {error}")
        return {'error': str(error)}, 500


def disconnect_linkedin(organization_id, user, tenant_id=None):
    """Disconnect LinkedIn integration by removing access token from database"""
    try:
        # Use tenant_id if provided, otherwise use organization_id (they're the same)
        actual_org_id = tenant_id or organization_id
        logger.info(f"Disconnecting LinkedIn for organization: {actual_org_id}")
        
        with db.engine.connect() as conn:
            result = conn.execute(text("""
                DELETE FROM linkedin_integrations 
                WHERE organization_id = :org_id
            """), {'org_id': actual_org_id})
            
            conn.commit()
            
            if result.rowcount > 0:
                logger.info(f"Successfully disconnected LinkedIn for organization {actual_org_id}")
                return {"success": True, "message": "LinkedIn disconnected successfully"}, 200
            else:
                logger.info(f"No LinkedIn integration found for organization {actual_org_id}")
                return {"success": False, "message": "No LinkedIn connection found"}, 404
        
    except Exception as error:
        logger.error(f"Error disconnecting LinkedIn: {error}")
        return {'error': str(error)}, 500
