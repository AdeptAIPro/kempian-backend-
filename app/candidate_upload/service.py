"""Service for uploading candidate data from CSV/JSON files to DynamoDB"""

import os
import csv
import json
import uuid
import boto3
from decimal import Decimal
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger("candidate_upload")

# AWS Configuration
REGION = os.getenv('AWS_REGION', 'ap-south-1')
TABLE_NAME = 'linkedin-table'

# Initialize DynamoDB
dynamodb = None
table = None
table_schema = None  # Store table schema info

def get_table_schema():
    """Get the table schema to understand primary key structure"""
    global table_schema
    if table_schema:
        return table_schema
    
    if not table:
        return None
    
    try:
        client = table.meta.client
        response = client.describe_table(TableName=TABLE_NAME)
        key_schema = response['Table']['KeySchema']
        
        # Extract primary key information
        partition_key = None
        sort_key = None
        
        for key in key_schema:
            if key['KeyType'] == 'HASH':
                partition_key = key['AttributeName']
            elif key['KeyType'] == 'RANGE':
                sort_key = key['AttributeName']
        
        table_schema = {
            'partition_key': partition_key,
            'sort_key': sort_key,
            'key_schema': key_schema
        }
        
        logger.info(f"Table schema for {TABLE_NAME}: Partition Key={partition_key}, Sort Key={sort_key}")
        return table_schema
    except Exception as e:
        logger.error(f"Failed to get table schema: {e}")
        return None

try:
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        table = dynamodb.Table(TABLE_NAME)
        logger.info(f"DynamoDB initialized successfully. Table: {TABLE_NAME}")
        # Get table schema on initialization
        get_table_schema()
    else:
        logger.warning("AWS credentials not found. DynamoDB upload will not be available.")
except Exception as e:
    logger.error(f"Failed to initialize DynamoDB: {e}")
    dynamodb = None
    table = None


def parse_csv_file(file_content: bytes) -> List[Dict[str, Any]]:
    """
    Parse CSV file content and return list of candidate dictionaries
    
    Args:
        file_content: Bytes content of the CSV file
        
    Returns:
        List of candidate dictionaries
    """
    try:
        # Decode the file content
        content = file_content.decode('utf-8-sig')  # Handle BOM
        lines = content.strip().split('\n')
        
        if not lines:
            raise ValueError("CSV file is empty")
        
        # Parse CSV
        reader = csv.DictReader(lines)
        candidates = []
        
        for row in reader:
            # Clean up the row (remove empty values, strip whitespace)
            cleaned_row = {k.strip(): v.strip() if v else '' for k, v in row.items() if k}
            
            # Skip empty rows
            if not any(cleaned_row.values()):
                continue
                
            candidates.append(cleaned_row)
        
        logger.info(f"Parsed {len(candidates)} candidates from CSV file")
        return candidates
        
    except Exception as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise ValueError(f"Failed to parse CSV file: {str(e)}")


def parse_json_file(file_content: bytes) -> List[Dict[str, Any]]:
    """
    Parse JSON file content and return list of candidate dictionaries
    
    Args:
        file_content: Bytes content of the JSON file
        
    Returns:
        List of candidate dictionaries
    """
    try:
        # Decode the file content
        content = file_content.decode('utf-8')
        
        # Remove control characters that are not allowed in JSON
        # JSON only allows: \n, \r, \t, and escaped characters
        import re
        # Remove control characters except newline, carriage return, and tab
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        # Parse JSON
        data = json.loads(content)
        
        # Handle both array and single object
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            # If it's a single object, wrap it in a list
            candidates = [data]
        else:
            raise ValueError("JSON must be an object or array of objects")
        
        # Remove unwanted fields from all candidates
        fields_to_exclude = ['peopleAlsoViewed']
        cleaned_candidates = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                cleaned_candidate = {k: v for k, v in candidate.items() if k not in fields_to_exclude}
                cleaned_candidates.append(cleaned_candidate)
            else:
                cleaned_candidates.append(candidate)
        
        logger.info(f"Parsed {len(cleaned_candidates)} candidates from JSON file (removed 'peopleAlsoViewed' field)")
        return cleaned_candidates
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        # Try to provide more helpful error message
        error_msg = str(e)
        if 'control character' in error_msg.lower():
            error_msg += ". The file contains invalid control characters. They have been removed, but the JSON may still be malformed."
        raise ValueError(f"Invalid JSON format: {error_msg}")
    except Exception as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise ValueError(f"Failed to parse JSON file: {str(e)}")


def normalize_candidate_data(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize candidate data to match DynamoDB schema
    
    Args:
        candidate: Raw candidate data from CSV/JSON
        
    Returns:
        Normalized candidate data for DynamoDB
    """
    # Remove fields that should not be uploaded
    fields_to_exclude = ['peopleAlsoViewed']
    candidate = {k: v for k, v in candidate.items() if k not in fields_to_exclude}
    
    # Common field mappings (handle various naming conventions)
    # Updated to match all fields from the LinkedIn export dataset
    field_mappings = {
        'linkedin_url': ['linkedinUrl', 'linkedin_url', 'LinkedInUrl', 'LINKEDIN_URL', 'linkedin', 'LinkedIn', 'linkedin_profile', 'LinkedInProfile', 'linkedinPublicUrl'],
        'first_name': ['firstName', 'first_name', 'FirstName', 'FIRST_NAME'],
        'last_name': ['lastName', 'last_name', 'LastName', 'LAST_NAME'],
        'full_name': ['fullName', 'full_name', 'FullName', 'FULL_NAME', 'name', 'Name', 'NAME'],
        'headline': ['headline', 'Headline', 'HEADLINE'],
        'connections': ['connections', 'Connections', 'CONNECTIONS'],
        'followers': ['followers', 'Followers', 'FOLLOWERS'],
        'email': ['email', 'Email', 'EMAIL', 'e-mail', 'E-mail'],
        'phone': ['mobileNumber', 'mobile_number', 'phone', 'Phone', 'PHONE', 'telephone', 'Telephone', 'mobile', 'Mobile'],
        'job_title': ['jobTitle', 'job_title', 'JobTitle', 'JOB_TITLE', 'title', 'Title', 'TITLE', 'current_title', 'CurrentTitle', 'position'],
        'job_started_on': ['jobStartedOn', 'job_started_on', 'JobStartedOn'],
        'job_location': ['jobLocation', 'job_location', 'JobLocation', 'JOB_LOCATION'],
        'job_still_working': ['jobStillWorking', 'job_still_working', 'JobStillWorking', 'JOB_STILL_WORKING'],
        'company_name': ['companyName', 'company_name', 'CompanyName', 'COMPANY_NAME'],
        'company_industry': ['companyIndustry', 'company_industry', 'CompanyIndustry'],
        'company_website': ['companyWebsite', 'company_website', 'CompanyWebsite'],
        'company_linkedin': ['companyLinkedin', 'company_linkedin', 'CompanyLinkedin'],
        'company_founded_in': ['companyFoundedIn', 'company_founded_in', 'CompanyFoundedIn'],
        'company_size': ['companySize', 'company_size', 'CompanySize'],
        'current_job_duration': ['currentJobDuration', 'current_job_duration', 'CurrentJobDuration'],
        'current_job_duration_yrs': ['currentJobDurationInYrs', 'current_job_duration_in_yrs', 'CurrentJobDurationInYrs'],
        'top_skills_by_endorsements': ['topSkillsByEndorsements', 'top_skills_by_endorsements', 'TopSkillsByEndorsements'],
        'address_country_only': ['addressCountryOnly', 'address_country_only', 'AddressCountryOnly'],
        'address_with_country': ['addressWithCountry', 'address_with_country', 'AddressWithCountry'],
        'address_without_country': ['addressWithoutCountry', 'address_without_country', 'AddressWithoutCountry'],
        'profile_pic': ['profilePic', 'profile_pic', 'ProfilePic', 'PROFILE_PIC'],
        'profile_pic_high_quality': ['profilePicHighQuality', 'profile_pic_high_quality', 'ProfilePicHighQuality'],
        'background_pic': ['backgroundPic', 'background_pic', 'BackgroundPic'],
        'linkedin_id': ['linkedinId', 'linkedin_id', 'LinkedInId'],
        'is_premium': ['isPremium', 'is_premium', 'IsPremium'],
        'is_verified': ['isVerified', 'is_verified', 'IsVerified'],
        'is_job_seeker': ['isJobSeeker', 'is_job_seeker', 'IsJobSeeker'],
        'is_retired': ['isRetired', 'is_retired', 'IsRetired'],
        'is_creator': ['isCreator', 'is_creator', 'IsCreator'],
        'is_influencer': ['isInfluencer', 'is_influencer', 'IsInfluencer'],
        'is_currently_employed': ['isCurrentlyEmployed', 'is_currently_employed', 'IsCurrentlyEmployed'],
        'about': ['about', 'About', 'ABOUT'],
        'public_identifier': ['publicIdentifier', 'public_identifier', 'PublicIdentifier'],
        'linkedin_public_url': ['linkedinPublicUrl', 'linkedin_public_url', 'LinkedInPublicUrl'],
        'open_connection': ['openConnection', 'open_connection', 'OpenConnection'],
        'urn': ['urn', 'URN'],
        'total_recommendations_received': ['totalRecommendationsReceived', 'total_recommendations_received', 'TotalRecommendationsReceived'],
        'total_recommendations_given': ['totalRecommendationsGiven', 'total_recommendations_given', 'TotalRecommendationsGiven'],
        'birthday': ['birthday', 'Birthday', 'BIRTHDAY'],
        'associated_hashtag': ['associatedHashtag', 'associated_hashtag', 'AssociatedHashtag'],
        'first_role_year': ['firstRoleYear', 'first_role_year', 'FirstRoleYear'],
        'total_experience_years': ['totalExperienceYears', 'total_experience_years', 'TotalExperienceYears'],
        'experiences_count': ['experiencesCount', 'experiences_count', 'ExperiencesCount'],
        'experiences': ['experiences', 'experience', 'Experiences', 'EXPERIENCES'],
        'updates': ['updates', 'Updates', 'UPDATES'],
        'skills': ['skills', 'Skills', 'SKILLS', 'skill', 'Skill', 'technical_skills', 'TechnicalSkills', 'topSkillsByEndorsements'],
        'creator_website': ['creatorWebsite', 'creator_website', 'CreatorWebsite'],
        'profile_pic_all_dimensions': ['profilePicAllDimensions', 'profile_pic_all_dimensions', 'ProfilePicAllDimensions'],
        'educations': ['educations', 'education', 'Education', 'EDUCATION', 'qualifications', 'qualification', 'Qualification'],
        'license_and_certificates': ['licenseAndCertificates', 'license_and_certificates', 'LicenseAndCertificates'],
        'honors_and_awards': ['honorsAndAwards', 'honors_and_awards', 'HonorsAndAwards'],
        'languages': ['languages', 'Languages', 'LANGUAGES', 'language', 'Language'],
        'volunteer_and_awards': ['volunteerAndAwards', 'volunteer_and_awards', 'VolunteerAndAwards'],
        'verifications': ['verifications', 'Verifications', 'VERIFICATIONS', 'verification', 'Verification'],
        'promos': ['promos', 'Promos', 'PROMOS'],
        'highlights': ['highlights', 'Highlights', 'HIGHLIGHTS', 'highlight', 'Highlight'],
        'projects': ['projects', 'Projects', 'PROJECTS', 'project', 'Project'],
        'publications': ['publications', 'Publications', 'PUBLICATIONS', 'publication', 'Publication'],
        'patents': ['patents', 'Patents', 'PATENTS', 'patent', 'Patent'],
        'courses': ['courses', 'Courses', 'COURSES', 'course', 'Course'],
        'test_scores': ['testScores', 'test_scores', 'TestScores'],
        'organizations': ['organizations', 'Organizations', 'ORGANIZATIONS', 'organization', 'Organization'],
        'volunteer_causes': ['volunteerCauses', 'volunteer_causes', 'VolunteerCauses'],
        'interests': ['interests', 'Interests', 'INTERESTS', 'interest', 'Interest'],
        'recommendations_received': ['recommendationsReceived', 'recommendations_received', 'RecommendationsReceived'],
        'recommendations': ['recommendations', 'Recommendations', 'RECOMMENDATIONS', 'recommendation', 'Recommendation'],
        'source': ['source', 'Source', 'SOURCE', 'source_url', 'SourceURL', 'sourceUrl', 'linkedinPublicUrl', 'linkedin_public_url']
    }
    
    normalized = {}
    
    # Helper function to find value by multiple possible keys
    def get_value(keys: List[str]) -> Optional[Any]:
        for key in keys:
            if key in candidate:
                value = candidate[key]
                if value is not None and value != '':
                    return value
        return None
    
    # Map email (keep as null if not present)
    email = get_value(field_mappings['email'])
    if email:
        normalized['email'] = email
    
    # Map full_name (keep as null if not present)
    full_name = get_value(field_mappings['full_name'])
    if not full_name:
        # Try to construct from first_name and last_name
        first_name = candidate.get('first_name') or candidate.get('firstName') or candidate.get('First Name') or ''
        last_name = candidate.get('last_name') or candidate.get('lastName') or candidate.get('Last Name') or ''
        if first_name or last_name:
            full_name = f"{first_name} {last_name}".strip()
    
    if full_name:
        normalized['fullName'] = full_name
    
    # Map headline
    headline = get_value(field_mappings['headline'])
    if headline:
        normalized['headline'] = str(headline)
    
    # Map connections
    connections = get_value(field_mappings['connections'])
    if connections is not None:
        try:
            normalized['connections'] = int(connections)
        except (ValueError, TypeError):
            pass
    
    # Map followers
    followers = get_value(field_mappings['followers'])
    if followers is not None:
        try:
            normalized['followers'] = int(followers)
        except (ValueError, TypeError):
            pass
    
    # Map phone
    phone = get_value(field_mappings['phone'])
    if phone:
        normalized['mobileNumber'] = str(phone)
    
    # Map job_title
    job_title = get_value(field_mappings['job_title'])
    if job_title:
        normalized['jobTitle'] = str(job_title)
    
    # Map job_started_on
    job_started_on = get_value(field_mappings['job_started_on'])
    if job_started_on:
        normalized['jobStartedOn'] = str(job_started_on)
    
    # Map job_location
    job_location = get_value(field_mappings['job_location'])
    if job_location:
        normalized['jobLocation'] = str(job_location)
    
    # Map job_still_working
    job_still_working = get_value(field_mappings['job_still_working'])
    if job_still_working is not None:
        normalized['jobStillWorking'] = bool(job_still_working)
    
    # Map company_name
    company_name = get_value(field_mappings['company_name'])
    if company_name:
        normalized['companyName'] = str(company_name)
    
    # Map company_industry
    company_industry = get_value(field_mappings['company_industry'])
    if company_industry:
        normalized['companyIndustry'] = str(company_industry)
    
    # Map company_website
    company_website = get_value(field_mappings['company_website'])
    if company_website:
        normalized['companyWebsite'] = str(company_website)
    
    # Map company_linkedin
    company_linkedin = get_value(field_mappings['company_linkedin'])
    if company_linkedin:
        normalized['companyLinkedin'] = str(company_linkedin)
    
    # Map company_founded_in
    company_founded_in = get_value(field_mappings['company_founded_in'])
    if company_founded_in:
        normalized['companyFoundedIn'] = str(company_founded_in)
    
    # Map company_size
    company_size = get_value(field_mappings['company_size'])
    if company_size:
        normalized['companySize'] = str(company_size)
    
    # Map current_job_duration
    current_job_duration = get_value(field_mappings['current_job_duration'])
    if current_job_duration:
        normalized['currentJobDuration'] = str(current_job_duration)
    
    # Map current_job_duration_yrs
    current_job_duration_yrs = get_value(field_mappings['current_job_duration_yrs'])
    if current_job_duration_yrs is not None:
        try:
            normalized['currentJobDurationInYrs'] = float(current_job_duration_yrs)
        except (ValueError, TypeError):
            pass
    
    # Map top_skills_by_endorsements
    top_skills_by_endorsements = get_value(field_mappings['top_skills_by_endorsements'])
    if top_skills_by_endorsements:
        if isinstance(top_skills_by_endorsements, list):
            normalized['topSkillsByEndorsements'] = top_skills_by_endorsements
        else:
            normalized['topSkillsByEndorsements'] = [str(top_skills_by_endorsements)]
    
    # Map address_country_only
    address_country_only = get_value(field_mappings['address_country_only'])
    if address_country_only:
        normalized['addressCountryOnly'] = str(address_country_only)
    
    # Map address_with_country (prefer addressWithCountry, fallback to addressWithoutCountry)
    address_with_country = None
    if 'addressWithCountry' in candidate and candidate.get('addressWithCountry'):
        address_with_country = candidate['addressWithCountry']
    elif 'address_with_country' in candidate and candidate.get('address_with_country'):
        address_with_country = candidate['address_with_country']
    elif 'addressWithoutCountry' in candidate and candidate.get('addressWithoutCountry'):
        address_with_country = candidate['addressWithoutCountry']
    elif 'address_without_country' in candidate and candidate.get('address_without_country'):
        address_with_country = candidate['address_without_country']
    else:
        address_with_country = get_value(field_mappings['address_with_country'])
    
    if address_with_country:
        normalized['addressWithCountry'] = str(address_with_country)
    
    # Map address_without_country
    address_without_country = get_value(field_mappings['address_without_country'])
    if address_without_country:
        normalized['addressWithoutCountry'] = str(address_without_country)
    
    # Map profile_pic
    profile_pic = get_value(field_mappings['profile_pic'])
    if profile_pic:
        normalized['profilePic'] = str(profile_pic)
    
    # Map profile_pic_high_quality
    profile_pic_high_quality = get_value(field_mappings['profile_pic_high_quality'])
    if profile_pic_high_quality:
        normalized['profilePicHighQuality'] = str(profile_pic_high_quality)
    
    # Map background_pic
    background_pic = get_value(field_mappings['background_pic'])
    if background_pic:
        normalized['backgroundPic'] = str(background_pic)
    
    # Map linkedin_id
    linkedin_id = get_value(field_mappings['linkedin_id'])
    if linkedin_id:
        normalized['linkedinId'] = str(linkedin_id)
    
    # Map is_premium
    is_premium = get_value(field_mappings['is_premium'])
    if is_premium is not None:
        normalized['isPremium'] = bool(is_premium)
    
    # Map is_verified
    is_verified = get_value(field_mappings['is_verified'])
    if is_verified is not None:
        normalized['isVerified'] = bool(is_verified)
    
    # Map is_job_seeker
    is_job_seeker = get_value(field_mappings['is_job_seeker'])
    if is_job_seeker is not None:
        normalized['isJobSeeker'] = bool(is_job_seeker)
    
    # Map is_retired
    is_retired = get_value(field_mappings['is_retired'])
    if is_retired is not None:
        normalized['isRetired'] = bool(is_retired)
    
    # Map is_creator
    is_creator = get_value(field_mappings['is_creator'])
    if is_creator is not None:
        normalized['isCreator'] = bool(is_creator)
    
    # Map is_influencer
    is_influencer = get_value(field_mappings['is_influencer'])
    if is_influencer is not None:
        normalized['isInfluencer'] = bool(is_influencer)
    
    # Map is_currently_employed
    is_currently_employed = get_value(field_mappings['is_currently_employed'])
    if is_currently_employed is not None:
        normalized['isCurrentlyEmployed'] = bool(is_currently_employed)
    
    # Map skills (convert to list if string or array of objects)
    skills = get_value(field_mappings['skills'])
    if skills:
        if isinstance(skills, str):
            # Handle comma-separated or semicolon-separated skills
            skills = [s.strip() for s in skills.replace(';', ',').split(',') if s.strip()]
        elif isinstance(skills, list):
            # Handle array of skill objects (like {"title": "JavaScript"}) or array of strings
            processed_skills = []
            for skill in skills:
                if isinstance(skill, dict):
                    # Extract title from skill object
                    skill_title = skill.get('title') or skill.get('name') or skill.get('skill')
                    if skill_title:
                        processed_skills.append(str(skill_title).strip())
                elif isinstance(skill, str):
                    processed_skills.append(skill.strip())
            skills = processed_skills if processed_skills else [str(s) for s in skills if s]
        else:
            skills = [str(skills)]
        normalized['skills'] = skills
    
    # Map updates (store as list)
    updates = get_value(field_mappings['updates'])
    if updates:
        if isinstance(updates, list):
            normalized['updates'] = updates if updates else []
        elif isinstance(updates, str):
            try:
                parsed = json.loads(updates)
                normalized['updates'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['updates'] = []
        else:
            normalized['updates'] = [updates]
    else:
        normalized['updates'] = []
    
    # Map creator_website (store as list)
    creator_website = get_value(field_mappings['creator_website'])
    if creator_website:
        if isinstance(creator_website, list):
            normalized['creatorWebsite'] = creator_website if creator_website else []
        elif isinstance(creator_website, str):
            try:
                parsed = json.loads(creator_website)
                normalized['creatorWebsite'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['creatorWebsite'] = []
        else:
            normalized['creatorWebsite'] = [creator_website]
    else:
        normalized['creatorWebsite'] = []
    
    # Map profile_pic_all_dimensions (store as list)
    profile_pic_all_dimensions = get_value(field_mappings['profile_pic_all_dimensions'])
    if profile_pic_all_dimensions:
        if isinstance(profile_pic_all_dimensions, list):
            normalized['profilePicAllDimensions'] = profile_pic_all_dimensions if profile_pic_all_dimensions else []
        elif isinstance(profile_pic_all_dimensions, str):
            try:
                parsed = json.loads(profile_pic_all_dimensions)
                normalized['profilePicAllDimensions'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['profilePicAllDimensions'] = []
        else:
            normalized['profilePicAllDimensions'] = [profile_pic_all_dimensions]
    else:
        normalized['profilePicAllDimensions'] = []
    
    # Map educations (store as list)
    educations = get_value(field_mappings['educations'])
    if educations:
        if isinstance(educations, list):
            normalized['educations'] = educations if educations else []
        elif isinstance(educations, str):
            # Try to parse JSON string
            try:
                parsed = json.loads(educations)
                normalized['educations'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['educations'] = []
        else:
            normalized['educations'] = [educations]
    else:
        normalized['educations'] = []
    
    # Map license_and_certificates (store as list)
    license_and_certificates = get_value(field_mappings['license_and_certificates'])
    if license_and_certificates:
        if isinstance(license_and_certificates, list):
            normalized['licenseAndCertificates'] = license_and_certificates if license_and_certificates else []
        elif isinstance(license_and_certificates, str):
            # Try to parse JSON string
            try:
                parsed = json.loads(license_and_certificates)
                normalized['licenseAndCertificates'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['licenseAndCertificates'] = []
        else:
            normalized['licenseAndCertificates'] = [license_and_certificates]
    else:
        normalized['licenseAndCertificates'] = []
    
    # Map honors_and_awards
    honors_and_awards = get_value(field_mappings['honors_and_awards'])
    if honors_and_awards:
        if isinstance(honors_and_awards, list):
            normalized['honorsAndAwards'] = json.dumps(honors_and_awards) if honors_and_awards else ''
        elif isinstance(honors_and_awards, str):
            normalized['honorsAndAwards'] = honors_and_awards
        else:
            normalized['honorsAndAwards'] = json.dumps(honors_and_awards)
    
    # Map languages
    languages = get_value(field_mappings['languages'])
    if languages:
        if isinstance(languages, list):
            normalized['languages'] = json.dumps(languages) if languages else ''
        elif isinstance(languages, str):
            normalized['languages'] = languages
        else:
            normalized['languages'] = json.dumps(languages)
    
    # Map volunteer_and_awards
    volunteer_and_awards = get_value(field_mappings['volunteer_and_awards'])
    if volunteer_and_awards:
        if isinstance(volunteer_and_awards, list):
            normalized['volunteerAndAwards'] = json.dumps(volunteer_and_awards) if volunteer_and_awards else ''
        elif isinstance(volunteer_and_awards, str):
            normalized['volunteerAndAwards'] = volunteer_and_awards
        else:
            normalized['volunteerAndAwards'] = json.dumps(volunteer_and_awards)
    
    # Map verifications
    verifications = get_value(field_mappings['verifications'])
    if verifications:
        if isinstance(verifications, list):
            normalized['verifications'] = json.dumps(verifications) if verifications else ''
        elif isinstance(verifications, str):
            normalized['verifications'] = verifications
        else:
            normalized['verifications'] = json.dumps(verifications)
    
    # Map highlights
    highlights = get_value(field_mappings['highlights'])
    if highlights:
        if isinstance(highlights, list):
            normalized['highlights'] = json.dumps(highlights) if highlights else ''
        elif isinstance(highlights, str):
            normalized['highlights'] = highlights
        else:
            normalized['highlights'] = json.dumps(highlights)
    
    # Map projects (store as list)
    projects = get_value(field_mappings['projects'])
    if projects:
        if isinstance(projects, list):
            normalized['projects'] = projects if projects else []
        elif isinstance(projects, str):
            # Try to parse JSON string
            try:
                parsed = json.loads(projects)
                normalized['projects'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['projects'] = []
        else:
            normalized['projects'] = [projects]
    else:
        normalized['projects'] = []
    
    # Map publications (store as list)
    publications = get_value(field_mappings['publications'])
    if publications:
        if isinstance(publications, list):
            normalized['publications'] = publications if publications else []
        elif isinstance(publications, str):
            # Try to parse JSON string
            try:
                parsed = json.loads(publications)
                normalized['publications'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['publications'] = []
        else:
            normalized['publications'] = [publications]
    else:
        normalized['publications'] = []
    
    # Map patents (store as list)
    patents = get_value(field_mappings['patents'])
    if patents:
        if isinstance(patents, list):
            normalized['patents'] = patents if patents else []
        elif isinstance(patents, str):
            try:
                parsed = json.loads(patents)
                normalized['patents'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['patents'] = []
        else:
            normalized['patents'] = [patents]
    else:
        normalized['patents'] = []
    
    # Map courses (store as list)
    courses = get_value(field_mappings['courses'])
    if courses:
        if isinstance(courses, list):
            normalized['courses'] = courses if courses else []
        elif isinstance(courses, str):
            try:
                parsed = json.loads(courses)
                normalized['courses'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['courses'] = []
        else:
            normalized['courses'] = [courses]
    else:
        normalized['courses'] = []
    
    # Map test_scores (store as list)
    test_scores = get_value(field_mappings['test_scores'])
    if test_scores:
        if isinstance(test_scores, list):
            normalized['testScores'] = test_scores if test_scores else []
        elif isinstance(test_scores, str):
            try:
                parsed = json.loads(test_scores)
                normalized['testScores'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['testScores'] = []
        else:
            normalized['testScores'] = [test_scores]
    else:
        normalized['testScores'] = []
    
    # Map organizations (store as list)
    organizations = get_value(field_mappings['organizations'])
    if organizations:
        if isinstance(organizations, list):
            normalized['organizations'] = organizations if organizations else []
        elif isinstance(organizations, str):
            try:
                parsed = json.loads(organizations)
                normalized['organizations'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['organizations'] = []
        else:
            normalized['organizations'] = [organizations]
    else:
        normalized['organizations'] = []
    
    # Map volunteer_causes (store as list)
    volunteer_causes = get_value(field_mappings['volunteer_causes'])
    if volunteer_causes:
        if isinstance(volunteer_causes, list):
            normalized['volunteerCauses'] = volunteer_causes if volunteer_causes else []
        elif isinstance(volunteer_causes, str):
            try:
                parsed = json.loads(volunteer_causes)
                normalized['volunteerCauses'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['volunteerCauses'] = []
        else:
            normalized['volunteerCauses'] = [volunteer_causes]
    else:
        normalized['volunteerCauses'] = []
    
    # Map interests (store as list)
    interests = get_value(field_mappings['interests'])
    if interests:
        if isinstance(interests, list):
            normalized['interests'] = interests if interests else []
        elif isinstance(interests, str):
            try:
                parsed = json.loads(interests)
                normalized['interests'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['interests'] = []
        else:
            normalized['interests'] = [interests]
    else:
        normalized['interests'] = []
    
    # Map recommendations_received (store as list)
    recommendations_received = get_value(field_mappings['recommendations_received'])
    if recommendations_received:
        if isinstance(recommendations_received, list):
            normalized['recommendationsReceived'] = recommendations_received if recommendations_received else []
        elif isinstance(recommendations_received, str):
            try:
                parsed = json.loads(recommendations_received)
                normalized['recommendationsReceived'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['recommendationsReceived'] = []
        else:
            normalized['recommendationsReceived'] = [recommendations_received]
    else:
        normalized['recommendationsReceived'] = []
    
    # Map recommendations (store as list)
    recommendations = get_value(field_mappings['recommendations'])
    if recommendations:
        if isinstance(recommendations, list):
            normalized['recommendations'] = recommendations if recommendations else []
        elif isinstance(recommendations, str):
            try:
                parsed = json.loads(recommendations)
                normalized['recommendations'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['recommendations'] = []
        else:
            normalized['recommendations'] = [recommendations]
    else:
        normalized['recommendations'] = []
    
    # Map promos (store as list)
    promos = get_value(field_mappings['promos'])
    if promos:
        if isinstance(promos, list):
            normalized['promos'] = promos if promos else []
        elif isinstance(promos, str):
            try:
                parsed = json.loads(promos)
                normalized['promos'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['promos'] = []
        else:
            normalized['promos'] = [promos]
    else:
        normalized['promos'] = []
    
    # Map about
    about = get_value(field_mappings['about'])
    if about:
        normalized['about'] = str(about)
    
    # Map public_identifier
    public_identifier = get_value(field_mappings['public_identifier'])
    if public_identifier:
        normalized['publicIdentifier'] = str(public_identifier)
    
    # Map linkedin_public_url
    linkedin_public_url = get_value(field_mappings['linkedin_public_url'])
    if linkedin_public_url:
        normalized['linkedinPublicUrl'] = str(linkedin_public_url)
    
    # Map open_connection
    open_connection = get_value(field_mappings['open_connection'])
    if open_connection is not None:
        normalized['openConnection'] = bool(open_connection)
    
    # Map urn
    urn = get_value(field_mappings['urn'])
    if urn:
        normalized['urn'] = str(urn)
    
    # Map total_recommendations_received
    total_recommendations_received = get_value(field_mappings['total_recommendations_received'])
    if total_recommendations_received is not None:
        try:
            normalized['totalRecommendationsReceived'] = int(total_recommendations_received)
        except (ValueError, TypeError):
            pass
    
    # Map total_recommendations_given
    total_recommendations_given = get_value(field_mappings['total_recommendations_given'])
    if total_recommendations_given is not None:
        try:
            normalized['totalRecommendationsGiven'] = int(total_recommendations_given)
        except (ValueError, TypeError):
            pass
    
    # Map birthday (object)
    birthday = get_value(field_mappings['birthday'])
    if birthday:
        if isinstance(birthday, dict):
            normalized['birthday'] = birthday
        elif isinstance(birthday, str):
            try:
                normalized['birthday'] = json.loads(birthday)
            except (json.JSONDecodeError, TypeError):
                pass
    
    # Map associated_hashtag (store as list)
    associated_hashtag = get_value(field_mappings['associated_hashtag'])
    if associated_hashtag:
        if isinstance(associated_hashtag, list):
            normalized['associatedHashtag'] = associated_hashtag if associated_hashtag else []
        else:
            normalized['associatedHashtag'] = [str(associated_hashtag)]
    else:
        normalized['associatedHashtag'] = []
    
    # Map first_role_year
    first_role_year = get_value(field_mappings['first_role_year'])
    if first_role_year is not None:
        try:
            normalized['firstRoleYear'] = int(first_role_year)
        except (ValueError, TypeError):
            pass
    
    # Map total_experience_years
    total_experience_years = get_value(field_mappings['total_experience_years'])
    if total_experience_years is not None:
        try:
            normalized['totalExperienceYears'] = float(total_experience_years)
        except (ValueError, TypeError):
            pass
    
    # Map experiences_count
    experiences_count = get_value(field_mappings['experiences_count'])
    if experiences_count is not None:
        try:
            normalized['experiencesCount'] = int(experiences_count)
        except (ValueError, TypeError):
            pass
    
    # Map experiences (store as list)
    experiences = None
    if 'experiences' in candidate and candidate.get('experiences'):
        experiences = candidate['experiences']
    else:
        experiences = get_value(field_mappings['experiences'])
    
    if experiences:
        if isinstance(experiences, list):
            # Store as list for DynamoDB
            normalized['experiences'] = experiences if experiences else []
        elif isinstance(experiences, str):
            # Try to parse JSON string
            try:
                parsed = json.loads(experiences)
                normalized['experiences'] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                normalized['experiences'] = []
        else:
            normalized['experiences'] = [experiences]
    else:
        normalized['experiences'] = []
    
    # Map source (keep as null if not present)
    source = get_value(field_mappings['source'])
    if source:
        normalized['sourceURL'] = str(source)
    
    # Add metadata
    normalized['updated_at'] = datetime.utcnow().isoformat()
    normalized['created_at'] = datetime.utcnow().isoformat()
    
    # Get table schema to determine primary key
    schema = get_table_schema()
    
    if schema:
        # Use the actual partition key from table schema
        partition_key = schema['partition_key']
        sort_key = schema.get('sort_key')
        
        # Map linkedinUrl (partition key)
        linkedin_url = get_value(field_mappings['linkedin_url'])
        if not linkedin_url:
            # Generate a default LinkedIn URL if not provided
            linkedin_url = f"https://www.linkedin.com/in/{email.split('@')[0].replace('.', '-')}"
            logger.warning(f"LinkedIn URL not provided for {email}, generating default: {linkedin_url}")
        
        # Always ensure partition key is set
        normalized[partition_key] = linkedin_url
        
        # Set sort key if it exists and is not set
        if sort_key:
            if sort_key not in normalized:
                # Common sort key patterns: timestamp, created_at, or empty string
                if sort_key in ['created_at', 'updated_at', 'timestamp']:
                    normalized[sort_key] = datetime.utcnow().isoformat()
                else:
                    # For other sort keys, use a default value
                    normalized[sort_key] = datetime.utcnow().isoformat()
                    logger.warning(f"Using timestamp for sort key '{sort_key}'")
    else:
        # Fallback: use linkedinUrl as partition key if schema not available
        linkedin_url = get_value(field_mappings['linkedin_url'])
        if not linkedin_url:
            # Try linkedinPublicUrl as fallback
            linkedin_url = candidate.get('linkedinPublicUrl') or candidate.get('linkedin_public_url')
        
        if not linkedin_url:
            logger.error(f"LinkedIn URL (partition key) is required but not found for candidate. Skipping candidate.")
            raise ValueError(f"Missing required partition key 'linkedinUrl'. LinkedIn URL is required.")
        
        normalized['linkedinUrl'] = linkedin_url
        logger.warning("Table schema not available, using 'linkedinUrl' as partition key")
    
    return normalized


def upload_candidates_to_dynamodb(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Upload candidates to DynamoDB
    
    Args:
        candidates: List of candidate dictionaries
        
    Returns:
        Dictionary with upload results
    """
    if not table:
        raise ValueError("DynamoDB table not initialized. Please check AWS credentials.")
    
    if not candidates:
        raise ValueError("No candidates to upload")
    
    results = {
        'total': len(candidates),
        'successful': 0,
        'failed': 0,
        'errors': []
    }
    
    # Use batch_writer for efficient batch writes
    try:
        with table.batch_writer() as batch:
            for idx, candidate in enumerate(candidates):
                try:
                    # Normalize candidate data
                    normalized = normalize_candidate_data(candidate)
                    
                    # Prepare item for DynamoDB
                    # DynamoDB requires specific types (Decimal instead of float)
                    def convert_to_dynamodb_type(value):
                        """Convert Python types to DynamoDB-compatible types"""
                        if value is None:
                            return None
                        elif isinstance(value, bool):
                            return value
                        elif isinstance(value, int):
                            return value
                        elif isinstance(value, float):
                            # DynamoDB requires Decimal, not float
                            return Decimal(str(value))
                        elif isinstance(value, str):
                            return value
                        elif isinstance(value, list):
                            # Recursively convert list items
                            return [convert_to_dynamodb_type(item) for item in value]
                        elif isinstance(value, dict):
                            # Recursively convert dict values
                            return {k: convert_to_dynamodb_type(v) for k, v in value.items()}
                        else:
                            # Convert other types to string
                            return str(value)
                    
                    item = {}
                    for key, value in normalized.items():
                        converted_value = convert_to_dynamodb_type(value)
                        if converted_value is not None:
                            item[key] = converted_value
                    
                    # Write to DynamoDB
                    batch.put_item(Item=item)
                    results['successful'] += 1
                    
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"Candidate {idx + 1}: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
                    
                    # Log schema information if it's a validation error
                    if 'ValidationException' in str(e) or 'key element does not match' in str(e):
                        schema = get_table_schema()
                        if schema:
                            logger.error(f"Schema mismatch. Expected partition key: {schema['partition_key']}, sort key: {schema.get('sort_key')}")
                            logger.error(f"Item keys: {list(item.keys())}")
                            logger.error(f"Item partition key value: {item.get(schema['partition_key'], 'MISSING')}")
                            if schema.get('sort_key'):
                                logger.error(f"Item sort key value: {item.get(schema['sort_key'], 'MISSING')}")
        
        logger.info(f"Upload completed: {results['successful']} successful, {results['failed']} failed out of {results['total']} candidates")
        
    except Exception as e:
        logger.error(f"Error during batch upload: {e}")
        raise ValueError(f"Failed to upload candidates to DynamoDB: {str(e)}")
    
    return results


def upload_file_to_dynamodb(file_content: bytes, file_type: str) -> Dict[str, Any]:
    """
    Parse and upload file content to DynamoDB
    
    Args:
        file_content: Bytes content of the file
        file_type: 'csv' or 'json'
        
    Returns:
        Dictionary with upload results
    """
    try:
        # Parse file based on type
        if file_type.lower() == 'csv':
            candidates = parse_csv_file(file_content)
        elif file_type.lower() == 'json':
            candidates = parse_json_file(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Supported types: csv, json")
        
        if not candidates:
            raise ValueError("No candidates found in file")
        
        # Upload to DynamoDB
        results = upload_candidates_to_dynamodb(candidates)
        
        return {
            'success': True,
            'message': f'Successfully uploaded {results["successful"]} candidates to DynamoDB',
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error uploading file to DynamoDB: {e}")
        return {
            'success': False,
            'message': f'Failed to upload file: {str(e)}',
            'error': str(e)
        }

