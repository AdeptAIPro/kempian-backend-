# Batch Jobseeker Signup for High-Performance Registration
# Handles 1500+ jobseekers with parallel processing and optimization

import os
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.exceptions import ClientError
from app.simple_logger import get_logger
from app.models import db, User, CandidateProfile, CandidateSkill, CandidateEducation, CandidateExperience
from app.auth.cognito import cognito_signup
from app.services.resume_parser import parse_resume_data

logger = get_logger("batch_signup")

@dataclass
class JobseekerData:
    """Jobseeker signup data structure"""
    email: str
    password: str
    first_name: str
    last_name: str
    phone: str
    location: str
    visa_status: str
    resume_file_path: str
    resume_filename: str

@dataclass
class SignupResult:
    """Result of jobseeker signup"""
    email: str
    success: bool
    user_id: Optional[int] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0

class BatchJobseekerSignup:
    """High-performance batch jobseeker signup system"""
    
    def __init__(self, 
                 max_concurrent: int = 50,
                 batch_size: int = 100,
                 s3_bucket: str = "resume-bucket-adept-ai-pro"):
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.s3_bucket = s3_bucket
        
        # Initialize AWS clients
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        # Performance metrics
        self.metrics = {
            'total_processed': 0,
            'successful_signups': 0,
            'failed_signups': 0,
            'average_processing_time': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def signup_jobseekers_batch(self, jobseeker_data_list: List[JobseekerData]) -> List[SignupResult]:
        """
        Signup multiple jobseekers in parallel batches
        """
        self.metrics['start_time'] = time.time()
        total_jobseekers = len(jobseeker_data_list)
        
        logger.info(f"Starting batch signup for {total_jobseekers} jobseekers")
        logger.info(f"Configuration: max_concurrent={self.max_concurrent}, batch_size={self.batch_size}")
        
        # Split into batches
        batches = [jobseeker_data_list[i:i + self.batch_size] 
                  for i in range(0, total_jobseekers, self.batch_size)]
        
        all_results = []
        
        # Process batches sequentially to avoid overwhelming the system
        for batch_num, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_num + 1}/{len(batches)} ({len(batch)} jobseekers)")
            
            # Process batch in parallel
            batch_results = await self._process_batch_parallel(batch)
            all_results.extend(batch_results)
            
            # Update metrics
            self._update_metrics(batch_results)
            
            # Log progress
            success_count = sum(1 for r in batch_results if r.success)
            logger.info(f"Batch {batch_num + 1} completed: {success_count}/{len(batch)} successful")
            
            # Small delay between batches to prevent rate limiting
            if batch_num < len(batches) - 1:
                await asyncio.sleep(1)
        
        self.metrics['end_time'] = time.time()
        self._log_final_metrics()
        
        return all_results
    
    async def _process_batch_parallel(self, batch: List[JobseekerData]) -> List[SignupResult]:
        """Process a single batch in parallel"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def signup_single_jobseeker(data: JobseekerData) -> SignupResult:
            async with semaphore:
                return await self._signup_single_jobseeker(data)
        
        # Create tasks for parallel execution
        tasks = [signup_single_jobseeker(data) for data in batch]
        
        # Execute tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SignupResult(
                    email=batch[i].email,
                    success=False,
                    error_message=str(result),
                    processing_time=0.0
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _signup_single_jobseeker(self, data: JobseekerData) -> SignupResult:
        """Signup a single jobseeker with optimized processing"""
        start_time = time.time()
        
        try:
            # Step 1: Create Cognito user
            cognito_result = await self._create_cognito_user_async(data)
            if not cognito_result['success']:
                return SignupResult(
                    email=data.email,
                    success=False,
                    error_message=cognito_result['error'],
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Upload resume to S3
            s3_result = await self._upload_resume_async(data)
            if not s3_result['success']:
                return SignupResult(
                    email=data.email,
                    success=False,
                    error_message=s3_result['error'],
                    processing_time=time.time() - start_time
                )
            
            # Step 3: Create database user
            db_result = await self._create_database_user_async(data, s3_result['s3_key'])
            if not db_result['success']:
                return SignupResult(
                    email=data.email,
                    success=False,
                    error_message=db_result['error'],
                    processing_time=time.time() - start_time
                )
            
            # Step 4: Process resume and create profile
            profile_result = await self._create_candidate_profile_async(data, db_result['user_id'], s3_result['s3_key'])
            if not profile_result['success']:
                return SignupResult(
                    email=data.email,
                    success=False,
                    error_message=profile_result['error'],
                    processing_time=time.time() - start_time
                )
            
            processing_time = time.time() - start_time
            
            return SignupResult(
                email=data.email,
                success=True,
                user_id=db_result['user_id'],
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error signing up {data.email}: {str(e)}")
            return SignupResult(
                email=data.email,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _create_cognito_user_async(self, data: JobseekerData) -> Dict[str, Any]:
        """Create Cognito user asynchronously"""
        try:
            # Run Cognito signup in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                cognito_signup,
                data.email,
                data.password,
                None,  # tenant_id
                'job_seeker',
                'job_seeker',
                f"{data.first_name} {data.last_name}",
                data.first_name,
                data.last_name
            )
            
            return {'success': True, 'result': result}
            
        except Exception as e:
            error_msg = str(e)
            if "UsernameExistsException" in error_msg:
                return {'success': False, 'error': 'User already exists'}
            else:
                return {'success': False, 'error': f'Cognito error: {error_msg}'}
    
    async def _upload_resume_async(self, data: JobseekerData) -> Dict[str, Any]:
        """Upload resume to S3 asynchronously"""
        try:
            # Generate S3 key
            s3_key = f"resumes/{data.email}/{data.resume_filename}"
            
            # Run S3 upload in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._upload_to_s3,
                data.resume_file_path,
                s3_key
            )
            
            return {'success': True, 's3_key': s3_key}
            
        except Exception as e:
            return {'success': False, 'error': f'S3 upload error: {str(e)}'}
    
    def _upload_to_s3(self, file_path: str, s3_key: str):
        """Upload file to S3 (synchronous)"""
        try:
            self.s3_client.upload_file(file_path, self.s3_bucket, s3_key)
        except ClientError as e:
            raise Exception(f"S3 upload failed: {str(e)}")
    
    async def _create_database_user_async(self, data: JobseekerData, s3_key: str) -> Dict[str, Any]:
        """Create database user asynchronously"""
        try:
            # Run database operations in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._create_database_user_sync,
                data,
                s3_key
            )
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Database error: {str(e)}'}
    
    def _create_database_user_sync(self, data: JobseekerData, s3_key: str) -> Dict[str, Any]:
        """Create database user (synchronous)"""
        try:
            # Create user record
            user = User(
                email=data.email,
                role='job_seeker',
                user_type='job_seeker',
                first_name=data.first_name,
                last_name=data.last_name,
                phone=data.phone,
                location=data.location,
                is_active=True
            )
            
            db.session.add(user)
            db.session.flush()  # Get user ID
            
            # Create candidate profile
            profile = CandidateProfile(
                user_id=user.id,
                full_name=f"{data.first_name} {data.last_name}",
                phone=data.phone,
                location=data.location,
                resume_s3_key=s3_key,
                resume_filename=data.resume_filename,
                resume_upload_date=time.time(),
                is_public=True,
                visa_status=data.visa_status
            )
            
            db.session.add(profile)
            db.session.commit()
            
            return {'success': True, 'user_id': user.id}
            
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'error': f'Database error: {str(e)}'}
    
    async def _create_candidate_profile_async(self, data: JobseekerData, user_id: int, s3_key: str) -> Dict[str, Any]:
        """Create candidate profile with resume parsing"""
        try:
            # Run resume parsing in thread pool
            loop = asyncio.get_event_loop()
            parse_result = await loop.run_in_executor(
                None,
                self._parse_resume_and_create_profile,
                data,
                user_id,
                s3_key
            )
            
            return parse_result
            
        except Exception as e:
            return {'success': False, 'error': f'Profile creation error: {str(e)}'}
    
    def _parse_resume_and_create_profile(self, data: JobseekerData, user_id: int, s3_key: str) -> Dict[str, Any]:
        """Parse resume and create detailed profile (synchronous)"""
        try:
            # Parse resume
            resume_data = parse_resume_data(data.resume_file_path)
            
            if resume_data:
                # Update candidate profile with parsed data
                profile = CandidateProfile.query.filter_by(user_id=user_id).first()
                if profile:
                    profile.summary = resume_data.get('summary', '')
                    profile.experience_years = resume_data.get('experience_years', 0)
                    
                    # Add skills
                    if 'skills' in resume_data:
                        for skill_name in resume_data['skills']:
                            skill = CandidateSkill(
                                profile_id=profile.id,
                                skill_name=skill_name,
                                proficiency_level='Intermediate'
                            )
                            db.session.add(skill)
                    
                    # Add education
                    if 'education' in resume_data:
                        for edu in resume_data['education']:
                            education = CandidateEducation(
                                profile_id=profile.id,
                                institution=edu.get('institution', ''),
                                degree=edu.get('degree', ''),
                                field_of_study=edu.get('field_of_study', '')
                            )
                            db.session.add(education)
                    
                    # Add experience
                    if 'experience' in resume_data:
                        for exp in resume_data['experience']:
                            experience = CandidateExperience(
                                profile_id=profile.id,
                                job_title=exp.get('job_title', ''),
                                company=exp.get('company', ''),
                                description=exp.get('description', ''),
                                start_date=exp.get('start_date'),
                                end_date=exp.get('end_date')
                            )
                            db.session.add(experience)
                    
                    db.session.commit()
            
            return {'success': True}
            
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'error': f'Resume parsing error: {str(e)}'}
    
    def _update_metrics(self, batch_results: List[SignupResult]):
        """Update performance metrics"""
        self.metrics['total_processed'] += len(batch_results)
        self.metrics['successful_signups'] += sum(1 for r in batch_results if r.success)
        self.metrics['failed_signups'] += sum(1 for r in batch_results if not r.success)
        
        # Update average processing time
        if batch_results:
            avg_batch_time = sum(r.processing_time for r in batch_results) / len(batch_results)
            self.metrics['average_processing_time'] = (
                (self.metrics['average_processing_time'] * (self.metrics['total_processed'] - len(batch_results)) + 
                 avg_batch_time * len(batch_results)) / self.metrics['total_processed']
            )
    
    def _log_final_metrics(self):
        """Log final performance metrics"""
        total_time = self.metrics['end_time'] - self.metrics['start_time']
        success_rate = (self.metrics['successful_signups'] / self.metrics['total_processed']) * 100
        
        logger.info("=== BATCH SIGNUP COMPLETED ===")
        logger.info(f"Total jobseekers: {self.metrics['total_processed']}")
        logger.info(f"Successful signups: {self.metrics['successful_signups']}")
        logger.info(f"Failed signups: {self.metrics['failed_signups']}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per signup: {self.metrics['average_processing_time']:.2f} seconds")
        logger.info(f"Signups per minute: {(self.metrics['total_processed'] / total_time) * 60:.2f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()

# Utility functions for easy usage
def create_jobseeker_data_from_dict(data: Dict[str, Any]) -> JobseekerData:
    """Create JobseekerData from dictionary"""
    return JobseekerData(
        email=data['email'],
        password=data['password'],
        first_name=data['first_name'],
        last_name=data['last_name'],
        phone=data['phone'],
        location=data['location'],
        visa_status=data.get('visa_status', ''),
        resume_file_path=data['resume_file_path'],
        resume_filename=data['resume_filename']
    )

async def signup_jobseekers_fast(jobseeker_data_list: List[Dict[str, Any]], 
                                max_concurrent: int = 50,
                                batch_size: int = 100) -> List[SignupResult]:
    """
    Fast jobseeker signup function
    """
    # Convert dictionaries to JobseekerData objects
    jobseeker_objects = [create_jobseeker_data_from_dict(data) for data in jobseeker_data_list]
    
    # Create batch signup instance
    batch_signup = BatchJobseekerSignup(
        max_concurrent=max_concurrent,
        batch_size=batch_size
    )
    
    # Process signups
    results = await batch_signup.signup_jobseekers_batch(jobseeker_objects)
    
    return results

# Example usage
if __name__ == "__main__":
    # Example jobseeker data
    jobseeker_data = [
        {
            'email': 'user1@example.com',
            'password': 'password123',
            'first_name': 'John',
            'last_name': 'Doe',
            'phone': '+1234567890',
            'location': 'New York',
            'visa_status': 'US Citizen',
            'resume_file_path': '/path/to/resume1.pdf',
            'resume_filename': 'resume1.pdf'
        },
        # Add more jobseekers...
    ]
    
    # Run signup
    async def main():
        results = await signup_jobseekers_fast(jobseeker_data, max_concurrent=50, batch_size=100)
        
        # Print results
        for result in results:
            if result.success:
                print(f"✅ {result.email}: Success (User ID: {result.user_id})")
            else:
                print(f"❌ {result.email}: Failed - {result.error_message}")
    
    # Run the example
    asyncio.run(main())
