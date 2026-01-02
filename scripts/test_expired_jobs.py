"""
Test script to check if expired jobs are returned
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from app import create_app
from app.models import Job
from app import db
from datetime import datetime, timedelta

app = create_app()

with app.app_context():
    # Get all jobs
    all_jobs = Job.query.filter_by(is_public=True, status='active').all()
    
    print("=" * 80)
    print("ALL PUBLIC ACTIVE JOBS:")
    print("=" * 80)
    
    for job in all_jobs:
        expires_at = job.expires_at
        is_expired = expires_at and expires_at < datetime.utcnow()
        
        print(f"Job ID: {job.id}")
        print(f"  Title: {job.title}")
        print(f"  Status: {job.status}")
        print(f"  Is Public: {job.is_public}")
        print(f"  Created At: {job.created_at}")
        print(f"  Expires At: {expires_at}")
        print(f"  Is Expired: {is_expired}")
        if expires_at:
            days_until_expiry = (expires_at - datetime.utcnow()).days
            print(f"  Days Until Expiry: {days_until_expiry}")
        print("-" * 80)
    
    print("\n")
    print("=" * 80)
    print("EXPIRED JOBS:")
    print("=" * 80)
    
    expired_jobs = [
        job for job in all_jobs 
        if job.expires_at and job.expires_at < datetime.utcnow()
    ]
    
    if not expired_jobs:
        print("No expired jobs found!")
    else:
        for job in expired_jobs:
            print(f"Job ID: {job.id} - {job.title}")
            print(f"  Expired: {job.expires_at}")
    
    print("\n")
    print("=" * 80)
    print("JOBS WITHOUT EXPIRATION:")
    print("=" * 80)
    
    no_expiry_jobs = [
        job for job in all_jobs 
        if job.expires_at is None
    ]
    
    if not no_expiry_jobs:
        print("All jobs have expiration dates!")
    else:
        for job in no_expiry_jobs:
            print(f"Job ID: {job.id} - {job.title}")

