"""
Script to set up S3 bucket for Jobvite document storage.

This script:
1. Creates S3 bucket (if it doesn't exist)
2. Configures bucket policy for secure access
3. Enables server-side encryption
4. Sets up lifecycle policies (optional)

Usage:
    python backend/scripts/setup_jobvite_s3.py
"""

import os
import boto3
import json
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# S3 Configuration
BUCKET_NAME = os.getenv('JOBVITE_DOCUMENTS_BUCKET', 'jobvite-documents')
AWS_REGION = os.getenv('AWS_REGION', 'ap-south-1')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

def create_s3_client():
    """Create S3 client with credentials"""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in .env")
    
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

def create_bucket(s3_client, bucket_name, region):
    """Create S3 bucket if it doesn't exist"""
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' already exists")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            # Bucket doesn't exist, create it
            try:
                if region == 'us-east-1':
                    # us-east-1 doesn't require LocationConstraint
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"✅ Created bucket '{bucket_name}' in region '{region}'")
                return True
            except ClientError as create_error:
                print(f"❌ Error creating bucket: {create_error}")
                return False
        else:
            print(f"❌ Error checking bucket: {e}")
            return False

def enable_encryption(s3_client, bucket_name):
    """Enable server-side encryption (AES256)"""
    try:
        s3_client.put_bucket_encryption(
            Bucket=bucket_name,
            ServerSideEncryptionConfiguration={
                'Rules': [{
                    'ApplyServerSideEncryptionByDefault': {
                        'SSEAlgorithm': 'AES256'
                    }
                }]
            }
        )
        print(f"✅ Enabled server-side encryption (AES256) for bucket '{bucket_name}'")
        return True
    except ClientError as e:
        print(f"❌ Error enabling encryption: {e}")
        return False

def set_bucket_policy(s3_client, bucket_name):
    """Set bucket policy for secure access"""
    # Policy: Only allow access from application with proper IAM role
    # Adjust this policy based on your security requirements
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowApplicationAccess",
                "Effect": "Allow",
                "Principal": {
                    "AWS": f"arn:aws:iam::{os.getenv('AWS_ACCOUNT_ID', 'YOUR_ACCOUNT_ID')}:root"
                },
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject"
                ],
                "Resource": f"arn:aws:s3:::{bucket_name}/jobvite/documents/*"
            },
            {
                "Sid": "DenyPublicAccess",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:*",
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ],
                "Condition": {
                    "Bool": {
                        "aws:PublicAccess": "true"
                    }
                }
            }
        ]
    }
    
    try:
        s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=json.dumps(policy)
        )
        print(f"✅ Set bucket policy for '{bucket_name}'")
        return True
    except ClientError as e:
        print(f"⚠️  Warning: Could not set bucket policy: {e}")
        print("   You may need to set this manually in AWS Console")
        return False

def block_public_access(s3_client, bucket_name):
    """Block all public access to the bucket"""
    try:
        s3_client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,
                'IgnorePublicAcls': True,
                'BlockPublicPolicy': True,
                'RestrictPublicBuckets': True
            }
        )
        print(f"✅ Blocked public access for bucket '{bucket_name}'")
        return True
    except ClientError as e:
        print(f"⚠️  Warning: Could not block public access: {e}")
        return False

def setup_lifecycle_policy(s3_client, bucket_name):
    """Set up lifecycle policy (optional - delete old documents after 90 days)"""
    try:
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration={
                'Rules': [{
                    'Id': 'DeleteOldDocuments',
                    'Status': 'Enabled',
                    'Prefix': 'jobvite/documents/',
                    'Expiration': {
                        'Days': 90  # Delete documents older than 90 days
                    }
                }]
            }
        )
        print(f"✅ Set lifecycle policy for bucket '{bucket_name}' (90-day retention)")
        return True
    except ClientError as e:
        print(f"⚠️  Warning: Could not set lifecycle policy: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("Jobvite S3 Bucket Setup")
    print("=" * 60)
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Region: {AWS_REGION}")
    print()
    
    try:
        s3_client = create_s3_client()
        
        # Step 1: Create bucket
        if not create_bucket(s3_client, BUCKET_NAME, AWS_REGION):
            print("❌ Failed to create bucket. Exiting.")
            return
        
        # Step 2: Enable encryption
        enable_encryption(s3_client, BUCKET_NAME)
        
        # Step 3: Block public access
        block_public_access(s3_client, BUCKET_NAME)
        
        # Step 4: Set bucket policy
        set_bucket_policy(s3_client, BUCKET_NAME)
        
        # Step 5: Set lifecycle policy (optional)
        response = input("\nSet up lifecycle policy (delete documents after 90 days)? (y/n): ")
        if response.lower() == 'y':
            setup_lifecycle_policy(s3_client, BUCKET_NAME)
        
        print("\n" + "=" * 60)
        print("✅ S3 bucket setup complete!")
        print("=" * 60)
        print(f"\nBucket URL: s3://{BUCKET_NAME}")
        print(f"Region: {AWS_REGION}")
        print("\nNext steps:")
        print("1. Verify bucket in AWS Console")
        print("2. Test document upload/retrieval")
        print("3. Monitor bucket usage and costs")
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("\nTroubleshooting:")
        print("1. Check AWS credentials in .env file")
        print("2. Verify AWS permissions (S3 full access)")
        print("3. Check AWS region is correct")

if __name__ == '__main__':
    main()

