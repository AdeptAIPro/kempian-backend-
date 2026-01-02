"""
Verification script for Jobvite integration.
Tests all components to ensure they're working correctly.

Usage:
    python backend/scripts/verify_integration.py

This script will:
1. Verify database models
2. Verify encryption functions
3. Verify API clients (if credentials provided)
4. Verify S3 access (if configured)
5. Run unit tests
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_step(text, status=None):
    """Print formatted step"""
    icon = "✅" if status is True else "❌" if status is False else "⏳"
    print(f"{icon} {text}")

def verify_database_models():
    """Verify database models are correct"""
    print_header("Verifying Database Models")
    
    try:
        from app import create_app
        from app.models import (
            JobviteSettings, JobviteJob, JobviteCandidate,
            JobviteCandidateDocument, JobviteOnboardingProcess,
            JobviteOnboardingTask, JobviteWebhookLog
        )
        
        app = create_app()
        with app.app_context():
            # Check if service account fields exist
            from sqlalchemy import inspect
            inspector = inspect(JobviteSettings)
            columns = [col.name for col in inspector.columns]
            
            required_fields = [
                'service_account_username',
                'service_account_password_encrypted'
            ]
            
            missing = [field for field in required_fields if field not in columns]
            
            if missing:
                print_step("Service account fields missing", False)
                print(f"   Missing: {', '.join(missing)}")
                print("   Run migration: python migrations/add_jobvite_service_account_fields.py")
                return False
            else:
                print_step("Service account fields present", True)
            
            # Verify all models
            models = [
                ('JobviteSettings', JobviteSettings),
                ('JobviteJob', JobviteJob),
                ('JobviteCandidate', JobviteCandidate),
                ('JobviteCandidateDocument', JobviteCandidateDocument),
                ('JobviteOnboardingProcess', JobviteOnboardingProcess),
                ('JobviteOnboardingTask', JobviteOnboardingTask),
                ('JobviteWebhookLog', JobviteWebhookLog)
            ]
            
            for name, model in models:
                try:
                    inspector = inspect(model)
                    print_step(f"{name} model verified", True)
                except Exception as e:
                    print_step(f"{name} model error", False)
                    print(f"   Error: {e}")
                    return False
            
            return True
    except Exception as e:
        print_step("Database verification failed", False)
        print(f"   Error: {e}")
        return False

def verify_encryption():
    """Verify encryption functions work"""
    print_header("Verifying Encryption Functions")
    
    try:
        from app.jobvite.crypto import (
            encrypt_at_rest, decrypt_at_rest,
            generate_rsa_key_pair,
            encrypt_onboarding_payload,
            decrypt_onboarding_response
        )
        
        # Test at-rest encryption
        test_secret = "test_secret_12345"
        encrypted = encrypt_at_rest(test_secret)
        decrypted = decrypt_at_rest(encrypted)
        
        if decrypted == test_secret:
            print_step("At-rest encryption/decryption", True)
        else:
            print_step("At-rest encryption/decryption", False)
            return False
        
        # Test RSA key generation
        private_key, public_key = generate_rsa_key_pair()
        if "BEGIN PRIVATE KEY" in private_key and "BEGIN PUBLIC KEY" in public_key:
            print_step("RSA key generation", True)
        else:
            print_step("RSA key generation", False)
            return False
        
        # Test Onboarding encryption/decryption
        test_data = {"candidateId": "12345", "processId": "67890"}
        encrypted_payload = encrypt_onboarding_payload(test_data, public_key)
        decrypted_data = decrypt_onboarding_response(
            encrypted_payload["key"],
            encrypted_payload["payload"],
            private_key
        )
        
        if decrypted_data == test_data:
            print_step("Onboarding API encryption/decryption", True)
        else:
            print_step("Onboarding API encryption/decryption", False)
            return False
        
        return True
    except Exception as e:
        print_step("Encryption verification failed", False)
        print(f"   Error: {e}")
        return False

def verify_s3_access():
    """Verify S3 access"""
    print_header("Verifying S3 Access")
    
    load_dotenv()
    
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_name = os.getenv('JOBVITE_DOCUMENTS_BUCKET')
    
    if not aws_key or not aws_secret:
        print_step("AWS credentials not configured", False)
        return False
    
    if not bucket_name:
        print_step("S3 bucket name not configured", False)
        return False
    
    try:
        import boto3
        s3_client = boto3.client('s3',
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=os.getenv('AWS_REGION', 'ap-south-1')
        )
        
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        print_step("S3 bucket accessible", True)
        
        # Test upload (small test file)
        from app.jobvite.storage import upload_document_from_base64
        import base64
        
        test_content = "test document content"
        test_base64 = base64.b64encode(test_content.encode()).decode()
        
        s3_key, s3_url, public_url = upload_document_from_base64(
            content_base64=test_base64,
            filename="test.txt",
            tenant_id=1,
            candidate_id=999,
            doc_type="test"
        )
        
        if s3_key:
            print_step("S3 document upload", True)
            
            # Clean up test file
            try:
                s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
                print_step("S3 test cleanup", True)
            except:
                pass
        else:
            print_step("S3 document upload", False)
            return False
        
        return True
    except Exception as e:
        print_step("S3 access verification failed", False)
        print(f"   Error: {e}")
        return False

def verify_api_clients():
    """Verify API clients (if credentials provided)"""
    print_header("Verifying API Clients")
    
    load_dotenv()
    
    # Check if test credentials are provided
    test_api_key = os.getenv('JOBVITE_TEST_API_KEY')
    test_api_secret = os.getenv('JOBVITE_TEST_API_SECRET')
    test_company_id = os.getenv('JOBVITE_TEST_COMPANY_ID')
    
    if not all([test_api_key, test_api_secret, test_company_id]):
        print_step("Test credentials not provided", None)
        print("   Set JOBVITE_TEST_API_KEY, JOBVITE_TEST_API_SECRET, JOBVITE_TEST_COMPANY_ID")
        print("   to test API client connectivity")
        return None  # Not a failure, just not tested
    
    try:
        from app.jobvite.client_v2 import JobviteV2Client
        from app.jobvite.utils import get_base_urls
        
        base_urls = get_base_urls('stage')
        client = JobviteV2Client(
            api_key=test_api_key,
            api_secret=test_api_secret,
            company_id=test_company_id,
            base_url=base_urls['v2']
        )
        
        # Test connection
        result = client.get_job(start=0, count=1)
        if 'jobs' in result:
            print_step("API v2 client connection", True)
            return True
        else:
            print_step("API v2 client connection", False)
            return False
    except Exception as e:
        print_step("API client verification failed", False)
        print(f"   Error: {e}")
        return False

def run_unit_tests():
    """Run unit tests"""
    print_header("Running Unit Tests")
    
    test_files = [
        "backend/tests/test_jobvite_crypto.py",
        "backend/tests/test_jobvite_client.py",
        "backend/tests/test_jobvite_storage.py"
    ]
    
    results = {}
    for test_file in test_files:
        test_path = Path(__file__).parent.parent.parent / test_file
        if test_path.exists():
            print(f"\nRunning: {test_file}")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(test_path), "-v"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    print_step(f"{test_file} - PASSED", True)
                    results[test_file] = True
                else:
                    print_step(f"{test_file} - FAILED", False)
                    print(result.stdout)
                    print(result.stderr)
                    results[test_file] = False
            except subprocess.TimeoutExpired:
                print_step(f"{test_file} - TIMEOUT", False)
                results[test_file] = False
            except Exception as e:
                print_step(f"{test_file} - ERROR", False)
                print(f"   Error: {e}")
                results[test_file] = False
        else:
            print_step(f"{test_file} - NOT FOUND", False)
            results[test_file] = False
    
    return all(results.values()) if results else False

def main():
    """Main verification function"""
    print_header("Jobvite Integration - Verification")
    
    results = {
        "database": verify_database_models(),
        "encryption": verify_encryption(),
        "s3": verify_s3_access(),
        "api_clients": verify_api_clients(),  # May be None
        "unit_tests": run_unit_tests()
    }
    
    print_header("Verification Summary")
    
    for check, result in results.items():
        if result is None:
            print_step(f"{check}: Not tested", None)
        elif result:
            print_step(f"{check}: PASSED", True)
        else:
            print_step(f"{check}: FAILED", False)
    
    # Overall status
    tested_results = {k: v for k, v in results.items() if v is not None}
    all_passed = all(tested_results.values())
    
    if all_passed:
        print("\n✅ All verifications passed!")
    else:
        print("\n⚠️  Some verifications failed. Please fix the issues above.")
    
    return all_passed

if __name__ == '__main__':
    main()

