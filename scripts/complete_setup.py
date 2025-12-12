"""
Complete setup script for Jobvite integration.
Automates all remaining infrastructure setup tasks.

Usage:
    python backend/scripts/complete_setup.py

This script will:
1. Check database connection
2. Run database migration
3. Generate encryption key
4. Check AWS credentials
5. Set up S3 bucket
6. Create .env file template
7. Verify setup
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_step(step_num, text):
    """Print formatted step"""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 60)

def check_database_connection():
    """Check if database connection is available"""
    print_step(1, "Checking database connection...")
    try:
        from app import create_app
        from app.models import db
        
        app = create_app()
        with app.app_context():
            db.engine.connect()
            print("✅ Database connection successful")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   Please ensure database is running and credentials are correct")
        return False

def run_database_migration():
    """Run database migration"""
    print_step(2, "Running database migration...")
    
    migration_file = Path(__file__).parent.parent / "migrations" / "add_jobvite_service_account_fields.sql"
    
    if not migration_file.exists():
        print(f"❌ Migration file not found: {migration_file}")
        return False
    
    print(f"📄 Migration file: {migration_file}")
    print("\nTo run migration, execute this SQL in your database:")
    print(f"   mysql -u user -p database < {migration_file}")
    print("\nOr use Alembic:")
    print("   alembic upgrade head")
    
    response = input("\nHave you run the migration? (y/n): ")
    if response.lower() == 'y':
        print("✅ Migration completed")
        return True
    else:
        print("⚠️  Migration not run. Please run it manually.")
        return False

def generate_encryption_key():
    """Generate encryption key"""
    print_step(3, "Generating encryption key...")
    
    try:
        import secrets
        import base64
        
        key_bytes = secrets.token_bytes(32)
        key_base64 = base64.b64encode(key_bytes).decode('utf-8')
        
        print("✅ Encryption key generated:")
        print(f"   {key_base64}")
        print("\n⚠️  IMPORTANT: Save this key securely!")
        print("   Add it to your .env file as: ENCRYPTION_KEY=<key>")
        
        return key_base64
    except Exception as e:
        print(f"❌ Failed to generate key: {e}")
        return None

def check_aws_credentials():
    """Check AWS credentials"""
    print_step(4, "Checking AWS credentials...")
    
    load_dotenv()
    
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'ap-south-1')
    
    if not aws_key or not aws_secret:
        print("❌ AWS credentials not found in environment")
        print("   Please set:")
        print("   - AWS_ACCESS_KEY_ID")
        print("   - AWS_SECRET_ACCESS_KEY")
        print("   - AWS_REGION (optional, default: ap-south-1)")
        return False
    
    print(f"✅ AWS credentials found")
    print(f"   Region: {aws_region}")
    print(f"   Access Key ID: {aws_key[:8]}...")
    
    return True

def setup_s3_bucket():
    """Set up S3 bucket"""
    print_step(5, "Setting up S3 bucket...")
    
    try:
        # Import and run S3 setup script
        sys.path.insert(0, str(Path(__file__).parent))
        from setup_jobvite_s3 import main as setup_s3_main
        
        print("Running S3 setup script...")
        setup_s3_main()
        return True
    except Exception as e:
        print(f"❌ S3 setup failed: {e}")
        print("   You can run it manually: python backend/scripts/setup_jobvite_s3.py")
        return False

def create_env_file():
    """Create .env file from template"""
    print_step(6, "Setting up environment variables...")
    
    env_template = Path(__file__).parent.parent / "env.jobvite.example"
    env_file = Path(__file__).parent.parent / ".env"
    
    if not env_template.exists():
        print(f"❌ Template file not found: {env_template}")
        return False
    
    if env_file.exists():
        response = input(f"⚠️  .env file already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("   Skipping .env file creation")
            return True
    
    # Read template
    with open(env_template, 'r') as f:
        template_content = f.read()
    
    # Generate encryption key if not provided
    encryption_key = generate_encryption_key()
    if encryption_key:
        template_content = template_content.replace(
            'ENCRYPTION_KEY=your_32_byte_base64_encoded_key',
            f'ENCRYPTION_KEY={encryption_key}'
        )
    
    # Write .env file
    with open(env_file, 'w') as f:
        f.write(template_content)
    
    print(f"✅ Created .env file: {env_file}")
    print("   Please fill in:")
    print("   - AWS_ACCESS_KEY_ID")
    print("   - AWS_SECRET_ACCESS_KEY")
    print("   - AWS_REGION")
    print("   - JOBVITE_DOCUMENTS_BUCKET")
    print("   - CELERY_BROKER_URL (if using Celery)")
    print("   - CELERY_RESULT_BACKEND (if using Celery)")
    
    return True

def verify_setup():
    """Verify setup is complete"""
    print_step(7, "Verifying setup...")
    
    checks = {
        "Database migration": False,
        "S3 bucket": False,
        "Environment variables": False,
        "Encryption key": False
    }
    
    # Check database
    try:
        from app import create_app
        from app.models import db, JobviteSettings
        
        app = create_app()
        with app.app_context():
            # Check if service account columns exist
            inspector = db.inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('jobvite_settings')]
            if 'service_account_username' in columns and 'service_account_password_encrypted' in columns:
                checks["Database migration"] = True
                print("✅ Database migration verified")
            else:
                print("❌ Database migration not complete")
    except Exception as e:
        print(f"⚠️  Could not verify database: {e}")
    
    # Check S3
    load_dotenv()
    bucket_name = os.getenv('JOBVITE_DOCUMENTS_BUCKET')
    if bucket_name:
        try:
            import boto3
            s3_client = boto3.client('s3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'ap-south-1')
            )
            s3_client.head_bucket(Bucket=bucket_name)
            checks["S3 bucket"] = True
            print("✅ S3 bucket verified")
        except Exception as e:
            print(f"⚠️  S3 bucket not accessible: {e}")
    else:
        print("⚠️  S3 bucket name not configured")
    
    # Check environment variables
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'ENCRYPTION_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    if not missing:
        checks["Environment variables"] = True
        print("✅ Environment variables verified")
    else:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
    
    # Check encryption key
    encryption_key = os.getenv('ENCRYPTION_KEY')
    if encryption_key and len(encryption_key) > 20:  # Basic check
        checks["Encryption key"] = True
        print("✅ Encryption key verified")
    else:
        print("⚠️  Encryption key not configured")
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP VERIFICATION SUMMARY")
    print("=" * 60)
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check}")
    
    all_complete = all(checks.values())
    if all_complete:
        print("\n✅ All setup tasks complete!")
    else:
        print("\n⚠️  Some setup tasks are incomplete. Please complete them manually.")
    
    return all_complete

def main():
    """Main setup function"""
    print_header("Jobvite Integration - Complete Setup")
    print("\nThis script will help you complete all remaining setup tasks.")
    print("Some steps require manual intervention (database migration, AWS credentials).")
    
    results = {
        "database_check": check_database_connection(),
        "migration": run_database_migration(),
        "encryption_key": generate_encryption_key() is not None,
        "aws_check": check_aws_credentials(),
        "s3_setup": False,  # Will be set by setup_s3_bucket
        "env_file": create_env_file()
    }
    
    # S3 setup (optional, can skip if AWS not configured)
    if results["aws_check"]:
        response = input("\nSet up S3 bucket now? (y/n): ")
        if response.lower() == 'y':
            results["s3_setup"] = setup_s3_bucket()
    else:
        print("\n⚠️  Skipping S3 setup (AWS credentials not configured)")
    
    # Verify setup
    verify_setup()
    
    print_header("Setup Complete")
    print("\nNext steps:")
    print("1. Complete any remaining manual tasks")
    print("2. Fill in environment variables in .env file")
    print("3. Run integration tests: pytest backend/tests/integration/test_jobvite_integration.py")
    print("4. Test the integration with real Jobvite credentials")
    print("\nFor detailed instructions, see: JOBVITE_COMPLETE_SETUP_GUIDE.md")

if __name__ == '__main__':
    main()

