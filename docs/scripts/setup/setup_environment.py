#!/usr/bin/env python3
"""
Environment Setup Script for AdeptAI Masters Algorithm
This script helps set up environment variables for optimal performance
"""

import os
from pathlib import Path

def check_environment():
    """Check current environment setup"""
    print("🔍 Checking Environment Setup")
    print("=" * 50)
    
    # Check AWS credentials
    aws_access = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if aws_access and aws_secret:
        print("✅ AWS credentials found")
    else:
        print("❌ AWS credentials not found - will use local storage only")
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith('your-ope'):
        print("✅ OpenAI API key found")
    else:
        print("❌ OpenAI API key not found or invalid - GPT-4 reranking disabled")
    
    # Check feedback system
    feedback_file = Path("feedback_data/feedback.json")
    if feedback_file.exists():
        print("✅ Local feedback system ready")
    else:
        print("⚠️  Local feedback system not set up")
    
    return True

def setup_local_only():
    """Set up for local-only operation (no external APIs)"""
    print("\n🔧 Setting up Local-Only Configuration")
    print("=" * 50)
    
    # Create .env file for local configuration
    env_content = """# AdeptAI Masters Algorithm - Local Configuration
# This file contains environment variables for local operation

# OpenAI API Key (optional - set to enable GPT-4 reranking)
# OPENAI_API_KEY=your-actual-openai-api-key-here

# AWS Credentials (optional - set to enable DynamoDB)
# AWS_ACCESS_KEY_ID=your-aws-access-key
# AWS_SECRET_ACCESS_KEY=your-aws-secret-key

# Algorithm Configuration
ADEPTAI_USE_LOCAL_STORAGE=true
ADEPTAI_ENABLE_GPT4=false
ADEPTAI_ENABLE_DYNAMODB=false
"""
    
    env_file = Path(".env")
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file for local configuration")
    print("   Edit this file to add your API keys when ready")
    
    return True

def create_startup_script():
    """Create a startup script for easy launching"""
    print("\n📝 Creating Startup Script")
    print("=" * 50)
    
    startup_content = """#!/usr/bin/env python3
"""
# AdeptAI Masters Algorithm - Startup Script
# This script starts the backend with proper environment setup

import os
import sys
from pathlib import Path

def load_env_file():
    \"\"\"Load environment variables from .env file\"\"\"
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

def main():
    \"\"\"Main startup function\"\"\"
    print("🚀 Starting AdeptAI Masters Algorithm Backend")
    print("=" * 50)
    
    # Load environment variables
    load_env_file()
    
    # Check if feedback system is ready
    feedback_file = Path("feedback_data/feedback.json")
    if not feedback_file.exists():
        print("⚠️  Feedback system not found. Running setup...")
        os.system("python create_feedback_table_local.py")
    
    # Start the Flask application
    print("✅ Starting Flask application...")
    os.system("python main.py")

if __name__ == "__main__":
    main()
"""
    
    startup_file = Path("start_adeptai.py")
    with open(startup_file, 'w') as f:
        f.write(startup_content)
    
    print("✅ Created startup script: start_adeptai.py")
    print("   Run this script to start the backend with proper configuration")
    
    return True

def main():
    """Main setup function"""
    print("🔧 AdeptAI Masters Algorithm - Environment Setup")
    print("=" * 60)
    
    # Check current environment
    check_environment()
    
    print("\nChoose setup option:")
    print("1. Set up for local-only operation (recommended)")
    print("2. Check environment only")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        if setup_local_only():
            if create_startup_script():
                print("\n🎉 Local setup completed successfully!")
                print("\n📋 Next Steps:")
                print("   1. Run: python start_adeptai.py")
                print("   2. The algorithm will work with local storage only")
                print("   3. Add API keys to .env file when ready for full features")
                return True
    
    elif choice == "2":
        print("\n✅ Environment check completed")
        return True
    
    elif choice == "3":
        print("Exiting...")
        return False
    
    else:
        print("❌ Invalid choice")
        return False
    
    return False

if __name__ == "__main__":
    main() 