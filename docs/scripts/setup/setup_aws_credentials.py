#!/usr/bin/env python3
"""
AWS Credentials Setup Script for AdeptAI Masters Algorithm
This script helps set up AWS credentials for DynamoDB access
"""

import os
import sys
from pathlib import Path

def check_aws_credentials():
    """Check if AWS credentials are available"""
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if access_key and secret_key:
        print("✅ AWS credentials found in environment variables")
        return True
    else:
        print("❌ AWS credentials not found in environment variables")
        return False

def setup_aws_credentials():
    """Interactive setup for AWS credentials"""
    print("\n🔧 AWS Credentials Setup")
    print("=" * 40)
    
    print("You need AWS credentials to create the DynamoDB table.")
    print("Please provide your AWS credentials:")
    
    access_key = input("AWS Access Key ID: ").strip()
    secret_key = input("AWS Secret Access Key: ").strip()
    
    if not access_key or not secret_key:
        print("❌ Credentials cannot be empty")
        return False
    
    # Set environment variables for current session
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    
    print("✅ AWS credentials set for current session")
    return True

def create_aws_config_file():
    """Create AWS config file for persistent credentials"""
    print("\n📁 Creating AWS config file for persistent credentials...")
    
    # Get user's home directory
    home_dir = Path.home()
    aws_dir = home_dir / ".aws"
    aws_dir.mkdir(exist_ok=True)
    
    config_file = aws_dir / "credentials"
    
    print(f"Creating credentials file at: {config_file}")
    
    access_key = input("AWS Access Key ID: ").strip()
    secret_key = input("AWS Secret Access Key: ").strip()
    
    if not access_key or not secret_key:
        print("❌ Credentials cannot be empty")
        return False
    
    # Write credentials to file
    with open(config_file, 'w') as f:
        f.write("[default]\n")
        f.write(f"aws_access_key_id = {access_key}\n")
        f.write(f"aws_secret_access_key = {secret_key}\n")
    
    print("✅ AWS credentials file created successfully")
    print(f"   Location: {config_file}")
    print("   Note: Keep this file secure and don't share it")
    
    return True

def test_aws_connection():
    """Test AWS connection with current credentials"""
    try:
        import boto3
        
        # Test connection
        sts = boto3.client('sts')
        response = sts.get_caller_identity()
        
        print("✅ AWS connection successful!")
        print(f"   Account ID: {response['Account']}")
        print(f"   User ARN: {response['Arn']}")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS connection failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 AWS Credentials Setup for AdeptAI Masters Algorithm")
    print("=" * 60)
    
    # Check if credentials already exist
    if check_aws_credentials():
        if test_aws_connection():
            print("\n🎉 AWS credentials are working! You can now run:")
            print("   python create_feedback_table.py")
            return True
    
    print("\nChoose setup option:")
    print("1. Set credentials for current session only")
    print("2. Create persistent AWS config file")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        if setup_aws_credentials():
            if test_aws_connection():
                print("\n🎉 AWS credentials set for current session!")
                print("   You can now run: python create_feedback_table.py")
                return True
    
    elif choice == "2":
        if create_aws_config_file():
            print("\n🎉 AWS credentials file created!")
            print("   You can now run: python create_feedback_table.py")
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