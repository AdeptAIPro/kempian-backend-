#!/usr/bin/env python3
"""
AdeptAI Startup Script
=====================

Clean startup script for the AdeptAI recruitment system.
Handles environment setup, dependency checking, and application startup.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import numpy
        import pandas
        import torch
        import transformers
        import faiss
        import boto3
        import redis
        print("✅ All core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment configuration"""
    print("🔍 Checking environment configuration...")
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("📝 Creating from template...")
        template_file = Path("env.template")
        if template_file.exists():
            with open(template_file, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("✅ .env file created from template")
        else:
            print("❌ env.template not found")
            return False
    else:
        print("✅ .env file found")
    
    # Check for required directories
    required_dirs = ['logs', 'cache', 'model']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
    
    return True

def run_security_check():
    """Run security validation"""
    try:
        print("🔒 Running security validation...")
        result = subprocess.run([sys.executable, 'security_config.py'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Security validation passed")
            return True
        else:
            print(f"⚠️  Security warnings: {result.stdout}")
            return True  # Continue with warnings
    except Exception as e:
        print(f"⚠️  Security check failed: {e}")
        return True  # Continue anyway

def main():
    """Main startup function"""
    print("🚀 Starting AdeptAI Recruitment System...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run security check
    run_security_check()
    
    # Start the application
    print("\n🌐 Starting AdeptAI server...")
    print("=" * 50)
    
    try:
        from main import main as app_main
        app_main()
    except KeyboardInterrupt:
        print("\n👋 Shutting down AdeptAI...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Failed to start AdeptAI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
