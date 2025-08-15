#!/usr/bin/env python3
"""
Backend Setup Script - Ensures all dependencies and components are ready
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def install_requirements():
    """Install all required packages"""
    print("📦 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def install_nltk_data():
    """Install required NLTK data"""
    print("📚 Installing NLTK data...")
    
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK data installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install NLTK data: {e}")
        return False

def check_imports():
    """Check if all critical imports work"""
    print("🔍 Checking imports...")
    
    required_packages = [
        'flask',
        'boto3', 
        'numpy',
        'pandas',
        'faiss',
        'torch',
        'sentence_transformers',
        'nltk',
        'sklearn',
        'lightgbm'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"⚠️ Failed imports: {failed_imports}")
        return False
    
    return True

def create_missing_directories():
    """Create any missing directories"""
    print("📁 Creating directories...")
    
    directories = [
        'logs',
        'data',
        'models',
        'cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/")

def setup_environment():
    """Setup environment variables"""
    print("🔧 Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path('.env')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("""# Backend Environment Variables
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=ap-south-1
""")
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")

def test_components():
    """Test all components"""
    print("🧪 Testing components...")
    
    try:
        # Test basic imports
        from app.search.service import AdeptAIMastersAlgorithm
        print("✅ Service import")
        
        # Test algorithm
        algorithm = AdeptAIMastersAlgorithm()
        print("✅ Algorithm initialization")
        
        # Test adeptai components
        from app.search.adeptai_components.enhanced_recruitment_search import EnhancedRecruitmentSearchSystem
        from app.search.adeptai_components.enhanced_candidate_matcher import EnhancedCandidateMatchingSystem
        from app.search.adeptai_components.advanced_query_parser import AdvancedJobQueryParser
        print("✅ AdeptAI components import")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Backend Setup Script")
    print("=" * 40)
    
    steps = [
        ("Install Requirements", install_requirements),
        ("Install NLTK Data", install_nltk_data),
        ("Check Imports", check_imports),
        ("Create Directories", create_missing_directories),
        ("Setup Environment", setup_environment),
        ("Test Components", test_components),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"❌ {step_name} failed")
            return False
        print(f"✅ {step_name} completed")
    
    print("\n" + "=" * 40)
    print("🎉 Backend setup completed successfully!")
    print("\nNext steps:")
    print("1. Configure your .env file with AWS credentials")
    print("2. Run: python test_complete_backend.py")
    print("3. Start the server: python app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 