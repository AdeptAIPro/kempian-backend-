#!/usr/bin/env python3
"""
Local Feedback System Setup for AdeptAI Masters Algorithm
This script sets up the feedback system using local storage only
"""

import os
import json
from pathlib import Path

def setup_local_feedback_system():
    """Set up local feedback storage system"""
    print("🔧 Setting up Local Feedback System for AdeptAI Masters Algorithm")
    print("=" * 70)
    
    # Create feedback directory
    feedback_dir = Path("backend/feedback_data")
    feedback_dir.mkdir(exist_ok=True)
    
    # Create feedback file
    feedback_file = feedback_dir / "feedback.json"
    
    # Initialize empty feedback data
    if not feedback_file.exists():
        initial_data = {}
        with open(feedback_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
        print(f"✅ Created feedback file: {feedback_file}")
    else:
        print(f"✅ Feedback file already exists: {feedback_file}")
    
    # Create sample feedback data for testing
    sample_feedback = {
        "test_candidate@example.com": {
            "positive": 2,
            "negative": 1,
            "last_updated": "2024-12-30T00:00:00.000Z"
        },
        "sample_user@test.com": {
            "positive": 1,
            "negative": 0,
            "last_updated": "2024-12-30T00:00:00.000Z"
        }
    }
    
    # Update feedback file with sample data
    with open(feedback_file, 'w') as f:
        json.dump(sample_feedback, f, indent=2)
    
    print("✅ Added sample feedback data for testing")
    
    # Create feedback configuration
    config_file = feedback_dir / "config.json"
    config = {
        "storage_type": "local",
        "feedback_file": str(feedback_file),
        "backup_enabled": True,
        "max_backup_files": 5,
        "created_at": "2024-12-30T00:00:00.000Z"
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created feedback configuration: {config_file}")
    
    return True

def test_local_feedback_system():
    """Test the local feedback system"""
    print("\n🧪 Testing Local Feedback System...")
    
    try:
        feedback_file = Path("backend/feedback_data/feedback.json")
        
        if not feedback_file.exists():
            print("❌ Feedback file not found")
            return False
        
        # Test reading feedback
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
        
        print(f"✅ Successfully read feedback data: {len(feedback_data)} entries")
        
        # Test writing feedback
        test_entry = {
            "test_write@example.com": {
                "positive": 1,
                "negative": 0,
                "last_updated": "2024-12-30T00:00:00.000Z"
            }
        }
        
        feedback_data.update(test_entry)
        
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        print("✅ Successfully wrote feedback data")
        
        # Clean up test entry
        del feedback_data["test_write@example.com"]
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        print("✅ Successfully cleaned up test data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing feedback system: {e}")
        return False

def update_service_for_local_storage():
    """Update the service.py to use local storage by default"""
    print("\n📝 Updating service.py for local storage...")
    
    service_file = Path("backend/app/search/service.py")
    
    if not service_file.exists():
        print("❌ service.py not found")
        return False
    
    # Read current service file
    with open(service_file, 'r') as f:
        content = f.read()
    
    # Update the feedback file path to use local storage
    updated_content = content.replace(
        "FEEDBACK_FILE = 'feedback.json'",
        "FEEDBACK_FILE = 'feedback_data/feedback.json'"
    )
    
    # Write updated content
    with open(service_file, 'w') as f:
        f.write(updated_content)
    
    print("✅ Updated service.py to use local feedback storage")
    return True

def create_backup_system():
    """Create a backup system for feedback data"""
    print("\n💾 Setting up backup system...")
    
    backup_dir = Path("backend/feedback_data/backups")
    backup_dir.mkdir(exist_ok=True)
    
    # Create backup script
    backup_script = Path("backend/feedback_data/backup_feedback.py")
    
    backup_code = '''#!/usr/bin/env python3
"""
Backup script for feedback data
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

def backup_feedback_data():
    """Create a backup of feedback data"""
    feedback_file = Path("backend/feedback_data/feedback.json")
    backup_dir = Path("backend/feedback_data/backups")
    
    if not feedback_file.exists():
        print("No feedback file to backup")
        return
    
    # Create timestamp for backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"feedback_backup_{timestamp}.json"
    
    # Copy feedback file
    shutil.copy2(feedback_file, backup_file)
    print(f"✅ Backup created: {backup_file}")

if __name__ == "__main__":
    backup_feedback_data()
'''
    
    with open(backup_script, 'w') as f:
        f.write(backup_code)
    
    print(f"✅ Created backup script: {backup_script}")
    return True

def main():
    """Main function"""
    print("🔧 Local Feedback System Setup for AdeptAI Masters Algorithm")
    print("=" * 70)
    
    # Set up local feedback system
    if setup_local_feedback_system():
        # Test the system
        if test_local_feedback_system():
            # Update service for local storage
            if update_service_for_local_storage():
                # Create backup system
                if create_backup_system():
                    print("\n🎉 Local feedback system setup completed successfully!")
                    print("\n📋 Summary:")
                    print("   ✅ Local feedback storage created")
                    print("   ✅ Sample data added for testing")
                    print("   ✅ Service.py updated for local storage")
                    print("   ✅ Backup system created")
                    print("\n🚀 The AdeptAI Masters algorithm can now use local feedback storage!")
                    print("   No AWS credentials required.")
                    return True
    
    print("\n❌ Setup failed")
    return False

if __name__ == "__main__":
    main() 