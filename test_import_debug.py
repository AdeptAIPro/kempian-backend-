"""
Test script to debug the import issues with accuracy enhancement system
Run this script to identify the exact import problem
"""

import sys
import os
import traceback

def test_imports():
    print("=" * 60)
    print("DEBUG: Testing accuracy enhancement system imports")
    print("=" * 60)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path first 5 entries:")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i}: {path}")
    print()
    
    # Test 1: Try importing the module directly
    print("Test 1: Direct import attempt")
    try:
        import app.search.accuracy_enhancement_system
        print("✅ SUCCESS: Direct import worked")
    except Exception as e:
        print(f"❌ FAILED: Direct import failed - {e}")
        print(f"   Exception type: {type(e)}")
        print(f"   Exception args: {e.args}")
        print(f"   Traceback: {traceback.format_exc()}")
    print()
    
    # Test 2: Try importing from app.search
    print("Test 2: From app.search import")
    try:
        from app.search import accuracy_enhancement_system
        print("✅ SUCCESS: From app.search import worked")
    except Exception as e:
        print(f"❌ FAILED: From app.search import failed - {e}")
        print(f"   Exception type: {type(e)}")
        print(f"   Exception args: {e.args}")
        print(f"   Traceback: {traceback.format_exc()}")
    print()
    
    # Test 3: Try importing the specific function
    print("Test 3: Import specific function")
    try:
        from app.search.accuracy_enhancement_system import enhance_search_accuracy
        print("✅ SUCCESS: Function import worked")
    except Exception as e:
        print(f"❌ FAILED: Function import failed - {e}")
        print(f"   Exception type: {type(e)}")
        print(f"   Exception args: {e.args}")
        print(f"   Traceback: {traceback.format_exc()}")
    print()
    
    # Test 4: Check if the file exists
    print("Test 4: Check file existence")
    file_path = "app/search/accuracy_enhancement_system.py"
    if os.path.exists(file_path):
        print(f"✅ SUCCESS: File exists at {file_path}")
        print(f"   File size: {os.path.getsize(file_path)} bytes")
    else:
        print(f"❌ FAILED: File not found at {file_path}")
        
        # Try alternative paths
        alt_paths = [
            "backend/app/search/accuracy_enhancement_system.py",
            "./app/search/accuracy_enhancement_system.py",
            "../app/search/accuracy_enhancement_system.py"
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"✅ Found at alternative path: {alt_path}")
                break
    print()
    
    # Test 5: Try to read the file content
    print("Test 5: Read file content")
    try:
        with open("app/search/accuracy_enhancement_system.py", "r") as f:
            content = f.read()
            print(f"✅ SUCCESS: File read successfully")
            print(f"   First 100 characters: {content[:100]}...")
            print(f"   Contains 'enhance_search_accuracy': {'enhance_search_accuracy' in content}")
    except Exception as e:
        print(f"❌ FAILED: Could not read file - {e}")
    print()
    
    # Test 6: Check for circular imports
    print("Test 6: Check for potential circular imports")
    try:
        # Check if accuracy_enhancement_system imports from service
        with open("app/search/accuracy_enhancement_system.py", "r") as f:
            content = f.read()
            if "from app.search.service" in content or "import service" in content:
                print("⚠️  WARNING: Potential circular import detected")
                print("   accuracy_enhancement_system.py imports from service")
            else:
                print("✅ SUCCESS: No obvious circular imports detected")
    except Exception as e:
        print(f"❌ FAILED: Could not check for circular imports - {e}")
    
    print("=" * 60)
    print("DEBUG: Import testing completed")
    print("=" * 60)

if __name__ == "__main__":
    test_imports()
