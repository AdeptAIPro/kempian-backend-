#!/usr/bin/env python3
"""
Simple test script for AdeptAI Masters Algorithm
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic algorithm functionality"""
    print("🧪 Testing AdeptAI Masters Algorithm...")
    
    try:
        # Test import
        from app.search.service import AdeptAIMastersAlgorithm
        print("✅ Algorithm imported successfully")
        
        # Test initialization
        algorithm = AdeptAIMastersAlgorithm()
        print("✅ Algorithm initialized successfully")
        
        # Test keyword extraction
        test_text = "Python developer with Django and React experience"
        keywords = algorithm.extract_keywords(test_text)
        print(f"✅ Keyword extraction: {keywords}")
        
        # Test domain detection
        domain = algorithm.detect_domain(keywords)
        print(f"✅ Domain detection: {domain}")
        
        # Test scoring
        score = algorithm.nlrga_score(['python', 'django'], 3, 'test@example.com')
        print(f"✅ Scoring: {score}")
        
        # Test grade
        grade = algorithm.get_grade(score)
        print(f"✅ Grade: {grade}")
        
        print("🎉 All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functionality() 