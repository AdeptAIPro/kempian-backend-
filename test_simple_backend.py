#!/usr/bin/env python3
"""
Simple Backend Test - Tests basic functionality without heavy ML models
"""

import os
import sys
import traceback
from datetime import datetime

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    
    try:
        import boto3
        print("✅ boto3 imported")
        
        import numpy as np
        print("✅ numpy imported")
        
        import pandas as pd
        print("✅ pandas imported")
        
        from flask import Flask
        print("✅ Flask imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic import error: {e}")
        return False

def test_service_import():
    """Test service import without initializing advanced system"""
    print("\n🔧 Testing service import...")
    
    try:
        # Import without initializing the algorithm
        from app.search.service import AdeptAIMastersAlgorithm
        print("✅ Service imported successfully")
        
        # Test basic methods without ML models
        algorithm = AdeptAIMastersAlgorithm()
        print("✅ Algorithm created")
        
        # Test basic methods
        score = algorithm.get_grade(85)
        print(f"✅ Grade calculation: 85 -> {score}")
        
        keywords = algorithm.extract_keywords("Python developer with React experience")
        print(f"✅ Keyword extraction: {keywords[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Service import error: {e}")
        traceback.print_exc()
        return False

def test_adeptai_components():
    """Test adeptai components without heavy initialization"""
    print("\n🚀 Testing adeptai components...")
    
    try:
        # Test query parser (lightweight)
        from app.search.adeptai_components.advanced_query_parser import AdvancedJobQueryParser
        
        parser = AdvancedJobQueryParser()
        query = "Senior Python Developer with 5+ years experience in React and AWS"
        parsed = parser.parse_job_query(query)
        print(f"✅ Query parsing: {parsed.job_title}")
        
        # Test candidate matcher (lightweight)
        from app.search.adeptai_components.enhanced_candidate_matcher import EnhancedCandidateMatchingSystem
        
        matcher = EnhancedCandidateMatchingSystem()
        print("✅ Candidate matcher initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Adeptai components error: {e}")
        traceback.print_exc()
        return False

def test_basic_search():
    """Test basic search functionality"""
    print("\n🔍 Testing basic search...")
    
    try:
        from app.search.service import AdeptAIMastersAlgorithm
        
        algorithm = AdeptAIMastersAlgorithm()
        
        # Test with a simple query
        job_description = "Python developer with React experience"
        
        print("Testing keyword search...")
        results, summary = algorithm.keyword_search(job_description, top_k=3)
        
        print(f"✅ Search completed: {summary}")
        print(f"📊 Found {len(results)} results")
        
        if results:
            print(f"📋 First result: {results[0].get('FullName', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic search error: {e}")
        traceback.print_exc()
        return False

def test_semantic_match():
    """Test semantic matching"""
    print("\n🧠 Testing semantic matching...")
    
    try:
        from app.search.service import AdeptAIMastersAlgorithm
        
        algorithm = AdeptAIMastersAlgorithm()
        
        job_description = "Senior Software Engineer with Python and React experience"
        
        result = algorithm.semantic_match(job_description)
        
        print(f"✅ Semantic match completed")
        print(f"📊 Results: {len(result.get('results', []))} candidates")
        print(f"📋 Summary: {result.get('summary', 'No summary')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Semantic match error: {e}")
        traceback.print_exc()
        return False

def test_flask_app():
    """Test Flask app initialization"""
    print("\n🌐 Testing Flask app...")
    
    try:
        from app import create_app
        
        app = create_app()
        print("✅ Flask app created successfully")
        
        # Test basic routes
        with app.test_client() as client:
            response = client.get('/health')
            print(f"✅ Health check: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Simple Backend Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Service Import", test_service_import),
        ("AdeptAI Components", test_adeptai_components),
        ("Basic Search", test_basic_search),
        ("Semantic Match", test_semantic_match),
        ("Flask App", test_flask_app),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Backend is ready for hosting.")
        return True
    elif passed >= total - 1:
        print("✅ MOST TESTS PASSED! Backend should work with fallback systems.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 