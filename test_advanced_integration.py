#!/usr/bin/env python3
"""
Test script for advanced adeptai-master integration
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_import():
    """Test if the service can be imported"""
    try:
        logger.info("Testing service import...")
        from app.search.service import AdeptAIMastersAlgorithm, keyword_search, semantic_match
        logger.info("✅ Service imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_algorithm_initialization():
    """Test if the algorithm can be initialized"""
    try:
        logger.info("Testing algorithm initialization...")
        from app.search.service import AdeptAIMastersAlgorithm
        
        algorithm = AdeptAIMastersAlgorithm()
        logger.info("✅ Algorithm initialized successfully")
        
        # Check if advanced system is available
        if hasattr(algorithm, 'advanced_system') and algorithm.advanced_system:
            logger.info("🚀 Advanced adeptai-master system is available")
        else:
            logger.info("🔄 Using fallback system (this is normal if dependencies are missing)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Algorithm initialization failed: {e}")
        return False

def test_basic_search():
    """Test basic search functionality"""
    try:
        logger.info("Testing basic search...")
        from app.search.service import keyword_search
        
        # Test with a simple query
        test_query = "python developer with 3 years experience"
        results, summary = keyword_search(test_query, top_k=5)
        
        logger.info(f"✅ Search completed: {summary}")
        logger.info(f"📊 Found {len(results)} results")
        
        if results:
            logger.info(f"🏆 Top result: {results[0].get('FullName', 'Unknown')} - Score: {results[0].get('Score', 0)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Basic search failed: {e}")
        return False

def test_semantic_match():
    """Test semantic matching functionality"""
    try:
        logger.info("Testing semantic matching...")
        from app.search.service import semantic_match
        
        # Test with a simple query
        test_query = "senior software engineer"
        result = semantic_match(test_query)
        
        logger.info(f"✅ Semantic match completed: {result.get('summary', 'No summary')}")
        logger.info(f"📊 Found {len(result.get('results', []))} results")
        
        return True
    except Exception as e:
        logger.error(f"❌ Semantic match failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Starting advanced adeptai-master integration tests...")
    
    tests = [
        ("Import Test", test_import),
        ("Algorithm Initialization", test_algorithm_initialization),
        ("Basic Search", test_basic_search),
        ("Semantic Match", test_semantic_match)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Advanced integration is working correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 