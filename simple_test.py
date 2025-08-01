#!/usr/bin/env python3
"""
Simple test for backend service
"""

try:
    print("Testing backend service import...")
    from app.search.service import AdeptAIMastersAlgorithm
    print("✅ Backend service imported successfully")
    
    # Test algorithm initialization
    print("Testing algorithm initialization...")
    algorithm = AdeptAIMastersAlgorithm()
    print("✅ Algorithm initialized successfully")
    
    # Test basic functionality
    print("Testing basic search...")
    results, summary = algorithm.keyword_search("python developer", top_k=3)
    print(f"✅ Search completed: {summary}")
    print(f"📊 Found {len(results)} results")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc() 