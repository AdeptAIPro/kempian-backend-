#!/usr/bin/env python3
"""
Performance monitoring script for search optimization
"""

import sys
import os
import time
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.search.accuracy_enhancement_system import enhance_search_accuracy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def create_test_candidates(count: int = 1000):
    """Create test candidates for performance testing"""
    candidates = []
    
    for i in range(count):
        candidate = {
            'id': f'test_{i}',
            'name': f'Test Candidate {i}',
            'title': f'Software Developer {i}',
            'skills': ['Python', 'JavaScript', 'React', 'AWS', 'Docker'],
            'experience_years': 3 + (i % 10),
            'resume_text': f'Experienced software developer with {3 + (i % 10)} years of experience in Python, JavaScript, React, AWS, and Docker. Strong background in full-stack development and cloud technologies.',
            'match_percentage': 60 + (i % 30)
        }
        candidates.append(candidate)
    
    return candidates

def test_performance():
    """Test search performance with different candidate counts"""
    
    test_query = "Senior Python Developer with React experience, 5+ years experience, AWS knowledge required"
    
    # Test with different candidate counts
    test_sizes = [50, 100, 200, 500, 1000]
    
    print("Search Performance Test")
    print("=" * 50)
    print(f"Test Query: {test_query}")
    print()
    
    results = []
    
    for size in test_sizes:
        print(f"Testing with {size} candidates...")
        
        # Create test candidates
        candidates = create_test_candidates(size)
        
        # Measure performance
        start_time = time.time()
        enhanced_results = enhance_search_accuracy(test_query, candidates, top_k=20)
        end_time = time.time()
        
        processing_time = end_time - start_time
        avg_time_per_candidate = processing_time / size
        
        results.append({
            'candidates': size,
            'processing_time': processing_time,
            'avg_time_per_candidate': avg_time_per_candidate,
            'results_returned': len(enhanced_results)
        })
        
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Avg time per candidate: {avg_time_per_candidate:.4f}s")
        print(f"  Results returned: {len(enhanced_results)}")
        print()
    
    # Summary
    print("Performance Summary:")
    print("-" * 50)
    print(f"{'Candidates':<12} {'Time (s)':<10} {'Per Candidate (s)':<18} {'Results':<8}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['candidates']:<12} {result['processing_time']:<10.2f} {result['avg_time_per_candidate']:<18.4f} {result['results_returned']:<8}")
    
    # Recommendations
    print("\nRecommendations:")
    print("-" * 50)
    
    if results[-1]['processing_time'] > 30:
        print("⚠️  Performance Issue: Processing time is too high")
        print("   - Consider reducing candidate limit to 200 or less")
        print("   - Enable caching for repeated queries")
        print("   - Use faster embedding models")
    else:
        print("✅ Performance is acceptable")
    
    if results[-1]['avg_time_per_candidate'] > 0.1:
        print("⚠️  Per-candidate processing is slow")
        print("   - Consider optimizing similarity calculations")
        print("   - Use batch processing for embeddings")
    else:
        print("✅ Per-candidate processing is efficient")

if __name__ == "__main__":
    test_performance()
