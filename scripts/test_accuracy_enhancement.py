#!/usr/bin/env python3
"""
Test script for the accuracy enhancement system
"""

import sys
import os
import time
import json

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_accuracy_enhancement():
    """Test the accuracy enhancement system"""
    print("Testing Accuracy Enhancement System")
    print("=" * 50)
    
    try:
        # Import the accuracy enhancement system
        from app.search.accuracy_enhancement_system import (
            get_accuracy_enhancement_system,
            enhance_search_accuracy
        )
        
        print("Successfully imported accuracy enhancement system")
        
        # Test data
        test_query = "Senior Python Developer with React experience, 5+ years experience, AWS knowledge required"
        
        test_candidates = [
            {
                'id': '1',
                'name': 'John Doe',
                'title': 'Senior Python Developer',
                'skills': ['Python', 'Django', 'React', 'AWS', 'Docker'],
                'experience_years': 6,
                'resume_text': 'Experienced Python developer with 6 years of experience in Django, React, and AWS. Strong background in full-stack development.',
                'match_percentage': 75.0
            },
            {
                'id': '2',
                'name': 'Jane Smith',
                'title': 'Full Stack Developer',
                'skills': ['JavaScript', 'Node.js', 'React', 'MongoDB'],
                'experience_years': 4,
                'resume_text': 'Full stack developer with 4 years experience in JavaScript, Node.js, and React. Some Python knowledge.',
                'match_percentage': 60.0
            },
            {
                'id': '3',
                'name': 'Bob Johnson',
                'title': 'Python Developer',
                'skills': ['Python', 'Flask', 'SQL', 'Git'],
                'experience_years': 3,
                'resume_text': 'Python developer with 3 years experience in Flask and SQL. Looking to learn React and AWS.',
                'match_percentage': 45.0
            },
            {
                'id': '4',
                'name': 'Alice Brown',
                'title': 'Senior Software Engineer',
                'skills': ['Java', 'Spring', 'Angular', 'AWS', 'Docker', 'Kubernetes'],
                'experience_years': 8,
                'resume_text': 'Senior software engineer with 8 years experience in Java, Spring, and cloud technologies. Some Python experience.',
                'match_percentage': 55.0
            },
            {
                'id': '5',
                'name': 'Charlie Wilson',
                'title': 'Python Backend Developer',
                'skills': ['Python', 'FastAPI', 'PostgreSQL', 'Redis', 'AWS'],
                'experience_years': 5,
                'resume_text': 'Python backend developer with 5 years experience in FastAPI, PostgreSQL, and AWS. No frontend experience.',
                'match_percentage': 70.0
            }
        ]
        
        print(f"\nTest Query: {test_query}")
        print(f"Test Candidates: {len(test_candidates)}")
        
        # Test accuracy enhancement
        print("\nApplying accuracy enhancement...")
        start_time = time.time()
        
        enhanced_results = enhance_search_accuracy(test_query, test_candidates, top_k=5)
        
        processing_time = time.time() - start_time
        
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Calculate improvements
        original_scores = [c.get('match_percentage', 0) for c in test_candidates]
        enhanced_scores = [c.get('accuracy_score', 0) for c in enhanced_results]
        
        # If no accuracy_score, use enhanced_semantic_score as fallback
        if not any(enhanced_scores):
            enhanced_scores = [c.get('enhanced_semantic_score', 0) for c in enhanced_results]
        
        avg_original = sum(original_scores) / len(original_scores)
        avg_enhanced = sum(enhanced_scores) / len(enhanced_scores)
        improvement = avg_enhanced - avg_original
        
        print(f"\nResults:")
        print(f"   Average Original Score: {avg_original:.1f}%")
        print(f"   Average Enhanced Score: {avg_enhanced:.1f}%")
        print(f"   Accuracy Improvement: {improvement:+.1f}%")
        
        # Display enhanced results
        print(f"\nEnhanced Results (Top {len(enhanced_results)}):")
        print("-" * 80)
        
        for i, candidate in enumerate(enhanced_results, 1):
            original_score = candidate.get('original_score', candidate.get('match_percentage', 0))
            enhanced_score = candidate.get('accuracy_score', 0)
            
            # If no accuracy_score, use enhanced_semantic_score
            if enhanced_score == 0:
                enhanced_score = candidate.get('enhanced_semantic_score', 0)
            
            improvement = enhanced_score - original_score
            
            print(f"{i}. {candidate.get('name', 'Unknown')} - {candidate.get('title', 'No Title')}")
            print(f"   Skills: {', '.join(candidate.get('skills', []))}")
            print(f"   Experience: {candidate.get('experience_years', 0)} years")
            print(f"   Original Score: {original_score:.1f}%")
            print(f"   Enhanced Score: {enhanced_score:.1f}%")
            print(f"   Improvement: {improvement:+.1f}%")
            print()
        
        # Test individual components
        print("Testing individual components...")
        
        # Test semantic matcher
        from app.search.accuracy_enhancement_system import AdvancedSemanticMatcher
        semantic_matcher = AdvancedSemanticMatcher()
        
        test_text1 = "Senior Python Developer with React experience"
        test_text2 = "Python developer with Django and React skills"
        
        similarity = semantic_matcher.calculate_advanced_similarity(test_text1, test_text2)
        print(f"   Semantic Similarity Test: {similarity:.2f}")
        
        # Test query expander
        from app.search.accuracy_enhancement_system import IntelligentQueryExpander
        query_expander = IntelligentQueryExpander()
        
        expanded_query = query_expander.expand_query("Python developer")
        print(f"   Query Expansion Test: {expanded_query[:100]}...")
        
        # Test result reranker
        from app.search.accuracy_enhancement_system import AdvancedResultReranker
        result_reranker = AdvancedResultReranker()
        
        reranked_results = result_reranker.rerank_results(test_candidates, test_query)
        print(f"   Result Re-ranking Test: {len(reranked_results)} candidates reranked")
        
        print("\nAll tests completed successfully!")
        
        # Get performance stats
        accuracy_system = get_accuracy_enhancement_system()
        stats = accuracy_system.get_performance_stats()
        
        print(f"\nPerformance Stats:")
        print(f"   Total Queries: {stats.get('total_queries', 0)}")
        print(f"   Avg Accuracy Improvement: {stats.get('avg_accuracy_improvement', 0):.1f}%")
        print(f"   Processing Time: {stats.get('processing_time', 0):.2f}s")
        
        return True
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("   Make sure all required dependencies are installed:")
        print("   pip install sentence-transformers scikit-learn jaro-winkler")
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoint():
    """Test the accuracy enhancement API endpoint"""
    print("\nTesting API Endpoint")
    print("=" * 30)
    
    try:
        import requests
        
        # Test data
        test_data = {
            'query': 'Senior Python Developer with React experience',
            'candidates': [
                {
                    'id': '1',
                    'name': 'John Doe',
                    'title': 'Senior Python Developer',
                    'skills': ['Python', 'Django', 'React', 'AWS'],
                    'experience_years': 6,
                    'match_percentage': 75.0
                }
            ]
        }
        
        # Make API request
        response = requests.post(
            'http://localhost:5000/api/search/test-accuracy',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("API endpoint working correctly")
            print(f"   Accuracy Improvement: {result.get('accuracy_improvement', 0):+.1f}%")
            print(f"   Processing Time: {result.get('processing_time', 0):.2f}s")
            return True
        else:
            print(f"API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("API server not running. Start the server to test the endpoint.")
        return False
    except Exception as e:
        print(f"API Test Error: {e}")
        return False

if __name__ == "__main__":
    print("Accuracy Enhancement System Test Suite")
    print("=" * 50)
    
    # Test the system
    system_test_passed = test_accuracy_enhancement()
    
    # Test the API endpoint
    api_test_passed = test_api_endpoint()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"   System Test: {'PASSED' if system_test_passed else 'FAILED'}")
    print(f"   API Test: {'PASSED' if api_test_passed else 'FAILED'}")
    
    if system_test_passed and api_test_passed:
        print("\nAll tests passed! Accuracy enhancement system is working correctly.")
    else:
        print("\nSome tests failed. Check the errors above.")
    
    print("\nTo improve accuracy further:")
    print("   1. Add more industry-specific knowledge")
    print("   2. Fine-tune the scoring weights")
    print("   3. Add more sophisticated NLP techniques")
    print("   4. Implement machine learning-based ranking")
