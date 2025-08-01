#!/usr/bin/env python3
"""
Debug script to identify None value errors
"""

import sys
import os
import traceback
sys.path.append('.')

def debug_none_errors():
    """Debug None value errors in the algorithm"""
    print("🔍 Debugging None Value Errors")
    print("=" * 50)
    
    try:
        from app.search.service import AdeptAIMastersAlgorithm
        
        # Initialize algorithm
        algorithm = AdeptAIMastersAlgorithm()
        print("✅ Algorithm initialized successfully")
        
        # Test with a sample job description
        job_description = "Python developer with Django experience"
        
        # Test the complete flow step by step
        print("\n📋 Testing step-by-step flow...")
        
        # Step 1: Test extract_keywords
        print("1. Testing extract_keywords...")
        try:
            keywords = algorithm.extract_keywords(job_description)
            print(f"   ✅ Keywords: {keywords}")
        except Exception as e:
            print(f"   ❌ Error in extract_keywords: {e}")
            traceback.print_exc()
        
        # Step 2: Test detect_domain
        print("2. Testing detect_domain...")
        try:
            domain = algorithm.detect_domain(keywords)
            print(f"   ✅ Domain: {domain}")
        except Exception as e:
            print(f"   ❌ Error in detect_domain: {e}")
            traceback.print_exc()
        
        # Step 3: Test query parsing
        print("3. Testing query parsing...")
        try:
            parsed_query = algorithm.query_parser.parse_job_query(job_description)
            print(f"   ✅ Parsed query: {parsed_query}")
        except Exception as e:
            print(f"   ❌ Error in query parsing: {e}")
            traceback.print_exc()
        
        # Step 4: Test with sample candidate data
        print("4. Testing with sample candidate data...")
        sample_candidate = {
            'full_name': 'Test Candidate',
            'email': 'test@example.com',
            'skills': ['python', 'django', 'react'],
            'experience': '5 years',
            'education': 'Bachelor of Computer Science',
            'location': 'New York',
            'resume_text': 'Experienced Python developer with Django and React skills.',
            'certifications': ['AWS Certified Developer'],
            'total_experience_years': 5
        }
        
        try:
            match_score = algorithm.calculate_enhanced_score(sample_candidate, job_description)
            print(f"   ✅ Match score calculated: {match_score.overall_score:.2f}")
        except Exception as e:
            print(f"   ❌ Error in calculate_enhanced_score: {e}")
            traceback.print_exc()
        
        # Step 5: Test with None values
        print("5. Testing with None values...")
        none_candidate = {
            'full_name': None,
            'email': None,
            'skills': [None, 'python', None, 'django'],
            'experience': None,
            'education': None,
            'location': None,
            'resume_text': None,
            'certifications': None,
            'total_experience_years': None
        }
        
        try:
            none_match_score = algorithm.calculate_enhanced_score(none_candidate, job_description)
            print(f"   ✅ None candidate score: {none_match_score.overall_score:.2f}")
        except Exception as e:
            print(f"   ❌ Error with None candidate: {e}")
            traceback.print_exc()
        
        # Step 6: Test keyword search
        print("6. Testing keyword search...")
        try:
            results, summary = algorithm.keyword_search(job_description, top_k=5)
            print(f"   ✅ Keyword search results: {len(results)} candidates")
            print(f"   ✅ Summary: {summary}")
        except Exception as e:
            print(f"   ❌ Error in keyword search: {e}")
            traceback.print_exc()
        
        print("\n🎉 Debug completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

def test_specific_methods():
    """Test specific methods that might have None issues"""
    print("\n🔧 Testing Specific Methods")
    print("=" * 50)
    
    try:
        from app.search.service import AdeptAIMastersAlgorithm
        
        algorithm = AdeptAIMastersAlgorithm()
        
        # Test _normalize_skills with None values
        print("1. Testing _normalize_skills with None...")
        try:
            skills_with_none = ['python', None, 'django', '', 'react']
            normalized = algorithm._normalize_skills(skills_with_none)
            print(f"   ✅ Normalized skills: {normalized}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            traceback.print_exc()
        
        # Test _skills_match with None
        print("2. Testing _skills_match with None...")
        try:
            match1 = algorithm._skills_match('python', None)
            match2 = algorithm._skills_match(None, 'python')
            match3 = algorithm._skills_match(None, None)
            print(f"   ✅ Matches: {match1}, {match2}, {match3}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            traceback.print_exc()
        
        # Test extract_keywords with None
        print("3. Testing extract_keywords with None...")
        try:
            keywords_none = algorithm.extract_keywords(None)
            keywords_empty = algorithm.extract_keywords("")
            print(f"   ✅ Keywords from None: {keywords_none}")
            print(f"   ✅ Keywords from empty: {keywords_empty}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            traceback.print_exc()
        
        # Test semantic_similarity with None
        print("4. Testing semantic_similarity with None...")
        try:
            sim1 = algorithm.semantic_similarity("Python", None)
            sim2 = algorithm.semantic_similarity(None, "Python")
            sim3 = algorithm.semantic_similarity(None, None)
            print(f"   ✅ Similarities: {sim1:.3f}, {sim2:.3f}, {sim3:.3f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            traceback.print_exc()
        
        print("✅ All specific method tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("🚀 Debugging None Value Errors in AdeptAI Masters Algorithm")
    print("=" * 70)
    
    success = True
    
    # Debug None errors
    if not debug_none_errors():
        success = False
    
    # Test specific methods
    if not test_specific_methods():
        success = False
    
    if success:
        print("\n🎉 All debug tests passed! No None value errors found.")
        print("\n📋 If you're still seeing errors, they might be coming from:")
        print("   • DynamoDB data with None values")
        print("   • External API calls")
        print("   • Other parts of the system")
        
    else:
        print("\n❌ Some debug tests failed. Check the errors above.")
    
    return success

if __name__ == "__main__":
    main() 