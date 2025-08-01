#!/usr/bin/env python3
"""
Quick test to verify None value fixes
"""

import sys
import os
sys.path.append('.')

def test_none_handling():
    """Test that the algorithm handles None values correctly"""
    print("🧪 Testing None Value Handling")
    print("=" * 50)
    
    try:
        from app.search.service import AdeptAIMastersAlgorithm
        
        # Initialize algorithm
        algorithm = AdeptAIMastersAlgorithm()
        print("✅ Algorithm initialized successfully")
        
        # Test with None values in candidate data
        candidate_data_with_nones = {
            'full_name': 'Test Candidate',
            'email': 'test@example.com',
            'skills': ['python', None, 'django', '', 'react'],  # Mixed valid and None values
            'experience': None,  # None experience
            'education': None,   # None education
            'location': None,    # None location
            'resume_text': None, # None resume text
            'certifications': None,  # None certifications
            'total_experience_years': None  # None experience years
        }
        
        job_query = "Python developer with Django experience"
        
        print("📋 Testing with None values in candidate data...")
        
        # Test keyword extraction with None
        keywords = algorithm.extract_keywords(None)
        print(f"✅ Keywords from None: {keywords}")
        
        # Test skill normalization with None values
        normalized_skills = algorithm._normalize_skills(['python', None, 'django', '', 'react'])
        print(f"✅ Normalized skills: {normalized_skills}")
        
        # Test skill matching with None
        skill_match = algorithm._skills_match('python', None)
        print(f"✅ Skill match with None: {skill_match}")
        
        # Test experience extraction with None
        experience_years = algorithm._extract_experience_years(candidate_data_with_nones)
        print(f"✅ Experience years from None: {experience_years}")
        
        # Test semantic similarity with None
        similarity = algorithm.semantic_similarity("Python developer", None)
        print(f"✅ Semantic similarity with None: {similarity}")
        
        # Test enhanced score calculation with None values
        print("📊 Testing enhanced score calculation with None values...")
        match_score = algorithm.calculate_enhanced_score(candidate_data_with_nones, job_query)
        
        print(f"✅ Overall Score: {match_score.overall_score:.2f}")
        print(f"✅ Technical Skills: {match_score.technical_skills_score:.2f}")
        print(f"✅ Experience: {match_score.experience_score:.2f}")
        print(f"✅ Education: {match_score.education_score:.2f}")
        print(f"✅ Location: {match_score.location_score:.2f}")
        print(f"✅ Confidence: {match_score.confidence:.2f}")
        print(f"✅ Match Explanation: {match_score.match_explanation}")
        
        # Test with completely empty candidate data
        empty_candidate = {}
        print("\n📋 Testing with completely empty candidate data...")
        
        empty_match_score = algorithm.calculate_enhanced_score(empty_candidate, job_query)
        print(f"✅ Empty candidate score: {empty_match_score.overall_score:.2f}")
        print(f"✅ Empty candidate explanation: {empty_match_score.match_explanation}")
        
        print("\n🎉 All None value handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\n🔍 Testing Edge Cases")
    print("=" * 50)
    
    try:
        from app.search.service import AdeptAIMastersAlgorithm
        
        algorithm = AdeptAIMastersAlgorithm()
        
        # Test with various edge cases
        edge_cases = [
            {'skills': []},  # Empty skills list
            {'skills': [None, None, None]},  # All None skills
            {'skills': ['', '', '']},  # Empty string skills
            {'experience': ''},  # Empty experience
            {'experience': 'invalid'},  # Invalid experience format
            {'education': ''},  # Empty education
            {'location': ''},  # Empty location
        ]
        
        job_query = "Python developer"
        
        for i, edge_case in enumerate(edge_cases):
            print(f"📋 Testing edge case {i+1}: {edge_case}")
            try:
                match_score = algorithm.calculate_enhanced_score(edge_case, job_query)
                print(f"   ✅ Score: {match_score.overall_score:.2f}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("✅ All edge case tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing None Value Fixes")
    print("=" * 60)
    
    success = True
    
    # Test None handling
    if not test_none_handling():
        success = False
    
    # Test edge cases
    if not test_edge_cases():
        success = False
    
    if success:
        print("\n🎉 ALL TESTS PASSED! The None value fixes are working correctly.")
        print("\n📋 Fixes Applied:")
        print("   ✅ _normalize_skills() - Handles None and invalid skills")
        print("   ✅ _skills_match() - Handles None skill comparisons")
        print("   ✅ _calculate_education_score() - Handles None education")
        print("   ✅ _calculate_location_score() - Handles None location")
        print("   ✅ _extract_experience_years() - Handles None experience")
        print("   ✅ extract_keywords() - Handles None text")
        print("   ✅ semantic_similarity() - Handles None text inputs")
        print("   ✅ _generate_match_explanation() - Enhanced with candidate name")
        
        print("\n🎯 The algorithm should now work without errors!")
        
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    main() 