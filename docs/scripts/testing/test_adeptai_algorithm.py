#!/usr/bin/env python3
"""
Test script for AdeptAI Masters Algorithm
This script tests all the major functionalities of the algorithm
"""

import os
import sys
import json
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the algorithm
from app.search.service import AdeptAIMastersAlgorithm, semantic_match, keyword_search, register_feedback

def test_keyword_extraction():
    """Test keyword extraction functionality"""
    print("🔍 Testing Keyword Extraction...")
    
    algorithm = AdeptAIMastersAlgorithm()
    
    # Test software job description
    software_jd = """
    We are looking for a Senior Python Developer with 5+ years of experience.
    Skills required: Python, Django, React, AWS, Docker, Kubernetes, SQL, NoSQL.
    Experience with machine learning, API development, and microservices architecture.
    """
    
    keywords = algorithm.extract_keywords(software_jd)
    print(f"Software JD Keywords: {keywords}")
    
    # Test healthcare job description
    healthcare_jd = """
    Registered Nurse position in ICU department.
    Requirements: BSN, RN license, ICU experience, patient care, medication administration.
    Skills: patient assessment, clinical documentation, emergency response.
    """
    
    keywords = algorithm.extract_keywords(healthcare_jd)
    print(f"Healthcare JD Keywords: {keywords}")
    
    print("✅ Keyword extraction test completed\n")

def test_domain_detection():
    """Test domain detection functionality"""
    print("🎯 Testing Domain Detection...")
    
    algorithm = AdeptAIMastersAlgorithm()
    
    # Test software keywords
    software_keywords = ['python', 'django', 'react', 'aws', 'docker']
    domain = algorithm.detect_domain(software_keywords)
    print(f"Software keywords domain: {domain}")
    
    # Test healthcare keywords
    healthcare_keywords = ['nurse', 'patient', 'clinical', 'medication', 'icu']
    domain = algorithm.detect_domain(healthcare_keywords)
    print(f"Healthcare keywords domain: {domain}")
    
    print("✅ Domain detection test completed\n")

def test_scoring():
    """Test scoring functionality"""
    print("📊 Testing Scoring System...")
    
    algorithm = AdeptAIMastersAlgorithm()
    
    # Test grade calculation
    scores = [95, 75, 55, 35]
    for score in scores:
        grade = algorithm.get_grade(score)
        print(f"Score {score} -> Grade {grade}")
    
    # Test NLRGA scoring
    matched_keywords = ['python', 'django', 'react']
    total_keywords = 5
    candidate_id = 'test_candidate@example.com'
    
    score = algorithm.nlrga_score(matched_keywords, total_keywords, candidate_id)
    print(f"NLRGA Score: {score}")
    
    print("✅ Scoring test completed\n")

def test_semantic_similarity():
    """Test semantic similarity functionality"""
    print("🧠 Testing Semantic Similarity...")
    
    algorithm = AdeptAIMastersAlgorithm()
    
    text1 = "Python developer with Django experience"
    text2 = "Software engineer skilled in Python web development"
    text3 = "Nurse with patient care experience"
    
    similarity1 = algorithm.semantic_similarity(text1, text2)
    similarity2 = algorithm.semantic_similarity(text1, text3)
    
    print(f"Similarity between software texts: {similarity1:.3f}")
    print(f"Similarity between software and healthcare: {similarity2:.3f}")
    
    print("✅ Semantic similarity test completed\n")

def test_feedback_system():
    """Test feedback system functionality"""
    print("🔄 Testing Feedback System...")
    
    candidate_id = 'test_candidate@example.com'
    
    # Register positive feedback
    register_feedback(candidate_id, positive=True)
    print(f"Registered positive feedback for {candidate_id}")
    
    # Register negative feedback
    register_feedback(candidate_id, positive=False)
    print(f"Registered negative feedback for {candidate_id}")
    
    # Test scoring with feedback
    algorithm = AdeptAIMastersAlgorithm()
    matched_keywords = ['python', 'django']
    total_keywords = 3
    
    score = algorithm.nlrga_score(matched_keywords, total_keywords, candidate_id)
    print(f"Score with feedback: {score}")
    
    print("✅ Feedback system test completed\n")

def test_keyword_search():
    """Test keyword search functionality"""
    print("🔎 Testing Keyword Search...")
    
    # Test software job search
    software_jd = """
    Senior Python Developer needed for web application development.
    Skills: Python, Django, React, AWS, Docker, SQL, API development.
    Experience with microservices and cloud deployment required.
    """
    
    try:
        results, summary = keyword_search(software_jd, top_k=5)
        print(f"Software search results: {len(results)} candidates found")
        print(f"Summary: {summary}")
        
        if results:
            print("Sample result:")
            sample = results[0]
            print(f"  Name: {sample.get('FullName', 'N/A')}")
            print(f"  Score: {sample.get('Score', 'N/A')}")
            print(f"  Grade: {sample.get('Grade', 'N/A')}")
            print(f"  Match %: {sample.get('MatchPercent', 'N/A')}")
        
    except Exception as e:
        print(f"Error in keyword search: {e}")
    
    print("✅ Keyword search test completed\n")

def test_semantic_match():
    """Test semantic matching functionality"""
    print("🎯 Testing Semantic Matching...")
    
    # Test software job search
    software_jd = """
    We need a Full Stack Developer with expertise in modern web technologies.
    Required skills: JavaScript, React, Node.js, Python, PostgreSQL, AWS.
    Experience with CI/CD, Docker, and agile methodologies preferred.
    """
    
    try:
        results = semantic_match(software_jd)
        print(f"Semantic match results: {len(results.get('results', []))} candidates found")
        print(f"Summary: {results.get('summary', 'N/A')}")
        
        if results.get('results'):
            print("Sample result:")
            sample = results['results'][0]
            print(f"  Name: {sample.get('FullName', 'N/A')}")
            print(f"  Score: {sample.get('Score', 'N/A')}")
            print(f"  Grade: {sample.get('Grade', 'N/A')}")
            print(f"  Semantic Score: {sample.get('SemanticScore', 'N/A')}")
        
    except Exception as e:
        print(f"Error in semantic match: {e}")
    
    print("✅ Semantic matching test completed\n")

def test_algorithm_initialization():
    """Test algorithm initialization"""
    print("🚀 Testing Algorithm Initialization...")
    
    try:
        algorithm = AdeptAIMastersAlgorithm()
        print("✅ Algorithm initialized successfully")
        
        # Test feedback manager
        feedback_manager = algorithm.feedback_manager
        print("✅ Feedback manager initialized")
        
        # Test model loading
        if hasattr(algorithm, 'semantic_cache'):
            print("✅ Semantic cache initialized")
        
    except Exception as e:
        print(f"❌ Error initializing algorithm: {e}")
    
    print("✅ Algorithm initialization test completed\n")

def main():
    """Run all tests"""
    print("🧪 Starting AdeptAI Masters Algorithm Tests")
    print("=" * 50)
    
    try:
        test_algorithm_initialization()
        test_keyword_extraction()
        test_domain_detection()
        test_scoring()
        test_semantic_similarity()
        test_feedback_system()
        test_keyword_search()
        test_semantic_match()
        
        print("🎉 All tests completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 