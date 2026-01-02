"""
Test script for candidate search history API endpoints
"""
import requests
import json
from datetime import datetime

# Test data
test_job_description = """
Senior Software Engineer - Full Stack Development

We are looking for an experienced software engineer to join our team. 
The ideal candidate should have:
- 5+ years of experience in full-stack development
- Proficiency in React, Node.js, and Python
- Experience with cloud platforms (AWS, Azure, or GCP)
- Strong problem-solving skills
- Excellent communication skills
"""

test_candidates = [
    {
        "id": "candidate_001",
        "name": "John Smith",
        "email": "john.smith@email.com",
        "phone": "+1-555-0123",
        "location": "San Francisco, CA",
        "match_score": 0.95,
        "data": {
            "experience": "6 years",
            "skills": ["React", "Node.js", "Python", "AWS"],
            "education": "Computer Science Degree"
        }
    },
    {
        "id": "candidate_002", 
        "name": "Sarah Johnson",
        "email": "sarah.johnson@email.com",
        "phone": "+1-555-0124",
        "location": "New York, NY",
        "match_score": 0.88,
        "data": {
            "experience": "4 years",
            "skills": ["React", "Python", "Docker"],
            "education": "Software Engineering Degree"
        }
    },
    {
        "id": "candidate_003",
        "name": "Mike Chen",
        "email": "mike.chen@email.com", 
        "phone": "+1-555-0125",
        "location": "Seattle, WA",
        "match_score": 0.82,
        "data": {
            "experience": "7 years",
            "skills": ["Node.js", "Python", "Azure"],
            "education": "Computer Science Masters"
        }
    }
]

def test_save_search_history():
    """Test saving a search to history"""
    print("Testing save search history...")
    
    data = {
        "job_description": test_job_description,
        "candidates": test_candidates,
        "search_criteria": {
            "location": "Remote",
            "experience_level": "Senior",
            "skills": ["React", "Python"]
        },
        "candidates_found": len(test_candidates),
        "search_duration_ms": 2500,
        "search_status": "completed"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/candidate-search-history",
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 201:
            result = response.json()
            print(f"SUCCESS: Search saved with ID {result.get('search_id')}")
            return result.get('search_id')
        else:
            print("FAILED: Could not save search history")
            return None
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None

def test_get_search_history():
    """Test retrieving search history"""
    print("\nTesting get search history...")
    
    try:
        response = requests.get(
            "http://localhost:8000/api/candidate-search-history",
            params={'limit': 10, 'include_expired': 'false'}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            searches = data.get('searches', [])
            print(f"SUCCESS: Retrieved {len(searches)} searches")
            
            for search in searches:
                print(f"  - Search ID: {search['id']}, Candidates: {search['candidates_found']}")
            
            return searches
        else:
            print(f"FAILED: {response.text}")
            return []
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return []

def test_get_search_stats():
    """Test retrieving search statistics"""
    print("\nTesting get search stats...")
    
    try:
        response = requests.get(
            "http://localhost:8000/api/candidate-search-history/stats"
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            print("SUCCESS: Retrieved search statistics")
            print(f"  - Total searches: {stats.get('total_searches', 0)}")
            print(f"  - Active searches: {stats.get('active_searches', 0)}")
            print(f"  - Total candidates: {stats.get('total_candidates', 0)}")
            print(f"  - Saved candidates: {stats.get('saved_candidates', 0)}")
            print(f"  - Contacted candidates: {stats.get('contacted_candidates', 0)}")
            return stats
        else:
            print(f"FAILED: {response.text}")
            return {}
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {}

def test_extend_search(search_id):
    """Test extending search expiry"""
    if not search_id:
        print("\nSkipping extend search test - no search ID available")
        return
        
    print(f"\nTesting extend search expiry for ID {search_id}...")
    
    data = {"days": 10}
    
    try:
        response = requests.post(
            f"http://localhost:8000/api/candidate-search-history/{search_id}/extend",
            json=data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"SUCCESS: Search extended, new expiry: {result.get('new_expiry')}")
        else:
            print(f"FAILED: {response.text}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    print("=== Candidate Search History API Test ===")
    print(f"Test started at: {datetime.now()}")
    
    # Test saving search history
    search_id = test_save_search_history()
    
    # Test retrieving search history
    searches = test_get_search_history()
    
    # Test getting stats
    stats = test_get_search_stats()
    
    # Test extending search
    test_extend_search(search_id)
    
    print(f"\n=== Test completed at: {datetime.now()} ===")
