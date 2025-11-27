"""
Candidate Search History Service
Handles saving and retrieving candidate search history
"""
import requests
import json
from typing import List, Dict, Optional
from app.simple_logger import get_logger

logger = get_logger('candidate_search_service')

class CandidateSearchService:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def save_search_history(
        self, 
        job_description: str, 
        candidates: List[Dict], 
        search_criteria: Optional[Dict] = None,
        search_duration_ms: Optional[int] = None,
        search_status: str = 'completed'
    ) -> bool:
        """Save candidate search to history"""
        try:
            data = {
                'job_description': job_description,
                'candidates': candidates,
                'search_criteria': search_criteria or {},
                'candidates_found': len(candidates),
                'search_duration_ms': search_duration_ms,
                'search_status': search_status
            }
            
            response = requests.post(
                f"{self.base_url}/api/candidate-search-history",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 201:
                logger.info(f"Successfully saved search history with {len(candidates)} candidates")
                return True
            else:
                logger.error(f"Failed to save search history: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving search history: {str(e)}")
            return False
    
    def get_search_history(self, limit: int = 20, include_expired: bool = False) -> List[Dict]:
        """Get candidate search history"""
        try:
            params = {
                'limit': limit,
                'include_expired': str(include_expired).lower()
            }
            
            response = requests.get(
                f"{self.base_url}/api/candidate-search-history",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('searches', [])
            else:
                logger.error(f"Failed to get search history: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting search history: {str(e)}")
            return []
    
    def get_search_details(self, search_id: int) -> Optional[Dict]:
        """Get detailed information about a specific search"""
        try:
            response = requests.get(
                f"{self.base_url}/api/candidate-search-history/{search_id}"
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('search')
            else:
                logger.error(f"Failed to get search details: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting search details: {str(e)}")
            return None
    
    def extend_search_expiry(self, search_id: int, days: int = 10) -> bool:
        """Extend the expiry date of a search"""
        try:
            data = {'days': days}
            
            response = requests.post(
                f"{self.base_url}/api/candidate-search-history/{search_id}/extend",
                json=data
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully extended search {search_id} by {days} days")
                return True
            else:
                logger.error(f"Failed to extend search: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error extending search: {str(e)}")
            return False
    
    def save_candidate(self, search_id: int, candidate_id: int) -> bool:
        """Mark a candidate as saved"""
        try:
            response = requests.post(
                f"{self.base_url}/api/candidate-search-history/{search_id}/candidates/{candidate_id}/save"
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully saved candidate {candidate_id}")
                return True
            else:
                logger.error(f"Failed to save candidate: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving candidate: {str(e)}")
            return False
    
    def mark_candidate_contacted(self, search_id: int, candidate_id: int) -> bool:
        """Mark a candidate as contacted"""
        try:
            response = requests.post(
                f"{self.base_url}/api/candidate-search-history/{search_id}/candidates/{candidate_id}/contact"
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully marked candidate {candidate_id} as contacted")
                return True
            else:
                logger.error(f"Failed to mark candidate as contacted: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error marking candidate as contacted: {str(e)}")
            return False
    
    def get_search_stats(self) -> Dict:
        """Get search statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/api/candidate-search-history/stats"
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('stats', {})
            else:
                logger.error(f"Failed to get search stats: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting search stats: {str(e)}")
            return {}

# Global instance
candidate_search_service = CandidateSearchService()
