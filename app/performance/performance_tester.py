# Performance Tester for 1000+ Candidates
# Comprehensive testing suite for backend performance optimization

import os
import time
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from app.simple_logger import get_logger
from app.models import db, CandidateProfile, CandidateSkill
from .optimized_candidate_handler import candidate_handler
from .optimized_search_system import search_system

logger = get_logger("performance_tester")

class PerformanceTester:
    """Comprehensive performance testing for 1000+ candidates"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {}
        self.auth_token = None
    
    def set_auth_token(self, token: str):
        """Set authentication token for API tests"""
        self.auth_token = token
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers
    
    def test_candidate_retrieval_performance(self, 
                                           page_sizes: List[int] = [10, 50, 100],
                                           total_pages: int = 10) -> Dict[str, Any]:
        """Test candidate retrieval performance with different page sizes"""
        logger.info("Testing candidate retrieval performance...")
        
        results = {}
        
        for page_size in page_sizes:
            logger.info(f"Testing with page size: {page_size}")
            
            times = []
            for page in range(1, min(total_pages + 1, 11)):  # Test up to 10 pages
                start_time = time.time()
                
                try:
                    # Test the optimized endpoint
                    response = requests.get(
                        f"{self.base_url}/performance/candidates/optimized",
                        params={
                            'page': page,
                            'per_page': page_size,
                            'sort_by': 'created_at',
                            'sort_order': 'desc'
                        },
                        headers=self.get_headers(),
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        end_time = time.time()
                        response_time = end_time - start_time
                        times.append(response_time)
                        
                        data = response.json()
                        logger.info(f"Page {page}: {response_time:.2f}s, {len(data.get('candidates', []))} candidates")
                    else:
                        logger.error(f"API error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    logger.error(f"Error testing page {page}: {str(e)}")
            
            if times:
                results[f"page_size_{page_size}"] = {
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_requests': len(times),
                    'candidates_per_second': page_size / (sum(times) / len(times))
                }
        
        self.test_results['candidate_retrieval'] = results
        return results
    
    def test_search_performance(self, 
                               queries: List[str] = None,
                               top_k_values: List[int] = [10, 20, 50]) -> Dict[str, Any]:
        """Test search performance with different queries and result counts"""
        logger.info("Testing search performance...")
        
        if queries is None:
            queries = [
                "Python developer",
                "Senior software engineer",
                "Data scientist with machine learning",
                "Full stack developer React Node.js",
                "DevOps engineer AWS Docker"
            ]
        
        results = {}
        
        for query in queries:
            logger.info(f"Testing search query: '{query}'")
            
            query_results = {}
            for top_k in top_k_values:
                times = []
                
                # Test multiple times for average
                for i in range(3):
                    start_time = time.time()
                    
                    try:
                        response = requests.post(
                            f"{self.base_url}/performance/candidates/search",
                            json={
                                'query': query,
                                'top_k': top_k,
                                'use_cache': True
                            },
                            headers=self.get_headers(),
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            end_time = time.time()
                            response_time = end_time - start_time
                            times.append(response_time)
                            
                            data = response.json()
                            logger.info(f"Query '{query}' (top_k={top_k}): {response_time:.2f}s, {len(data.get('results', []))} results")
                        else:
                            logger.error(f"Search API error: {response.status_code} - {response.text}")
                            
                    except Exception as e:
                        logger.error(f"Error testing search: {str(e)}")
                
                if times:
                    query_results[f"top_k_{top_k}"] = {
                        'average_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'total_requests': len(times)
                    }
            
            results[query] = query_results
        
        self.test_results['search_performance'] = results
        return results
    
    def test_batch_processing_performance(self, 
                                        batch_sizes: List[int] = [10, 50, 100, 200]) -> Dict[str, Any]:
        """Test batch processing performance"""
        logger.info("Testing batch processing performance...")
        
        # First, get some candidate IDs for testing
        try:
            response = requests.get(
                f"{self.base_url}/performance/candidates/optimized",
                params={'page': 1, 'per_page': 1000},
                headers=self.get_headers(),
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Could not get candidate IDs: {response.status_code}")
                return {}
            
            data = response.json()
            candidate_ids = [c['id'] for c in data.get('candidates', [])]
            
            if not candidate_ids:
                logger.warning("No candidates found for batch processing test")
                return {}
            
        except Exception as e:
            logger.error(f"Error getting candidate IDs: {str(e)}")
            return {}
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch processing with batch size: {batch_size}")
            
            # Test with different batch sizes
            test_ids = candidate_ids[:min(batch_size * 2, len(candidate_ids))]
            
            times = []
            for i in range(0, len(test_ids), batch_size):
                batch = test_ids[i:i + batch_size]
                
                start_time = time.time()
                
                try:
                    response = requests.post(
                        f"{self.base_url}/performance/candidates/batch-process",
                        json={
                            'candidate_ids': batch,
                            'operation': 'export'
                        },
                        headers=self.get_headers(),
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        end_time = time.time()
                        response_time = end_time - start_time
                        times.append(response_time)
                        
                        data = response.json()
                        logger.info(f"Batch {len(batch)} candidates: {response_time:.2f}s")
                    else:
                        logger.error(f"Batch processing API error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    logger.error(f"Error testing batch processing: {str(e)}")
            
            if times:
                results[f"batch_size_{batch_size}"] = {
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_batches': len(times),
                    'candidates_per_second': batch_size / (sum(times) / len(times))
                }
        
        self.test_results['batch_processing'] = results
        return results
    
    def test_concurrent_requests(self, 
                               num_requests: int = 10,
                               request_type: str = 'candidates') -> Dict[str, Any]:
        """Test concurrent request handling"""
        logger.info(f"Testing {num_requests} concurrent {request_type} requests...")
        
        def make_request(request_id: int) -> Dict[str, Any]:
            start_time = time.time()
            
            try:
                if request_type == 'candidates':
                    response = requests.get(
                        f"{self.base_url}/performance/candidates/optimized",
                        params={'page': 1, 'per_page': 50},
                        headers=self.get_headers(),
                        timeout=30
                    )
                elif request_type == 'search':
                    response = requests.post(
                        f"{self.base_url}/performance/candidates/search",
                        json={'query': 'Python developer', 'top_k': 20},
                        headers=self.get_headers(),
                        timeout=30
                    )
                else:
                    return {'error': 'Invalid request type'}
                
                end_time = time.time()
                response_time = end_time - start_time
                
                return {
                    'request_id': request_id,
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'success': response.status_code == 200
                }
                
            except Exception as e:
                return {
                    'request_id': request_id,
                    'error': str(e),
                    'success': False
                }
        
        # Execute concurrent requests
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=min(num_requests, 20)) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r.get('success', False)]
        failed_requests = [r for r in results if not r.get('success', False)]
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0
        
        concurrent_results = {
            'total_requests': num_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / num_requests,
            'total_time': total_time,
            'requests_per_second': num_requests / total_time,
            'average_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'failed_request_details': failed_requests
        }
        
        self.test_results['concurrent_requests'] = concurrent_results
        return concurrent_results
    
    def test_database_performance(self) -> Dict[str, Any]:
        """Test database performance directly"""
        logger.info("Testing database performance...")
        
        results = {}
        
        # Test 1: Count all candidates
        start_time = time.time()
        total_candidates = db.session.query(CandidateProfile).count()
        count_time = time.time() - start_time
        
        results['count_all_candidates'] = {
            'time': count_time,
            'count': total_candidates
        }
        
        # Test 2: Complex query with joins
        start_time = time.time()
        candidates_with_skills = db.session.query(CandidateProfile).join(
            CandidateSkill
        ).distinct().count()
        join_time = time.time() - start_time
        
        results['candidates_with_skills'] = {
            'time': join_time,
            'count': candidates_with_skills
        }
        
        # Test 3: Pagination query
        start_time = time.time()
        paginated_candidates = db.session.query(CandidateProfile).offset(0).limit(50).all()
        pagination_time = time.time() - start_time
        
        results['pagination_query'] = {
            'time': pagination_time,
            'count': len(paginated_candidates)
        }
        
        # Test 4: Search query
        start_time = time.time()
        search_candidates = db.session.query(CandidateProfile).filter(
            CandidateProfile.full_name.ilike('%Python%')
        ).limit(20).all()
        search_time = time.time() - start_time
        
        results['search_query'] = {
            'time': search_time,
            'count': len(search_candidates)
        }
        
        self.test_results['database_performance'] = results
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all performance tests"""
        logger.info("Running comprehensive performance test...")
        
        start_time = time.time()
        
        # Run all tests
        self.test_candidate_retrieval_performance()
        self.test_search_performance()
        self.test_batch_processing_performance()
        self.test_concurrent_requests()
        self.test_database_performance()
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_test_summary(total_time)
        
        self.test_results['summary'] = summary
        self.test_results['total_test_time'] = total_time
        
        logger.info(f"Comprehensive test completed in {total_time:.2f}s")
        
        return self.test_results
    
    def _generate_test_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate test summary and recommendations"""
        summary = {
            'total_test_time': total_time,
            'recommendations': [],
            'performance_score': 0
        }
        
        # Analyze candidate retrieval performance
        if 'candidate_retrieval' in self.test_results:
            retrieval_results = self.test_results['candidate_retrieval']
            avg_times = [r['average_time'] for r in retrieval_results.values()]
            
            if avg_times:
                avg_retrieval_time = sum(avg_times) / len(avg_times)
                if avg_retrieval_time > 2.0:
                    summary['recommendations'].append("Candidate retrieval is slow - consider adding database indexes")
                elif avg_retrieval_time < 0.5:
                    summary['recommendations'].append("Candidate retrieval performance is excellent")
        
        # Analyze search performance
        if 'search_performance' in self.test_results:
            search_results = self.test_results['search_performance']
            all_search_times = []
            
            for query_results in search_results.values():
                for result in query_results.values():
                    all_search_times.append(result['average_time'])
            
            if all_search_times:
                avg_search_time = sum(all_search_times) / len(all_search_times)
                if avg_search_time > 1.0:
                    summary['recommendations'].append("Search performance is slow - consider rebuilding search index")
                elif avg_search_time < 0.3:
                    summary['recommendations'].append("Search performance is excellent")
        
        # Analyze concurrent request performance
        if 'concurrent_requests' in self.test_results:
            concurrent_results = self.test_results['concurrent_requests']
            success_rate = concurrent_results['success_rate']
            
            if success_rate < 0.9:
                summary['recommendations'].append("Concurrent request handling needs improvement - consider increasing server resources")
            elif success_rate >= 0.95:
                summary['recommendations'].append("Concurrent request handling is excellent")
        
        # Calculate performance score
        score = 100
        if 'candidate_retrieval' in self.test_results:
            avg_times = [r['average_time'] for r in self.test_results['candidate_retrieval'].values()]
            if avg_times and sum(avg_times) / len(avg_times) > 1.0:
                score -= 20
        
        if 'concurrent_requests' in self.test_results:
            success_rate = self.test_results['concurrent_requests']['success_rate']
            if success_rate < 0.9:
                score -= 30
            elif success_rate < 0.95:
                score -= 15
        
        summary['performance_score'] = max(0, score)
        
        return summary
    
    def save_test_results(self, filename: str = None) -> str:
        """Save test results to file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"performance_test_results_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {filepath}")
        return filepath

# Example usage
if __name__ == "__main__":
    tester = PerformanceTester()
    
    # Set auth token if needed
    # tester.set_auth_token("your_auth_token_here")
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Save results
    tester.save_test_results()
    
    print("Performance test completed!")
    print(f"Performance score: {results['summary']['performance_score']}/100")
    print("Recommendations:")
    for rec in results['summary']['recommendations']:
        print(f"- {rec}")
