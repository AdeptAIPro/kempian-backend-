# domain_integration.py - Integration code for your existing system

import sys
import os
import time
import logging
from typing import List, Dict, Any, Tuple
from app.simple_logger import get_logger

# Import the domain-aware search system
from enhanced_domain_aware_search import DomainAwareSearchSystem, DomainClassifier, integrate_domain_awareness

logger = get_logger("search")

class DomainAwareEnhancedRecruitmentSearchSystem:
    """
    Wrapper for your existing Enhanced Recruitment Search System with domain awareness
    """
    
    def __init__(self, base_search_system):
        self.base_search_system = base_search_system
        self.domain_aware_system = integrate_domain_awareness(base_search_system)
        
        # Enable domain filtering by default
        self.domain_filtering_enabled = True
        
        logger.info("✅ Domain-Aware Enhanced Search System initialized")

    def search(self, query: str, top_k: int = 10, enable_domain_filtering: bool = None) -> List[Dict[str, Any]]:
        """
        Enhanced search with domain awareness
        """
        # Use instance setting if not specified
        if enable_domain_filtering is None:
            enable_domain_filtering = self.domain_filtering_enabled
        
        try:
            # Use domain-aware search
            results, summary, metadata = self.domain_aware_system.domain_aware_search(
                query, top_k, enable_domain_filtering
            )
            
            # Log domain filtering results
            if metadata.get('domain_filtering_applied'):
                logger.info(f"🎯 Domain filtering applied: {metadata['query_domain']} domain")
                logger.info(f"📊 Filtered out {metadata['candidates_filtered']} mismatched candidates")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Domain-aware search failed: {e}")
            # Fallback to base search
            logger.info("🔄 Falling back to base search system")
            return self.base_search_system.search(query, top_k)

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get domain classification statistics"""
        return self.domain_aware_system.get_domain_statistics()

    def toggle_domain_filtering(self, enabled: bool):
        """Enable or disable domain filtering"""
        self.domain_filtering_enabled = enabled
        logger.info(f"🔧 Domain filtering {'enabled' if enabled else 'disabled'}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance stats including domain filtering"""
        base_stats = self.base_search_system.get_performance_stats()
        domain_stats = self.get_domain_statistics()
        
        return {
            **base_stats,
            'domain_filtering_enabled': self.domain_filtering_enabled,
            'domain_statistics': domain_stats
        }

# Modified main.py integration - ADD THIS TO YOUR main.py

def initialize_domain_aware_enhanced_search():
    """
    Modified initialization function for main.py
    Replace your existing initialize_enhanced_search() with this
    """
    global enhanced_search_system
    
    with enhanced_search_lock:
        if enhanced_search_system is not None:
            logger.info("✅ Enhanced search system already initialized.")
            return True

        try:
            logger.info("🚀 Initializing Domain-Aware Enhanced Search System...")

            if table is None:
                logger.error("❌ DynamoDB table not available")
                return False

            # Create base enhanced search system first
            from enhanced_recruitment_search import create_ultra_fast_search_system
            success, base_enhanced_system = create_ultra_fast_search_system(table)

            if success and base_enhanced_system:
                # Wrap with domain awareness
                enhanced_search_system = DomainAwareEnhancedRecruitmentSearchSystem(base_enhanced_system)
                logger.info("✅ Domain-Aware Enhanced Search System initialized successfully!")
                return True
            else:
                logger.error("❌ Base enhanced search initialization failed")
                return False

        except Exception as e:
            logger.error(f"❌ Domain-aware enhanced search initialization failed: {e}", exc_info=True)
            enhanced_search_system = None
            return False

# Modified search endpoint - REPLACE YOUR EXISTING /search ENDPOINT WITH THIS

def modified_search_endpoint():
    """
    Modified search endpoint with domain awareness
    Replace the content of your @app.route('/search', methods=['POST']) with this
    """
    start_time = time.time()

    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        algorithm = data.get('algorithm', 'enhanced')
        top_k = min(data.get('top_k', 10), 50)
        enable_domain_filtering = data.get('enable_domain_filtering', True)  # New parameter

        if not query:
            return jsonify({"error": "Empty query"}), 400

        logger.info(f"🔍 Domain-aware search: query='{query[:50]}...', algorithm='{algorithm}'")

        # Use enhanced search with domain awareness if available
        if algorithm == 'enhanced' and enhanced_search_system:
            try:
                logger.info(f"🚀 Using Domain-Aware Enhanced Search")
                
                # Call domain-aware search
                results = enhanced_search_system.search(
                    query, 
                    top_k=top_k, 
                    enable_domain_filtering=enable_domain_filtering
                )

                # Format results with domain information
                formatted_results = []
                for result in results:
                    formatted_result = {
                        'FullName': result.get('full_name', 'Unknown'),
                        'email': result.get('email', ''),
                        'phone': result.get('phone', ''),
                        'Skills': result.get('skills', []),
                        'Experience': f"{result.get('experience_years', 0)} years",
                        'sourceURL': result.get('source_url', ''),
                        'Score': int(result.get('overall_score', 0)),
                        'Grade': result.get('grade', 'C'),
                        'SemanticScore': result.get('overall_score', 0) / 100,

                        # Enhanced fields
                        'AdvancedMatchScore': result.get('advanced_match_score', 0),
                        'SkillMatchScore': result.get('skill_match_score', 0),
                        'ExperienceRelevance': result.get('experience_relevance', 0),
                        'SeniorityMatch': result.get('seniority_match', 0),
                        'EducationMatch': result.get('education_match', 0),
                        'SoftSkillsMatch': result.get('soft_skills_match', 0),
                        'LocationMatch': result.get('location_match', 0),
                        'Confidence': result.get('confidence', 60),
                        'SeniorityLevel': result.get('seniority_level', 'Mid'),
                        'MatchExplanation': result.get('match_explanation', 'Standard match'),
                        'MissingRequirements': result.get('missing_requirements', []),
                        'StrengthAreas': result.get('strength_areas', []),
                        'MatchingAlgorithm': result.get('matching_algorithm', 'domain_aware_v2.0'),
                        
                        # NEW: Domain-specific fields
                        'Domain': result.get('domain', 'unknown'),
                        'DomainConfidence': result.get('domain_confidence', 0),
                        'DomainKeywords': result.get('domain_keywords', []),
                        'DomainFiltered': enable_domain_filtering
                    }
                    formatted_results.append(formatted_result)

                search_time = time.time() - start_time

                # Get domain statistics
                domain_stats = enhanced_search_system.get_domain_statistics()

                # Enhanced summary with domain information
                top_score = formatted_results[0]['Score'] if formatted_results else 0
                avg_score = sum(r['Score'] for r in formatted_results) / len(formatted_results) if formatted_results else 0
                domain_info = f" | Domain: {domain_stats.get('query_domain', 'mixed')}" if enable_domain_filtering else ""
                
                summary = f"Found {len(results)} candidates using Domain-Aware AI Search{domain_info} (Top: {top_score}%, Avg: {avg_score:.1f}%)"

                return jsonify({
                    "results": formatted_results,
                    "summary": summary,
                    "search_time": round(search_time, 3),
                    "algorithm": "domain_aware_enhanced_v2.0",
                    "performance": enhanced_search_system.get_performance_stats(),
                    "enhanced": True,
                    "domain_aware": True,
                    "domain_filtering_enabled": enable_domain_filtering,
                    "domain_statistics": domain_stats,
                    "query_complexity": "high" if len(query.split()) > 20 else "medium"
                })

            except Exception as e:
                logger.error(f"❌ Domain-aware enhanced search failed: {e}")
                import traceback
                logger.error(f"Full error traceback: {traceback.format_exc()}")
                # Fall back to dimension-safe search
                algorithm = 'fallback'

        # Fallback to your existing search logic here...
        # [Keep your existing fallback code]

    except Exception as e:
        search_time = time.time() - start_time
        logger.error(f"❌ Search error: {e}")
        return jsonify({
            "error": "Search failed",
            "details": str(e),
            "search_time": round(search_time, 3),
            "algorithm": algorithm
        }), 500

# Test scenarios for validation
def create_domain_test_scenarios():
    """Create test scenarios to validate domain separation"""
    return {
        'healthcare_queries': [
            "Looking for experienced ICU nurses with ACLS certification",
            "Registered Nurse needed for emergency department",
            "BSN nursing graduate for medical surgical unit",
            "Experienced RN with Epic EMR experience",
            "Nurse practitioner for primary care clinic",
            "Clinical nurse specialist in cardiology"
        ],
        'technology_queries': [
            "Senior Python developer with AWS experience",
            "Full stack engineer with React and Node.js skills",
            "Java developer with Spring Boot expertise",
            "DevOps engineer with Docker and Kubernetes",
            "Data scientist with machine learning experience",
            "Frontend developer with JavaScript and TypeScript"
        ],
        'mixed_queries': [
            "Healthcare IT specialist with programming experience",
            "Software engineer for medical device development",
            "Technical writer for healthcare documentation"
        ]
    }

def test_domain_separation(search_system, test_scenarios):
    """Test that domain separation is working correctly"""
    print("🧪 Testing Domain Separation")
    print("=" * 60)
    
    for domain, queries in test_scenarios.items():
        print(f"\n📋 Testing {domain.upper()} queries:")
        print("-" * 40)
        
        for query in queries[:3]:  # Test first 3 queries
            print(f"\nQuery: '{query}'")
            
            # Search with domain filtering enabled
            results, summary, metadata = search_system.domain_aware_search(query, top_k=5, enable_domain_filtering=True)
            
            if results:
                print(f"Results ({len(results)}): Domain filtering applied: {metadata.get('domain_filtering_applied', False)}")
                
                # Show domain distribution of results
                result_domains = [r.get('domain', 'unknown') for r in results]
                domain_dist = {domain: result_domains.count(domain) for domain in set(result_domains)}
                print(f"Domain distribution: {domain_dist}")
                
                # Show top result
                top_result = results[0]
                print(f"Top result: {top_result.get('full_name', 'Unknown')} "
                      f"(Domain: {top_result.get('domain', 'unknown')}, "
                      f"Score: {top_result.get('overall_score', 0):.1f}%)")
            else:
                print("No results found")

# Add new API endpoints for domain management
def add_domain_management_endpoints(app):
    """Add domain management endpoints to your Flask app"""
    
    @app.route('/api/domain/statistics', methods=['GET'])
    def get_domain_statistics():
        """Get domain classification statistics"""
        try:
            if enhanced_search_system and hasattr(enhanced_search_system, 'get_domain_statistics'):
                stats = enhanced_search_system.get_domain_statistics()
                return jsonify(stats)
            else:
                return jsonify({"error": "Domain-aware search not available"}), 503
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/domain/toggle', methods=['POST'])
    def toggle_domain_filtering():
        """Toggle domain filtering on/off"""
        try:
            data = request.get_json()
            enabled = data.get('enabled', True)
            
            if enhanced_search_system and hasattr(enhanced_search_system, 'toggle_domain_filtering'):
                enhanced_search_system.toggle_domain_filtering(enabled)
                return jsonify({
                    "status": "success",
                    "domain_filtering_enabled": enabled,
                    "message": f"Domain filtering {'enabled' if enabled else 'disabled'}"
                })
            else:
                return jsonify({"error": "Domain-aware search not available"}), 503
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/domain/test', methods=['POST'])
    def test_domain_classification():
        """Test domain classification on sample text"""
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({"error": "No text provided"}), 400
            
            if enhanced_search_system and hasattr(enhanced_search_system, 'domain_aware_system'):
                classifier = enhanced_search_system.domain_aware_system.domain_classifier
                classification = classifier.classify_text(text)
                
                return jsonify({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "primary_domain": classification.primary_domain,
                    "confidence": classification.confidence,
                    "domain_scores": classification.domain_scores,
                    "keywords_matched": classification.keywords_matched
                })
            else:
                return jsonify({"error": "Domain classifier not available"}), 503
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Instructions for updating your main.py
MAIN_PY_INTEGRATION_INSTRUCTIONS = """
🔧 INTEGRATION INSTRUCTIONS FOR main.py:

1. ADD IMPORTS at the top of main.py:
   ```python
   from domain_integration import (
       DomainAwareEnhancedRecruitmentSearchSystem,
       initialize_domain_aware_enhanced_search,
       add_domain_management_endpoints
   )
   ```

2. REPLACE your initialize_enhanced_search() function with:
   ```python
   def initialize_enhanced_search():
       return initialize_domain_aware_enhanced_search()
   ```

3. ADD domain management endpoints after your existing routes:
   ```python
   # Add domain management endpoints
   add_domain_management_endpoints(app)
   ```

4. UPDATE your search endpoint to handle domain filtering:
   Add this parameter to your search endpoint:
   ```python
   enable_domain_filtering = data.get('enable_domain_filtering', True)
   ```
   
   And pass it to the search call:
   ```python
   results = enhanced_search_system.search(
       query, 
       top_k=top_k, 
       enable_domain_filtering=enable_domain_filtering
   )
   ```

5. UPDATE your frontend to include domain filtering toggle (optional):
   Add this to your HTML template:
   ```html
   <div class="domain-filter-toggle">
       <label>
           <input type="checkbox" id="domainFilterToggle" checked>
           🎯 Smart Domain Filtering (Healthcare/Tech separation)
       </label>
   </div>
   ```
   
   And update your search function to include:
   ```javascript
   const enableDomainFiltering = document.getElementById('domainFilterToggle').checked;
   // Add to your fetch request body:
   body: JSON.stringify({ 
       query: query,
       algorithm: currentAlgorithm,
       top_k: 10,
       enable_domain_filtering: enableDomainFiltering  // NEW
   })
   ```
"""

print(MAIN_PY_INTEGRATION_INSTRUCTIONS)