"""
SageMaker Query Enhancement Service
Replaces OpenAI/Claude query enhancement with SageMaker-hosted models
Uses Llama 3.1 8B fine-tuned for recruitment query enhancement
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .sagemaker_llm_client import (
    SageMakerLLMClient,
    ModelType,
    LLMRequest,
    LLMResponse,
    get_sagemaker_client
)

logger = logging.getLogger(__name__)


@dataclass
class QueryEnhancement:
    """Query enhancement result structure"""
    original_query: str
    synonyms: List[str]
    related_skills: List[str]
    job_variations: List[str]
    industry_terms: List[str]
    intent: str
    expanded_terms: List[str]
    variations: List[str]
    must_terms: List[str]
    should_terms: List[str]
    must_not_terms: List[str]
    confidence: float = 0.0


class SageMakerQueryEnhancer:
    """
    Query enhancement service using SageMaker-hosted LLM
    Replaces OpenAI/Claude query enhancement
    """
    
    def __init__(
        self,
        client: Optional[SageMakerLLMClient] = None,
        model_name: str = "llama-3.1-8b-instruct",
        temperature: float = 0.3,
        max_tokens: int = 512
    ):
        """
        Initialize SageMaker Query Enhancer
        
        Args:
            client: SageMaker LLM client (uses global client if None)
            model_name: Name of the model (for reference)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.client = client or get_sagemaker_client()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Cache for enhanced queries
        self.cache: Dict[str, QueryEnhancement] = {}
    
    def _create_enhancement_prompt(self, query: str) -> str:
        """Create prompt for query enhancement"""
        return f"""You are an expert recruitment AI assistant specializing in job search query enhancement.

Given this job search query: "{query}"

Generate a comprehensive query enhancement with the following components:

1. **Synonyms and Related Terms**: List synonyms and closely related terms for key words in the query
2. **Related Skills**: Based on the query context, list related skills that candidates might have
   (e.g., if query mentions "Python", include "Django", "Flask", "NumPy", "Pandas", etc.)
3. **Job Title Variations**: Generate variations of job titles/roles mentioned
   (e.g., "Python developer" → ["Python engineer", "Backend developer", "Python programmer"])
4. **Industry-Specific Terms**: Add industry-specific terminology and jargon
5. **Query Intent**: Identify the intent (skill_search, role_search, experience_search, location_search, general_search)

Return ONLY a valid JSON object with the following structure:
{{
    "synonyms": ["term1", "term2", ...],
    "related_skills": ["skill1", "skill2", ...],
    "job_variations": ["variation1", "variation2", ...],
    "industry_terms": ["term1", "term2", ...],
    "intent": "skill_search|role_search|experience_search|location_search|general_search",
    "expanded_terms": ["all", "combined", "terms", ...],
    "variations": ["variation1", "variation2", ...],
    "must_terms": [],
    "should_terms": [],
    "must_not_terms": []
}}

Only return the JSON object, no additional text or explanation."""
    
    def _parse_enhancement_response(self, response: str, original_query: str) -> QueryEnhancement:
        """Parse LLM response into QueryEnhancement object"""
        try:
            # Clean response - remove markdown formatting if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Ensure all required fields exist
            enhancement = QueryEnhancement(
                original_query=original_query,
                synonyms=data.get('synonyms', []),
                related_skills=data.get('related_skills', []),
                job_variations=data.get('job_variations', []),
                industry_terms=data.get('industry_terms', []),
                intent=data.get('intent', 'general_search'),
                expanded_terms=data.get('expanded_terms', []),
                variations=data.get('variations', data.get('job_variations', [])),
                must_terms=data.get('must_terms', []),
                should_terms=data.get('should_terms', []),
                must_not_terms=data.get('must_not_terms', []),
                confidence=0.9  # High confidence for structured output
            )
            
            # If expanded_terms is empty, combine all terms
            if not enhancement.expanded_terms:
                all_terms = set()
                all_terms.update(enhancement.synonyms)
                all_terms.update(enhancement.related_skills)
                all_terms.update(enhancement.industry_terms)
                all_terms.update(original_query.split())
                enhancement.expanded_terms = list(all_terms)
            
            return enhancement
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse enhancement response as JSON: {e}")
            logger.debug(f"Response: {response}")
            return self._create_fallback_enhancement(original_query)
        except Exception as e:
            logger.error(f"Error parsing enhancement response: {e}")
            return self._create_fallback_enhancement(original_query)
    
    def _create_fallback_enhancement(self, query: str) -> QueryEnhancement:
        """Create fallback enhancement using simple rules"""
        query_lower = query.lower()
        words = query_lower.split()
        
        # Simple synonym mapping
        synonyms = []
        related_skills = []
        job_variations = []
        
        if 'developer' in query_lower:
            synonyms.extend(['programmer', 'coder', 'engineer'])
            job_variations.append(query.replace('developer', 'engineer'))
        
        if 'python' in query_lower:
            related_skills.extend(['django', 'flask', 'fastapi', 'numpy', 'pandas'])
        
        if 'javascript' in query_lower:
            related_skills.extend(['react', 'vue', 'node', 'typescript'])
        
        if 'aws' in query_lower:
            related_skills.extend(['ec2', 's3', 'lambda', 'cloudformation'])
        
        return QueryEnhancement(
            original_query=query,
            synonyms=list(set(synonyms)),
            related_skills=list(set(related_skills)),
            job_variations=list(set(job_variations)),
            industry_terms=[],
            intent='general_search',
            expanded_terms=list(set(words + synonyms + related_skills)),
            variations=job_variations,
            must_terms=[],
            should_terms=[],
            must_not_terms=[],
            confidence=0.5
        )
    
    def enhance_query(
        self,
        query: str,
        use_cache: bool = True,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Enhance search query using SageMaker LLM
        
        Args:
            query: Original search query
            use_cache: Whether to use cached results
            use_llm: Whether to use LLM (if False, uses fallback)
            
        Returns:
            Dictionary with enhanced query information
        """
        if not query or not query.strip():
            return self._create_fallback_enhancement("").__dict__
        
        query = query.strip()
        
        # Check cache first
        if use_cache and query in self.cache:
            cached = self.cache[query]
            logger.debug(f"Using cached enhancement for query: {query[:50]}...")
            return cached.__dict__
        
        # Try LLM enhancement if enabled
        if use_llm:
            try:
                prompt = self._create_enhancement_prompt(query)
                
                # Create LLM request
                llm_request = LLMRequest(
                    prompt=prompt,
                    system_prompt="You are a recruitment AI expert. Always return valid JSON only.",
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    top_k=50
                )
                
                # Invoke SageMaker endpoint
                response = self.client.invoke(
                    ModelType.QUERY_ENHANCER,
                    llm_request,
                    use_cache=use_cache
                )
                
                # Parse response
                enhancement = self._parse_enhancement_response(
                    response.content,
                    query
                )
                
                # Update confidence based on response quality
                enhancement.confidence = response.confidence_score if response.confidence_score > 0 else 0.9
                
                # Save to cache
                if use_cache:
                    self.cache[query] = enhancement
                
                logger.info(f"Query enhanced using SageMaker: {query[:50]}...")
                return enhancement.__dict__
                
            except Exception as e:
                logger.warning(f"SageMaker query enhancement failed: {e}. Using fallback.")
        
        # Fallback to rule-based enhancement
        enhancement = self._create_fallback_enhancement(query)
        
        # Save to cache
        if use_cache:
            self.cache[query] = enhancement
        
        return enhancement.__dict__
    
    def batch_enhance(
        self,
        queries: List[str],
        use_cache: bool = True,
        use_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Enhance multiple queries efficiently
        
        Args:
            queries: List of queries to enhance
            use_cache: Whether to use cache
            use_llm: Whether to use LLM
            
        Returns:
            List of enhanced query dictionaries
        """
        results = []
        for query in queries:
            try:
                enhancement = self.enhance_query(query, use_cache, use_llm)
                results.append(enhancement)
            except Exception as e:
                logger.error(f"Failed to enhance query '{query}': {e}")
                # Add fallback enhancement
                fallback = self._create_fallback_enhancement(query)
                results.append(fallback.__dict__)
        
        return results
    
    def clear_cache(self):
        """Clear query enhancement cache"""
        self.cache.clear()
        logger.info("Query enhancement cache cleared")


# Global instance
_query_enhancer: Optional[SageMakerQueryEnhancer] = None


def get_query_enhancer() -> SageMakerQueryEnhancer:
    """Get or create global query enhancer instance"""
    global _query_enhancer
    if _query_enhancer is None:
        _query_enhancer = SageMakerQueryEnhancer()
    return _query_enhancer

