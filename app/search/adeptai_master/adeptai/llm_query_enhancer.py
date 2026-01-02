"""
LLM-Based Query Enhancement

Uses Large Language Models (GPT-4, Claude) for intelligent query expansion
instead of rule-based synonym matching.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import time

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")

# Try to import Anthropic (Claude)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available. Install with: pip install anthropic")


class LLMQueryEnhancer:
    """
    LLM-based query enhancement using GPT-4 or Claude
    
    Provides:
    - Context-aware synonym expansion
    - Related skills inference
    - Job title variations
    - Industry-specific terms
    - Query intent understanding
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        use_cache: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize LLM Query Enhancer
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo", "claude-3-opus")
            use_cache: Whether to cache query enhancements
            cache_size: Maximum cache size
        """
        self.provider = provider
        self.model = model
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.cache_size = cache_size
        
        # Initialize LLM client
        self.client = None
        self._init_client()
        
        # Fallback to rule-based enhancer
        self._init_fallback_enhancer()
    
    def _init_client(self):
        """Initialize LLM client"""
        if self.provider == "openai" and OPENAI_AVAILABLE:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                    logger.info(f"Initialized OpenAI client with model {self.model}")
                else:
                    logger.warning("OPENAI_API_KEY not found. LLM enhancement disabled.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            try:
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if api_key:
                    self.client = Anthropic(api_key=api_key)
                    logger.info(f"Initialized Anthropic client with model {self.model}")
                else:
                    logger.warning("ANTHROPIC_API_KEY not found. LLM enhancement disabled.")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None
        else:
            logger.warning(f"LLM provider {self.provider} not available. Using fallback.")
            self.client = None
    
    def _init_fallback_enhancer(self):
        """Initialize fallback rule-based enhancer"""
        self.synonyms = {
            'developer': ['programmer', 'coder', 'engineer', 'software engineer'],
            'engineer': ['developer', 'programmer', 'architect', 'software engineer'],
            'python': ['py', 'python3', 'django', 'flask', 'fastapi'],
            'javascript': ['js', 'node', 'nodejs', 'react', 'vue'],
            'aws': ['amazon web services', 'amazon', 'cloud computing'],
            'docker': ['containerization', 'containers'],
            'kubernetes': ['k8s', 'container orchestration'],
            'nurse': ['rn', 'registered nurse', 'nursing'],
            'doctor': ['physician', 'md', 'medical doctor']
        }
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        import hashlib
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def _get_from_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Get enhancement from cache"""
        if not self.use_cache or not self.cache:
            return None
        
        cache_key = self._get_cache_key(query)
        return self.cache.get(cache_key)
    
    def _save_to_cache(self, query: str, enhancement: Dict[str, Any]):
        """Save enhancement to cache"""
        if not self.use_cache or not self.cache:
            return
        
        # Evict old entries if cache is full
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        cache_key = self._get_cache_key(query)
        self.cache[cache_key] = enhancement
    
    def _create_enhancement_prompt(self, query: str) -> str:
        """Create prompt for LLM query enhancement"""
        prompt = f"""Given this job search query: "{query}"

Generate a comprehensive query enhancement with the following components:

1. **Synonyms and Related Terms**: List synonyms and closely related terms for key words in the query
2. **Related Skills**: Based on the query context, list related skills that candidates might have
   (e.g., if query mentions "Python", include "Django", "Flask", "NumPy", "Pandas", etc.)
3. **Job Title Variations**: Generate variations of job titles/roles mentioned
   (e.g., "Python developer" → ["Python engineer", "Backend developer", "Python programmer"])
4. **Industry-Specific Terms**: Add industry-specific terminology and jargon
5. **Query Intent**: Identify the intent (skill_search, role_search, experience_search, etc.)

Return the response as a JSON object with the following structure:
{{
    "synonyms": ["term1", "term2", ...],
    "related_skills": ["skill1", "skill2", ...],
    "job_variations": ["variation1", "variation2", ...],
    "industry_terms": ["term1", "term2", ...],
    "intent": "skill_search|role_search|experience_search|location_search",
    "expanded_terms": ["all", "combined", "terms", ...]
}}

Only return the JSON object, no additional text."""
        return prompt
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API"""
        if not self.client:
            return None
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for job search query enhancement. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_tokens=1000
                )
                return response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into enhancement structure"""
        try:
            # Try to extract JSON from response
            # Handle cases where LLM adds markdown formatting
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            data = json.loads(response)
            
            # Ensure all required fields exist
            enhancement = {
                'original_query': '',
                'synonyms': data.get('synonyms', []),
                'related_skills': data.get('related_skills', []),
                'job_variations': data.get('job_variations', []),
                'industry_terms': data.get('industry_terms', []),
                'intent': data.get('intent', 'unknown'),
                'expanded_terms': data.get('expanded_terms', []),
                'variations': data.get('job_variations', []),
                'must_terms': [],
                'should_terms': [],
                'must_not_terms': []
            }
            
            # Combine all terms into expanded_terms if not provided
            if not enhancement['expanded_terms']:
                all_terms = set()
                all_terms.update(enhancement['synonyms'])
                all_terms.update(enhancement['related_skills'])
                all_terms.update(enhancement['industry_terms'])
                enhancement['expanded_terms'] = list(all_terms)
            
            return enhancement
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response: {response}")
            return self._create_fallback_enhancement("")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_enhancement("")
    
    def _create_fallback_enhancement(self, query: str) -> Dict[str, Any]:
        """Create fallback enhancement using rule-based approach"""
        query_lower = query.lower()
        words = query_lower.split()
        
        synonyms = []
        related_skills = []
        job_variations = []
        
        for word in words:
            if word in self.synonyms:
                synonyms.extend(self.synonyms[word])
        
        # Simple skill inference
        if 'python' in query_lower:
            related_skills.extend(['django', 'flask', 'fastapi', 'numpy', 'pandas'])
        if 'javascript' in query_lower:
            related_skills.extend(['react', 'vue', 'node', 'typescript'])
        if 'aws' in query_lower:
            related_skills.extend(['ec2', 's3', 'lambda', 'cloudformation'])
        
        # Job variations
        if 'developer' in query_lower:
            job_variations.append(query_lower.replace('developer', 'engineer'))
            job_variations.append(query_lower.replace('developer', 'programmer'))
        
        return {
            'original_query': query,
            'synonyms': list(set(synonyms)),
            'related_skills': list(set(related_skills)),
            'job_variations': list(set(job_variations)),
            'industry_terms': [],
            'intent': 'unknown',
            'expanded_terms': list(set(words + synonyms + related_skills)),
            'variations': job_variations,
            'must_terms': [],
            'should_terms': [],
            'must_not_terms': []
        }
    
    def enhance_query(self, query: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Enhance query using LLM or fallback
        
        Args:
            query: Original search query
            use_llm: Whether to use LLM (if available) or fallback
            
        Returns:
            Enhanced query dictionary
        """
        if not query:
            return self._create_fallback_enhancement("")
        
        # Check cache first
        if self.use_cache:
            cached = self._get_from_cache(query)
            if cached:
                cached['original_query'] = query
                logger.debug(f"Using cached enhancement for query: {query[:50]}...")
                return cached
        
        # Try LLM enhancement if available and enabled
        if use_llm and self.client:
            try:
                prompt = self._create_enhancement_prompt(query)
                response = self._call_llm(prompt)
                
                if response:
                    enhancement = self._parse_llm_response(response)
                    enhancement['original_query'] = query
                    
                    # Save to cache
                    if self.use_cache:
                        self._save_to_cache(query, enhancement)
                    
                    logger.info(f"LLM-enhanced query: {query[:50]}...")
                    return enhancement
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}. Using fallback.")
        
        # Fallback to rule-based enhancement
        enhancement = self._create_fallback_enhancement(query)
        
        # Save to cache
        if self.use_cache:
            self._save_to_cache(query, enhancement)
        
        return enhancement
    
    def batch_enhance(self, queries: List[str], use_llm: bool = True) -> List[Dict[str, Any]]:
        """
        Enhance multiple queries (with rate limiting)
        
        Args:
            queries: List of queries to enhance
            use_llm: Whether to use LLM
            
        Returns:
            List of enhanced query dictionaries
        """
        results = []
        for i, query in enumerate(queries):
            if i > 0 and use_llm and self.client:
                # Rate limiting: wait between requests
                time.sleep(0.5)  # 2 requests per second max
            
            results.append(self.enhance_query(query, use_llm=use_llm))
        
        return results


# Global instance
_llm_query_enhancer = None


def get_llm_query_enhancer(
    provider: str = "openai",
    model: str = "gpt-4",
    use_cache: bool = True
) -> LLMQueryEnhancer:
    """Get or create global LLM query enhancer instance"""
    global _llm_query_enhancer
    if _llm_query_enhancer is None:
        _llm_query_enhancer = LLMQueryEnhancer(
            provider=provider,
            model=model,
            use_cache=use_cache
        )
    return _llm_query_enhancer

