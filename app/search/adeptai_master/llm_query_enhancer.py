"""
LLM-Based Query Enhancement

Uses Large Language Models (Hugging Face, OpenAI GPT-4, Claude) for intelligent query expansion
instead of rule-based synonym matching. Falls back to rule-based enhancement if LLMs are unavailable.

This module provides intelligent query enhancement using various LLM providers to expand
search queries with synonyms, related skills, job variations, and industry-specific terms.

Classes:
    LLMQueryEnhancer: Main query enhancement class with multi-provider support

Functions:
    get_llm_query_enhancer: Factory function to get or create global instance

Example:
    >>> from adeptai.llm_query_enhancer import get_llm_query_enhancer
    >>> enhancer = get_llm_query_enhancer(provider="openai", model="gpt-4")
    >>> result = enhancer.enhance_query("Python developer")
    >>> print(result['expanded_terms'])
    ['python', 'django', 'flask', 'developer', 'engineer', ...]
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import time

logger = logging.getLogger(__name__)

# Try to import Hugging Face transformers (preferred for local models)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HF_TRANSFORMERS_AVAILABLE = True
    logger.info("Hugging Face transformers available")
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False
    logger.warning("Hugging Face transformers not available. Install with: pip install transformers torch")

# Try to import Hugging Face Hub for model access
try:
    from huggingface_hub import login, HfFolder
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("Hugging Face Hub not available. Install with: pip install huggingface_hub")

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
    LLM-based query enhancement using Hugging Face models, OpenAI GPT-4, or Claude
    
    Provides:
    - Context-aware synonym expansion
    - Related skills inference
    - Job title variations
    - Industry-specific terms
    - Query intent understanding
    """
    
    def __init__(
        self,
        provider: str = "huggingface",
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        use_cache: bool = True,
        cache_size: int = 1000,
        device: Optional[str] = None
    ):
        """
        Initialize LLM Query Enhancer
        
        Args:
            provider: LLM provider ("huggingface", "openai", or "anthropic")
            model: Model name 
                - For Hugging Face: model ID (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
                - For OpenAI: "gpt-4", "gpt-3.5-turbo"
                - For Anthropic: "claude-3-opus", "claude-sonnet-4-20250514"
            use_cache: Whether to cache query enhancements
            cache_size: Maximum cache size
            device: Device for Hugging Face models ("cuda", "cpu", or None for auto)
        """
        self.provider = provider.lower()
        self.model = model
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.cache_size = cache_size
        self.device = device
        
        # Initialize Hugging Face components
        self.hf_tokenizer = None
        self.hf_model = None
        self.hf_pipeline = None
        
        # Initialize external API clients
        self.client = None
        
        # Initialize LLM client/provider
        self._init_client()
        
        # Fallback to rule-based enhancer
        self._init_fallback_enhancer()
    
    def _init_client(self) -> None:
        """
        Initialize LLM client (Hugging Face, OpenAI, or Anthropic).
        
        Attempts to initialize the specified provider in order:
        1. Hugging Face (if provider is "huggingface", "hf", or "auto")
        2. OpenAI (if provider is "openai" or "auto")
        3. Anthropic (if provider is "anthropic" or "auto")
        
        Falls back to rule-based enhancement if all providers fail.
        
        Note:
            Requires appropriate API keys/tokens in environment variables:
            - HUGGINGFACE_TOKEN for Hugging Face
            - OPENAI_API_KEY for OpenAI
            - ANTHROPIC_API_KEY for Anthropic
        """
        # Determine device for Hugging Face
        if self.device is None:
            if HF_TRANSFORMERS_AVAILABLE:
                try:
                    if torch.cuda.is_available():
                        self.device = "cuda"
                    else:
                        self.device = "cpu"
                except Exception:
                    self.device = "cpu"
            else:
                self.device = "cpu"
        
        # Try Hugging Face first (preferred for local models)
        if self.provider in ["huggingface", "hf", "auto"] and HF_TRANSFORMERS_AVAILABLE:
            try:
                # Get token from environment variable (required for private models)
                hf_token = os.getenv('HUGGINGFACE_TOKEN')
                
                # Authenticate with Hugging Face if token provided
                if HF_HUB_AVAILABLE and hf_token:
                    try:
                        login(token=hf_token)
                        HfFolder.save_token(hf_token)
                        logger.info("Authenticated with Hugging Face Hub")
                    except Exception as e:
                        logger.warning(f"Hugging Face authentication failed: {e}")
                
                # Load model and tokenizer
                logger.info(f"Loading Hugging Face model: {self.model} on {self.device}")
                
                # Use pipeline for easier inference
                # Note: trust_remote_code should be passed directly to pipeline, not in model_kwargs
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.model,
                    device=0 if self.device == "cuda" else -1,
                    token=hf_token if hf_token else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                
                logger.info(f"Initialized Hugging Face model: {self.model}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Hugging Face model: {e}")
                if self.provider == "huggingface":
                    # Don't fallback if explicitly requested
                    return
        
        # Fallback to OpenAI
        if self.provider in ["openai", "auto"] and OPENAI_AVAILABLE:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                    self.provider = "openai"
                    logger.info(f"Initialized OpenAI client with model {self.model}")
                    return
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Fallback to Anthropic
        if self.provider in ["anthropic", "auto"] and ANTHROPIC_AVAILABLE:
            try:
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if api_key:
                    self.client = Anthropic(api_key=api_key)
                    self.provider = "anthropic"
                    logger.info(f"Initialized Anthropic client with model {self.model}")
                    return
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
        
        logger.warning(f"LLM provider {self.provider} not available. Using rule-based fallback.")
    
    def _init_fallback_enhancer(self) -> None:
        """
        Initialize fallback rule-based query enhancer.
        
        Creates a simple synonym dictionary for common terms when LLM
        providers are unavailable or fail.
        """
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
        """
        Create prompt for LLM query enhancement.
        
        Generates a structured prompt that instructs the LLM to enhance
        the query with synonyms, related skills, job variations, and industry terms.
        
        Args:
            query: Original search query to enhance
            
        Returns:
            Formatted prompt string for the LLM
        """
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
        """
        Call LLM provider with the given prompt.
        
        Attempts to call the configured LLM provider (Hugging Face, OpenAI,
        or Anthropic) and returns the generated response.
        
        Args:
            prompt: Formatted prompt for the LLM
            
        Returns:
            Generated text response from LLM, or None if call fails
            
        Note:
            Handles provider-specific formatting and error cases.
            Logs errors but doesn't raise exceptions.
        """
        # Try Hugging Face first
        if self.hf_pipeline:
            try:
                # Format prompt for instruction-tuned models (Llama format)
                formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for job search query enhancement. Return only valid JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
                
                # Generate response
                outputs = self.hf_pipeline(
                    formatted_prompt,
                    max_new_tokens=512,
                    temperature=0.3,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    return_full_text=False,
                    truncation=True,
                    pad_token_id=self.hf_pipeline.tokenizer.eos_token_id
                )
                
                # Extract generated text
                if isinstance(outputs, list) and len(outputs) > 0:
                    generated_text = outputs[0].get('generated_text', '')
                    # Clean up the response
                    generated_text = generated_text.strip()
                    return generated_text
            except Exception as e:
                logger.error(f"Hugging Face model call failed: {e}")
                return None
        
        # Fallback to OpenAI
        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for job search query enhancement. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                return None
        
        # Fallback to Anthropic
        if self.provider == "anthropic" and self.client:
            try:
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
                logger.error(f"Anthropic API call failed: {e}")
                return None
        
        return None
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured enhancement dictionary.
        
        Extracts JSON from LLM response (handling markdown formatting)
        and validates/structures the enhancement data.
        
        Args:
            response: Raw text response from LLM
            
        Returns:
            Dictionary containing:
            - synonyms: List of synonym terms
            - related_skills: List of related skills
            - job_variations: List of job title variations
            - industry_terms: List of industry-specific terms
            - intent: Query intent (skill_search, role_search, etc.)
            - expanded_terms: Combined list of all terms
            
        Note:
            Falls back to empty enhancement if parsing fails.
        """
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
        """
        Create fallback enhancement using rule-based approach.
        
        Uses simple synonym matching and keyword-based inference when
        LLM providers are unavailable.
        
        Args:
            query: Original search query
            
        Returns:
            Dictionary with basic enhancement structure (same format as LLM response)
        """
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
        Enhance query using LLM or fallback.
        
        Attempts to use LLM enhancement if available and enabled, otherwise
        falls back to rule-based enhancement. Results are cached if caching is enabled.
        
        Args:
            query: Original search query to enhance
            use_llm: Whether to attempt LLM enhancement (if available).
                   If False, uses rule-based fallback immediately.
            
        Returns:
            Dictionary containing:
            - original_query: Original query text
            - synonyms: List of synonym terms
            - related_skills: List of related skills
            - job_variations: List of job title variations
            - industry_terms: List of industry-specific terms
            - intent: Query intent classification
            - expanded_terms: Combined list of all expanded terms
            - variations: Job title variations (alias for job_variations)
            - must_terms: Terms that must match (currently empty)
            - should_terms: Terms that should match (currently empty)
            - must_not_terms: Terms that must not match (currently empty)
            
        Example:
            >>> enhancer = LLMQueryEnhancer(provider="openai")
            >>> result = enhancer.enhance_query("Python developer")
            >>> print(result['expanded_terms'])
            ['python', 'django', 'flask', 'developer', ...]
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
        if use_llm and (self.hf_pipeline or self.client):
            try:
                prompt = self._create_enhancement_prompt(query)
                response = self._call_llm(prompt)
                
                if response:
                    enhancement = self._parse_llm_response(response)
                    enhancement['original_query'] = query
                    
                    # Save to cache
                    if self.use_cache:
                        self._save_to_cache(query, enhancement)
                    
                    provider_name = "Hugging Face" if self.hf_pipeline else self.provider
                    logger.info(f"LLM-enhanced query ({provider_name}): {query[:50]}...")
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
        Enhance multiple queries with rate limiting.
        
        Processes multiple queries sequentially with rate limiting to avoid
        API throttling when using external LLM providers.
        
        Args:
            queries: List of queries to enhance
            use_llm: Whether to use LLM enhancement (if available)
            
        Returns:
            List of enhanced query dictionaries (one per input query)
            
        Note:
            Adds 0.5 second delay between requests when using external APIs
            to respect rate limits (2 requests per second max).
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
    provider: str = "huggingface",
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    use_cache: bool = True,
    device: Optional[str] = None
) -> LLMQueryEnhancer:
    """
    Get or create global LLM query enhancer instance
    
    Args:
        provider: LLM provider ("huggingface", "openai", "anthropic", or "auto")
        model: Model name or ID
            - Hugging Face: "meta-llama/Meta-Llama-3.1-8B-Instruct"
            - OpenAI: "gpt-4", "gpt-3.5-turbo"
            - Anthropic: "claude-sonnet-4-20250514"
        use_cache: Whether to cache query enhancements
        device: Device for Hugging Face ("cuda", "cpu", or None for auto)
    
    Returns:
        LLMQueryEnhancer instance
    """
    global _llm_query_enhancer
    if _llm_query_enhancer is None:
        _llm_query_enhancer = LLMQueryEnhancer(
            provider=provider,
            model=model,
            use_cache=use_cache,
            device=device
        )
    return _llm_query_enhancer

