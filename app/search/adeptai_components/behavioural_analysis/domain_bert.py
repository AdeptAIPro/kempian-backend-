"""
Domain-Specific BERT Manager
Handles automatic domain detection and specialized BERT model selection
"""

from __future__ import annotations
from typing import Dict, Optional
from enum import Enum

# Try to import torch and transformers, fallback if not available
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Domain-specific BERT features will be disabled.")
    # Create dummy classes for fallback
    class AutoTokenizer:
        def __init__(self, *args, **kwargs):
            pass
        def from_pretrained(self, *args, **kwargs):
            return DummyTokenizer()
    
    class AutoModel:
        def __init__(self, *args, **kwargs):
            pass
        def from_pretrained(self, *args, **kwargs):
            return DummyModel()
    
    class DummyTokenizer:
        def __call__(self, *args, **kwargs):
            return DummyInputs()
    
    class DummyModel:
        def __call__(self, *args, **kwargs):
            return DummyOutputs()
    
    class DummyInputs:
        def __init__(self):
            self.input_ids = None
    
    class DummyOutputs:
        def __init__(self):
            self.last_hidden_state = None
    
    # Create dummy torch tensor
    class torch:
        @staticmethod
        def tensor(data, *args, **kwargs):
            return DummyTensor()
    
    class DummyTensor:
        def __getitem__(self, *args):
            return self
        def shape(self):
            return (1, 1)
        def __len__(self):
            return 1


class DomainType(Enum):
    GENERAL = "general"
    HEALTHCARE = "healthcare"
    RESEARCH = "research"
    TECH = "tech"
    FINANCE = "finance"
    LEGAL = "legal"


class DomainSpecificBERT:
    """
    Manager for domain-specific BERT variants
    Automatically selects appropriate model based on resume content
    """
    
    DOMAIN_MODELS = {
        DomainType.HEALTHCARE: "dmis-lab/biobert-base-cased-v1.1",
        DomainType.RESEARCH: "allenai/scibert_scivocab_uncased", 
        DomainType.TECH: "microsoft/codebert-base",
        DomainType.FINANCE: "ProsusAI/finbert",
        DomainType.LEGAL: "nlpaueb/legal-bert-base-uncased",
        DomainType.GENERAL: "bert-base-uncased"
    }
    
    DOMAIN_KEYWORDS = {
        DomainType.HEALTHCARE: ["medical", "clinical", "patient", "diagnosis", "treatment", "healthcare", "medicine"],
        DomainType.RESEARCH: ["research", "publication", "experiment", "analysis", "methodology", "academic", "phd"],
        DomainType.TECH: ["software", "programming", "development", "code", "algorithm", "system", "technology"],
        DomainType.FINANCE: ["finance", "banking", "investment", "trading", "risk", "portfolio", "financial"],
        DomainType.LEGAL: ["legal", "law", "attorney", "counsel", "litigation", "contract", "compliance"]
    }
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: Domain-specific BERT disabled due to missing transformers")
            self.disabled = True
        else:
            self.disabled = False
        
    def detect_domain(self, text: str) -> DomainType:
        """Detect domain based on keyword frequency and semantic similarity"""
        text_lower = text.lower()
        
        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score / len(keywords)
        
        # Return domain with highest score, default to GENERAL
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain if domain_scores[best_domain] > 0.1 else DomainType.GENERAL
    
    def get_model(self, domain: DomainType):
        """Lazy loading of domain-specific models"""
        if self.disabled:
            return DummyModel(), DummyTokenizer()
        
        if domain not in self.models:
            model_name = self.DOMAIN_MODELS[domain]
            try:
                self.tokenizers[domain] = AutoTokenizer.from_pretrained(model_name)
                self.models[domain] = AutoModel.from_pretrained(model_name)
            except Exception as e:
                print(f"Failed to load {model_name}, falling back to general BERT: {e}")
                # Fallback to general BERT
                model_name = self.DOMAIN_MODELS[DomainType.GENERAL]
                self.tokenizers[domain] = AutoTokenizer.from_pretrained(model_name)
                self.models[domain] = AutoModel.from_pretrained(model_name)
        
        return self.models[domain], self.tokenizers[domain]
    
    def encode(self, text: str, domain: DomainType) -> torch.Tensor:
        """Encode text using domain-specific BERT"""
        if self.disabled:
            # Return dummy embedding when transformers are not available
            return torch.tensor([[0.1] * 768])
        
        model, tokenizer = self.get_model(domain)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings