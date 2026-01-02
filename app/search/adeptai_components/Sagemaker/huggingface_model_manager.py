"""
Hugging Face Model Manager
Manages model downloads, caching, and loading from Hugging Face Hub
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from huggingface_hub import login, snapshot_download, hf_hub_download
from huggingface_hub.utils import HfFolder
from .huggingface_models_config import (
    get_model_for_use_case,
    ModelUseCase,
    get_huggingface_token,
    DEFAULT_HF_TOKEN
)

logger = logging.getLogger(__name__)


class HuggingFaceModelManager:
    """
    Manager for Hugging Face models
    Handles authentication, downloading, and caching models
    """
    
    def __init__(self, token: Optional[str] = None, cache_dir: str = "/opt/ml/model/hf_cache"):
        """
        Initialize Hugging Face Model Manager
        
        Args:
            token: Hugging Face token (uses default if None)
            cache_dir: Directory for caching models
        """
        self.token = token or get_huggingface_token() or DEFAULT_HF_TOKEN
        self.cache_dir = cache_dir
        self.authenticated = False
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Authenticate with Hugging Face
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Hugging Face"""
        try:
            login(token=self.token)
            HfFolder.save_token(self.token)
            self.authenticated = True
            logger.info("Authenticated with Hugging Face successfully")
        except Exception as e:
            logger.error(f"Hugging Face authentication failed: {e}")
            raise
    
    def download_model(
        self,
        model_id: str,
        use_case: ModelUseCase,
        force_download: bool = False
    ) -> str:
        """
        Download model from Hugging Face Hub
        
        Args:
            model_id: Hugging Face model ID (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
            use_case: Model use case
            force_download: Force re-download even if cached
            
        Returns:
            Path to downloaded model
        """
        if not self.authenticated:
            self._authenticate()
        
        try:
            # Create model-specific cache directory
            model_cache_dir = os.path.join(self.cache_dir, model_id.replace("/", "_"))
            
            # Check if model already exists
            if not force_download and os.path.exists(model_cache_dir):
                logger.info(f"Model {model_id} already cached at {model_cache_dir}")
                return model_cache_dir
            
            logger.info(f"Downloading model {model_id} from Hugging Face Hub...")
            
            # Download model snapshot
            model_path = snapshot_download(
                repo_id=model_id,
                token=self.token,
                cache_dir=self.cache_dir,
                local_dir=model_cache_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"Model {model_id} downloaded successfully to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            raise
    
    def get_model_path(
        self,
        use_case: ModelUseCase,
        priority: int = 1
    ) -> str:
        """
        Get model path for a use case, downloading if necessary
        
        Args:
            use_case: Model use case
            priority: Model priority (1 = primary, 2 = alternative, 3 = fallback)
            
        Returns:
            Path to model directory
        """
        model_config = get_model_for_use_case(use_case, priority)
        
        if not model_config:
            raise ValueError(f"No model found for use case {use_case.value} with priority {priority}")
        
        # Download model if not already cached
        model_path = self.download_model(
            model_config.model_id,
            use_case
        )
        
        return model_path
    
    def list_available_models(self, use_case: ModelUseCase) -> List[Dict[str, Any]]:
        """
        List all available models for a use case
        
        Args:
            use_case: Model use case
            
        Returns:
            List of model configurations
        """
        from .huggingface_models_config import HUGGINGFACE_MODELS
        
        models = HUGGINGFACE_MODELS.get(use_case, [])
        return [
            {
                "model_id": model.model_id,
                "description": model.description,
                "size": model.size,
                "instance_type": model.instance_type,
                "priority": model.priority,
                "notes": model.notes
            }
            for model in models
        ]
    
    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model
        
        Args:
            model_id: Hugging Face model ID
            
        Returns:
            Model configuration dictionary or None
        """
        from .huggingface_models_config import HUGGINGFACE_MODELS
        
        for use_case, models in HUGGINGFACE_MODELS.items():
            for model in models:
                if model.model_id == model_id:
                    return {
                        "model_id": model.model_id,
                        "use_case": model.use_case.value,
                        "description": model.description,
                        "size": model.size,
                        "instance_type": model.instance_type,
                        "max_tokens": model.max_tokens,
                        "temperature": model.temperature,
                        "priority": model.priority,
                        "notes": model.notes
                    }
        
        return None
    
    def clear_cache(self, model_id: Optional[str] = None):
        """
        Clear model cache
        
        Args:
            model_id: Specific model ID to clear (None = clear all)
        """
        if model_id:
            model_cache_dir = os.path.join(self.cache_dir, model_id.replace("/", "_"))
            if os.path.exists(model_cache_dir):
                import shutil
                shutil.rmtree(model_cache_dir)
                logger.info(f"Cleared cache for {model_id}")
        else:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info("Cleared all model cache")


# Global model manager instance
_model_manager: Optional[HuggingFaceModelManager] = None


def get_model_manager() -> HuggingFaceModelManager:
    """Get or create global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = HuggingFaceModelManager()
    return _model_manager


def initialize_model_manager(token: Optional[str] = None, cache_dir: Optional[str] = None) -> HuggingFaceModelManager:
    """Initialize global model manager with custom configuration"""
    global _model_manager
    kwargs = {}
    if token:
        kwargs['token'] = token
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    _model_manager = HuggingFaceModelManager(**kwargs)
    return _model_manager

