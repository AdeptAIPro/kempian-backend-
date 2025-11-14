"""
SageMaker Inference Handler (Optional - Reference Only)

NOTE: This file is optional and not required for deployment.
SageMaker's managed Hugging Face containers include built-in inference handlers
that work with standard Hugging Face models automatically.

Use this file only if you need:
- Custom inference logic beyond standard Hugging Face models
- Custom preprocessing/postprocessing
- Special model loading requirements

For standard deployments, SageMaker's managed containers handle everything automatically.
No Docker image or custom inference code is needed.
"""

import json
import logging
import os
import torch
from typing import Dict, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteriaList,
    StoppingCriteria
)
from huggingface_hub import login, snapshot_download
from huggingface_hub.utils import HfFolder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StopOnTokens(StoppingCriteria):
    """Stop generation when specific tokens are encountered"""
    def __init__(self, stop_token_ids: list):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class ModelHandler:
    """Model handler for SageMaker inference"""
    
    def __init__(self):
        """Initialize model handler"""
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
    
    def _authenticate_huggingface(self):
        """Authenticate with Hugging Face using token"""
        try:
            # Get Hugging Face token from environment or use default
            hf_token = os.environ.get('HUGGINGFACE_TOKEN', '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')
            
            # Login to Hugging Face
            login(token=hf_token)
            HfFolder.save_token(hf_token)
            
            logger.info("Authenticated with Hugging Face successfully")
        except Exception as e:
            logger.warning(f"Hugging Face authentication failed: {e}. Continuing without authentication.")
    
    def _load_model(self):
        """Load model and tokenizer from local path or Hugging Face Hub"""
        if self.model_loaded:
            return
        
        try:
            # Authenticate with Hugging Face
            self._authenticate_huggingface()
            
            # Get model configuration from environment
            model_path = os.environ.get('MODEL_PATH', '/opt/ml/model')
            hf_model_id = os.environ.get('HF_MODEL_ID', '')
            
            # Determine if loading from Hugging Face Hub or local path
            use_huggingface = hf_model_id and hf_model_id.startswith(('meta-llama', 'mistralai', 'Qwen', 'microsoft', 'google'))
            
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info("Using GPU for inference")
            else:
                self.device = torch.device('cpu')
                logger.info("Using CPU for inference")
            
            # Load from Hugging Face Hub if specified
            if use_huggingface:
                logger.info(f"Loading model from Hugging Face Hub: {hf_model_id}")
                
                # Get Hugging Face token
                hf_token = os.environ.get('HUGGINGFACE_TOKEN', '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')
                
                # Load tokenizer from Hugging Face
                self.tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_id,
                    token=hf_token,
                    trust_remote_code=True
                )
                
                # Set padding token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model from Hugging Face
                model_kwargs = {
                    "token": hf_token,
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if self.device.type == 'cuda' else torch.float32,
                }
                
                # Use device_map for multi-GPU setups
                if torch.cuda.device_count() > 1:
                    model_kwargs["device_map"] = "auto"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    **model_kwargs
                )
            else:
                # Load from local path
                logger.info(f"Loading model from local path: {model_path}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                # Set padding token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model with appropriate settings
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if self.device.type == 'cuda' else torch.float32,
                }
                
                # Use device_map for multi-GPU setups
                if torch.cuda.device_count() > 1:
                    model_kwargs["device_map"] = "auto"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
            
            # Move to device if not using device_map
            if not hasattr(self.model, 'hf_device_map'):
                self.model.to(self.device)
            
            self.model.eval()
            self.model_loaded = True
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _prepare_stopping_criteria(self, stop_sequences: Optional[list]) -> Optional[StoppingCriteriaList]:
        """Prepare stopping criteria from stop sequences"""
        if not stop_sequences:
            return None
        
        stop_token_ids = []
        for stop_seq in stop_sequences:
            if isinstance(stop_seq, str):
                tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                if tokens:
                    stop_token_ids.extend(tokens)
        
        if stop_token_ids:
            return StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        return None
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction
        
        Args:
            inputs: Dictionary with 'inputs' (prompt) and 'parameters' (generation params)
            
        Returns:
            Dictionary with generated text
        """
        # Load model if not loaded
        if not self.model_loaded:
            self._load_model()
        
        try:
            # Extract inputs
            prompt = inputs.get('inputs', '')
            parameters = inputs.get('parameters', {})
            
            # Get generation parameters
            max_new_tokens = parameters.get('max_new_tokens', 512)
            temperature = parameters.get('temperature', 0.7)
            top_p = parameters.get('top_p', 0.9)
            top_k = parameters.get('top_k', 50)
            repetition_penalty = parameters.get('repetition_penalty', 1.1)
            stop_sequences = parameters.get('stop', [])
            
            # Handle system prompt if provided
            system_prompt = inputs.get('system_prompt', '')
            if system_prompt:
                # Format prompt with system prompt (Llama 3.1 format)
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                formatted_prompt = prompt
            
            # Tokenize input
            input_ids = self.tokenizer.encode(
                formatted_prompt,
                return_tensors='pt',
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Prepare stopping criteria
            stopping_criteria = self._prepare_stopping_criteria(stop_sequences)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Calculate tokens used
            tokens_used = outputs.shape[1]
            
            # Prepare response
            response = {
                "generated_text": generated_text,
                "tokens_used": int(tokens_used),
                "confidence_score": 0.9  # Can be enhanced with model confidence if available
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


# Global model handler instance
_model_handler: Optional[ModelHandler] = None


def model_fn(model_dir: str) -> ModelHandler:
    """
    Load model during container startup
    
    Args:
        model_dir: Directory containing model files or Hugging Face model ID
        
    Returns:
        ModelHandler instance
    """
    global _model_handler
    
    # Check if model_dir is a Hugging Face model ID
    # Format: "hf://model_id" or just "model_id" if it contains "/"
    if model_dir.startswith("hf://"):
        hf_model_id = model_dir[5:]  # Remove "hf://" prefix
        os.environ['HF_MODEL_ID'] = hf_model_id
        logger.info(f"Loading Hugging Face model: {hf_model_id}")
    elif "/" in model_dir and not os.path.exists(model_dir):
        # Likely a Hugging Face model ID
        os.environ['HF_MODEL_ID'] = model_dir
        logger.info(f"Loading Hugging Face model: {model_dir}")
    else:
        os.environ['MODEL_PATH'] = model_dir
        logger.info(f"Loading local model from: {model_dir}")
    
    # Set Hugging Face token if not already set
    if 'HUGGINGFACE_TOKEN' not in os.environ:
        os.environ['HUGGINGFACE_TOKEN'] = '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'
    
    _model_handler = ModelHandler()
    _model_handler._load_model()
    return _model_handler


def input_fn(request_body: bytes, request_content_type: str) -> Dict[str, Any]:
    """
    Parse input request
    
    Args:
        request_body: Request body bytes
        request_content_type: Content type of request
        
    Returns:
        Parsed input dictionary
    """
    if request_content_type == 'application/json':
        return json.loads(request_body.decode('utf-8'))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: Dict[str, Any], model: ModelHandler) -> Dict[str, Any]:
    """
    Generate prediction
    
    Args:
        input_data: Parsed input dictionary
        model: ModelHandler instance
        
    Returns:
        Prediction dictionary
    """
    return model.predict(input_data)


def output_fn(prediction: Dict[str, Any], response_content_type: str) -> bytes:
    """
    Format output
    
    Args:
        prediction: Prediction dictionary
        response_content_type: Desired response content type
        
    Returns:
        Formatted response bytes
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction).encode('utf-8')
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")

