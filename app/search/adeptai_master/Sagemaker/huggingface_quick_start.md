# Hugging Face Models Quick Start Guide

## Overview

This guide provides a quick start for using Hugging Face models in the SageMaker implementation. All models are accessed using the provided token: `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`

## Quick Setup

### 1. Environment Configuration

```bash
# Set Hugging Face token (already configured in Dockerfile)
export HUGGINGFACE_TOKEN=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

# Set cache directories
export HF_HOME=/opt/ml/model/hf_cache
export TRANSFORMERS_CACHE=/opt/ml/model/hf_cache
```

### 2. Model Selection

```python
from Sagemaker import ModelUseCase, get_huggingface_model_id

# Get model ID for each use case
query_model = get_huggingface_model_id(ModelUseCase.QUERY_ENHANCEMENT)
# Returns: "meta-llama/Meta-Llama-3.1-8B-Instruct"

behavioral_model = get_huggingface_model_id(ModelUseCase.BEHAVIORAL_ANALYSIS)
# Returns: "meta-llama/Meta-Llama-3.1-70B-Instruct"
```

### 3. Service Usage

```python
from Sagemaker import get_query_enhancer

# Use query enhancer (automatically uses Hugging Face model)
enhancer = get_query_enhancer()
result = enhancer.enhance_query("Python developer with AWS experience")
print(result)
```

## Available Models

### Primary Models (Pre-configured)

1. **Query Enhancement**
   - Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
   - Alternative: `mistralai/Mistral-7B-Instruct-v0.3`

2. **Behavioral Analysis**
   - Model: `meta-llama/Meta-Llama-3.1-70B-Instruct`
   - Alternative: `mistralai/Mixtral-8x7B-Instruct-v0.1`

3. **Market Intelligence**
   - Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
   - Alternative: `Qwen/Qwen2.5-7B-Instruct`

4. **Job Parsing**
   - Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
   - Alternative: `microsoft/Phi-3-medium-4k-instruct`

5. **Explanation Generation**
   - Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
   - Alternative: `mistralai/Mistral-7B-Instruct-v0.3`

## Deployment

### Using Hugging Face Models Directly

```python
from sagemaker.huggingface import HuggingFaceModel

# Deploy model from Hugging Face Hub
huggingface_model = HuggingFaceModel(
    model_data=None,  # Download from Hugging Face Hub
    role=role,
    transformers_version="4.35",
    pytorch_version="2.1",
    py_version="py310",
    env={
        "HUGGINGFACE_TOKEN": "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",
        "HF_MODEL_ID": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "HF_HOME": "/opt/ml/model/hf_cache"
    }
)

# Deploy endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="adeptai-query-enhancer-v1"
)
```

## Model Caching

Models are automatically cached in `/opt/ml/model/hf_cache` to avoid re-downloading. The cache persists across container restarts.

## Token Usage

The Hugging Face token (`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`) provides:
- Access to gated models (Llama series)
- Enhanced rate limits
- Private model access (if applicable)

## Troubleshooting

### Authentication Issues
- Verify token: `echo $HUGGINGFACE_TOKEN`
- Check token validity on Hugging Face website
- Re-authenticate if needed

### Download Issues
- Check network connectivity
- Verify model ID is correct
- Check disk space for cache
- Review Hugging Face rate limits

### Model Loading Issues
- Verify model is compatible with transformers version
- Check GPU memory availability
- Review model size vs instance type

## Next Steps

1. Review `huggingface_models_config.py` for all available models
2. Check `model_selection_guide.md` for detailed model comparisons
3. Review `HUGGINGFACE_MODELS_GUIDE.md` for comprehensive guide
4. Use `deploy_huggingface_models.py` for automated deployment

