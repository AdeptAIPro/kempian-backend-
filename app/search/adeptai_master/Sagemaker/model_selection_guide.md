# Hugging Face Model Selection Guide

## Model Selection by Use Case

This guide helps you select the most suitable Hugging Face models for each use case in AdeptAI.

## Available Models

### Meta Llama 3.1 Series
- **Llama 3.1 8B Instruct**: Fast, efficient, excellent instruction following
- **Llama 3.1 70B Instruct**: Deep reasoning, best for complex tasks
- **Access**: Requires Meta approval (automated via token)

### Mistral AI Series
- **Mistral 7B Instruct v0.3**: Apache 2.0 license, cost-effective
- **Mixtral 8x7B Instruct**: Mixture of experts, balanced performance
- **Access**: Open access

### Qwen Series
- **Qwen 2.5 7B Instruct**: Strong multilingual support
- **Qwen 2.5 72B Instruct**: Large model with multilingual capabilities
- **Access**: Open access

### Microsoft Phi Series
- **Phi-3 Medium 4k Instruct**: Optimized for structured output
- **Access**: Open access

## Use Case Recommendations

### 1. Query Enhancement
**Best Choice:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Why:** Fast, accurate query expansion
- **Cost:** Low (ml.g5.xlarge)
- **Latency:** < 100ms
- **Alternative:** `mistralai/Mistral-7B-Instruct-v0.3` (if license is important)

### 2. Behavioral Analysis
**Best Choice:** `meta-llama/Meta-Llama-3.1-70B-Instruct`
- **Why:** Deep reasoning for nuanced analysis
- **Cost:** Medium (ml.g5.12xlarge)
- **Latency:** < 800ms
- **Alternative:** `mistralai/Mixtral-8x7B-Instruct-v0.1` (if cost is concern)

### 3. Market Intelligence
**Best Choice:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Why:** Fast enough, good structured output
- **Cost:** Low (ml.g5.2xlarge)
- **Latency:** < 500ms
- **Alternative:** `Qwen/Qwen2.5-7B-Instruct` (for multilingual markets)

### 4. Job Parsing
**Best Choice:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Why:** Excellent structured extraction
- **Cost:** Low (ml.g5.xlarge)
- **Latency:** < 150ms
- **Alternative:** `microsoft/Phi-3-medium-4k-instruct` (optimized for structured output)

### 5. Explanation Generation
**Best Choice:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Why:** Natural language generation
- **Cost:** Low (ml.g5.xlarge)
- **Latency:** < 200ms
- **Alternative:** `mistralai/Mistral-7B-Instruct-v0.3`

## Model Comparison Matrix

| Model | Size | Speed | Quality | Cost | Best For |
|-------|------|-------|---------|------|----------|
| Llama 3.1 8B | 8B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low | Most use cases |
| Llama 3.1 70B | 70B | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High | Complex analysis |
| Mistral 7B | 7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low | Cost-sensitive |
| Mixtral 8x7B | 47B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | Balanced |
| Qwen 2.5 7B | 7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low | Multilingual |
| Qwen 2.5 72B | 72B | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High | Multilingual large |
| Phi-3 Medium | 14B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low | Structured output |

## Selection Criteria

### Performance Requirements
- **Low Latency (< 100ms)**: Use 8B models
- **Medium Latency (< 500ms)**: Use 8B-14B models
- **High Quality Required**: Use 70B models

### Cost Constraints
- **Budget-Conscious**: Use Mistral 7B or Phi-3
- **Balanced**: Use Llama 3.1 8B
- **Quality-First**: Use Llama 3.1 70B or Mixtral

### License Requirements
- **Apache 2.0**: Mistral, Qwen, Phi-3
- **Custom License**: Llama (requires approval)

### Multilingual Support
- **English Only**: Llama, Mistral, Phi-3
- **Multilingual**: Qwen 2.5 series

### Specialized Tasks
- **Structured Output**: Phi-3 Medium
- **Complex Reasoning**: Llama 3.1 70B, Mixtral
- **Fast Processing**: Llama 3.1 8B, Mistral 7B

## Configuration Examples

### Query Enhancement
```python
from Sagemaker import ModelUseCase, get_huggingface_model_id

model_id = get_huggingface_model_id(ModelUseCase.QUERY_ENHANCEMENT)
# Returns: "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

### Behavioral Analysis
```python
model_id = get_huggingface_model_id(ModelUseCase.BEHAVIORAL_ANALYSIS)
# Returns: "meta-llama/Meta-Llama-3.1-70B-Instruct"
```

### Using Alternative Models
```python
from Sagemaker.huggingface_models_config import get_model_for_use_case, ModelUseCase

# Get alternative model (priority=2)
model_config = get_model_for_use_case(ModelUseCase.QUERY_ENHANCEMENT, priority=2)
print(f"Alternative model: {model_config.model_id}")
# Returns: "mistralai/Mistral-7B-Instruct-v0.3"
```

## Model Updates

Models are regularly updated on Hugging Face. To use newer versions:
1. Check Hugging Face for latest model versions
2. Update `huggingface_models_config.py` with new model IDs
3. Test new models in staging
4. Deploy to production

## Token Usage

The provided token (`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`) provides:
- Access to gated models (Llama series)
- Enhanced rate limits
- Private model access (if applicable)

Monitor token usage on Hugging Face dashboard.

