"""
Example: Deploying Hugging Face Models to SageMaker
Demonstrates how to deploy and use Hugging Face models on SageMaker
"""

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.huggingface.model import HuggingFacePredictor
import os
import json

# Configuration
HF_TOKEN = "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"
REGION = "us-east-1"
ROLE = "arn:aws:iam::<account-id>:role/SageMakerExecutionRole"

# Initialize SageMaker session
session = sagemaker.Session()
sagemaker_client = boto3.client('sagemaker', region_name=REGION)


def deploy_query_enhancer_model():
    """Deploy query enhancement model"""
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Create HuggingFace model using SageMaker's managed container
    # No Docker required - SageMaker provides pre-built Hugging Face containers
    huggingface_model = HuggingFaceModel(
        model_data=None,  # None = download from Hugging Face Hub during deployment
        role=ROLE,
        transformers_version="4.35",  # SageMaker managed container version
        pytorch_version="2.1",         # SageMaker managed container version
        py_version="py310",            # SageMaker managed container version
        # Note: No image_uri needed - SageMaker uses managed containers automatically
        env={
            "HUGGINGFACE_TOKEN": HF_TOKEN,
            "HF_MODEL_ID": model_id,
            "HF_HOME": "/opt/ml/model/hf_cache",
            "TRANSFORMERS_CACHE": "/opt/ml/model/hf_cache"
        }
    )
    
    # Deploy endpoint
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name="adeptai-query-enhancer-v1",
        wait=True
    )
    
    print(f"Query enhancer endpoint deployed: {predictor.endpoint_name}")
    return predictor


def deploy_behavioral_analyzer_model():
    """Deploy behavioral analysis model"""
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    
    # Create HuggingFace model
    huggingface_model = HuggingFaceModel(
        model_data=None,
        role=ROLE,
        transformers_version="4.35",
        pytorch_version="2.1",
        py_version="py310",
        env={
            "HUGGINGFACE_TOKEN": HF_TOKEN,
            "HF_MODEL_ID": model_id,
            "HF_HOME": "/opt/ml/model/hf_cache",
            "TRANSFORMERS_CACHE": "/opt/ml/model/hf_cache"
        }
    )
    
    # Deploy endpoint
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.12xlarge",
        endpoint_name="adeptai-behavioral-analyzer-v1",
        wait=True
    )
    
    print(f"Behavioral analyzer endpoint deployed: {predictor.endpoint_name}")
    return predictor


def test_endpoint(predictor: HuggingFacePredictor):
    """Test deployed endpoint"""
    test_prompt = "Enhance this query: Python developer with AWS experience"
    
    payload = {
        "inputs": test_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50
        }
    }
    
    response = predictor.predict(payload)
    print(f"Response: {response}")
    return response


def configure_auto_scaling(endpoint_name: str):
    """Configure auto-scaling for endpoint"""
    client = boto3.client('application-autoscaling', region_name=REGION)
    
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    
    # Register scalable target
    client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=1,
        MaxCapacity=10
    )
    
    # Create scaling policy
    client.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scaling-policy",
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 70.0,
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
            },
            'ScaleInCooldown': 300,
            'ScaleOutCooldown': 60
        }
    )
    
    print(f"Auto-scaling configured for {endpoint_name}")


if __name__ == "__main__":
    # Deploy query enhancer
    print("Deploying query enhancer model...")
    query_predictor = deploy_query_enhancer_model()
    
    # Configure auto-scaling
    configure_auto_scaling("adeptai-query-enhancer-v1")
    
    # Test endpoint
    print("\nTesting endpoint...")
    test_endpoint(query_predictor)
    
    print("\nDeployment completed successfully!")

