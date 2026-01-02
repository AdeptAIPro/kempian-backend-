"""
Deployment Script for Hugging Face Models on SageMaker
Automates deployment of Hugging Face models to SageMaker endpoints
"""

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel, HuggingFace
from sagemaker.huggingface.model import HuggingFacePredictor
from typing import Dict, Any, Optional
import os
import json

# SageMaker session
session = sagemaker.Session()
role = os.getenv('SAGEMAKER_ROLE', 'arn:aws:iam::<account-id>:role/SageMakerExecutionRole')
region = os.getenv('AWS_REGION', 'us-east-1')

# Hugging Face token
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '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')

# Model configurations
MODEL_CONFIGS = {
    'query_enhancer': {
        'model_id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'instance_type': 'ml.g5.xlarge',
        'endpoint_name': 'adeptai-query-enhancer-v1',
        'max_tokens': 512,
        'temperature': 0.3
    },
    'behavioral_analyzer': {
        'model_id': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'instance_type': 'ml.g5.12xlarge',
        'endpoint_name': 'adeptai-behavioral-analyzer-v1',
        'max_tokens': 1024,
        'temperature': 0.7
    },
    'market_intelligence': {
        'model_id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'instance_type': 'ml.g5.2xlarge',
        'endpoint_name': 'adeptai-market-intelligence-v1',
        'max_tokens': 2048,
        'temperature': 0.7
    },
    'job_parser': {
        'model_id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'instance_type': 'ml.g5.xlarge',
        'endpoint_name': 'adeptai-job-parser-v1',
        'max_tokens': 512,
        'temperature': 0.3
    },
    'explanation_generator': {
        'model_id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'instance_type': 'ml.g5.xlarge',
        'endpoint_name': 'adeptai-explanation-generator-v1',
        'max_tokens': 512,
        'temperature': 0.7
    }
}


def deploy_huggingface_model(
    model_id: str,
    endpoint_name: str,
    instance_type: str,
    initial_instance_count: int = 1,
    max_instance_count: int = 10
) -> HuggingFacePredictor:
    """
    Deploy Hugging Face model to SageMaker endpoint
    
    Args:
        model_id: Hugging Face model ID (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
        endpoint_name: SageMaker endpoint name
        instance_type: SageMaker instance type
        initial_instance_count: Initial instance count
        max_instance_count: Maximum instance count for auto-scaling
        
    Returns:
        HuggingFacePredictor instance
    """
    print(f"Deploying Hugging Face model: {model_id}")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Instance type: {instance_type}")
    
    # Create HuggingFace model using SageMaker's managed container
    # No Docker image needed - SageMaker provides pre-built Hugging Face containers
    huggingface_model = HuggingFaceModel(
        model_data=None,  # None = download from Hugging Face Hub during deployment
        role=role,
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
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        endpoint_config_name=f"{endpoint_name}-config",
        wait=True
    )
    
    print(f"Endpoint deployed successfully: {endpoint_name}")
    return predictor


def deploy_all_models():
    """Deploy all configured models"""
    predictors = {}
    
    for use_case, config in MODEL_CONFIGS.items():
        try:
            predictor = deploy_huggingface_model(
                model_id=config['model_id'],
                endpoint_name=config['endpoint_name'],
                instance_type=config['instance_type']
            )
            predictors[use_case] = predictor
            print(f"✓ {use_case} deployed successfully\n")
        except Exception as e:
            print(f"✗ {use_case} deployment failed: {e}\n")
    
    return predictors


def update_endpoint_config(
    endpoint_name: str,
    instance_type: Optional[str] = None,
    initial_instance_count: int = 1
):
    """
    Update endpoint configuration
    
    Args:
        endpoint_name: Endpoint name
        instance_type: New instance type (optional)
        initial_instance_count: New instance count
    """
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    # Get current endpoint config
    endpoint_info = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    current_config = endpoint_info['EndpointConfigName']
    
    # Get current endpoint config details
    config_info = sagemaker_client.describe_endpoint_config(EndpointConfigName=current_config)
    production_variant = config_info['ProductionVariants'][0]
    
    # Update instance type if provided
    if instance_type:
        production_variant['InstanceType'] = instance_type
    
    # Update instance count
    production_variant['InitialInstanceCount'] = initial_instance_count
    
    # Create new endpoint config
    new_config_name = f"{current_config}-v2"
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=new_config_name,
        ProductionVariants=[production_variant]
    )
    
    # Update endpoint
    sagemaker_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=new_config_name
    )
    
    print(f"Endpoint configuration updated: {endpoint_name}")


def configure_auto_scaling(endpoint_name: str, min_capacity: int = 1, max_capacity: int = 10):
    """
    Configure auto-scaling for endpoint
    
    Args:
        endpoint_name: Endpoint name
        min_capacity: Minimum instance count
        max_capacity: Maximum instance count
    """
    client = boto3.client('application-autoscaling', region_name=region)
    
    # Register scalable target
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    
    try:
        client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity
        )
        print(f"Scalable target registered: {resource_id}")
    except client.exceptions.ValidationException:
        print(f"Scalable target already registered: {resource_id}")
    
    # Create scaling policy
    try:
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
        print(f"Auto-scaling policy created: {endpoint_name}")
    except Exception as e:
        print(f"Error creating auto-scaling policy: {e}")


def list_deployed_endpoints():
    """List all deployed endpoints"""
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    endpoints = sagemaker_client.list_endpoints(
        NameContains='adeptai',
        MaxResults=100
    )
    
    print("Deployed endpoints:")
    for endpoint in endpoints.get('Endpoints', []):
        status = endpoint['EndpointStatus']
        name = endpoint['EndpointName']
        print(f"  - {name}: {status}")
    
    return endpoints.get('Endpoints', [])


def delete_endpoint(endpoint_name: str):
    """Delete SageMaker endpoint"""
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint deletion initiated: {endpoint_name}")
    except Exception as e:
        print(f"Error deleting endpoint: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "deploy":
            # Deploy all models
            deploy_all_models()
        elif command == "list":
            # List endpoints
            list_deployed_endpoints()
        elif command == "deploy-single":
            # Deploy single model
            if len(sys.argv) < 3:
                print("Usage: deploy-single <use_case>")
                sys.exit(1)
            use_case = sys.argv[2]
            if use_case in MODEL_CONFIGS:
                config = MODEL_CONFIGS[use_case]
                deploy_huggingface_model(
                    model_id=config['model_id'],
                    endpoint_name=config['endpoint_name'],
                    instance_type=config['instance_type']
                )
        elif command == "configure-scaling":
            # Configure auto-scaling
            if len(sys.argv) < 3:
                print("Usage: configure-scaling <endpoint_name>")
                sys.exit(1)
            endpoint_name = sys.argv[2]
            configure_auto_scaling(endpoint_name)
        elif command == "delete":
            # Delete endpoint
            if len(sys.argv) < 3:
                print("Usage: delete <endpoint_name>")
                sys.exit(1)
            endpoint_name = sys.argv[2]
            delete_endpoint(endpoint_name)
        else:
            print(f"Unknown command: {command}")
            print("Available commands: deploy, deploy-single, list, configure-scaling, delete")
    else:
        print("Usage: python deploy_huggingface_models.py <command>")
        print("Commands:")
        print("  deploy            - Deploy all models")
        print("  deploy-single    - Deploy single model")
        print("  list             - List all endpoints")
        print("  configure-scaling - Configure auto-scaling")
        print("  delete           - Delete endpoint")

