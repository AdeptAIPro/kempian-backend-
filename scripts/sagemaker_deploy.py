"""
SageMaker Deployment Script
Deploys Kempian LLM to AWS SageMaker
"""

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role
import os
import sys
import json
from datetime import datetime

# Configuration
REGION = os.getenv("AWS_REGION", "ap-south-1")
BUCKET_NAME = os.getenv("S3_BUCKET", "")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::888577040598:role/SageMakerExecutionRoleProd")
MODEL_NAME = os.getenv("SAGEMAKER_MODEL_NAME", "kempian-llm")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", f"{MODEL_NAME}-endpoint")
INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.g4dn.xlarge")

def deploy_from_huggingface():
    """Deploy base model from Hugging Face"""
    print("=" * 60)
    print("Deploying Kempian LLM to SageMaker")
    print("=" * 60)
    
    # Initialize SageMaker session
    sess = sagemaker.Session()
    
    # Get role
    try:
        if ROLE_ARN:
            role = ROLE_ARN
            print(f"Using provided role: {role}")
        else:
            role = get_execution_role()
            print(f"Using default execution role: {role}")
    except Exception as e:
        print(f"Error getting role: {e}")
        print("\nPlease set SAGEMAKER_ROLE_ARN in environment or configure AWS credentials")
        sys.exit(1)
    
    print(f"Region: {REGION}")
    print(f"Instance Type: {INSTANCE_TYPE}")
    print(f"Endpoint Name: {ENDPOINT_NAME}")
    print()
    
    # Create HuggingFace model
    print("Creating HuggingFace model...")
    huggingface_model = HuggingFaceModel(
        model_data=None,  # Will use model_id instead
        role=role,
        transformers_version="4.37",
        pytorch_version="2.1",
        py_version="py310",
        model_server_workers=1,
        env={
            "HF_MODEL_ID": "mistralai/Mistral-7B-Instruct-v0.2",
            "HF_TASK": "text-generation",
            "MAX_INPUT_LENGTH": "2048",
            "MAX_TOTAL_TOKENS": "4096",
        }
    )
    
    # Deploy model
    print(f"\nDeploying to {INSTANCE_TYPE}...")
    print("⏳ This will take 10-15 minutes...")
    print("   (SageMaker is creating the endpoint)")
    print()
    
    try:
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type=INSTANCE_TYPE,
            endpoint_name=ENDPOINT_NAME,
            wait=True
        )
        
        print("\n" + "=" * 60)
        print("✅ Model deployed successfully!")
        print("=" * 60)
        print(f"Endpoint Name: {ENDPOINT_NAME}")
        try:
            sm_client = boto3.client("sagemaker", region_name=REGION)
            endpoint_arn = sm_client.describe_endpoint(
                EndpointName=ENDPOINT_NAME
            )["EndpointArn"]
            print(f"Endpoint ARN: {endpoint_arn}")
        except Exception as describe_err:
            print(f"⚠️  Could not fetch endpoint ARN automatically: {describe_err}")
        print(f"Region: {REGION}")
        print()
        print("Next steps:")
        print(f"1. Update backend/.env:")
        print(f"   USE_SAGEMAKER=true")
        print(f"   SAGEMAKER_ENDPOINT_NAME={ENDPOINT_NAME}")
        print(f"   AWS_REGION={REGION}")
        print("2. Restart backend server")
        print("3. Test the endpoint")
        print()
        
        return predictor
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check AWS credentials: aws configure")
        print("2. Verify IAM role has SageMaker permissions")
        print("3. Check service quotas in AWS Console")
        raise

def deploy_custom_model(model_path: str):
    """Deploy custom fine-tuned model"""
    print("=" * 60)
    print("Deploying Custom Fine-Tuned Model to SageMaker")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        sys.exit(1)
    
    if not BUCKET_NAME:
        print("❌ S3_BUCKET not set. Please set it in environment variables.")
        sys.exit(1)
    
    sess = sagemaker.Session()
    
    # Get role
    try:
        if ROLE_ARN:
            role = ROLE_ARN
        else:
            role = get_execution_role()
    except Exception as e:
        print(f"Error getting role: {e}")
        sys.exit(1)
    
    print(f"Model path: {model_path}")
    print(f"S3 Bucket: {BUCKET_NAME}")
    print(f"Region: {REGION}")
    print(f"Instance Type: {INSTANCE_TYPE}")
    print()
    
    # Upload model to S3
    print("Uploading model to S3...")
    model_s3_path = sess.upload_data(
        path=model_path,
        bucket=BUCKET_NAME,
        key_prefix=f"models/{MODEL_NAME}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    print(f"✅ Model uploaded to: {model_s3_path}")
    print()
    
    # Check for custom inference code
    inference_code_path = os.path.join(os.path.dirname(model_path), "..", "sagemaker_code")
    if not os.path.exists(inference_code_path):
        inference_code_path = "sagemaker_code"
    
    if os.path.exists(inference_code_path):
        print(f"Using custom inference code from: {inference_code_path}")
        entry_point = "inference.py"
        source_dir = inference_code_path
    else:
        print("⚠️  No custom inference code found, using default")
        entry_point = None
        source_dir = None
    
    # Create model
    print("Creating SageMaker model...")
    huggingface_model = HuggingFaceModel(
        model_data=model_s3_path,
        role=role,
        transformers_version="4.37",
        pytorch_version="2.1",
        py_version="py310",
        entry_point=entry_point,
        source_dir=source_dir,
        model_server_workers=1,
        env={
            "HF_TASK": "text-generation",
            "MAX_INPUT_LENGTH": "2048",
            "MAX_TOTAL_TOKENS": "4096",
        }
    )
    
    # Deploy
    print(f"\nDeploying to {INSTANCE_TYPE}...")
    print("⏳ This will take 10-15 minutes...")
    print()
    
    try:
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type=INSTANCE_TYPE,
            endpoint_name=ENDPOINT_NAME,
            wait=True
        )
        
        print("\n" + "=" * 60)
        print("✅ Custom model deployed successfully!")
        print("=" * 60)
        print(f"Endpoint Name: {ENDPOINT_NAME}")
        try:
            sm_client = boto3.client("sagemaker", region_name=REGION)
            endpoint_arn = sm_client.describe_endpoint(
                EndpointName=ENDPOINT_NAME
            )["EndpointArn"]
            print(f"Endpoint ARN: {endpoint_arn}")
        except Exception as describe_err:
            print(f"⚠️  Could not fetch endpoint ARN automatically: {describe_err}")
        print()
        
        return predictor
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        raise

def test_endpoint():
    """Test the deployed endpoint"""
    import boto3
    
    print("Testing endpoint...")
    
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=REGION)
    
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps({
                "prompt": "Extract job data: Need a senior React developer",
                "max_tokens": 200,
                "temperature": 0.7
            })
        )
        
        result = json.loads(response['Body'].read())
        print("✅ Endpoint is working!")
        print(f"Response: {result.get('response', '')[:200]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "custom":
        model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/kempian-llm-v1.0/final"
        deploy_custom_model(model_path)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test_endpoint()
    else:
        deploy_from_huggingface()
        print("\nTesting endpoint...")
        test_endpoint()

