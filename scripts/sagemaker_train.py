"""
SageMaker Training Script
Trains Kempian LLM on AWS SageMaker
"""

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role
import os
import sys
from datetime import datetime

# Configuration
REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET", "")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
TRAINING_DATA_S3 = os.getenv("TRAINING_DATA_S3", "")
INSTANCE_TYPE = os.getenv("TRAINING_INSTANCE_TYPE", "ml.g4dn.xlarge")
USE_SPOT = os.getenv("USE_SPOT_INSTANCES", "true").lower() == "true"
JOB_NAME = f"kempian-llm-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def train_on_sagemaker():
    """Train model on SageMaker"""
    
    print("=" * 60)
    print("SageMaker Training Job")
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
        print(f"❌ Error getting role: {e}")
        print("\nPlease set SAGEMAKER_ROLE_ARN in environment or configure AWS credentials")
        sys.exit(1)
    
    if not BUCKET_NAME:
        print("❌ S3_BUCKET not set")
        print("Set it: export S3_BUCKET=your-bucket-name")
        sys.exit(1)
    
    if not TRAINING_DATA_S3:
        print("❌ TRAINING_DATA_S3 not set")
        print("Set it: export TRAINING_DATA_S3=s3://bucket/training-data/")
        print("\nOr upload data first:")
        print("  aws s3 cp training_data.json s3://${S3_BUCKET}/training-data/")
        sys.exit(1)
    
    print(f"Region: {REGION}")
    print(f"Instance: {INSTANCE_TYPE}")
    print(f"Training Data: {TRAINING_DATA_S3}")
    print(f"Job Name: {JOB_NAME}")
    print(f"Use Spot Instances: {USE_SPOT} ({'70% cost savings' if USE_SPOT else 'no savings'})")
    print()
    
    # Check if training code directory exists
    training_code_dir = "sagemaker_training_code"
    if not os.path.exists(training_code_dir):
        print(f"❌ Training code directory not found: {training_code_dir}")
        print("\nPlease create it with train.py inside")
        print("See SAGEMAKER_TRAINING_GUIDE.md for details")
        sys.exit(1)
    
    # Create HuggingFace estimator
    print("Creating training job...")
    
    estimator_config = {
        'entry_point': 'train.py',
        'source_dir': training_code_dir,
        'instance_type': INSTANCE_TYPE,
        'instance_count': 1,
        'role': role,
        'transformers_version': '4.37',
        'pytorch_version': '2.1',
        'py_version': 'py310',
        'hyperparameters': {
            'epochs': 3,
            'train_batch_size': 4,
            'eval_batch_size': 4,
            'learning_rate': 2e-4,
            'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
        },
        'output_path': f's3://{BUCKET_NAME}/models/',
    }
    
    # Add spot instances if enabled
    if USE_SPOT:
        estimator_config['use_spot_instances'] = True
        estimator_config['max_wait'] = 36000  # 10 hours max wait
    
    huggingface_estimator = HuggingFace(**estimator_config)
    
    # Start training
    print("Starting training job...")
    print("⏳ This will take 2-6 hours depending on data size...")
    print(f"   Job name: {JOB_NAME}")
    print(f"   Monitor: aws sagemaker describe-training-job --training-job-name {JOB_NAME}")
    print()
    
    try:
        huggingface_estimator.fit(
            inputs={
                'training': TRAINING_DATA_S3
            },
            job_name=JOB_NAME,
            wait=True  # Wait for completion
        )
        
        print("\n" + "=" * 60)
        print("✅ Training complete!")
        print("=" * 60)
        print(f"Model artifacts: {huggingface_estimator.model_data}")
        print()
        print("Next steps:")
        print("1. Deploy model:")
        print(f"   export MODEL_S3_PATH={huggingface_estimator.model_data}")
        print(f"   python scripts/sagemaker_deploy.py custom $MODEL_S3_PATH")
        print()
        print("2. Or download and use locally:")
        print(f"   aws s3 cp {huggingface_estimator.model_data} ./models/")
        print()
        print("3. View training metrics:")
        print(f"   aws sagemaker describe-training-job --training-job-name {JOB_NAME}")
        
        return huggingface_estimator
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check CloudWatch logs:")
        print(f"   aws logs tail /aws/sagemaker/TrainingJobs/{JOB_NAME} --follow")
        print("2. Verify training data format")
        print("3. Check IAM role permissions")
        print("4. Verify instance type is available in region")
        raise

def list_training_jobs():
    """List recent training jobs"""
    sagemaker_client = boto3.client('sagemaker', region_name=REGION)
    
    response = sagemaker_client.list_training_jobs(
        MaxResults=10,
        SortBy='CreationTime',
        SortOrder='Descending'
    )
    
    print("Recent Training Jobs:")
    print("-" * 60)
    for job in response['TrainingJobSummaries']:
        status = job['TrainingJobStatus']
        name = job['TrainingJobName']
        created = job['CreationTime']
        
        status_icon = "✅" if status == "Completed" else "⏳" if status == "InProgress" else "❌"
        print(f"{status_icon} {name}")
        print(f"   Status: {status}")
        print(f"   Created: {created}")
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_training_jobs()
    else:
        train_on_sagemaker()

