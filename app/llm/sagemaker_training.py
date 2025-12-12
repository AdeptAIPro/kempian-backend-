"""
SageMaker Training Service
Handles model training on AWS SageMaker
"""

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role
import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class SageMakerTrainingService:
    """Service for training models on SageMaker"""
    
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET", "kempian-llm-models")
        self.role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
        self.instance_type = os.getenv("TRAINING_INSTANCE_TYPE", "ml.g4dn.xlarge")
        self.use_spot = os.getenv("USE_SPOT_INSTANCES", "true").lower() == "true"
        
        # Initialize SageMaker session
        try:
            self.sess = sagemaker.Session()
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker session: {e}")
            raise
        
        # Get role
        if self.role_arn:
            self.role = self.role_arn
        else:
            try:
                self.role = get_execution_role()
            except Exception as e:
                logger.error(f"Failed to get SageMaker role: {e}")
                raise
    
    def prepare_training_data(
        self,
        training_data: List[Dict],
        output_path: str = None
    ) -> str:
        """Prepare and upload training data to S3"""
        if output_path is None:
            output_path = f"s3://{self.bucket}/training-data/"
        
        # Format as instruction-following dataset
        formatted_data = []
        for item in training_data:
            formatted_data.append({
                "instruction": item.get("input", ""),
                "input": json.dumps(item.get("context", {})),
                "output": item.get("output", "")
            })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(formatted_data, f, indent=2)
            temp_path = f.name
        
        # Upload to S3
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"training-data/training_data_{timestamp}.json"
        
        try:
            self.sess.upload_data(
                path=temp_path,
                bucket=self.bucket,
                key_prefix="training-data/"
            )
            
            s3_path = f"s3://{self.bucket}/{s3_key}"
            logger.info(f"Training data uploaded to: {s3_path}")
            return s3_path
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def create_training_job(
        self,
        training_data_s3: str,
        model_version: str = None,
        hyperparameters: Dict = None
    ) -> str:
        """Create SageMaker training job"""
        
        if model_version is None:
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job_name = f"kempian-llm-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                "epochs": "5",
                "per_device_train_batch_size": "4",
                "per_device_eval_batch_size": "4",
                "gradient_accumulation_steps": "4",
                "learning_rate": "2e-4",
                "warmup_steps": "100",
                "max_length": "2048",
                "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
                "use_lora": "true",
                "lora_r": "16",
                "lora_alpha": "32",
                "lora_dropout": "0.05"
            }
        
        # Create HuggingFace estimator
        estimator = HuggingFace(
            entry_point="train.py",
            source_dir="training_scripts/",
            instance_type=self.instance_type,
            instance_count=1,
            role=self.role,
            transformers_version="4.35.0",
            pytorch_version="2.0.0",
            py_version="py310",
            hyperparameters=hyperparameters,
            use_spot_instances=self.use_spot,
            max_wait=3600 if self.use_spot else None,
            output_path=f"s3://{self.bucket}/model-artifacts/",
            base_job_name="kempian-llm",
            tags=[
                {"Key": "Project", "Value": "KempianLLM"},
                {"Key": "ModelVersion", "Value": model_version}
            ]
        )
        
        # Start training
        estimator.fit({"training": training_data_s3}, job_name=job_name)
        
        logger.info(f"Training job started: {job_name}")
        return job_name
    
    def register_model(
        self,
        model_artifact_s3: str,
        model_version: str,
        description: str = None
    ) -> str:
        """Register model in SageMaker Model Registry"""
        
        from sagemaker.model import Model
        
        model_name = f"kempian-llm-{model_version}"
        
        # Create model
        model = Model(
            model_data=model_artifact_s3,
            role=self.role,
            image_uri=None,  # Will use default HuggingFace image
            name=model_name
        )
        
        # Register in model registry
        model_package = model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.g4dn.xlarge", "ml.g4dn.2xlarge"],
            transform_instances=["ml.g4dn.xlarge"],
            model_package_group_name="kempian-llm-models",
            description=description or f"Kempian LLM Model {model_version}",
            approval_status="PendingManualApproval"
        )
        
        logger.info(f"Model registered: {model_package.model_package_arn}")
        return model_package.model_package_arn
    
    def deploy_model(
        self,
        model_package_arn: str,
        endpoint_name: str = None,
        instance_type: str = None
    ) -> str:
        """Deploy model to SageMaker endpoint"""
        
        from sagemaker import ModelPackage
        
        if endpoint_name is None:
            endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME", "kempian-llm-endpoint")
        
        if instance_type is None:
            instance_type = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.g4dn.xlarge")
        
        # Create model from package
        model = ModelPackage(
            role=self.role,
            model_package_arn=model_package_arn,
            sagemaker_session=self.sess
        )
        
        # Deploy
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            update_endpoint=True
        )
        
        logger.info(f"Model deployed to endpoint: {endpoint_name}")
        return endpoint_name

