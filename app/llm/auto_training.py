"""
Auto-Training Service
Integrates with auto-evolution system to trigger training
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime
from .auto_evolution import AutoEvolutionSystem

logger = logging.getLogger(__name__)


class AutoTrainingService:
    """Service for automatic model training"""
    
    def __init__(self):
        self.evolution_system = AutoEvolutionSystem()
        try:
            from .sagemaker_training import SageMakerTrainingService
            self.training_service = SageMakerTrainingService()
        except Exception as e:
            logger.warning(f"SageMaker training service not available: {e}")
            self.training_service = None
    
    def check_and_train(self) -> Optional[Dict]:
        """Check if training is needed and trigger if so"""
        
        if self.training_service is None:
            return {
                "success": False,
                "error": "SageMaker training service not available"
            }
        
        # Check if should retrain
        if not self.evolution_system.should_retrain():
            logger.info("Not enough data for retraining yet")
            return None
        
        # Get training data
        training_data = self.evolution_system.get_training_data(
            min_quality=0.7,
            limit=5000
        )
        
        if len(training_data) < 100:
            logger.warning(f"Not enough high-quality data: {len(training_data)}")
            return None
        
        logger.info(f"Starting auto-training with {len(training_data)} examples")
        
        try:
            # Prepare training data
            training_data_s3 = self.training_service.prepare_training_data(
                training_data=training_data
            )
            
            # Create training job
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            job_name = self.training_service.create_training_job(
                training_data_s3=training_data_s3,
                model_version=model_version
            )
            
            # Log training run
            self.evolution_system.log_training_run(
                model_version=model_version,
                examples_used=len(training_data),
                training_loss=0.0,
                deployed=False
            )
            
            return {
                "success": True,
                "job_name": job_name,
                "model_version": model_version,
                "examples_used": len(training_data),
                "training_data_s3": training_data_s3
            }
            
        except Exception as e:
            logger.error(f"Auto-training failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def monitor_training_job(self, job_name: str) -> Dict:
        """Monitor training job status"""
        import boto3
        
        region = os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client('sagemaker', region_name=region)
        
        try:
            response = client.describe_training_job(TrainingJobName=job_name)
            
            status = response['TrainingJobStatus']
            training_loss = None
            validation_loss = None
            
            # Get metrics if available
            if 'FinalMetricDataList' in response:
                for metric in response['FinalMetricDataList']:
                    if metric['MetricName'] == 'train_loss':
                        training_loss = metric['Value']
                    elif metric['MetricName'] == 'eval_loss':
                        validation_loss = metric['Value']
            
            return {
                "status": status,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "creation_time": str(response.get('CreationTime')),
                "training_end_time": str(response.get('TrainingEndTime')) if response.get('TrainingEndTime') else None,
                "model_artifacts": response.get('ModelArtifacts', {}).get('S3ModelArtifacts')
            }
            
        except Exception as e:
            logger.error(f"Error monitoring training job: {e}")
            return {
                "status": "Unknown",
                "error": str(e)
            }
    
    def complete_training_workflow(
        self,
        job_name: str,
        model_version: str,
        auto_deploy: bool = False
    ) -> Dict:
        """Complete workflow after training: register and optionally deploy"""
        
        if self.training_service is None:
            return {
                "success": False,
                "error": "SageMaker training service not available"
            }
        
        # Get model artifacts
        import boto3
        import os
        
        region = os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client('sagemaker', region_name=region)
        
        try:
            response = client.describe_training_job(TrainingJobName=job_name)
            model_artifacts_s3 = response['ModelArtifacts']['S3ModelArtifacts']
            
            # Register model
            model_package_arn = self.training_service.register_model(
                model_artifact_s3=model_artifacts_s3,
                model_version=model_version,
                description=f"Auto-trained model from {len(self.evolution_system.get_training_data())} examples"
            )
            
            result = {
                "success": True,
                "model_package_arn": model_package_arn,
                "model_artifacts_s3": model_artifacts_s3
            }
            
            # Auto-deploy if requested
            if auto_deploy:
                endpoint_name = self.training_service.deploy_model(
                    model_package_arn=model_package_arn
                )
                result["endpoint_name"] = endpoint_name
                
                # Update training run as deployed
                self.evolution_system.log_training_run(
                    model_version=model_version,
                    examples_used=0,
                    training_loss=0.0,
                    deployed=True
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error completing training workflow: {e}")
            return {
                "success": False,
                "error": str(e)
            }

