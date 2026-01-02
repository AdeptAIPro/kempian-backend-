"""
Kempian LLM Routes
Flask routes for custom LLM API
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from .service import LLMService

logger = logging.getLogger(__name__)

# Create Blueprint
llm_bp = Blueprint('llm', __name__, url_prefix='/api/llm')

# Initialize LLM Service
llm_service = LLMService()


@llm_bp.route('/health', methods=['GET'])
def health_check():
    """Check LLM service health"""
    try:
        status = llm_service.get_health_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "unknown"
        }), 503


@llm_bp.route('/generate', methods=['POST'])
@jwt_required()
def generate():
    """
    Generate text from prompt
    
    Expected JSON:
    {
        "prompt": "Your prompt here",
        "max_tokens": 512,
        "temperature": 0.7,
        "context": {}
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                "success": False,
                "error": "Prompt is required"
            }), 400
        
        prompt = data['prompt']
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        context = data.get('context', {})
        
        # Generate response
        response = llm_service.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context
        )
        
        return jsonify({
            "success": True,
            "response": response,
            "model_version": llm_service.model_version
        }), 200
        
    except Exception as e:
        logger.error(f"Generate error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "response": "I encountered an error processing your request."
        }), 500


@llm_bp.route('/extract-job', methods=['POST'])
@jwt_required()
def extract_job():
    """
    Extract structured job data from requirement
    
    Expected JSON:
    {
        "requirement": "Need a senior React developer..."
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'requirement' not in data:
            return jsonify({
                "success": False,
                "error": "Requirement is required"
            }), 400
        
        requirement = data['requirement']
        result = llm_service.extract_job_from_requirement(requirement)
        
        return jsonify(result), 200 if result.get('success') else 500
        
    except Exception as e:
        logger.error(f"Extract job error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@llm_bp.route('/analyze-candidates', methods=['POST'])
@jwt_required()
def analyze_candidates():
    """
    Analyze candidates based on query
    
    Expected JSON:
    {
        "message": "Tell me about candidate 1",
        "candidates": [...],
        "job_description": "...",
        "conversation_history": [...]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "Message is required"
            }), 400
        
        message = data['message']
        candidates = data.get('candidates', [])
        job_description = data.get('job_description')
        conversation_history = data.get('conversation_history', [])
        
        result = llm_service.analyze_candidates(
            message=message,
            candidates=candidates,
            job_description=job_description,
            conversation_history=conversation_history
        )
        
        return jsonify(result), 200 if result.get('success') else 500
        
    except Exception as e:
        logger.error(f"Analyze candidates error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "response": "I encountered an error analyzing candidates."
        }), 500


@llm_bp.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        return jsonify({
            "success": True,
            "model_path": llm_service.config["base_model"],
            "model_version": llm_service.model_version,
            "device": llm_service.config["device"],
            "quantization": "4-bit" if llm_service.config["load_in_4bit"] else "none",
            "max_length": llm_service.config["max_length"],
            "loaded": llm_service.model_loaded
        }), 200
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@llm_bp.route('/training/check', methods=['POST'])
@jwt_required()
def check_and_train():
    """Check if training is needed and trigger if so"""
    try:
        from .auto_training import AutoTrainingService
        service = AutoTrainingService()
        result = service.check_and_train()
        
        if result:
            return jsonify(result), 200
        else:
            return jsonify({
                "success": False,
                "message": "Not enough data for training"
            }), 200
            
    except Exception as e:
        logger.error(f"Training check error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@llm_bp.route('/training/status/<job_name>', methods=['GET'])
@jwt_required()
def training_status(job_name):
    """Get training job status"""
    try:
        from .auto_training import AutoTrainingService
        service = AutoTrainingService()
        status = service.monitor_training_job(job_name)
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Training status error: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@llm_bp.route('/training/deploy/<job_name>', methods=['POST'])
@jwt_required()
def deploy_trained_model(job_name):
    """Deploy trained model"""
    try:
        from .auto_training import AutoTrainingService
        service = AutoTrainingService()
        data = request.get_json()
        model_version = data.get('model_version')
        auto_deploy = data.get('auto_deploy', True)
        
        result = service.complete_training_workflow(
            job_name=job_name,
            model_version=model_version,
            auto_deploy=auto_deploy
        )
        
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Deploy error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
