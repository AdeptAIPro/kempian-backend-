"""
AI Routes for Kempian Platform
RESTful API endpoints for AI functionality
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from .service import AIService

logger = logging.getLogger(__name__)

# Create Blueprint
ai_bp = Blueprint('ai', __name__, url_prefix='/api/ai')

# Initialize AI Service
ai_service = AIService()

@ai_bp.route('/health', methods=['GET'])
def health_check():
    """Check AI service health status"""
    try:
        status = ai_service.get_health_status()
        return jsonify(status), 200 if status['service_status'] == 'healthy' else 503
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "service_status": "unhealthy",
            "error": str(e),
            "timestamp": "unknown"
        }), 503

@ai_bp.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    """
    Main chat endpoint for AI conversations
    
    Expected JSON payload:
    {
        "message": "User's message",
        "context": {
            "user_profile": {...},
            "job_data": {...},
            "conversation_history": "..."
        },
        "options": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
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
        context = data.get('context', {})
        options = data.get('options', {})
        
        # Add user ID to context
        user_id = get_jwt_identity()
        context['user_id'] = user_id
        
        # Generate AI response
        result = ai_service.generate_response(
            prompt=message,
            context=context,
            temperature=options.get('temperature'),
            max_tokens=options.get('max_tokens')
        )
        
        return jsonify(result), 200 if result['success'] else 500
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "response": "I encountered an error processing your request."
        }), 500

@ai_bp.route('/analyze-resume', methods=['POST'])
@jwt_required()
def analyze_resume():
    """
    Analyze resume and provide optimization suggestions
    
    Expected JSON payload:
    {
        "resume_text": "Resume content...",
        "job_description": "Optional job description..."
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'resume_text' not in data:
            return jsonify({
                "success": False,
                "error": "Resume text is required"
            }), 400
        
        resume_text = data['resume_text']
        job_description = data.get('job_description')
        
        # Analyze resume
        result = ai_service.analyze_resume(resume_text, job_description)
        
        return jsonify(result), 200 if result['success'] else 500
        
    except Exception as e:
        logger.error(f"Resume analysis error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "response": "I encountered an error analyzing your resume."
        }), 500

@ai_bp.route('/job-recommendations', methods=['POST'])
@jwt_required()
def job_recommendations():
    """
    Generate job recommendations based on user profile
    
    Expected JSON payload:
    {
        "user_profile": {
            "skills": ["Python", "React", "AWS"],
            "experience": "3 years",
            "preferences": {...}
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'user_profile' not in data:
            return jsonify({
                "success": False,
                "error": "User profile is required"
            }), 400
        
        user_profile = data['user_profile']
        
        # Generate recommendations
        result = ai_service.generate_job_recommendations(user_profile)
        
        return jsonify(result), 200 if result['success'] else 500
        
    except Exception as e:
        logger.error(f"Job recommendations error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "response": "I encountered an error generating recommendations."
        }), 500

@ai_bp.route('/interview-prep', methods=['POST'])
@jwt_required()
def interview_preparation():
    """
    Generate interview questions and preparation tips
    
    Expected JSON payload:
    {
        "job_title": "Software Engineer",
        "company": "Tech Corp",
        "user_skills": ["Python", "React", "AWS"]
    }
    """
    try:
        data = request.get_json()
        
        required_fields = ['job_title', 'company', 'user_skills']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"{field} is required"
                }), 400
        
        job_title = data['job_title']
        company = data['company']
        user_skills = data['user_skills']
        
        # Generate interview prep
        result = ai_service.prepare_interview_questions(job_title, company, user_skills)
        
        return jsonify(result), 200 if result['success'] else 500
        
    except Exception as e:
        logger.error(f"Interview prep error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "response": "I encountered an error preparing interview questions."
        }), 500

@ai_bp.route('/models', methods=['GET'])
def get_models():
    """Get list of available AI models"""
    try:
        models = ai_service.get_available_models()
        return jsonify({
            "success": True,
            "models": models,
            "current_model": ai_service.model_name
        }), 200
    except Exception as e:
        logger.error(f"Get models error: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve models"
        }), 500

@ai_bp.route('/test', methods=['POST'])
def test_ai():
    """
    Test endpoint for AI functionality
    No authentication required for testing
    """
    try:
        data = request.get_json() or {}
        message = data.get('message', 'Hello! Can you help me with my job search?')
        
        # Simple test without context
        result = ai_service.generate_response(message)
        
        return jsonify({
            "success": True,
            "test_message": message,
            "ai_response": result
        }), 200
        
    except Exception as e:
        logger.error(f"AI test error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@ai_bp.route('/test-resume', methods=['POST'])
def test_resume_analysis():
    """
    Test endpoint for resume analysis
    No authentication required for testing
    """
    try:
        data = request.get_json() or {}
        resume_text = data.get('resume_text', '')
        job_description = data.get('job_description')
        
        if not resume_text:
            return jsonify({
                "success": False,
                "error": "Resume text is required"
            }), 400
        
        # Analyze resume
        result = ai_service.analyze_resume(resume_text, job_description)
        
        return jsonify({
            "success": True,
            "test_resume": resume_text[:100] + "..." if len(resume_text) > 100 else resume_text,
            "ai_response": result
        }), 200
        
    except Exception as e:
        logger.error(f"Resume test error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@ai_bp.route('/test-job-recommendations', methods=['POST'])
def test_job_recommendations():
    """
    Test endpoint for job recommendations
    No authentication required for testing
    """
    try:
        data = request.get_json() or {}
        user_profile = data.get('user_profile', {
            "skills": ["Python", "React", "AWS"],
            "experience": "3 years",
            "preferences": "Remote work, startup environment"
        })
        
        # Generate recommendations
        result = ai_service.generate_job_recommendations(user_profile)
        
        return jsonify({
            "success": True,
            "test_profile": user_profile,
            "ai_response": result
        }), 200
        
    except Exception as e:
        logger.error(f"Job recommendations test error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@ai_bp.route('/test-interview-prep', methods=['POST'])
def test_interview_preparation():
    """
    Test endpoint for interview preparation
    No authentication required for testing
    """
    try:
        data = request.get_json() or {}
        job_title = data.get('job_title', 'Software Engineer')
        company = data.get('company', 'Tech Corp')
        user_skills = data.get('user_skills', ['Python', 'React', 'AWS'])
        
        # Generate interview prep
        result = ai_service.prepare_interview_questions(job_title, company, user_skills)
        
        return jsonify({
            "success": True,
            "test_data": {
                "job_title": job_title,
                "company": company,
                "user_skills": user_skills
            },
            "ai_response": result
        }), 200
        
    except Exception as e:
        logger.error(f"Interview prep test error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
