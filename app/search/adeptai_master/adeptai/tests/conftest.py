"""
Pytest configuration and fixtures
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import Settings


@pytest.fixture
def test_settings():
    """Fixture providing test settings"""
    return Settings(
        secret_key="test-secret-key",
        debug=True,
        testing=True,
        log_level="DEBUG",
        cors_allowed_origins=["http://localhost:3000", "http://localhost:5055"],
        max_content_length=1024 * 1024,  # 1MB for testing
        search_index_path="test_search_index",
        aws_region="us-west-2",
        dynamodb_table_name="test_table",
        dynamodb_feedback_table="test_feedback_table"
    )


@pytest.fixture
def mock_flask_app():
    """Fixture providing a mock Flask app"""
    app = Mock()
    app.extensions = {}
    app.config = {}
    app.register_blueprint = Mock()
    return app


@pytest.fixture
def mock_service_container():
    """Fixture providing a mock service container"""
    container = Mock()
    container._services = {}
    container._initialized = False
    container.register = Mock()
    container.get = Mock(return_value=None)
    container.is_initialized = Mock(return_value=False)
    container.initialize_services = Mock()
    return container


@pytest.fixture
def mock_search_system():
    """Fixture providing a mock search system"""
    search_system = Mock()
    search_system.search = Mock(return_value=[{"id": "1", "score": 0.9}])
    search_system.get_performance_stats = Mock(return_value={"searches": 10, "avg_time": 0.1})
    return search_system


@pytest.fixture
def mock_behavioral_pipeline():
    """Fixture providing a mock behavioral analysis pipeline"""
    pipeline = Mock()
    pipeline.analyze = Mock(return_value={"score": 0.8, "confidence": 0.9})
    pipeline.analyze_comprehensive_profile = Mock(return_value={
        "behavioral_profile": {"overall_score": 0.8},
        "career_trajectory": {"trend": "upward"},
        "detected_domain": "technology"
    })
    return pipeline


@pytest.fixture
def mock_ml_service():
    """Fixture providing a mock ML service"""
    ml_service = Mock()
    ml_service.predict = Mock(return_value=0.85)
    ml_service.get_feature_importance = Mock(return_value={"feature1": 0.3, "feature2": 0.7})
    return ml_service


@pytest.fixture
def mock_embedding_service():
    """Fixture providing a mock embedding service"""
    embedding_service = Mock()
    embedding_service.encode_single = Mock(return_value=[0.1, 0.2, 0.3])
    embedding_service.encode_batch = Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return embedding_service


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    yield
    
    # Cleanup after test
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
    if 'LOG_LEVEL' in os.environ:
        del os.environ['LOG_LEVEL']


@pytest.fixture
def sample_search_request():
    """Fixture providing sample search request data"""
    return {
        "query": "python developer with 5 years experience",
        "top_k": 10,
        "filters": {
            "location": "San Francisco",
            "experience_level": "senior"
        }
    }


@pytest.fixture
def sample_candidate_data():
    """Fixture providing sample candidate data"""
    return {
        "email": "john.doe@example.com",
        "full_name": "John Doe",
        "skills": ["Python", "Django", "PostgreSQL", "AWS"],
        "experience_years": 5,
        "location": "San Francisco, CA",
        "resume_text": "Experienced Python developer with 5 years of experience..."
    }


@pytest.fixture
def sample_job_description():
    """Fixture providing sample job description"""
    return """
    We are looking for a Senior Python Developer to join our team.
    
    Requirements:
    - 5+ years of Python experience
    - Experience with Django or Flask
    - Knowledge of PostgreSQL
    - AWS experience preferred
    - Strong problem-solving skills
    
    Responsibilities:
    - Develop and maintain web applications
    - Collaborate with cross-functional teams
    - Write clean, maintainable code
    - Participate in code reviews
    """
