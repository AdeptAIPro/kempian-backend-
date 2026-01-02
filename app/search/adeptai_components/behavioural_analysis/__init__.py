"""
Behavioural Analysis – Enhanced Modular AI Pipeline
--------------------------------------------------
Next-generation behavioral analysis with cutting-edge AI integrations:

🚀 Core Features:
- Domain-specific BERT models (BioBERT, SciBERT, CodeBERT, FinBERT, Legal-BERT)
- Multi-source profile analysis (Resume, LinkedIn, GitHub, Portfolio, References)
- Graph Neural Networks for career trajectory analysis
- Enhanced semantic matching and behavioral scoring
- Multi-model validation system

📦 Usage:
    from .pipeline import create_enhanced_pipeline, create_multi_source_profile
    
    # Create multi-source profile
    profile = create_multi_source_profile(
        resume_text="...",
        linkedin_data={...},
        github_data={...}
    )
    
    # Create pipeline with all cutting-edge features
    pipeline = create_enhanced_pipeline({
        "enable_gnn": True,
        "enable_validation": True
    })
    
    # Run comprehensive analysis
    results = pipeline.analyze_comprehensive_profile(
        source_data=profile,
        target_role="Senior Software Engineer",
        job_description="..."
    )
    
    # Access new insights
    print(f"Domain: {results['detected_domain']}")
    print(f"Overall Score: {results['behavioral_profile'].overall_score}")
    print(f"Career Trajectory: {results['career_trajectory']}")

🔧 Components Available:
"""

__version__ = "3.0.0-enhanced-semantic"

__all__ = [
    # Core pipeline
    "pipeline",
    "BehavioralPipeline",           # Original class (backwards compatible)
    "EnhancedBehavioralPipeline",   # New enhanced class
    "create_enhanced_pipeline",     # Factory function
    "create_multi_source_profile",  # Multi-source data helper
    
    # Original components (enhanced)
    "entity_extractor",
    "semantic_analyzer", 
    "emotion_analyzer",
    "behavioral_scorer",
    
    # New cutting-edge components
    "domain_bert",
    "career_gnn",
    
    # New analysis components
    "SemanticSkillAnalyzer",
    "CareerTrajectoryAnalyzer", 
    "MultiModelBehavioralValidator",
    
    # Data structures
    "MultiSourceProfile",
    "BehavioralProfile",
    "CareerTrajectory",
    "ProfileDataSource",
    
    # Utility classes
    "DomainType",
    "DomainSpecificBERT",
    "CareerGraphGNN",
    "CareerGraphBuilder",
    "PretrainedCareerGNN",
    "CareerNode",
]

# Initialize variables to None first
BehavioralPipeline = None
EnhancedBehavioralPipeline = None
create_enhanced_pipeline = None
create_multi_source_profile = None

# Import core components for easy access
try:
    from .pipeline import (
        BehavioralPipeline,
        EnhancedBehavioralPipeline, 
        create_enhanced_pipeline,
        create_multi_source_profile,
        SemanticSkillAnalyzer,
        CareerTrajectoryAnalyzer,
        MultiModelBehavioralValidator,
        MultiSourceProfile,
        BehavioralProfile,
        CareerTrajectory,
        ProfileDataSource
    )
    
    from .domain_bert import DomainType, DomainSpecificBERT
    from .career_gnn import CareerGraphGNN, CareerGraphBuilder, PretrainedCareerGNN, CareerNode
    
except ImportError as e:
    print(f"Warning: Some enhanced components may not be available: {e}")
    print("Install additional dependencies: pip install torch torch-geometric transformers")
    
    # Fallback to original components only
    try:
        from .pipeline import BehavioralPipeline
    except ImportError:
        print("Error: Core pipeline components not available")
        # Create a minimal fallback class
        class BehavioralPipeline:
            """Fallback behavioral analysis pipeline"""
            
            def __init__(self):
                self.name = "Fallback Behavioral Pipeline"
                self.version = "1.0.0-fallback"
            
            def analyze(self, resume_text, job_description):
                """Basic analysis fallback"""
                return {
                    'final_score': 0.7,
                    'behavioral_score': 0.7,
                    'technical_score': 0.7,
                    'cultural_fit': 0.7,
                    'risk_factors': [],
                    'strengths': ['Basic analysis available'],
                    'recommendations': ['Consider upgrading to enhanced pipeline'],
                    'pipeline_type': 'fallback'
                }

# Create fallback function if import failed
if create_enhanced_pipeline is None:
    def create_enhanced_pipeline(config=None):
        """Fallback enhanced pipeline creation"""
        if config is None:
            config = {"enable_gnn": False, "enable_validation": False}
        
        try:
            return BehavioralPipeline()
        except Exception as e:
            print(f"Failed to create enhanced pipeline: {e}")
            return BehavioralPipeline()

if create_multi_source_profile is None:
    def create_multi_source_profile(resume_text, **kwargs):
        """Fallback multi-source profile creation"""
        return MultiSourceProfile(resume_text=resume_text) if 'MultiSourceProfile' in globals() else None

# Metadata for different deployment scenarios
DEPLOYMENT_CONFIGS = {
    "lightweight": {
        "enable_gnn": False,
        "enable_validation": False,
        "description": "Original pipeline only - minimal resource usage"
    },
    
    "domain_enhanced": {
        "enable_gnn": False, 
        "enable_validation": True,
        "description": "Domain-specific BERT with validation - moderate resource usage"
    },
    
    "full_featured": {
        "enable_gnn": True,
        "enable_validation": True,
        "description": "All cutting-edge features - high resource usage"
    },
    
    "production": {
        "enable_gnn": True,
        "enable_validation": True,
        "description": "Production-ready with all analysis features"
    }
}

def get_pipeline(config_name: str = "lightweight"):
    """
    Get preconfigured pipeline for different use cases
    
    Args:
        config_name: One of 'lightweight', 'domain_enhanced', 'full_featured', 'production'
    
    Returns:
        Configured pipeline instance
    """
    if config_name not in DEPLOYMENT_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(DEPLOYMENT_CONFIGS.keys())}")
    
    config = DEPLOYMENT_CONFIGS[config_name]
    print(f"Loading pipeline: {config['description']}")
    
    try:
        return create_enhanced_pipeline(config)
    except Exception as e:
        print(f"Failed to create enhanced pipeline: {e}")
        print("Falling back to original pipeline...")
        return BehavioralPipeline()

# Quick start examples
QUICK_START_EXAMPLES = {
    "basic_usage": '''
# Basic Usage (Original Pipeline - Backwards Compatible)
from behavioural_analysis import BehavioralPipeline

pipeline = BehavioralPipeline()
results = pipeline.analyze(resume_text, job_description)
print(f"Final Score: {results['final_score']}")
''',

    "enhanced_usage": '''
# Enhanced Usage (Multi-Source Analysis)
from behavioural_analysis import create_enhanced_pipeline, create_multi_source_profile

# Create multi-source profile
profile_data = create_multi_source_profile(
    resume_text=resume_text,
    linkedin_data={'summary': '...', 'experience_descriptions': [...]},
    github_data={'bio': '...', 'project_descriptions': [...]}
)

# Create enhanced pipeline
pipeline = create_enhanced_pipeline({
    "enable_gnn": True,
    "enable_validation": True
})

# Comprehensive analysis
results = pipeline.analyze_comprehensive_profile(
    source_data=profile_data,
    target_role="Senior Software Engineer",
    job_description=job_description
)

print(f"Overall Score: {results['behavioral_profile'].overall_score}")
print(f"Leadership Score: {results['behavioral_profile'].leadership_score}")
print(f"Career Trajectory: {results['career_trajectory']}")
print(f"Validation Confidence: {results['validation_results']['overall_confidence']}")
''',

    "production_ready": '''
# Production Configuration
from behavioural_analysis import get_pipeline, create_multi_source_profile

# Use preconfigured production setup
pipeline = get_pipeline("production")

# Create comprehensive profile
profile_data = create_multi_source_profile(
    resume_text=resume_text,
    linkedin_data=linkedin_data,  # Optional
    github_data=github_data,      # Optional
    portfolio_data=portfolio_data # Optional
)

# Full analysis
results = pipeline.analyze_comprehensive_profile(
    source_data=profile_data,
    target_role=target_role,
    job_description=job_description
)

# Access comprehensive insights
behavioral_profile = results['behavioral_profile']
career_insights = results['career_trajectory']
role_fit = results['role_fit_analysis']
'''
}

def show_examples():
    """Display quick start examples"""
    for name, code in QUICK_START_EXAMPLES.items():
        print(f"\n=== {name.replace('_', ' ').title()} ===")
        print(code)

# Module info
def info():
    """Display module information"""
    print(f"🚀 Behavioral Analysis Pipeline v{__version__}")
    print("\n📦 Available Configurations:")
    for name, config in DEPLOYMENT_CONFIGS.items():
        print(f"  - {name}: {config['description']}")
    
    print(f"\n🔧 Components: {len(__all__)} available")
    print("\n💡 Quick start: run show_examples() for usage examples")
    print("\n🔄 Migration Guide:")
    print("  - Old code using BehavioralPipeline.analyze() still works")
    print("  - New code can use create_enhanced_pipeline() for advanced features")
    print("  - Multi-source analysis available via analyze_comprehensive_profile()")