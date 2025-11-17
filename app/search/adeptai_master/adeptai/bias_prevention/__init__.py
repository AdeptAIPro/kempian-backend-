# Bias Prevention Package for Recruitment System - Focused on Race, Culture, and Religion

# Import the core classes
from .sanitizer import QuerySanitizer, ResumeSanitizer
from .monitor import BiasMonitor
from .patterns import (
    RACE_PATTERNS, 
    CULTURE_PATTERNS, 
    RELIGION_PATTERNS, 
    PROTECTED_CHARACTERISTIC_PATTERNS
)
from .config import (
    PROTECTED_CHARACTERISTICS,
    INTERSECTIONAL_THRESHOLDS,
    CULTURAL_SENSITIVITY,
    RELIGIOUS_ACCOMMODATION,
    MONITORING_CONFIG,
    ALERT_CONFIG,
    COMPLIANCE_FRAMEWORKS
)
from .utils import (
    check_cultural_sensitivity,
    get_religious_accommodation_guidelines,
    analyze_protected_characteristics,
    generate_bias_report
)

# Make sanitizer available at module level as a class reference
# This allows bias_prevention.sanitizer.QuerySanitizer() to work
sanitizer = QuerySanitizer

# Export all public components
__all__ = [
    # Core Classes
    'QuerySanitizer',
    'ResumeSanitizer', 
    'BiasMonitor',
    'sanitizer',
    
    # Pattern Collections
    'RACE_PATTERNS',
    'CULTURE_PATTERNS',
    'RELIGION_PATTERNS',
    'PROTECTED_CHARACTERISTIC_PATTERNS',
    
    # Configuration
    'PROTECTED_CHARACTERISTICS',
    'INTERSECTIONAL_THRESHOLDS',
    'CULTURAL_SENSITIVITY',
    'RELIGIOUS_ACCOMMODATION',
    'MONITORING_CONFIG',
    'ALERT_CONFIG',
    'COMPLIANCE_FRAMEWORKS',
    
    # Utility Functions
    'check_cultural_sensitivity',
    'get_religious_accommodation_guidelines',
    'analyze_protected_characteristics',
    'generate_bias_report'
]

# Version information
__version__ = '2.0.0'
__description__ = 'Race, Culture, and Religion Focused Bias Prevention System'
__author__ = 'AdeptAI Team'
__last_updated__ = '2024'
