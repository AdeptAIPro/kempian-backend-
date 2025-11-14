# Bias Prevention Configuration - Focused on Race, Culture, and Religion Only

# PROTECTED CHARACTERISTICS CONFIGURATION
PROTECTED_CHARACTERISTICS = {
    'race': {
        'enabled': True,
        'monitoring_threshold': 0.15,  # 15% minimum representation
        'high_risk_threshold': 0.05,   # 5% triggers high risk alert
        'categories': ['African/Black', 'Asian', 'Hispanic/Latino', 'Middle Eastern', 
                      'Native American/Indigenous', 'Pacific Islander', 'White/Caucasian']
    },
    'culture': {
        'enabled': True,
        'monitoring_threshold': 0.10,  # 10% minimum representation
        'high_risk_threshold': 0.03,   # 3% triggers high risk alert
        'categories': ['Cultural Traditions', 'Cultural Media', 'Cultural Heritage', 'Multicultural']
    },
    'religion': {
        'enabled': True,
        'monitoring_threshold': 0.08,  # 8% minimum representation
        'high_risk_threshold': 0.02,   # 2% triggers high risk alert
        'categories': ['Christian', 'Muslim/Islamic', 'Jewish', 'Hindu', 'Buddhist', 'Sikh', 'Non-religious']
    }
}

# INTERSECTIONAL DIVERSITY THRESHOLDS
INTERSECTIONAL_THRESHOLDS = {
    'race_culture': 0.05,     # 5% minimum for race+culture combinations
    'race_religion': 0.04,    # 4% minimum for race+religion combinations
    'culture_religion': 0.03, # 3% minimum for culture+religion combinations
    'triple_intersection': 0.02  # 2% minimum for race+culture+religion combinations
}

# CULTURAL SENSITIVITY SETTINGS
CULTURAL_SENSITIVITY = {
    'enabled': True,
    'awareness_level': 'high',  # low, medium, high
    'holiday_recognition': True,
    'cultural_practices_respect': True,
    'language_preferences': True,
    'traditional_customs': True
}

# RELIGIOUS ACCOMMODATION PREFERENCES
RELIGIOUS_ACCOMMODATION = {
    'enabled': True,
    'prayer_time_accommodation': True,
    'dietary_restrictions': True,
    'religious_holiday_observance': True,
    'dress_code_flexibility': True,
    'worship_space_consideration': True
}

# MONITORING AND REPORTING CONFIGURATION
MONITORING_CONFIG = {
    'real_time_monitoring': True,
    'daily_reports': True,
    'weekly_summaries': True,
    'monthly_analytics': True,
    'compliance_reporting': True,
    'audit_trail_retention_days': 365
}

# ALERT AND NOTIFICATION SETTINGS
ALERT_CONFIG = {
    'underrepresentation_alerts': True,
    'intersectional_bias_alerts': True,
    'cultural_sensitivity_alerts': True,
    'religious_accommodation_alerts': True,
    'alert_threshold': 'medium',  # low, medium, high
    'notification_channels': ['email', 'dashboard', 'api']
}

# COMPLIANCE FRAMEWORK SETTINGS
COMPLIANCE_FRAMEWORKS = {
    'eeo_compliance': True,      # Equal Employment Opportunity
    'ada_compliance': False,     # Americans with Disabilities Act (disabled)
    'age_discrimination': False, # Age Discrimination in Employment Act
    'gender_equity': False,      # Gender-based discrimination
    'lgbtq_protection': False,   # Sexual orientation and gender identity
    'veteran_protection': False  # Veteran status protection
}

# DATA PROCESSING CONFIGURATION
DATA_PROCESSING = {
    'anonymization_enabled': True,
    'pii_protection': True,
    'encryption_enabled': True,
    'data_retention_days': 90,
    'backup_frequency': 'daily'
}

# PERFORMANCE AND SCALABILITY SETTINGS
PERFORMANCE_CONFIG = {
    'batch_processing_size': 1000,
    'concurrent_processing': True,
    'cache_enabled': True,
    'cache_ttl_seconds': 3600,
    'max_query_complexity': 'high'
}

# LEGACY CONFIGURATION (DEPRECATED - FOR BACKWARD COMPATIBILITY)
REPLACEMENT_TOKENS = {
    'CANDIDATE_NAME': '[CANDIDATE]',
    'LOCATION': '[LOCATION]',
    'ORGANIZATION': '[ORG]',
    'SCHOOL': '[EDU_INST]',
    'PHONE': '[PHONE]',
    'EMAIL': 'candidate@domain.com'
}

# DEPRECATED - Use PROTECTED_CHARACTERISTICS instead
BIAS_CONFIG = {
    "anonymization_enabled": True,
    "query_sanitization_enabled": True,
    "monitoring_enabled": True,
    "fair_embedding_enabled": False
}
