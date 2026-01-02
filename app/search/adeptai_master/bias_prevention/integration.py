from .sanitizer import ResumeSanitizer, QuerySanitizer
from .monitor import BiasMonitor
from .config import PROTECTED_CHARACTERISTICS, MONITORING_CONFIG, ALERT_CONFIG

def integrate_bias_prevention(search_system):
    """Wrap a search system with race, culture, and religion bias prevention"""
    sanitizer = ResumeSanitizer()
    q_sanitizer = QuerySanitizer()
    monitor = BiasMonitor()

    original_search = getattr(search_system, 'search', None)
    if not original_search:
        raise ValueError("Search system missing 'search' method")

    def bias_safe_search(query, top_k=10, **kwargs):
        """Enhanced search with race, culture, and religion bias prevention"""
        # Sanitize query to remove bias terms
        sanitized_query = q_sanitizer.sanitize_query(query)
        sanitization_report = q_sanitizer.get_sanitization_report(query)
        
        # Perform search with sanitized query
        results = original_search(sanitized_query, top_k, **kwargs)
        
        # Assess diversity and representation balance
        if isinstance(results, list):
            diversity_assessment = monitor.assess_diversity(sanitized_query, results)
            
            # Store assessment results
            search_system.latest_diversity_assessment = diversity_assessment
            search_system.latest_sanitization_report = sanitization_report
            
            # Check for bias incidents and trigger alerts if configured
            if ALERT_CONFIG['underrepresentation_alerts'] and diversity_assessment['bias_flags']:
                search_system.bias_incidents_detected = diversity_assessment['bias_flags']
                
            # Store monitoring data for compliance reporting
            if MONITORING_CONFIG['compliance_reporting']:
                search_system.compliance_data = {
                    'query': sanitized_query,
                    'original_query': query,
                    'diversity_assessment': diversity_assessment,
                    'sanitization_report': sanitization_report,
                    'timestamp': diversity_assessment['timestamp']
                }
        
        return results

    def bias_safe_resume_search(resume_data, top_k=10, **kwargs):
        """Search with resume data sanitization for race, culture, and religion bias"""
        # Sanitize resume data
        sanitized_resume = sanitizer.sanitize_resume(resume_data)
        
        # Perform search with sanitized resume
        results = original_search(sanitized_resume, top_k, **kwargs)
        
        # Assess diversity of results
        if isinstance(results, list):
            diversity_assessment = monitor.assess_diversity("resume_search", results)
            search_system.latest_resume_diversity_assessment = diversity_assessment
            
            # Store sanitization details
            search_system.resume_sanitization_details = sanitized_resume.get('sanitization_details', {})
        
        return results

    def get_diversity_metrics():
        """Get current diversity metrics and monitoring summary"""
        return {
            'monitoring_summary': monitor.get_monitoring_summary(),
            'protected_characteristics': PROTECTED_CHARACTERISTICS,
            'monitoring_config': MONITORING_CONFIG,
            'alert_config': ALERT_CONFIG
        }

    def export_compliance_report(filepath=None):
        """Export compliance and diversity monitoring data"""
        return monitor.export_monitoring_data(filepath)

    def check_cultural_sensitivity(text):
        """Check text for cultural sensitivity issues"""
        from .utils import check_cultural_sensitivity
        return check_cultural_sensitivity(text)

    def get_religious_accommodation_guidelines():
        """Get religious accommodation guidelines"""
        from .utils import get_religious_accommodation_guidelines
        return get_religious_accommodation_guidelines()

    # Replace original search method
    search_system.search = bias_safe_search
    
    # Add new methods for race, culture, and religion focus
    search_system.bias_safe_resume_search = bias_safe_resume_search
    search_system.get_diversity_metrics = get_diversity_metrics
    search_system.export_compliance_report = export_compliance_report
    search_system.check_cultural_sensitivity = check_cultural_sensitivity
    search_system.get_religious_accommodation_guidelines = get_religious_accommodation_guidelines
    
    # Store monitoring and configuration references
    search_system.bias_monitor = monitor
    search_system.bias_sanitizer = sanitizer
    search_system.query_sanitizer = q_sanitizer
    search_system.protected_characteristics_config = PROTECTED_CHARACTERISTICS
    
    return search_system
