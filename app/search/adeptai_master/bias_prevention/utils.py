# Bias Prevention Utilities - Focused on Race, Culture, and Religion

import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# Import patterns for protected characteristics
from .patterns import RACE_PATTERNS, CULTURE_PATTERNS, RELIGION_PATTERNS

def check_cultural_sensitivity(text: str) -> Dict[str, Any]:
    """
    Check text for cultural sensitivity issues related to race, culture, and religion
    
    Args:
        text: Text to analyze for cultural sensitivity
        
    Returns:
        Dictionary containing sensitivity analysis results
    """
    if not isinstance(text, str):
        return {'error': 'Text must be a string', 'sensitivity_score': 0}
    
    text_lower = text.lower()
    sensitivity_issues = []
    sensitivity_score = 100  # Start with perfect score
    
    # Check for potentially insensitive language
    insensitive_patterns = {
        'racial_stereotypes': [
            r'\b(exotic|oriental|colored|ethnic|foreign)\b',
            r'\b(primitive|backward|uncivilized|savage)\b',
            r'\b(articulate|well-spoken)\s+(black|african|hispanic)\b'
        ],
        'cultural_appropriation': [
            r'\b(costume|party|theme)\s+(mexican|chinese|indian|arabic)\b',
            r'\b(cultural|ethnic)\s+(fashion|style|trend)\b',
            r'\b(traditional|authentic)\s+(dress|clothing|outfit)\b'
        ],
        'religious_insensitivity': [
            r'\b(heathen|infidel|pagan|heretic)\b',
            r'\b(religious\s+fanatic|extremist)\b',
            r'\b(backward\s+beliefs|primitive\s+religion)\b'
        ]
    }
    
    for category, patterns in insensitive_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                sensitivity_issues.append({
                    'category': category,
                    'pattern': pattern,
                    'severity': 'high' if 'stereotype' in category else 'medium'
                })
                sensitivity_score -= 20  # Deduct points for each issue
    
    # Check for cultural awareness indicators
    cultural_awareness_patterns = [
        r'\b(respect|honor|celebrate|acknowledge)\s+(culture|tradition|heritage)\b',
        r'\b(cultural\s+diversity|inclusion|representation)\b',
        r'\b(religious\s+freedom|accommodation|tolerance)\b'
    ]
    
    awareness_bonus = 0
    for pattern in cultural_awareness_patterns:
        if re.search(pattern, text_lower):
            awareness_bonus += 5
    
    sensitivity_score = max(0, min(100, sensitivity_score + awareness_bonus))
    
    return {
        'sensitivity_score': sensitivity_score,
        'issues_found': sensitivity_issues,
        'awareness_bonus': awareness_bonus,
        'recommendations': _generate_sensitivity_recommendations(sensitivity_issues),
        'analysis_timestamp': datetime.now().isoformat()
    }

def get_religious_accommodation_guidelines() -> Dict[str, Any]:
    """
    Get comprehensive religious accommodation guidelines for workplace settings
    
    Returns:
        Dictionary containing religious accommodation guidelines
    """
    guidelines = {
        'prayer_time_accommodation': {
            'muslim': {
                'daily_prayers': ['Fajr', 'Dhuhr', 'Asr', 'Maghrib', 'Isha'],
                'flexible_breaks': True,
                'prayer_space': 'Quiet, clean area',
                'special_considerations': 'Friday prayer (Jummah) may require longer break'
            },
            'jewish': {
                'daily_prayers': ['Shacharit', 'Mincha', 'Maariv'],
                'flexible_breaks': True,
                'prayer_space': 'Private area with mezuzah',
                'special_considerations': 'Shabbat restrictions on certain days'
            },
            'hindu': {
                'daily_prayers': ['Morning and evening prayers'],
                'flexible_breaks': True,
                'prayer_space': 'Clean, quiet area',
                'special_considerations': 'May need time for meditation'
            }
        },
        'dietary_restrictions': {
            'muslim': {
                'halal_diet': True,
                'pork_prohibition': True,
                'alcohol_prohibition': True,
                'ramadan_fasting': 'Adjust meal schedules during Ramadan'
            },
            'jewish': {
                'kosher_diet': True,
                'pork_prohibition': True,
                'shellfish_prohibition': True,
                'passover_restrictions': 'Special dietary needs during Passover'
            },
            'hindu': {
                'vegetarian_preference': True,
                'beef_prohibition': True,
                'alcohol_restriction': 'May avoid alcohol',
                'fasting_days': 'Various fasting days throughout year'
            }
        },
        'religious_holiday_observance': {
            'muslim': ['Eid al-Fitr', 'Eid al-Adha', 'Mawlid', 'Ashura'],
            'jewish': ['Rosh Hashanah', 'Yom Kippur', 'Passover', 'Hanukkah'],
            'hindu': ['Diwali', 'Holi', 'Raksha Bandhan', 'Navratri'],
            'buddhist': ['Vesak', 'Magha Puja', 'Asala Puja'],
            'christian': ['Christmas', 'Easter', 'Good Friday', 'Ash Wednesday']
        },
        'dress_code_flexibility': {
            'muslim': ['Hijab', 'Modest clothing', 'Religious head coverings'],
            'jewish': ['Kippah', 'Modest dress', 'Religious garments'],
            'hindu': ['Traditional clothing', 'Religious symbols', 'Modest dress'],
            'sikh': ['Turban', 'Kara', 'Modest clothing']
        }
    }
    
    return {
        'guidelines': guidelines,
        'last_updated': datetime.now().isoformat(),
        'version': '1.0',
        'compliance_note': 'These guidelines should be implemented in consultation with legal counsel'
    }

def analyze_protected_characteristics(text: str) -> Dict[str, Any]:
    """
    Analyze text for presence of race, culture, and religion indicators
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    if not isinstance(text, str):
        return {'error': 'Text must be a string'}
    
    analysis = {
        'race_indicators': [],
        'culture_indicators': [],
        'religion_indicators': [],
        'intersectional_combinations': [],
        'confidence_scores': {},
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    # Analyze race indicators
    for pattern in RACE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            analysis['race_indicators'].extend(matches)
    
    # Analyze culture indicators
    for pattern in CULTURE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            analysis['culture_indicators'].extend(matches)
    
    # Analyze religion indicators
    for pattern in RELIGION_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            analysis['religion_indicators'].extend(matches)
    
    # Find intersectional combinations
    if analysis['race_indicators'] and analysis['culture_indicators']:
        analysis['intersectional_combinations'].append('race_culture')
    if analysis['race_indicators'] and analysis['religion_indicators']:
        analysis['intersectional_combinations'].append('race_religion')
    if analysis['culture_indicators'] and analysis['religion_indicators']:
        analysis['intersectional_combinations'].append('culture_religion')
    if all([analysis['race_indicators'], analysis['culture_indicators'], analysis['religion_indicators']]):
        analysis['intersectional_combinations'].append('triple_intersection')
    
    # Calculate confidence scores
    analysis['confidence_scores'] = {
        'race': min(100, len(analysis['race_indicators']) * 20),
        'culture': min(100, len(analysis['culture_indicators']) * 20),
        'religion': min(100, len(analysis['religion_indicators']) * 20)
    }
    
    return analysis

def generate_bias_report(assessment_data: Dict[str, Any], report_type: str = 'comprehensive') -> Dict[str, Any]:
    """
    Generate bias report based on assessment data
    
    Args:
        assessment_data: Data from bias assessment
        report_type: Type of report to generate
        
    Returns:
        Dictionary containing formatted report
    """
    if report_type == 'executive_summary':
        return _generate_executive_summary(assessment_data)
    elif report_type == 'compliance':
        return _generate_compliance_report(assessment_data)
    elif report_type == 'detailed':
        return _generate_detailed_report(assessment_data)
    else:
        return _generate_comprehensive_report(assessment_data)

def _generate_executive_summary(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate executive summary report"""
    return {
        'report_type': 'executive_summary',
        'timestamp': datetime.now().isoformat(),
        'overall_diversity_score': assessment_data.get('diversity_score', 0),
        'bias_incidents_count': len(assessment_data.get('bias_flags', [])),
        'key_findings': _extract_key_findings(assessment_data),
        'recommendations': assessment_data.get('recommendations', [])[:3],  # Top 3
        'risk_level': _calculate_risk_level(assessment_data)
    }

def _generate_compliance_report(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate compliance-focused report"""
    return {
        'report_type': 'compliance',
        'timestamp': datetime.now().isoformat(),
        'eeo_compliance_status': _assess_eeo_compliance(assessment_data),
        'protected_characteristics_analysis': _analyze_protected_characteristics_compliance(assessment_data),
        'bias_incidents_details': assessment_data.get('bias_flags', []),
        'corrective_actions_required': _identify_corrective_actions(assessment_data),
        'compliance_score': _calculate_compliance_score(assessment_data)
    }

def _generate_detailed_report(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed technical report"""
    return {
        'report_type': 'detailed',
        'timestamp': datetime.now().isoformat(),
        'full_assessment_data': assessment_data,
        'statistical_analysis': _perform_statistical_analysis(assessment_data),
        'trend_analysis': _analyze_trends(assessment_data),
        'methodology': _document_methodology(),
        'data_sources': _document_data_sources(assessment_data)
    }

def _generate_comprehensive_report(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive report combining all aspects"""
    return {
        'report_type': 'comprehensive',
        'timestamp': datetime.now().isoformat(),
        'executive_summary': _generate_executive_summary(assessment_data),
        'compliance_analysis': _generate_compliance_report(assessment_data),
        'detailed_analysis': _generate_detailed_report(assessment_data),
        'action_plan': _generate_action_plan(assessment_data),
        'monitoring_schedule': _generate_monitoring_schedule()
    }

def _extract_key_findings(assessment_data: Dict[str, Any]) -> List[str]:
    """Extract key findings from assessment data"""
    findings = []
    
    diversity_score = assessment_data.get('diversity_score', 0)
    if diversity_score < 50:
        findings.append(f"Low overall diversity score: {diversity_score}")
    
    bias_flags = assessment_data.get('bias_flags', [])
    if bias_flags:
        findings.append(f"Found {len(bias_flags)} bias incidents requiring attention")
    
    diversity_metrics = assessment_data.get('diversity_metrics', {})
    for category, data in diversity_metrics.items():
        if isinstance(data, dict) and data.get('count', 0) > 0:
            findings.append(f"{category.title()} diversity: {data['count']} categories identified")
    
    return findings

def _calculate_risk_level(assessment_data: Dict[str, Any]) -> str:
    """Calculate overall risk level"""
    diversity_score = assessment_data.get('diversity_score', 0)
    bias_flags = assessment_data.get('bias_flags', [])
    
    if diversity_score < 30 or len(bias_flags) > 5:
        return 'high'
    elif diversity_score < 60 or len(bias_flags) > 2:
        return 'medium'
    else:
        return 'low'

def _assess_eeo_compliance(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assess EEO compliance status"""
    bias_flags = assessment_data.get('bias_flags', [])
    compliance_issues = []
    
    for flag in bias_flags:
        if flag.get('type') == 'underrepresentation':
            compliance_issues.append({
                'issue': f"Underrepresentation in {flag.get('category', 'unknown')}",
                'severity': flag.get('severity', 'unknown'),
                'recommendation': 'Implement diversity recruitment strategies'
            })
    
    return {
        'compliant': len(compliance_issues) == 0,
        'issues_found': compliance_issues,
        'compliance_score': max(0, 100 - len(compliance_issues) * 20)
    }

def _analyze_protected_characteristics_compliance(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze compliance for each protected characteristic"""
    diversity_metrics = assessment_data.get('diversity_metrics', {})
    compliance_analysis = {}
    
    for category in ['race', 'culture', 'religion']:
        if category in diversity_metrics:
            category_data = diversity_metrics[category]
            compliance_analysis[category] = {
                'representation_count': category_data.get('count', 0),
                'diversity_index': category_data.get('diversity_index', 0),
                'compliance_status': 'compliant' if category_data.get('count', 0) >= 2 else 'non_compliant'
            }
    
    return compliance_analysis

def _identify_corrective_actions(assessment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify required corrective actions"""
    actions = []
    bias_flags = assessment_data.get('bias_flags', [])
    
    for flag in bias_flags:
        if flag.get('type') == 'underrepresentation':
            actions.append({
                'action': f"Address {flag.get('category', 'unknown')} underrepresentation",
                'priority': 'high' if flag.get('severity') == 'high' else 'medium',
                'timeline': '30 days',
                'responsible_party': 'Recruitment Team'
            })
    
    return actions

def _calculate_compliance_score(assessment_data: Dict[str, Any]) -> int:
    """Calculate overall compliance score"""
    base_score = 100
    bias_flags = assessment_data.get('bias_flags', [])
    
    # Deduct points for each bias incident
    deduction = len(bias_flags) * 10
    
    return max(0, base_score - deduction)

def _perform_statistical_analysis(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform statistical analysis on assessment data"""
    # Placeholder for statistical analysis
    return {
        'statistical_methods': 'Simpson Diversity Index, Representation Analysis',
        'confidence_intervals': '95% confidence level',
        'statistical_significance': 'p < 0.05 threshold'
    }

def _analyze_trends(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze trends in assessment data"""
    # Placeholder for trend analysis
    return {
        'trend_period': 'Last 30 days',
        'trend_direction': 'stable',
        'trend_significance': 'moderate'
    }

def _document_methodology() -> Dict[str, Any]:
    """Document methodology used for bias assessment"""
    return {
        'assessment_framework': 'Race, Culture, and Religion Focus',
        'detection_methods': 'Pattern matching, Statistical analysis, Intersectional analysis',
        'thresholds': 'Configurable representation thresholds',
        'validation': 'Manual review and statistical validation'
    }

def _document_data_sources(assessment_data: Dict[str, Any]) -> List[str]:
    """Document data sources used in assessment"""
    return [
        'Resume text analysis',
        'Query sanitization logs',
        'Diversity metrics calculation',
        'Intersectional analysis results'
    ]

def _generate_action_plan(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate action plan based on assessment results"""
    return {
        'immediate_actions': _identify_corrective_actions(assessment_data),
        'short_term_goals': ['Improve diversity metrics', 'Reduce bias incidents'],
        'long_term_strategy': 'Implement comprehensive diversity and inclusion program',
        'success_metrics': ['Diversity score > 70', 'Bias incidents < 2 per month']
    }

def _generate_monitoring_schedule() -> Dict[str, Any]:
    """Generate monitoring schedule for ongoing bias prevention"""
    return {
        'daily_monitoring': 'Real-time bias detection',
        'weekly_review': 'Diversity metrics analysis',
        'monthly_reporting': 'Compliance and diversity reports',
        'quarterly_assessment': 'Comprehensive bias prevention review'
    }

def _generate_sensitivity_recommendations(issues: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations for improving cultural sensitivity"""
    recommendations = []
    
    if not issues:
        recommendations.append("Text shows good cultural sensitivity awareness")
        return recommendations
    
    for issue in issues:
        if issue.get('category') == 'racial_stereotypes':
            recommendations.append("Avoid using potentially stereotypical language when describing people")
        elif issue.get('category') == 'cultural_appropriation':
            recommendations.append("Be mindful of cultural appropriation in descriptions and themes")
        elif issue.get('category') == 'religious_insensitivity':
            recommendations.append("Use respectful language when discussing religious beliefs and practices")
    
    recommendations.append("Consider cultural sensitivity training for content creators")
    recommendations.append("Review content with diverse stakeholders before publication")
    
    return recommendations
