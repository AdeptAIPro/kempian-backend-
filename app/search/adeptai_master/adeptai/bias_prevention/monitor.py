import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json

class BiasMonitor:
    """Monitors search results for race, culture, and religion diversity and representation balance"""

    def __init__(self):
        self.metrics = {
            'queries_analyzed': 0,
            'bias_incidents_detected': 0,
            'diversity_scores': [],
            'representation_balance': defaultdict(list),
            'intersectional_analysis': []
        }
        self.diversity_thresholds = {
            'race': 0.15,      # Minimum 15% representation for any race
            'culture': 0.10,   # Minimum 10% representation for any culture
            'religion': 0.08   # Minimum 8% representation for any religion
        }
        self.intersectional_thresholds = {
            'race_culture': 0.05,    # Minimum 5% for race+culture combinations
            'race_religion': 0.04,   # Minimum 4% for race+religion combinations
            'culture_religion': 0.03  # Minimum 3% for culture+religion combinations
        }

    def assess_diversity(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess diversity and representation balance for race, culture, and religion"""
        self.metrics['queries_analyzed'] += 1
        
        if not results:
            return self._create_empty_assessment(query)
        
        assessment = {
            'query': query,
            'result_count': len(results),
            'timestamp': datetime.now().isoformat(),
            'diversity_metrics': self._calculate_diversity_metrics(results),
            'representation_balance': self._analyze_representation_balance(results),
            'intersectional_analysis': self._analyze_intersectional_diversity(results),
            'bias_flags': [],
            'diversity_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall diversity score
        assessment['diversity_score'] = self._calculate_overall_diversity_score(assessment)
        
        # Identify bias incidents
        assessment['bias_flags'] = self._identify_bias_incidents(assessment)
        
        # Generate recommendations
        assessment['recommendations'] = self._generate_diversity_recommendations(assessment)
        
        # Update global metrics
        self._update_global_metrics(assessment)
        
        return assessment

    def _calculate_diversity_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate diversity metrics for race, culture, and religion"""
        diversity_data = {
            'race': defaultdict(int),
            'culture': defaultdict(int),
            'religion': defaultdict(int),
            'total_identifiable': 0
        }
        
        for result in results:
            # Extract protected characteristics from result
            race_info = self._extract_race_indicators(result)
            culture_info = self._extract_culture_indicators(result)
            religion_info = self._extract_religion_indicators(result)
            
            if race_info:
                diversity_data['race'][race_info] += 1
            if culture_info:
                diversity_data['culture'][culture_info] += 1
            if religion_info:
                diversity_data['religion'][religion_info] += 1
            
            if race_info or culture_info or religion_info:
                diversity_data['total_identifiable'] += 1
        
        # Calculate percentages and diversity indices
        return self._compute_diversity_statistics(diversity_data, len(results))

    def _extract_race_indicators(self, result: Dict[str, Any]) -> str:
        """Extract race indicators from result data"""
        # Check various fields for race indicators
        fields_to_check = ['name', 'full_name', 'resume_text', 'description', 'summary']
        
        for field in fields_to_check:
            if field in result and isinstance(result[field], str):
                text = result[field].lower()
                
                # Check for specific race indicators
                if any(indicator in text for indicator in ['african', 'black', 'african-american']):
                    return 'African/Black'
                elif any(indicator in text for indicator in ['asian', 'chinese', 'japanese', 'korean', 'vietnamese']):
                    return 'Asian'
                elif any(indicator in text for indicator in ['hispanic', 'latino', 'mexican', 'puerto rican']):
                    return 'Hispanic/Latino'
                elif any(indicator in text for indicator in ['middle eastern', 'arab', 'persian', 'turkish']):
                    return 'Middle Eastern'
                elif any(indicator in text for indicator in ['native american', 'indigenous', 'aboriginal']):
                    return 'Native American/Indigenous'
                elif any(indicator in text for indicator in ['pacific islander', 'hawaiian', 'polynesian']):
                    return 'Pacific Islander'
                elif any(indicator in text for indicator in ['white', 'caucasian', 'european']):
                    return 'White/Caucasian'
        
        return None

    def _extract_culture_indicators(self, result: Dict[str, Any]) -> str:
        """Extract culture indicators from result data"""
        fields_to_check = ['resume_text', 'description', 'summary', 'education', 'experience']
        
        for field in fields_to_check:
            if field in result and isinstance(result[field], str):
                text = result[field].lower()
                
                # Check for cultural practices and traditions
                if any(indicator in text for indicator in ['diwali', 'holi', 'ramadan', 'eid']):
                    return 'Cultural Traditions'
                elif any(indicator in text for indicator in ['bollywood', 'k-pop', 'anime', 'manga']):
                    return 'Cultural Media'
                elif any(indicator in text for indicator in ['traditional', 'cultural', 'heritage']):
                    return 'Cultural Heritage'
                elif any(indicator in text for indicator in ['international', 'multicultural', 'diverse']):
                    return 'Multicultural'
        
        return None

    def _extract_religion_indicators(self, result: Dict[str, Any]) -> str:
        """Extract religion indicators from result data"""
        fields_to_check = ['resume_text', 'description', 'summary', 'education', 'experience']
        
        for field in fields_to_check:
            if field in result and isinstance(result[field], str):
                text = result[field].lower()
                
                # Check for religious indicators
                if any(indicator in text for indicator in ['christian', 'catholic', 'protestant', 'baptist']):
                    return 'Christian'
                elif any(indicator in text for indicator in ['muslim', 'islam', 'islamic', 'mosque']):
                    return 'Muslim/Islamic'
                elif any(indicator in text for indicator in ['jewish', 'judaism', 'synagogue', 'kosher']):
                    return 'Jewish'
                elif any(indicator in text for indicator in ['hindu', 'hinduism', 'temple', 'mandir']):
                    return 'Hindu'
                elif any(indicator in text for indicator in ['buddhist', 'buddhism', 'meditation', 'zen']):
                    return 'Buddhist'
                elif any(indicator in text for indicator in ['sikh', 'sikhism', 'gurdwara']):
                    return 'Sikh'
                elif any(indicator in text for indicator in ['atheist', 'agnostic', 'secular']):
                    return 'Non-religious'
        
        return None

    def _compute_diversity_statistics(self, diversity_data: Dict[str, Any], total_results: int) -> Dict[str, Any]:
        """Compute diversity statistics and indices"""
        stats = {}
        
        for category in ['race', 'culture', 'religion']:
            category_data = diversity_data[category]
            if not category_data:
                stats[category] = {'count': 0, 'percentage': 0.0, 'diversity_index': 0.0}
                continue
            
            total_category = sum(category_data.values())
            percentages = {k: (v / total_results) * 100 for k, v in category_data.items()}
            
            # Calculate Simpson's Diversity Index
            diversity_index = 1 - sum((v / total_category) ** 2 for v in category_data.values())
            
            stats[category] = {
                'count': len(category_data),
                'total_identified': total_category,
                'percentages': percentages,
                'diversity_index': diversity_index,
                'representation': category_data
            }
        
        stats['total_identifiable'] = diversity_data['total_identifiable']
        stats['identifiable_percentage'] = (diversity_data['total_identifiable'] / total_results) * 100
        
        return stats

    def _analyze_representation_balance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze representation balance across protected characteristics"""
        balance_analysis = {
            'race_balance': self._check_representation_balance('race', results),
            'culture_balance': self._check_representation_balance('culture', results),
            'religion_balance': self._check_representation_balance('religion', results),
            'overall_balance_score': 0.0
        }
        
        # Calculate overall balance score
        balance_scores = [
            balance_analysis['race_balance']['score'],
            balance_analysis['culture_balance']['score'],
            balance_analysis['religion_balance']['score']
        ]
        balance_analysis['overall_balance_score'] = np.mean([s for s in balance_scores if s > 0])
        
        return balance_analysis

    def _check_representation_balance(self, category: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check representation balance for a specific category"""
        # This would be implemented based on the diversity metrics
        # For now, returning a placeholder structure
        return {
            'score': 0.0,
            'status': 'insufficient_data',
            'details': f'Insufficient {category} data for balance analysis'
        }

    def _analyze_intersectional_diversity(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze intersectional diversity (combinations of protected characteristics)"""
        intersectional_data = {
            'race_culture': defaultdict(int),
            'race_religion': defaultdict(int),
            'culture_religion': defaultdict(int),
            'triple_intersection': defaultdict(int)
        }
        
        for result in results:
            race = self._extract_race_indicators(result)
            culture = self._extract_culture_indicators(result)
            religion = self._extract_religion_indicators(result)
            
            # Record intersectional combinations
            if race and culture:
                intersectional_data['race_culture'][f"{race}+{culture}"] += 1
            if race and religion:
                intersectional_data['race_religion'][f"{race}+{religion}"] += 1
            if culture and religion:
                intersectional_data['culture_religion'][f"{culture}+{religion}"] += 1
            if race and culture and religion:
                intersectional_data['triple_intersection'][f"{race}+{culture}+{religion}"] += 1
        
        return intersectional_data

    def _calculate_overall_diversity_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate overall diversity score (0-100)"""
        diversity_metrics = assessment['diversity_metrics']
        
        if not diversity_metrics:
            return 0.0
        
        scores = []
        
        # Race diversity score
        if 'race' in diversity_metrics and diversity_metrics['race']['count'] > 0:
            race_score = diversity_metrics['race']['diversity_index'] * 100
            scores.append(race_score)
        
        # Culture diversity score
        if 'culture' in diversity_metrics and diversity_metrics['culture']['count'] > 0:
            culture_score = diversity_metrics['culture']['diversity_index'] * 100
            scores.append(culture_score)
        
        # Religion diversity score
        if 'religion' in diversity_metrics and diversity_metrics['religion']['count'] > 0:
            religion_score = diversity_metrics['religion']['diversity_index'] * 100
            scores.append(religion_score)
        
        # Representation balance score
        if assessment['representation_balance']['overall_balance_score'] > 0:
            balance_score = assessment['representation_balance']['overall_balance_score']
            scores.append(balance_score)
        
        return np.mean(scores) if scores else 0.0

    def _identify_bias_incidents(self, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential bias incidents based on diversity metrics"""
        bias_flags = []
        diversity_metrics = assessment['diversity_metrics']
        
        # Check for underrepresentation
        for category, threshold in self.diversity_thresholds.items():
            if category in diversity_metrics:
                category_data = diversity_metrics[category]
                if category_data['count'] > 0:
                    for representation, percentage in category_data['percentages'].items():
                        if percentage < (threshold * 100):
                            bias_flags.append({
                                'type': 'underrepresentation',
                                'category': category,
                                'group': representation,
                                'percentage': percentage,
                                'threshold': threshold * 100,
                                'severity': 'high' if percentage < (threshold * 50) else 'medium'
                            })
        
        # Check for intersectional bias
        intersectional_data = assessment['intersectional_analysis']
        for combination_type, combinations in intersectional_data.items():
            if combinations:
                total_results = assessment['result_count']
                for combination, count in combinations.items():
                    percentage = (count / total_results) * 100
                    threshold = self.intersectional_thresholds.get(combination_type, 0.03)
                    if percentage < (threshold * 100):
                        bias_flags.append({
                            'type': 'intersectional_underrepresentation',
                            'combination_type': combination_type,
                            'combination': combination,
                            'percentage': percentage,
                            'threshold': threshold * 100,
                            'severity': 'high' if percentage < (threshold * 50) else 'medium'
                        })
        
        return bias_flags

    def _generate_diversity_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving diversity"""
        recommendations = []
        bias_flags = assessment['bias_flags']
        
        if not bias_flags:
            recommendations.append("Diversity metrics look good. Continue current practices.")
            return recommendations
        
        # Generate specific recommendations based on bias flags
        for flag in bias_flags:
            if flag['type'] == 'underrepresentation':
                recommendations.append(
                    f"Consider expanding search criteria to include more {flag['category']} candidates, "
                    f"particularly {flag['group']} representation (currently {flag['percentage']:.1f}%)"
                )
            elif flag['type'] == 'intersectional_underrepresentation':
                recommendations.append(
                    f"Address intersectional underrepresentation: {flag['combination']} "
                    f"(currently {flag['percentage']:.1f}% representation)"
                )
        
        # General recommendations
        if assessment['diversity_score'] < 50:
            recommendations.append("Overall diversity score is low. Review recruitment strategies and search algorithms.")
        
        if assessment['diversity_metrics'].get('total_identifiable', 0) < len(assessment['results']) * 0.3:
            recommendations.append("Low identification rate of protected characteristics. Consider improving data collection methods.")
        
        return recommendations

    def _update_global_metrics(self, assessment: Dict[str, Any]):
        """Update global monitoring metrics"""
        self.metrics['diversity_scores'].append(assessment['diversity_score'])
        
        if assessment['bias_flags']:
            self.metrics['bias_incidents_detected'] += len(assessment['bias_flags'])
        
        # Store representation balance data
        for category in ['race', 'culture', 'religion']:
            if category in assessment['diversity_metrics']:
                self.metrics['representation_balance'][category].append(
                    assessment['diversity_metrics'][category]['diversity_index']
                )
        
        # Store intersectional analysis
        if assessment['intersectional_analysis']:
            self.metrics['intersectional_analysis'].append(assessment['intersectional_analysis'])

    def _create_empty_assessment(self, query: str) -> Dict[str, Any]:
        """Create assessment for empty results"""
        return {
            'query': query,
            'result_count': 0,
            'timestamp': datetime.now().isoformat(),
            'diversity_metrics': {},
            'representation_balance': {},
            'intersectional_analysis': {},
            'bias_flags': [],
            'diversity_score': 0.0,
            'recommendations': ['No results found. Consider broadening search criteria.']
        }

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of all monitoring activities"""
        return {
            'total_queries_analyzed': self.metrics['queries_analyzed'],
            'total_bias_incidents': self.metrics['bias_incidents_detected'],
            'average_diversity_score': np.mean(self.metrics['diversity_scores']) if self.metrics['diversity_scores'] else 0.0,
            'representation_trends': dict(self.metrics['representation_balance']),
            'monitoring_period': {
                'start': min(self.metrics['diversity_scores']) if self.metrics['diversity_scores'] else None,
                'current': max(self.metrics['diversity_scores']) if self.metrics['diversity_scores'] else None
            }
        }

    def export_monitoring_data(self, filepath: str = None) -> str:
        """Export monitoring data to JSON file"""
        export_data = {
            'monitoring_summary': self.get_monitoring_summary(),
            'detailed_metrics': dict(self.metrics),
            'export_timestamp': datetime.now().isoformat()
        }
        
        if filepath is None:
            filepath = f"bias_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath
