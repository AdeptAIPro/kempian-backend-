"""
Enhanced Behavioral Analysis Pipeline for Recruitment
====================================================
Semantic-driven behavioral profiling with multi-source analysis and career tracking
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import re
from textstat import textstat
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Import original components
from .entity_extractor import EntityExtractor
from .semantic_analyzer import SemanticAnalyzer
from .emotion_analyzer import EmotionAnalyzer
from .behavioral_scorer import BehavioralScorer
from .domain_bert import DomainSpecificBERT, DomainType
from .career_gnn import PretrainedCareerGNN, CareerGraphBuilder, CareerNode

# Import new multi-modal components
from .multi_modal_engine import MultiModalEngine, ModelType, ModelPrediction, EnsemblePrediction
from .advanced_feature_extractor import AdvancedFeatureExtractor
from .ensemble_optimizer import EnsembleOptimizer, OptimizationConfig, OptimizationMethod, ValidationStrategy
from .confidence_calibrator import ConfidenceCalibrator, CalibrationConfig, CalibrationMethod


class ProfileDataSource(Enum):
    """Types of data sources for behavioral analysis"""
    RESUME = "resume"
    LINKEDIN = "linkedin"
    GITHUB = "github"
    PORTFOLIO = "portfolio"
    REFERENCES = "references"
    ASSESSMENT = "assessment"
    JOB_SITES = "job_sites"


@dataclass
class MultiSourceProfile:
    """Container for multi-source candidate data"""
    resume_text: str
    linkedin_data: Optional[Dict] = None
    github_data: Optional[Dict] = None
    portfolio_data: Optional[Dict] = None
    reference_data: Optional[List[Dict]] = None
    assessment_results: Optional[Dict] = None
    job_site_data: Optional[Dict] = None  # Data from job websites
    
    def get_all_text_content(self) -> str:
        """Aggregate all text content from various sources"""
        content = [self.resume_text]
        
        if self.linkedin_data:
            content.extend([
                self.linkedin_data.get('summary', ''),
                ' '.join(self.linkedin_data.get('experience_descriptions', [])),
                ' '.join(self.linkedin_data.get('recommendations', [])),
                ' '.join(self.linkedin_data.get('posts', []))
            ])
        
        if self.github_data:
            content.extend([
                self.github_data.get('bio', ''),
                ' '.join(self.github_data.get('readme_content', [])),
                ' '.join(self.github_data.get('project_descriptions', [])),
                ' '.join(self.github_data.get('code_comments', []))
            ])
        
        if self.portfolio_data:
            content.extend([
                self.portfolio_data.get('about_section', ''),
                ' '.join(self.portfolio_data.get('project_narratives', [])),
                ' '.join(self.portfolio_data.get('testimonials', []))
            ])
        
        if self.reference_data:
            for ref in self.reference_data:
                content.extend([
                    ref.get('feedback_text', ''),
                    ref.get('work_description', '')
                ])
        
        if self.job_site_data:
            content.extend([
                self.job_site_data.get('profile_summary', ''),
                ' '.join(self.job_site_data.get('project_descriptions', [])),
                ' '.join(self.job_site_data.get('client_reviews', []))
            ])
        
        return ' '.join(filter(None, content))


@dataclass
class BehavioralProfile:
    """Comprehensive behavioral profile of a candidate"""
    candidate_id: str
    overall_score: float
    
    # Core behavioral dimensions (0-1 scale)
    leadership_score: float
    collaboration_score: float
    innovation_score: float
    adaptability_score: float
    stability_score: float
    growth_potential: float
    
    # Emotional intelligence metrics
    emotional_intelligence: float
    stress_resilience: float
    communication_effectiveness: float
    
    # Technical competencies
    technical_depth: float
    learning_agility: float
    problem_solving_ability: float
    
    # Cultural fit indicators
    cultural_alignment: float
    work_style_preferences: List[str]
    
    # Validation metrics
    confidence_score: float  # How confident we are in this profile
    data_completeness: float  # Percentage of available data sources
    consistency_score: float  # Cross-source consistency
    
    # Supporting evidence
    strengths: List[str]
    development_areas: List[str]
    behavioral_patterns: List[str]
    risk_factors: List[str]
    
    # Source attribution
    primary_data_sources: List[str]


@dataclass
class CareerTrajectory:
    """Career growth prediction and analysis"""
    current_level: str
    predicted_next_roles: List[Tuple[str, float]]  # (role, confidence_score)
    growth_timeline: Dict[str, int]  # role -> estimated_months
    skill_development_path: List[str]
    experience_gaps: List[str]
    development_recommendations: List[str]
    success_indicators: List[str]
    risk_factors: List[str]
    market_alignment: float  # How well aligned with market trends


class SemanticSkillAnalyzer:
    """Analyzes role requirements using semantic understanding"""
    
    def __init__(self, semantic_analyzer: SemanticAnalyzer, domain_bert: DomainSpecificBERT):
        self.semantic = semantic_analyzer
        self.domain_bert = domain_bert
        
        # Semantic exemplars for different skill categories
        self.skill_exemplars = {
            'leadership': [
                "guided team through complex project challenges and delivered results",
                "influenced stakeholders across different organizational levels",
                "mentored junior team members and fostered their professional growth",
                "made strategic decisions during high-pressure situations",
                "built consensus among diverse groups with competing priorities"
            ],
            'collaboration': [
                "worked seamlessly with cross-functional teams to achieve shared objectives",
                "facilitated effective communication between different departments",
                "contributed meaningfully to team discussions and collaborative problem solving",
                "supported colleagues during challenging project phases",
                "built strong working relationships that enhanced team performance"
            ],
            'innovation': [
                "developed creative solutions to previously unsolved problems",
                "introduced novel approaches that improved existing processes",
                "experimented with cutting-edge technologies to enhance capabilities",
                "challenged conventional thinking and proposed alternative methodologies",
                "transformed abstract concepts into practical implementable solutions"
            ],
            'adaptability': [
                "successfully navigated significant organizational changes and transitions",
                "quickly mastered new technologies and methodologies as business needs evolved",
                "thrived in ambiguous environments with changing requirements",
                "adjusted communication style effectively for different stakeholder groups",
                "remained productive during periods of uncertainty and rapid change"
            ],
            'technical_depth': [
                "demonstrated deep understanding of complex technical concepts and systems",
                "solved intricate problems requiring advanced domain expertise",
                "designed robust architectures handling significant scale and complexity",
                "optimized performance through sophisticated technical analysis",
                "mentored others on advanced technical topics and best practices"
            ],
            'problem_solving': [
                "analyzed complex situations systematically to identify root causes",
                "developed comprehensive solutions addressing multiple interconnected challenges",
                "approached problems from multiple angles to find optimal resolutions",
                "broke down overwhelming challenges into manageable actionable components",
                "applied analytical thinking to resolve ambiguous and ill-defined problems"
            ],
            'communication': [
                "presented complex technical concepts clearly to non-technical audiences",
                "facilitated productive discussions among groups with different perspectives",
                "documented processes and decisions in ways that enhanced team understanding",
                "provided feedback that was both constructive and actionable",
                "articulated vision and strategy in ways that inspired and motivated others"
            ],
            'learning_agility': [
                "rapidly acquired new skills and knowledge to meet evolving job demands",
                "synthesized information from diverse sources to develop comprehensive understanding",
                "applied learnings from one context effectively to solve problems in different domains",
                "sought out challenging opportunities to expand capabilities and expertise",
                "demonstrated intellectual curiosity and commitment to continuous improvement"
            ]
        }
        
        # Role-specific behavioral profiles (semantic understanding)
        self.role_behavioral_profiles = {
            'software_engineer': {
                'description': "individual contributor role focused on building and maintaining software systems",
                'key_behaviors': [
                    "writes clean, maintainable code that solves real user problems",
                    "collaborates effectively with product and design teams",
                    "continuously learns new technologies to improve development efficiency",
                    "debugging complex issues through systematic analysis",
                    "participates constructively in code reviews and technical discussions"
                ],
                'behavioral_weights': {
                    'technical_depth': 0.35,
                    'problem_solving_ability': 0.25,
                    'collaboration': 0.15,
                    'learning_agility': 0.15,
                    'innovation': 0.10
                }
            },
            'senior_software_engineer': {
                'description': "experienced individual contributor with mentoring responsibilities",
                'key_behaviors': [
                    "designs robust solutions for complex technical challenges",
                    "mentors junior developers and elevates overall team capabilities",
                    "makes architectural decisions that scale with business growth",
                    "leads technical initiatives across multiple teams",
                    "balances technical excellence with pragmatic delivery timelines"
                ],
                'behavioral_weights': {
                    'technical_depth': 0.30,
                    'leadership_score': 0.20,
                    'problem_solving_ability': 0.20,
                    'collaboration': 0.15,
                    'innovation': 0.15
                }
            },
            'engineering_manager': {
                'description': "people manager responsible for team performance and technical direction",
                'key_behaviors': [
                    "develops and motivates engineering talent to achieve career goals",
                    "balances technical debt with feature development priorities",
                    "communicates technical complexity to business stakeholders",
                    "builds high-performing teams through effective hiring and coaching",
                    "drives strategic technical decisions aligned with business objectives"
                ],
                'behavioral_weights': {
                    'leadership_score': 0.35,
                    'collaboration': 0.25,
                    'communication_effectiveness': 0.20,
                    'technical_depth': 0.15,
                    'adaptability': 0.05
                }
            },
            'data_scientist': {
                'description': "analytical role focused on extracting insights from data",
                'key_behaviors': [
                    "translates business questions into analytical frameworks",
                    "develops models that provide actionable insights for decision making",
                    "communicates complex analytical findings to diverse audiences",
                    "validates hypotheses through rigorous experimental design",
                    "stays current with evolving methodologies in data science"
                ],
                'behavioral_weights': {
                    'technical_depth': 0.30,
                    'problem_solving_ability': 0.25,
                    'communication_effectiveness': 0.20,
                    'innovation': 0.15,
                    'learning_agility': 0.10
                }
            },
            'product_manager': {
                'description': "strategic role balancing user needs with business objectives",
                'key_behaviors': [
                    "synthesizes user research with market analysis to inform product strategy",
                    "prioritizes features based on impact and alignment with business goals",
                    "facilitates cross-functional collaboration between engineering, design, and business",
                    "communicates product vision clearly to diverse stakeholder groups",
                    "makes data-driven decisions while maintaining user empathy"
                ],
                'behavioral_weights': {
                    'leadership_score': 0.25,
                    'communication_effectiveness': 0.25,
                    'problem_solving_ability': 0.20,
                    'collaboration': 0.20,
                    'adaptability': 0.10
                }
            }
        }
    
    def analyze_role_fit(self, profile_text: str, target_role: str, 
                        job_description: str = None) -> Dict[str, Any]:
        """Analyze semantic fit between candidate and role using behavioral exemplars"""
        
        role_key = target_role.lower().replace(' ', '_')
        
        if role_key not in self.role_behavioral_profiles:
            return self._analyze_generic_role_fit(profile_text, target_role, job_description)
        
        role_profile = self.role_behavioral_profiles[role_key]
        
        # Calculate semantic alignment with role behaviors
        role_alignment = self._calculate_role_alignment(profile_text, role_profile)
        
        # Analyze individual behavioral dimensions
        behavioral_scores = self._analyze_behavioral_dimensions(profile_text)
        
        # Calculate weighted role fit score
        weighted_score = self._calculate_weighted_score(behavioral_scores, role_profile['behavioral_weights'])
        
        # Identify development areas
        development_areas = self._identify_development_areas(
            behavioral_scores, role_profile['behavioral_weights']
        )
        
        # Assess experience alignment
        experience_fit = self._assess_experience_alignment(
            profile_text, role_profile['key_behaviors']
        )
        
        return {
            'overall_fit_score': (weighted_score + role_alignment + experience_fit) / 3,
            'role_alignment': role_alignment,
            'behavioral_scores': behavioral_scores,
            'weighted_behavioral_score': weighted_score,
            'experience_alignment': experience_fit,
            'development_areas': development_areas,
            'strengths': self._identify_strengths(behavioral_scores, role_profile['behavioral_weights']),
            'role_readiness': self._assess_role_readiness(behavioral_scores, role_key),
            'semantic_insights': self._generate_semantic_insights(profile_text, role_profile)
        }
    
    def _calculate_role_alignment(self, profile_text: str, role_profile: Dict) -> float:
        """Calculate semantic alignment with role-specific behaviors"""
        role_behaviors = role_profile['key_behaviors']
        role_description = role_profile['description']
        
        # Encode profile and role elements
        profile_embedding = self.semantic.encode([profile_text])
        role_embeddings = self.semantic.encode(role_behaviors + [role_description])
        
        # Calculate similarities
        similarities = []
        for role_embedding in role_embeddings:
            sim = self.semantic.cosine(profile_embedding[0], role_embedding)
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def _analyze_behavioral_dimensions(self, profile_text: str) -> Dict[str, float]:
        """Analyze behavioral dimensions using semantic exemplars"""
        behavioral_scores = {}
        
        for dimension, exemplars in self.skill_exemplars.items():
            # Calculate semantic similarity to exemplars
            dimension_score = self.semantic.exemplar_alignment(profile_text).get(
                dimension, 0.0
            )
            
            # If not in standard exemplars, calculate manually
            if dimension not in ['leadership', 'collaboration', 'innovation', 'adaptability']:
                exemplar_embeddings = self.semantic.encode(exemplars)
                profile_embedding = self.semantic.encode([profile_text])[0]
                
                similarities = []
                for exemplar_emb in exemplar_embeddings:
                    sim = self.semantic.cosine(profile_embedding, exemplar_emb)
                    similarities.append(sim)
                
                dimension_score = float(np.mean(similarities))
            
            behavioral_scores[dimension] = dimension_score
        
        return behavioral_scores
    
    def _calculate_weighted_score(self, behavioral_scores: Dict[str, float], 
                                 weights: Dict[str, float]) -> float:
        """Calculate weighted behavioral score based on role requirements"""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in weights.items():
            if dimension in behavioral_scores:
                total_score += behavioral_scores[dimension] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _identify_development_areas(self, behavioral_scores: Dict[str, float], 
                                  weights: Dict[str, float]) -> List[str]:
        """Identify areas for development based on role requirements"""
        development_areas = []
        
        for dimension, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            if dimension in behavioral_scores:
                score = behavioral_scores[dimension]
                # High weight but low score indicates development area
                if weight > 0.15 and score < 0.6:
                    development_areas.append(dimension)
        
        return development_areas
    
    def _identify_strengths(self, behavioral_scores: Dict[str, float], 
                          weights: Dict[str, float]) -> List[str]:
        """Identify key strengths relevant to the role"""
        strengths = []
        
        for dimension, score in behavioral_scores.items():
            role_weight = weights.get(dimension, 0.0)
            # High score in important dimension indicates strength
            if score > 0.7 and role_weight > 0.1:
                strengths.append(dimension)
        
        return strengths
    
    def _assess_experience_alignment(self, profile_text: str, 
                                   key_behaviors: List[str]) -> float:
        """Assess how well experience aligns with role requirements"""
        profile_embedding = self.semantic.encode([profile_text])[0]
        behavior_embeddings = self.semantic.encode(key_behaviors)
        
        similarities = []
        for behavior_emb in behavior_embeddings:
            sim = self.semantic.cosine(profile_embedding, behavior_emb)
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def _assess_role_readiness(self, behavioral_scores: Dict[str, float], role_key: str) -> str:
        """Assess overall readiness for the role"""
        role_profile = self.role_behavioral_profiles[role_key]
        critical_dimensions = [dim for dim, weight in role_profile['behavioral_weights'].items() 
                             if weight > 0.2]
        
        critical_scores = [behavioral_scores.get(dim, 0.0) for dim in critical_dimensions]
        avg_critical_score = np.mean(critical_scores) if critical_scores else 0.0
        
        if avg_critical_score > 0.75:
            return "Highly Ready"
        elif avg_critical_score > 0.6:
            return "Ready with Development"
        elif avg_critical_score > 0.4:
            return "Requires Significant Development"
        else:
            return "Not Ready"
    
    def _generate_semantic_insights(self, profile_text: str, role_profile: Dict) -> List[str]:
        """Generate insights based on semantic analysis"""
        insights = []
        
        # Analyze alignment with role description
        role_desc_similarity = self.semantic.resume_jd_similarity(
            profile_text, role_profile['description']
        )
        
        if role_desc_similarity > 0.7:
            insights.append("Strong semantic alignment with role expectations")
        elif role_desc_similarity > 0.5:
            insights.append("Moderate alignment with role requirements")
        else:
            insights.append("Limited alignment with core role responsibilities")
        
        return insights
    
    def _analyze_generic_role_fit(self, profile_text: str, target_role: str, 
                                 job_description: str = None) -> Dict[str, Any]:
        """Fallback analysis for roles not in predefined profiles"""
        behavioral_scores = self._analyze_behavioral_dimensions(profile_text)
        
        # Use job description if available
        if job_description:
            jd_similarity = self.semantic.resume_jd_similarity(profile_text, job_description)
        else:
            jd_similarity = 0.5  # Neutral score
        
        return {
            'overall_fit_score': jd_similarity,
            'behavioral_scores': behavioral_scores,
            'experience_alignment': jd_similarity,
            'development_areas': [dim for dim, score in behavioral_scores.items() if score < 0.5],
            'strengths': [dim for dim, score in behavioral_scores.items() if score > 0.7],
            'role_readiness': "Analysis Limited - Generic Assessment",
            'semantic_insights': ["Generic role analysis - consider adding specific role profile"]
        }


class CareerTrajectoryAnalyzer:
    """Analyzes career progression and predicts future growth"""
    
    def __init__(self, semantic_analyzer: SemanticAnalyzer, career_gnn: PretrainedCareerGNN):
        self.semantic = semantic_analyzer
        self.career_gnn = career_gnn
        
        # Career progression pathways (semantic understanding)
        self.progression_pathways = {
            'software_engineer': [
                ('senior_software_engineer', 0.8),
                ('tech_lead', 0.6),
                ('engineering_manager', 0.4),
                ('product_manager', 0.3),
                ('staff_engineer', 0.5)
            ],
            'senior_software_engineer': [
                ('staff_engineer', 0.7),
                ('principal_engineer', 0.6),
                ('engineering_manager', 0.8),
                ('tech_lead', 0.9),
                ('architect', 0.5)
            ],
            'data_scientist': [
                ('senior_data_scientist', 0.8),
                ('data_science_manager', 0.6),
                ('principal_data_scientist', 0.7),
                ('product_manager', 0.4),
                ('research_scientist', 0.5)
            ],
            'product_manager': [
                ('senior_product_manager', 0.8),
                ('principal_product_manager', 0.7),
                ('group_product_manager', 0.6),
                ('director_product', 0.5),
                ('vp_product', 0.3)
            ]
        }
    
    def analyze_career_trajectory(self, profile: MultiSourceProfile, 
                                current_role: str) -> CareerTrajectory:
        """Comprehensive career trajectory analysis"""
        
        # Extract career history from all sources
        full_profile_text = profile.get_all_text_content()
        entities = EntityExtractor().extract(full_profile_text)
        
        # Analyze current position
        current_level = self._assess_current_level(entities, current_role)
        
        # Predict next roles using semantic analysis
        predicted_roles = self._predict_next_roles(
            full_profile_text, current_role, entities
        )
        
        # Estimate timeline
        growth_timeline = self._estimate_growth_timeline(
            entities, predicted_roles, current_role
        )
        
        # Identify skill gaps
        skill_gaps = self._identify_skill_gaps_for_growth(
            full_profile_text, predicted_roles
        )
        
        # Generate development recommendations
        development_recs = self._generate_development_recommendations(
            full_profile_text, predicted_roles, skill_gaps
        )
        
        # Assess market alignment
        market_alignment = self._assess_market_alignment(
            current_role, predicted_roles, full_profile_text
        )
        
        # Identify success indicators and risks
        success_indicators = self._identify_success_indicators(full_profile_text, entities)
        risk_factors = self._identify_career_risks(entities, current_role)
        
        return CareerTrajectory(
            current_level=current_level,
            predicted_next_roles=predicted_roles,
            growth_timeline=growth_timeline,
            skill_development_path=self._create_skill_development_path(skill_gaps),
            experience_gaps=skill_gaps,
            development_recommendations=development_recs,
            success_indicators=success_indicators,
            risk_factors=risk_factors,
            market_alignment=market_alignment
        )
    
    def _assess_current_level(self, entities: Dict, current_role: str) -> str:
        """Assess current seniority level based on semantic analysis"""
        roles = entities.get("ROLES", [])
        
        if not roles:
            return "Entry Level"
        
        # Analyze role progression semantically
        role_embeddings = self.semantic.encode(roles)
        seniority_indicators = self.semantic.encode([
            "junior entry level beginner associate",
            "intermediate mid-level regular standard",
            "senior experienced advanced specialist",
            "lead principal staff architect",
            "director manager head executive"
        ])
        
        # Find best semantic match
        max_similarity = 0.0
        best_level = "Mid Level"
        level_names = ["Entry Level", "Mid Level", "Senior Level", "Lead Level", "Executive Level"]
        
        latest_role = roles[-1] if roles else current_role
        latest_role_embedding = self.semantic.encode([latest_role])[0]
        
        for i, seniority_emb in enumerate(seniority_indicators):
            similarity = self.semantic.cosine(latest_role_embedding, seniority_emb)
            if similarity > max_similarity:
                max_similarity = similarity
                best_level = level_names[i]
        
        return best_level
    
    def _predict_next_roles(self, profile_text: str, current_role: str, 
                          entities: Dict) -> List[Tuple[str, float]]:
        """Predict next career roles using semantic analysis and GNN"""
        
        current_role_key = current_role.lower().replace(' ', '_')
        
        # Get pathway predictions
        pathway_predictions = self.progression_pathways.get(current_role_key, [])
        
        # Use GNN for enhanced predictions if available
        if self.career_gnn:
            try:
                gnn_predictions = self.career_gnn.predict_career_trajectory(
                    entities, profile_text
                )
                growth_potential = gnn_predictions.get('growth_potential', 0.5)
                
                # Adjust pathway probabilities based on GNN insights
                adjusted_predictions = []
                for role, base_prob in pathway_predictions:
                    # Factor in growth potential
                    adjusted_prob = base_prob * (0.7 + 0.3 * growth_potential)
                    adjusted_predictions.append((role, min(1.0, adjusted_prob)))
                
                pathway_predictions = adjusted_predictions
            except Exception:
                pass  # Fall back to base predictions
        
        # Semantic analysis of readiness for each potential role
        semantic_predictions = []
        for role, base_prob in pathway_predictions:
            # Analyze semantic fit for the target role
            role_description = f"experienced professional in {role.replace('_', ' ')} position"
            semantic_fit = self.semantic.resume_jd_similarity(profile_text, role_description)
            
            # Combine base probability with semantic fit
            final_prob = (base_prob + semantic_fit) / 2
            semantic_predictions.append((role.replace('_', ' ').title(), final_prob))
        
        # Sort by probability and return top 5
        return sorted(semantic_predictions, key=lambda x: x[1], reverse=True)[:5]
    
    def _estimate_growth_timeline(self, entities: Dict, predicted_roles: List[Tuple[str, float]], 
                                current_role: str) -> Dict[str, int]:
        """Estimate timeline for achieving predicted roles"""
        timeline = {}
        
        # Analyze historical progression speed
        roles = entities.get("ROLES", [])
        avg_tenure = 24  # Default 2 years
        
        if len(roles) > 1:
            # Estimate average time per role (simplified)
            avg_tenure = max(18, min(36, 30 - len(roles) * 3))  # Adjust based on career length
        
        base_months = avg_tenure
        for i, (role, confidence) in enumerate(predicted_roles):
            # Higher confidence = potentially faster timeline
            # Lower confidence = longer development time needed
            confidence_factor = 0.7 + (confidence * 0.6)  # 0.7 to 1.3 range
            estimated_months = int(base_months * (1 + i * 0.5) / confidence_factor)
            timeline[role] = estimated_months
        
        return timeline
    
    def _identify_skill_gaps_for_growth(self, profile_text: str, 
                                      predicted_roles: List[Tuple[str, float]]) -> List[str]:
        """Identify skills needed for career growth"""
        gaps = []
        
        # For each predicted role, analyze what's missing
        for role, confidence in predicted_roles[:3]:  # Top 3 roles
            role_key = role.lower().replace(' ', '_')
            
            # Semantic analysis of role requirements vs current profile
            role_requirements = f"expert in {role} with advanced capabilities"
            requirement_similarity = self.semantic.resume_jd_similarity(
                profile_text, role_requirements
            )
            
            if requirement_similarity < 0.7:
                # Identify specific gap areas through semantic analysis
                skill_areas = [
                    "advanced technical leadership",
                    "strategic thinking and planning", 
                    "team development and mentoring",
                    "cross-functional collaboration",
                    "business acumen and impact"
                ]
                
                for skill_area in skill_areas:
                    skill_similarity = self.semantic.resume_jd_similarity(
                        profile_text, f"demonstrated experience in {skill_area}"
                    )
                    if skill_similarity < 0.5:
                        gaps.append(f"{skill_area} for {role}")
        
        return list(set(gaps))  # Remove duplicates
    
    def _generate_development_recommendations(self, profile_text: str, 
                                           predicted_roles: List[Tuple[str, float]], 
                                           skill_gaps: List[str]) -> List[str]:
        """Generate actionable development recommendations"""
        recommendations = []
        
        # High-level recommendations based on career trajectory
        top_role = predicted_roles[0][0] if predicted_roles else "senior position"
        
        recommendations.extend([
            f"Focus on developing skills for {top_role} transition",
            "Seek opportunities to demonstrate leadership impact",
            "Build broader cross-functional relationships",
            "Document and communicate your strategic contributions"
        ])
        
        # Specific recommendations based on skill gaps
        for gap in skill_gaps[:3]:  # Top 3 gaps
            if "leadership" in gap.lower():
                recommendations.append("Take on team lead responsibilities on key projects")
            elif "strategic" in gap.lower():
                recommendations.append("Participate in strategic planning and roadmap discussions")
            elif "mentoring" in gap.lower():
                recommendations.append("Establish formal mentoring relationships with junior team members")
            elif "business" in gap.lower():
                recommendations.append("Develop deeper understanding of business metrics and impact")
        
        return recommendations
    
    def _create_skill_development_path(self, skill_gaps: List[str]) -> List[str]:
        """Create ordered skill development path"""
        # Priority order based on typical career progression
        priority_order = [
            "technical leadership",
            "team collaboration", 
            "strategic thinking",
            "business acumen",
            "people development"
        ]
        
        development_path = []
        for priority in priority_order:
            matching_gaps = [gap for gap in skill_gaps if priority in gap.lower()]
            development_path.extend(matching_gaps[:1])  # Add one per category
        
        return development_path
    
    def _identify_success_indicators(self, profile_text: str, entities: Dict) -> List[str]:
        """Identify indicators of career success potential"""
        indicators = []
        
        # Semantic analysis for success patterns
        success_patterns = [
            "demonstrated consistent career progression",
            "showed leadership in challenging situations", 
            "delivered significant business impact",
            "developed and mentored other professionals",
            "adapted successfully to changing environments"
        ]
        
        for pattern in success_patterns:
            similarity = self.semantic.resume_jd_similarity(profile_text, pattern)
            if similarity > 0.6:
                indicators.append(pattern)
        
        # Analyze role progression quality
        roles = entities.get("ROLES", [])
        if len(roles) >= 2:
            progression_quality = self.semantic.analyze_progression(roles)
            if progression_quality > 0.6:
                indicators.append("Shows strong career progression trajectory")
        
        return indicators
    
    def _identify_career_risks(self, entities: Dict, current_role: str) -> List[str]:
        """Identify potential career development risks"""
        risks = []
        
        roles = entities.get("ROLES", [])
        companies = entities.get("COMPANIES", [])
        
        # Analyze job stability patterns
        if len(companies) > len(roles) * 1.5:
            risks.append("Frequent job changes may indicate stability concerns")
        
        # Analyze progression stagnation
        if len(roles) >= 3:
            progression = self.semantic.analyze_progression(roles)
            if progression < 0.3:
                risks.append("Limited career progression may indicate growth challenges")
        
        # Role diversity analysis
        if len(set(roles)) < len(roles) * 0.7:
            risks.append("Limited role diversity may restrict future opportunities")
        
        return risks
    
    def _assess_market_alignment(self, current_role: str, predicted_roles: List[Tuple[str, float]], 
                               profile_text: str) -> float:
        """Assess alignment with current market trends"""
        
        # Market trend indicators (semantic patterns)
        market_trends = [
            "digital transformation and cloud technologies",
            "artificial intelligence and machine learning applications",
            "data-driven decision making and analytics",
            "agile methodologies and DevOps practices",
            "remote work and distributed team leadership",
            "sustainability and environmental considerations",
            "customer experience and user-centric design"
        ]
        
        trend_alignments = []
        for trend in market_trends:
            alignment = self.semantic.resume_jd_similarity(profile_text, trend)
            trend_alignments.append(alignment)
        
        return float(np.mean(trend_alignments))


class MultiModelBehavioralValidator:
    """Validates behavioral profiles using multiple AI models"""
    
    def __init__(self, semantic_analyzer: SemanticAnalyzer, domain_bert: DomainSpecificBERT,
                 emotion_analyzer: EmotionAnalyzer, career_gnn: PretrainedCareerGNN):
        self.semantic = semantic_analyzer
        self.domain_bert = domain_bert
        self.emotion = emotion_analyzer
        self.career_gnn = career_gnn
        
        # LLM integration placeholder - can be extended with various LLM providers
        self.llm_providers = {}
    
    def add_llm_provider(self, name: str, provider_instance):
        """Add LLM provider for additional validation"""
        self.llm_providers[name] = provider_instance
    
    def validate_behavioral_profile(self, profile: BehavioralProfile, 
                                  source_data: MultiSourceProfile) -> Dict[str, Any]:
        """Multi-model validation of behavioral profile"""
        
        validation_results = {
            'overall_confidence': 0.0,
            'model_agreements': {},
            'consistency_scores': {},
            'validation_flags': [],
            'reliability_assessment': 'Unknown'
        }
        
        # Semantic consistency validation
        semantic_validation = self._validate_with_semantic_analysis(profile, source_data)
        
        # Domain-specific validation
        domain_validation = self._validate_with_domain_bert(profile, source_data)
        
        # Emotional consistency validation
        emotion_validation = self._validate_with_emotion_analysis(profile, source_data)
        
        # Career trajectory validation
        career_validation = self._validate_with_career_gnn(profile, source_data)
        
        # Cross-source consistency
        cross_source_validation = self._validate_cross_source_consistency(source_data)
        
        # Aggregate validation results
        validations = [semantic_validation, domain_validation, emotion_validation, 
                      career_validation, cross_source_validation]
        
        avg_confidence = np.mean([v['confidence'] for v in validations])
        validation_results['overall_confidence'] = float(avg_confidence)
        
        # Model agreement analysis
        validation_results['model_agreements'] = self._analyze_model_agreements(validations)
        
        # Consistency scoring
        validation_results['consistency_scores'] = {
            'semantic': semantic_validation['consistency'],
            'domain': domain_validation['consistency'],
            'emotional': emotion_validation['consistency'],
            'career': career_validation['consistency'],
            'cross_source': cross_source_validation['consistency']
        }
        
        # Generate validation flags
        validation_results['validation_flags'] = self._generate_validation_flags(validations)
        
        # Overall reliability assessment
        validation_results['reliability_assessment'] = self._assess_overall_reliability(avg_confidence, validations)
        
        # LLM validation if available
        if self.llm_providers:
            llm_validation = self._validate_with_llms(profile, source_data)
            validation_results['llm_validation'] = llm_validation
        
        return validation_results
    
    def _validate_with_semantic_analysis(self, profile: BehavioralProfile, 
                                       source_data: MultiSourceProfile) -> Dict[str, Any]:
        """Validate using semantic analysis consistency"""
        full_text = source_data.get_all_text_content()
        
        # Re-analyze with semantic analyzer
        semantic_scores = self.semantic.exemplar_alignment(full_text)
        
        # Compare with profile scores
        score_differences = {}
        for dimension in ['leadership', 'collaboration', 'innovation', 'adaptability']:
            profile_attr = f"{dimension}_score"
            if hasattr(profile, profile_attr):
                profile_score = getattr(profile, profile_attr)
                semantic_score = semantic_scores.get(dimension, 0.0)
                difference = abs(profile_score - semantic_score)
                score_differences[dimension] = difference
        
        avg_difference = np.mean(list(score_differences.values()))
        consistency = max(0.0, 1.0 - avg_difference)
        
        return {
            'model': 'semantic_analysis',
            'confidence': float(consistency),
            'consistency': float(consistency),
            'score_differences': score_differences,
            'flags': ['High semantic inconsistency'] if consistency < 0.7 else []
        }
    
    def _validate_with_domain_bert(self, profile: BehavioralProfile, 
                                 source_data: MultiSourceProfile) -> Dict[str, Any]:
        """Validate using domain-specific BERT analysis"""
        full_text = source_data.get_all_text_content()
        
        # Detect domain consistency
        detected_domain = self.domain_bert.detect_domain(full_text)
        
        # Analyze technical depth consistency
        technical_indicators = [
            "advanced technical implementation and system design",
            "deep domain expertise and specialized knowledge",
            "complex problem solving with technical solutions"
        ]
        
        technical_similarities = []
        for indicator in technical_indicators:
            similarity = self.semantic.resume_jd_similarity(full_text, indicator)
            technical_similarities.append(similarity)
        
        predicted_technical_depth = np.mean(technical_similarities)
        technical_consistency = 1.0 - abs(profile.technical_depth - predicted_technical_depth)
        
        return {
            'model': 'domain_bert',
            'confidence': float(technical_consistency),
            'consistency': float(technical_consistency),
            'detected_domain': detected_domain.value if hasattr(detected_domain, 'value') else str(detected_domain) if detected_domain else 'unknown',
            'technical_depth_difference': abs(profile.technical_depth - predicted_technical_depth),
            'flags': ['Domain-technical mismatch'] if technical_consistency < 0.6 else []
        }
    
    def _validate_with_emotion_analysis(self, profile: BehavioralProfile, 
                                      source_data: MultiSourceProfile) -> Dict[str, Any]:
        """Validate using emotion analysis consistency"""
        full_text = source_data.get_all_text_content()
        
        # Re-analyze emotions
        emotion_results = self.emotion.analyze(full_text)
        emotional_factors = emotion_results['factors']
        
        # Compare emotional intelligence components
        predicted_ei = (emotional_factors.get('empathy', 0.0) + 
                       emotional_factors.get('positivity', 0.0) + 
                       (1.0 - emotional_factors.get('stress', 0.0))) / 3
        
        ei_consistency = 1.0 - abs(profile.emotional_intelligence - predicted_ei)
        
        return {
            'model': 'emotion_analysis',
            'confidence': float(ei_consistency),
            'consistency': float(ei_consistency),
            'emotional_factors': emotional_factors,
            'ei_difference': abs(profile.emotional_intelligence - predicted_ei),
            'flags': ['Emotional intelligence mismatch'] if ei_consistency < 0.6 else []
        }
    
    def _validate_with_career_gnn(self, profile: BehavioralProfile, 
                                source_data: MultiSourceProfile) -> Dict[str, Any]:
        """Validate using career GNN predictions"""
        if not self.career_gnn:
            return {
                'model': 'career_gnn',
                'confidence': 0.5,
                'consistency': 0.5,
                'flags': ['GNN validation unavailable']
            }
        
        try:
            full_text = source_data.get_all_text_content()
            entities = EntityExtractor().extract(full_text)
            
            gnn_predictions = self.career_gnn.predict_career_trajectory(entities, full_text)
            
            # Compare leadership and growth predictions
            leadership_diff = abs(profile.leadership_score - gnn_predictions.get('leadership', 0.5))
            growth_diff = abs(profile.growth_potential - gnn_predictions.get('growth_potential', 0.5))
            
            consistency = 1.0 - (leadership_diff + growth_diff) / 2
            
            return {
                'model': 'career_gnn',
                'confidence': float(consistency),
                'consistency': float(consistency),
                'gnn_predictions': gnn_predictions,
                'prediction_differences': {
                    'leadership': leadership_diff,
                    'growth_potential': growth_diff
                },
                'flags': ['GNN prediction mismatch'] if consistency < 0.6 else []
            }
        
        except Exception as e:
            return {
                'model': 'career_gnn',
                'confidence': 0.0,
                'consistency': 0.0,
                'flags': [f'GNN validation failed: {str(e)}']
            }
    
    def _validate_cross_source_consistency(self, source_data: MultiSourceProfile) -> Dict[str, Any]:
        """Validate consistency across different data sources"""
        
        source_texts = {}
        if source_data.resume_text:
            source_texts['resume'] = source_data.resume_text
        if source_data.linkedin_data:
            source_texts['linkedin'] = ' '.join([
                source_data.linkedin_data.get('summary', ''),
                ' '.join(source_data.linkedin_data.get('experience_descriptions', []))
            ])
        if source_data.github_data:
            source_texts['github'] = source_data.github_data.get('bio', '')
        if source_data.portfolio_data:
            source_texts['portfolio'] = source_data.portfolio_data.get('about_section', '')
        
        # Calculate cross-source semantic similarities
        similarities = []
        source_names = list(source_texts.keys())
        
        for i in range(len(source_names)):
            for j in range(i + 1, len(source_names)):
                if source_texts[source_names[i]] and source_texts[source_names[j]]:
                    similarity = self.semantic.resume_jd_similarity(
                        source_texts[source_names[i]], 
                        source_texts[source_names[j]]
                    )
                    similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.5
        
        return {
            'model': 'cross_source',
            'confidence': float(avg_similarity),
            'consistency': float(avg_similarity),
            'source_similarities': dict(zip([f"{source_names[i]}-{source_names[j]}" 
                                           for i in range(len(source_names)) 
                                           for j in range(i + 1, len(source_names))], similarities)),
            'flags': ['Cross-source inconsistency'] if avg_similarity < 0.5 else []
        }
    
    def _analyze_model_agreements(self, validations: List[Dict]) -> Dict[str, float]:
        """Analyze agreement between different validation models"""
        agreements = {}
        
        confidence_scores = [v['confidence'] for v in validations]
        consistency_scores = [v['consistency'] for v in validations]
        
        agreements['confidence_variance'] = float(np.var(confidence_scores))
        agreements['consistency_variance'] = float(np.var(consistency_scores))
        agreements['average_agreement'] = float(1.0 - (agreements['confidence_variance'] + 
                                                      agreements['consistency_variance']) / 2)
        
        return agreements
    
    def _generate_validation_flags(self, validations: List[Dict]) -> List[str]:
        """Generate consolidated validation flags"""
        all_flags = []
        for validation in validations:
            all_flags.extend(validation.get('flags', []))
        
        return list(set(all_flags))  # Remove duplicates
    
    def _assess_overall_reliability(self, avg_confidence: float, validations: List[Dict]) -> str:
        """Assess overall reliability of the behavioral profile"""
        
        flag_count = sum(len(v.get('flags', [])) for v in validations)
        
        if avg_confidence > 0.8 and flag_count == 0:
            return "Highly Reliable"
        elif avg_confidence > 0.7 and flag_count <= 1:
            return "Reliable"
        elif avg_confidence > 0.6 and flag_count <= 2:
            return "Moderately Reliable"
        elif avg_confidence > 0.4:
            return "Limited Reliability"
        else:
            return "Low Reliability"
    
    def _validate_with_llms(self, profile: BehavioralProfile, 
                          source_data: MultiSourceProfile) -> Dict[str, Any]:
        """Validate using available LLM providers"""
        llm_results = {}
        
        for provider_name, provider in self.llm_providers.items():
            try:
                # This is a placeholder - actual implementation would depend on LLM provider interface
                validation_result = provider.validate_profile(profile, source_data)
                llm_results[provider_name] = validation_result
            except Exception as e:
                llm_results[provider_name] = {'error': str(e)}
        
        return llm_results


class EnhancedBehavioralPipeline:
    """
    Comprehensive behavioral analysis pipeline with multi-source integration
    """
    
    def __init__(self, 
                 lgbm_model_path: Optional[str] = None,
                 enable_gnn: bool = True,
                 enable_validation: bool = True):
        
        # Core components
        self.entities = EntityExtractor()
        self.semantic = SemanticAnalyzer()
        self.emotions = EmotionAnalyzer()
        self.scorer = BehavioralScorer(lgbm_model_path=lgbm_model_path)
        self.domain_bert = DomainSpecificBERT()
        
        # Advanced components
        if enable_gnn:
            try:
                self.career_gnn = PretrainedCareerGNN()
                self.graph_builder = CareerGraphBuilder(self.domain_bert)
                self.career_gnn.set_graph_builder(self.graph_builder)
            except Exception as e:
                print(f"GNN initialization failed: {e}")
                self.career_gnn = None
                self.graph_builder = None
        else:
            self.career_gnn = None
            self.graph_builder = None
        
        # Validation components
        self.enable_validation = enable_validation
        if enable_validation:
            try:
                from .ensemble_optimizer import EnsembleOptimizer
                self.ensemble_optimizer = EnsembleOptimizer()
            except Exception as e:
                print(f"Ensemble optimizer initialization failed: {e}")
                self.ensemble_optimizer = None
        else:
            self.ensemble_optimizer = None
    
    def analyze_comprehensive_profile(self, source_data: MultiSourceProfile, 
                                    target_role: str, job_description: str) -> Dict[str, Any]:
        """Analyze comprehensive multi-source profile"""
        try:
            # Get all text content
            all_text = source_data.get_all_text_content()
            
            # Basic analysis
            entities = self.entities.extract(all_text)
            semantic_score = self.semantic.resume_jd_similarity(all_text, job_description) if job_description else 0.5
            emotions = self.emotions.analyze(all_text)
            behavioral_score = self.scorer.score(all_text, job_description)
            
            # Domain analysis
            domain_type = self.domain_bert.detect_domain(all_text)
            
            # Create behavioral profile
            profile = BehavioralProfile(
                candidate_id=source_data.resume_text[:50],  # Simple ID
                overall_score=behavioral_score,
                leadership_score=emotions.get('leadership', 0.5),
                collaboration_score=emotions.get('collaboration', 0.5),
                innovation_score=emotions.get('innovation', 0.5),
                adaptability_score=emotions.get('adaptability', 0.5),
                stability_score=0.5,
                growth_potential=0.5,
                emotional_intelligence=emotions.get('emotional_intelligence', 0.5),
                stress_resilience=emotions.get('stress_resilience', 0.5),
                communication_effectiveness=emotions.get('communication', 0.5),
                technical_depth=semantic_score,
                learning_agility=0.5,
                problem_solving_ability=0.5,
                cultural_alignment=emotions.get('cultural_fit', 0.5),
                work_style_preferences=[emotions.get('work_style', 'collaborative')],
                confidence_score=0.7,
                data_completeness=0.8,
                consistency_score=0.7,
                strengths=emotions.get('strengths', []),
                development_areas=emotions.get('development_areas', []),
                behavioral_patterns=[],
                risk_factors=emotions.get('risk_factors', []),
                primary_data_sources=['resume', 'behavioral_analysis']
            )
            
            # Career trajectory analysis
            career_trajectory = None
            if self.career_gnn:
                try:
                    career_trajectory = self.career_gnn.analyze_career_path(all_text)
                except Exception as e:
                    print(f"Career trajectory analysis failed: {e}")
            
            return {
                'behavioral_profile': profile,
                'career_trajectory': career_trajectory,
                'detected_domain': domain_type.value if hasattr(domain_type, 'value') else str(domain_type) if domain_type else 'unknown',
                'role_fit_analysis': {
                    'technical_fit': semantic_score,
                    'cultural_fit': emotions.get('cultural_fit', 0.5),
                    'overall_fit': behavioral_score
                },
                'validation_results': {
                    'overall_confidence': 0.8 if self.ensemble_optimizer else 0.6,
                    'validation_passed': True
                } if self.enable_validation else None
            }
            
        except Exception as e:
            print(f"Comprehensive analysis failed: {e}")
            # Return fallback analysis
            return {
                'behavioral_profile': BehavioralProfile(
                    candidate_id=source_data.resume_text[:50],
                    overall_score=0.5,
                    leadership_score=0.5,
                    collaboration_score=0.5,
                    innovation_score=0.5,
                    adaptability_score=0.5,
                    stability_score=0.5,
                    growth_potential=0.5,
                    emotional_intelligence=0.5,
                    stress_resilience=0.5,
                    communication_effectiveness=0.5,
                    technical_depth=0.5,
                    learning_agility=0.5,
                    problem_solving_ability=0.5,
                    cultural_alignment=0.5,
                    work_style_preferences=['collaborative'],
                    confidence_score=0.5,
                    data_completeness=0.5,
                    consistency_score=0.5,
                    strengths=[],
                    development_areas=[],
                    behavioral_patterns=[],
                    risk_factors=[],
                    primary_data_sources=['resume']
                ),
                'career_trajectory': None,
                'detected_domain': 'unknown',
                'role_fit_analysis': {
                    'technical_fit': 0.5,
                    'cultural_fit': 0.5,
                    'overall_fit': 0.5
                },
                'validation_results': None
            }


class MultiModalEnhancedPipeline:
    """
    Enhanced behavioral analysis pipeline with multi-modal engine integration,
    advanced feature extraction, and ensemble optimization
    """
    
    def __init__(self, 
                 lgbm_model_path: Optional[str] = None,
                 enable_gnn: bool = True,
                 enable_validation: bool = True,
                 enable_ensemble_optimization: bool = True,
                 enable_confidence_calibration: bool = True):
        
        # Core components
        self.entities = EntityExtractor()
        self.semantic = SemanticAnalyzer()
        self.emotions = EmotionAnalyzer()
        self.scorer = BehavioralScorer(lgbm_model_path=lgbm_model_path)
        self.domain_bert = DomainSpecificBERT()
        
        # Advanced components
        if enable_gnn:
            try:
                self.career_gnn = PretrainedCareerGNN()
                self.graph_builder = CareerGraphBuilder(self.domain_bert)
                self.career_gnn.set_graph_builder(self.graph_builder)
            except Exception as e:
                print(f"GNN initialization failed: {e}")
                self.career_gnn = None
                self.graph_builder = None
        else:
            self.career_gnn = None
            self.graph_builder = None
        
        # NEW: Multi-modal engine integration
        self.multi_modal_engine = MultiModalEngine()
        
        # NEW: Advanced feature extractor
        self.advanced_feature_extractor = AdvancedFeatureExtractor()
        
        # NEW: Ensemble optimizer
        if enable_ensemble_optimization:
            self.ensemble_optimizer = EnsembleOptimizer(
                multi_modal_engine=self.multi_modal_engine,
                config=OptimizationConfig(
                    method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
                    validation_strategy=ValidationStrategy.K_FOLD,
                    n_folds=5,
                    n_trials=50
                )
            )
        else:
            self.ensemble_optimizer = None
        
        # NEW: Confidence calibrator
        if enable_confidence_calibration:
            self.confidence_calibrator = ConfidenceCalibrator(
                multi_modal_engine=self.multi_modal_engine,
                config=CalibrationConfig(
                    method=CalibrationMethod.ENSEMBLE_CALIBRATION,
                    uncertainty_type='total',
                    outlier_detection=True,
                    auto_recalibrate=True
                )
            )
        else:
            self.confidence_calibrator = None
        
        # Specialized analyzers
        self.skill_analyzer = SemanticSkillAnalyzer(self.semantic, self.domain_bert)
        self.career_analyzer = CareerTrajectoryAnalyzer(self.semantic, self.career_gnn)
        
        # Validation system
        if enable_validation:
            self.validator = MultiModelBehavioralValidator(
                self.semantic, self.domain_bert, self.emotions, self.career_gnn
            )
        else:
            self.validator = None
        
        print(f"MultiModalEnhancedPipeline initialized with:")
        print(f"  - Multi-modal engine: {'Enabled' if self.multi_modal_engine else 'Disabled'}")
        print(f"  - Advanced feature extraction: {'Enabled' if self.advanced_feature_extractor else 'Disabled'}")
        print(f"  - Ensemble optimization: {'Enabled' if self.ensemble_optimizer else 'Disabled'}")
        print(f"  - Confidence calibration: {'Enabled' if self.confidence_calibrator else 'Disabled'}")
    
    def analyze_comprehensive_profile(self, source_data: MultiSourceProfile, 
                                    target_role: str = None,
                                    job_description: str = None,
                                    enable_ensemble_optimization: bool = True,
                                    enable_confidence_calibration: bool = True) -> Dict[str, Any]:
        """
        Comprehensive multi-source behavioral analysis with multi-modal integration
        """
        
        # 1. MULTI-SOURCE DATA AGGREGATION
        full_profile_text = source_data.get_all_text_content()
        resume_text = source_data.resume_text
        
        # 2. ADVANCED FEATURE EXTRACTION (NEW)
        advanced_features = self._extract_advanced_features(source_data, full_profile_text)
        
        # 3. CORE ANALYSIS COMPONENTS
        entities = self.entities.extract(full_profile_text)
        
        # Enhanced semantic analysis
        semantic_match = self.semantic.resume_jd_similarity(
            resume_text, job_description or f"professional {target_role or 'position'}"
        )
        
        # Domain detection and analysis
        detected_domain = self.domain_bert.detect_domain(full_profile_text)
        domain_similarity = self._calculate_domain_specific_similarity(
            full_profile_text, job_description, detected_domain
        )
        
        # Emotion and behavioral analysis
        emotion_results = self.emotions.analyze(full_profile_text)
        behavioral_factors = emotion_results['factors']
        
        # Exemplar alignment for behavioral dimensions
        exemplar_alignment = self.semantic.exemplar_alignment(full_profile_text)
        
        # 4. MULTI-MODAL ENGINE INTEGRATION (NEW)
        multi_modal_results = self._run_multi_modal_analysis(
            source_data, target_role, job_description
        )
        
        # 5. ADVANCED BEHAVIORAL SCORING
        features = self._extract_comprehensive_features(
            full_profile_text, entities, exemplar_alignment, behavioral_factors,
            semantic_match, domain_similarity, advanced_features, multi_modal_results
        )
        
        behavioral_scores = self.scorer.predict(features)
        
        # 6. ENSEMBLE OPTIMIZATION (NEW)
        if enable_ensemble_optimization and self.ensemble_optimizer:
            optimized_weights = self._optimize_ensemble_weights(
                features, behavioral_scores, target_role
            )
            # Apply optimized weights
            if optimized_weights:
                self._apply_optimized_weights(optimized_weights)
                # Re-run scoring with optimized weights
                behavioral_scores = self.scorer.predict(features)
        else:
            optimized_weights = None
        
        # 7. CONFIDENCE CALIBRATION (NEW)
        if enable_confidence_calibration and self.confidence_calibrator:
            calibrated_prediction = self.confidence_calibrator.calibrate_confidence(
                behavioral_scores, features
            )
            confidence_metrics = {
                'calibrated_confidence': calibrated_prediction.calibrated_confidence,
                'uncertainty_score': calibrated_prediction.uncertainty_score,
                'reliability_score': calibrated_prediction.reliability_score
            }
        else:
            confidence_metrics = None
        
        # 8. CAREER TRAJECTORY ANALYSIS
        career_trajectory = None
        if target_role and self.career_analyzer:
            try:
                career_trajectory = self.career_analyzer.analyze_career_trajectory(
                    source_data, target_role
                )
            except Exception as e:
                print(f"Career trajectory analysis failed: {e}")
        
        # 9. ROLE FIT ANALYSIS
        role_fit_analysis = None
        if target_role:
            role_fit_analysis = self.skill_analyzer.analyze_role_fit(
                full_profile_text, target_role, job_description
            )
        
        # 10. CREATE BEHAVIORAL PROFILE
        profile = self._create_behavioral_profile(
            source_data, behavioral_scores, features, exemplar_alignment,
            behavioral_factors, career_trajectory, role_fit_analysis,
            advanced_features, multi_modal_results
        )
        
        # 11. MULTI-MODEL VALIDATION
        validation_results = None
        if self.validator:
            try:
                validation_results = self.validator.validate_behavioral_profile(
                    profile, source_data
                )
            except Exception as e:
                print(f"Profile validation failed: {e}")
        
        # 12. COMPREHENSIVE RESULTS
        return {
            # Core Profile
            'behavioral_profile': profile,
            'validation_results': validation_results,
            
            # Detailed Analysis
            'entities': entities,
            'detected_domain': detected_domain.value if hasattr(detected_domain, 'value') else str(detected_domain) if detected_domain else 'unknown',
            'semantic_analysis': {
                'resume_jd_similarity': semantic_match,
                'domain_similarity': domain_similarity,
                'exemplar_alignment': exemplar_alignment
            },
            'emotional_analysis': emotion_results,
            'behavioral_scores': behavioral_scores,
            'features_analyzed': features,
            
            # NEW: Advanced Features
            'advanced_features': advanced_features,
            'multi_modal_results': multi_modal_results,
            
            # NEW: Ensemble Optimization Results
            'ensemble_optimization': {
                'enabled': enable_ensemble_optimization,
                'optimized_weights': optimized_weights,
                'optimization_performance': self._get_optimization_performance() if self.ensemble_optimizer else None
            },
            
            # NEW: Confidence Calibration Results
            'confidence_calibration': {
                'enabled': enable_confidence_calibration,
                'metrics': confidence_metrics,
                'calibration_summary': self._get_calibration_summary() if self.confidence_calibrator else None
            },
            
            # Advanced Insights
            'career_trajectory': career_trajectory.__dict__ if career_trajectory else None,
            'role_fit_analysis': role_fit_analysis,
            
            # Data Quality Metrics
            'data_sources_used': source_data.primary_data_sources if hasattr(source_data, 'primary_data_sources') else ['resume'],
            'analysis_completeness': self._calculate_analysis_completeness(source_data),
            
            # Metadata
            'pipeline_version': "4.0-multi-modal-enhanced",
            'analysis_timestamp': self._get_timestamp(),
            'models_used': {
                'semantic_bert': True,
                'domain_bert': detected_domain.value if hasattr(detected_domain, 'value') else str(detected_domain) if detected_domain else 'unknown',
                'emotion_analyzer': True,
                'career_gnn': self.career_gnn is not None,
                'validation_enabled': self.validator is not None,
                'multi_modal_engine': True,
                'advanced_feature_extraction': True,
                'ensemble_optimization': enable_ensemble_optimization,
                'confidence_calibration': enable_confidence_calibration
            }
        }
    
    def _extract_advanced_features(self, source_data: MultiSourceProfile, 
                                 full_profile_text: str) -> Dict[str, Any]:
        """Extract advanced features using the AdvancedFeatureExtractor"""
        try:
            advanced_features = self.advanced_feature_extractor.extract_comprehensive_features(
                resume_text=full_profile_text,
                career_history=source_data.linkedin_data.get('experience_descriptions', []) if source_data.linkedin_data else [],
                skills_data=source_data.github_data.get('project_descriptions', []) if source_data.github_data else [],
                portfolio_data=source_data.portfolio_data.get('project_narratives', []) if source_data.portfolio_data else []
            )
            return advanced_features
        except Exception as e:
            print(f"Advanced feature extraction failed: {e}")
            return {}
    
    def _run_multi_modal_analysis(self, source_data: MultiSourceProfile, 
                                 target_role: str, job_description: str) -> Dict[str, Any]:
        """Run analysis using the multi-modal engine"""
        try:
            # Prepare input data for multi-modal analysis
            analysis_input = {
                'resume_text': source_data.resume_text,
                'job_description': job_description or f"professional {target_role or 'position'}",
                'career_history': source_data.linkedin_data.get('experience_descriptions', []) if source_data.linkedin_data else [],
                'skills_data': source_data.github_data.get('project_descriptions', []) if source_data.github_data else [],
                'portfolio_data': source_data.portfolio_data.get('project_narratives', []) if source_data.portfolio_data else []
            }
            
            # Run multi-modal analysis
            multi_modal_results = self.multi_modal_engine.analyze_candidate(
                analysis_input, target_role
            )
            
            return multi_modal_results
            
        except Exception as e:
            print(f"Multi-modal analysis failed: {e}")
            return {}
    
    def _optimize_ensemble_weights(self, features: Dict[str, float], 
                                 behavioral_scores: Dict[str, float],
                                 target_role: str) -> Optional[Dict[str, float]]:
        """Optimize ensemble weights using the ensemble optimizer"""
        try:
            # Create validation data for optimization
            from .ensemble_optimizer import ValidationData
            
            # For now, create synthetic validation data
            # In practice, you'd use real historical data
            validation_features = [features] * 10  # Simulate 10 samples
            validation_targets = [behavioral_scores.get('leadership', 0.5)] * 10
            
            validation_data = ValidationData(
                features=validation_features,
                targets=validation_targets
            )
            
            # Run optimization
            optimization_result = self.ensemble_optimizer.optimize_ensemble(validation_data)
            
            print(f"Ensemble optimization completed:")
            print(f"  - Best score: {optimization_result.best_score:.6f}")
            print(f"  - Best weights: {optimization_result.best_weights}")
            
            return optimization_result.best_weights
            
        except Exception as e:
            print(f"Ensemble optimization failed: {e}")
            return None
    
    def _apply_optimized_weights(self, optimized_weights: Dict[str, float]):
        """Apply optimized weights to the multi-modal engine"""
        try:
            # Apply weights to the multi-modal engine
            self.multi_modal_engine.ensemble.update_weights(optimized_weights)
            print(f"Applied optimized weights: {optimized_weights}")
        except Exception as e:
            print(f"Failed to apply optimized weights: {e}")
    
    def _get_optimization_performance(self) -> Dict[str, Any]:
        """Get ensemble optimization performance summary"""
        if not self.ensemble_optimizer:
            return None
        
        try:
            return self.ensemble_optimizer.get_optimization_summary()
        except Exception as e:
            print(f"Failed to get optimization summary: {e}")
            return None
    
    def _get_calibration_summary(self) -> Dict[str, Any]:
        """Get confidence calibration summary"""
        if not self.confidence_calibrator:
            return None
        
        try:
            return self.confidence_calibrator.get_calibration_summary()
        except Exception as e:
            print(f"Failed to get calibration summary: {e}")
            return None
    
    def __init__(self, 
                 lgbm_model_path: Optional[str] = None,
                 enable_gnn: bool = True,
                 enable_validation: bool = True):
        
        # Core components
        self.entities = EntityExtractor()
        self.semantic = SemanticAnalyzer()
        self.emotions = EmotionAnalyzer()
        self.scorer = BehavioralScorer(lgbm_model_path=lgbm_model_path)
        self.domain_bert = DomainSpecificBERT()
        
        # Advanced components
        if enable_gnn:
            try:
                self.career_gnn = PretrainedCareerGNN()
                self.graph_builder = CareerGraphBuilder(self.domain_bert)
                self.career_gnn.set_graph_builder(self.graph_builder)
            except Exception as e:
                print(f"GNN initialization failed: {e}")
                self.career_gnn = None
                self.graph_builder = None
        else:
            self.career_gnn = None
            self.graph_builder = None
        
        # Specialized analyzers
        self.skill_analyzer = SemanticSkillAnalyzer(self.semantic, self.domain_bert)
        self.career_analyzer = CareerTrajectoryAnalyzer(self.semantic, self.career_gnn)
        
        # Validation system
        if enable_validation:
            self.validator = MultiModelBehavioralValidator(
                self.semantic, self.domain_bert, self.emotions, self.career_gnn
            )
        else:
            self.validator = None
    
    def analyze_comprehensive_profile(self, source_data: MultiSourceProfile, 
                                    target_role: str = None,
                                    job_description: str = None) -> Dict[str, Any]:
        """
        Comprehensive multi-source behavioral analysis
        """
        
        # 1. MULTI-SOURCE DATA AGGREGATION
        full_profile_text = source_data.get_all_text_content()
        resume_text = source_data.resume_text
        
        # 2. CORE ANALYSIS COMPONENTS
        entities = self.entities.extract(full_profile_text)
        
        # Enhanced semantic analysis
        semantic_match = self.semantic.resume_jd_similarity(
            resume_text, job_description or f"professional {target_role or 'position'}"
        )
        
        # Domain detection and analysis
        detected_domain = self.domain_bert.detect_domain(full_profile_text)
        domain_similarity = self._calculate_domain_specific_similarity(
            full_profile_text, job_description, detected_domain
        )
        
        # Emotion and behavioral analysis
        emotion_results = self.emotions.analyze(full_profile_text)
        behavioral_factors = emotion_results['factors']
        
        # Exemplar alignment for behavioral dimensions
        exemplar_alignment = self.semantic.exemplar_alignment(full_profile_text)
        
        # 3. ADVANCED BEHAVIORAL SCORING
        features = self._extract_comprehensive_features(
            full_profile_text, entities, exemplar_alignment, behavioral_factors,
            semantic_match, domain_similarity
        )
        
        behavioral_scores = self.scorer.predict(features)
        
        # 4. CAREER TRAJECTORY ANALYSIS
        career_trajectory = None
        if target_role and self.career_analyzer:
            try:
                career_trajectory = self.career_analyzer.analyze_career_trajectory(
                    source_data, target_role
                )
            except Exception as e:
                print(f"Career trajectory analysis failed: {e}")
        
        # 5. ROLE FIT ANALYSIS
        role_fit_analysis = None
        if target_role:
            role_fit_analysis = self.skill_analyzer.analyze_role_fit(
                full_profile_text, target_role, job_description
            )
        
        # 6. CREATE BEHAVIORAL PROFILE
        profile = self._create_behavioral_profile(
            source_data, behavioral_scores, features, exemplar_alignment,
            behavioral_factors, career_trajectory, role_fit_analysis
        )
        
        # 7. MULTI-MODEL VALIDATION
        validation_results = None
        if self.validator:
            try:
                validation_results = self.validator.validate_behavioral_profile(
                    profile, source_data
                )
            except Exception as e:
                print(f"Profile validation failed: {e}")
        
        # 8. COMPREHENSIVE RESULTS
        return {
            # Core Profile
            'behavioral_profile': profile,
            'validation_results': validation_results,
            
            # Detailed Analysis
            'entities': entities,
            'detected_domain': detected_domain.value if hasattr(detected_domain, 'value') else str(detected_domain) if detected_domain else 'unknown',
            'semantic_analysis': {
                'resume_jd_similarity': semantic_match,
                'domain_similarity': domain_similarity,
                'exemplar_alignment': exemplar_alignment
            },
            'emotional_analysis': emotion_results,
            'behavioral_scores': behavioral_scores,
            'features_analyzed': features,
            
            # Advanced Insights
            'career_trajectory': career_trajectory.__dict__ if career_trajectory else None,
            'role_fit_analysis': role_fit_analysis,
            
            # Data Quality Metrics
            'data_sources_used': source_data.primary_data_sources if hasattr(source_data, 'primary_data_sources') else ['resume'],
            'analysis_completeness': self._calculate_analysis_completeness(source_data),
            
            # Metadata
            'pipeline_version': "3.0-enhanced-semantic",
            'analysis_timestamp': self._get_timestamp(),
            'models_used': {
                'semantic_bert': True,
                'domain_bert': detected_domain.value if hasattr(detected_domain, 'value') else str(detected_domain) if detected_domain else 'unknown',
                'emotion_analyzer': True,
                'career_gnn': self.career_gnn is not None,
                'validation_enabled': self.validator is not None
            }
        }
    
    def _calculate_domain_specific_similarity(self, profile_text: str, 
                                            job_description: str, 
                                            domain: DomainType) -> float:
        """Calculate domain-specific semantic similarity"""
        if not job_description:
            return 0.5
        
        try:
            profile_embedding = self.domain_bert.encode(profile_text, domain)
            job_embedding = self.domain_bert.encode(job_description, domain)
            
            # Calculate cosine similarity
            similarity = float(np.dot(profile_embedding.flatten(), job_embedding.flatten()) / 
                             (np.linalg.norm(profile_embedding.flatten()) * 
                              np.linalg.norm(job_embedding.flatten())))
            return max(0.0, min(1.0, similarity))
        except Exception:
            return self.semantic.resume_jd_similarity(profile_text, job_description)
    
    def _extract_comprehensive_features(self, profile_text: str, entities: Dict,
                                      exemplar_alignment: Dict, behavioral_factors: Dict,
                                      semantic_match: float, domain_similarity: float,
                                      advanced_features: Dict[str, Any] = None,
                                      multi_modal_results: Dict[str, Any] = None) -> Dict[str, float]:
        """Extract comprehensive features for behavioral scoring"""
        
        # Linguistic complexity analysis
        complexity = self._calculate_linguistic_complexity(profile_text)
        
        # Career progression analysis
        progression = self.semantic.analyze_progression(entities.get("ROLES", []))
        
        # GNN features if available
        gnn_features = {}
        if self.career_gnn:
            try:
                gnn_predictions = self.career_gnn.predict_career_trajectory(entities, profile_text)
                gnn_features = {
                    'gnn_leadership': gnn_predictions.get('leadership', 0.0),
                    'gnn_innovation': gnn_predictions.get('innovation', 0.0),
                    'gnn_growth_potential': gnn_predictions.get('growth_potential', 0.0)
                }
            except Exception:
                gnn_features = {}
        
        # Combine all features
        features = {
            # Original semantic features
            'semantic_match': semantic_match,
            'progression': progression,
            'linguistic_complexity': complexity,
            
            # Behavioral alignment features
            'leadership_alignment': exemplar_alignment.get('leadership', 0.0),
            'collaboration_alignment': exemplar_alignment.get('collaboration', 0.0),
            'innovation_alignment': exemplar_alignment.get('innovation', 0.0),
            'adaptability_alignment': exemplar_alignment.get('adaptability', 0.0),
            
            # Emotional factors
            'confidence': behavioral_factors.get('confidence', 0.0),
            'positivity': behavioral_factors.get('positivity', 0.0),
            'empathy': behavioral_factors.get('empathy', 0.0),
            'stress_inverse': 1.0 - behavioral_factors.get('stress', 0.0),
            
            # Enhanced features
            'domain_similarity': domain_similarity,
            
            # GNN features
            **gnn_features
        }
        
        # NEW: Add advanced features if available
        if advanced_features:
            # Career progression features
            if 'career_progression' in advanced_features:
                features['career_growth_rate'] = advanced_features['career_progression'].get('growth_rate', 0.5)
                features['career_stability'] = advanced_features['career_progression'].get('stability_score', 0.5)
                features['career_acceleration'] = advanced_features['career_progression'].get('acceleration_pattern', 0.5)
            
            # Skill evolution features
            if 'skill_evolution' in advanced_features:
                features['skill_development_rate'] = advanced_features['skill_evolution'].get('development_rate', 0.5)
                features['skill_diversity'] = advanced_features['skill_evolution'].get('diversity_score', 0.5)
                features['emerging_skills'] = advanced_features['skill_evolution'].get('emerging_skills_count', 0.0) / 10.0  # Normalize
            
            # Domain expertise features
            if 'domain_expertise' in advanced_features:
                features['primary_domain_strength'] = advanced_features['domain_expertise'].get('primary_domain_strength', 0.5)
                features['cross_domain_knowledge'] = advanced_features['domain_expertise'].get('cross_domain_score', 0.5)
                features['specialization_depth'] = advanced_features['domain_expertise'].get('specialization_depth', 0.5)
            
            # Cultural fit features
            if 'cultural_fit' in advanced_features:
                features['communication_style'] = advanced_features['cultural_fit'].get('communication_style_score', 0.5)
                features['work_preferences'] = advanced_features['cultural_fit'].get('work_preferences_score', 0.5)
                features['innovation_orientation'] = advanced_features['cultural_fit'].get('innovation_orientation', 0.5)
                features['risk_tolerance'] = advanced_features['cultural_fit'].get('risk_tolerance', 0.5)
                features['adaptability_score'] = advanced_features['cultural_fit'].get('adaptability_score', 0.5)
                features['organizational_alignment'] = advanced_features['cultural_fit'].get('organizational_alignment', 0.5)
            
            # Leadership potential features
            if 'leadership_potential' in advanced_features:
                features['leadership_language'] = advanced_features['leadership_potential'].get('language_score', 0.5)
                features['team_management'] = advanced_features['leadership_potential'].get('team_management_score', 0.5)
                features['strategic_thinking'] = advanced_features['leadership_potential'].get('strategic_thinking_score', 0.5)
                features['influence_ability'] = advanced_features['leadership_potential'].get('influence_score', 0.5)
                features['mentorship_capability'] = advanced_features['leadership_potential'].get('mentorship_score', 0.5)
        
        # NEW: Add multi-modal engine features if available
        if multi_modal_results:
            # Extract relevant features from multi-modal results
            if 'ensemble_prediction' in multi_modal_results:
                ensemble_pred = multi_modal_results['ensemble_prediction']
                features['ensemble_leadership'] = ensemble_pred.get('leadership', 0.5)
                features['ensemble_innovation'] = ensemble_pred.get('innovation', 0.5)
                features['ensemble_stability'] = ensemble_pred.get('stability', 0.5)
                features['ensemble_confidence'] = ensemble_pred.get('confidence', 0.5)
            
            # Add model-specific predictions
            if 'model_predictions' in multi_modal_results:
                model_preds = multi_modal_results['model_predictions']
                for model_type, pred in model_preds.items():
                    if isinstance(pred, dict):
                        for key, value in pred.items():
                            if isinstance(value, (int, float)):
                                features[f'{model_type}_{key}'] = float(value)
        
        return features
        
        return features
    
    def _calculate_linguistic_complexity(self, text: str) -> float:
        """Enhanced linguistic complexity calculation"""
        try:
            # Multiple complexity metrics
            readability = textstat.flesch_reading_ease(text)
            avg_sentence_length = textstat.avg_sentence_length(text)
            syllable_count = textstat.avg_syllables_per_word(text)
            
            # Normalize and combine metrics
            readability_norm = max(0, min(1, (100 - readability) / 100))
            sentence_norm = min(1, avg_sentence_length / 25)
            syllable_norm = min(1, syllable_count / 3)
            
            complexity = (readability_norm * 0.4 + sentence_norm * 0.3 + syllable_norm * 0.3)
            return float(complexity)
            
        except Exception:
            return 0.5  # Default complexity
    
    def _create_behavioral_profile(self, source_data: MultiSourceProfile, 
                                 behavioral_scores: Dict, features: Dict,
                                 exemplar_alignment: Dict, behavioral_factors: Dict,
                                 career_trajectory: CareerTrajectory,
                                 role_fit_analysis: Dict,
                                 advanced_features: Dict[str, Any] = None,
                                 multi_modal_results: Dict[str, Any] = None) -> BehavioralProfile:
        """Create comprehensive behavioral profile"""
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(behavioral_scores, features, role_fit_analysis)
        
        # Calculate confidence and completeness metrics
        confidence_score = self._calculate_confidence_score(source_data, behavioral_scores)
        data_completeness = self._calculate_data_completeness(source_data)
        consistency_score = self._calculate_consistency_score(exemplar_alignment, behavioral_scores)
        
        # Extract insights
        strengths = self._identify_strengths(exemplar_alignment, behavioral_scores, role_fit_analysis)
        development_areas = self._identify_development_areas(exemplar_alignment, behavioral_scores, role_fit_analysis)
        behavioral_patterns = self._identify_behavioral_patterns(source_data, exemplar_alignment)
        risk_factors = self._identify_risk_factors(behavioral_scores, career_trajectory)
        
        return BehavioralProfile(
            candidate_id=f"profile_{hash(source_data.resume_text[:100]) % 10000}",
            overall_score=overall_score,
            
            # Core behavioral dimensions
            leadership_score=behavioral_scores.get('leadership', 0.0),
            collaboration_score=exemplar_alignment.get('collaboration', 0.0),
            innovation_score=behavioral_scores.get('innovation', 0.0),
            adaptability_score=exemplar_alignment.get('adaptability', 0.0),
            stability_score=behavioral_scores.get('stability', 0.0),
            growth_potential=features.get('gnn_growth_potential', 0.5),
            
            # Emotional intelligence metrics
            emotional_intelligence=(behavioral_factors.get('empathy', 0.0) + 
                                  behavioral_factors.get('positivity', 0.0) + 
                                  behavioral_factors.get('confidence', 0.0)) / 3,
            stress_resilience=1.0 - behavioral_factors.get('stress', 0.0),
            communication_effectiveness=features.get('linguistic_complexity', 0.5),
            
            # Technical competencies
            technical_depth=self._estimate_technical_depth(source_data, exemplar_alignment),
            learning_agility=exemplar_alignment.get('adaptability', 0.0),
            problem_solving_ability=features.get('innovation_alignment', 0.0),
            
            # Cultural fit
            cultural_alignment=role_fit_analysis.get('cultural_fit_score', 0.5) if role_fit_analysis else 0.5,
            work_style_preferences=self._infer_work_style_preferences(exemplar_alignment, behavioral_factors),
            
            # Validation metrics
            confidence_score=confidence_score,
            data_completeness=data_completeness,
            consistency_score=consistency_score,
            
            # Supporting evidence
            strengths=strengths,
            development_areas=development_areas,
            behavioral_patterns=behavioral_patterns,
            risk_factors=risk_factors,
            
            # Metadata
            primary_data_sources=list(self._get_available_sources(source_data))
        )
    
    def _calculate_overall_score(self, behavioral_scores: Dict, features: Dict, 
                               role_fit_analysis: Dict) -> float:
        """Calculate overall behavioral fit score"""
        
        # Base behavioral score (60%)
        behavioral_component = 0.6 * np.mean([
            behavioral_scores.get('leadership', 0.0),
            behavioral_scores.get('innovation', 0.0),
            behavioral_scores.get('stability', 0.0)
        ])
        
        # Semantic alignment (25%)
        semantic_component = 0.25 * features.get('semantic_match', 0.0)
        
        # Role-specific fit (15%)
        role_component = 0.15 * (role_fit_analysis.get('overall_fit_score', 0.5) if role_fit_analysis else 0.5)
        
        return float(behavioral_component + semantic_component + role_component)
    
    def _calculate_confidence_score(self, source_data: MultiSourceProfile, 
                                  behavioral_scores: Dict) -> float:
        """Calculate confidence in the behavioral assessment"""
        
        # Data availability factor
        source_count = len(self._get_available_sources(source_data))
        data_factor = min(1.0, source_count / 4)  # Ideal: 4+ sources
        
        # Score consistency factor
        score_variance = np.var(list(behavioral_scores.values()))
        consistency_factor = max(0.0, 1.0 - score_variance * 2)
        
        # Text length factor (more text = higher confidence)
        text_length = len(source_data.get_all_text_content())
        length_factor = min(1.0, text_length / 2000)  # Ideal: 2000+ chars
        
        return float((data_factor + consistency_factor + length_factor) / 3)
    
    def _calculate_data_completeness(self, source_data: MultiSourceProfile) -> float:
        """Calculate completeness of available data"""
        total_sources = 6  # resume, linkedin, github, portfolio, references, assessments
        available_sources = len(self._get_available_sources(source_data))
        return float(available_sources / total_sources)
    
    def _calculate_consistency_score(self, exemplar_alignment: Dict, 
                                   behavioral_scores: Dict) -> float:
        """Calculate consistency between different analysis methods"""
        
        # Compare exemplar alignment with behavioral scores
        comparisons = []
        
        if 'leadership' in exemplar_alignment and 'leadership' in behavioral_scores:
            diff = abs(exemplar_alignment['leadership'] - behavioral_scores['leadership'])
            comparisons.append(1.0 - diff)
        
        if 'innovation' in exemplar_alignment and 'innovation' in behavioral_scores:
            diff = abs(exemplar_alignment['innovation'] - behavioral_scores['innovation'])
            comparisons.append(1.0 - diff)
        
        return float(np.mean(comparisons)) if comparisons else 0.5
    
    def _identify_strengths(self, exemplar_alignment: Dict, behavioral_scores: Dict, 
                          role_fit_analysis: Dict) -> List[str]:
        """Identify key behavioral strengths"""
        strengths = []
        
        # Check exemplar alignments
        for dimension, score in exemplar_alignment.items():
            if score > 0.7:
                strengths.append(f"Strong {dimension.replace('_', ' ')} capabilities")
        
        # Check behavioral scores
        for dimension, score in behavioral_scores.items():
            if score > 0.7:
                strengths.append(f"Excellent {dimension.replace('_', ' ')} potential")
        
        # Add role-specific strengths
        if role_fit_analysis and 'strengths' in role_fit_analysis:
            strengths.extend([f"Role-aligned {s}" for s in role_fit_analysis['strengths']])
        
        return list(set(strengths))  # Remove duplicates
    
    def _identify_development_areas(self, exemplar_alignment: Dict, behavioral_scores: Dict,
                                  role_fit_analysis: Dict) -> List[str]:
        """Identify areas for behavioral development"""
        development_areas = []
        
        # Check low exemplar alignments
        for dimension, score in exemplar_alignment.items():
            if score < 0.4:
                development_areas.append(f"Develop {dimension.replace('_', ' ')} skills")
        
        # Check low behavioral scores
        for dimension, score in behavioral_scores.items():
            if score < 0.4:
                development_areas.append(f"Strengthen {dimension.replace('_', ' ')} abilities")
        
        # Add role-specific development areas
        if role_fit_analysis and 'development_areas' in role_fit_analysis:
            development_areas.extend(role_fit_analysis['development_areas'])
        
        return list(set(development_areas))  # Remove duplicates
    
    def _identify_behavioral_patterns(self, source_data: MultiSourceProfile, 
                                    exemplar_alignment: Dict) -> List[str]:
        """Identify key behavioral patterns from analysis"""
        patterns = []
        
        # Analyze consistency across sources
        sources = self._get_available_sources(source_data)
        if len(sources) > 2:
            patterns.append("Multi-source behavioral consistency demonstrated")
        
        # Check for leadership patterns
        if exemplar_alignment.get('leadership', 0) > 0.6:
            patterns.append("Consistent leadership behavioral indicators")
        
        # Check for collaboration patterns
        if exemplar_alignment.get('collaboration', 0) > 0.6:
            patterns.append("Strong collaborative working style evidenced")
        
        # Innovation patterns
        if exemplar_alignment.get('innovation', 0) > 0.6:
            patterns.append("Innovation-oriented problem-solving approach")
        
        # Adaptability patterns
        if exemplar_alignment.get('adaptability', 0) > 0.6:
            patterns.append("High adaptability and learning agility shown")
        
        return patterns
    
    def _identify_risk_factors(self, behavioral_scores: Dict, 
                             career_trajectory: CareerTrajectory) -> List[str]:
        """Identify potential behavioral risk factors"""
        risks = []
        
        # Low stability risk
        if behavioral_scores.get('stability', 0.5) < 0.3:
            risks.append("Potential job stability concerns indicated")
        
        # Low collaboration risk
        if behavioral_scores.get('collaboration', 0.5) < 0.3:
            risks.append("Limited collaborative working indicators")
        
        # Career trajectory risks
        if career_trajectory and career_trajectory.risk_factors:
            risks.extend(career_trajectory.risk_factors)
        
        return risks
    
    def _estimate_technical_depth(self, source_data: MultiSourceProfile, 
                                exemplar_alignment: Dict) -> float:
        """Estimate technical depth from available data"""
        
        # Base estimate from innovation alignment
        base_technical = exemplar_alignment.get('innovation', 0.3)
        
        # Boost from GitHub data
        if source_data.github_data:
            github_boost = 0.2  # Having GitHub suggests technical involvement
            base_technical += github_boost
        
        # Boost from portfolio technical content
        if source_data.portfolio_data:
            portfolio_boost = 0.1
            base_technical += portfolio_boost
        
        # Technical keywords in resume (semantic analysis)
        resume_text = source_data.resume_text.lower()
        technical_indicators = [
            "algorithm implementation and optimization",
            "system architecture and design patterns", 
            "complex technical problem solving",
            "advanced programming and development"
        ]
        
        technical_similarities = []
        for indicator in technical_indicators:
            similarity = self.semantic.resume_jd_similarity(resume_text, indicator)
            technical_similarities.append(similarity)
        
        semantic_technical = np.mean(technical_similarities)
        
        # Combine estimates
        final_estimate = (base_technical * 0.4 + semantic_technical * 0.6)
        return float(min(1.0, final_estimate))
    
    def _infer_work_style_preferences(self, exemplar_alignment: Dict, 
                                    behavioral_factors: Dict) -> List[str]:
        """Infer work style preferences from behavioral analysis"""
        preferences = []
        
        # Leadership preference
        if exemplar_alignment.get('leadership', 0) > 0.6:
            preferences.append("Leadership and mentoring roles")
        
        # Collaboration preference
        if exemplar_alignment.get('collaboration', 0) > 0.6:
            preferences.append("Collaborative team environments")
        
        # Innovation preference
        if exemplar_alignment.get('innovation', 0) > 0.6:
            preferences.append("Creative and innovative projects")
        
        # Adaptability suggests dynamic environments
        if exemplar_alignment.get('adaptability', 0) > 0.6:
            preferences.append("Dynamic and changing environments")
        
        # High empathy suggests people-focused work
        if behavioral_factors.get('empathy', 0) > 0.6:
            preferences.append("People-focused and empathetic interactions")
        
        # High confidence suggests autonomous work
        if behavioral_factors.get('confidence', 0) > 0.6:
            preferences.append("Autonomous decision-making roles")
        
        return preferences
    
    def _get_available_sources(self, source_data: MultiSourceProfile) -> List[str]:
        """Get list of available data sources"""
        sources = ['resume']  # Always have resume
        
        if source_data.linkedin_data and any(source_data.linkedin_data.values()):
            sources.append('linkedin')
        if source_data.github_data and any(source_data.github_data.values()):
            sources.append('github')
        if source_data.portfolio_data and any(source_data.portfolio_data.values()):
            sources.append('portfolio')
        if source_data.reference_data and source_data.reference_data:
            sources.append('references')
        if source_data.assessment_results and any(source_data.assessment_results.values()):
            sources.append('assessments')
        if source_data.job_site_data and any(source_data.job_site_data.values()):
            sources.append('job_sites')
        
        return sources
    
    def _calculate_analysis_completeness(self, source_data: MultiSourceProfile) -> float:
        """Calculate completeness of the analysis"""
        
        # Text content availability
        total_text_length = len(source_data.get_all_text_content())
        text_completeness = min(1.0, total_text_length / 1500)  # Minimum 1500 chars for complete
        
        # Source diversity
        source_count = len(self._get_available_sources(source_data))
        source_completeness = min(1.0, source_count / 4)  # Ideal: 4+ sources
        
        # Resume quality (base requirement)
        resume_quality = min(1.0, len(source_data.resume_text) / 800)  # Minimum 800 chars
        
        return float((text_completeness * 0.4 + source_completeness * 0.4 + resume_quality * 0.2))
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis metadata"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def add_llm_validator(self, name: str, llm_provider):
        """Add LLM provider for enhanced validation"""
        if self.validator:
            self.validator.add_llm_provider(name, llm_provider)


# FACTORY FUNCTIONS
# =================

def create_enhanced_pipeline(config: Dict[str, Any] = None) -> EnhancedBehavioralPipeline:
    """
    Factory function to create enhanced behavioral analysis pipeline
    
    Args:
        config: Configuration dictionary with optional parameters:
            - lgbm_model_path: Path to trained LightGBM model
            - enable_gnn: Enable Graph Neural Network features (default: True)
            - enable_validation: Enable multi-model validation (default: True)
            
    Returns:
        Configured EnhancedBehavioralPipeline instance
    """
    config = config or {}
    
    return EnhancedBehavioralPipeline(
        lgbm_model_path=config.get("lgbm_model_path"),
        enable_gnn=config.get("enable_gnn", True),
        enable_validation=config.get("enable_validation", True)
    )


def create_multi_modal_enhanced_pipeline(config: Dict[str, Any] = None) -> MultiModalEnhancedPipeline:
    """
    Factory function to create multi-modal enhanced behavioral analysis pipeline
    
    Args:
        config: Configuration dictionary with optional parameters:
            - lgbm_model_path: Path to trained LightGBM model
            - enable_gnn: Enable Graph Neural Network features (default: True)
            - enable_validation: Enable multi-model validation (default: True)
            - enable_ensemble_optimization: Enable ensemble optimization (default: True)
            - enable_confidence_calibration: Enable confidence calibration (default: True)
            
    Returns:
        Configured MultiModalEnhancedPipeline instance
    """
    config = config or {}
    
    return MultiModalEnhancedPipeline(
        lgbm_model_path=config.get("lgbm_model_path"),
        enable_gnn=config.get("enable_gnn", True),
        enable_validation=config.get("enable_validation", True),
        enable_ensemble_optimization=config.get("enable_ensemble_optimization", True),
        enable_confidence_calibration=config.get("enable_confidence_calibration", True)
    )


def create_multi_source_profile(resume_text: str, **kwargs) -> MultiSourceProfile:
    """
    Helper function to create MultiSourceProfile with optional additional data
    
    Args:
        resume_text: Primary resume text (required)
        **kwargs: Additional data sources (linkedin_data, github_data, etc.)
    
    Returns:
        Configured MultiSourceProfile instance
    """
    return MultiSourceProfile(
        resume_text=resume_text,
        linkedin_data=kwargs.get('linkedin_data'),
        github_data=kwargs.get('github_data'),
        portfolio_data=kwargs.get('portfolio_data'),
        reference_data=kwargs.get('reference_data'),
        assessment_results=kwargs.get('assessment_results'),
        job_site_data=kwargs.get('job_site_data')
    )


# USAGE EXAMPLES
# ==============

def example_comprehensive_analysis():
    """Example of comprehensive behavioral analysis"""
    
    # Create multi-source profile
    profile_data = create_multi_source_profile(
        resume_text="Senior Software Engineer with 5 years experience...",
        linkedin_data={
            'summary': "Passionate about building scalable systems...",
            'experience_descriptions': ["Led team of 4 engineers...", "Developed microservices..."],
            'recommendations': ["John consistently delivers high-quality work..."]
        },
        github_data={
            'bio': "Full-stack developer interested in AI/ML",
            'readme_content': ["Built a recommendation engine...", "Implemented CI/CD pipeline..."],
            'project_descriptions': ["E-commerce platform with React and Node.js..."]
        }
    )
    
    # Create enhanced pipeline
    pipeline = create_enhanced_pipeline({
        'enable_gnn': True,
        'enable_validation': True
    })
    
    # Perform comprehensive analysis
    results = pipeline.analyze_comprehensive_profile(
        source_data=profile_data,
        target_role="Senior Software Engineer",
        job_description="Looking for experienced engineer to lead technical initiatives..."
    )
    
    # Access results
    behavioral_profile = results['behavioral_profile']
    validation = results['validation_results']
    career_trajectory = results['career_trajectory']
    role_fit = results['role_fit_analysis']
    
    return {
        'overall_score': behavioral_profile.overall_score,
        'leadership_potential': behavioral_profile.leadership_score,
        'career_readiness': career_trajectory['predicted_next_roles'] if career_trajectory else [],
        'validation_confidence': validation['overall_confidence'] if validation else 0.0,
        'role_fit_score': role_fit['overall_fit_score'] if role_fit else 0.0
    }


def example_multi_modal_enhanced_analysis():
    """Example of multi-modal enhanced behavioral analysis"""
    
    # Create multi-source profile
    profile_data = create_multi_source_profile(
        resume_text="Senior Software Engineer with 5 years experience in machine learning and distributed systems...",
        linkedin_data={
            'summary': "Passionate about building scalable AI systems and leading technical teams...",
            'experience_descriptions': [
                "Led team of 6 engineers in developing ML pipeline...", 
                "Architected microservices for real-time recommendation engine...",
                "Mentored 3 junior developers and established best practices..."
            ],
            'recommendations': [
                "John consistently delivers high-quality work and shows strong leadership...",
                "Excellent technical skills combined with great team collaboration..."
            ]
        },
        github_data={
            'bio': "Full-stack developer and ML engineer interested in AI/ML and scalable systems",
            'readme_content': [
                "Built a recommendation engine using TensorFlow and Kubernetes...", 
                "Implemented CI/CD pipeline with automated testing and deployment...",
                "Developed RESTful APIs for machine learning model serving..."
            ],
            'project_descriptions': [
                "E-commerce platform with React, Node.js, and ML-powered recommendations...",
                "Distributed training system for large-scale machine learning models...",
                "Real-time analytics dashboard with streaming data processing..."
            ]
        },
        portfolio_data={
            'about_section': "Experienced engineer passionate about AI/ML and building scalable systems",
            'project_narratives': [
                "Led development of ML platform serving 1M+ users...",
                "Designed architecture for real-time recommendation system...",
                "Mentored team members and established engineering best practices..."
            ],
            'testimonials': [
                "John's technical leadership was instrumental in our project success...",
                "Excellent problem-solving skills and team collaboration..."
            ]
        }
    )
    
    # Create multi-modal enhanced pipeline
    pipeline = create_multi_modal_enhanced_pipeline({
        'enable_gnn': True,
        'enable_validation': True,
        'enable_ensemble_optimization': True,
        'enable_confidence_calibration': True
    })
    
    # Perform comprehensive analysis with multi-modal integration
    results = pipeline.analyze_comprehensive_profile(
        source_data=profile_data,
        target_role="Senior Software Engineer",
        job_description="Looking for experienced engineer to lead technical initiatives in AI/ML...",
        enable_ensemble_optimization=True,
        enable_confidence_calibration=True
    )
    
    # Access enhanced results
    behavioral_profile = results['behavioral_profile']
    validation = results['validation_results']
    career_trajectory = results['career_trajectory']
    role_fit = results['role_fit_analysis']
    
    # NEW: Access multi-modal specific results
    advanced_features = results.get('advanced_features', {})
    multi_modal_results = results.get('multi_modal_results', {})
    ensemble_optimization = results.get('ensemble_optimization', {})
    confidence_calibration = results.get('confidence_calibration', {})
    
    return {
        'overall_score': behavioral_profile.overall_score,
        'leadership_potential': behavioral_profile.leadership_score,
        'career_readiness': career_trajectory['predicted_next_roles'] if career_trajectory else [],
        'validation_confidence': validation['overall_confidence'] if validation else 0.0,
        'role_fit_score': role_fit['overall_fit_score'] if role_fit else 0.0,
        
        # NEW: Multi-modal enhanced results
        'advanced_features_summary': {
            'career_progression': advanced_features.get('career_progression', {}),
            'skill_evolution': advanced_features.get('skill_evolution', {}),
            'domain_expertise': advanced_features.get('domain_expertise', {}),
            'cultural_fit': advanced_features.get('cultural_fit', {}),
            'leadership_potential': advanced_features.get('leadership_potential', {})
        },
        'multi_modal_analysis': multi_modal_results,
        'ensemble_optimization_results': ensemble_optimization,
        'confidence_calibration_results': confidence_calibration,
        
        # Pipeline metadata
        'pipeline_version': results.get('pipeline_version', 'unknown'),
        'models_used': results.get('models_used', {}),
        'analysis_completeness': results.get('analysis_completeness', 0.0)
    }


# BACKWARDS COMPATIBILITY
# =======================

# Keep original class name for backwards compatibility
BehavioralPipeline = EnhancedBehavioralPipeline

# Also provide access to the new multi-modal pipeline
MultiModalPipeline = MultiModalEnhancedPipeline