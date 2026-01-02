"""
ATS (Applicant Tracking System) Analyzer Service
Analyzes resumes for ATS compatibility and provides improvement suggestions
"""
import os
import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from app.simple_logger import get_logger

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = get_logger(__name__.split('.')[-1])


@dataclass
class ATSSuggestion:
    """Represents a single ATS improvement suggestion"""
    category: str  # 'formatting', 'keywords', 'structure', 'content', 'technical'
    priority: str  # 'critical', 'high', 'medium', 'low'
    title: str
    description: str
    current_state: Optional[str] = None
    recommended_action: Optional[str] = None
    impact: Optional[str] = None  # Expected impact on ATS score


@dataclass
class ATSSectionScore:
    """Score for a specific resume section"""
    section: str
    score: int  # 0-100
    max_score: int = 100
    issues: List[str] = None
    strengths: List[str] = None


@dataclass
class ATSAnalysisResult:
    """Complete ATS analysis result"""
    overall_score: int  # 0-100
    ats_compatibility: str  # 'excellent', 'good', 'fair', 'poor'
    section_scores: Dict[str, ATSSectionScore]
    suggestions: List[ATSSuggestion]
    critical_issues: List[str]
    keyword_analysis: Dict[str, Any]
    formatting_analysis: Dict[str, Any]
    structure_analysis: Dict[str, Any]
    estimated_improvement_potential: int  # Potential score improvement if all suggestions applied


class ATSAnalyzer:
    """Service for analyzing resume ATS compatibility"""
    
    def __init__(self):
        # Common ATS-friendly keywords by category
        self.action_verbs = [
            'achieved', 'improved', 'increased', 'decreased', 'managed', 'led',
            'developed', 'created', 'designed', 'implemented', 'optimized',
            'analyzed', 'resolved', 'delivered', 'executed', 'coordinated',
            'collaborated', 'established', 'maintained', 'enhanced', 'streamlined'
        ]
        
        # ATS-unfriendly elements
        self.problematic_elements = [
            r'<img[^>]*>',  # Images
            r'<table[^>]*>.*?</table>',  # Complex tables
            r'[^\x00-\x7F]+',  # Special characters that might break parsing
        ]
        
        # Required sections for ATS compatibility
        self.required_sections = [
            'contact', 'experience', 'education', 'skills'
        ]
        
        # Initialize OpenAI if available
        self.openai_client = None
        self.openai_available = False
        if OPENAI_AVAILABLE:
            openai_api_key = os.getenv('OPENAI_API_KEY') or os.getenv('CHATGPT_API_KEY')
            if openai_api_key:
                try:
                    try:
                        from openai import OpenAI
                        self.openai_client = OpenAI(api_key=openai_api_key)
                        self.openai_available = True
                        logger.info("OpenAI API initialized for ATS analysis")
                    except (ImportError, AttributeError):
                        openai.api_key = openai_api_key
                        self.openai_client = openai
                        self.openai_available = True
                        logger.info("OpenAI API initialized for ATS analysis (legacy)")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI for ATS analysis: {e}")
    
    def analyze_resume(self, resume_text: str, parsed_data: Dict[str, Any] = None) -> ATSAnalysisResult:
        """
        Analyze resume for ATS compatibility
        
        Args:
            resume_text: Raw text extracted from resume
            parsed_data: Optional parsed resume data from resume_parser
            
        Returns:
            ATSAnalysisResult with scores, suggestions, and analysis
        """
        if not resume_text or len(resume_text.strip()) < 50:
            return self._create_empty_result("Resume text is too short or empty")
        
        # Normalize text for analysis
        normalized_text = resume_text.lower()
        
        # Analyze different aspects
        formatting_score, formatting_issues = self._analyze_formatting(resume_text, normalized_text)
        structure_score, structure_issues = self._analyze_structure(resume_text, normalized_text, parsed_data)
        keyword_score, keyword_analysis = self._analyze_keywords(resume_text, normalized_text, parsed_data)
        content_score, content_issues = self._analyze_content(resume_text, normalized_text, parsed_data)
        technical_score, technical_issues = self._analyze_technical_aspects(resume_text, normalized_text)
        
        # Calculate overall score (weighted average)
        section_scores = {
            'formatting': ATSSectionScore('formatting', formatting_score, issues=formatting_issues),
            'structure': ATSSectionScore('structure', structure_score, issues=structure_issues),
            'keywords': ATSSectionScore('keywords', keyword_score, issues=keyword_analysis.get('issues', [])),
            'content': ATSSectionScore('content', content_score, issues=content_issues),
            'technical': ATSSectionScore('technical', technical_score, issues=technical_issues)
        }
        
        # Weighted overall score
        overall_score = int(
            formatting_score * 0.20 +
            structure_score * 0.25 +
            keyword_score * 0.25 +
            content_score * 0.20 +
            technical_score * 0.10
        )
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            formatting_score, structure_score, keyword_score, content_score, technical_score,
            formatting_issues, structure_issues, keyword_analysis, content_issues, technical_issues,
            resume_text, parsed_data
        )
        
        # Identify critical issues
        critical_issues = [
            issue for suggestion in suggestions 
            if suggestion.priority == 'critical' 
            for issue in [suggestion.title]
        ]
        
        # Determine compatibility level
        if overall_score >= 85:
            compatibility = 'excellent'
        elif overall_score >= 70:
            compatibility = 'good'
        elif overall_score >= 50:
            compatibility = 'fair'
        else:
            compatibility = 'poor'
        
        # Estimate improvement potential
        improvement_potential = min(100 - overall_score, sum(
            10 if s.priority == 'critical' else 5 if s.priority == 'high' else 2 if s.priority == 'medium' else 1
            for s in suggestions
        ))
        
        return ATSAnalysisResult(
            overall_score=overall_score,
            ats_compatibility=compatibility,
            section_scores=section_scores,
            suggestions=suggestions,
            critical_issues=critical_issues,
            keyword_analysis=keyword_analysis,
            formatting_analysis={'score': formatting_score, 'issues': formatting_issues},
            structure_analysis={'score': structure_score, 'issues': structure_issues},
            estimated_improvement_potential=improvement_potential
        )
    
    def _analyze_formatting(self, resume_text: str, normalized_text: str) -> tuple[int, List[str]]:
        """Analyze resume formatting for ATS compatibility"""
        score = 100
        issues = []
        
        # Check for problematic elements
        if re.search(r'<img[^>]*>', resume_text, re.IGNORECASE):
            score -= 20
            issues.append("Contains images which ATS systems cannot parse")
        
        # Check for complex tables
        table_count = len(re.findall(r'<table[^>]*>', resume_text, re.IGNORECASE))
        if table_count > 2:
            score -= 15
            issues.append(f"Contains {table_count} tables which may confuse ATS parsing")
        
        # Check for special characters that might break parsing
        special_chars = re.findall(r'[^\x00-\x7F]', resume_text)
        if len(special_chars) > 50:
            score -= 10
            issues.append("Contains many special characters that may cause parsing issues")
        
        # Check for proper line breaks
        if '\n\n' not in resume_text and '\r\n\r\n' not in resume_text:
            score -= 10
            issues.append("Missing proper paragraph breaks - may affect section detection")
        
        # Check for consistent formatting
        if len(set(re.findall(r'^\s*[•\-\*]\s*', resume_text, re.MULTILINE))) > 3:
            score -= 5
            issues.append("Inconsistent bullet point formatting")
        
        return max(0, score), issues
    
    def _analyze_structure(self, resume_text: str, normalized_text: str, parsed_data: Dict = None) -> tuple[int, List[str]]:
        """Analyze resume structure"""
        score = 100
        issues = []
        
        # Check for required sections
        has_contact = bool(re.search(r'(email|phone|contact)', normalized_text)) or (parsed_data and parsed_data.get('email'))
        has_experience = bool(re.search(r'(experience|work|employment|position|role)', normalized_text)) or (parsed_data and parsed_data.get('work_experience'))
        has_education = bool(re.search(r'(education|degree|university|college|school)', normalized_text)) or (parsed_data and parsed_data.get('education'))
        has_skills = bool(re.search(r'(skills|competencies|proficiencies)', normalized_text)) or (parsed_data and parsed_data.get('skills'))
        
        if not has_contact:
            score -= 25
            issues.append("Missing contact information section")
        if not has_experience:
            score -= 30
            issues.append("Missing work experience section")
        if not has_education:
            score -= 15
            issues.append("Missing education section")
        if not has_skills:
            score -= 20
            issues.append("Missing skills section")
        
        # Check section organization
        section_headers = re.findall(r'^[A-Z][A-Z\s]+$', resume_text, re.MULTILINE)
        if len(section_headers) < 3:
            score -= 10
            issues.append("Resume lacks clear section headers")
        
        # Check for proper chronological order (experience should be reverse chronological)
        experience_section = re.search(r'(experience|work history|employment).*?(?=education|skills|$)', resume_text, re.IGNORECASE | re.DOTALL)
        if experience_section:
            dates = re.findall(r'(\d{4}|\w+\s+\d{4})', experience_section.group(0))
            if len(dates) >= 2:
                # Simple check - more recent dates should appear first
                pass  # Could add more sophisticated date ordering check
        
        return max(0, score), issues
    
    def _analyze_keywords(self, resume_text: str, normalized_text: str, parsed_data: Dict = None) -> tuple[int, Dict[str, Any]]:
        """Analyze keyword usage and optimization"""
        score = 100
        issues = []
        strengths = []
        
        # Count action verbs
        action_verb_count = sum(1 for verb in self.action_verbs if verb in normalized_text)
        if action_verb_count < 5:
            score -= 15
            issues.append(f"Only {action_verb_count} action verbs found - use more achievement-focused language")
        elif action_verb_count >= 10:
            strengths.append(f"Strong use of {action_verb_count} action verbs")
        
        # Check for quantified achievements
        quantifiers = re.findall(r'\d+%|\d+\+|\$\d+|\d+\s*(?:years?|months?|people|team|projects?)', resume_text, re.IGNORECASE)
        if len(quantifiers) < 3:
            score -= 20
            issues.append("Limited quantified achievements - add metrics and numbers")
        elif len(quantifiers) >= 5:
            strengths.append(f"Good use of {len(quantifiers)} quantified achievements")
        
        # Check for skills keywords
        skills = parsed_data.get('skills', []) if parsed_data else []
        if isinstance(skills, list) and len(skills) < 5:
            score -= 15
            issues.append(f"Only {len(skills)} skills listed - add more relevant technical and soft skills")
        elif isinstance(skills, list) and len(skills) >= 10:
            strengths.append(f"Comprehensive list of {len(skills)} skills")
        
        # Check for industry-specific keywords
        tech_keywords = ['software', 'development', 'programming', 'code', 'api', 'database', 'system']
        healthcare_keywords = ['patient', 'care', 'medical', 'health', 'clinical', 'treatment']
        business_keywords = ['management', 'strategy', 'business', 'marketing', 'sales', 'revenue']
        
        keyword_matches = {
            'tech': sum(1 for kw in tech_keywords if kw in normalized_text),
            'healthcare': sum(1 for kw in healthcare_keywords if kw in normalized_text),
            'business': sum(1 for kw in business_keywords if kw in normalized_text)
        }
        
        max_matches = max(keyword_matches.values())
        if max_matches < 3:
            score -= 10
            issues.append("Limited industry-specific keywords - tailor resume to target industry")
        
        return max(0, score), {
            'issues': issues,
            'strengths': strengths,
            'action_verbs_count': action_verb_count,
            'quantifiers_count': len(quantifiers),
            'skills_count': len(skills) if isinstance(skills, list) else 0,
            'keyword_matches': keyword_matches
        }
    
    def _analyze_content(self, resume_text: str, normalized_text: str, parsed_data: Dict = None) -> tuple[int, List[str]]:
        """Analyze content quality"""
        score = 100
        issues = []
        
        # Check resume length
        word_count = len(resume_text.split())
        if word_count < 200:
            score -= 20
            issues.append(f"Resume is too short ({word_count} words) - expand with more details")
        elif word_count > 1000:
            score -= 10
            issues.append(f"Resume is very long ({word_count} words) - consider condensing to 1-2 pages")
        
        # Check for summary/objective
        has_summary = bool(re.search(r'(summary|objective|profile|about)', normalized_text)) or (parsed_data and parsed_data.get('summary'))
        if not has_summary:
            score -= 10
            issues.append("Missing professional summary or objective statement")
        
        # Check for achievements vs responsibilities
        achievement_words = ['achieved', 'improved', 'increased', 'reduced', 'saved', 'won', 'awarded']
        responsibility_words = ['responsible', 'duties', 'tasks', 'assigned']
        
        achievement_count = sum(1 for word in achievement_words if word in normalized_text)
        responsibility_count = sum(1 for word in responsibility_words if word in normalized_text)
        
        if responsibility_count > achievement_count * 2:
            score -= 15
            issues.append("Too many responsibilities listed - focus more on achievements and results")
        
        # Check for dates in experience
        if parsed_data and parsed_data.get('work_experience'):
            exp_text = str(parsed_data.get('work_experience', ''))
            dates = re.findall(r'\d{4}', exp_text)
            if len(dates) < 2:
                score -= 10
                issues.append("Work experience missing dates - add employment dates for each position")
        
        return max(0, score), issues
    
    def _analyze_technical_aspects(self, resume_text: str, normalized_text: str) -> tuple[int, List[str]]:
        """Analyze technical ATS compatibility"""
        score = 100
        issues = []
        
        # Check file format compatibility (this would be checked at upload, but good to mention)
        # Check for proper encoding
        try:
            resume_text.encode('utf-8')
        except UnicodeEncodeError:
            score -= 20
            issues.append("Encoding issues detected - may cause ATS parsing problems")
        
        # Check for headers/footers that might confuse ATS
        if re.search(r'page\s*\d+', normalized_text):
            score -= 5
            issues.append("Contains page numbers - remove for better ATS compatibility")
        
        # Check for proper spacing
        if re.search(r'\s{4,}', resume_text):
            score -= 5
            issues.append("Contains excessive spacing - use consistent formatting")
        
        # Check for email format
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, resume_text)
        if not emails:
            score -= 15
            issues.append("No email address found - ensure contact information is included")
        elif len(emails) > 1:
            score -= 5
            issues.append("Multiple email addresses found - use one primary email")
        
        # Check for phone format
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,9}'
        phones = re.findall(phone_pattern, resume_text)
        if not phones:
            score -= 10
            issues.append("No phone number found - add contact phone number")
        
        return max(0, score), issues
    
    def _generate_suggestions(
        self, formatting_score: int, structure_score: int, keyword_score: int,
        content_score: int, technical_score: int,
        formatting_issues: List[str], structure_issues: List[str],
        keyword_analysis: Dict, content_issues: List[str], technical_issues: List[str],
        resume_text: str, parsed_data: Dict = None
    ) -> List[ATSSuggestion]:
        """Generate actionable suggestions for improvement"""
        suggestions = []
        
        # Formatting suggestions
        if formatting_score < 80:
            priority = 'critical' if formatting_score < 50 else 'high' if formatting_score < 70 else 'medium'
            suggestions.append(ATSSuggestion(
                category='formatting',
                priority=priority,
                title="Improve Resume Formatting",
                description="Your resume contains formatting elements that may confuse ATS systems",
                current_state=f"Formatting score: {formatting_score}/100",
                recommended_action="Remove images, simplify tables, use standard fonts, ensure proper spacing",
                impact="Can improve ATS parsing accuracy by 15-25%"
            ))
        
        # Structure suggestions
        if structure_score < 80:
            priority = 'critical' if structure_score < 50 else 'high' if structure_score < 70 else 'medium'
            missing_sections = [issue for issue in structure_issues if 'missing' in issue.lower()]
            suggestions.append(ATSSuggestion(
                category='structure',
                priority=priority,
                title="Improve Resume Structure",
                description="Your resume is missing key sections or has structural issues",
                current_state=f"Structure score: {formatting_score}/100. Issues: {', '.join(missing_sections[:2])}",
                recommended_action="Add missing sections (Contact, Experience, Education, Skills) with clear headers",
                impact="Critical for ATS systems to properly categorize your information"
            ))
        
        # Keyword suggestions
        if keyword_score < 80:
            priority = 'high' if keyword_score < 60 else 'medium'
            keyword_issues = keyword_analysis.get('issues', [])
            suggestions.append(ATSSuggestion(
                category='keywords',
                priority=priority,
                title="Optimize Keywords and Language",
                description="Your resume needs better keyword optimization for ATS matching",
                current_state=f"Keyword score: {keyword_score}/100",
                recommended_action=f"{'; '.join(keyword_issues[:2])}. Use more action verbs and industry-specific terms",
                impact="Can improve job matching by 20-30%"
            ))
        
        # Content suggestions
        if content_score < 80:
            priority = 'high' if content_score < 60 else 'medium'
            suggestions.append(ATSSuggestion(
                category='content',
                priority=priority,
                title="Enhance Content Quality",
                description="Your resume content needs improvement for better ATS performance",
                current_state=f"Content score: {content_score}/100",
                recommended_action="Focus on achievements over responsibilities, add quantified results, include professional summary",
                impact="Can improve overall ATS score by 10-20%"
            ))
        
        # Technical suggestions
        if technical_score < 90:
            priority = 'high' if technical_score < 70 else 'medium'
            suggestions.append(ATSSuggestion(
                category='technical',
                priority=priority,
                title="Fix Technical Issues",
                description="Your resume has technical issues that may affect ATS parsing",
                current_state=f"Technical score: {technical_score}/100",
                recommended_action="Ensure proper encoding, remove page numbers, verify contact information format",
                impact="Critical for ATS systems to read your resume correctly"
            ))
        
        # Specific actionable suggestions based on issues
        all_issues = formatting_issues + structure_issues + content_issues + technical_issues
        for issue in all_issues[:5]:  # Top 5 issues
            if 'missing' in issue.lower() and 'section' in issue.lower():
                suggestions.append(ATSSuggestion(
                    category='structure',
                    priority='critical',
                    title=f"Add Missing Section",
                    description=issue,
                    recommended_action="Create a dedicated section with a clear header",
                    impact="Essential for ATS parsing"
                ))
        
        return suggestions
    
    def _create_empty_result(self, reason: str) -> ATSAnalysisResult:
        """Create an empty result when analysis cannot be performed"""
        return ATSAnalysisResult(
            overall_score=0,
            ats_compatibility='poor',
            section_scores={},
            suggestions=[ATSSuggestion(
                category='technical',
                priority='critical',
                title="Cannot Analyze Resume",
                description=reason
            )],
            critical_issues=[reason],
            keyword_analysis={},
            formatting_analysis={},
            structure_analysis={},
            estimated_improvement_potential=0
        )
    
    def get_improved_template_suggestions(self, analysis_result: ATSAnalysisResult) -> Dict[str, Any]:
        """Generate template improvement suggestions based on analysis"""
        template_suggestions = {
            'recommended_format': 'PDF',
            'recommended_structure': [],
            'section_order': ['Contact', 'Summary', 'Experience', 'Education', 'Skills', 'Certifications'],
            'formatting_guidelines': [
                'Use standard fonts (Arial, Calibri, Times New Roman)',
                'Use consistent heading styles',
                'Avoid images and graphics',
                'Use simple bullet points',
                'Keep formatting simple and clean'
            ],
            'keyword_optimization': {
                'action_verbs': self.action_verbs[:10],
                'suggested_quantifiers': ['%', 'years', 'people', 'projects', '$']
            }
        }
        
        # Add specific recommendations based on analysis
        if analysis_result.section_scores.get('structure', ATSSectionScore('structure', 0)).score < 70:
            template_suggestions['recommended_structure'].append('Add clear section headers')
        
        if analysis_result.section_scores.get('keywords', ATSSectionScore('keywords', 0)).score < 70:
            template_suggestions['recommended_structure'].append('Include more industry keywords')
        
        return template_suggestions


# Global instance
ats_analyzer = ATSAnalyzer()

