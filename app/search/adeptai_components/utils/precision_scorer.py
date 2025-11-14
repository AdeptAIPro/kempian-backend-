"""
Precision scoring utilities
"""
import re
from typing import Dict, List
from app.simple_logger import get_logger
from collections import Counter

class AdvancedSkillMatcher:
    """Advanced skill matching with context awareness"""
    
    def __init__(self):
        self.skill_synonyms = {
            'javascript': ['js', 'javascript', 'ecmascript'],
            'python': ['python', 'py'],
            'react': ['react', 'reactjs', 'react.js'],
            'node': ['node', 'nodejs', 'node.js'],
            'aws': ['aws', 'amazon web services'],
            'sql': ['sql', 'mysql', 'postgresql', 'sqlite']
        }
        
        self.proficiency_indicators = {
            'expert': ['expert', 'mastery', 'guru', '10+ years', 'architect'],
            'senior': ['senior', 'advanced', '5+ years', 'lead'],
            'intermediate': ['intermediate', 'experienced', '2-4 years'],
            'junior': ['junior', 'basic', 'learning', '0-2 years']
        }
    
    def extract_skills_from_text(self, text: str) -> Dict[str, Dict]:
        """Extract skills with proficiency levels"""
        text_lower = text.lower()
        found_skills = {}
        
        for canonical_skill, synonyms in self.skill_synonyms.items():
            for synonym in synonyms:
                if synonym in text_lower:
                    # Extract context around the skill
                    context = self._get_skill_context(text_lower, synonym)
                    proficiency = self._determine_proficiency(context)
                    
                    found_skills[canonical_skill] = {
                        'proficiency': proficiency,
                        'context': context,
                        'mentioned_as': synonym
                    }
                    break
        
        return found_skills
    
    def _get_skill_context(self, text: str, skill: str) -> str:
        """Get context around skill mention"""
        words = text.split()
        skill_words = skill.split()
        
        for i in range(len(words) - len(skill_words) + 1):
            if ' '.join(words[i:i+len(skill_words)]) == skill:
                start = max(0, i - 5)
                end = min(len(words), i + len(skill_words) + 5)
                return ' '.join(words[start:end])
        return ""
    
    def _determine_proficiency(self, context: str) -> str:
        """Determine proficiency level from context"""
        for level, indicators in self.proficiency_indicators.items():
            if any(indicator in context for indicator in indicators):
                return level
        return 'intermediate'  # default
    
    def calculate_skill_match_score(self, candidate_skills: List[str], required_skills: List[str]) -> float:
        """Calculate advanced skill matching score"""
        if not required_skills:
            return 1.0
        
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        matches = 0
        
        for required_skill in required_skills:
            required_lower = required_skill.lower()
            
            # Direct match
            if required_lower in candidate_skills_lower:
                matches += 1
                continue
            
            # Synonym match
            if required_lower in self.skill_synonyms:
                synonyms = self.skill_synonyms[required_lower]
                if any(syn in candidate_skills_lower for syn in synonyms):
                    matches += 0.8  # Slightly lower score for synonym match
                    continue
        
        return min(matches / len(required_skills), 1.0)

# Global instance
skill_matcher = AdvancedSkillMatcher()