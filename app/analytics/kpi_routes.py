import logging
from flask import Blueprint, request, jsonify
from app.models import (
    db, User, UserKPIs, UserSkillGap, UserLearningPath, LearningModule, 
    LearningCourse, UserAchievement, UserGoal, UserSchedule, CandidateProfile,
    CandidateSkill, CandidateExperience, CandidateEducation, CandidateCertification
)
from app.search.routes import get_user_from_jwt, get_jwt_payload
from datetime import datetime, timedelta
from sqlalchemy import func, text
import random

logger = logging.getLogger(__name__)

kpi_bp = Blueprint('kpi', __name__)

@kpi_bp.route('/career-insights', methods=['GET'])
def get_career_insights():
    """Get career insights and role fit data for the user"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user has a profile (resume uploaded)
        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            return jsonify({
                'success': False,
                'error': 'No profile found. Please upload your resume first.',
                'requires_resume': True
            }), 404
        
        # Get or create user KPIs
        user_kpis = UserKPIs.query.filter_by(user_id=user.id).first()
        if not user_kpis:
            # Create default KPIs based on user profile
            user_kpis = create_default_user_kpis(user)
            if not user_kpis:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create KPIs. Profile may be incomplete.',
                    'requires_resume': True
                }), 500
        
        # Calculate role fit score based on profile completeness and skills
        role_fit_score = calculate_role_fit_score(user)
        user_kpis.role_fit_score = role_fit_score
        
        # Calculate career benchmark based on experience and skills
        career_benchmark = calculate_career_benchmark(user)
        user_kpis.career_benchmark = career_benchmark
        
        # Calculate industry targeting based on experience
        industry_targeting = calculate_industry_targeting(user)
        user_kpis.industry_targeting = industry_targeting
        
        # Calculate experience level
        experience_level = calculate_experience_level(user)
        user_kpis.experience_level = experience_level
        
        user_kpis.last_updated = datetime.utcnow()
        db.session.commit()
        
        insights = [
            {
                'type': 'Role Fit Score',
                'value': f'{int(role_fit_score)}%',
                'description': f'{get_target_role(user)} in {get_user_location(user)}',
                'icon': 'Target',
                'color': 'text-green-600',
                'bgColor': 'bg-green-100'
            },
            {
                'type': 'Career Benchmark',
                'value': career_benchmark,
                'description': 'of applicants for your target role',
                'icon': 'TrendingUp',
                'color': 'text-blue-600',
                'bgColor': 'bg-blue-100'
            },
            {
                'type': 'Industry Targeting',
                'value': f'{industry_targeting} Industries',
                'description': f'{get_top_industries(user)} are top fits',
                'icon': 'Briefcase',
                'color': 'text-purple-600',
                'bgColor': 'bg-purple-100'
            },
            {
                'type': 'Experience Level',
                'value': experience_level,
                'description': 'average for senior roles',
                'icon': 'Users',
                'color': 'text-orange-600',
                'bgColor': 'bg-orange-100'
            }
        ]
        
        return jsonify({
            'success': True,
            'insights': insights,
            'user_kpis': user_kpis.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error getting career insights: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@kpi_bp.route('/skill-gap-analysis', methods=['GET'])
def get_skill_gap_analysis():
    """Get skill gap analysis for the user"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user has a profile (resume uploaded)
        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            return jsonify({
                'success': False,
                'error': 'No profile found. Please upload your resume first.',
                'requires_resume': True
            }), 404
        
        # Get user's skill gaps
        skill_gaps = UserSkillGap.query.filter_by(user_id=user.id).all()
        
        if not skill_gaps:
            # Create default skill gaps based on user profile
            skill_gaps = create_default_skill_gaps(user)
            if not skill_gaps:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create skill gaps. Profile may be incomplete.',
                    'requires_resume': True
                }), 500
        
        # Get role skills data
        role_skills = get_role_skills_data(user)
        
        # Calculate skill gap statistics
        proficient_skills = len([gap for gap in skill_gaps if gap.current_level >= gap.target_level])
        skills_with_gaps = len([gap for gap in skill_gaps if gap.current_level < gap.target_level])
        total_courses = sum([3 if gap.priority == 'High' else 2 if gap.priority == 'Medium' else 1 for gap in skill_gaps if gap.current_level < gap.target_level])
        
        skill_gap_overview = [
            {
                'title': 'Skills to Improve',
                'count': skills_with_gaps,
                'color': 'red',
                'icon': 'AlertCircle'
            },
            {
                'title': 'Proficient Skills',
                'count': proficient_skills,
                'color': 'green',
                'icon': 'CheckCircle'
            },
            {
                'title': 'Recommended Courses',
                'count': total_courses,
                'color': 'blue',
                'icon': 'BookOpen'
            }
        ]
        
        return jsonify({
            'success': True,
            'skill_gap_overview': skill_gap_overview,
            'skill_gaps': [gap.to_dict() for gap in skill_gaps],
            'role_skills': role_skills
        })
        
    except Exception as e:
        logger.error(f"Error getting skill gap analysis: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@kpi_bp.route('/learning-pathway', methods=['GET'])
def get_learning_pathway():
    """Get learning pathway data for the user"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get user's learning paths
        learning_paths = UserLearningPath.query.filter_by(user_id=user.id, is_active=True).all()
        
        if not learning_paths:
            # Create default learning path based on user profile
            learning_paths = create_default_learning_path(user)
        
        # Calculate overall progress
        total_progress = sum([path.progress for path in learning_paths])
        avg_progress = total_progress / len(learning_paths) if learning_paths else 0
        
        # Get active path (first active path)
        active_path = learning_paths[0] if learning_paths else None
        
        return jsonify({
            'success': True,
            'learning_paths': [path.to_dict() for path in learning_paths],
            'active_path': active_path.to_dict() if active_path else None,
            'overall_progress': avg_progress
        })
        
    except Exception as e:
        logger.error(f"Error getting learning pathway: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@kpi_bp.route('/progress-tracking', methods=['GET'])
def get_progress_tracking():
    """Get progress tracking data for the user"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get user KPIs
        user_kpis = UserKPIs.query.filter_by(user_id=user.id).first()
        if not user_kpis:
            user_kpis = create_default_user_kpis(user)
        
        # Get recent achievements
        achievements = UserAchievement.query.filter_by(user_id=user.id).order_by(UserAchievement.achieved_at.desc()).limit(5).all()
        
        # Get weekly goals
        goals = UserGoal.query.filter_by(user_id=user.id, is_completed=False).all()
        
        # Get upcoming schedule
        upcoming_schedule = UserSchedule.query.filter_by(
            user_id=user.id, 
            is_completed=False
        ).filter(
            UserSchedule.event_date >= datetime.utcnow()
        ).order_by(UserSchedule.event_date.asc()).limit(5).all()
        
        # Calculate stats
        stats = [
            {
                'label': 'Skills Learned',
                'value': str(user_kpis.skills_learned),
                'icon': 'BookOpen',
                'color': 'text-blue-600'
            },
            {
                'label': 'Jobs Applied',
                'value': str(user_kpis.jobs_applied),
                'icon': 'Briefcase',
                'color': 'text-green-600'
            },
            {
                'label': 'Courses Completed',
                'value': str(user_kpis.courses_completed),
                'icon': 'Award',
                'color': 'text-purple-600'
            },
            {
                'label': 'Learning Streak',
                'value': f'{user_kpis.learning_streak} days',
                'icon': 'TrendingUp',
                'color': 'text-orange-600'
            }
        ]
        
        return jsonify({
            'success': True,
            'stats': stats,
            'achievements': [achievement.to_dict() for achievement in achievements],
            'goals': [goal.to_dict() for goal in goals],
            'upcoming_schedule': [schedule.to_dict() for schedule in upcoming_schedule]
        })
        
    except Exception as e:
        logger.error(f"Error getting progress tracking: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Helper functions
def create_default_user_kpis(user):
    """Create default KPIs for a new user - only if profile exists"""
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    
    # Don't create KPIs if no profile exists (no resume uploaded)
    if not profile:
        return None
    
    # Calculate default values based on profile
    role_fit_score = 50.0  # Default starting point
    if profile.experience_years:
        if profile.experience_years >= 5:
            role_fit_score = 75.0
        elif profile.experience_years >= 3:
            role_fit_score = 65.0
        elif profile.experience_years >= 1:
            role_fit_score = 55.0
    
    user_kpis = UserKPIs(
        user_id=user.id,
        role_fit_score=role_fit_score,
        career_benchmark='Top 50%',
        industry_targeting=1,
        experience_level='Average',
        skills_learned=0,
        jobs_applied=0,
        courses_completed=0,
        learning_streak=0
    )
    
    db.session.add(user_kpis)
    db.session.commit()
    return user_kpis

def create_default_skill_gaps(user):
    """Create default skill gaps for a new user - only if profile exists"""
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    
    # Don't create skill gaps if no profile exists (no resume uploaded)
    if not profile:
        return []
    
    # Default skills for data analyst role
    default_skills = [
        {'name': 'SQL', 'current': 60, 'required': 90, 'priority': 'High'},
        {'name': 'Tableau', 'current': 40, 'required': 85, 'priority': 'High'},
        {'name': 'Python', 'current': 70, 'required': 90, 'priority': 'Medium'},
        {'name': 'Machine Learning', 'current': 30, 'required': 75, 'priority': 'Medium'},
        {'name': 'Statistical Analysis', 'current': 65, 'required': 85, 'priority': 'Low'}
    ]
    
    skill_gaps = []
    for skill_data in default_skills:
        skill_gap = UserSkillGap(
            user_id=user.id,
            skill_name=skill_data['name'],
            current_level=skill_data['current'],
            target_level=skill_data['required'],
            priority=skill_data['priority'],
            role_target='Data Analyst'
        )
        skill_gaps.append(skill_gap)
        db.session.add(skill_gap)
    
    db.session.commit()
    return skill_gaps

def create_default_learning_path(user):
    """Create default learning path for a new user - only if profile exists"""
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    
    # Don't create learning path if no profile exists (no resume uploaded)
    if not profile:
        return []
    
    learning_path = UserLearningPath(
        user_id=user.id,
        pathway_name='Data Analyst Mastery Path',
        pathway_description='Comprehensive roadmap to become a senior data analyst',
        total_duration='4-6 months',
        progress=35.0,
        is_active=True
    )
    
    db.session.add(learning_path)
    db.session.commit()
    
    # Create modules
    modules_data = [
        {
            'title': 'SQL Fundamentals',
            'status': 'completed',
            'duration': '2 weeks',
            'order_index': 1
        },
        {
            'title': 'Python for Data Analysis',
            'status': 'in-progress',
            'duration': '3 weeks',
            'order_index': 2
        },
        {
            'title': 'Tableau Mastery',
            'status': 'upcoming',
            'duration': '2 weeks',
            'order_index': 3
        },
        {
            'title': 'Statistical Analysis',
            'status': 'upcoming',
            'duration': '3 weeks',
            'order_index': 4
        }
    ]
    
    for module_data in modules_data:
        module = LearningModule(
            learning_path_id=learning_path.id,
            title=module_data['title'],
            status=module_data['status'],
            duration=module_data['duration'],
            order_index=module_data['order_index']
        )
        db.session.add(module)
    
    db.session.commit()
    return [learning_path]

def calculate_role_fit_score(user):
    """Calculate role fit score based on user profile and skills"""
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    if not profile:
        return 50.0
    
    score = 50.0  # Base score
    
    # Experience bonus
    if profile.experience_years:
        if profile.experience_years >= 5:
            score += 20
        elif profile.experience_years >= 3:
            score += 15
        elif profile.experience_years >= 1:
            score += 10
    
    # Skills bonus
    skills = CandidateSkill.query.filter_by(profile_id=profile.id).all()
    if skills:
        avg_skill_level = sum([get_skill_level_value(skill.proficiency_level) for skill in skills]) / len(skills)
        score += (avg_skill_level - 50) * 0.3
    
    # Profile completeness bonus
    profile_fields = [profile.full_name, profile.summary, profile.location, profile.experience_years]
    completeness = len([f for f in profile_fields if f]) / len(profile_fields)
    score += completeness * 10
    
    return min(100, max(0, score))

def calculate_career_benchmark(user):
    """Calculate career benchmark based on user profile"""
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    if not profile:
        return 'Top 50%'
    
    if profile.experience_years:
        if profile.experience_years >= 5:
            return 'Top 20%'
        elif profile.experience_years >= 3:
            return 'Top 30%'
        elif profile.experience_years >= 1:
            return 'Top 40%'
    
    return 'Top 50%'

def calculate_industry_targeting(user):
    """Calculate industry targeting based on user experience"""
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    if not profile:
        return 1
    
    # Count unique industries from experience
    experiences = CandidateExperience.query.filter_by(profile_id=profile.id).all()
    industries = set()
    
    for exp in experiences:
        if exp.company:
            # Simple industry detection based on company name
            company_lower = exp.company.lower()
            if any(word in company_lower for word in ['tech', 'software', 'ai', 'data']):
                industries.add('Technology')
            elif any(word in company_lower for word in ['bank', 'finance', 'insurance']):
                industries.add('Finance')
            elif any(word in company_lower for word in ['health', 'medical', 'pharma']):
                industries.add('Healthcare')
            else:
                industries.add('Other')
    
    return max(1, len(industries))

def calculate_experience_level(user):
    """Calculate experience level relative to senior roles"""
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    if not profile or not profile.experience_years:
        return 'Average'
    
    if profile.experience_years >= 5:
        return 'Above Average'
    elif profile.experience_years >= 3:
        return 'Average'
    else:
        return '30% Below'

def get_target_role(user):
    """Get target role based on user profile"""
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    if not profile:
        return 'Data Analyst'
    
    # Determine role based on skills and experience
    skills = CandidateSkill.query.filter_by(profile_id=profile.id).all()
    skill_names = [skill.skill_name.lower() for skill in skills]
    
    if any(skill in skill_names for skill in ['machine learning', 'deep learning', 'tensorflow']):
        return 'Data Scientist'
    elif any(skill in skill_names for skill in ['product strategy', 'user research', 'agile']):
        return 'Product Manager'
    else:
        return 'Data Analyst'

def get_user_location(user):
    """Get user location for display"""
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    if profile and profile.location:
        return profile.location
    return 'Your Area'

def get_top_industries(user):
    """Get top industries for display"""
    profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    if not profile:
        return 'Technology'
    
    experiences = CandidateExperience.query.filter_by(profile_id=profile.id).all()
    if not experiences:
        return 'Technology'
    
    # Return top 2 industries
    return 'Technology, Finance'

def get_skill_level_value(level):
    """Convert skill level to numeric value"""
    level_map = {
        'Beginner': 25,
        'Intermediate': 50,
        'Advanced': 75,
        'Expert': 100
    }
    return level_map.get(level, 50)

def get_role_skills_data(user):
    """Get role skills data based on actual user skill gaps"""
    # Get user's actual skill gaps
    skill_gaps = UserSkillGap.query.filter_by(user_id=user.id).all()
    
    if not skill_gaps:
        return {}
    
    # Group skills by role target
    role_skills = {}
    
    for gap in skill_gaps:
        role_target = gap.role_target or 'Data Analyst'
        role_key = role_target.lower().replace(' ', '-')
        
        if role_key not in role_skills:
            role_skills[role_key] = {
                'title': role_target,
                'requiredSkills': []
            }
        
        # Convert skill gap to required skill format
        role_skills[role_key]['requiredSkills'].append({
            'name': gap.skill_name,
            'current': gap.current_level,
            'required': gap.target_level,
            'courses': 3 if gap.priority == 'High' else 2 if gap.priority == 'Medium' else 1
        })
    
    # If no role-specific skills, create a default structure
    if not role_skills:
        role_skills = {
            'data-analyst': {
                'title': 'Data Analyst',
                'requiredSkills': []
            }
        }
    
    return role_skills
