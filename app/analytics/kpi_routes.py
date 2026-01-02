import logging
import re
import json as json_lib
from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import (
    db, User, UserKPIs, UserSkillGap, UserLearningPath, LearningModule, 
    LearningCourse, UserAchievement, UserGoal, UserSchedule, CandidateProfile,
    CandidateSkill, CandidateExperience, CandidateEducation, CandidateCertification,
    CandidateProject
)
from app.search.routes import get_user_from_jwt, get_jwt_payload
from datetime import datetime, timedelta
from sqlalchemy import func, text
import random

logger = get_logger("analytics")

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

@kpi_bp.route('/talent-insights', methods=['GET', 'OPTIONS'])
def get_talent_insights():
    """
    Get comprehensive talent insights generated from resume data.
    This endpoint generates all insights (KPIs, skill gaps, benchmarking, learning paths, courses)
    from the user's resume/profile data using AI services.
    
    Query params:
    - role: Optional target role for analysis (defaults to user's desired role or current position)
    
    Returns:
    {
        "success": true,
        "user_kpis": {...},
        "skill_gaps": [...],
        "chatgpt_skill_gaps": [...],
        "chatgpt_benchmarking": [...],
        "learning_paths": [...],
        "progress_tracking": {...},
        "platform_courses": [...],
        "ai_career_insight": "..."
    }
    """
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response, 200
    
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
        
        # Get target role from query params or profile
        target_role = request.args.get('role', '')
        if not target_role:
            # Try to get from saved target role for insights
            if profile.target_role_for_insights:
                target_role = profile.target_role_for_insights
            else:
                # Get from current experience
                current_exp = CandidateExperience.query.filter_by(
                    profile_id=profile.id, 
                    is_current=True
                ).first()
                if current_exp:
                    target_role = current_exp.position
                else:
                    target_role = 'Your Target Role'
        
        # Get all profile data from resume
        skills = CandidateSkill.query.filter_by(profile_id=profile.id).all()
        experiences = CandidateExperience.query.filter_by(profile_id=profile.id).all()
        education = CandidateEducation.query.filter_by(profile_id=profile.id).all()
        certifications = CandidateCertification.query.filter_by(profile_id=profile.id).all()
        projects = CandidateProject.query.filter_by(profile_id=profile.id).all()
        
        # Build profile data for AI analysis
        profile_data = {
            'full_name': profile.full_name or '',
            'summary': profile.summary or '',
            'experience_years': profile.experience_years or 0,
            'location': profile.location or '',
            'skills': [skill.skill_name for skill in skills],
            'experiences': [
                {
                    'position': exp.position,
                    'company': exp.company,
                    'start_date': str(exp.start_date) if exp.start_date else None,
                    'end_date': str(exp.end_date) if exp.end_date else None,
                    'is_current': exp.is_current,
                    'description': exp.description or ''
                }
                for exp in experiences
            ],
            'education': [
                {
                    'institution': edu.institution,
                    'degree': edu.degree,
                    'field_of_study': edu.field_of_study
                }
                for edu in education
            ],
            'certifications': [cert.name for cert in certifications],
            'projects': [proj.name for proj in projects],
            'target_role': target_role
        }
        
        # Get or create user KPIs
        user_kpis = UserKPIs.query.filter_by(user_id=user.id).first()
        if not user_kpis:
            user_kpis = create_default_user_kpis(user)
        
        # Calculate role fit score
        role_fit_score = calculate_role_fit_score(user)
        if user_kpis:
            user_kpis.role_fit_score = role_fit_score
            user_kpis.last_updated = datetime.utcnow()
            db.session.commit()
        
        # Get skill gaps from database
        skill_gaps = UserSkillGap.query.filter_by(user_id=user.id).all()
        if not skill_gaps:
            skill_gaps = create_default_skill_gaps(user)
        
        # Check if insights were recently generated (within last 24 hours)
        # If so, use cached data from database
        use_cached_insights = False
        if profile.insights_generated_at:
            time_diff = datetime.utcnow() - profile.insights_generated_at
            if time_diff.total_seconds() < 86400:  # 24 hours
                use_cached_insights = True
                logger.info(f"Using cached insights for user {user.id} (generated {time_diff.total_seconds()/3600:.1f} hours ago)")
        
        # Generate AI insights using AI service
        chatgpt_skill_gaps = []
        chatgpt_benchmarking = []
        platform_courses = []
        ai_career_insight = ''
        
        # Use cached insights if available and recent
        if use_cached_insights:
            if profile.ai_career_insight:
                ai_career_insight = profile.ai_career_insight
            if profile.benchmarking_data:
                chatgpt_benchmarking = profile.benchmarking_data
            if profile.recommended_courses:
                platform_courses = profile.recommended_courses
            # Get skill gaps from database (already saved)
            chatgpt_skill_gaps = [
                {
                    'skill_name': gap.skill_name,
                    'current_level': gap.current_level,
                    'target_level': gap.target_level
                }
                for gap in UserSkillGap.query.filter_by(user_id=user.id, role_target=target_role).all()
            ]
        else:
            # Generate new insights
            try:
                from app.ai.service import AIService
                ai_service = AIService()
                
                # Build comprehensive profile context for accurate analysis
                experience_summary = ""
                if profile_data['experiences']:
                    exp_list = []
                    for exp in profile_data['experiences'][:5]:  # Top 5 experiences
                        exp_str = f"{exp['position']} at {exp.get('company', 'Unknown')}"
                        if exp.get('start_date'):
                            exp_str += f" ({exp['start_date']}"
                            if exp.get('end_date'):
                                exp_str += f" - {exp['end_date']}"
                            elif exp.get('is_current'):
                                exp_str += " - Present"
                            exp_str += ")"
                        if exp.get('description'):
                            exp_str += f": {exp['description'][:200]}"
                        exp_list.append(exp_str)
                    experience_summary = "\n".join(exp_list)
                
                education_summary = ""
                if profile_data['education']:
                    edu_list = []
                    for edu in profile_data['education'][:3]:  # Top 3 education entries
                        edu_str = f"{edu.get('degree', 'Degree')}"
                        if edu.get('institution'):
                            edu_str += f" from {edu['institution']}"
                        if edu.get('field_of_study'):
                            edu_str += f" in {edu['field_of_study']}"
                        edu_list.append(edu_str)
                    education_summary = "\n".join(edu_list)
                
                projects_summary = ""
                if profile_data['projects']:
                    projects_summary = ", ".join(profile_data['projects'][:10])
                
                certifications_summary = ""
                if profile_data['certifications']:
                    certifications_summary = ", ".join(profile_data['certifications'][:10])
                
                # Generate skill gap analysis and benchmarking using AI with comprehensive context
                ai_prompt = f"""You are an expert career analyst and talent assessment specialist. Analyze the following candidate profile extracted from their resume and provide accurate, data-driven insights.

CANDIDATE PROFILE:
Name: {profile_data['full_name']}
Total Experience: {profile_data['experience_years']} years
Location: {profile_data.get('location', 'Not specified')}
Professional Summary: {profile_data.get('summary', 'Not provided')[:500]}

CURRENT SKILLS (from resume):
{', '.join(profile_data['skills']) if profile_data['skills'] else 'No skills listed'}

WORK EXPERIENCE:
{experience_summary if experience_summary else 'No experience listed'}

EDUCATION:
{education_summary if education_summary else 'No education listed'}

CERTIFICATIONS:
{certifications_summary if certifications_summary else 'No certifications listed'}

PROJECTS:
{projects_summary if projects_summary else 'No projects listed'}

TARGET ROLE: {target_role}

ANALYSIS REQUIREMENTS:

1. SKILL GAPS ANALYSIS:
   - Identify 5-8 critical skills required for {target_role} role
   - Assess candidate's current skill level (0-100) based on:
     * Years of experience with each skill
     * Depth of experience (mentioned in multiple roles = higher level)
     * Education/certifications related to the skill
     * Projects demonstrating the skill
   - Set target level (0-100) based on industry standards for {target_role}
   - Focus on skills that are: (a) Critical for the role, (b) Have a measurable gap, (c) Are achievable to improve

2. INDUSTRY BENCHMARKING:
   - Compare candidate's skills to top 10% performers in {target_role}
   - For each skill, provide:
     * Current level vs top performers level
     * Gap percentage (how far behind top performers)
     * Specific, actionable recommendation (1-2 sentences) on how to close the gap
   - Be realistic and encouraging - focus on achievable improvements

3. CAREER INSIGHT:
   - Write a 3-4 sentence personalized career summary that:
     * Highlights their strengths and experience
     * Identifies their readiness for {target_role}
     * Mentions 1-2 key areas for growth
     * Provides encouragement and direction
   - Be specific, professional, and actionable

RETURN FORMAT (JSON only, no markdown):
{{
    "skillGaps": [
        {{"skill_name": "exact skill name", "current_level": 0-100, "target_level": 0-100}}
    ],
    "benchmarking": [
        {{"skill_name": "exact skill name", "current_level": 0-100, "top_performers_level": 0-100, "gap_percentage": 0-100, "actionable": "specific recommendation"}}
    ],
    "career_insight": "personalized 3-4 sentence summary"
}}

IMPORTANT:
- Use realistic skill levels based on actual experience evidence
- Target levels should reflect industry standards for {target_role}
- Gap percentage = ((top_performers_level - current_level) / top_performers_level) * 100
- Return ONLY valid JSON, no markdown, no explanations, no code blocks"""
                
                ai_result = ai_service.generate_response(
                    prompt=ai_prompt,
                    context={'task_type': 'talent_insights', 'target_role': target_role},
                    temperature=0.3,  # Lower temperature for more accurate, consistent results
                    max_tokens=2500  # Increased for more detailed analysis
                )
                
                if ai_result.get('success'):
                    raw_text = ai_result.get('response', '')
                    try:
                        # Try to extract JSON
                        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                        if match:
                            ai_data = json_lib.loads(match.group(0))
                            chatgpt_skill_gaps = ai_data.get('skillGaps', [])
                            chatgpt_benchmarking = ai_data.get('benchmarking', [])
                            ai_career_insight = ai_data.get('career_insight', '')
                            
                            # Save AI-generated skill gaps to database
                            if chatgpt_skill_gaps:
                                for skill_gap_data in chatgpt_skill_gaps:
                                    skill_name = skill_gap_data.get('skill_name', '')
                                    current_level = skill_gap_data.get('current_level', 0)
                                    target_level = skill_gap_data.get('target_level', 0)
                                    
                                    if skill_name:
                                        # Check if skill gap already exists for this user and role
                                        existing_gap = UserSkillGap.query.filter_by(
                                            user_id=user.id,
                                            skill_name=skill_name,
                                            role_target=target_role
                                        ).first()
                                        
                                        if existing_gap:
                                            # Update existing gap
                                            existing_gap.current_level = current_level
                                            existing_gap.target_level = target_level
                                            existing_gap.updated_at = datetime.utcnow()
                                            # Determine priority based on gap size
                                            gap_size = target_level - current_level
                                            if gap_size >= 30:
                                                existing_gap.priority = 'High'
                                            elif gap_size >= 15:
                                                existing_gap.priority = 'Medium'
                                            else:
                                                existing_gap.priority = 'Low'
                                        else:
                                            # Create new skill gap
                                            gap_size = target_level - current_level
                                            priority = 'High' if gap_size >= 30 else 'Medium' if gap_size >= 15 else 'Low'
                                            new_gap = UserSkillGap(
                                                user_id=user.id,
                                                skill_name=skill_name,
                                                current_level=current_level,
                                                target_level=target_level,
                                                role_target=target_role,
                                                priority=priority
                                            )
                                            db.session.add(new_gap)
                            
                            # Save benchmarking data and career insight to profile
                            profile.benchmarking_data = chatgpt_benchmarking if chatgpt_benchmarking else None
                            profile.ai_career_insight = ai_career_insight if ai_career_insight else None
                            profile.target_role_for_insights = target_role  # Save the selected role
                            profile.insights_generated_at = datetime.utcnow()
                            profile.updated_at = datetime.utcnow()
                            
                            db.session.commit()
                            logger.info(f"Saved AI insights to database for user {user.id} with target role: {target_role}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse AI insights: {e}")
                        db.session.rollback()
                
                # Generate course recommendations with better context
                if profile_data['skills']:
                    # Identify top skill gaps for course recommendations
                    top_gaps = sorted(chatgpt_skill_gaps, key=lambda x: x.get('target_level', 0) - x.get('current_level', 0), reverse=True)[:5]
                    gap_skills = [gap.get('skill_name', '') for gap in top_gaps if gap.get('skill_name')]
                    
                    course_prompt = f"""You are a learning and development expert. Recommend 4-6 highly relevant online courses to help this candidate transition to {target_role}.

CANDIDATE CONTEXT:
- Current Skills: {', '.join(profile_data['skills'][:20])}
- Experience Level: {profile_data['experience_years']} years
- Target Role: {target_role}
- Top Skill Gaps to Address: {', '.join(gap_skills) if gap_skills else 'General role requirements'}
- Education Background: {education_summary if education_summary else 'Not specified'}

REQUIREMENTS:
1. Recommend courses that directly address the skill gaps for {target_role}
2. Mix of platforms: Udemy, Coursera, Khan Academy, edX, or Pluralsight
3. Courses should match their experience level ({profile_data['experience_years']} years)
4. Include a mix of: foundational courses (if gaps are large) and advanced courses (if they have good base)
5. Prioritize courses with high ratings (4.0+) and practical, hands-on content
6. Consider their learning style - prefer courses with projects and real-world applications

For each course, provide:
- Title: Specific, descriptive course name
- Platform: One of "Udemy", "Coursera", "Khan Academy", "edX", or "Pluralsight"
- Description: 1-2 sentences explaining why this course is relevant (mention specific skill it addresses)
- URL: Realistic course URL format (e.g., "https://www.udemy.com/course/[course-name]/" or "https://www.coursera.org/learn/[course-name]")
- Rating: Number between 4.0 and 5.0 (realistic rating)
- Duration: Realistic duration (e.g., "8 hours", "4 weeks", "6 hours", "12 hours")
- Level: "Beginner", "Intermediate", or "Advanced" (match to their current level + gap)
- Price: Realistic price (e.g., "$49.99", "$79.99", "Free", "$39.99")

RETURN FORMAT (JSON array only):
[
    {{"title": "Course Title", "platform": "Udemy", "description": "Why relevant", "url": "https://...", "rating": 4.5, "duration": "8 hours", "level": "Intermediate", "price": "$49.99"}}
]

Return ONLY valid JSON array, no markdown, no explanations."""
                
                course_result = ai_service.generate_response(
                    prompt=course_prompt,
                    context={'task_type': 'course_recommendations', 'target_role': target_role},
                    temperature=0.4,  # Slightly lower for more focused recommendations
                    max_tokens=1200  # Increased for better course descriptions
                )
                
                if course_result.get('success'):
                    try:
                        match = re.search(r'\[.*\]', course_result.get('response', ''), re.DOTALL)
                        if match:
                            platform_courses = json_lib.loads(match.group(0))
                            
                            # Save course recommendations to profile
                            if platform_courses:
                                profile.recommended_courses = platform_courses
                                profile.updated_at = datetime.utcnow()
                                db.session.commit()
                                logger.info(f"Saved course recommendations to database for user {user.id}")
                    except Exception as e:
                        logger.warning(f"Failed to parse course recommendations: {e}")
                        db.session.rollback()
        
            except Exception as ai_error:
                logger.error(f"Error generating AI insights: {ai_error}")
                # Continue without AI insights, but try to use any existing cached data
                if profile.ai_career_insight:
                    ai_career_insight = profile.ai_career_insight
                if profile.benchmarking_data:
                    chatgpt_benchmarking = profile.benchmarking_data
                if profile.recommended_courses:
                    platform_courses = profile.recommended_courses
        
        # Get learning paths
        learning_paths = UserLearningPath.query.filter_by(user_id=user.id, is_active=True).all()
        if not learning_paths:
            learning_paths = create_default_learning_path(user)
        
        # Get progress tracking
        achievements = UserAchievement.query.filter_by(user_id=user.id).order_by(UserAchievement.achieved_at.desc()).limit(5).all()
        goals = UserGoal.query.filter_by(user_id=user.id, is_completed=False).all()
        upcoming_schedule = UserSchedule.query.filter_by(
            user_id=user.id, 
            is_completed=False
        ).filter(
            UserSchedule.event_date >= datetime.utcnow()
        ).order_by(UserSchedule.event_date.asc()).limit(5).all()
        
        progress_stats = []
        if user_kpis:
            progress_stats = [
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
            'user_kpis': user_kpis.to_dict() if user_kpis else None,
            'skill_gaps': [gap.to_dict() for gap in skill_gaps],
            'chatgpt_skill_gaps': chatgpt_skill_gaps,
            'chatgpt_benchmarking': chatgpt_benchmarking,
            'learning_paths': [path.to_dict() for path in learning_paths],
            'progress_tracking': {
                'stats': progress_stats,
                'achievements': [achievement.to_dict() for achievement in achievements],
                'goals': [goal.to_dict() for goal in goals],
                'upcoming_schedule': [schedule.to_dict() for schedule in upcoming_schedule]
            },
            'platform_courses': platform_courses,
            'ai_career_insight': ai_career_insight,
            'target_role': target_role
        })
        
    except Exception as e:
        logger.error(f"Error getting talent insights: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500
