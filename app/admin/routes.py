from flask import Blueprint, jsonify, request
from app.simple_logger import get_logger
from app.models import User, db, CandidateProfile, UserSocialLinks, UserTrial, CeipalIntegration, JDSearchLog, UserImage, UserKPIs, UserSkillGap, UserLearningPath, UserAchievement, UserGoal, UserSchedule, Tenant, TenantAlert, CandidateSkill, CandidateEducation, CandidateExperience, CandidateCertification, CandidateProject, UnlimitedQuotaUser, SavedCandidate, SearchHistory
from app.utils.unlimited_quota_production import (
    production_unlimited_quota_manager as unlimited_quota_manager, 
    is_unlimited_quota_user,
    get_unlimited_quota_info
)
from app.utils.admin_auth import require_admin_auth
import logging
from datetime import datetime, timedelta
from sqlalchemy import func, desc

logger = get_logger("admin")

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/unlimited-quota/users', methods=['GET'])
@require_admin_auth
def list_unlimited_quota_users():
    """List all unlimited quota users (admin only)"""
    try:
        users = unlimited_quota_manager.list_unlimited_users()
        stats = unlimited_quota_manager.get_stats()
        
        return jsonify({
            'users': users,
            'stats': stats,
            'message': 'Unlimited quota users retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing unlimited quota users: {e}")
        return jsonify({'error': 'Failed to retrieve unlimited quota users'}), 500

@admin_bp.route('/unlimited-quota/users', methods=['POST'])
@require_admin_auth
def add_unlimited_quota_user():
    """Add a new unlimited quota user (admin only)"""
    try:
        data = request.get_json()
        email = data.get('email')
        reason = data.get('reason', 'Admin granted')
        added_by = data.get('added_by', 'admin')
        quota_limit = data.get('quota_limit', -1)
        daily_limit = data.get('daily_limit', -1)
        monthly_limit = data.get('monthly_limit', -1)
        expires = data.get('expires')
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        # Check if user exists in database
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found in database'}), 404
        
        # Add unlimited quota
        success = unlimited_quota_manager.add_unlimited_user(
            email=email,
            reason=reason,
            added_by=added_by,
            quota_limit=quota_limit,
            daily_limit=daily_limit,
            monthly_limit=monthly_limit,
            expires=expires
        )
        
        if success:
            return jsonify({
                'message': f'Unlimited quota added for {email}',
                'user': get_unlimited_quota_info(email)
            }), 201
        else:
            return jsonify({'error': 'Failed to add unlimited quota'}), 500
            
    except Exception as e:
        logger.error(f"Error adding unlimited quota user: {e}")
        return jsonify({'error': 'Failed to add unlimited quota user'}), 500

@admin_bp.route('/unlimited-quota/users/<email>', methods=['PUT'])
@require_admin_auth
def update_unlimited_quota_user(email):
    """Update unlimited quota user settings (admin only)"""
    try:
        data = request.get_json()
        updates = {}
        
        # Only allow updating specific fields
        allowed_fields = ['quota_limit', 'daily_limit', 'monthly_limit', 'reason', 'expires', 'active']
        for field in allowed_fields:
            if field in data:
                updates[field] = data[field]
        
        if not updates:
            return jsonify({'error': 'No valid fields to update'}), 400
        
        updated_by = data.get('updated_by', 'admin')
        
        # Update user quota
        success = unlimited_quota_manager.update_user_quota(
            email=email,
            updates=updates,
            updated_by=updated_by
        )
        
        if success:
            return jsonify({
                'message': f'Unlimited quota updated for {email}',
                'user': get_unlimited_quota_info(email)
            }), 200
        else:
            return jsonify({'error': 'Failed to update unlimited quota'}), 500
            
    except Exception as e:
        logger.error(f"Error updating unlimited quota user: {e}")
        return jsonify({'error': 'Failed to update unlimited quota user'}), 500

@admin_bp.route('/unlimited-quota/users/<email>', methods=['DELETE'])
@require_admin_auth
def remove_unlimited_quota_user(email):
    """Remove unlimited quota for a user (admin only)"""
    try:
        removed_by = request.json.get('removed_by', 'admin') if request.json else 'admin'
        
        # Remove unlimited quota
        success = unlimited_quota_manager.remove_unlimited_user(
            email=email,
            removed_by=removed_by
        )
        
        if success:
            return jsonify({
                'message': f'Unlimited quota removed for {email}'
            }), 200
        else:
            return jsonify({'error': 'Failed to remove unlimited quota'}), 500
            
    except Exception as e:
        logger.error(f"Error removing unlimited quota user: {e}")
        return jsonify({'error': 'Failed to remove unlimited quota user'}), 500

@admin_bp.route('/unlimited-quota/users/<email>', methods=['GET'])
@require_admin_auth
def get_unlimited_quota_user(email):
    """Get unlimited quota info for a specific user (admin only)"""
    try:
        user_info = get_unlimited_quota_info(email)
        
        if not user_info:
            return jsonify({'error': 'User not found or no unlimited quota'}), 404
        
        return jsonify({
            'user': user_info,
            'message': 'Unlimited quota user info retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting unlimited quota user info: {e}")
        return jsonify({'error': 'Failed to get unlimited quota user info'}), 500

@admin_bp.route('/unlimited-quota/stats', methods=['GET'])
@require_admin_auth
def get_unlimited_quota_stats():
    """Get statistics about unlimited quota users (admin only)"""
    try:
        stats = unlimited_quota_manager.get_stats()
        
        return jsonify({
            'stats': stats,
            'message': 'Unlimited quota stats retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting unlimited quota stats: {e}")
        return jsonify({'error': 'Failed to get unlimited quota stats'}), 500

@admin_bp.route('/candidates', methods=['GET'])
@require_admin_auth
def list_candidates():
    """List all candidates with detailed information (admin only)"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        role_filter = request.args.get('role', None)
        search = request.args.get('search', None)
        
        # Base query for all user roles (admin can see all users)
        query = User.query.filter(User.role.in_(['job_seeker', 'employee', 'recruiter', 'employer', 'admin', 'owner', 'subuser']))
        
        # Apply role filter
        if role_filter:
            query = query.filter(User.role == role_filter)
        
        # Apply search filter
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (User.email.ilike(search_term)) |
                (User.company_name.ilike(search_term))
            )
        
        # Get paginated results
        pagination = query.order_by(desc(User.created_at)).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        candidates = []
        for user in pagination.items:
            # Get candidate profile
            profile = CandidateProfile.query.filter_by(user_id=user.id).first()
            
            # Get related data counts
            skills_count = CandidateSkill.query.filter_by(profile_id=profile.id).count() if profile else 0
            education_count = CandidateEducation.query.filter_by(profile_id=profile.id).count() if profile else 0
            experience_count = CandidateExperience.query.filter_by(profile_id=profile.id).count() if profile else 0
            certifications_count = CandidateCertification.query.filter_by(profile_id=profile.id).count() if profile else 0
            projects_count = CandidateProject.query.filter_by(profile_id=profile.id).count() if profile else 0
            
            # Get social links
            social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
            
            # Get trial info
            trial = UserTrial.query.filter_by(user_id=user.id).first()
            
            # Get search logs count
            search_logs_count = JDSearchLog.query.filter_by(user_id=user.id).count()
            
            # Get KPIs
            kpis = UserKPIs.query.filter_by(user_id=user.id).first()
            
            # Get other related data counts
            skill_gaps_count = UserSkillGap.query.filter_by(user_id=user.id).count()
            learning_paths_count = UserLearningPath.query.filter_by(user_id=user.id).count()
            achievements_count = UserAchievement.query.filter_by(user_id=user.id).count()
            goals_count = UserGoal.query.filter_by(user_id=user.id).count()
            schedules_count = UserSchedule.query.filter_by(user_id=user.id).count()
            
            candidate_data = {
                'id': user.id,
                'email': user.email,
                'role': user.role,
                'user_type': user.user_type,
                'company_name': user.company_name,
                'linkedin_id': user.linkedin_id,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'updated_at': user.updated_at.isoformat() if user.updated_at else None,
                'tenant_id': user.tenant_id,
                'profile': {
                    'id': profile.id if profile else None,
                    'full_name': profile.full_name if profile else None,
                    'phone': profile.phone if profile else None,
                    'location': profile.location if profile else None,
                    'summary': profile.summary if profile else None,
                    'skills_count': skills_count,
                    'education_count': education_count,
                    'experience_count': experience_count,
                    'certifications_count': certifications_count,
                    'projects_count': projects_count
                } if profile else None,
                'social_links': {
                    'linkedin': social_links.linkedin if social_links else None,
                    'facebook': social_links.facebook if social_links else None,
                    'x': social_links.x if social_links else None,
                    'github': social_links.github if social_links else None
                } if social_links else None,
                'trial': {
                    'is_active': trial.is_active if trial else None,
                    'trial_start_date': trial.trial_start_date.isoformat() if trial and trial.trial_start_date else None,
                    'trial_end_date': trial.trial_end_date.isoformat() if trial and trial.trial_end_date else None,
                    'searches_used_today': trial.searches_used_today if trial else None,
                    'last_search_date': trial.last_search_date.isoformat() if trial and trial.last_search_date else None
                } if trial else None,
                'stats': {
                    'search_logs_count': search_logs_count,
                    'skill_gaps_count': skill_gaps_count,
                    'learning_paths_count': learning_paths_count,
                    'achievements_count': achievements_count,
                    'goals_count': goals_count,
                    'schedules_count': schedules_count
                },
                'kpis': kpis.to_dict() if kpis else None
            }
            
            candidates.append(candidate_data)
        
        return jsonify({
            'candidates': candidates,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            },
            'message': 'Candidates retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing candidates: {e}")
        return jsonify({'error': 'Failed to retrieve candidates'}), 500

@admin_bp.route('/candidates/<int:candidate_id>', methods=['GET'])
@require_admin_auth
def get_candidate_details(candidate_id):
    """Get detailed information about a specific candidate (admin only)"""
    try:
        user = User.query.get(candidate_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Allow viewing details of all user roles (admin can view all users)
        # No role restriction needed
        
        # Get all related data
        profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
        trial = UserTrial.query.filter_by(user_id=user.id).first()
        ceipal_integration = CeipalIntegration.query.filter_by(user_id=user.id).first()
        user_image = UserImage.query.filter_by(user_id=user.id).first()
        kpis = UserKPIs.query.filter_by(user_id=user.id).first()
        
        # Get all related records
        skills = CandidateSkill.query.filter_by(profile_id=profile.id).all() if profile else []
        education = CandidateEducation.query.filter_by(profile_id=profile.id).all() if profile else []
        experience = CandidateExperience.query.filter_by(profile_id=profile.id).all() if profile else []
        certifications = CandidateCertification.query.filter_by(profile_id=profile.id).all() if profile else []
        projects = CandidateProject.query.filter_by(profile_id=profile.id).all() if profile else []
        search_logs = JDSearchLog.query.filter_by(user_id=user.id).all()
        skill_gaps = UserSkillGap.query.filter_by(user_id=user.id).all()
        learning_paths = UserLearningPath.query.filter_by(user_id=user.id).all()
        achievements = UserAchievement.query.filter_by(user_id=user.id).all()
        goals = UserGoal.query.filter_by(user_id=user.id).all()
        schedules = UserSchedule.query.filter_by(user_id=user.id).all()
        
        candidate_details = {
            'id': user.id,
            'email': user.email,
            'role': user.role,
            'user_type': user.user_type,
            'company_name': user.company_name,
            'linkedin_id': user.linkedin_id,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'updated_at': user.updated_at.isoformat() if user.updated_at else None,
            'tenant_id': user.tenant_id,
            'profile': profile.to_dict() if profile else None,
            'social_links': social_links.to_dict() if social_links else None,
            'trial': trial.to_dict() if trial else None,
            'ceipal_integration': ceipal_integration.to_dict() if ceipal_integration else None,
            'user_image': {
                'has_image': bool(user_image),
                'image_type': user_image.image_type if user_image else None,
                'file_name': user_image.file_name if user_image else None,
                'file_size': user_image.file_size if user_image else None,
                'uploaded_at': user_image.uploaded_at.isoformat() if user_image and user_image.uploaded_at else None
            },
            'kpis': kpis.to_dict() if kpis else None,
            'skills': [skill.to_dict() for skill in skills],
            'education': [edu.to_dict() for edu in education],
            'experience': [exp.to_dict() for exp in experience],
            'certifications': [cert.to_dict() for cert in certifications],
            'projects': [proj.to_dict() for proj in projects],
            'search_logs': [log.to_dict() for log in search_logs],
            'skill_gaps': [gap.to_dict() for gap in skill_gaps],
            'learning_paths': [path.to_dict() for path in learning_paths],
            'achievements': [achievement.to_dict() for achievement in achievements],
            'goals': [goal.to_dict() for goal in goals],
            'schedules': [schedule.to_dict() for schedule in schedules]
        }
        
        return jsonify({
            'candidate': candidate_details,
            'message': 'Candidate details retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting candidate details: {e}")
        return jsonify({'error': 'Failed to retrieve candidate details'}), 500

@admin_bp.route('/candidates/<int:candidate_id>', methods=['DELETE'])
@require_admin_auth
def delete_candidate(candidate_id):
    """Delete a candidate and all related data (admin only)"""
    try:
        # Get current user from the authentication decorator
        from flask import request
        from app.utils import get_current_user_flexible
        
        current_user = get_current_user_flexible()
        if not current_user:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_email = current_user.get('email')
        if not user_email:
            return jsonify({'error': 'Valid user email required'}), 401
        
        # Check if the requesting user is vinit@adeptaipro.com
        if user_email != 'vinit@adeptaipro.com':
            logger.warning(f"Unauthorized deletion attempt by {user_email}")
            return jsonify({
                'error': 'Access denied. Only vinit@adeptaipro.com can delete candidates.',
                'success': False
            }), 403
        
        user = User.query.get(candidate_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Allow deletion of all user roles (admin can manage all users)
        # No role restriction needed
        
        logger.info(f"Starting deletion process for user: {user.email} (ID: {user.id}, Role: {user.role})")
        
        deleted_data = {
            'user_id': user.id,
            'email': user.email,
            'role': user.role,
            'tenant_id': user.tenant_id,
            'deleted_at': datetime.utcnow().isoformat()
        }
        
        # Note: SQLAlchemy sessions have implicit transactions, no need to call begin()
        
        # 1. Delete candidate profile and all related data
        candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        if candidate_profile:
            logger.info(f"Deleting candidate profile (ID: {candidate_profile.id})")
            
            # Count related records before deletion
            skills_count = CandidateSkill.query.filter_by(profile_id=candidate_profile.id).count()
            education_count = CandidateEducation.query.filter_by(profile_id=candidate_profile.id).count()
            experience_count = CandidateExperience.query.filter_by(profile_id=candidate_profile.id).count()
            certifications_count = CandidateCertification.query.filter_by(profile_id=candidate_profile.id).count()
            projects_count = CandidateProject.query.filter_by(profile_id=candidate_profile.id).count()
            
            deleted_data['candidate_profile'] = {
                'profile_id': candidate_profile.id,
                'full_name': candidate_profile.full_name,
                'skills_deleted': skills_count,
                'education_deleted': education_count,
                'experience_deleted': experience_count,
                'certifications_deleted': certifications_count,
                'projects_deleted': projects_count
            }
            
            # Delete candidate profile (cascade will handle related records)
            db.session.delete(candidate_profile)
            logger.info(f"Deleted candidate profile and {skills_count} skills, {education_count} education records, {experience_count} experience records, {certifications_count} certifications, {projects_count} projects")
        
        # 2. Delete user social links
        social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
        if social_links:
            db.session.delete(social_links)
            deleted_data['social_links'] = True
            logger.info("Deleted user social links")
        
        # 3. Delete user trial
        user_trial = UserTrial.query.filter_by(user_id=user.id).first()
        if user_trial:
            db.session.delete(user_trial)
            deleted_data['user_trial'] = True
            logger.info("Deleted user trial")
        
        # 4. Delete Ceipal integration
        ceipal_integration = CeipalIntegration.query.filter_by(user_id=user.id).first()
        if ceipal_integration:
            db.session.delete(ceipal_integration)
            deleted_data['ceipal_integration'] = True
            logger.info("Deleted Ceipal integration")
        
        # 5. Delete job search logs
        search_logs = JDSearchLog.query.filter_by(user_id=user.id).all()
        if search_logs:
            for log in search_logs:
                db.session.delete(log)
            deleted_data['search_logs_deleted'] = len(search_logs)
            logger.info(f"Deleted {len(search_logs)} search logs")
        
        # 6. Delete user image
        user_image = UserImage.query.filter_by(user_id=user.id).first()
        if user_image:
            db.session.delete(user_image)
            deleted_data['user_image'] = True
            logger.info("Deleted user image")
        
        # 7. Delete user KPIs
        user_kpis = UserKPIs.query.filter_by(user_id=user.id).all()
        if user_kpis:
            for kpi in user_kpis:
                db.session.delete(kpi)
            deleted_data['user_kpis_deleted'] = len(user_kpis)
            logger.info(f"Deleted {len(user_kpis)} user KPIs")
        
        # 8. Delete user skill gaps
        skill_gaps = UserSkillGap.query.filter_by(user_id=user.id).all()
        if skill_gaps:
            for gap in skill_gaps:
                db.session.delete(gap)
            deleted_data['skill_gaps_deleted'] = len(skill_gaps)
            logger.info(f"Deleted {len(skill_gaps)} skill gaps")
        
        # 9. Delete user learning paths
        learning_paths = UserLearningPath.query.filter_by(user_id=user.id).all()
        if learning_paths:
            for path in learning_paths:
                db.session.delete(path)
            deleted_data['learning_paths_deleted'] = len(learning_paths)
            logger.info(f"Deleted {len(learning_paths)} learning paths")
        
        # 10. Delete user achievements
        achievements = UserAchievement.query.filter_by(user_id=user.id).all()
        if achievements:
            for achievement in achievements:
                db.session.delete(achievement)
            deleted_data['achievements_deleted'] = len(achievements)
            logger.info(f"Deleted {len(achievements)} achievements")
        
        # 11. Delete user goals
        goals = UserGoal.query.filter_by(user_id=user.id).all()
        if goals:
            for goal in goals:
                db.session.delete(goal)
            deleted_data['goals_deleted'] = len(goals)
            logger.info(f"Deleted {len(goals)} goals")
        
        # 12. Delete user schedules
        schedules = UserSchedule.query.filter_by(user_id=user.id).all()
        if schedules:
            for schedule in schedules:
                db.session.delete(schedule)
            deleted_data['schedules_deleted'] = len(schedules)
            logger.info(f"Deleted {len(schedules)} schedules")
        
        # 13. Delete ALL possible tables with foreign key references (comprehensive cleanup)
        # Comprehensive list of all tables that might reference users table
        all_possible_tables = [
            # Tables we know exist from errors
            'ai_recommendations', 'career_insights', 'learning_pathways', 
            'progress_achievements', 'password_reset_otps', 'skill_gaps',
            'user_stats', 'weekly_goals', 'monthly_goals', 'daily_goals',
            
            # Common user-related tables
            'user_notes', 'user_feedback', 'user_preferences', 'user_settings',
            'user_notifications', 'user_activities', 'user_sessions',
            'user_logs', 'user_analytics', 'user_metrics', 'user_reports',
            
            # Learning and course related
            'user_courses', 'user_lessons', 'user_assignments', 'user_quizzes',
            'user_certificates', 'user_badges', 'user_progress',
            
            # Goal and achievement related
            'user_milestones', 'user_challenges', 'user_rewards',
            'user_leaderboard', 'user_rankings', 'user_scores',
            
            # Communication and social
            'user_messages', 'user_comments', 'user_reviews', 'user_ratings',
            'user_follows', 'user_connections', 'user_invitations',
            
            # System and admin
            'user_permissions', 'user_roles', 'user_access_logs',
            'user_audit_trail', 'user_activity_logs',
            
            # Additional tables that might exist
            'user_workflows', 'user_templates', 'user_forms', 'user_submissions',
            'user_uploads', 'user_downloads', 'user_shares', 'user_bookmarks',
            'user_favorites', 'user_watchlist', 'user_history', 'user_timeline'
        ]
        
        # Try to delete from all possible tables
        for table in all_possible_tables:
            try:
                # First check if table exists and has user_id column
                count_result = db.session.execute(
                    db.text(f"SELECT COUNT(*) FROM {table} WHERE user_id = :user_id"),
                    {"user_id": user.id}
                ).scalar()
                
                if count_result > 0:
                    db.session.execute(
                        db.text(f"DELETE FROM {table} WHERE user_id = :user_id"),
                        {"user_id": user.id}
                    )
                    deleted_data[f'{table}_deleted'] = count_result
                    logger.info(f"Deleted {count_result} records from {table}")
            except Exception as e:
                # Table might not exist or have different structure, continue silently
                pass
        
        # 14. Delete the user
        db.session.delete(user)
        logger.info(f"Deleted user: {user.email}")
        
        # 15. Check if tenant has other users, if not, delete tenant
        tenant = Tenant.query.get(user.tenant_id)
        if tenant:
            remaining_users = User.query.filter_by(tenant_id=tenant.id).count()
            if remaining_users == 0:
                # Delete tenant alerts
                tenant_alerts = TenantAlert.query.filter_by(tenant_id=tenant.id).all()
                for alert in tenant_alerts:
                    db.session.delete(alert)
                
                # Delete tenant
                db.session.delete(tenant)
                deleted_data['tenant_deleted'] = True
                deleted_data['tenant_alerts_deleted'] = len(tenant_alerts)
                logger.info(f"Deleted tenant (ID: {tenant.id}) and {len(tenant_alerts)} tenant alerts")
            else:
                deleted_data['tenant_preserved'] = True
                deleted_data['remaining_users_in_tenant'] = remaining_users
                logger.info(f"Tenant preserved - {remaining_users} users still exist")
        
        # Commit the transaction
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Successfully deleted candidate: {user.email}',
            'deleted_data': deleted_data
        }), 200
        
    except Exception as e:
        # Rollback on error
        db.session.rollback()
        logger.error(f"Error deleting candidate {candidate_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error deleting candidate: {str(e)}'
        }), 500

@admin_bp.route('/candidates/stats', methods=['GET'])
@require_admin_auth
def get_candidates_stats():
    """Get statistics about candidates (admin only)"""
    try:
        # Total candidates
        total_candidates = User.query.filter(User.role.in_(['job_seeker', 'employee'])).count()
        
        # Candidates by role
        role_stats = db.session.query(
            User.role, 
            func.count(User.id).label('count')
        ).filter(User.role.in_(['job_seeker', 'employee'])).group_by(User.role).all()
        
        # Candidates with profiles
        candidates_with_profiles = db.session.query(User.id).join(CandidateProfile).filter(
            User.role.in_(['job_seeker', 'employee'])
        ).count()
        
        # Candidates with social links
        candidates_with_social = db.session.query(User.id).join(UserSocialLinks).filter(
            User.role.in_(['job_seeker', 'employee'])
        ).count()
        
        # Candidates with trials
        candidates_with_trials = db.session.query(User.id).join(UserTrial).filter(
            User.role.in_(['job_seeker', 'employee'])
        ).count()
        
        # Recent candidates (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_candidates = User.query.filter(
            User.role.in_(['job_seeker', 'employee']),
            User.created_at >= thirty_days_ago
        ).count()
        
        # Average profile completeness
        profiles = CandidateProfile.query.join(User).filter(
            User.role.in_(['job_seeker', 'employee'])
        ).all()
        
        completeness_scores = []
        for profile in profiles:
            score = 0
            if profile.full_name: score += 20
            if profile.phone: score += 10
            if profile.location: score += 10
            if profile.summary: score += 20
            if profile.skills: score += 20
            if profile.education: score += 10
            if profile.experience: score += 10
            completeness_scores.append(score)
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        stats = {
            'total_candidates': total_candidates,
            'role_distribution': {role: count for role, count in role_stats},
            'candidates_with_profiles': candidates_with_profiles,
            'candidates_with_social': candidates_with_social,
            'candidates_with_trials': candidates_with_trials,
            'recent_candidates': recent_candidates,
            'average_profile_completeness': round(avg_completeness, 2)
        }
        
        return jsonify({
            'stats': stats,
            'message': 'Candidate statistics retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting candidate stats: {e}")
        return jsonify({'error': 'Failed to retrieve candidate statistics'}), 500

@admin_bp.route('/all-emails', methods=['GET'])
@require_admin_auth
def get_all_emails():
    """Get all emails from all tables (admin only)"""
    try:
        all_emails = []
        
        # 1. Get emails from users table
        users = User.query.all()
        for user in users:
            all_emails.append({
                'email': user.email,
                'source': 'users',
                'user_id': user.id,
                'role': user.role,
                'user_type': user.user_type,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'is_candidate': user.role in ['job_seeker', 'employee']
            })
        
        # 2. Get emails from ceipal_integrations table
        ceipal_integrations = CeipalIntegration.query.all()
        for integration in ceipal_integrations:
            all_emails.append({
                'email': integration.ceipal_email,
                'source': 'ceipal_integrations',
                'user_id': integration.user_id,
                'role': 'integration',
                'user_type': 'ceipal',
                'created_at': integration.created_at.isoformat() if integration.created_at else None,
                'is_candidate': False
            })
        
        # 3. Get emails from unlimited_quota_users table
        unlimited_users = UnlimitedQuotaUser.query.all()
        for quota_user in unlimited_users:
            all_emails.append({
                'email': quota_user.email,
                'source': 'unlimited_quota_users',
                'user_id': None,  # This table doesn't have user_id
                'role': 'unlimited_quota',
                'user_type': 'quota_exempt',
                'created_at': quota_user.added_date.isoformat() if quota_user.added_date else None,
                'is_candidate': False,
                'quota_info': {
                    'reason': quota_user.reason,
                    'quota_limit': quota_user.quota_limit,
                    'daily_limit': quota_user.daily_limit,
                    'monthly_limit': quota_user.monthly_limit,
                    'active': quota_user.active,
                    'expires': quota_user.expires.isoformat() if quota_user.expires else None
                }
            })
        
        # Sort by email for easier reading
        all_emails.sort(key=lambda x: x['email'].lower())
        
        # Get statistics
        stats = {
            'total_emails': len(all_emails),
            'users_table': len([e for e in all_emails if e['source'] == 'users']),
            'ceipal_integrations': len([e for e in all_emails if e['source'] == 'ceipal_integrations']),
            'unlimited_quota': len([e for e in all_emails if e['source'] == 'unlimited_quota_users']),
            'candidates': len([e for e in all_emails if e['is_candidate']]),
            'non_candidates': len([e for e in all_emails if not e['is_candidate']])
        }
        
        return jsonify({
            'emails': all_emails,
            'stats': stats,
            'message': 'All emails retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting all emails: {e}")
        return jsonify({'error': 'Failed to retrieve all emails'}), 500

@admin_bp.route('/emails/search', methods=['GET'])
@require_admin_auth
def search_emails():
    """Search for specific emails across all tables (admin only)"""
    try:
        search_term = request.args.get('q', '').strip()
        
        if not search_term:
            return jsonify({'error': 'Search term is required'}), 400
        
        results = []
        
        # Search in users table
        users = User.query.filter(User.email.ilike(f'%{search_term}%')).all()
        for user in users:
            results.append({
                'email': user.email,
                'source': 'users',
                'user_id': user.id,
                'role': user.role,
                'user_type': user.user_type,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'is_candidate': user.role in ['job_seeker', 'employee']
            })
        
        # Search in ceipal_integrations table
        ceipal_integrations = CeipalIntegration.query.filter(
            CeipalIntegration.ceipal_email.ilike(f'%{search_term}%')
        ).all()
        for integration in ceipal_integrations:
            results.append({
                'email': integration.ceipal_email,
                'source': 'ceipal_integrations',
                'user_id': integration.user_id,
                'role': 'integration',
                'user_type': 'ceipal',
                'created_at': integration.created_at.isoformat() if integration.created_at else None,
                'is_candidate': False
            })
        
        # Search in unlimited_quota_users table
        unlimited_users = UnlimitedQuotaUser.query.filter(
            UnlimitedQuotaUser.email.ilike(f'%{search_term}%')
        ).all()
        for quota_user in unlimited_users:
            results.append({
                'email': quota_user.email,
                'source': 'unlimited_quota_users',
                'user_id': None,
                'role': 'unlimited_quota',
                'user_type': 'quota_exempt',
                'created_at': quota_user.added_date.isoformat() if quota_user.added_date else None,
                'is_candidate': False,
                'quota_info': {
                    'reason': quota_user.reason,
                    'quota_limit': quota_user.quota_limit,
                    'daily_limit': quota_user.daily_limit,
                    'monthly_limit': quota_user.monthly_limit,
                    'active': quota_user.active,
                    'expires': quota_user.expires.isoformat() if quota_user.expires else None
                }
            })
        
        # Sort by email
        results.sort(key=lambda x: x['email'].lower())
        
        return jsonify({
            'results': results,
            'count': len(results),
            'search_term': search_term,
            'message': f'Found {len(results)} emails matching "{search_term}"'
        }), 200
        
    except Exception as e:
        logger.error(f"Error searching emails: {e}")
        return jsonify({'error': 'Failed to search emails'}), 500
