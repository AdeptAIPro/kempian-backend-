# from flask import Blueprint, jsonify, request
# from app.simple_logger import get_logger
# from app.models import User, db, CandidateProfile, UserSocialLinks, UserTrial, CeipalIntegration, JDSearchLog, UserImage, UserKPIs, UserSkillGap, UserLearningPath, UserAchievement, UserGoal, UserSchedule, Tenant, TenantAlert, CandidateSkill, CandidateEducation, CandidateExperience, CandidateCertification, CandidateProject, UnlimitedQuotaUser, SavedCandidate, SearchHistory, Job, JobApplication, AdminActivityLog, AdminSession, OnboardingSubmission
# from app.utils.unlimited_quota_production import (
#     production_unlimited_quota_manager as unlimited_quota_manager, 
#     is_unlimited_quota_user,
#     get_unlimited_quota_info
# )
# from app.utils.admin_auth import require_admin_auth
# from app.utils.admin_activity_decorator import log_admin_activity
# import logging
# from datetime import datetime, timedelta
# from sqlalchemy import func, desc

# logger = get_logger("admin")

# admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# @admin_bp.route('/unlimited-quota/users', methods=['GET'])
# @require_admin_auth
# @log_admin_activity("List Unlimited Quota Users")
# def list_unlimited_quota_users():
#     """List all unlimited quota users (admin only)"""
#     try:
#         users = unlimited_quota_manager.list_unlimited_users()
#         stats = unlimited_quota_manager.get_stats()
        
#         return jsonify({
#             'users': users,
#             'stats': stats,
#             'message': 'Unlimited quota users retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error listing unlimited quota users: {e}")
#         return jsonify({'error': 'Failed to retrieve unlimited quota users'}), 500

# @admin_bp.route('/unlimited-quota/users', methods=['POST'])
# @require_admin_auth
# @log_admin_activity("Add Unlimited Quota User", include_request_data=True)
# def add_unlimited_quota_user():
#     """Add a new unlimited quota user (admin only)"""
#     try:
#         data = request.get_json()
#         email = data.get('email')
#         reason = data.get('reason', 'Admin granted')
#         added_by = data.get('added_by', 'admin')
#         quota_limit = data.get('quota_limit', -1)
#         daily_limit = data.get('daily_limit', -1)
#         monthly_limit = data.get('monthly_limit', -1)
#         expires = data.get('expires')
        
#         if not email:
#             return jsonify({'error': 'Email is required'}), 400
        
#         # Check if user exists in database
#         user = User.query.filter_by(email=email).first()
#         if not user:
#             return jsonify({'error': 'User not found in database'}), 404
        
#         # Add unlimited quota
#         success = unlimited_quota_manager.add_unlimited_user(
#             email=email,
#             reason=reason,
#             added_by=added_by,
#             quota_limit=quota_limit,
#             daily_limit=daily_limit,
#             monthly_limit=monthly_limit,
#             expires=expires
#         )
        
#         if success:
#             return jsonify({
#                 'message': f'Unlimited quota added for {email}',
#                 'user': get_unlimited_quota_info(email)
#             }), 201
#         else:
#             return jsonify({'error': 'Failed to add unlimited quota'}), 500
            
#     except Exception as e:
#         logger.error(f"Error adding unlimited quota user: {e}")
#         return jsonify({'error': 'Failed to add unlimited quota user'}), 500

# @admin_bp.route('/unlimited-quota/users/<email>', methods=['PUT'])
# @require_admin_auth
# def update_unlimited_quota_user(email):
#     """Update unlimited quota user settings (admin only)"""
#     try:
#         data = request.get_json()
#         updates = {}
        
#         # Only allow updating specific fields
#         allowed_fields = ['quota_limit', 'daily_limit', 'monthly_limit', 'reason', 'expires', 'active']
#         for field in allowed_fields:
#             if field in data:
#                 updates[field] = data[field]
        
#         if not updates:
#             return jsonify({'error': 'No valid fields to update'}), 400
        
#         updated_by = data.get('updated_by', 'admin')
        
#         # Update user quota
#         success = unlimited_quota_manager.update_user_quota(
#             email=email,
#             updates=updates,
#             updated_by=updated_by
#         )
        
#         if success:
#             return jsonify({
#                 'message': f'Unlimited quota updated for {email}',
#                 'user': get_unlimited_quota_info(email)
#             }), 200
#         else:
#             return jsonify({'error': 'Failed to update unlimited quota'}), 500
            
#     except Exception as e:
#         logger.error(f"Error updating unlimited quota user: {e}")
#         return jsonify({'error': 'Failed to update unlimited quota user'}), 500

# @admin_bp.route('/unlimited-quota/users/<email>', methods=['DELETE'])
# @require_admin_auth
# def remove_unlimited_quota_user(email):
#     """Remove unlimited quota for a user (admin only)"""
#     try:
#         removed_by = request.json.get('removed_by', 'admin') if request.json else 'admin'
        
#         # Remove unlimited quota
#         success = unlimited_quota_manager.remove_unlimited_user(
#             email=email,
#             removed_by=removed_by
#         )
        
#         if success:
#             return jsonify({
#                 'message': f'Unlimited quota removed for {email}'
#             }), 200
#         else:
#             return jsonify({'error': 'Failed to remove unlimited quota'}), 500
            
#     except Exception as e:
#         logger.error(f"Error removing unlimited quota user: {e}")
#         return jsonify({'error': 'Failed to remove unlimited quota user'}), 500

# @admin_bp.route('/unlimited-quota/users/<email>', methods=['GET'])
# @require_admin_auth
# def get_unlimited_quota_user(email):
#     """Get unlimited quota info for a specific user (admin only)"""
#     try:
#         user_info = get_unlimited_quota_info(email)
        
#         if not user_info:
#             return jsonify({'error': 'User not found or no unlimited quota'}), 404
        
#         return jsonify({
#             'user': user_info,
#             'message': 'Unlimited quota user info retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting unlimited quota user info: {e}")
#         return jsonify({'error': 'Failed to get unlimited quota user info'}), 500

# @admin_bp.route('/unlimited-quota/stats', methods=['GET'])
# @require_admin_auth
# def get_unlimited_quota_stats():
#     """Get statistics about unlimited quota users (admin only)"""
#     try:
#         stats = unlimited_quota_manager.get_stats()
        
#         return jsonify({
#             'stats': stats,
#             'message': 'Unlimited quota stats retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting unlimited quota stats: {e}")
#         return jsonify({'error': 'Failed to get unlimited quota stats'}), 500

# @admin_bp.route('/candidates', methods=['GET'])
# @require_admin_auth
# @log_admin_activity("List All Candidates")
# def list_candidates():
#     """List all candidates with detailed information (admin only)"""
#     try:
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 50, type=int)
#         role_filter = request.args.get('role', None)
#         search = request.args.get('search', None)
        
#         # Base query for all user roles (admin can see all users)
#         query = User.query.filter(User.role.in_(['job_seeker', 'employee', 'recruiter', 'employer', 'admin', 'owner', 'subuser']))
        
#         # Apply role filter
#         if role_filter:
#             query = query.filter(User.role == role_filter)
        
#         # Apply search filter
#         if search:
#             search_term = f"%{search}%"
#             query = query.filter(
#                 (User.email.ilike(search_term)) |
#                 (User.company_name.ilike(search_term))
#             )
        
#         # Get paginated results
#         pagination = query.order_by(desc(User.created_at)).paginate(
#             page=page, per_page=per_page, error_out=False
#         )
        
#         candidates = []
#         for user in pagination.items:
#             # Get candidate profile
#             profile = CandidateProfile.query.filter_by(user_id=user.id).first()
            
#             # Get related data counts
#             skills_count = CandidateSkill.query.filter_by(profile_id=profile.id).count() if profile else 0
#             education_count = CandidateEducation.query.filter_by(profile_id=profile.id).count() if profile else 0
#             experience_count = CandidateExperience.query.filter_by(profile_id=profile.id).count() if profile else 0
#             certifications_count = CandidateCertification.query.filter_by(profile_id=profile.id).count() if profile else 0
#             projects_count = CandidateProject.query.filter_by(profile_id=profile.id).count() if profile else 0
            
#             # Get social links
#             social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
            
#             # Get trial info
#             trial = UserTrial.query.filter_by(user_id=user.id).first()
            
#             # Get search logs count
#             search_logs_count = JDSearchLog.query.filter_by(user_id=user.id).count()
            
#             # Get KPIs
#             kpis = UserKPIs.query.filter_by(user_id=user.id).first()
            
#             # Get other related data counts
#             skill_gaps_count = UserSkillGap.query.filter_by(user_id=user.id).count()
#             learning_paths_count = UserLearningPath.query.filter_by(user_id=user.id).count()
#             achievements_count = UserAchievement.query.filter_by(user_id=user.id).count()
#             goals_count = UserGoal.query.filter_by(user_id=user.id).count()
#             schedules_count = UserSchedule.query.filter_by(user_id=user.id).count()
            
#             candidate_data = {
#                 'id': user.id,
#                 'email': user.email,
#                 'role': user.role,
#                 'user_type': user.user_type,
#                 'company_name': user.company_name,
#                 'linkedin_id': user.linkedin_id,
#                 'created_at': user.created_at.isoformat() if user.created_at else None,
#                 'updated_at': user.updated_at.isoformat() if user.updated_at else None,
#                 'tenant_id': user.tenant_id,
#                 'profile': {
#                     'id': profile.id if profile else None,
#                     'full_name': profile.full_name if profile else None,
#                     'phone': profile.phone if profile else None,
#                     'location': profile.location if profile else None,
#                     'summary': profile.summary if profile else None,
#                     'skills_count': skills_count,
#                     'education_count': education_count,
#                     'experience_count': experience_count,
#                     'certifications_count': certifications_count,
#                     'projects_count': projects_count
#                 } if profile else None,
#                 'social_links': {
#                     'linkedin': social_links.linkedin if social_links else None,
#                     'facebook': social_links.facebook if social_links else None,
#                     'x': social_links.x if social_links else None,
#                     'github': social_links.github if social_links else None
#                 } if social_links else None,
#                 'trial': {
#                     'is_active': trial.is_active if trial else None,
#                     'trial_start_date': trial.trial_start_date.isoformat() if trial and trial.trial_start_date else None,
#                     'trial_end_date': trial.trial_end_date.isoformat() if trial and trial.trial_end_date else None,
#                     'searches_used_today': trial.searches_used_today if trial else None,
#                     'last_search_date': trial.last_search_date.isoformat() if trial and trial.last_search_date else None
#                 } if trial else None,
#                 'stats': {
#                     'search_logs_count': search_logs_count,
#                     'skill_gaps_count': skill_gaps_count,
#                     'learning_paths_count': learning_paths_count,
#                     'achievements_count': achievements_count,
#                     'goals_count': goals_count,
#                     'schedules_count': schedules_count
#                 },
#                 'kpis': kpis.to_dict() if kpis else None
#             }
            
#             candidates.append(candidate_data)
        
#         return jsonify({
#             'candidates': candidates,
#             'pagination': {
#                 'page': page,
#                 'per_page': per_page,
#                 'total': pagination.total,
#                 'pages': pagination.pages,
#                 'has_next': pagination.has_next,
#                 'has_prev': pagination.has_prev
#             },
#             'message': 'Candidates retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error listing candidates: {e}")
#         return jsonify({'error': 'Failed to retrieve candidates'}), 500

# @admin_bp.route('/candidates/<int:candidate_id>', methods=['GET'])
# @require_admin_auth
# def get_candidate_details(candidate_id):
#     """Get detailed information about a specific candidate (admin only)"""
#     try:
#         user = User.query.get(candidate_id)
        
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
        
#         # Allow viewing details of all user roles (admin can view all users)
#         # No role restriction needed
        
#         # Get all related data
#         profile = CandidateProfile.query.filter_by(user_id=user.id).first()
#         social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
#         trial = UserTrial.query.filter_by(user_id=user.id).first()
#         ceipal_integration = CeipalIntegration.query.filter_by(user_id=user.id).first()
#         user_image = UserImage.query.filter_by(user_id=user.id).first()
#         kpis = UserKPIs.query.filter_by(user_id=user.id).first()
        
#         # Get all related records
#         skills = CandidateSkill.query.filter_by(profile_id=profile.id).all() if profile else []
#         education = CandidateEducation.query.filter_by(profile_id=profile.id).all() if profile else []
#         experience = CandidateExperience.query.filter_by(profile_id=profile.id).all() if profile else []
#         certifications = CandidateCertification.query.filter_by(profile_id=profile.id).all() if profile else []
#         projects = CandidateProject.query.filter_by(profile_id=profile.id).all() if profile else []
#         search_logs = JDSearchLog.query.filter_by(user_id=user.id).all()
#         skill_gaps = UserSkillGap.query.filter_by(user_id=user.id).all()
#         learning_paths = UserLearningPath.query.filter_by(user_id=user.id).all()
#         achievements = UserAchievement.query.filter_by(user_id=user.id).all()
#         goals = UserGoal.query.filter_by(user_id=user.id).all()
#         schedules = UserSchedule.query.filter_by(user_id=user.id).all()
        
#         candidate_details = {
#             'id': user.id,
#             'email': user.email,
#             'role': user.role,
#             'user_type': user.user_type,
#             'company_name': user.company_name,
#             'linkedin_id': user.linkedin_id,
#             'created_at': user.created_at.isoformat() if user.created_at else None,
#             'updated_at': user.updated_at.isoformat() if user.updated_at else None,
#             'tenant_id': user.tenant_id,
#             'profile': profile.to_dict() if profile else None,
#             'social_links': social_links.to_dict() if social_links else None,
#             'trial': trial.to_dict() if trial else None,
#             'ceipal_integration': ceipal_integration.to_dict() if ceipal_integration else None,
#             'user_image': {
#                 'has_image': bool(user_image),
#                 'image_type': user_image.image_type if user_image else None,
#                 'file_name': user_image.file_name if user_image else None,
#                 'file_size': user_image.file_size if user_image else None,
#                 'uploaded_at': user_image.uploaded_at.isoformat() if user_image and user_image.uploaded_at else None
#             },
#             'kpis': kpis.to_dict() if kpis else None,
#             'skills': [skill.to_dict() for skill in skills],
#             'education': [edu.to_dict() for edu in education],
#             'experience': [exp.to_dict() for exp in experience],
#             'certifications': [cert.to_dict() for cert in certifications],
#             'projects': [proj.to_dict() for proj in projects],
#             'search_logs': [log.to_dict() for log in search_logs],
#             'skill_gaps': [gap.to_dict() for gap in skill_gaps],
#             'learning_paths': [path.to_dict() for path in learning_paths],
#             'achievements': [achievement.to_dict() for achievement in achievements],
#             'goals': [goal.to_dict() for goal in goals],
#             'schedules': [schedule.to_dict() for schedule in schedules]
#         }
        
#         return jsonify({
#             'candidate': candidate_details,
#             'message': 'Candidate details retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting candidate details: {e}")
#         return jsonify({'error': 'Failed to retrieve candidate details'}), 500

# @admin_bp.route('/candidates/<int:candidate_id>', methods=['DELETE'])
# @require_admin_auth
# def delete_candidate(candidate_id):
#     """Delete a candidate and all related data (admin only)"""
#     try:
#         # Get current user from the authentication decorator
#         from flask import request
#         from app.utils import get_current_user_flexible
        
#         current_user = get_current_user_flexible()
#         if not current_user:
#             return jsonify({'error': 'Authentication required'}), 401
        
#         user_email = current_user.get('email')
#         if not user_email:
#             return jsonify({'error': 'Valid user email required'}), 401
        
#         # Check if the requesting user is authorized
#         if user_email not in ['vinit@adeptaipro.com', 'contact@kempian.ai']:
#             logger.warning(f"Unauthorized deletion attempt by {user_email}")
#             return jsonify({
#                 'error': 'Access denied. Only authorized admins can delete candidates.',
#                 'success': False
#             }), 403
        
#         user = User.query.get(candidate_id)
        
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
        
#         # Allow deletion of all user roles (admin can manage all users)
#         # No role restriction needed
        
#         logger.info(f"Starting deletion process for user: {user.email} (ID: {user.id}, Role: {user.role})")
        
#         deleted_data = {
#             'user_id': user.id,
#             'email': user.email,
#             'role': user.role,
#             'tenant_id': user.tenant_id,
#             'deleted_at': datetime.utcnow().isoformat()
#         }
        
#         # Note: SQLAlchemy sessions have implicit transactions, no need to call begin()
        
#         # 1. Delete candidate profile and all related data
#         candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
#         if candidate_profile:
#             logger.info(f"Deleting candidate profile (ID: {candidate_profile.id})")
            
#             # Count related records before deletion
#             skills_count = CandidateSkill.query.filter_by(profile_id=candidate_profile.id).count()
#             education_count = CandidateEducation.query.filter_by(profile_id=candidate_profile.id).count()
#             experience_count = CandidateExperience.query.filter_by(profile_id=candidate_profile.id).count()
#             certifications_count = CandidateCertification.query.filter_by(profile_id=candidate_profile.id).count()
#             projects_count = CandidateProject.query.filter_by(profile_id=candidate_profile.id).count()
            
#             deleted_data['candidate_profile'] = {
#                 'profile_id': candidate_profile.id,
#                 'full_name': candidate_profile.full_name,
#                 'skills_deleted': skills_count,
#                 'education_deleted': education_count,
#                 'experience_deleted': experience_count,
#                 'certifications_deleted': certifications_count,
#                 'projects_deleted': projects_count
#             }
            
#             # Delete candidate profile (cascade will handle related records)
#             db.session.delete(candidate_profile)
#             logger.info(f"Deleted candidate profile and {skills_count} skills, {education_count} education records, {experience_count} experience records, {certifications_count} certifications, {projects_count} projects")
        
#         # 2. Delete user social links
#         social_links = UserSocialLinks.query.filter_by(user_id=user.id).first()
#         if social_links:
#             db.session.delete(social_links)
#             deleted_data['social_links'] = True
#             logger.info("Deleted user social links")
        
#         # 3. Delete user trial
#         user_trial = UserTrial.query.filter_by(user_id=user.id).first()
#         if user_trial:
#             db.session.delete(user_trial)
#             deleted_data['user_trial'] = True
#             logger.info("Deleted user trial")
        
#         # 4. Delete Ceipal integration
#         ceipal_integration = CeipalIntegration.query.filter_by(user_id=user.id).first()
#         if ceipal_integration:
#             db.session.delete(ceipal_integration)
#             deleted_data['ceipal_integration'] = True
#             logger.info("Deleted Ceipal integration")
        
#         # 5. Delete job search logs
#         search_logs = JDSearchLog.query.filter_by(user_id=user.id).all()
#         if search_logs:
#             for log in search_logs:
#                 db.session.delete(log)
#             deleted_data['search_logs_deleted'] = len(search_logs)
#             logger.info(f"Deleted {len(search_logs)} search logs")
        
#         # 6. Delete user image
#         user_image = UserImage.query.filter_by(user_id=user.id).first()
#         if user_image:
#             db.session.delete(user_image)
#             deleted_data['user_image'] = True
#             logger.info("Deleted user image")
        
#         # 7. Delete user KPIs
#         user_kpis = UserKPIs.query.filter_by(user_id=user.id).all()
#         if user_kpis:
#             for kpi in user_kpis:
#                 db.session.delete(kpi)
#             deleted_data['user_kpis_deleted'] = len(user_kpis)
#             logger.info(f"Deleted {len(user_kpis)} user KPIs")
        
#         # 8. Delete user skill gaps
#         skill_gaps = UserSkillGap.query.filter_by(user_id=user.id).all()
#         if skill_gaps:
#             for gap in skill_gaps:
#                 db.session.delete(gap)
#             deleted_data['skill_gaps_deleted'] = len(skill_gaps)
#             logger.info(f"Deleted {len(skill_gaps)} skill gaps")
        
#         # 9. Delete user learning paths
#         learning_paths = UserLearningPath.query.filter_by(user_id=user.id).all()
#         if learning_paths:
#             for path in learning_paths:
#                 db.session.delete(path)
#             deleted_data['learning_paths_deleted'] = len(learning_paths)
#             logger.info(f"Deleted {len(learning_paths)} learning paths")
        
#         # 10. Delete user achievements
#         achievements = UserAchievement.query.filter_by(user_id=user.id).all()
#         if achievements:
#             for achievement in achievements:
#                 db.session.delete(achievement)
#             deleted_data['achievements_deleted'] = len(achievements)
#             logger.info(f"Deleted {len(achievements)} achievements")
        
#         # 11. Delete user goals
#         goals = UserGoal.query.filter_by(user_id=user.id).all()
#         if goals:
#             for goal in goals:
#                 db.session.delete(goal)
#             deleted_data['goals_deleted'] = len(goals)
#             logger.info(f"Deleted {len(goals)} goals")
        
#         # 12. Delete user schedules
#         schedules = UserSchedule.query.filter_by(user_id=user.id).all()
#         if schedules:
#             for schedule in schedules:
#                 db.session.delete(schedule)
#             deleted_data['schedules_deleted'] = len(schedules)
#             logger.info(f"Deleted {len(schedules)} schedules")
        
#         # 13. Delete job applications (this was causing the foreign key constraint error)
#         job_applications = JobApplication.query.filter_by(applicant_id=user.id).all()
#         if job_applications:
#             for application in job_applications:
#                 db.session.delete(application)
#             deleted_data['job_applications_deleted'] = len(job_applications)
#             logger.info(f"Deleted {len(job_applications)} job applications")
        
#         # 14. Delete ALL possible tables with foreign key references (comprehensive cleanup)
#         # Comprehensive list of all tables that might reference users table
#         all_possible_tables = [
#             # Tables we know exist from errors
#             'ai_recommendations', 'career_insights', 'learning_pathways', 
#             'progress_achievements', 'password_reset_otps', 'skill_gaps',
#             'user_stats', 'weekly_goals', 'monthly_goals', 'daily_goals',
            
#             # Common user-related tables
#             'user_notes', 'user_feedback', 'user_preferences', 'user_settings',
#             'user_notifications', 'user_activities', 'user_sessions',
#             'user_logs', 'user_analytics', 'user_metrics', 'user_reports',
            
#             # Learning and course related
#             'user_courses', 'user_lessons', 'user_assignments', 'user_quizzes',
#             'user_certificates', 'user_badges', 'user_progress',
            
#             # Goal and achievement related
#             'user_milestones', 'user_challenges', 'user_rewards',
#             'user_leaderboard', 'user_rankings', 'user_scores',
            
#             # Communication and social
#             'user_messages', 'user_comments', 'user_reviews', 'user_ratings',
#             'user_follows', 'user_connections', 'user_invitations',
            
#             # System and admin
#             'user_permissions', 'user_roles', 'user_access_logs',
#             'user_audit_trail', 'user_activity_logs',
            
#             # Additional tables that might exist
#             'user_workflows', 'user_templates', 'user_forms', 'user_submissions',
#             'user_uploads', 'user_downloads', 'user_shares', 'user_bookmarks',
#             'user_favorites', 'user_watchlist', 'user_history', 'user_timeline'
#         ]
        
#         # Try to delete from all possible tables
#         for table in all_possible_tables:
#             try:
#                 # First check if table exists and has user_id column
#                 count_result = db.session.execute(
#                     db.text(f"SELECT COUNT(*) FROM {table} WHERE user_id = :user_id"),
#                     {"user_id": user.id}
#                 ).scalar()
                
#                 if count_result > 0:
#                     db.session.execute(
#                         db.text(f"DELETE FROM {table} WHERE user_id = :user_id"),
#                         {"user_id": user.id}
#                     )
#                     deleted_data[f'{table}_deleted'] = count_result
#                     logger.info(f"Deleted {count_result} records from {table}")
#             except Exception as e:
#                 # Table might not exist or have different structure, continue silently
#                 pass
        
#         # 15. Delete the user
#         db.session.delete(user)
#         logger.info(f"Deleted user: {user.email}")
        
#         # 16. Check if tenant has other users, if not, delete tenant
#         tenant = Tenant.query.get(user.tenant_id)
#         if tenant:
#             remaining_users = User.query.filter_by(tenant_id=tenant.id).count()
#             if remaining_users == 0:
#                 # Delete tenant alerts FIRST (before deleting tenant)
#                 tenant_alerts = TenantAlert.query.filter_by(tenant_id=tenant.id).all()
#                 for alert in tenant_alerts:
#                     db.session.delete(alert)
#                 db.session.flush()  # Ensure tenant alerts are deleted before tenant
                
#                 # Delete tenant
#                 db.session.delete(tenant)
#                 deleted_data['tenant_deleted'] = True
#                 deleted_data['tenant_alerts_deleted'] = len(tenant_alerts)
#                 logger.info(f"Deleted tenant (ID: {tenant.id}) and {len(tenant_alerts)} tenant alerts")
#             else:
#                 deleted_data['tenant_preserved'] = True
#                 deleted_data['remaining_users_in_tenant'] = remaining_users
#                 logger.info(f"Tenant preserved - {remaining_users} users still exist")
        
#         # Commit the transaction
#         db.session.commit()
        
#         return jsonify({
#             'success': True,
#             'message': f'Successfully deleted candidate: {user.email}',
#             'deleted_data': deleted_data
#         }), 200
        
#     except Exception as e:
#         # Rollback on error
#         db.session.rollback()
#         logger.error(f"Error deleting candidate {candidate_id}: {str(e)}")
#         return jsonify({
#             'success': False,
#             'error': f'Error deleting candidate: {str(e)}'
#         }), 500

# @admin_bp.route('/candidates/stats', methods=['GET'])
# @require_admin_auth
# def get_candidates_stats():
#     """Get statistics about candidates (admin only)"""
#     try:
#         # Total candidates
#         total_candidates = User.query.filter(User.role.in_(['job_seeker', 'employee'])).count()
        
#         # Candidates by role
#         role_stats = db.session.query(
#             User.role, 
#             func.count(User.id).label('count')
#         ).filter(User.role.in_(['job_seeker', 'employee'])).group_by(User.role).all()
        
#         # Candidates with profiles
#         candidates_with_profiles = db.session.query(User.id).join(CandidateProfile).filter(
#             User.role.in_(['job_seeker', 'employee'])
#         ).count()
        
#         # Candidates with social links
#         candidates_with_social = db.session.query(User.id).join(UserSocialLinks).filter(
#             User.role.in_(['job_seeker', 'employee'])
#         ).count()
        
#         # Candidates with trials
#         candidates_with_trials = db.session.query(User.id).join(UserTrial).filter(
#             User.role.in_(['job_seeker', 'employee'])
#         ).count()
        
#         # Recent candidates (last 30 days)
#         thirty_days_ago = datetime.utcnow() - timedelta(days=30)
#         recent_candidates = User.query.filter(
#             User.role.in_(['job_seeker', 'employee']),
#             User.created_at >= thirty_days_ago
#         ).count()
        
#         # Average profile completeness
#         profiles = CandidateProfile.query.join(User).filter(
#             User.role.in_(['job_seeker', 'employee'])
#         ).all()
        
#         completeness_scores = []
#         for profile in profiles:
#             score = 0
#             if profile.full_name: score += 20
#             if profile.phone: score += 10
#             if profile.location: score += 10
#             if profile.summary: score += 20
#             if profile.skills: score += 20
#             if profile.education: score += 10
#             if profile.experience: score += 10
#             completeness_scores.append(score)
        
#         avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
#         stats = {
#             'total_candidates': total_candidates,
#             'role_distribution': {role: count for role, count in role_stats},
#             'candidates_with_profiles': candidates_with_profiles,
#             'candidates_with_social': candidates_with_social,
#             'candidates_with_trials': candidates_with_trials,
#             'recent_candidates': recent_candidates,
#             'average_profile_completeness': round(avg_completeness, 2)
#         }
        
#         return jsonify({
#             'stats': stats,
#             'message': 'Candidate statistics retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting candidate stats: {e}")
#         return jsonify({'error': 'Failed to retrieve candidate statistics'}), 500

# @admin_bp.route('/all-emails', methods=['GET'])
# @require_admin_auth
# def get_all_emails():
#     """Get all emails from all tables (admin only)"""
#     try:
#         all_emails = []
        
#         # 1. Get emails from users table
#         users = User.query.all()
#         for user in users:
#             all_emails.append({
#                 'email': user.email,
#                 'source': 'users',
#                 'user_id': user.id,
#                 'role': user.role,
#                 'user_type': user.user_type,
#                 'created_at': user.created_at.isoformat() if user.created_at else None,
#                 'is_candidate': user.role in ['job_seeker', 'employee']
#             })
        
#         # 2. Get emails from ceipal_integrations table
#         ceipal_integrations = CeipalIntegration.query.all()
#         for integration in ceipal_integrations:
#             all_emails.append({
#                 'email': integration.ceipal_email,
#                 'source': 'ceipal_integrations',
#                 'user_id': integration.user_id,
#                 'role': 'integration',
#                 'user_type': 'ceipal',
#                 'created_at': integration.created_at.isoformat() if integration.created_at else None,
#                 'is_candidate': False
#             })
        
#         # 3. Get emails from unlimited_quota_users table
#         unlimited_users = UnlimitedQuotaUser.query.all()
#         for quota_user in unlimited_users:
#             all_emails.append({
#                 'email': quota_user.email,
#                 'source': 'unlimited_quota_users',
#                 'user_id': None,  # This table doesn't have user_id
#                 'role': 'unlimited_quota',
#                 'user_type': 'quota_exempt',
#                 'created_at': quota_user.added_date.isoformat() if quota_user.added_date else None,
#                 'is_candidate': False,
#                 'quota_info': {
#                     'reason': quota_user.reason,
#                     'quota_limit': quota_user.quota_limit,
#                     'daily_limit': quota_user.daily_limit,
#                     'monthly_limit': quota_user.monthly_limit,
#                     'active': quota_user.active,
#                     'expires': quota_user.expires.isoformat() if quota_user.expires else None
#                 }
#             })
        
#         # Sort by email for easier reading
#         all_emails.sort(key=lambda x: x['email'].lower())
        
#         # Get statistics
#         stats = {
#             'total_emails': len(all_emails),
#             'users_table': len([e for e in all_emails if e['source'] == 'users']),
#             'ceipal_integrations': len([e for e in all_emails if e['source'] == 'ceipal_integrations']),
#             'unlimited_quota': len([e for e in all_emails if e['source'] == 'unlimited_quota_users']),
#             'candidates': len([e for e in all_emails if e['is_candidate']]),
#             'non_candidates': len([e for e in all_emails if not e['is_candidate']])
#         }
        
#         return jsonify({
#             'emails': all_emails,
#             'stats': stats,
#             'message': 'All emails retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting all emails: {e}")
#         return jsonify({'error': 'Failed to retrieve all emails'}), 500

# @admin_bp.route('/emails/search', methods=['GET'])
# @require_admin_auth
# def search_emails():
#     """Search for specific emails across all tables (admin only)"""
#     try:
#         search_term = request.args.get('q', '').strip()
        
#         if not search_term:
#             return jsonify({'error': 'Search term is required'}), 400
        
#         results = []
        
#         # Search in users table
#         users = User.query.filter(User.email.ilike(f'%{search_term}%')).all()
#         for user in users:
#             results.append({
#                 'email': user.email,
#                 'source': 'users',
#                 'user_id': user.id,
#                 'role': user.role,
#                 'user_type': user.user_type,
#                 'created_at': user.created_at.isoformat() if user.created_at else None,
#                 'is_candidate': user.role in ['job_seeker', 'employee']
#             })
        
#         # Search in ceipal_integrations table
#         ceipal_integrations = CeipalIntegration.query.filter(
#             CeipalIntegration.ceipal_email.ilike(f'%{search_term}%')
#         ).all()
#         for integration in ceipal_integrations:
#             results.append({
#                 'email': integration.ceipal_email,
#                 'source': 'ceipal_integrations',
#                 'user_id': integration.user_id,
#                 'role': 'integration',
#                 'user_type': 'ceipal',
#                 'created_at': integration.created_at.isoformat() if integration.created_at else None,
#                 'is_candidate': False
#             })
        
#         # Search in unlimited_quota_users table
#         unlimited_users = UnlimitedQuotaUser.query.filter(
#             UnlimitedQuotaUser.email.ilike(f'%{search_term}%')
#         ).all()
#         for quota_user in unlimited_users:
#             results.append({
#                 'email': quota_user.email,
#                 'source': 'unlimited_quota_users',
#                 'user_id': None,
#                 'role': 'unlimited_quota',
#                 'user_type': 'quota_exempt',
#                 'created_at': quota_user.added_date.isoformat() if quota_user.added_date else None,
#                 'is_candidate': False,
#                 'quota_info': {
#                     'reason': quota_user.reason,
#                     'quota_limit': quota_user.quota_limit,
#                     'daily_limit': quota_user.daily_limit,
#                     'monthly_limit': quota_user.monthly_limit,
#                     'active': quota_user.active,
#                     'expires': quota_user.expires.isoformat() if quota_user.expires else None
#                 }
#             })
        
#         # Sort by email
#         results.sort(key=lambda x: x['email'].lower())
        
#         return jsonify({
#             'results': results,
#             'count': len(results),
#             'search_term': search_term,
#             'message': f'Found {len(results)} emails matching "{search_term}"'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error searching emails: {e}")
#         return jsonify({'error': 'Failed to search emails'}), 500

# @admin_bp.route('/activity-logs', methods=['GET'])
# @require_admin_auth
# @log_admin_activity("View Admin Activity Logs")
# def get_admin_activity_logs():
#     """Get admin activity logs (admin only)"""
#     try:
#         from app.services.admin_activity_logger import AdminActivityLogger
        
#         # Get query parameters
#         admin_email = request.args.get('admin_email')
#         activity_type = request.args.get('activity_type')
#         start_date = request.args.get('start_date')
#         end_date = request.args.get('end_date')
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 50, type=int)
        
#         # Parse dates
#         start_date_parsed = None
#         end_date_parsed = None
#         if start_date:
#             try:
#                 start_date_parsed = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
#             except ValueError:
#                 return jsonify({'error': 'Invalid start_date format. Use ISO format.'}), 400
        
#         if end_date:
#             try:
#                 end_date_parsed = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
#             except ValueError:
#                 return jsonify({'error': 'Invalid end_date format. Use ISO format.'}), 400
        
#         # Get activities
#         result = AdminActivityLogger.get_admin_activities(
#             admin_email=admin_email,
#             activity_type=activity_type,
#             start_date=start_date_parsed,
#             end_date=end_date_parsed,
#             page=page,
#             per_page=per_page
#         )
        
#         if result is None:
#             return jsonify({'error': 'Failed to retrieve activity logs'}), 500
        
#         return jsonify({
#             'success': True,
#             'data': result
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting admin activity logs: {e}")
#         return jsonify({'error': 'Failed to retrieve activity logs'}), 500

# @admin_bp.route('/sessions', methods=['GET'])
# @require_admin_auth
# @log_admin_activity("View Admin Sessions")
# def get_admin_sessions():
#     """Get admin sessions (admin only)"""
#     try:
#         from app.services.admin_activity_logger import AdminActivityLogger
        
#         # Get query parameters
#         admin_email = request.args.get('admin_email')
#         is_active = request.args.get('is_active')
#         start_date = request.args.get('start_date')
#         end_date = request.args.get('end_date')
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 50, type=int)
        
#         # Parse boolean
#         is_active_parsed = None
#         if is_active is not None:
#             is_active_parsed = is_active.lower() in ['true', '1', 'yes']
        
#         # Parse dates
#         start_date_parsed = None
#         end_date_parsed = None
#         if start_date:
#             try:
#                 start_date_parsed = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
#             except ValueError:
#                 return jsonify({'error': 'Invalid start_date format. Use ISO format.'}), 400
        
#         if end_date:
#             try:
#                 end_date_parsed = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
#             except ValueError:
#                 return jsonify({'error': 'Invalid end_date format. Use ISO format.'}), 400
        
#         # Get sessions
#         result = AdminActivityLogger.get_admin_sessions(
#             admin_email=admin_email,
#             is_active=is_active_parsed,
#             start_date=start_date_parsed,
#             end_date=end_date_parsed,
#             page=page,
#             per_page=per_page
#         )
        
#         if result is None:
#             return jsonify({'error': 'Failed to retrieve sessions'}), 500
        
#         return jsonify({
#             'success': True,
#             'data': result
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting admin sessions: {e}")
#         return jsonify({'error': 'Failed to retrieve sessions'}), 500

# @admin_bp.route('/activity-stats', methods=['GET'])
# @require_admin_auth
# @log_admin_activity("View Admin Activity Stats")
# def get_admin_activity_stats():
#     """Get admin activity statistics (admin only)"""
#     try:
#         from app.services.admin_activity_logger import AdminActivityLogger
        
#         # Get query parameters
#         admin_email = request.args.get('admin_email')
#         days = request.args.get('days', 30, type=int)
        
#         # Get stats
#         result = AdminActivityLogger.get_admin_stats(
#             admin_email=admin_email,
#             days=days
#         )
        
#         if result is None:
#             return jsonify({'error': 'Failed to retrieve activity stats'}), 500
        
#         return jsonify({
#             'success': True,
#             'data': result
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting admin activity stats: {e}")
#         return jsonify({'error': 'Failed to retrieve activity stats'}), 500

# @admin_bp.route('/cleanup-logs', methods=['POST'])
# @require_admin_auth
# @log_admin_activity("Cleanup Old Admin Logs", include_request_data=True)
# def cleanup_admin_logs():
#     """Clean up old admin activity logs (admin only)"""
#     try:
#         from app.services.admin_activity_logger import AdminActivityLogger
        
#         data = request.get_json() or {}
#         days_to_keep = data.get('days_to_keep', 90)
        
#         if not isinstance(days_to_keep, int) or days_to_keep < 1:
#             return jsonify({'error': 'days_to_keep must be a positive integer'}), 400
        
#         # Clean up logs
#         deleted_count = AdminActivityLogger.cleanup_old_logs(days_to_keep)
        
#         return jsonify({
#             'success': True,
#             'message': f'Cleaned up {deleted_count} old log entries',
#             'deleted_count': deleted_count
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error cleaning up admin logs: {e}")
#         return jsonify({'error': 'Failed to cleanup logs'}), 500

# @admin_bp.route('/jobs', methods=['GET'])
# @require_admin_auth
# def list_jobs():
#     """List all jobs with creator information (admin only)"""
#     try:
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 50, type=int)
#         status_filter = request.args.get('status', None)
#         search = request.args.get('search', None)
#         creator_filter = request.args.get('creator', None)
        
#         # Base query for all jobs
#         query = Job.query
        
#         # Apply status filter
#         if status_filter:
#             query = query.filter(Job.status == status_filter)
        
#         # Apply search filter
#         if search:
#             search_term = f"%{search}%"
#             query = query.filter(
#                 (Job.title.ilike(search_term)) |
#                 (Job.company_name.ilike(search_term)) |
#                 (Job.description.ilike(search_term))
#             )
        
#         # Apply creator filter
#         if creator_filter:
#             query = query.filter(Job.created_by == creator_filter)
        
#         # Get paginated results
#         pagination = query.order_by(desc(Job.created_at)).paginate(
#             page=page, per_page=per_page, error_out=False
#         )
        
#         jobs = []
#         for job in pagination.items:
#             # Get creator information
#             creator = User.query.get(job.created_by)
            
#             # Get application count
#             applications_count = JobApplication.query.filter_by(job_id=job.id).count()
            
#             job_data = {
#                 'id': job.id,
#                 'title': job.title,
#                 'description': job.description,
#                 'location': job.location,
#                 'company_name': job.company_name,
#                 'employment_type': job.employment_type,
#                 'experience_level': job.experience_level,
#                 'salary_min': job.salary_min,
#                 'salary_max': job.salary_max,
#                 'currency': job.currency,
#                 'remote_allowed': job.remote_allowed,
#                 'skills_required': job.skills_required,
#                 'benefits': job.benefits,
#                 'requirements': job.requirements,
#                 'responsibilities': job.responsibilities,
#                 'status': job.status,
#                 'is_public': job.is_public,
#                 'views_count': job.views_count,
#                 'applications_count': applications_count,
#                 'created_at': job.created_at.isoformat() if job.created_at else None,
#                 'updated_at': job.updated_at.isoformat() if job.updated_at else None,
#                 'expires_at': job.expires_at.isoformat() if job.expires_at else None,
#                 'created_by': job.created_by,
#                 'tenant_id': job.tenant_id,
#                 'creator': {
#                     'id': creator.id if creator else None,
#                     'email': creator.email if creator else None,
#                     'role': creator.role if creator else None,
#                     'user_type': creator.user_type if creator else None,
#                     'company_name': creator.company_name if creator else None
#                 } if creator else None
#             }
            
#             jobs.append(job_data)
        
#         return jsonify({
#             'jobs': jobs,
#             'pagination': {
#                 'page': page,
#                 'per_page': per_page,
#                 'total': pagination.total,
#                 'pages': pagination.pages,
#                 'has_next': pagination.has_next,
#                 'has_prev': pagination.has_prev
#             },
#             'message': 'Jobs retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error listing jobs: {e}")
#         return jsonify({'error': 'Failed to retrieve jobs'}), 500

# @admin_bp.route('/jobs/<int:job_id>', methods=['GET'])
# @require_admin_auth
# def get_job_details(job_id):
#     """Get detailed information about a specific job (admin only)"""
#     try:
#         job = Job.query.get(job_id)
        
#         if not job:
#             return jsonify({'error': 'Job not found'}), 404
        
#         # Get creator information
#         creator = User.query.get(job.created_by)
        
#         # Get all applications for this job
#         applications = JobApplication.query.filter_by(job_id=job.id).all()
        
#         job_details = {
#             'id': job.id,
#             'title': job.title,
#             'description': job.description,
#             'location': job.location,
#             'company_name': job.company_name,
#             'employment_type': job.employment_type,
#             'experience_level': job.experience_level,
#             'salary_min': job.salary_min,
#             'salary_max': job.salary_max,
#             'currency': job.currency,
#             'remote_allowed': job.remote_allowed,
#             'skills_required': job.skills_required,
#             'benefits': job.benefits,
#             'requirements': job.requirements,
#             'responsibilities': job.responsibilities,
#             'status': job.status,
#             'is_public': job.is_public,
#             'views_count': job.views_count,
#             'applications_count': job.applications_count,
#             'created_at': job.created_at.isoformat() if job.created_at else None,
#             'updated_at': job.updated_at.isoformat() if job.updated_at else None,
#             'expires_at': job.expires_at.isoformat() if job.expires_at else None,
#             'created_by': job.created_by,
#             'tenant_id': job.tenant_id,
#             'creator': {
#                 'id': creator.id if creator else None,
#                 'email': creator.email if creator else None,
#                 'role': creator.role if creator else None,
#                 'user_type': creator.user_type if creator else None,
#                 'company_name': creator.company_name if creator else None,
#                 'created_at': creator.created_at.isoformat() if creator and creator.created_at else None
#             } if creator else None,
#             'applications': [app.to_dict() for app in applications]
#         }
        
#         return jsonify({
#             'job': job_details,
#             'message': 'Job details retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting job details: {e}")
#         return jsonify({'error': 'Failed to retrieve job details'}), 500

# @admin_bp.route('/jobs/stats', methods=['GET'])
# @require_admin_auth
# def get_jobs_stats():
#     """Get statistics about jobs (admin only)"""
#     try:
#         # Total jobs
#         total_jobs = Job.query.count()
        
#         # Jobs by status
#         status_stats = db.session.query(
#             Job.status, 
#             func.count(Job.id).label('count')
#         ).group_by(Job.status).all()
        
#         # Jobs by creator
#         creator_stats = db.session.query(
#             User.email,
#             func.count(Job.id).label('count')
#         ).join(Job, User.id == Job.created_by).group_by(User.email).order_by(desc('count')).limit(10).all()
        
#         # Jobs by company
#         company_stats = db.session.query(
#             Job.company_name,
#             func.count(Job.id).label('count')
#         ).group_by(Job.company_name).order_by(desc('count')).limit(10).all()
        
#         # Recent jobs (last 30 days)
#         thirty_days_ago = datetime.utcnow() - timedelta(days=30)
#         recent_jobs = Job.query.filter(Job.created_at >= thirty_days_ago).count()
        
#         # Jobs with applications
#         jobs_with_applications = db.session.query(Job.id).join(JobApplication).distinct().count()
        
#         # Average applications per job
#         avg_applications = db.session.query(
#             func.avg(Job.applications_count)
#         ).scalar() or 0
        
#         stats = {
#             'total_jobs': total_jobs,
#             'status_distribution': {status: count for status, count in status_stats},
#             'top_creators': [{'email': email, 'count': count} for email, count in creator_stats],
#             'top_companies': [{'company': company, 'count': count} for company, count in company_stats],
#             'recent_jobs': recent_jobs,
#             'jobs_with_applications': jobs_with_applications,
#             'average_applications_per_job': round(float(avg_applications), 2)
#         }
        
#         return jsonify({
#             'stats': stats,
#             'message': 'Job statistics retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting job stats: {e}")
#         return jsonify({'error': 'Failed to retrieve job statistics'}), 500

# @admin_bp.route('/jobs/creators', methods=['GET'])
# @require_admin_auth
# def get_job_creators():
#     """Get all users who have created jobs (admin only)"""
#     try:
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 50, type=int)
#         search = request.args.get('search', None)
        
#         # Get users who have created jobs
#         query = db.session.query(User).join(Job, User.id == Job.created_by).distinct()
        
#         # Apply search filter
#         if search:
#             search_term = f"%{search}%"
#             query = query.filter(
#                 (User.email.ilike(search_term)) |
#                 (User.company_name.ilike(search_term))
#             )
        
#         # Get paginated results
#         pagination = query.order_by(desc(User.created_at)).paginate(
#             page=page, per_page=per_page, error_out=False
#         )
        
#         creators = []
#         for user in pagination.items:
#             # Get job statistics for this creator
#             jobs_created = Job.query.filter_by(created_by=user.id).count()
#             active_jobs = Job.query.filter_by(created_by=user.id, status='active').count()
#             total_applications = db.session.query(func.sum(Job.applications_count)).filter_by(created_by=user.id).scalar() or 0
#             total_views = db.session.query(func.sum(Job.views_count)).filter_by(created_by=user.id).scalar() or 0
            
#             # Get recent jobs
#             recent_jobs = Job.query.filter_by(created_by=user.id).order_by(desc(Job.created_at)).limit(5).all()
            
#             creator_data = {
#                 'id': user.id,
#                 'email': user.email,
#                 'role': user.role,
#                 'user_type': user.user_type,
#                 'company_name': user.company_name,
#                 'created_at': user.created_at.isoformat() if user.created_at else None,
#                 'tenant_id': user.tenant_id,
#                 'job_stats': {
#                     'jobs_created': jobs_created,
#                     'active_jobs': active_jobs,
#                     'total_applications': total_applications,
#                     'total_views': total_views
#                 },
#                 'recent_jobs': [{
#                     'id': job.id,
#                     'title': job.title,
#                     'company_name': job.company_name,
#                     'status': job.status,
#                     'created_at': job.created_at.isoformat() if job.created_at else None
#                 } for job in recent_jobs]
#             }
            
#             creators.append(creator_data)
        
#         return jsonify({
#             'creators': creators,
#             'pagination': {
#                 'page': page,
#                 'per_page': per_page,
#                 'total': pagination.total,
#                 'pages': pagination.pages,
#                 'has_next': pagination.has_next,
#                 'has_prev': pagination.has_prev
#             },
#             'message': 'Job creators retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting job creators: {e}")
#         return jsonify({'error': 'Failed to retrieve job creators'}), 500

# @admin_bp.route('/jobs/<int:job_id>/applicants', methods=['GET'])
# @require_admin_auth
# def get_job_applicants(job_id):
#     """Get all applicants for a specific job (admin only)"""
#     try:
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 50, type=int)
#         status_filter = request.args.get('status', None)
#         search = request.args.get('search', None)
        
#         # Check if job exists
#         job = Job.query.get(job_id)
#         if not job:
#             return jsonify({'error': 'Job not found'}), 404
        
#         # Base query for applications
#         query = JobApplication.query.filter_by(job_id=job_id)
        
#         # Apply status filter
#         if status_filter:
#             query = query.filter(JobApplication.status == status_filter)
        
#         # Apply search filter
#         if search:
#             search_term = f"%{search}%"
#             query = query.join(User, JobApplication.applicant_id == User.id).filter(
#                 User.email.ilike(search_term)
#             )
        
#         # Get paginated results
#         pagination = query.order_by(desc(JobApplication.applied_at)).paginate(
#             page=page, per_page=per_page, error_out=False
#         )
        
#         applicants = []
#         for application in pagination.items:
#             # Get applicant information
#             applicant = User.query.get(application.applicant_id)
            
#             # Get applicant profile
#             profile = CandidateProfile.query.filter_by(user_id=application.applicant_id).first()
            
#             applicant_data = {
#                 'id': application.id,
#                 'job_id': application.job_id,
#                 'job_title': job.title,
#                 'applicant_id': application.applicant_id,
#                 'applicant': {
#                     'id': applicant.id if applicant else None,
#                     'email': applicant.email if applicant else None,
#                     'role': applicant.role if applicant else None,
#                     'user_type': applicant.user_type if applicant else None,
#                     'company_name': applicant.company_name if applicant else None,
#                     'created_at': applicant.created_at.isoformat() if applicant and applicant.created_at else None
#                 } if applicant else None,
#                 'profile': {
#                     'full_name': profile.full_name if profile else None,
#                     'phone': profile.phone if profile else None,
#                     'location': profile.location if profile else None,
#                     'summary': profile.summary if profile else None,
#                     'experience_years': profile.experience_years if profile else None
#                 } if profile else None,
#                 'application': {
#                     'cover_letter': application.cover_letter,
#                     'resume_filename': application.resume_filename,
#                     'status': application.status,
#                     'notes': application.notes,
#                     'interview_scheduled': application.interview_scheduled,
#                     'interview_date': application.interview_date.isoformat() if application.interview_date else None,
#                     'interview_meeting_link': application.interview_meeting_link,
#                     'interview_meeting_type': application.interview_meeting_type,
#                     'interview_notes': application.interview_notes,
#                     'additional_answers': application.additional_answers,
#                     'applied_at': application.applied_at.isoformat() if application.applied_at else None,
#                     'updated_at': application.updated_at.isoformat() if application.updated_at else None,
#                     'reviewed_at': application.reviewed_at.isoformat() if application.reviewed_at else None
#                 }
#             }
            
#             applicants.append(applicant_data)
        
#         return jsonify({
#             'applicants': applicants,
#             'job': {
#                 'id': job.id,
#                 'title': job.title,
#                 'company_name': job.company_name,
#                 'location': job.location,
#                 'status': job.status
#             },
#             'pagination': {
#                 'page': page,
#                 'per_page': per_page,
#                 'total': pagination.total,
#                 'pages': pagination.pages,
#                 'has_next': pagination.has_next,
#                 'has_prev': pagination.has_prev
#             },
#             'message': 'Job applicants retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting job applicants: {e}")
#         return jsonify({'error': 'Failed to retrieve job applicants'}), 500

# @admin_bp.route('/jobs/applicants', methods=['GET'])
# @require_admin_auth
# def list_all_applicants():
#     """Get all job applicants across all jobs (admin only)"""
#     try:
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 50, type=int)
#         status_filter = request.args.get('status', None)
#         job_filter = request.args.get('job_id', None)
#         search = request.args.get('search', None)
        
#         # Base query for applications
#         query = JobApplication.query
        
#         # Apply status filter
#         if status_filter:
#             query = query.filter(JobApplication.status == status_filter)
        
#         # Apply job filter
#         if job_filter:
#             query = query.filter(JobApplication.job_id == job_filter)
        
#         # Apply search filter
#         if search:
#             search_term = f"%{search}%"
#             query = query.join(User, JobApplication.applicant_id == User.id).join(Job, JobApplication.job_id == Job.id).filter(
#                 (User.email.ilike(search_term)) |
#                 (Job.title.ilike(search_term)) |
#                 (Job.company_name.ilike(search_term))
#             )
        
#         # Get paginated results
#         pagination = query.order_by(desc(JobApplication.applied_at)).paginate(
#             page=page, per_page=per_page, error_out=False
#         )
        
#         applicants = []
#         for application in pagination.items:
#             # Get applicant and job information
#             applicant = User.query.get(application.applicant_id)
#             job = Job.query.get(application.job_id)
#             profile = CandidateProfile.query.filter_by(user_id=application.applicant_id).first()
            
#             applicant_data = {
#                 'id': application.id,
#                 'job_id': application.job_id,
#                 'job': {
#                     'id': job.id if job else None,
#                     'title': job.title if job else None,
#                     'company_name': job.company_name if job else None,
#                     'location': job.location if job else None,
#                     'status': job.status if job else None
#                 } if job else None,
#                 'applicant_id': application.applicant_id,
#                 'applicant': {
#                     'id': applicant.id if applicant else None,
#                     'email': applicant.email if applicant else None,
#                     'role': applicant.role if applicant else None,
#                     'user_type': applicant.user_type if applicant else None,
#                     'company_name': applicant.company_name if applicant else None,
#                     'created_at': applicant.created_at.isoformat() if applicant and applicant.created_at else None
#                 } if applicant else None,
#                 'profile': {
#                     'full_name': profile.full_name if profile else None,
#                     'phone': profile.phone if profile else None,
#                     'location': profile.location if profile else None,
#                     'summary': profile.summary if profile else None,
#                     'experience_years': profile.experience_years if profile else None
#                 } if profile else None,
#                 'application': {
#                     'cover_letter': application.cover_letter,
#                     'resume_filename': application.resume_filename,
#                     'status': application.status,
#                     'notes': application.notes,
#                     'interview_scheduled': application.interview_scheduled,
#                     'interview_date': application.interview_date.isoformat() if application.interview_date else None,
#                     'interview_meeting_link': application.interview_meeting_link,
#                     'interview_meeting_type': application.interview_meeting_type,
#                     'interview_notes': application.interview_notes,
#                     'additional_answers': application.additional_answers,
#                     'applied_at': application.applied_at.isoformat() if application.applied_at else None,
#                     'updated_at': application.updated_at.isoformat() if application.updated_at else None,
#                     'reviewed_at': application.reviewed_at.isoformat() if application.reviewed_at else None
#                 }
#             }
            
#             applicants.append(applicant_data)
        
#         return jsonify({
#             'applicants': applicants,
#             'pagination': {
#                 'page': page,
#                 'per_page': per_page,
#                 'total': pagination.total,
#                 'pages': pagination.pages,
#                 'has_next': pagination.has_next,
#                 'has_prev': pagination.has_prev
#             },
#             'message': 'All job applicants retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting all applicants: {e}")
#         return jsonify({'error': 'Failed to retrieve applicants'}), 500

# @admin_bp.route('/jobs/applicants/stats', methods=['GET'])
# @require_admin_auth
# def get_applicants_stats():
#     """Get statistics about job applicants (admin only)"""
#     try:
#         # Total applications
#         total_applications = JobApplication.query.count()
        
#         # Applications by status
#         status_stats = db.session.query(
#             JobApplication.status, 
#             func.count(JobApplication.id).label('count')
#         ).group_by(JobApplication.status).all()
        
#         # Applications by job
#         job_stats = db.session.query(
#             Job.title,
#             Job.company_name,
#             func.count(JobApplication.id).label('count')
#         ).join(JobApplication, Job.id == JobApplication.job_id).group_by(Job.id, Job.title, Job.company_name).order_by(desc('count')).limit(10).all()
        
#         # Applications by applicant (top applicants)
#         applicant_stats = db.session.query(
#             User.email,
#             func.count(JobApplication.id).label('count')
#         ).join(JobApplication, User.id == JobApplication.applicant_id).group_by(User.email).order_by(desc('count')).limit(10).all()
        
#         # Recent applications (last 30 days)
#         thirty_days_ago = datetime.utcnow() - timedelta(days=30)
#         recent_applications = JobApplication.query.filter(JobApplication.applied_at >= thirty_days_ago).count()
        
#         # Applications with interviews scheduled
#         interview_scheduled = JobApplication.query.filter(JobApplication.interview_scheduled == True).count()
        
#         # Average applications per job
#         avg_applications = db.session.query(
#             func.avg(Job.applications_count)
#         ).scalar() or 0
        
#         stats = {
#             'total_applications': total_applications,
#             'status_distribution': {status: count for status, count in status_stats},
#             'top_jobs': [{'title': title, 'company': company, 'count': count} for title, company, count in job_stats],
#             'top_applicants': [{'email': email, 'count': count} for email, count in applicant_stats],
#             'recent_applications': recent_applications,
#             'interview_scheduled': interview_scheduled,
#             'average_applications_per_job': round(float(avg_applications), 2)
#         }
        
#         return jsonify({
#             'stats': stats,
#             'message': 'Applicant statistics retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting applicant stats: {e}")
#         return jsonify({'error': 'Failed to retrieve applicant statistics'}), 500

# @admin_bp.route('/onboarding-submissions', methods=['GET'])
# @require_admin_auth
# @log_admin_activity("List Onboarding Submissions")
# def list_onboarding_submissions():
#     """List all onboarding submissions (admin only)"""
#     try:
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 50, type=int)
#         search = request.args.get('search', None)
        
#         # Base query for all onboarding submissions
#         query = OnboardingSubmission.query
        
#         # Apply search filter (search by user email or company name)
#         if search:
#             search_term = f"%{search}%"
#             query = query.join(User, OnboardingSubmission.user_id == User.id).filter(
#                 (User.email.ilike(search_term)) |
#                 (User.company_name.ilike(search_term)) |
#                 (User.first_name.ilike(search_term)) |
#                 (User.last_name.ilike(search_term))
#             )
        
#         # Get paginated results
#         pagination = query.order_by(desc(OnboardingSubmission.created_at)).paginate(
#             page=page, per_page=per_page, error_out=False
#         )
        
#         submissions = []
#         for submission in pagination.items:
#             user = User.query.get(submission.user_id)
#             submission_data = submission.data or {}
            
#             # Get user profile if available
#             profile = None
#             if user:
#                 profile = CandidateProfile.query.filter_by(user_id=user.id).first()
            
#             submissions.append({
#                 'id': submission.id,
#                 'user_id': submission.user_id,
#                 'user': {
#                     'id': user.id if user else None,
#                     'email': user.email if user else None,
#                     'full_name': profile.full_name if profile else None,
#                     'company_name': user.company_name if user else None,
#                     'role': user.role if user else None
#                 },
#                 'contact_name': submission_data.get('contactName') or submission_data.get('full_name'),
#                 'email': submission_data.get('email') or (user.email if user else None),
#                 'phone': submission_data.get('phone'),
#                 'company_name': submission_data.get('companyName') or submission_data.get('company_name') or (user.company_name if user else None),
#                 'website': submission_data.get('website'),
#                 'company_size': submission_data.get('companySize'),
#                 'location': submission_data.get('location') or submission_data.get('address'),
#                 'tools': submission_data.get('tools', []),
#                 'goals': submission_data.get('goals') or submission_data.get('business_goals'),
#                 'services': submission_data.get('services', []),
#                 'scope': submission_data.get('scope') or submission_data.get('project_scope'),
#                 'timeline': submission_data.get('timeline'),
#                 'budget': submission_data.get('budget'),
#                 'billing_address': submission_data.get('billingAddress'),
#                 'payment_method': submission_data.get('paymentMethod'),
#                 'consent_data': submission_data.get('consentData', False),
#                 'consent_communication': submission_data.get('consentCommunication', False),
#                 'notes': submission_data.get('notes'),
#                 'created_at': submission.created_at.isoformat() if submission.created_at else None,
#                 'updated_at': submission.updated_at.isoformat() if submission.updated_at else None
#             })
        
#         return jsonify({
#             'success': True,
#             'data': {
#                 'submissions': submissions,
#                 'pagination': {
#                     'page': page,
#                     'per_page': per_page,
#                     'total': pagination.total,
#                     'pages': pagination.pages,
#                     'has_next': pagination.has_next,
#                     'has_prev': pagination.has_prev
#                 }
#             },
#             'message': 'Onboarding submissions retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error listing onboarding submissions: {e}")
#         return jsonify({'error': 'Failed to retrieve onboarding submissions'}), 500

# @admin_bp.route('/onboarding-submissions/<int:submission_id>', methods=['GET'])
# @require_admin_auth
# @log_admin_activity("View Onboarding Submission Details")
# def get_onboarding_submission(submission_id):
#     """Get a specific onboarding submission by ID (admin only)"""
#     try:
#         submission = OnboardingSubmission.query.get(submission_id)
#         if not submission:
#             return jsonify({'error': 'Onboarding submission not found'}), 404
        
#         user = User.query.get(submission.user_id)
#         submission_data = submission.data or {}
        
#         # Get user profile if available
#         profile = None
#         if user:
#             profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        
#         # Flatten the data structure for easier frontend access
#         submission_details = {
#             'id': submission.id,
#             'user_id': submission.user_id,
#             'user': {
#                 'id': user.id if user else None,
#                 'email': user.email if user else None,
#                 'full_name': profile.full_name if profile else None,
#                 'company_name': user.company_name if user else None,
#                 'role': user.role if user else None
#             },
#             # Flatten submission data to top level
#             'contact_name': submission_data.get('contactName') or submission_data.get('full_name'),
#             'email': submission_data.get('email') or (user.email if user else None),
#             'phone': submission_data.get('phone'),
#             'company_name': submission_data.get('companyName') or submission_data.get('company_name') or (user.company_name if user else None),
#             'website': submission_data.get('website'),
#             'company_size': submission_data.get('companySize'),
#             'location': submission_data.get('location') or submission_data.get('address'),
#             'tools': submission_data.get('tools', []),
#             'goals': submission_data.get('goals') or submission_data.get('business_goals'),
#             'services': submission_data.get('services', []),
#             'scope': submission_data.get('scope') or submission_data.get('project_scope'),
#             'timeline': submission_data.get('timeline'),
#             'budget': submission_data.get('budget'),
#             'billing_address': submission_data.get('billingAddress'),
#             'payment_method': submission_data.get('paymentMethod'),
#             'consent_data': submission_data.get('consentData', False),
#             'consent_communication': submission_data.get('consentCommunication', False),
#             'notes': submission_data.get('notes'),
#             'created_at': submission.created_at.isoformat() if submission.created_at else None,
#             'updated_at': submission.updated_at.isoformat() if submission.updated_at else None,
#             # Also include raw data for reference
#             'data': submission_data
#         }
        
#         return jsonify({
#             'success': True,
#             'data': submission_details,
#             'message': 'Onboarding submission retrieved successfully'
#         }), 200
        
#     except Exception as e:
#         logger.error(f"Error getting onboarding submission: {e}")
#         return jsonify({'error': 'Failed to retrieve onboarding submission'}), 500

from flask import Blueprint, jsonify, request
from app.simple_logger import get_logger
from app.models import User, db, CandidateProfile, UserSocialLinks, UserTrial, CeipalIntegration, StafferlinkIntegration, JobAdderIntegration, JDSearchLog, UserImage, UserKPIs, UserSkillGap, UserLearningPath, UserAchievement, UserGoal, UserSchedule, Tenant, TenantAlert, CandidateSkill, CandidateEducation, CandidateExperience, CandidateCertification, CandidateProject, UnlimitedQuotaUser, SavedCandidate, SearchHistory, Job, JobApplication, AdminActivityLog, AdminSession, OnboardingSubmission, OnboardingFlag, UserFunctionalityPreferences, MessageTemplate, CandidateCommunication, EmployeeProfile, OrganizationMetadata, Timesheet, IntegrationSubmission, UserModuleAccess
from app.utils.unlimited_quota_production import (
    production_unlimited_quota_manager as unlimited_quota_manager, 
    is_unlimited_quota_user,
    get_unlimited_quota_info
)
from app.utils.admin_auth import require_admin_auth
from app.utils.admin_activity_decorator import log_admin_activity
import logging
from datetime import datetime, timedelta
from sqlalchemy import func, desc

logger = get_logger("admin")

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/unlimited-quota/users', methods=['GET'])
@require_admin_auth
@log_admin_activity("List Unlimited Quota Users")
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
@log_admin_activity("Add Unlimited Quota User", include_request_data=True)
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
@log_admin_activity("List All Candidates")
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
        stafferlink_integration = StafferlinkIntegration.query.filter_by(user_id=user.id).first()
        jobadder_integration = JobAdderIntegration.query.filter_by(user_id=user.id).first()
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
            'stafferlink_integration': stafferlink_integration.to_dict() if stafferlink_integration else None,
            'jobadder_integration': jobadder_integration.to_dict() if jobadder_integration else None,
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
        
        # Check if the requesting user is authorized
        if user_email not in ['vinit@adeptaipro.com', 'contact@kempian.ai']:
            logger.warning(f"Unauthorized deletion attempt by {user_email}")
            return jsonify({
                'error': 'Access denied. Only authorized admins can delete candidates.',
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
        
        # 13. Delete jobs created by the user (must be done before deleting user to avoid foreign key constraint)
        jobs_created = Job.query.filter_by(created_by=user.id).all()
        if jobs_created:
            # Delete job applications for these jobs first (cascade will handle this, but being explicit)
            for job in jobs_created:
                job_applications_for_job = JobApplication.query.filter_by(job_id=job.id).all()
                for application in job_applications_for_job:
                    db.session.delete(application)
                db.session.delete(job)
            deleted_data['jobs_created_deleted'] = len(jobs_created)
            logger.info(f"Deleted {len(jobs_created)} jobs created by user")
        
        # 14. Delete job applications (this was causing the foreign key constraint error)
        job_applications = JobApplication.query.filter_by(applicant_id=user.id).all()
        if job_applications:
            for application in job_applications:
                db.session.delete(application)
            deleted_data['job_applications_deleted'] = len(job_applications)
            logger.info(f"Deleted {len(job_applications)} job applications")
        
        # 15. Delete saved candidates (must be done before deleting user to avoid foreign key constraint)
        saved_candidates = SavedCandidate.query.filter_by(user_id=user.id).all()
        if saved_candidates:
            for saved_candidate in saved_candidates:
                db.session.delete(saved_candidate)
            deleted_data['saved_candidates_deleted'] = len(saved_candidates)
            logger.info(f"Deleted {len(saved_candidates)} saved candidates")
        
        # 16. Delete search history (must be done before deleting user to avoid foreign key constraint)
        search_history = SearchHistory.query.filter_by(user_id=user.id).all()
        if search_history:
            for history in search_history:
                db.session.delete(history)
            deleted_data['search_history_deleted'] = len(search_history)
            logger.info(f"Deleted {len(search_history)} search history records")
        
        # 17. Delete onboarding flags (must be done before deleting user to avoid foreign key constraint)
        onboarding_flag = OnboardingFlag.query.filter_by(user_id=user.id).first()
        if onboarding_flag:
            db.session.delete(onboarding_flag)
            deleted_data['onboarding_flag_deleted'] = True
            logger.info(f"Deleted onboarding flag for user {user.id}")
        
        # 18. Delete onboarding submissions (must be done before deleting user to avoid foreign key constraint)
        onboarding_submissions = OnboardingSubmission.query.filter_by(user_id=user.id).all()
        if onboarding_submissions:
            for submission in onboarding_submissions:
                db.session.delete(submission)
            deleted_data['onboarding_submissions_deleted'] = len(onboarding_submissions)
            logger.info(f"Deleted {len(onboarding_submissions)} onboarding submissions")
        
        # 19. Delete user functionality preferences (must be done before deleting user to avoid foreign key constraint)
        user_functionality_preferences = UserFunctionalityPreferences.query.filter_by(user_id=user.id).first()
        if user_functionality_preferences:
            db.session.delete(user_functionality_preferences)
            deleted_data['user_functionality_preferences_deleted'] = True
            logger.info(f"Deleted user functionality preferences for user {user.id}")
        
        # 19.5. Delete message templates, candidate communications, employee profiles, and timesheets (must be done before deleting user to avoid foreign key constraint)
        # Use no_autoflush to prevent premature flushes that could trigger foreign key constraint errors
        with db.session.no_autoflush:
            message_templates = MessageTemplate.query.filter_by(user_id=user.id).all()
            if message_templates:
                for template in message_templates:
                    db.session.delete(template)
                deleted_data['message_templates_deleted'] = len(message_templates)
                logger.info(f"Deleted {len(message_templates)} message templates for user {user.id}")
            
            candidate_communications = CandidateCommunication.query.filter_by(user_id=user.id).all()
            if candidate_communications:
                for communication in candidate_communications:
                    db.session.delete(communication)
                deleted_data['candidate_communications_deleted'] = len(candidate_communications)
                logger.info(f"Deleted {len(candidate_communications)} candidate communications for user {user.id}")
            
            employee_profile = EmployeeProfile.query.filter_by(user_id=user.id).first()
            if employee_profile:
                db.session.delete(employee_profile)
                deleted_data['employee_profile_deleted'] = True
                logger.info(f"Deleted employee profile for user {user.id}")
            
            timesheets = Timesheet.query.filter_by(user_id=user.id).all()
            if timesheets:
                for timesheet in timesheets:
                    db.session.delete(timesheet)
                deleted_data['timesheets_deleted'] = len(timesheets)
                logger.info(f"Deleted {len(timesheets)} timesheets for user {user.id}")
        
        # 20. Delete ALL possible tables with foreign key references (comprehensive cleanup)
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
        
        # 20.5. Delete integration submissions, admin activity logs, and admin sessions (must be done before deleting user to avoid foreign key constraint)
        # Use no_autoflush to prevent premature flushes that could trigger foreign key constraint errors
        with db.session.no_autoflush:
            # Delete integration submissions
            integration_submissions = IntegrationSubmission.query.filter_by(user_id=user.id).all()
            if integration_submissions:
                for submission in integration_submissions:
                    db.session.delete(submission)
                deleted_data['integration_submissions_deleted'] = len(integration_submissions)
                logger.info(f"Deleted {len(integration_submissions)} integration submissions for user {user.id}")
            
            # Delete admin activity logs
            admin_activity_logs = AdminActivityLog.query.filter_by(admin_id=user.id).all()
            if admin_activity_logs:
                for log in admin_activity_logs:
                    db.session.delete(log)
                deleted_data['admin_activity_logs_deleted'] = len(admin_activity_logs)
                logger.info(f"Deleted {len(admin_activity_logs)} admin activity logs for user {user.id}")
            
            # Delete admin sessions
            admin_sessions = AdminSession.query.filter_by(admin_id=user.id).all()
            if admin_sessions:
                for session in admin_sessions:
                    db.session.delete(session)
                deleted_data['admin_sessions_deleted'] = len(admin_sessions)
                logger.info(f"Deleted {len(admin_sessions)} admin sessions for user {user.id}")
            
            # Set created_by to NULL for organization_metadata records created by this user
            # (We preserve the organization but remove the reference to the deleted user)
            org_metadata_created_by_user = OrganizationMetadata.query.filter_by(created_by=user.id).all()
            if org_metadata_created_by_user:
                for org_meta in org_metadata_created_by_user:
                    org_meta.created_by = None
                deleted_data['organization_metadata_updated'] = len(org_metadata_created_by_user)
                logger.info(f"Updated {len(org_metadata_created_by_user)} organization metadata records (set created_by to NULL) for user {user.id}")
        
        # 21. Delete the user
        db.session.delete(user)
        logger.info(f"Deleted user: {user.email}")
        
        # 22. Check if tenant has other users, if not, delete tenant
        tenant = Tenant.query.get(user.tenant_id)
        if tenant:
            remaining_users = User.query.filter_by(tenant_id=tenant.id).count()
            if remaining_users == 0:
                # Delete organization metadata FIRST (before deleting tenant)
                # Use no_autoflush to prevent premature flushes that could trigger foreign key constraint errors
                with db.session.no_autoflush:
                    org_metadata = OrganizationMetadata.query.filter_by(tenant_id=tenant.id).first()
                    if org_metadata:
                        db.session.delete(org_metadata)
                        deleted_data['organization_metadata_deleted'] = True
                        logger.info(f"Deleted organization metadata for tenant {tenant.id}")
                
                # Flush to ensure organization_metadata is deleted before proceeding
                db.session.flush()
                
                # Delete tenant alerts (before deleting tenant)
                tenant_alerts = TenantAlert.query.filter_by(tenant_id=tenant.id).all()
                for alert in tenant_alerts:
                    db.session.delete(alert)
                db.session.flush()  # Ensure tenant alerts are deleted before tenant
                
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
        
        # 3. Get emails from stafferlink_integrations table
        stafferlink_integrations = StafferlinkIntegration.query.all()
        for integration in stafferlink_integrations:
            all_emails.append({
                'email': integration.stafferlink_email,
                'source': 'stafferlink_integrations',
                'user_id': integration.user_id,
                'role': 'integration',
                'user_type': 'stafferlink',
                'created_at': integration.created_at.isoformat() if integration.created_at else None,
                'is_candidate': False
            })
        
        # 4. Get emails from jobadder_integrations table
        jobadder_integrations = JobAdderIntegration.query.all()
        for integration in jobadder_integrations:
            all_emails.append({
                'email': integration.account_email or f"jobadder_{integration.user_id}@jobadder.com",
                'source': 'jobadder_integrations',
                'user_id': integration.user_id,
                'role': 'integration',
                'user_type': 'jobadder',
                'created_at': integration.created_at.isoformat() if integration.created_at else None,
                'is_candidate': False,
                'account_name': integration.account_name
            })
        
        # 5. Get emails from unlimited_quota_users table
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
            'stafferlink_integrations': len([e for e in all_emails if e['source'] == 'stafferlink_integrations']),
            'jobadder_integrations': len([e for e in all_emails if e['source'] == 'jobadder_integrations']),
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
        
        # Search in stafferlink_integrations table
        stafferlink_integrations = StafferlinkIntegration.query.filter(
            StafferlinkIntegration.stafferlink_email.ilike(f'%{search_term}%')
        ).all()
        for integration in stafferlink_integrations:
            results.append({
                'email': integration.stafferlink_email,
                'source': 'stafferlink_integrations',
                'user_id': integration.user_id,
                'role': 'integration',
                'user_type': 'stafferlink',
                'created_at': integration.created_at.isoformat() if integration.created_at else None,
                'is_candidate': False
            })
        
        # Search in jobadder_integrations table
        jobadder_integrations = JobAdderIntegration.query.filter(
            (JobAdderIntegration.account_email.ilike(f'%{search_term}%')) |
            (JobAdderIntegration.account_name.ilike(f'%{search_term}%'))
        ).all()
        for integration in jobadder_integrations:
            results.append({
                'email': integration.account_email or f"jobadder_{integration.user_id}@jobadder.com",
                'source': 'jobadder_integrations',
                'user_id': integration.user_id,
                'role': 'integration',
                'user_type': 'jobadder',
                'created_at': integration.created_at.isoformat() if integration.created_at else None,
                'is_candidate': False,
                'account_name': integration.account_name
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

@admin_bp.route('/activity-logs', methods=['GET'])
@require_admin_auth
@log_admin_activity("View Admin Activity Logs")
def get_admin_activity_logs():
    """Get admin activity logs (admin only)"""
    try:
        from app.services.admin_activity_logger import AdminActivityLogger
        
        # Get query parameters
        admin_email = request.args.get('admin_email')
        activity_type = request.args.get('activity_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        # Parse dates
        start_date_parsed = None
        end_date_parsed = None
        if start_date:
            try:
                start_date_parsed = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'error': 'Invalid start_date format. Use ISO format.'}), 400
        
        if end_date:
            try:
                end_date_parsed = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'error': 'Invalid end_date format. Use ISO format.'}), 400
        
        # Get activities
        result = AdminActivityLogger.get_admin_activities(
            admin_email=admin_email,
            activity_type=activity_type,
            start_date=start_date_parsed,
            end_date=end_date_parsed,
            page=page,
            per_page=per_page
        )
        
        if result is None:
            return jsonify({'error': 'Failed to retrieve activity logs'}), 500
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting admin activity logs: {e}")
        return jsonify({'error': 'Failed to retrieve activity logs'}), 500

@admin_bp.route('/sessions', methods=['GET'])
@require_admin_auth
@log_admin_activity("View Admin Sessions")
def get_admin_sessions():
    """Get admin sessions (admin only)"""
    try:
        from app.services.admin_activity_logger import AdminActivityLogger
        
        # Get query parameters
        admin_email = request.args.get('admin_email')
        is_active = request.args.get('is_active')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        # Parse boolean
        is_active_parsed = None
        if is_active is not None:
            is_active_parsed = is_active.lower() in ['true', '1', 'yes']
        
        # Parse dates
        start_date_parsed = None
        end_date_parsed = None
        if start_date:
            try:
                start_date_parsed = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'error': 'Invalid start_date format. Use ISO format.'}), 400
        
        if end_date:
            try:
                end_date_parsed = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'error': 'Invalid end_date format. Use ISO format.'}), 400
        
        # Get sessions
        result = AdminActivityLogger.get_admin_sessions(
            admin_email=admin_email,
            is_active=is_active_parsed,
            start_date=start_date_parsed,
            end_date=end_date_parsed,
            page=page,
            per_page=per_page
        )
        
        if result is None:
            return jsonify({'error': 'Failed to retrieve sessions'}), 500
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting admin sessions: {e}")
        return jsonify({'error': 'Failed to retrieve sessions'}), 500

@admin_bp.route('/activity-stats', methods=['GET'])
@require_admin_auth
@log_admin_activity("View Admin Activity Stats")
def get_admin_activity_stats():
    """Get admin activity statistics (admin only)"""
    try:
        from app.services.admin_activity_logger import AdminActivityLogger
        
        # Get query parameters
        admin_email = request.args.get('admin_email')
        days = request.args.get('days', 30, type=int)
        
        # Get stats
        result = AdminActivityLogger.get_admin_stats(
            admin_email=admin_email,
            days=days
        )
        
        if result is None:
            return jsonify({'error': 'Failed to retrieve activity stats'}), 500
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting admin activity stats: {e}")
        return jsonify({'error': 'Failed to retrieve activity stats'}), 500

@admin_bp.route('/cleanup-logs', methods=['POST'])
@require_admin_auth
@log_admin_activity("Cleanup Old Admin Logs", include_request_data=True)
def cleanup_admin_logs():
    """Clean up old admin activity logs (admin only)"""
    try:
        from app.services.admin_activity_logger import AdminActivityLogger
        
        data = request.get_json() or {}
        days_to_keep = data.get('days_to_keep', 90)
        
        if not isinstance(days_to_keep, int) or days_to_keep < 1:
            return jsonify({'error': 'days_to_keep must be a positive integer'}), 400
        
        # Clean up logs
        deleted_count = AdminActivityLogger.cleanup_old_logs(days_to_keep)
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up {deleted_count} old log entries',
            'deleted_count': deleted_count
        }), 200
        
    except Exception as e:
        logger.error(f"Error cleaning up admin logs: {e}")
        return jsonify({'error': 'Failed to cleanup logs'}), 500

@admin_bp.route('/jobs', methods=['GET'])
@require_admin_auth
def list_jobs():
    """List all jobs with creator information (admin only)"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        status_filter = request.args.get('status', None)
        search = request.args.get('search', None)
        creator_filter = request.args.get('creator', None)
        
        # Base query for all jobs
        query = Job.query
        
        # Apply status filter
        if status_filter:
            query = query.filter(Job.status == status_filter)
        
        # Apply search filter
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (Job.title.ilike(search_term)) |
                (Job.company_name.ilike(search_term)) |
                (Job.description.ilike(search_term))
            )
        
        # Apply creator filter
        if creator_filter:
            query = query.filter(Job.created_by == creator_filter)
        
        # Get paginated results
        pagination = query.order_by(desc(Job.created_at)).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        jobs = []
        for job in pagination.items:
            # Get creator information
            creator = User.query.get(job.created_by)
            
            # Get application count
            applications_count = JobApplication.query.filter_by(job_id=job.id).count()
            
            job_data = {
                'id': job.id,
                'title': job.title,
                'description': job.description,
                'location': job.location,
                'company_name': job.company_name,
                'employment_type': job.employment_type,
                'experience_level': job.experience_level,
                'salary_min': job.salary_min,
                'salary_max': job.salary_max,
                'currency': job.currency,
                'remote_allowed': job.remote_allowed,
                'skills_required': job.skills_required,
                'benefits': job.benefits,
                'requirements': job.requirements,
                'responsibilities': job.responsibilities,
                'status': job.status,
                'is_public': job.is_public,
                'views_count': job.views_count,
                'applications_count': applications_count,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'updated_at': job.updated_at.isoformat() if job.updated_at else None,
                'expires_at': job.expires_at.isoformat() if job.expires_at else None,
                'created_by': job.created_by,
                'tenant_id': job.tenant_id,
                'creator': {
                    'id': creator.id if creator else None,
                    'email': creator.email if creator else None,
                    'role': creator.role if creator else None,
                    'user_type': creator.user_type if creator else None,
                    'company_name': creator.company_name if creator else None
                } if creator else None
            }
            
            jobs.append(job_data)
        
        return jsonify({
            'jobs': jobs,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            },
            'message': 'Jobs retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return jsonify({'error': 'Failed to retrieve jobs'}), 500

@admin_bp.route('/jobs/<int:job_id>', methods=['GET'])
@require_admin_auth
def get_job_details(job_id):
    """Get detailed information about a specific job (admin only)"""
    try:
        job = Job.query.get(job_id)
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Get creator information
        creator = User.query.get(job.created_by)
        
        # Get all applications for this job
        applications = JobApplication.query.filter_by(job_id=job.id).all()
        
        job_details = {
            'id': job.id,
            'title': job.title,
            'description': job.description,
            'location': job.location,
            'company_name': job.company_name,
            'employment_type': job.employment_type,
            'experience_level': job.experience_level,
            'salary_min': job.salary_min,
            'salary_max': job.salary_max,
            'currency': job.currency,
            'remote_allowed': job.remote_allowed,
            'skills_required': job.skills_required,
            'benefits': job.benefits,
            'requirements': job.requirements,
            'responsibilities': job.responsibilities,
            'status': job.status,
            'is_public': job.is_public,
            'views_count': job.views_count,
            'applications_count': job.applications_count,
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'updated_at': job.updated_at.isoformat() if job.updated_at else None,
            'expires_at': job.expires_at.isoformat() if job.expires_at else None,
            'created_by': job.created_by,
            'tenant_id': job.tenant_id,
            'creator': {
                'id': creator.id if creator else None,
                'email': creator.email if creator else None,
                'role': creator.role if creator else None,
                'user_type': creator.user_type if creator else None,
                'company_name': creator.company_name if creator else None,
                'created_at': creator.created_at.isoformat() if creator and creator.created_at else None
            } if creator else None,
            'applications': [app.to_dict() for app in applications]
        }
        
        return jsonify({
            'job': job_details,
            'message': 'Job details retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting job details: {e}")
        return jsonify({'error': 'Failed to retrieve job details'}), 500

@admin_bp.route('/jobs/stats', methods=['GET'])
@require_admin_auth
def get_jobs_stats():
    """Get statistics about jobs (admin only)"""
    try:
        # Total jobs
        total_jobs = Job.query.count()
        
        # Jobs by status
        status_stats = db.session.query(
            Job.status, 
            func.count(Job.id).label('count')
        ).group_by(Job.status).all()
        
        # Jobs by creator
        creator_stats = db.session.query(
            User.email,
            func.count(Job.id).label('count')
        ).join(Job, User.id == Job.created_by).group_by(User.email).order_by(desc('count')).limit(10).all()
        
        # Jobs by company
        company_stats = db.session.query(
            Job.company_name,
            func.count(Job.id).label('count')
        ).group_by(Job.company_name).order_by(desc('count')).limit(10).all()
        
        # Recent jobs (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_jobs = Job.query.filter(Job.created_at >= thirty_days_ago).count()
        
        # Jobs with applications
        jobs_with_applications = db.session.query(Job.id).join(JobApplication).distinct().count()
        
        # Average applications per job
        avg_applications = db.session.query(
            func.avg(Job.applications_count)
        ).scalar() or 0
        
        stats = {
            'total_jobs': total_jobs,
            'status_distribution': {status: count for status, count in status_stats},
            'top_creators': [{'email': email, 'count': count} for email, count in creator_stats],
            'top_companies': [{'company': company, 'count': count} for company, count in company_stats],
            'recent_jobs': recent_jobs,
            'jobs_with_applications': jobs_with_applications,
            'average_applications_per_job': round(float(avg_applications), 2)
        }
        
        return jsonify({
            'stats': stats,
            'message': 'Job statistics retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting job stats: {e}")
        return jsonify({'error': 'Failed to retrieve job statistics'}), 500

@admin_bp.route('/jobs/creators', methods=['GET'])
@require_admin_auth
def get_job_creators():
    """Get all users who have created jobs (admin only)"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        search = request.args.get('search', None)
        
        # Get users who have created jobs
        query = db.session.query(User).join(Job, User.id == Job.created_by).distinct()
        
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
        
        creators = []
        for user in pagination.items:
            # Get job statistics for this creator
            jobs_created = Job.query.filter_by(created_by=user.id).count()
            active_jobs = Job.query.filter_by(created_by=user.id, status='active').count()
            total_applications = db.session.query(func.sum(Job.applications_count)).filter_by(created_by=user.id).scalar() or 0
            total_views = db.session.query(func.sum(Job.views_count)).filter_by(created_by=user.id).scalar() or 0
            
            # Get recent jobs
            recent_jobs = Job.query.filter_by(created_by=user.id).order_by(desc(Job.created_at)).limit(5).all()
            
            creator_data = {
                'id': user.id,
                'email': user.email,
                'role': user.role,
                'user_type': user.user_type,
                'company_name': user.company_name,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'tenant_id': user.tenant_id,
                'job_stats': {
                    'jobs_created': jobs_created,
                    'active_jobs': active_jobs,
                    'total_applications': total_applications,
                    'total_views': total_views
                },
                'recent_jobs': [{
                    'id': job.id,
                    'title': job.title,
                    'company_name': job.company_name,
                    'status': job.status,
                    'created_at': job.created_at.isoformat() if job.created_at else None
                } for job in recent_jobs]
            }
            
            creators.append(creator_data)
        
        return jsonify({
            'creators': creators,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            },
            'message': 'Job creators retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting job creators: {e}")
        return jsonify({'error': 'Failed to retrieve job creators'}), 500

@admin_bp.route('/jobs/<int:job_id>/applicants', methods=['GET'])
@require_admin_auth
def get_job_applicants(job_id):
    """Get all applicants for a specific job (admin only)"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        status_filter = request.args.get('status', None)
        search = request.args.get('search', None)
        
        # Check if job exists
        job = Job.query.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Base query for applications
        query = JobApplication.query.filter_by(job_id=job_id)
        
        # Apply status filter
        if status_filter:
            query = query.filter(JobApplication.status == status_filter)
        
        # Apply search filter
        if search:
            search_term = f"%{search}%"
            query = query.join(User, JobApplication.applicant_id == User.id).filter(
                User.email.ilike(search_term)
            )
        
        # Get paginated results
        pagination = query.order_by(desc(JobApplication.applied_at)).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        applicants = []
        for application in pagination.items:
            # Get applicant information
            applicant = User.query.get(application.applicant_id)
            
            # Get applicant profile
            profile = CandidateProfile.query.filter_by(user_id=application.applicant_id).first()
            
            applicant_data = {
                'id': application.id,
                'job_id': application.job_id,
                'job_title': job.title,
                'applicant_id': application.applicant_id,
                'applicant': {
                    'id': applicant.id if applicant else None,
                    'email': applicant.email if applicant else None,
                    'role': applicant.role if applicant else None,
                    'user_type': applicant.user_type if applicant else None,
                    'company_name': applicant.company_name if applicant else None,
                    'created_at': applicant.created_at.isoformat() if applicant and applicant.created_at else None
                } if applicant else None,
                'profile': {
                    'full_name': profile.full_name if profile else None,
                    'phone': profile.phone if profile else None,
                    'location': profile.location if profile else None,
                    'summary': profile.summary if profile else None,
                    'experience_years': profile.experience_years if profile else None
                } if profile else None,
                'application': {
                    'cover_letter': application.cover_letter,
                    'resume_filename': application.resume_filename,
                    'status': application.status,
                    'notes': application.notes,
                    'interview_scheduled': application.interview_scheduled,
                    'interview_date': application.interview_date.isoformat() if application.interview_date else None,
                    'interview_meeting_link': application.interview_meeting_link,
                    'interview_meeting_type': application.interview_meeting_type,
                    'interview_notes': application.interview_notes,
                    'additional_answers': application.additional_answers,
                    'applied_at': application.applied_at.isoformat() if application.applied_at else None,
                    'updated_at': application.updated_at.isoformat() if application.updated_at else None,
                    'reviewed_at': application.reviewed_at.isoformat() if application.reviewed_at else None
                }
            }
            
            applicants.append(applicant_data)
        
        return jsonify({
            'applicants': applicants,
            'job': {
                'id': job.id,
                'title': job.title,
                'company_name': job.company_name,
                'location': job.location,
                'status': job.status
            },
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            },
            'message': 'Job applicants retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting job applicants: {e}")
        return jsonify({'error': 'Failed to retrieve job applicants'}), 500

@admin_bp.route('/jobs/applicants', methods=['GET'])
@require_admin_auth
def list_all_applicants():
    """Get all job applicants across all jobs (admin only)"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        status_filter = request.args.get('status', None)
        job_filter = request.args.get('job_id', None)
        search = request.args.get('search', None)
        
        # Base query for applications
        query = JobApplication.query
        
        # Apply status filter
        if status_filter:
            query = query.filter(JobApplication.status == status_filter)
        
        # Apply job filter
        if job_filter:
            query = query.filter(JobApplication.job_id == job_filter)
        
        # Apply search filter
        if search:
            search_term = f"%{search}%"
            query = query.join(User, JobApplication.applicant_id == User.id).join(Job, JobApplication.job_id == Job.id).filter(
                (User.email.ilike(search_term)) |
                (Job.title.ilike(search_term)) |
                (Job.company_name.ilike(search_term))
            )
        
        # Get paginated results
        pagination = query.order_by(desc(JobApplication.applied_at)).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        applicants = []
        for application in pagination.items:
            # Get applicant and job information
            applicant = User.query.get(application.applicant_id)
            job = Job.query.get(application.job_id)
            profile = CandidateProfile.query.filter_by(user_id=application.applicant_id).first()
            
            applicant_data = {
                'id': application.id,
                'job_id': application.job_id,
                'job': {
                    'id': job.id if job else None,
                    'title': job.title if job else None,
                    'company_name': job.company_name if job else None,
                    'location': job.location if job else None,
                    'status': job.status if job else None
                } if job else None,
                'applicant_id': application.applicant_id,
                'applicant': {
                    'id': applicant.id if applicant else None,
                    'email': applicant.email if applicant else None,
                    'role': applicant.role if applicant else None,
                    'user_type': applicant.user_type if applicant else None,
                    'company_name': applicant.company_name if applicant else None,
                    'created_at': applicant.created_at.isoformat() if applicant and applicant.created_at else None
                } if applicant else None,
                'profile': {
                    'full_name': profile.full_name if profile else None,
                    'phone': profile.phone if profile else None,
                    'location': profile.location if profile else None,
                    'summary': profile.summary if profile else None,
                    'experience_years': profile.experience_years if profile else None
                } if profile else None,
                'application': {
                    'cover_letter': application.cover_letter,
                    'resume_filename': application.resume_filename,
                    'status': application.status,
                    'notes': application.notes,
                    'interview_scheduled': application.interview_scheduled,
                    'interview_date': application.interview_date.isoformat() if application.interview_date else None,
                    'interview_meeting_link': application.interview_meeting_link,
                    'interview_meeting_type': application.interview_meeting_type,
                    'interview_notes': application.interview_notes,
                    'additional_answers': application.additional_answers,
                    'applied_at': application.applied_at.isoformat() if application.applied_at else None,
                    'updated_at': application.updated_at.isoformat() if application.updated_at else None,
                    'reviewed_at': application.reviewed_at.isoformat() if application.reviewed_at else None
                }
            }
            
            applicants.append(applicant_data)
        
        return jsonify({
            'applicants': applicants,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            },
            'message': 'All job applicants retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting all applicants: {e}")
        return jsonify({'error': 'Failed to retrieve applicants'}), 500

@admin_bp.route('/jobs/applicants/stats', methods=['GET'])
@require_admin_auth
def get_applicants_stats():
    """Get statistics about job applicants (admin only)"""
    try:
        # Total applications
        total_applications = JobApplication.query.count()
        
        # Applications by status
        status_stats = db.session.query(
            JobApplication.status, 
            func.count(JobApplication.id).label('count')
        ).group_by(JobApplication.status).all()
        
        # Applications by job
        job_stats = db.session.query(
            Job.title,
            Job.company_name,
            func.count(JobApplication.id).label('count')
        ).join(JobApplication, Job.id == JobApplication.job_id).group_by(Job.id, Job.title, Job.company_name).order_by(desc('count')).limit(10).all()
        
        # Applications by applicant (top applicants)
        applicant_stats = db.session.query(
            User.email,
            func.count(JobApplication.id).label('count')
        ).join(JobApplication, User.id == JobApplication.applicant_id).group_by(User.email).order_by(desc('count')).limit(10).all()
        
        # Recent applications (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_applications = JobApplication.query.filter(JobApplication.applied_at >= thirty_days_ago).count()
        
        # Applications with interviews scheduled
        interview_scheduled = JobApplication.query.filter(JobApplication.interview_scheduled == True).count()
        
        # Average applications per job
        avg_applications = db.session.query(
            func.avg(Job.applications_count)
        ).scalar() or 0
        
        stats = {
            'total_applications': total_applications,
            'status_distribution': {status: count for status, count in status_stats},
            'top_jobs': [{'title': title, 'company': company, 'count': count} for title, company, count in job_stats],
            'top_applicants': [{'email': email, 'count': count} for email, count in applicant_stats],
            'recent_applications': recent_applications,
            'interview_scheduled': interview_scheduled,
            'average_applications_per_job': round(float(avg_applications), 2)
        }
        
        return jsonify({
            'stats': stats,
            'message': 'Applicant statistics retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting applicant stats: {e}")
        return jsonify({'error': 'Failed to retrieve applicant statistics'}), 500

@admin_bp.route('/onboarding-submissions', methods=['GET'])
@require_admin_auth
@log_admin_activity("List Onboarding Submissions")
def list_onboarding_submissions():
    """List all onboarding submissions (admin only)"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        search = request.args.get('search', None)
        
        # Base query for all onboarding submissions
        query = OnboardingSubmission.query
        
        # Apply search filter (search by user email or company name)
        if search:
            search_term = f"%{search}%"
            query = query.join(User, OnboardingSubmission.user_id == User.id).filter(
                (User.email.ilike(search_term)) |
                (User.company_name.ilike(search_term)) |
                (User.first_name.ilike(search_term)) |
                (User.last_name.ilike(search_term))
            )
        
        # Get paginated results
        pagination = query.order_by(desc(OnboardingSubmission.created_at)).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        submissions = []
        for submission in pagination.items:
            user = User.query.get(submission.user_id)
            submission_data = submission.data or {}
            
            # Get user profile if available
            profile = None
            if user:
                profile = CandidateProfile.query.filter_by(user_id=user.id).first()
            
            submissions.append({
                'id': submission.id,
                'user_id': submission.user_id,
                'user': {
                    'id': user.id if user else None,
                    'email': user.email if user else None,
                    'full_name': profile.full_name if profile else None,
                    'company_name': user.company_name if user else None,
                    'role': user.role if user else None
                },
                'contact_name': submission_data.get('contactName') or submission_data.get('full_name'),
                'email': submission_data.get('email') or (user.email if user else None),
                'phone': submission_data.get('phone'),
                'company_name': submission_data.get('companyName') or submission_data.get('company_name') or (user.company_name if user else None),
                'website': submission_data.get('website'),
                'company_size': submission_data.get('companySize'),
                'location': submission_data.get('location') or submission_data.get('address'),
                'tools': submission_data.get('tools', []),
                'goals': submission_data.get('goals') or submission_data.get('business_goals'),
                'services': submission_data.get('services', []),
                'scope': submission_data.get('scope') or submission_data.get('project_scope'),
                'timeline': submission_data.get('timeline'),
                'budget': submission_data.get('budget'),
                'billing_address': submission_data.get('billingAddress'),
                'payment_method': submission_data.get('paymentMethod'),
                'consent_data': submission_data.get('consentData', False),
                'consent_communication': submission_data.get('consentCommunication', False),
                'notes': submission_data.get('notes'),
                'created_at': submission.created_at.isoformat() if submission.created_at else None,
                'updated_at': submission.updated_at.isoformat() if submission.updated_at else None
            })
        
        return jsonify({
            'success': True,
            'data': {
                'submissions': submissions,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': pagination.total,
                    'pages': pagination.pages,
                    'has_next': pagination.has_next,
                    'has_prev': pagination.has_prev
                }
            },
            'message': 'Onboarding submissions retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing onboarding submissions: {e}")
        return jsonify({'error': 'Failed to retrieve onboarding submissions'}), 500

@admin_bp.route('/onboarding-submissions/<int:submission_id>', methods=['GET'])
@require_admin_auth
@log_admin_activity("View Onboarding Submission Details")
def get_onboarding_submission(submission_id):
    """Get a specific onboarding submission by ID (admin only)"""
    try:
        submission = OnboardingSubmission.query.get(submission_id)
        if not submission:
            return jsonify({'error': 'Onboarding submission not found'}), 404
        
        user = User.query.get(submission.user_id)
        submission_data = submission.data or {}
        
        # Get user profile if available
        profile = None
        if user:
            profile = CandidateProfile.query.filter_by(user_id=user.id).first()
        
        # Flatten the data structure for easier frontend access
        submission_details = {
            'id': submission.id,
            'user_id': submission.user_id,
            'user': {
                'id': user.id if user else None,
                'email': user.email if user else None,
                'full_name': profile.full_name if profile else None,
                'company_name': user.company_name if user else None,
                'role': user.role if user else None
            },
            # Flatten submission data to top level
            'contact_name': submission_data.get('contactName') or submission_data.get('full_name'),
            'email': submission_data.get('email') or (user.email if user else None),
            'phone': submission_data.get('phone'),
            'company_name': submission_data.get('companyName') or submission_data.get('company_name') or (user.company_name if user else None),
            'website': submission_data.get('website'),
            'company_size': submission_data.get('companySize'),
            'location': submission_data.get('location') or submission_data.get('address'),
            'tools': submission_data.get('tools', []),
            'goals': submission_data.get('goals') or submission_data.get('business_goals'),
            'services': submission_data.get('services', []),
            'scope': submission_data.get('scope') or submission_data.get('project_scope'),
            'timeline': submission_data.get('timeline'),
            'budget': submission_data.get('budget'),
            'billing_address': submission_data.get('billingAddress'),
            'payment_method': submission_data.get('paymentMethod'),
            'consent_data': submission_data.get('consentData', False),
            'consent_communication': submission_data.get('consentCommunication', False),
            'notes': submission_data.get('notes'),
            'created_at': submission.created_at.isoformat() if submission.created_at else None,
            'updated_at': submission.updated_at.isoformat() if submission.updated_at else None,
            # Also include raw data for reference
            'data': submission_data
        }
        
        return jsonify({
            'success': True,
            'data': submission_details,
            'message': 'Onboarding submission retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting onboarding submission: {e}")
        return jsonify({'error': 'Failed to retrieve onboarding submission'}), 500

# ==================== Module Access Management Routes ====================

@admin_bp.route('/modules/available', methods=['GET'])
@require_admin_auth
@log_admin_activity("List Available Modules")
def list_available_modules():
    """Get list of all available modules in the system (admin only)"""
    try:
        from app.utils.module_role_mapper import MODULE_ROLE_MAPPING
        
        modules = [
            {'name': 'payroll', 'display_name': 'Payroll', 'role': MODULE_ROLE_MAPPING.get('payroll')},
            {'name': 'talent_matchmaker', 'display_name': 'Talent Matchmaker', 'role': MODULE_ROLE_MAPPING.get('talent_matchmaker')},
            {'name': 'jobseeker', 'display_name': 'Job Seeker', 'role': MODULE_ROLE_MAPPING.get('jobseeker')},
            {'name': 'jobs', 'display_name': 'Jobs', 'role': MODULE_ROLE_MAPPING.get('jobs')},
            {'name': 'recruiter', 'display_name': 'Recruiter', 'role': MODULE_ROLE_MAPPING.get('recruiter')},
            {'name': 'employer', 'display_name': 'Employer', 'role': MODULE_ROLE_MAPPING.get('employer')},
            {'name': 'employee', 'display_name': 'Employee', 'role': MODULE_ROLE_MAPPING.get('employee')},
        ]
        
        return jsonify({
            'success': True,
            'modules': modules,
            'message': 'Available modules retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing available modules: {e}")
        return jsonify({'error': 'Failed to retrieve available modules'}), 500

@admin_bp.route('/users/<int:user_id>/modules', methods=['GET'])
@require_admin_auth
@log_admin_activity("Get User Modules")
def get_user_modules(user_id):
    """Get all modules assigned to a user (admin only)"""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get all module access records for the user
        module_access = UserModuleAccess.query.filter_by(user_id=user_id).all()
        
        modules = [access.to_dict() for access in module_access]
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'user_email': user.email,
            'current_role': user.role,
            'modules': modules,
            'message': 'User modules retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting user modules: {e}")
        return jsonify({'error': 'Failed to retrieve user modules'}), 500

@admin_bp.route('/users/<int:user_id>/modules', methods=['POST'])
@require_admin_auth
@log_admin_activity("Assign Module to User", include_request_data=True)
def assign_module_to_user(user_id):
    """Assign a module to a user (admin only)"""
    try:
        from flask import g
        from app.utils.module_role_mapper import update_user_role_from_modules
        
        # Get current admin user
        admin_user = User.query.filter_by(email=g.admin_email).first()
        if not admin_user:
            return jsonify({'error': 'Admin user not found'}), 404
        
        data = request.get_json()
        module_name = data.get('module_name')
        
        if not module_name:
            return jsonify({'error': 'module_name is required'}), 400
        
        # Validate user exists
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Don't allow changing modules for admin/owner
        if user.role in ['admin', 'owner']:
            return jsonify({'error': 'Cannot modify modules for admin/owner users'}), 403
        
        # Check if module access already exists
        existing_access = UserModuleAccess.query.filter_by(
            user_id=user_id,
            module_name=module_name.lower()
        ).first()
        
        if existing_access:
            # Reactivate if inactive
            if not existing_access.is_active:
                existing_access.is_active = True
                existing_access.granted_by = admin_user.id
                existing_access.granted_at = datetime.utcnow()
                existing_access.updated_at = datetime.utcnow()
                db.session.commit()
                
                # Update user role based on modules
                success, new_role, message = update_user_role_from_modules(user_id)
                
                return jsonify({
                    'success': True,
                    'message': f'Module {module_name} reactivated for user',
                    'module': existing_access.to_dict(),
                    'user_role_updated': success,
                    'new_role': new_role,
                    'role_message': message
                }), 200
            else:
                return jsonify({
                    'success': True,
                    'message': f'User already has access to module {module_name}',
                    'module': existing_access.to_dict()
                }), 200
        
        # Create new module access
        new_access = UserModuleAccess(
            user_id=user_id,
            module_name=module_name.lower(),
            granted_by=admin_user.id,
            is_active=True
        )
        
        db.session.add(new_access)
        db.session.commit()
        
        # Update user role based on modules
        success, new_role, message = update_user_role_from_modules(user_id)
        
        return jsonify({
            'success': True,
            'message': f'Module {module_name} assigned to user',
            'module': new_access.to_dict(),
            'user_role_updated': success,
            'new_role': new_role,
            'role_message': message
        }), 201
        
    except Exception as e:
        logger.error(f"Error assigning module to user: {e}")
        db.session.rollback()
        return jsonify({'error': f'Failed to assign module: {str(e)}'}), 500

@admin_bp.route('/users/<int:user_id>/modules', methods=['PUT'])
@require_admin_auth
@log_admin_activity("Update User Modules", include_request_data=True)
def update_user_modules(user_id):
    """Update multiple modules for a user at once (admin only)"""
    try:
        from flask import g
        from app.utils.module_role_mapper import update_user_role_from_modules
        
        # Get current admin user
        admin_user = User.query.filter_by(email=g.admin_email).first()
        if not admin_user:
            return jsonify({'error': 'Admin user not found'}), 404
        
        data = request.get_json()
        module_names = data.get('modules', [])  # List of module names to assign
        
        if not isinstance(module_names, list):
            return jsonify({'error': 'modules must be a list'}), 400
        
        # Validate user exists
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Don't allow changing modules for admin/owner
        if user.role in ['admin', 'owner']:
            return jsonify({'error': 'Cannot modify modules for admin/owner users'}), 403
        
        # Get existing modules
        existing_modules = UserModuleAccess.query.filter_by(user_id=user_id).all()
        existing_module_names = {m.module_name for m in existing_modules}
        new_module_names = {m.lower() for m in module_names}
        
        # Deactivate modules not in the new list
        for existing in existing_modules:
            if existing.module_name not in new_module_names:
                existing.is_active = False
                existing.updated_at = datetime.utcnow()
        
        # Add new modules
        for module_name in new_module_names:
            if module_name not in existing_module_names:
                new_access = UserModuleAccess(
                    user_id=user_id,
                    module_name=module_name,
                    granted_by=admin_user.id,
                    is_active=True
                )
                db.session.add(new_access)
            else:
                # Reactivate if it was inactive
                existing = next(m for m in existing_modules if m.module_name == module_name)
                if not existing.is_active:
                    existing.is_active = True
                    existing.granted_by = admin_user.id
                    existing.granted_at = datetime.utcnow()
                    existing.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # Update user role based on modules
        success, new_role, message = update_user_role_from_modules(user_id)
        
        # Get updated modules
        updated_modules = UserModuleAccess.query.filter_by(user_id=user_id).all()
        
        return jsonify({
            'success': True,
            'message': 'User modules updated successfully',
            'modules': [m.to_dict() for m in updated_modules],
            'user_role_updated': success,
            'new_role': new_role,
            'role_message': message
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating user modules: {e}")
        db.session.rollback()
        return jsonify({'error': f'Failed to update modules: {str(e)}'}), 500

@admin_bp.route('/users/<int:user_id>/modules/<module_name>', methods=['DELETE'])
@require_admin_auth
@log_admin_activity("Remove Module from User", include_request_data=True)
def remove_module_from_user(user_id, module_name):
    """Remove a module from a user (admin only)"""
    try:
        from app.utils.module_role_mapper import update_user_role_from_modules
        
        # Validate user exists
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Don't allow removing modules for admin/owner
        if user.role in ['admin', 'owner']:
            return jsonify({'error': 'Cannot modify modules for admin/owner users'}), 403
        
        # Find module access
        module_access = UserModuleAccess.query.filter_by(
            user_id=user_id,
            module_name=module_name.lower()
        ).first()
        
        if not module_access:
            return jsonify({'error': 'Module access not found'}), 404
        
        # Deactivate instead of deleting (soft delete)
        module_access.is_active = False
        module_access.updated_at = datetime.utcnow()
        db.session.commit()
        
        # Update user role based on remaining modules
        success, new_role, message = update_user_role_from_modules(user_id)
        
        return jsonify({
            'success': True,
            'message': f'Module {module_name} removed from user',
            'user_role_updated': success,
            'new_role': new_role,
            'role_message': message
        }), 200
        
    except Exception as e:
        logger.error(f"Error removing module from user: {e}")
        db.session.rollback()
        return jsonify({'error': f'Failed to remove module: {str(e)}'}), 500
