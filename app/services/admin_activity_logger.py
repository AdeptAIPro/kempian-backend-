"""
Admin Activity Logging Service
Handles logging of admin activities, logins, and actions
"""

import uuid
import json
import time
from datetime import datetime, timedelta
from flask import request, g
from app import db
from app.models import AdminActivityLog, AdminSession
from app.simple_logger import get_logger

logger = get_logger("admin_activity")

class AdminActivityLogger:
    """Service for logging admin activities and sessions"""
    
    @staticmethod
    def log_admin_login(admin_email, admin_id, admin_role, tenant_id=None):
        """Log admin login activity"""
        try:
            session_id = str(uuid.uuid4())
            ip_address = request.remote_addr if request else None
            user_agent = request.headers.get('User-Agent') if request else None
            
            # Create new session
            session = AdminSession(
                session_id=session_id,
                admin_email=admin_email,
                admin_id=admin_id,
                admin_role=admin_role,
                ip_address=ip_address,
                user_agent=user_agent,
                tenant_id=tenant_id
            )
            db.session.add(session)
            
            # Log login activity
            activity = AdminActivityLog(
                admin_email=admin_email,
                admin_id=admin_id,
                admin_role=admin_role,
                activity_type='login',
                action='Admin login',
                endpoint=request.endpoint if request else None,
                method=request.method if request else None,
                ip_address=ip_address,
                user_agent=user_agent,
                tenant_id=tenant_id,
                session_id=session_id,
                status_code=200
            )
            db.session.add(activity)
            db.session.commit()
            
            # Store session ID in Flask g for tracking
            g.admin_session_id = session_id
            
            logger.info(f"Admin login logged: {admin_email} (session: {session_id})")
            return session_id
            
        except Exception as e:
            logger.error(f"Error logging admin login: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def log_admin_logout(admin_email, session_id=None):
        """Log admin logout activity"""
        try:
            if not session_id:
                session_id = getattr(g, 'admin_session_id', None)
            
            if session_id:
                # Update session
                session = AdminSession.query.filter_by(session_id=session_id).first()
                if session:
                    session.is_active = False
                    session.logout_time = datetime.utcnow()
                    db.session.commit()
                
                # Log logout activity
                activity = AdminActivityLog(
                    admin_email=admin_email,
                    admin_id=session.admin_id if session else None,
                    admin_role=session.admin_role if session else 'admin',
                    activity_type='logout',
                    action='Admin logout',
                    endpoint=request.endpoint if request else None,
                    method=request.method if request else None,
                    ip_address=request.remote_addr if request else None,
                    user_agent=request.headers.get('User-Agent') if request else None,
                    tenant_id=session.tenant_id if session else None,
                    session_id=session_id,
                    status_code=200
                )
                db.session.add(activity)
                db.session.commit()
                
                logger.info(f"Admin logout logged: {admin_email} (session: {session_id})")
            
        except Exception as e:
            logger.error(f"Error logging admin logout: {e}")
            db.session.rollback()
    
    @staticmethod
    def log_admin_action(admin_email, admin_id, admin_role, action, endpoint=None, method=None, 
                        request_data=None, status_code=200, response_time_ms=None, 
                        error_message=None, tenant_id=None):
        """Log admin action activity"""
        try:
            session_id = getattr(g, 'admin_session_id', None)
            
            # Update last activity for session
            if session_id:
                session = AdminSession.query.filter_by(session_id=session_id).first()
                if session:
                    session.last_activity = datetime.utcnow()
                    db.session.commit()
            
            # Log the activity
            activity = AdminActivityLog(
                admin_email=admin_email,
                admin_id=admin_id,
                admin_role=admin_role,
                activity_type='action',
                action=action,
                endpoint=endpoint or (request.endpoint if request else None),
                method=method or (request.method if request else None),
                ip_address=request.remote_addr if request else None,
                user_agent=request.headers.get('User-Agent') if request else None,
                request_data=json.dumps(request_data) if request_data else None,
                status_code=status_code,
                response_time_ms=response_time_ms,
                tenant_id=tenant_id,
                session_id=session_id,
                error_message=error_message
            )
            db.session.add(activity)
            db.session.commit()
            
            logger.info(f"Admin action logged: {admin_email} - {action}")
            
        except Exception as e:
            logger.error(f"Error logging admin action: {e}")
            db.session.rollback()
    
    @staticmethod
    def get_admin_activities(admin_email=None, activity_type=None, start_date=None, 
                           end_date=None, page=1, per_page=50):
        """Get admin activities with filtering and pagination"""
        try:
            query = AdminActivityLog.query
            
            # Apply filters
            if admin_email:
                query = query.filter(AdminActivityLog.admin_email == admin_email)
            if activity_type:
                query = query.filter(AdminActivityLog.activity_type == activity_type)
            if start_date:
                query = query.filter(AdminActivityLog.created_at >= start_date)
            if end_date:
                query = query.filter(AdminActivityLog.created_at <= end_date)
            
            # Order by created_at desc
            query = query.order_by(AdminActivityLog.created_at.desc())
            
            # Paginate
            pagination = query.paginate(
                page=page, per_page=per_page, error_out=False
            )
            
            return {
                'activities': [activity.to_dict() for activity in pagination.items],
                'total': pagination.total,
                'pages': pagination.pages,
                'current_page': pagination.page,
                'per_page': pagination.per_page,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
            
        except Exception as e:
            logger.error(f"Error getting admin activities: {e}")
            return None
    
    @staticmethod
    def get_admin_sessions(admin_email=None, is_active=None, start_date=None, 
                          end_date=None, page=1, per_page=50):
        """Get admin sessions with filtering and pagination"""
        try:
            query = AdminSession.query
            
            # Apply filters
            if admin_email:
                query = query.filter(AdminSession.admin_email == admin_email)
            if is_active is not None:
                query = query.filter(AdminSession.is_active == is_active)
            if start_date:
                query = query.filter(AdminSession.login_time >= start_date)
            if end_date:
                query = query.filter(AdminSession.login_time <= end_date)
            
            # Order by login_time desc
            query = query.order_by(AdminSession.login_time.desc())
            
            # Paginate
            pagination = query.paginate(
                page=page, per_page=per_page, error_out=False
            )
            
            return {
                'sessions': [session.to_dict() for session in pagination.items],
                'total': pagination.total,
                'pages': pagination.pages,
                'current_page': pagination.page,
                'per_page': pagination.per_page,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
            
        except Exception as e:
            logger.error(f"Error getting admin sessions: {e}")
            return None
    
    @staticmethod
    def get_admin_stats(admin_email=None, days=30):
        """Get admin activity statistics"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            query = AdminActivityLog.query.filter(
                AdminActivityLog.created_at >= start_date,
                AdminActivityLog.created_at <= end_date
            )
            
            if admin_email:
                query = query.filter(AdminActivityLog.admin_email == admin_email)
            
            # Get activity counts by type
            activity_counts = db.session.query(
                AdminActivityLog.activity_type,
                db.func.count(AdminActivityLog.id).label('count')
            ).filter(
                AdminActivityLog.created_at >= start_date,
                AdminActivityLog.created_at <= end_date
            )
            
            if admin_email:
                activity_counts = activity_counts.filter(AdminActivityLog.admin_email == admin_email)
            
            activity_counts = activity_counts.group_by(AdminActivityLog.activity_type).all()
            
            # Get unique admin count
            unique_admins = query.with_entities(AdminActivityLog.admin_email).distinct().count()
            
            # Get total activities
            total_activities = query.count()
            
            # Get recent activities (last 10)
            recent_activities = query.order_by(AdminActivityLog.created_at.desc()).limit(10).all()
            
            return {
                'period_days': days,
                'total_activities': total_activities,
                'unique_admins': unique_admins,
                'activity_counts': {item[0]: item[1] for item in activity_counts},
                'recent_activities': [activity.to_dict() for activity in recent_activities]
            }
            
        except Exception as e:
            logger.error(f"Error getting admin stats: {e}")
            return None
    
    @staticmethod
    def cleanup_old_logs(days_to_keep=90):
        """Clean up old admin activity logs"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old activity logs
            deleted_activities = AdminActivityLog.query.filter(
                AdminActivityLog.created_at < cutoff_date
            ).delete()
            
            # Delete old inactive sessions
            deleted_sessions = AdminSession.query.filter(
                AdminSession.is_active == False,
                AdminSession.logout_time < cutoff_date
            ).delete()
            
            db.session.commit()
            
            logger.info(f"Cleaned up {deleted_activities} old activity logs and {deleted_sessions} old sessions")
            return deleted_activities + deleted_sessions
            
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
            db.session.rollback()
            return 0
