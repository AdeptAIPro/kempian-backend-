#!/usr/bin/env python3
"""
Daily Quota Reset Cron Job
Automatically resets daily search quotas for all users at midnight
"""

import logging
from datetime import datetime, date, timedelta
from app.simple_logger import get_logger
from sqlalchemy import text
from app import create_app
from app.models import db, UserTrial

logger = get_logger(__name__.split('.')[-1])

def reset_daily_quotas():
    """Reset daily search quotas for all users"""
    try:
        app = create_app()
        with app.app_context():
            today = date.today()
            
            # Reset daily search counts for all active trials
            updated_trials = db.session.execute(
                text("""
                    UPDATE user_trials 
                    SET searches_used_today = 0, 
                        last_search_date = :today,
                        updated_at = :now
                    WHERE is_active = true 
                    AND (last_search_date != :today OR last_search_date IS NULL)
                """),
                {
                    'today': today,
                    'now': datetime.utcnow()
                }
            )
            
            db.session.commit()
            
            logger.info(f"✅ Daily quota reset completed. Updated {updated_trials.rowcount} trials.")
            return updated_trials.rowcount
            
    except Exception as e:
        logger.error(f"❌ Error resetting daily quotas: {str(e)}")
        if 'db' in locals():
            db.session.rollback()
        return 0

def reset_monthly_quotas():
    """Reset monthly search quotas for all tenants - PRESERVES ANALYTICS DATA"""
    try:
        app = create_app()
        with app.app_context():
            # FIXED: Only clean up search logs older than 6 months to preserve analytics
            # This ensures we keep enough data for meaningful analytics while cleaning up old data
            six_months_ago = datetime.utcnow() - timedelta(days=180)
            
            # Check how many logs would be deleted
            logs_to_delete = db.session.execute(
                text("""
                    SELECT COUNT(*) FROM jd_search_logs 
                    WHERE searched_at < :six_months_ago
                """),
                {'six_months_ago': six_months_ago}
            ).scalar() or 0
            
            if logs_to_delete > 0:
                # Only delete if there are actually old logs
                deleted_logs = db.session.execute(
                    text("""
                        DELETE FROM jd_search_logs 
                        WHERE searched_at < :six_months_ago
                    """),
                    {'six_months_ago': six_months_ago}
                )
                
                db.session.commit()
                
                logger.info(f"✅ Monthly cleanup completed. Deleted {deleted_logs.rowcount} old search logs (older than 6 months).")
                logger.info(f"📊 Preserved search logs for analytics: {logs_to_delete} logs kept for last 6 months.")
                return deleted_logs.rowcount
            else:
                logger.info("ℹ️ No old search logs to clean up. All logs are within 6 months.")
                return 0
            
    except Exception as e:
        logger.error(f"❌ Error cleaning monthly quotas: {str(e)}")
        if 'db' in locals():
            db.session.rollback()
        return 0

def preserve_analytics_data():
    """Ensure we preserve enough data for analytics"""
    try:
        app = create_app()
        with app.app_context():
            # Count total search logs
            total_logs = db.session.execute(
                text("SELECT COUNT(*) FROM jd_search_logs")
            ).scalar() or 0
            
            # Count logs from last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_logs = db.session.execute(
                text("""
                    SELECT COUNT(*) FROM jd_search_logs 
                    WHERE searched_at >= :thirty_days_ago
                """),
                {'thirty_days_ago': thirty_days_ago}
            ).scalar() or 0
            
            logger.info(f"📊 Analytics Data Status:")
            logger.info(f"   Total search logs: {total_logs}")
            logger.info(f"   Logs from last 30 days: {recent_logs}")
            logger.info(f"   Data preservation: {'✅ Good' if total_logs > 0 and recent_logs > 0 else '⚠️ Low'}")
            
            return {
                'total_logs': total_logs,
                'recent_logs': recent_logs,
                'status': 'good' if total_logs > 0 and recent_logs > 0 else 'low'
            }
            
    except Exception as e:
        logger.error(f"❌ Error checking analytics data: {str(e)}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("🔄 Starting daily quota reset...")
    
    # Check analytics data before reset
    analytics_status = preserve_analytics_data()
    
    # Run daily quota reset
    daily_reset_count = reset_daily_quotas()
    
    # Run monthly cleanup (preserves analytics data)
    monthly_cleanup_count = reset_monthly_quotas()
    
    # Check analytics data after reset
    final_analytics_status = preserve_analytics_data()
    
    logger.info(f"✅ Quota reset completed: {daily_reset_count} daily resets, {monthly_cleanup_count} monthly cleanups")
    
    if analytics_status and final_analytics_status:
        if final_analytics_status['total_logs'] < analytics_status['total_logs']:
            logger.warning(f"⚠️ Search logs decreased from {analytics_status['total_logs']} to {final_analytics_status['total_logs']}")
        else:
            logger.info(f"✅ Analytics data preserved: {final_analytics_status['total_logs']} total logs")
