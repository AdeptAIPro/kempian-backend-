from datetime import datetime, timedelta, date
from app.simple_logger import get_logger
from app.models import User, UserTrial, Plan, db
import logging

logger = get_logger(__name__.split('.')[-1])

def create_user_trial(user_id):
    """Create a trial for a new user"""
    try:
        # Check if user already has a trial
        existing_trial = UserTrial.query.filter_by(user_id=user_id).first()
        if existing_trial:
            return existing_trial
        
        # Create new trial - 7 days from now
        trial_end_date = datetime.utcnow() + timedelta(days=7)
        trial = UserTrial(
            user_id=user_id,
            trial_end_date=trial_end_date,
            searches_used_today=0,
            last_search_date=date.today(),  # Set today's date explicitly
            is_active=True
        )
        
        db.session.add(trial)
        db.session.commit()
        
        logger.info(f"Created trial for user {user_id}, expires: {trial_end_date}")
        return trial
        
    except Exception as e:
        logger.error(f"Error creating trial for user {user_id}: {str(e)}")
        db.session.rollback()
        return None

def get_user_trial_status(user_id):
    """Get the trial status for a user"""
    try:
        trial = UserTrial.query.filter_by(user_id=user_id).first()
        if not trial:
            return None
        
        # Ensure daily quota is reset if it's a new day
        ensure_daily_quota_reset(trial)
        
        # Log trial data for debugging
        logger.debug(f"Trial data for user {user_id}: {trial.trial_end_date} (type: {type(trial.trial_end_date)})")
        
        # Calculate days remaining safely
        try:
            if trial.trial_end_date:
                days_remaining = max(0, (trial.trial_end_date - datetime.utcnow()).days)
            else:
                days_remaining = 0
        except Exception as e:
            logger.warning(f"Error calculating days remaining for user {user_id}: {str(e)}")
            days_remaining = 0
        
        return {
            'is_active': trial.is_active,
            'is_valid': trial.is_trial_valid(),
            'can_search_today': trial.can_search_today(),
            'trial_start_date': trial.trial_start_date,
            'trial_end_date': trial.trial_end_date,
            'searches_used_today': trial.searches_used_today,
            'last_search_date': trial.last_search_date,
            'days_remaining': days_remaining
        }
        
    except Exception as e:
        logger.error(f"Error getting trial status for user {user_id}: {str(e)}")
        return None

def ensure_daily_quota_reset(trial):
    """Ensure daily quota is reset if it's a new day"""
    try:
        today = date.today()
        if trial.last_search_date != today:
            # New day, reset daily quota
            trial.searches_used_today = 0
            trial.last_search_date = today
            trial.updated_at = datetime.utcnow()
            db.session.commit()
            logger.info(f"Reset daily quota for user {trial.user_id} - new day detected")
    except Exception as e:
        logger.error(f"Error resetting daily quota for user {trial.user_id}: {str(e)}")
        db.session.rollback()

def check_and_increment_trial_search(user_id):
    """Check if user can search and increment the count if they can"""
    try:
        trial = UserTrial.query.filter_by(user_id=user_id).first()
        if not trial:
            return False, "No trial found"
        
        if not trial.is_trial_valid():
            return False, "Trial expired"
        
        # Ensure daily quota is reset before checking
        ensure_daily_quota_reset(trial)
        
        if not trial.can_search_today():
            return False, "Daily search limit reached"
        
        # Increment the search count
        trial.increment_search_count()
        db.session.commit()
        
        return True, f"Search allowed. Used today: {trial.searches_used_today}/5"
        
    except Exception as e:
        logger.error(f"Error checking trial search for user {user_id}: {str(e)}")
        db.session.rollback()
        return False, "Error checking trial status"

def deactivate_user_trial(user_id):
    """Deactivate a user's trial (when they purchase a plan)"""
    try:
        trial = UserTrial.query.filter_by(user_id=user_id).first()
        if trial:
            trial.is_active = False
            db.session.commit()
            logger.info(f"Deactivated trial for user {user_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error deactivating trial for user {user_id}: {str(e)}")
        db.session.rollback()
        return False

def get_free_trial_plan():
    """Get the free trial plan"""
    try:
        return Plan.query.filter_by(is_trial=True).first()
    except Exception as e:
        logger.error(f"Error getting free trial plan: {str(e)}")
        return None

def force_daily_quota_reset():
    """Force reset daily quotas for all active trials (for cron jobs)"""
    try:
        today = date.today()
        active_trials = UserTrial.query.filter_by(is_active=True).all()
        
        reset_count = 0
        for trial in active_trials:
            if trial.last_search_date != today:
                trial.searches_used_today = 0
                trial.last_search_date = today
                trial.updated_at = datetime.utcnow()
                reset_count += 1
        
        if reset_count > 0:
            db.session.commit()
            logger.info(f"Force reset daily quotas for {reset_count} trials")
        
        return reset_count
        
    except Exception as e:
        logger.error(f"Error in force daily quota reset: {str(e)}")
        db.session.rollback()
        return 0 