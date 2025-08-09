from datetime import datetime, timedelta
from app.models import User, UserTrial, Plan, db
import logging

logger = logging.getLogger(__name__)

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
        
        return {
            'is_active': trial.is_active,
            'is_valid': trial.is_trial_valid(),
            'can_search_today': trial.can_search_today(),
            'trial_start_date': trial.trial_start_date,
            'trial_end_date': trial.trial_end_date,
            'searches_used_today': trial.searches_used_today,
            'last_search_date': trial.last_search_date,
            'days_remaining': max(0, (trial.trial_end_date - datetime.utcnow()).days)
        }
        
    except Exception as e:
        logger.error(f"Error getting trial status for user {user_id}: {str(e)}")
        return None

def check_and_increment_trial_search(user_id):
    """Check if user can search and increment the count if they can"""
    try:
        trial = UserTrial.query.filter_by(user_id=user_id).first()
        if not trial:
            return False, "No trial found"
        
        if not trial.is_trial_valid():
            return False, "Trial expired"
        
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