import logging
from typing import List, Dict, Any
from app.simple_logger import get_logger
from datetime import datetime
from app.models import UnlimitedQuotaUser, db

logger = get_logger(__name__.split('.')[-1])

class ProductionUnlimitedQuotaManager:
    """Production-ready unlimited quota manager using database"""
    
    def is_unlimited_user(self, email: str) -> bool:
        """Check if a user has unlimited quota"""
        if not email:
            return False
        
        try:
            user = UnlimitedQuotaUser.query.filter_by(
                email=email.lower(),
                active=True
            ).first()
            
            if not user:
                return False
            
            # Check if unlimited quota is set
            if user.quota_limit == -1:
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking unlimited quota for {email}: {e}")
            return False
    
    def get_user_quota_info(self, email: str) -> Dict[str, Any]:
        """Get quota information for a user"""
        if not email:
            return {}
        
        try:
            user = UnlimitedQuotaUser.query.filter_by(
                email=email.lower(),
                active=True
            ).first()
            
            if not user:
                return {}
            
            return user.to_dict()
        except Exception as e:
            logger.error(f"Error getting quota info for {email}: {e}")
            return {}
    
    def add_unlimited_user(self, email: str, reason: str, added_by: str, 
                          quota_limit: int = -1, daily_limit: int = -1, 
                          monthly_limit: int = -1, expires: datetime = None) -> bool:
        """Add a new unlimited quota user"""
        try:
            email = email.lower()
            
            # Check if user already exists
            existing_user = UnlimitedQuotaUser.query.filter_by(email=email).first()
            if existing_user:
                logger.warning(f"User {email} already has unlimited quota")
                return False
            
            # Create new user
            new_user = UnlimitedQuotaUser(
                email=email,
                reason=reason,
                quota_limit=quota_limit,
                daily_limit=daily_limit,
                monthly_limit=monthly_limit,
                added_by=added_by,
                added_date=datetime.utcnow(),
                expires=expires,
                active=True
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            logger.info(f"Added unlimited quota for user {email}: {reason}")
            return True
                
        except Exception as e:
            logger.error(f"Error adding unlimited quota user {email}: {e}")
            db.session.rollback()
            return False
    
    def remove_unlimited_user(self, email: str, removed_by: str) -> bool:
        """Remove unlimited quota for a user (hard delete)"""
        try:
            email = email.lower()
            
            user = UnlimitedQuotaUser.query.filter_by(email=email).first()
            if not user:
                logger.warning(f"User {email} doesn't have unlimited quota")
                return False
            
            # Actually delete the record from the database
            db.session.delete(user)
            db.session.commit()
            logger.info(f"Removed unlimited quota for user {email} (deleted by {removed_by})")
            return True
                
        except Exception as e:
            logger.error(f"Error removing unlimited quota user {email}: {e}")
            db.session.rollback()
            return False
    
    def update_user_quota(self, email: str, updates: Dict[str, Any], updated_by: str) -> bool:
        """Update quota settings for a user"""
        try:
            email = email.lower()
            
            user = UnlimitedQuotaUser.query.filter_by(email=email).first()
            if not user:
                logger.warning(f"User {email} doesn't have unlimited quota")
                return False
            
            # Update fields
            allowed_fields = ['quota_limit', 'daily_limit', 'monthly_limit', 'reason', 'expires', 'active']
            for field in allowed_fields:
                if field in updates:
                    setattr(user, field, updates[field])
            
            # Add update metadata
            user.updated_by = updated_by
            user.updated_date = datetime.utcnow()
            
            db.session.commit()
            logger.info(f"Updated quota for user {email}")
            return True
                
        except Exception as e:
            logger.error(f"Error updating quota for user {email}: {e}")
            db.session.rollback()
            return False
    
    def list_unlimited_users(self) -> List[Dict[str, Any]]:
        """List all active unlimited quota users"""
        try:
            # Only return active users
            users = UnlimitedQuotaUser.query.filter_by(active=True).all()
            return [user.to_dict() for user in users]
        except Exception as e:
            logger.error(f"Error listing unlimited quota users: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about unlimited quota users"""
        try:
            total_users = UnlimitedQuotaUser.query.count()
            active_users = UnlimitedQuotaUser.query.filter_by(active=True).count()
            inactive_users = total_users - active_users
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'inactive_users': inactive_users,
                'unlimited_quota_users': active_users
            }
        except Exception as e:
            logger.error(f"Error getting unlimited quota stats: {e}")
            return {}

# Global instance
production_unlimited_quota_manager = ProductionUnlimitedQuotaManager()

def is_unlimited_quota_user(email: str) -> bool:
    """Check if a user has unlimited quota (production function)"""
    return production_unlimited_quota_manager.is_unlimited_user(email)

def get_unlimited_quota_info(email: str) -> Dict[str, Any]:
    """Get unlimited quota info for a user (production function)"""
    return production_unlimited_quota_manager.get_user_quota_info(email)
