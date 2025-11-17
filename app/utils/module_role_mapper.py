"""
Utility functions for mapping modules to user roles
"""
from app.models import User, UserModuleAccess, db
from app.simple_logger import get_logger

logger = get_logger("module_role_mapper")

# Module to role mapping
# This defines which role a user should have based on the modules they have access to
MODULE_ROLE_MAPPING = {
    'payroll': 'employee',  # Payroll access typically means employee role
    'talent_matchmaker': 'recruiter',  # Talent matching is for recruiters
    'jobseeker': 'job_seeker',  # Job seeker module for job seekers
    'jobs': 'employer',  # Job posting module for employers
    'recruiter': 'recruiter',  # Recruiter module
    'employer': 'employer',  # Employer module
    'employee': 'employee',  # Employee module
}

# Priority order for role assignment when user has multiple modules
# Higher priority roles come first
ROLE_PRIORITY = ['admin', 'owner', 'employer', 'recruiter', 'employee', 'job_seeker', 'subuser']

def get_role_from_modules(module_names):
    """
    Determine user role based on assigned modules
    
    Args:
        module_names: List of module names the user has access to
        
    Returns:
        str: The role that should be assigned to the user
    """
    if not module_names:
        return 'subuser'  # Default role if no modules assigned
    
    # Get roles for each module
    roles = []
    for module in module_names:
        role = MODULE_ROLE_MAPPING.get(module.lower())
        if role:
            roles.append(role)
    
    if not roles:
        return 'subuser'  # Default if no valid role mapping found
    
    # If user has multiple roles, return the highest priority one
    # Sort by priority (higher priority first)
    roles_sorted = sorted(roles, key=lambda r: ROLE_PRIORITY.index(r) if r in ROLE_PRIORITY else len(ROLE_PRIORITY))
    return roles_sorted[0]

def update_user_role_from_modules(user_id):
    """
    Update a user's role based on their assigned modules
    
    Args:
        user_id: The ID of the user to update
        
    Returns:
        tuple: (success: bool, new_role: str, message: str)
    """
    try:
        user = User.query.get(user_id)
        if not user:
            return False, None, "User not found"
        
        # Don't change admin or owner roles
        if user.role in ['admin', 'owner']:
            return True, user.role, "Admin/Owner role cannot be changed"
        
        # Get all active modules for the user
        active_modules = UserModuleAccess.query.filter_by(
            user_id=user_id,
            is_active=True
        ).all()
        
        module_names = [access.module_name for access in active_modules]
        
        # Determine new role
        new_role = get_role_from_modules(module_names)
        
        # Update user role if it changed
        if user.role != new_role:
            old_role = user.role
            user.role = new_role
            user.user_type = new_role  # Also update user_type for consistency
            db.session.commit()
            
            # Update Cognito role as well
            try:
                from app.auth.cognito import cognito_admin_update_user_attributes
                cognito_admin_update_user_attributes(user.email, {
                    "custom:role": new_role,
                    "custom:user_type": new_role
                })
                logger.info(f"Updated Cognito role for user {user.email} to {new_role}")
            except Exception as e:
                logger.warning(f"Failed to update Cognito role for user {user.email}: {e}")
                # Don't fail the whole operation if Cognito update fails
            
            logger.info(f"Updated user {user_id} role from {old_role} to {new_role} based on modules: {module_names}")
            return True, new_role, f"Role updated from {old_role} to {new_role}"
        else:
            return True, user.role, "Role unchanged"
            
    except Exception as e:
        logger.error(f"Error updating user role from modules: {e}")
        db.session.rollback()
        return False, None, f"Error updating role: {str(e)}"

def get_user_modules(user_id):
    """
    Get all active modules for a user
    
    Args:
        user_id: The ID of the user
        
    Returns:
        list: List of module names the user has access to
    """
    try:
        active_modules = UserModuleAccess.query.filter_by(
            user_id=user_id,
            is_active=True
        ).all()
        
        return [access.module_name for access in active_modules]
    except Exception as e:
        logger.error(f"Error getting user modules: {e}")
        return []

def has_module_access(user_id, module_name):
    """
    Check if a user has access to a specific module
    
    Args:
        user_id: The ID of the user
        module_name: The name of the module to check
        
    Returns:
        bool: True if user has access, False otherwise
    """
    try:
        access = UserModuleAccess.query.filter_by(
            user_id=user_id,
            module_name=module_name.lower(),
            is_active=True
        ).first()
        
        return access is not None
    except Exception as e:
        logger.error(f"Error checking module access: {e}")
        return False

