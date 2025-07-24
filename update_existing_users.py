from app import create_app
from app.models import db, User
from sqlalchemy import text

app = create_app()

with app.app_context():
    # Get all users that don't have a user_type set
    users_without_type = User.query.filter(User.user_type.is_(None)).all()
    
    print(f"Found {len(users_without_type)} users without user_type")
    
    for user in users_without_type:
        # Set default user_type based on role
        if user.role == 'owner':
            # For existing owners, set as 'employer' (most likely scenario)
            user.user_type = 'employer'
            print(f"Updated user {user.email} (role: {user.role}) -> user_type: employer")
        elif user.role == 'subuser':
            # For existing subusers, set as 'employee'
            user.user_type = 'employee'
            print(f"Updated user {user.email} (role: {user.role}) -> user_type: employee")
        else:
            # For any other roles, set as 'job_seeker'
            user.user_type = 'job_seeker'
            print(f"Updated user {user.email} (role: {user.role}) -> user_type: job_seeker")
    
    # Commit the changes
    db.session.commit()
    print("Successfully updated existing users with user_type!")
    
    # Show summary
    all_users = User.query.all()
    print(f"\nSummary of all users:")
    for user in all_users:
        print(f"- {user.email}: role={user.role}, user_type={user.user_type}") 