from app import create_app
from app.models import db, User
from sqlalchemy import text

app = create_app()

with app.app_context():
    # Add user_type column if it doesn't exist
    try:
        db.session.execute(text("ALTER TABLE users ADD COLUMN user_type VARCHAR(50)"))
        print("Added user_type column to users table")
    except Exception as e:
        print(f"user_type column might already exist: {e}")
    
    # Update role enum to include new roles
    try:
        # For MySQL, we need to modify the enum
        db.session.execute(text("ALTER TABLE users MODIFY COLUMN role ENUM('owner', 'subuser', 'job_seeker', 'employee', 'recruiter', 'employer', 'admin') NOT NULL"))
        print("Updated role enum to include new roles")
    except Exception as e:
        print(f"Role enum update might have failed: {e}")
    
    db.session.commit()
    print("Migration completed successfully!") 