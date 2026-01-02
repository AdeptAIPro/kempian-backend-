"""
Migration script to add company_name and visa_status fields
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

def run_migration():
    """Run the migration to add company_name and visa_status fields"""
    
    # Database connection - use the same URL as the Flask app
    database_url = os.environ.get('DATABASE_URL', 'mysql+pymysql://localhost:3307/kempianDB')
    engine = create_engine(database_url)
    
    with engine.connect() as connection:
        # Add company_name column to users table
        try:
            connection.execute(text("""
                ALTER TABLE users 
                ADD COLUMN company_name VARCHAR(255) NULL
            """))
            print("✅ Added company_name column to users table")
        except Exception as e:
            print(f"⚠️  company_name column might already exist: {e}")
        
        # Add visa_status column to candidate_profiles table
        try:
            connection.execute(text("""
                ALTER TABLE candidate_profiles 
                ADD COLUMN visa_status VARCHAR(100) NULL
            """))
            print("✅ Added visa_status column to candidate_profiles table")
        except Exception as e:
            print(f"⚠️  visa_status column might already exist: {e}")
        
        connection.commit()
        print("🎉 Migration completed successfully!")

if __name__ == "__main__":
    run_migration()
