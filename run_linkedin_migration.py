#!/usr/bin/env python
"""
Migration script to create linkedin_integrations table
Run this once to set up the database table
"""
from app.db import db
from app import create_app
from app.config import Config
from sqlalchemy import text

def create_linkedin_table():
    """Create linkedin_integrations table if it doesn't exist"""
    try:
        app = create_app()
        
        with app.app_context():
            # Connect to database using SQLAlchemy
            with db.engine.connect() as connection:
                # Create table
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS linkedin_integrations (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        organization_id VARCHAR(255) NOT NULL,
                        access_token TEXT NOT NULL,
                        created_by VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        UNIQUE KEY unique_org (organization_id),
                        INDEX idx_org_id (organization_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """))
                
                connection.commit()
                print("✅ Successfully created linkedin_integrations table")
        
    except Exception as e:
        print(f"❌ Error creating linkedin_integrations table: {e}")
        raise

if __name__ == "__main__":
    create_linkedin_table()

