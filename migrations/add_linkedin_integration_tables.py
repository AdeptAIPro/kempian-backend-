"""
Migration: Add LinkedIn Integration Tables
Created: 2024
Description: Adds tables for LinkedIn OAuth integration and candidate import functionality
"""

import os
import sys
from sqlalchemy import text

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.db import db

def upgrade():
    """Add LinkedIn integration tables"""
    app = create_app()
    
    with app.app_context():
        try:
            # Create linkedin_integrations table
            db.engine.execute(text("""
                CREATE TABLE IF NOT EXISTS linkedin_integrations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    organization_id VARCHAR(255) NOT NULL,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT,
                    expires_at TIMESTAMP NULL,
                    created_by VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_org_linkedin (organization_id),
                    INDEX idx_organization (organization_id),
                    INDEX idx_created_by (created_by)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """))
            
            print("✅ Created linkedin_integrations table")
            
            # Check if candidates table exists, if not create it
            result = db.engine.execute(text("""
                SELECT COUNT(*) as count 
                FROM information_schema.tables 
                WHERE table_schema = DATABASE() 
                AND table_name = 'candidates'
            """))
            
            table_exists = result.fetchone()[0] > 0
            
            if not table_exists:
                # Create candidates table for LinkedIn imports
                db.engine.execute(text("""
                    CREATE TABLE IF NOT EXISTS candidates (
                        id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
                        organization_id VARCHAR(36) NOT NULL,
                        candidate_id VARCHAR(50) NOT NULL,
                        first_name VARCHAR(100) NOT NULL,
                        last_name VARCHAR(100) NOT NULL,
                        email VARCHAR(255) NOT NULL,
                        phone VARCHAR(20),
                        current_location VARCHAR(255),
                        experience_years INT DEFAULT 0,
                        skills JSON,
                        linkedin_url VARCHAR(500),
                        source VARCHAR(50) DEFAULT 'linkedin',
                        status VARCHAR(20) DEFAULT 'active',
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        UNIQUE KEY unique_candidate_org (organization_id, candidate_id),
                        INDEX idx_organization (organization_id),
                        INDEX idx_email (email),
                        INDEX idx_status (status),
                        INDEX idx_source (source)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """))
                
                print("✅ Created candidates table")
            else:
                print("ℹ️  candidates table already exists")
            
            # Add LinkedIn-specific columns to existing candidates table if they don't exist
            try:
                db.engine.execute(text("""
                    ALTER TABLE candidates 
                    ADD COLUMN IF NOT EXISTS linkedin_url VARCHAR(500),
                    ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'linkedin'
                """))
                print("✅ Added LinkedIn-specific columns to candidates table")
            except Exception as e:
                print(f"ℹ️  LinkedIn columns may already exist: {e}")
            
            print("🎉 LinkedIn integration tables migration completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during migration: {e}")
            raise

def downgrade():
    """Remove LinkedIn integration tables"""
    app = create_app()
    
    with app.app_context():
        try:
            # Drop linkedin_integrations table
            db.engine.execute(text("DROP TABLE IF EXISTS linkedin_integrations"))
            print("✅ Dropped linkedin_integrations table")
            
            # Note: We don't drop the candidates table as it might contain other data
            # If you want to remove LinkedIn-specific data, you can run:
            # DELETE FROM candidates WHERE source = 'linkedin';
            
            print("🎉 LinkedIn integration tables rollback completed!")
            
        except Exception as e:
            print(f"❌ Error during rollback: {e}")
            raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "downgrade":
        downgrade()
    else:
        upgrade()
