"""
Database migration script for candidate match logs table
Run this script to create the candidate_match_logs table and update candidate_search_results table

This script will:
1. Add match_reasons column to candidate_search_results table (if it doesn't exist)
2. Create candidate_match_logs table for long-term storage of match logs

Usage:
    python backend/migrations/create_candidate_match_logs_table.py
    OR
    cd backend && python migrations/create_candidate_match_logs_table.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from app.models import CandidateMatchLog, CandidateSearchResult
from sqlalchemy import inspect, text
from app.simple_logger import get_logger

logger = get_logger('migration')

def add_match_reasons_column():
    """Add match_reasons column to candidate_search_results table if it doesn't exist"""
    try:
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('candidate_search_results')]
        
        if 'match_reasons' not in columns:
            logger.info("Adding match_reasons column to candidate_search_results table...")
            
            # Get database dialect
            dialect = db.engine.dialect.name
            
            if dialect == 'mysql':
                db.session.execute(text(
                    "ALTER TABLE candidate_search_results "
                    "ADD COLUMN match_reasons TEXT NULL AFTER match_score"
                ))
            elif dialect == 'postgresql':
                db.session.execute(text(
                    "ALTER TABLE candidate_search_results "
                    "ADD COLUMN match_reasons TEXT"
                ))
            else:  # SQLite or other
                db.session.execute(text(
                    "ALTER TABLE candidate_search_results "
                    "ADD COLUMN match_reasons TEXT"
                ))
            
            db.session.commit()
            logger.info("✅ Successfully added match_reasons column to candidate_search_results")
            return True
        else:
            logger.info("ℹ️  match_reasons column already exists in candidate_search_results")
            return False
    except Exception as e:
        db.session.rollback()
        logger.warning(f"Could not add match_reasons column (may already exist): {str(e)}")
        return False

def create_candidate_match_logs_table():
    """Create candidate_match_logs table"""
    app = create_app()
    
    with app.app_context():
        try:
            logger.info("=" * 60)
            logger.info("Creating candidate match logs table...")
            logger.info("=" * 60)
            
            # First, add match_reasons column to existing table
            add_match_reasons_column()
            
            # Import model to register it with SQLAlchemy
            from app.models import CandidateMatchLog
            
            # Check if table already exists
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'candidate_match_logs' in tables:
                logger.info("ℹ️  candidate_match_logs table already exists")
                logger.info("   Verifying table structure...")
                
                # Verify columns
                columns = inspector.get_columns('candidate_match_logs')
                column_names = [col['name'] for col in columns]
                expected_columns = [
                    'id', 'search_history_id', 'candidate_result_id', 'tenant_id', 'user_id',
                    'candidate_id', 'candidate_name', 'candidate_email', 'job_description',
                    'search_query', 'search_criteria', 'match_score', 'match_reasons',
                    'match_explanation', 'match_details', 'algorithm_version',
                    'search_duration_ms', 'created_at'
                ]
                
                missing_columns = [col for col in expected_columns if col not in column_names]
                if missing_columns:
                    logger.warning(f"⚠️  Missing columns: {', '.join(missing_columns)}")
                    logger.info("   Recreating table...")
                    # Drop and recreate
                    db.session.execute(text("DROP TABLE IF EXISTS candidate_match_logs"))
                    db.session.commit()
                else:
                    logger.info("✅ Table structure is correct")
                    return True
            
            # Create the table
            logger.info("Creating candidate_match_logs table...")
            CandidateMatchLog.__table__.create(db.engine, checkfirst=True)
            
            # Verify table was created
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'candidate_match_logs' in tables:
                logger.info("✅ Successfully created candidate_match_logs table")
                
                # Show table structure
                columns = inspector.get_columns('candidate_match_logs')
                logger.info("\nTable structure:")
                for column in columns:
                    nullable = "NULL" if column['nullable'] else "NOT NULL"
                    logger.info(f"  - {column['name']}: {str(column['type'])} {nullable}")
                
                logger.info("\n" + "=" * 60)
                logger.info("✅ Migration completed successfully!")
                logger.info("=" * 60)
                return True
            else:
                logger.error("❌ Table candidate_match_logs was not created")
                return False
                
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Error creating table: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    create_candidate_match_logs_table()

