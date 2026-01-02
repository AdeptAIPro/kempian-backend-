"""
Database migration script for job_suggested_candidates table
Run this script to create the job_suggested_candidates table

This script will:
1. Create job_suggested_candidates table for storing top 3 suggested candidates per job

Usage:
    python backend/migrations/create_job_suggested_candidates_table.py
    OR
    cd backend && python migrations/create_job_suggested_candidates_table.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from app.models import JobSuggestedCandidates
from sqlalchemy import inspect, text
from app.simple_logger import get_logger

logger = get_logger('migration')

def create_job_suggested_candidates_table():
    """Create job_suggested_candidates table"""
    app = create_app()
    
    with app.app_context():
        try:
            logger.info("=" * 60)
            logger.info("Creating job_suggested_candidates table...")
            logger.info("=" * 60)
            
            # Import model to register it with SQLAlchemy
            from app.models import JobSuggestedCandidates
            
            # Check if table already exists
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'job_suggested_candidates' in tables:
                logger.info("ℹ️  job_suggested_candidates table already exists")
                logger.info("   Verifying table structure...")
                
                # Verify columns
                columns = inspector.get_columns('job_suggested_candidates')
                column_names = [col['name'] for col in columns]
                expected_columns = [
                    'id', 'job_id', 'candidates_data', 'algorithm_used',
                    'generated_at', 'updated_at'
                ]
                
                missing_columns = [col for col in expected_columns if col not in column_names]
                if missing_columns:
                    logger.warning(f"⚠️  Missing columns: {', '.join(missing_columns)}")
                    logger.info("   Table structure incomplete. Please check manually.")
                    return False
                else:
                    logger.info("✅ Table structure is correct")
                    
                    # Check for unique constraint on job_id
                    try:
                        indexes = inspector.get_indexes('job_suggested_candidates')
                        unique_indexes = [idx for idx in indexes if idx.get('unique', False)]
                        has_job_id_unique = any(
                            'job_id' in idx.get('column_names', []) for idx in unique_indexes
                        )
                        if not has_job_id_unique:
                            logger.warning("⚠️  Unique constraint on job_id may be missing")
                        else:
                            logger.info("✅ Unique constraint on job_id verified")
                    except Exception as e:
                        logger.warning(f"Could not verify unique constraint: {str(e)}")
                    
                    return True
            
            # Create the table
            logger.info("Creating job_suggested_candidates table...")
            JobSuggestedCandidates.__table__.create(db.engine, checkfirst=True)
            
            # Verify table was created
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'job_suggested_candidates' in tables:
                logger.info("✅ Successfully created job_suggested_candidates table")
                
                # Show table structure
                columns = inspector.get_columns('job_suggested_candidates')
                logger.info("\nTable structure:")
                for column in columns:
                    nullable = "NULL" if column['nullable'] else "NOT NULL"
                    default = f" DEFAULT {column.get('default', '')}" if column.get('default') else ""
                    logger.info(f"  - {column['name']}: {str(column['type'])} {nullable}{default}")
                
                # Show indexes
                try:
                    indexes = inspector.get_indexes('job_suggested_candidates')
                    if indexes:
                        logger.info("\nIndexes:")
                        for index in indexes:
                            unique = "UNIQUE " if index.get('unique', False) else ""
                            cols = ', '.join(index.get('column_names', []))
                            logger.info(f"  - {unique}INDEX: {cols}")
                except Exception as e:
                    logger.debug(f"Could not show indexes: {str(e)}")
                
                logger.info("\n" + "=" * 60)
                logger.info("✅ Migration completed successfully!")
                logger.info("=" * 60)
                logger.info("\nThe job_suggested_candidates table is ready to store")
                logger.info("top 3 suggested candidates for each job posting.")
                return True
            else:
                logger.error("❌ Table job_suggested_candidates was not created")
                return False
                
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Error creating table: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    create_job_suggested_candidates_table()

