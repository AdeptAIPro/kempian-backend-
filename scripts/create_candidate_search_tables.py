"""
Database migration script for candidate search history tables
Run this script to create the necessary tables for the candidate search history feature
"""
from app import create_app
from app.db import db
from app.models import CandidateSearchHistory, CandidateSearchResult
from app.simple_logger import get_logger

logger = get_logger('migration')

def create_candidate_search_tables():
    """Create candidate search history tables"""
    app = create_app()
    
    with app.app_context():
        try:
            # Create tables
            db.create_all()
            logger.info("Candidate search history tables created successfully")
            
            # Verify tables exist
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'candidate_search_history' in tables:
                logger.info("SUCCESS: candidate_search_history table created")
            else:
                logger.error("ERROR: candidate_search_history table not found")
                
            if 'candidate_search_results' in tables:
                logger.info("SUCCESS: candidate_search_results table created")
            else:
                logger.error("ERROR: candidate_search_results table not found")
                
            logger.info("Migration completed successfully")
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            raise

if __name__ == "__main__":
    create_candidate_search_tables()
