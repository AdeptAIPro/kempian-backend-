"""
Migration script to create all Jobvite integration tables.

This creates:
- jobvite_settings
- jobvite_jobs
- jobvite_candidates
- jobvite_candidate_documents
- jobvite_onboarding_processes
- jobvite_onboarding_tasks
- jobvite_webhook_logs

Usage:
    python backend/migrations/create_jobvite_tables.py
"""
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from sqlalchemy import inspect
from app import create_app
from app.db import db
from app.models import (
    JobviteSettings,
    JobviteJob,
    JobviteCandidate,
    JobviteCandidateDocument,
    JobviteOnboardingProcess,
    JobviteOnboardingTask,
    JobviteWebhookLog
)


def create_jobvite_tables():
    """Create all Jobvite integration tables"""
    app = create_app()
    
    with app.app_context():
        tables_to_create = [
            ('jobvite_settings', JobviteSettings),
            ('jobvite_jobs', JobviteJob),
            ('jobvite_candidates', JobviteCandidate),
            ('jobvite_candidate_documents', JobviteCandidateDocument),
            ('jobvite_onboarding_processes', JobviteOnboardingProcess),
            ('jobvite_onboarding_tasks', JobviteOnboardingTask),
            ('jobvite_webhook_logs', JobviteWebhookLog),
        ]
        
        connection = db.engine.connect()
        try:
            for table_name, model_class in tables_to_create:
                print(f"▶️ Creating {table_name} table...")
                try:
                    model_class.__table__.create(db.engine, checkfirst=True)
                    
                    # Re-create inspector after each table to avoid cached metadata
                    inspector = inspect(db.engine)
                    tables = inspector.get_table_names()
                    if table_name in tables:
                        print(f"✅ {table_name} table created successfully.")
                        
                        # Show table structure
                        columns = inspector.get_columns(table_name)
                        print(f"   Columns ({len(columns)}):")
                        for column in columns:
                            nullable = "NULL" if column['nullable'] else "NOT NULL"
                            print(f"     - {column['name']}: {column['type']} {nullable}")
                        
                        # Show indexes
                        indexes = inspector.get_indexes(table_name)
                        if indexes:
                            print(f"   Indexes ({len(indexes)}):")
                            for idx in indexes:
                                print(f"     - {idx['name']}: {idx['column_names']}")
                    else:
                        # Fallback check using dialect API in case inspector cache misses it
                        exists = db.engine.dialect.has_table(connection, table_name)
                        if exists:
                            print(f"✅ {table_name} table created successfully (dialect check).")
                        else:
                            print(f"❌ {table_name} table was not created.")
                            
                except Exception as e:
                    print(f"❌ Error creating {table_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        finally:
            connection.close()
        
        print("\n✅ Jobvite tables migration completed!")


if __name__ == "__main__":
    create_jobvite_tables()

