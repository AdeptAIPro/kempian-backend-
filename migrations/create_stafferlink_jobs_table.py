"""
Migration script to create stafferlink_jobs table and add last_job_sync_at column.

Usage:
    python backend/migrations/create_stafferlink_jobs_table.py
"""
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from sqlalchemy import inspect, text  # noqa: E402

from app import create_app  # noqa: E402
from app.db import db  # noqa: E402
from app.models import StafferlinkJob  # noqa: E402


def create_stafferlink_jobs_table():
    app = create_app()

    with app.app_context():
        inspector = inspect(db.engine)

        print("▶️ Ensuring stafferlink_jobs table exists...")
        StafferlinkJob.__table__.create(db.engine, checkfirst=True)

        tables = inspector.get_table_names()
        if 'stafferlink_jobs' in tables:
            print("✅ stafferlink_jobs table is present.")
            columns = inspector.get_columns('stafferlink_jobs')
            for column in columns:
                print(f"    - {column['name']}: {column['type']}")
        else:
            print("❌ Failed to create stafferlink_jobs table.")

        print("\n▶️ Checking for last_job_sync_at column on stafferlink_integrations...")
        integration_columns = [col['name'] for col in inspector.get_columns('stafferlink_integrations')]

        if 'last_job_sync_at' in integration_columns:
            print("ℹ️  last_job_sync_at already exists on stafferlink_integrations.")
        else:
            dialect = db.engine.dialect.name
            if dialect == 'postgresql':
                column_type = "TIMESTAMP WITHOUT TIME ZONE"
            elif dialect == 'mysql':
                column_type = "DATETIME"
            else:
                column_type = "DATETIME"

            db.session.execute(text(
                f"ALTER TABLE stafferlink_integrations "
                f"ADD COLUMN last_job_sync_at {column_type}"
            ))
            db.session.commit()
            print("✅ last_job_sync_at column added to stafferlink_integrations.")


if __name__ == "__main__":
    create_stafferlink_jobs_table()


