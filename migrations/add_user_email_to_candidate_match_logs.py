"""
Migration script to add user_email column to candidate_match_logs table.

Usage:
    python backend/migrations/add_user_email_to_candidate_match_logs.py
"""
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from sqlalchemy import inspect, text


def add_user_email_column():
    app = create_app()

    with app.app_context():
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('candidate_match_logs')]

        if 'user_email' in columns:
            print("ℹ️  user_email column already exists.")
            return

        print("Adding user_email column to candidate_match_logs table...")

        dialect = db.engine.dialect.name

        if dialect == 'mysql':
            db.session.execute(text(
                "ALTER TABLE candidate_match_logs "
                "ADD COLUMN user_email VARCHAR(255) NULL AFTER user_id"
            ))
        elif dialect == 'postgresql':
            db.session.execute(text(
                "ALTER TABLE candidate_match_logs "
                "ADD COLUMN user_email VARCHAR(255)"
            ))
        else:  # SQLite or others
            db.session.execute(text(
                "ALTER TABLE candidate_match_logs "
                "ADD COLUMN user_email TEXT"
            ))

        db.session.commit()
        print("✅ user_email column added successfully.")


if __name__ == "__main__":
    add_user_email_column()

