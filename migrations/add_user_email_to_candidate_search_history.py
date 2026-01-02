"""
Migration script to add user_email column to candidate_search_history table.

Usage:
    python backend/migrations/add_user_email_to_candidate_search_history.py
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
        table_name = 'candidate_search_history'

        if table_name not in inspector.get_table_names():
            print(f"❌ '{table_name}' table does not exist. Please run the base migrations first.")
            return

        columns = [col['name'] for col in inspector.get_columns(table_name)]

        if 'user_email' in columns:
            print("ℹ️  user_email column already exists on candidate_search_history.")
            return

        print("Adding user_email column to candidate_search_history table...")

        dialect = db.engine.dialect.name

        if dialect == 'mysql':
            db.session.execute(text(
                "ALTER TABLE candidate_search_history "
                "ADD COLUMN user_email VARCHAR(255) NULL AFTER user_id"
            ))
        elif dialect == 'postgresql':
            db.session.execute(text(
                "ALTER TABLE candidate_search_history "
                "ADD COLUMN user_email VARCHAR(255)"
            ))
        else:  # SQLite or others
            db.session.execute(text(
                "ALTER TABLE candidate_search_history "
                "ADD COLUMN user_email TEXT"
            ))

        db.session.commit()
        print("✅ user_email column added successfully to candidate_search_history.")


if __name__ == "__main__":
    add_user_email_column()


