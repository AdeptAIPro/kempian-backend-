"""Utility script to add the salary_type column and backfill data.

Usage:
    python -m scripts.add_salary_type_column
"""
import os
import sys
from typing import Optional

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip heavy startup pieces while running the script
os.environ.setdefault("SKIP_HEAVY_INIT", "1")
os.environ.setdefault("SKIP_SEARCH_INIT", "1")
os.environ.setdefault("MIGRATION_MODE", "1")

from flask import Flask
from sqlalchemy import inspect, text
from app.db import db
from app.config import Config
from dotenv import load_dotenv

# Load environment variables from backend/.env if present
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def create_minimal_app() -> Flask:
    """Create a minimal Flask app instance initialized with DB only."""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app


def column_exists(inspector, table_name: str, column_name: str) -> bool:
    """Check if a column exists on a table."""
    try:
        columns = inspector.get_columns(table_name)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[ERROR] Unable to inspect table '{table_name}': {exc}")
        return False

    for column in columns:
        if column.get("name") == column_name:
            return True
    return False


def add_salary_type_column() -> None:
    """Add salary_type column to employee_profiles and default existing rows."""
    app = create_minimal_app()

    with app.app_context():
        engine = db.engine
        inspector = inspect(engine)

        table_name = "employee_profiles"
        column_name = "salary_type"
        altered = False

        if column_exists(inspector, table_name, column_name):
            print("[INFO] Column salary_type already exists. Skipping ALTER TABLE.")
        else:
            alter_sql = text(
                """
                ALTER TABLE employee_profiles
                    ADD COLUMN salary_type VARCHAR(20) NOT NULL DEFAULT 'monthly'
                    AFTER salary_currency
                """
            )
            with engine.begin() as connection:
                print("[INFO] Adding salary_type column to employee_profiles...")
                connection.execute(alter_sql)
                altered = True
                print("[SUCCESS] Column salary_type added successfully.")

        # Refresh inspector if table definition may have changed
        if altered:
            inspector = inspect(engine)

        if column_exists(inspector, table_name, column_name):
            update_sql = text(
                """
                UPDATE employee_profiles
                SET salary_type = 'monthly'
                WHERE salary_type IS NULL OR salary_type = ''
                """
            )
            with engine.begin() as connection:
                print("[INFO] Backfilling salary_type for existing rows (default 'monthly')...")
                result = connection.execute(update_sql)
                if hasattr(result, "rowcount"):
                    print(f"[SUCCESS] Updated {result.rowcount} rows.")
                else:  # pragma: no cover - compatibility branch
                    print("[SUCCESS] Backfill command executed.")
        else:
            print("[ERROR] Column salary_type still missing after attempted ALTER. Please check manually.")


if __name__ == "__main__":
    print("=" * 60)
    print("Adding salary_type column to employee_profiles")
    print("=" * 60)
    try:
        add_salary_type_column()
        print("\n[SUCCESS] Script completed.")
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - main guard
        print(f"\n[ERROR] Script failed: {exc}")
        raise
