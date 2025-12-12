"""
Migration helper to add `last_job_sync_at` to `stafferlink_integrations`.

Usage:
    python backend/migrations/add_stafferlink_last_job_sync_at.py

Reads DATABASE_URL from environment (or .env) and adds the column if missing.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv


def load_env():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def column_exists(engine, table, column):
    inspector = inspect(engine)
    for col in inspector.get_columns(table):
        if col["name"] == column:
            return True
    return False


def add_column(engine):
    ddl = text(
        """
        ALTER TABLE stafferlink_integrations
        ADD COLUMN last_job_sync_at DATETIME NULL AFTER updated_at
        """
    )
    with engine.begin() as conn:
        conn.execute(ddl)


def main():
    load_env()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set. Update backend/.env and retry.")
        return

    engine = create_engine(database_url)

    try:
        if column_exists(engine, "stafferlink_integrations", "last_job_sync_at"):
            print("✅ Column last_job_sync_at already exists. Nothing to do.")
            return

        add_column(engine)

        if column_exists(engine, "stafferlink_integrations", "last_job_sync_at"):
            print("✅ Added last_job_sync_at to stafferlink_integrations.")
        else:
            print("❌ Column still missing after migration. Check permissions.")

    except SQLAlchemyError as exc:
        print(f"❌ Migration failed: {exc}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

