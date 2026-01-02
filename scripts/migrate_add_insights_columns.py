#!/usr/bin/env python3
"""
Migration script to add AI insights columns to candidate_profiles table.

This script adds the following columns to the candidate_profiles table:
- ai_career_insight (TEXT)
- benchmarking_data (JSON)
- recommended_courses (JSON)
- insights_generated_at (DATETIME)
- target_role_for_insights (VARCHAR(255))

Usage:
    python migrate_add_insights_columns.py
"""

import os
import sys
from sqlalchemy import text, inspect
from sqlalchemy.exc import OperationalError, ProgrammingError

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.db import db
from app.simple_logger import get_logger

logger = get_logger(__name__)

def column_exists(connection, table_name, column_name):
    """Check if a column exists in a table"""
    inspector = inspect(connection)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns

def add_column_if_not_exists(connection, table_name, column_name, column_type, nullable=True):
    """Add a column to a table if it doesn't exist"""
    try:
        if not column_exists(connection, table_name, column_name):
            # Get the database dialect
            dialect = connection.dialect.name
            
            if dialect == 'sqlite':
                # SQLite doesn't support adding columns with constraints easily
                # For JSON, we'll use TEXT in SQLite
                if column_type.upper() == 'JSON':
                    sql_type = 'TEXT'
                else:
                    sql_type = column_type
                
                alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sql_type}"
                if not nullable:
                    alter_sql += " NOT NULL"
                    
            elif dialect in ['postgresql', 'mysql', 'mariadb']:
                # PostgreSQL and MySQL support JSON type
                sql_type = column_type
                alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sql_type}"
                if not nullable:
                    alter_sql += " NOT NULL"
                else:
                    alter_sql += " NULL"
            else:
                # Generic SQL
                sql_type = column_type if column_type.upper() != 'JSON' else 'TEXT'
                alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sql_type}"
                if not nullable:
                    alter_sql += " NOT NULL"
            
            connection.execute(text(alter_sql))
            connection.commit()
            logger.info(f"✓ Added column '{column_name}' to table '{table_name}'")
            return True
        else:
            logger.info(f"⊘ Column '{column_name}' already exists in table '{table_name}'")
            return False
    except Exception as e:
        logger.error(f"✗ Error adding column '{column_name}': {str(e)}")
        connection.rollback()
        raise

def migrate():
    """Run the migration"""
    app = create_app()
    
    with app.app_context():
        try:
            connection = db.engine.connect()
            table_name = 'candidate_profiles'
            
            logger.info("=" * 60)
            logger.info("Starting migration: Add insights columns to candidate_profiles")
            logger.info("=" * 60)
            
            # Check if table exists
            inspector = inspect(connection)
            if table_name not in inspector.get_table_names():
                logger.error(f"Table '{table_name}' does not exist!")
                return False
            
            logger.info(f"Table '{table_name}' found. Checking columns...")
            
            # Get database dialect for proper type handling
            dialect = connection.dialect.name
            logger.info(f"Database dialect: {dialect}")
            
            # Determine JSON type based on database
            if dialect == 'postgresql':
                json_type = 'JSON'
            elif dialect in ['mysql', 'mariadb']:
                json_type = 'JSON'
            else:  # SQLite or others
                json_type = 'TEXT'  # SQLite doesn't have native JSON type
            
            # Add columns
            columns_added = 0
            
            # 1. ai_career_insight (TEXT)
            if add_column_if_not_exists(connection, table_name, 'ai_career_insight', 'TEXT', nullable=True):
                columns_added += 1
            
            # 2. benchmarking_data (JSON)
            if add_column_if_not_exists(connection, table_name, 'benchmarking_data', json_type, nullable=True):
                columns_added += 1
            
            # 3. recommended_courses (JSON)
            if add_column_if_not_exists(connection, table_name, 'recommended_courses', json_type, nullable=True):
                columns_added += 1
            
            # 4. insights_generated_at (DATETIME)
            if dialect == 'postgresql':
                datetime_type = 'TIMESTAMP'
            elif dialect in ['mysql', 'mariadb']:
                datetime_type = 'DATETIME'
            else:  # SQLite
                datetime_type = 'DATETIME'
            
            if add_column_if_not_exists(connection, table_name, 'insights_generated_at', datetime_type, nullable=True):
                columns_added += 1
            
            # 5. target_role_for_insights (VARCHAR/STRING)
            if add_column_if_not_exists(connection, table_name, 'target_role_for_insights', 'VARCHAR(255)', nullable=True):
                columns_added += 1
            
            logger.info("=" * 60)
            if columns_added > 0:
                logger.info(f"✓ Migration completed successfully! Added {columns_added} column(s).")
            else:
                logger.info("✓ All columns already exist. No changes needed.")
            logger.info("=" * 60)
            
            connection.close()
            return True
            
        except Exception as e:
            logger.error(f"✗ Migration failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Candidate Profiles Insights Columns Migration")
    print("=" * 60 + "\n")
    
    success = migrate()
    
    if success:
        print("\n✓ Migration completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Migration failed. Please check the logs above.")
        sys.exit(1)

