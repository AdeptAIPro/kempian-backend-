"""
Migration Script: Create Communication Tables for Twilio Integration
Run this script to create the necessary tables for message templates,
candidate communications, and communication replies.

Usage:
    python backend/migrations/create_communication_tables.py
    OR
    cd backend && python migrations/create_communication_tables.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from sqlalchemy import text

def create_communication_tables():
    """Create communication tables using SQL"""
    app = create_app()
    
    with app.app_context():
        try:
            # Read SQL file
            sql_file_path = os.path.join(
                os.path.dirname(__file__),
                'create_communication_tables.sql'
            )
            
            if not os.path.exists(sql_file_path):
                print(f"[ERROR] SQL file not found: {sql_file_path}")
                return
            
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            
            # Remove comments and split by semicolon
            lines = sql_script.split('\n')
            cleaned_lines = []
            for line in lines:
                # Remove single-line comments
                if '--' in line:
                    line = line[:line.index('--')]
                cleaned_lines.append(line)
            cleaned_script = '\n'.join(cleaned_lines)
            
            # Split by semicolon and execute each statement
            statements = [s.strip() for s in cleaned_script.split(';') if s.strip()]
            
            executed_count = 0
            for statement in statements:
                if statement and len(statement) > 10:  # Ignore very short statements
                    try:
                        db.session.execute(text(statement))
                        executed_count += 1
                        # Extract table name from CREATE TABLE statement
                        if 'CREATE TABLE' in statement.upper():
                            # Handle "CREATE TABLE IF NOT EXISTS table_name"
                            parts = statement.upper().split()
                            if 'IF' in parts and 'NOT' in parts and 'EXISTS' in parts:
                                idx = parts.index('EXISTS')
                                if idx + 1 < len(parts):
                                    table_name = parts[idx + 1].split('(')[0]
                                else:
                                    table_name = 'unknown'
                            elif len(parts) >= 3:
                                table_name = parts[2].split('(')[0]
                            else:
                                table_name = 'unknown'
                            print(f"[OK] Created table: {table_name}")
                    except Exception as e:
                        # Table might already exist, which is okay
                        error_str = str(e).lower()
                        if "already exists" in error_str or "duplicate" in error_str or "1050" in error_str:
                            print(f"[SKIP] Table already exists (skipping)")
                        else:
                            print(f"[ERROR] Error executing statement: {str(e)}")
                            raise
            
            if executed_count == 0:
                print("[WARNING] No statements were executed. Tables may already exist.")
            
            db.session.commit()
            print("\n[SUCCESS] Successfully created communication tables!")
            print("   - message_templates")
            print("   - candidate_communications")
            print("   - communication_replies")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error creating tables: {str(e)}")
            raise

if __name__ == '__main__':
    create_communication_tables()

