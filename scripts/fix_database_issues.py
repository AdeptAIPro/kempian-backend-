#!/usr/bin/env python3
"""
Script to fix database issues:
1. Add missing columns to support_tickets table (category, etc.)
2. Fix ticket_attachments foreign key constraint issue
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import create_app, db
    from sqlalchemy import text, inspect
except ImportError as e:
    print(f"❌ Error importing app: {e}")
    print("Make sure you're running this script from the backend directory")
    sys.exit(1)


def fix_support_tickets_table():
    """Add missing columns to support_tickets table"""
    app = create_app()
    
    with app.app_context():
        try:
            inspector = inspect(db.engine)
            
            # Check if support_tickets table exists
            if 'support_tickets' not in inspector.get_table_names():
                print("[WARNING] support_tickets table does not exist. It will be created by create_all_tables.py")
                return True
            
            # Get existing columns
            existing_columns = {col['name'] for col in inspector.get_columns('support_tickets')}
            print(f"Existing columns in support_tickets: {sorted(existing_columns)}")
            
            dialect = db.engine.dialect.name
            print(f"Database dialect: {dialect}")
            
            # Columns that should exist according to the model
            if dialect == 'mysql':
                required_columns = {
                    'category': "ENUM('bug', 'feature_request', 'question', 'billing', 'technical', 'account', 'integration', 'other')",
                    'assigned_to': 'INT',
                    'assigned_at': 'DATETIME',
                    'internal_notes': 'TEXT',
                    'notes_updated_by': 'INT',
                    'notes_updated_at': 'DATETIME',
                    'source': "ENUM('help_widget', 'email', 'dashboard', 'api', 'other')",
                    'source_url': 'VARCHAR(500)',
                    'due_date': 'DATETIME',
                    'first_response_at': 'DATETIME',
                    'rating': 'INT',
                    'feedback': 'TEXT',
                    'rated_at': 'DATETIME',
                    'tags': 'JSON'
                }
            else:
                required_columns = {
                    'category': "VARCHAR(20)",
                    'assigned_to': 'INTEGER',
                    'assigned_at': 'TIMESTAMP',
                    'internal_notes': 'TEXT',
                    'notes_updated_by': 'INTEGER',
                    'notes_updated_at': 'TIMESTAMP',
                    'source': "VARCHAR(20)",
                    'source_url': 'VARCHAR(500)',
                    'due_date': 'TIMESTAMP',
                    'first_response_at': 'TIMESTAMP',
                    'rating': 'INTEGER',
                    'feedback': 'TEXT',
                    'rated_at': 'TIMESTAMP',
                    'tags': 'JSON'
                }
            
            dialect = db.engine.dialect.name
            print(f"Database dialect: {dialect}")
            
            alter_statements = []
            
            for col_name, col_type in required_columns.items():
                if col_name not in existing_columns:
                    if dialect == 'mysql':
                        # MySQL-specific syntax
                        if col_name == 'source':
                            alter_statements.append(
                                f"ALTER TABLE support_tickets ADD COLUMN {col_name} {col_type} NOT NULL DEFAULT 'help_widget'"
                            )
                        else:
                            alter_statements.append(
                                f"ALTER TABLE support_tickets ADD COLUMN {col_name} {col_type}"
                            )
                    else:
                        # PostgreSQL or other
                        if col_name == 'source':
                            alter_statements.append(
                                f"ALTER TABLE support_tickets ADD COLUMN {col_name} {col_type} NOT NULL DEFAULT 'help_widget'"
                            )
                        else:
                            alter_statements.append(
                                f"ALTER TABLE support_tickets ADD COLUMN {col_name} {col_type}"
                            )
                    print(f"  Will add column: {col_name}")
            
            if alter_statements:
                print(f"\nAdding {len(alter_statements)} missing columns...")
                for stmt in alter_statements:
                    try:
                        db.session.execute(text(stmt))
                        db.session.commit()
                        col_name = stmt.split()[4]
                        print(f"  [OK] Added column: {col_name}")
                    except Exception as e:
                        print(f"  [ERROR] Error adding column: {e}")
                        db.session.rollback()
                        # Try without NOT NULL constraint
                        if 'NOT NULL' in stmt:
                            stmt_retry = stmt.replace(' NOT NULL', '')
                            try:
                                db.session.execute(text(stmt_retry))
                                db.session.commit()
                                col_name = stmt_retry.split()[4]
                                print(f"  [OK] Added column (nullable): {col_name}")
                            except Exception as e2:
                                print(f"  [ERROR] Failed even without NOT NULL: {e2}")
                                db.session.rollback()
                print("[SUCCESS] support_tickets table updated")
            else:
                print("[SUCCESS] All required columns already exist in support_tickets")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error fixing support_tickets table: {e}")
            import traceback
            print(traceback.format_exc())
            db.session.rollback()
            return False


def fix_ticket_attachments_table():
    """Fix ticket_attachments table foreign key constraint"""
    app = create_app()
    
    with app.app_context():
        try:
            inspector = inspect(db.engine)
            dialect = db.engine.dialect.name
            
            # Check if support_tickets exists
            if 'support_tickets' not in inspector.get_table_names():
                print("[WARNING] support_tickets table does not exist. Cannot fix ticket_attachments.")
                return False
            
            # Get the exact type of support_tickets.id
            support_tickets_id_type = None
            if dialect == 'mysql':
                query = text("""
                    SELECT COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = 'support_tickets' 
                    AND COLUMN_NAME = 'id'
                """)
                result = db.session.execute(query).fetchone()
                if result:
                    support_tickets_id_type = result[0]
                    print(f"  support_tickets.id type: {support_tickets_id_type}")
            
            # Check if ticket_attachments exists
            if 'ticket_attachments' in inspector.get_table_names():
                print("ticket_attachments table exists. Checking foreign key...")
                
                # Check if foreign key exists and is correct
                fk_query = text("""
                    SELECT CONSTRAINT_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = 'ticket_attachments'
                    AND COLUMN_NAME = 'ticket_id'
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                """)
                fk_result = db.session.execute(fk_query).fetchone()
                
                if fk_result:
                    print(f"  Foreign key exists: {fk_result[0]}")
                    # Check if there's a type mismatch
                    ticket_id_type_query = text("""
                        SELECT COLUMN_TYPE
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = DATABASE() 
                        AND TABLE_NAME = 'ticket_attachments' 
                        AND COLUMN_NAME = 'ticket_id'
                    """)
                    ticket_id_result = db.session.execute(ticket_id_type_query).fetchone()
                    if ticket_id_result and support_tickets_id_type:
                        ticket_id_type = ticket_id_result[0]
                        print(f"  ticket_attachments.ticket_id type: {ticket_id_type}")
                        if ticket_id_type != support_tickets_id_type:
                            print(f"  [WARNING] Type mismatch detected!")
                            print(f"     Dropping and recreating table...")
                            
                            # Drop the table
                            db.session.execute(text("DROP TABLE IF EXISTS ticket_attachments"))
                            db.session.commit()
                            print("  [OK] Dropped ticket_attachments table")
                        else:
                            print("  [OK] Foreign key types match. Table is correct.")
                            return True
                else:
                    print("  No foreign key found. Table will be recreated.")
                    db.session.execute(text("DROP TABLE IF EXISTS ticket_attachments"))
                    db.session.commit()
            
            # Now create the table with correct types (only if it doesn't exist)
            if 'ticket_attachments' in inspector.get_table_names():
                print("  [INFO] ticket_attachments table already exists and is correct.")
                return True
                
            print("\nCreating ticket_attachments table with correct foreign key...")
            
            if dialect == 'mysql':
                # Use the exact type from support_tickets.id
                if support_tickets_id_type:
                    ticket_id_col = f"ticket_id {support_tickets_id_type} NOT NULL"
                else:
                    ticket_id_col = "ticket_id INT NOT NULL"
                
                # Get users.id type
                users_id_type = None
                query = text("""
                    SELECT COLUMN_TYPE
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = 'users' 
                    AND COLUMN_NAME = 'id'
                """)
                result = db.session.execute(query).fetchone()
                if result:
                    users_id_type = result[0]
                    print(f"  users.id type: {users_id_type}")
                
                uploaded_by_col = f"uploaded_by {users_id_type} NOT NULL" if users_id_type else "uploaded_by INT NOT NULL"
                
                create_table_sql = text(f"""
                    CREATE TABLE ticket_attachments (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        {ticket_id_col},
                        file_name VARCHAR(255) NOT NULL,
                        file_path VARCHAR(500) NOT NULL,
                        file_size INT NOT NULL,
                        content_type VARCHAR(100) NOT NULL,
                        {uploaded_by_col},
                        uploaded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        is_internal TINYINT(1) NOT NULL DEFAULT 0,
                        FOREIGN KEY(ticket_id) REFERENCES support_tickets(id) ON DELETE CASCADE,
                        FOREIGN KEY(uploaded_by) REFERENCES users(id)
                    )
                """)
            else:
                # PostgreSQL or other
                create_table_sql = text("""
                    CREATE TABLE ticket_attachments (
                        id SERIAL PRIMARY KEY,
                        ticket_id INTEGER NOT NULL,
                        file_name VARCHAR(255) NOT NULL,
                        file_path VARCHAR(500) NOT NULL,
                        file_size INTEGER NOT NULL,
                        content_type VARCHAR(100) NOT NULL,
                        uploaded_by INTEGER NOT NULL,
                        uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        is_internal BOOLEAN NOT NULL DEFAULT FALSE,
                        FOREIGN KEY(ticket_id) REFERENCES support_tickets(id) ON DELETE CASCADE,
                        FOREIGN KEY(uploaded_by) REFERENCES users(id)
                    )
                """)
            
            db.session.execute(create_table_sql)
            db.session.commit()
            print("  [OK] Created ticket_attachments table")
            
            # Create indexes
            indexes = [
                "CREATE INDEX idx_ticket_attachment_ticket ON ticket_attachments(ticket_id)",
                "CREATE INDEX idx_ticket_attachment_uploader ON ticket_attachments(uploaded_by)"
            ]
            
            for idx_sql in indexes:
                try:
                    db.session.execute(text(idx_sql))
                    db.session.commit()
                except Exception as e:
                    # Index might already exist
                    if "Duplicate key name" not in str(e) and "already exists" not in str(e):
                        print(f"  [WARNING] Could not create index: {e}")
            
            print("[SUCCESS] ticket_attachments table fixed")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error fixing ticket_attachments table: {e}")
            import traceback
            print(traceback.format_exc())
            db.session.rollback()
            return False


def main():
    print("=" * 80)
    print("FIXING DATABASE ISSUES")
    print("=" * 80)
    print()
    
    success1 = fix_support_tickets_table()
    print()
    success2 = fix_ticket_attachments_table()
    
    print()
    print("=" * 80)
    if success1 and success2:
        print("[SUCCESS] All database issues fixed!")
    else:
        print("[WARNING] Some issues may remain. Check the output above.")
    print("=" * 80)


if __name__ == "__main__":
    main()

