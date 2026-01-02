"""
Migration script to enhance support_tickets table with new fields
and create ticket_attachments table

Run this script to add new fields to the support_tickets table and create ticket_attachments table.

Usage:
    python -m backend.migrations.enhance_support_tickets_table
    OR
    python backend/migrations/enhance_support_tickets_table.py
"""

import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import db
from app import create_app
from sqlalchemy import text

def enhance_support_tickets_table():
    """Enhance the support_tickets table with new fields and create ticket_attachments table"""
    app = create_app()
    
    with app.app_context():
        try:
            inspector = db.inspect(db.engine)
            table_exists = 'support_tickets' in inspector.get_table_names()
            
            if not table_exists:
                print("❌ Table 'support_tickets' does not exist. Please run create_support_tickets_table.py first.")
                return
            
            print("Enhancing support_tickets table...")
            
            # Detect database dialect
            dialect = db.engine.dialect.name
            print(f"Database dialect detected: {dialect}")
            
            # Determine JSON type based on database
            if dialect == 'postgresql':
                json_type = 'JSONB'
            elif dialect in ['mysql', 'mariadb']:
                json_type = 'JSON'
            else:  # SQLite or others
                json_type = 'TEXT'  # SQLite doesn't have native JSON type
            
            # Check which columns already exist
            existing_columns = [col['name'] for col in inspector.get_columns('support_tickets')]
            
            # Add new columns if they don't exist
            alter_statements = []
            
            if 'category' not in existing_columns:
                # MySQL doesn't support CHECK constraints in ALTER TABLE well, so we'll add it without constraint
                # Validation will be handled at application level
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN category VARCHAR(20);
                """)
            
            if 'assigned_to' not in existing_columns:
                # Add column first, foreign key constraint will be handled by SQLAlchemy model
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN assigned_to INTEGER;
                """)
            
            if 'assigned_at' not in existing_columns:
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN assigned_at TIMESTAMP;
                """)
            
            if 'internal_notes' not in existing_columns:
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN internal_notes TEXT;
                """)
            
            if 'notes_updated_by' not in existing_columns:
                # Add column first, foreign key constraint will be handled by SQLAlchemy model
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN notes_updated_by INTEGER;
                """)
            
            if 'notes_updated_at' not in existing_columns:
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN notes_updated_at TIMESTAMP;
                """)
            
            if 'source' not in existing_columns:
                # MySQL doesn't support CHECK constraints in ALTER TABLE well
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN source VARCHAR(20) NOT NULL DEFAULT 'help_widget';
                """)
            
            if 'source_url' not in existing_columns:
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN source_url VARCHAR(500);
                """)
            
            if 'due_date' not in existing_columns:
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN due_date TIMESTAMP;
                """)
            
            if 'first_response_at' not in existing_columns:
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN first_response_at TIMESTAMP;
                """)
            
            if 'rating' not in existing_columns:
                # MySQL doesn't support CHECK constraints in ALTER TABLE well
                # Validation will be handled at application level
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN rating INTEGER;
                """)
            
            if 'feedback' not in existing_columns:
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN feedback TEXT;
                """)
            
            if 'rated_at' not in existing_columns:
                alter_statements.append("""
                    ALTER TABLE support_tickets 
                    ADD COLUMN rated_at TIMESTAMP;
                """)
            
            if 'tags' not in existing_columns:
                alter_statements.append(f"""
                    ALTER TABLE support_tickets 
                    ADD COLUMN tags {json_type};
                """)
            
            # Execute alter statements
            for stmt in alter_statements:
                db.session.execute(text(stmt))
            
            db.session.commit()
            print("✓ New columns added to support_tickets table")
            
            # Create indexes for new columns
            print("Creating indexes for new columns...")
            index_statements = []
            
            existing_indexes = [idx['name'] for idx in inspector.get_indexes('support_tickets')]
            
            if 'idx_support_ticket_category' not in existing_indexes:
                index_statements.append("CREATE INDEX idx_support_ticket_category ON support_tickets(category);")
            
            if 'idx_support_ticket_assigned' not in existing_indexes:
                index_statements.append("CREATE INDEX idx_support_ticket_assigned ON support_tickets(assigned_to);")
            
            if 'idx_support_ticket_source' not in existing_indexes:
                index_statements.append("CREATE INDEX idx_support_ticket_source ON support_tickets(source);")
            
            for idx_stmt in index_statements:
                try:
                    db.session.execute(text(idx_stmt))
                except Exception as e:
                    print(f"  Note: Index may already exist: {e}")
            
            db.session.commit()
            print("✓ Indexes created")
            
            # Create ticket_attachments table
            print("Creating ticket_attachments table...")
            if 'ticket_attachments' in inspector.get_table_names():
                print("  Table 'ticket_attachments' already exists. Skipping creation.")
            else:
                # Get the actual column type of support_tickets.id to match it exactly
                support_tickets_columns = inspector.get_columns('support_tickets')
                ticket_id_type = None
                user_id_type = None
                
                for col in support_tickets_columns:
                    if col['name'] == 'id':
                        ticket_id_type = str(col['type'])
                    elif col['name'] == 'user_id':
                        user_id_type = str(col['type'])
                
                print(f"  Detected support_tickets.id type: {ticket_id_type}")
                print(f"  Detected users.id type (from user_id): {user_id_type}")
                
                # Use MySQL-compatible syntax
                if dialect in ['mysql', 'mariadb']:
                    # Query the actual column definition from information_schema
                    # This ensures we get the exact type including signed/unsigned, size, etc.
                    try:
                        ticket_id_query = text("""
                            SELECT COLUMN_TYPE 
                            FROM INFORMATION_SCHEMA.COLUMNS 
                            WHERE TABLE_SCHEMA = DATABASE() 
                            AND TABLE_NAME = 'support_tickets' 
                            AND COLUMN_NAME = 'id'
                        """)
                        result = db.session.execute(ticket_id_query).fetchone()
                        if result:
                            exact_ticket_id_type = result[0]
                            print(f"  Exact support_tickets.id type from DB: {exact_ticket_id_type}")
                        else:
                            exact_ticket_id_type = 'INT'
                    except Exception as e:
                        print(f"  Warning: Could not query column type: {e}")
                        exact_ticket_id_type = 'INT'
                    
                    try:
                        user_id_query = text("""
                            SELECT COLUMN_TYPE 
                            FROM INFORMATION_SCHEMA.COLUMNS 
                            WHERE TABLE_SCHEMA = DATABASE() 
                            AND TABLE_NAME = 'users' 
                            AND COLUMN_NAME = 'id'
                        """)
                        result = db.session.execute(user_id_query).fetchone()
                        if result:
                            exact_user_id_type = result[0]
                            print(f"  Exact users.id type from DB: {exact_user_id_type}")
                        else:
                            exact_user_id_type = 'INT'
                    except Exception as e:
                        print(f"  Warning: Could not query users.id type: {e}")
                        exact_user_id_type = 'INT'
                    
                    # Create table without foreign keys first
                    create_attachments_table_sql = text(f"""
                        CREATE TABLE ticket_attachments (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            ticket_id {exact_ticket_id_type} NOT NULL,
                            file_name VARCHAR(255) NOT NULL,
                            file_path VARCHAR(500) NOT NULL,
                            file_size INT NOT NULL,
                            content_type VARCHAR(100) NOT NULL,
                            uploaded_by {exact_user_id_type} NOT NULL,
                            uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            is_internal TINYINT(1) NOT NULL DEFAULT 0
                        );
                    """)
                    
                    db.session.execute(create_attachments_table_sql)
                    db.session.commit()
                    print("✓ Table 'ticket_attachments' created")
                    
                    # Add foreign keys separately
                    try:
                        add_fk_ticket = text("""
                            ALTER TABLE ticket_attachments
                            ADD CONSTRAINT fk_ticket_attachment_ticket 
                            FOREIGN KEY (ticket_id) REFERENCES support_tickets(id) ON DELETE CASCADE;
                        """)
                        db.session.execute(add_fk_ticket)
                        db.session.commit()
                        print("✓ Foreign key to support_tickets added")
                    except Exception as e:
                        print(f"  Warning: Could not add foreign key to support_tickets: {e}")
                        db.session.rollback()
                    
                    try:
                        add_fk_user = text("""
                            ALTER TABLE ticket_attachments
                            ADD CONSTRAINT fk_ticket_attachment_uploader 
                            FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE CASCADE;
                        """)
                        db.session.execute(add_fk_user)
                        db.session.commit()
                        print("✓ Foreign key to users added")
                    except Exception as e:
                        print(f"  Warning: Could not add foreign key to users: {e}")
                        db.session.rollback()
                else:
                    # PostgreSQL syntax
                    create_attachments_table_sql = text("""
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
                            CONSTRAINT fk_ticket_attachment_ticket FOREIGN KEY (ticket_id) 
                                REFERENCES support_tickets(id) ON DELETE CASCADE,
                            CONSTRAINT fk_ticket_attachment_uploader FOREIGN KEY (uploaded_by) 
                                REFERENCES users(id) ON DELETE CASCADE
                        );
                    """)
                    
                    db.session.execute(create_attachments_table_sql)
                    db.session.commit()
                    print("✓ Table 'ticket_attachments' created")
                
                # Create indexes for attachments
                attachment_indexes = [
                    text("CREATE INDEX idx_ticket_attachment_ticket ON ticket_attachments(ticket_id);"),
                    text("CREATE INDEX idx_ticket_attachment_uploader ON ticket_attachments(uploaded_by);"),
                ]
                
                for idx_sql in attachment_indexes:
                    db.session.execute(idx_sql)
                db.session.commit()
                print("✓ Indexes created for ticket_attachments")
            
            print("\n✅ Migration completed successfully!")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Error during migration: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    enhance_support_tickets_table()

