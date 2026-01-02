"""
Migration script to create support_tickets table

Run this script to create the support_tickets table in your database.

Usage:
    python -m backend.migrations.create_support_tickets_table
    OR
    python backend/migrations/create_support_tickets_table.py
"""

import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import db
from app import create_app
from sqlalchemy import text

def create_support_tickets_table():
    """Create the support_tickets table"""
    app = create_app()
    
    with app.app_context():
        try:
            # Check if table already exists
            inspector = db.inspect(db.engine)
            if 'support_tickets' in inspector.get_table_names():
                print("Table 'support_tickets' already exists. Skipping creation.")
                return
            
            # Create table SQL
            create_table_sql = text("""
                CREATE TABLE support_tickets (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    user_email VARCHAR(255) NOT NULL,
                    subject VARCHAR(255),
                    message TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'open',
                    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
                    admin_reply TEXT,
                    replied_by INTEGER,
                    replied_at TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_support_ticket_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    CONSTRAINT fk_support_ticket_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                    CONSTRAINT fk_support_ticket_replier FOREIGN KEY (replied_by) REFERENCES users(id) ON DELETE SET NULL,
                    CONSTRAINT chk_support_ticket_status CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
                    CONSTRAINT chk_support_ticket_priority CHECK (priority IN ('low', 'medium', 'high'))
                );
            """)
            
            # Create indexes
            create_indexes_sql = [
                text("CREATE INDEX idx_support_ticket_user ON support_tickets(user_id);"),
                text("CREATE INDEX idx_support_ticket_status ON support_tickets(status);"),
                text("CREATE INDEX idx_support_ticket_created ON support_tickets(created_at);"),
                text("CREATE INDEX idx_support_ticket_tenant ON support_tickets(tenant_id);"),
            ]
            
            # Execute table creation
            print("Creating support_tickets table...")
            db.session.execute(create_table_sql)
            db.session.commit()
            print("✓ Table 'support_tickets' created successfully")
            
            # Execute index creation
            print("Creating indexes...")
            for index_sql in create_indexes_sql:
                db.session.execute(index_sql)
            db.session.commit()
            print("✓ Indexes created successfully")
            
            print("\n✅ Migration completed successfully!")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Error creating table: {e}")
            raise

if __name__ == '__main__':
    create_support_tickets_table()

