"""
Alembic migration to add service account fields to jobvite_settings table.

Run this migration to add:
- service_account_username (VARCHAR(255), nullable)
- service_account_password_encrypted (TEXT, nullable)

Usage:
    python -m alembic upgrade head
    OR
    Run the SQL directly in your database
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'add_jobvite_service_account_fields'
down_revision = 'create_jobvite_tables'  # Update with your last migration revision
branch_labels = None
depends_on = None

def upgrade():
    """Add service account fields to jobvite_settings table"""
    op.add_column('jobvite_settings', 
        sa.Column('service_account_username', sa.String(255), nullable=True))
    op.add_column('jobvite_settings', 
        sa.Column('service_account_password_encrypted', sa.Text(), nullable=True))
    
    # Add index for service account username if needed
    op.create_index('idx_jobvite_settings_service_account', 
                   'jobvite_settings', 
                   ['service_account_username'])

def downgrade():
    """Remove service account fields from jobvite_settings table"""
    op.drop_index('idx_jobvite_settings_service_account', table_name='jobvite_settings')
    op.drop_column('jobvite_settings', 'service_account_password_encrypted')
    op.drop_column('jobvite_settings', 'service_account_username')

# Alternative: Direct SQL migration (if not using Alembic)
SQL_MIGRATION = """
-- Add service account fields to jobvite_settings table
ALTER TABLE jobvite_settings 
ADD COLUMN service_account_username VARCHAR(255) NULL,
ADD COLUMN service_account_password_encrypted TEXT NULL;

-- Add index for faster lookups
CREATE INDEX idx_jobvite_settings_service_account 
ON jobvite_settings(service_account_username);

-- Verify migration
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'jobvite_settings' 
AND column_name IN ('service_account_username', 'service_account_password_encrypted');
"""

