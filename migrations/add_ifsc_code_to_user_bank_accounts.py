"""
Alembic migration to add ifsc_code column to user_bank_accounts table.

Run this migration to add:
- ifsc_code (VARCHAR(11), nullable) - Indian Financial System Code

Usage:
    python -m alembic upgrade head
    OR
    Run the SQL directly in your database
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'add_ifsc_code_to_user_bank_accounts'
down_revision = 'add_jobvite_service_account_fields'  # Update with your last migration revision
branch_labels = None
depends_on = None

def upgrade():
    """Add ifsc_code column to user_bank_accounts table"""
    op.add_column('user_bank_accounts',
        sa.Column('ifsc_code', sa.String(11), nullable=True))

def downgrade():
    """Remove ifsc_code column from user_bank_accounts table"""
    op.drop_column('user_bank_accounts', 'ifsc_code')
