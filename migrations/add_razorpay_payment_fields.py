"""
Migration Script: Add Razorpay Payment Fields
Adds new fields for Razorpay payment integration to existing tables.

New Fields:
- payroll_settings table:
  * razorpay_webhook_secret (VARCHAR(255), nullable)
  * razorpay_fund_account_validated (BOOLEAN, default=False)
  * razorpay_fund_account_validated_at (DATETIME, nullable)

- user_bank_accounts table:
  * razorpay_contact_id (VARCHAR(255), nullable)
  * razorpay_fund_account_id (VARCHAR(255), nullable)
  * razorpay_contact_created_at (DATETIME, nullable)
  * razorpay_fund_account_created_at (DATETIME, nullable)

Usage:
    python backend/migrations/add_razorpay_payment_fields.py
    OR
    cd backend && python migrations/add_razorpay_payment_fields.py
"""

import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError, ProgrammingError


def column_exists(table_name, column_name, inspector):
    """Check if a column exists in a table"""
    try:
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except Exception as e:
        print(f"  ⚠️  Error checking column {column_name} in {table_name}: {str(e)}")
        return False


def add_column_if_not_exists(table_name, column_name, column_type, default=None, nullable=True):
    """Add a column to a table if it doesn't exist"""
    app = create_app()
    
    with app.app_context():
        inspector = inspect(db.engine)
        
        # Check if table exists
        if table_name not in inspector.get_table_names():
            print(f"  ⚠️  Table '{table_name}' does not exist. Skipping column '{column_name}'.")
            return False
        
        # Check if column already exists
        if column_exists(table_name, column_name, inspector):
            print(f"  ✓ Column '{column_name}' already exists in '{table_name}'")
            return True
        
        try:
            # Detect database dialect
            dialect = db.engine.dialect.name
            
            # Convert column types for MySQL compatibility
            if dialect in ['mysql', 'mariadb']:
                if column_type.upper() == 'BOOLEAN':
                    column_type = 'TINYINT(1)'
                elif column_type.upper() == 'DATETIME':
                    column_type = 'DATETIME'
                # VARCHAR is fine as-is
            
            # Build ALTER TABLE statement
            if default is not None:
                if isinstance(default, bool):
                    if dialect in ['mysql', 'mariadb']:
                        default_value = '1' if default else '0'
                    else:
                        default_value = 'TRUE' if default else 'FALSE'
                elif isinstance(default, str):
                    default_value = f"'{default}'"
                else:
                    default_value = str(default)
                alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type} DEFAULT {default_value}"
            else:
                alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            
            if not nullable:
                alter_sql += " NOT NULL"
            
            # Execute the ALTER TABLE statement
            with db.engine.connect() as conn:
                conn.execute(text(alter_sql))
                conn.commit()
            
            print(f"  ✓ Added column '{column_name}' to '{table_name}'")
            return True
            
        except Exception as e:
            print(f"  ❌ Error adding column '{column_name}' to '{table_name}': {str(e)}")
            return False


def create_tables_if_not_exist():
    """Create tables if they don't exist (for new installations)"""
    app = create_app()
    
    with app.app_context():
        try:
            print("\n[INFO] Creating all tables (if they don't exist)...")
            from app.models import PayrollSettings, UserBankAccount
            db.create_all()
            print("  ✓ Tables created/verified")
            return True
        except Exception as e:
            print(f"  ⚠️  Error creating tables: {str(e)}")
            return False


def migrate_razorpay_fields():
    """Add Razorpay payment fields to existing tables"""
    app = create_app()
    
    with app.app_context():
        print("=" * 80)
        print("RAZORPAY PAYMENT FIELDS MIGRATION")
        print("=" * 80)
        print()
        
        # First, ensure tables exist
        create_tables_if_not_exist()
        
        print("\n[INFO] Adding new columns to existing tables...")
        print("-" * 80)
        
        success_count = 0
        total_count = 0
        
        # Add fields to payroll_settings table
        print("\n📋 Migrating 'payroll_settings' table:")
        total_count += 3
        
        if add_column_if_not_exists(
            'payroll_settings',
            'razorpay_webhook_secret',
            'VARCHAR(255)',
            nullable=True
        ):
            success_count += 1
        
        if add_column_if_not_exists(
            'payroll_settings',
            'razorpay_fund_account_validated',
            'BOOLEAN',
            default=False,
            nullable=False
        ):
            success_count += 1
        
        if add_column_if_not_exists(
            'payroll_settings',
            'razorpay_fund_account_validated_at',
            'DATETIME',
            nullable=True
        ):
            success_count += 1
        
        # Add fields to user_bank_accounts table
        print("\n📋 Migrating 'user_bank_accounts' table:")
        total_count += 4
        
        if add_column_if_not_exists(
            'user_bank_accounts',
            'razorpay_contact_id',
            'VARCHAR(255)',
            nullable=True
        ):
            success_count += 1
        
        if add_column_if_not_exists(
            'user_bank_accounts',
            'razorpay_fund_account_id',
            'VARCHAR(255)',
            nullable=True
        ):
            success_count += 1
        
        if add_column_if_not_exists(
            'user_bank_accounts',
            'razorpay_contact_created_at',
            'DATETIME',
            nullable=True
        ):
            success_count += 1
        
        if add_column_if_not_exists(
            'user_bank_accounts',
            'razorpay_fund_account_created_at',
            'DATETIME',
            nullable=True
        ):
            success_count += 1
        
        # Summary
        print("\n" + "=" * 80)
        print("MIGRATION SUMMARY")
        print("=" * 80)
        print(f"Total columns to add: {total_count}")
        print(f"Successfully added: {success_count}")
        print(f"Already existed: {total_count - success_count}")
        
        if success_count == total_count:
            print("\n✅ All new columns added successfully!")
        elif success_count > 0:
            print(f"\n⚠️  {success_count} columns added, {total_count - success_count} already existed.")
        else:
            print("\n⚠️  No new columns were added (all may already exist).")
        
        print("=" * 80)
        
        return success_count > 0 or total_count == 0


def verify_migration():
    """Verify that all columns were added successfully"""
    app = create_app()
    
    with app.app_context():
        print("\n" + "=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        
        inspector = inspect(db.engine)
        
        # Verify payroll_settings columns
        print("\n📋 Verifying 'payroll_settings' table:")
        payroll_columns = [
            'razorpay_webhook_secret',
            'razorpay_fund_account_validated',
            'razorpay_fund_account_validated_at'
        ]
        
        if 'payroll_settings' in inspector.get_table_names():
            existing_columns = [col['name'] for col in inspector.get_columns('payroll_settings')]
            for col in payroll_columns:
                if col in existing_columns:
                    print(f"  ✓ {col}")
                else:
                    print(f"  ✗ {col} - MISSING!")
        else:
            print("  ⚠️  Table 'payroll_settings' does not exist")
        
        # Verify user_bank_accounts columns
        print("\n📋 Verifying 'user_bank_accounts' table:")
        bank_columns = [
            'razorpay_contact_id',
            'razorpay_fund_account_id',
            'razorpay_contact_created_at',
            'razorpay_fund_account_created_at'
        ]
        
        if 'user_bank_accounts' in inspector.get_table_names():
            existing_columns = [col['name'] for col in inspector.get_columns('user_bank_accounts')]
            for col in bank_columns:
                if col in existing_columns:
                    print(f"  ✓ {col}")
                else:
                    print(f"  ✗ {col} - MISSING!")
        else:
            print("  ⚠️  Table 'user_bank_accounts' does not exist")
        
        print("=" * 80)


if __name__ == '__main__':
    print("\n🚀 Starting Razorpay payment fields migration...\n")
    
    try:
        success = migrate_razorpay_fields()
        
        if success:
            verify_migration()
            print("\n✨ Migration completed successfully!")
            print("\n📝 Next steps:")
            print("   1. Verify the columns in your database")
            print("   2. Test the payment functionality")
            print("   3. Configure Razorpay settings in the payroll settings page")
            sys.exit(0)
        else:
            verify_migration()
            print("\n⚠️  Migration completed with warnings. Please review above.")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Migration failed: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        sys.exit(1)


