"""
Migration Script: Add ifsc_code column to user_bank_accounts table
Run this script to add the missing ifsc_code column that was added to the model but not migrated.

Usage:
    python backend/migrations/add_ifsc_code_column.py
    OR
    cd backend && python migrations/add_ifsc_code_column.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from sqlalchemy import text

def add_ifsc_code_column():
    """Add ifsc_code column to user_bank_accounts table"""
    app = create_app()

    with app.app_context():
        try:
            print("🔄 Checking if ifsc_code column exists...")

            # Check if column already exists
            result = db.session.execute(text("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'user_bank_accounts'
                AND COLUMN_NAME = 'ifsc_code'
                AND TABLE_SCHEMA = DATABASE()
            """)).fetchone()

            if result:
                print("✅ ifsc_code column already exists in user_bank_accounts table")
                return True

            print("📝 Adding ifsc_code column to user_bank_accounts table...")

            # Add the column
            db.session.execute(text("""
                ALTER TABLE user_bank_accounts
                ADD COLUMN ifsc_code VARCHAR(11) NULL
                COMMENT 'Indian Financial System Code'
            """))

            db.session.commit()

            print("✅ Successfully added ifsc_code column to user_bank_accounts table")
            print("📋 Column details:")
            print("   - Name: ifsc_code")
            print("   - Type: VARCHAR(11)")
            print("   - Nullable: Yes")
            print("   - Purpose: Indian Financial System Code")

            return True

        except Exception as e:
            db.session.rollback()
            print(f"❌ Error adding ifsc_code column: {str(e)}")
            return False

if __name__ == "__main__":
    print("🚀 Adding ifsc_code column to user_bank_accounts table...")
    print()

    success = add_ifsc_code_column()

    if success:
        print()
        print("🎉 Migration completed successfully!")
        print("💡 The ifsc_code column has been added to the user_bank_accounts table.")
        print("📝 This should resolve the 'Unknown column' error in the auth routes.")
    else:
        print()
        print("❌ Migration failed. Please check the error messages above.")
        sys.exit(1)
