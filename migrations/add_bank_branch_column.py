"""
Simple Migration Script: Add bank_branch column to user_bank_accounts table
Run this script to add the missing bank_branch column that was added to the model but not migrated.

Usage:
    python backend/migrations/add_bank_branch_column.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from sqlalchemy import text

def add_bank_branch_column():
    """Add bank_branch column to user_bank_accounts table"""
    app = create_app()

    with app.app_context():
        try:
            print("🔄 Checking if bank_branch column exists...")

            # Check if column already exists
            result = db.session.execute(text("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'user_bank_accounts'
                AND COLUMN_NAME = 'bank_branch'
                AND TABLE_SCHEMA = DATABASE()
            """)).fetchone()

            if result:
                print("✅ bank_branch column already exists!")
                return True

            print("📝 Adding bank_branch column...")

            # Add the column
            db.session.execute(text("""
                ALTER TABLE user_bank_accounts
                ADD COLUMN bank_branch VARCHAR(255) NULL
                COMMENT 'Bank branch name'
            """))

            db.session.commit()

            print("✅ Successfully added bank_branch column!")
            print("📋 Column details:")
            print("   - Name: bank_branch")
            print("   - Type: VARCHAR(255)")
            print("   - Nullable: Yes")
            print("   - Purpose: Bank branch name")

            return True

        except Exception as e:
            db.session.rollback()
            print(f"❌ Error adding bank_branch column: {str(e)}")
            return False

if __name__ == "__main__":
    print("🚀 Adding bank_branch column to user_bank_accounts table...")
    print()

    success = add_bank_branch_column()

    if success:
        print()
        print("🎉 Migration completed successfully!")
        print("💡 The bank_branch column has been added to the user_bank_accounts table.")
        print("📝 This should resolve the 'Unknown column' error in the auth routes.")
    else:
        print()
        print("❌ Migration failed. Please check the error messages above.")
