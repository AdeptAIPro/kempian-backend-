"""
Simple Migration Script: Add ifsc_code column to user_bank_accounts table
This script runs directly against the database without loading the full Flask app.

Usage:
    python backend/migrations/run_ifsc_migration.py
"""
import os
import pymysql
from pymysql.cursors import DictCursor

def get_db_config():
    """Get database configuration from environment variables"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'kempian'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'charset': 'utf8mb4',
        'cursorclass': DictCursor
    }

def run_migration():
    """Run the migration to add ifsc_code column"""
    config = get_db_config()

    try:
        print("🔌 Connecting to database...")
        connection = pymysql.connect(**config)

        with connection.cursor() as cursor:
            print("🔄 Checking if ifsc_code column exists...")

            # Check if column exists
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'user_bank_accounts'
                AND COLUMN_NAME = 'ifsc_code'
                AND TABLE_SCHEMA = DATABASE()
            """)

            result = cursor.fetchone()

            if result:
                print("✅ ifsc_code column already exists!")
                return True

            print("📝 Adding ifsc_code column...")

            # Add the column
            cursor.execute("""
                ALTER TABLE user_bank_accounts
                ADD COLUMN ifsc_code VARCHAR(11) NULL
                COMMENT 'Indian Financial System Code'
            """)

            connection.commit()

            print("✅ Successfully added ifsc_code column!")
            print("📋 Column details:")
            print("   - Name: ifsc_code")
            print("   - Type: VARCHAR(11)")
            print("   - Nullable: Yes")
            print("   - Purpose: Indian Financial System Code")

            return True

    except pymysql.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    print("🚀 Running ifsc_code column migration...")
    print()

    success = run_migration()

    if success:
        print()
        print("🎉 Migration completed successfully!")
        print("💡 The ifsc_code column has been added to the user_bank_accounts table.")
        print("📝 This should resolve the 'Unknown column' error in the auth routes.")
    else:
        print()
        print("❌ Migration failed. Please check:")
        print("   - Database connection settings")
        print("   - User permissions")
        print("   - Database server is running")
        print()
        print("💡 Alternative: Run the SQL file directly:")
        print("   mysql -u [username] -p [database] < backend/migrations/add_ifsc_code_column.sql")
