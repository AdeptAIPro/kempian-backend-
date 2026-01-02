"""
Fix missing bank columns in user_bank_accounts table
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

def add_missing_columns():
    """Add missing columns to user_bank_accounts table"""
    config = get_db_config()

    try:
        print("🔌 Connecting to database...")
        connection = pymysql.connect(**config)

        with connection.cursor() as cursor:
            # Check and add bank_branch column
            print("🔄 Checking bank_branch column...")
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'user_bank_accounts'
                AND COLUMN_NAME = 'bank_branch'
                AND TABLE_SCHEMA = DATABASE()
            """)

            if not cursor.fetchone():
                print("📝 Adding bank_branch column...")
                cursor.execute("""
                    ALTER TABLE user_bank_accounts
                    ADD COLUMN bank_branch VARCHAR(255) NULL
                    COMMENT 'Bank branch name'
                """)
                print("✅ Added bank_branch column")
            else:
                print("⚠️ bank_branch column already exists")

            # Check and add bank_address column (in case it's also missing)
            print("🔄 Checking bank_address column...")
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'user_bank_accounts'
                AND COLUMN_NAME = 'bank_address'
                AND TABLE_SCHEMA = DATABASE()
            """)

            if not cursor.fetchone():
                print("📝 Adding bank_address column...")
                cursor.execute("""
                    ALTER TABLE user_bank_accounts
                    ADD COLUMN bank_address TEXT NULL
                    COMMENT 'Bank branch address'
                """)
                print("✅ Added bank_address column")
            else:
                print("⚠️ bank_address column already exists")

            connection.commit()
            print("🎉 All missing columns added successfully!")
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
    print("🔧 Fixing missing bank columns in user_bank_accounts table...")
    print()

    success = add_missing_columns()

    if success:
        print()
        print("✅ Database schema updated successfully!")
        print("💡 The missing bank_branch and bank_address columns have been added.")
        print("📝 The auth routes should now work without 'Unknown column' errors.")
    else:
        print()
        print("❌ Schema update failed. Please check:")
        print("   - Database connection settings")
        print("   - User permissions")
        print("   - Database server is running")
        print()
        print("🔧 Alternative: Run the SQL files manually:")
        print("   mysql -u [username] -p [database] < backend/migrations/add_bank_branch_column.sql")
