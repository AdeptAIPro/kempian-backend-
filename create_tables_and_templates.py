"""
Combined Script: Create Communication Tables and Default Templates
This script creates the database tables first, then creates default templates.

Usage:
    python backend/create_tables_and_templates.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def main():
    """Run both migration and template creation"""
    print("=" * 60)
    print("Communication Tables & Templates Setup")
    print("=" * 60)
    print()
    
    # Step 1: Create tables
    print("Step 1: Creating database tables...")
    print("-" * 60)
    try:
        from migrations.create_communication_tables import create_communication_tables
        create_communication_tables()
        print()
    except Exception as e:
        print(f"[ERROR] Failed to create tables: {str(e)}")
        print("\n[WARNING] Please create tables manually using SQL script:")
        print("   backend/migrations/create_communication_tables.sql")
        return
    
    # Step 2: Create templates
    print("Step 2: Creating default templates...")
    print("-" * 60)
    try:
        from create_default_templates import create_default_templates
        create_default_templates()
        print()
    except Exception as e:
        print(f"[ERROR] Failed to create templates: {str(e)}")
        print("\n[WARNING] Tables were created, but template creation failed.")
        print("   You can run template creation separately:")
        print("   python backend/create_default_templates.py")
        return
    
    print("=" * 60)
    print("[SUCCESS] Setup Complete!")
    print("=" * 60)
    print("\nSummary:")
    print("   [OK] Database tables created")
    print("   [OK] Default templates created (3 per channel)")
    print("\nYou can now use the communication features!")

if __name__ == '__main__':
    main()

