"""
Quick script to create user_activity_logs table
Can be run from project root or backend directory

Usage:
    python backend/create_user_activity_logs_table.py
    OR
    cd backend && python create_user_activity_logs_table.py
"""
import sys
import os

# Add backend directory to path
if os.path.basename(os.getcwd()) == 'backend':
    # Already in backend directory
    sys.path.insert(0, os.getcwd())
else:
    # In project root, add backend to path
    backend_dir = os.path.join(os.getcwd(), 'backend')
    if os.path.exists(backend_dir):
        sys.path.insert(0, backend_dir)
    else:
        # Try to find backend directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)

# Import and run the migration
from migrations.create_user_activity_logs_table import create_user_activity_logs_table

if __name__ == '__main__':
    success = create_user_activity_logs_table()
    sys.exit(0 if success else 1)

