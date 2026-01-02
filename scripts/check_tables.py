#!/usr/bin/env python3
"""
Script to verify all tables defined in models.py have been created in the database.

Usage:
    python check_tables.py
"""

import sys
import os
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import create_app, db
    import app.models as models_module
except ImportError as e:
    print(f"Error importing app: {e}")
    print("Make sure you're running this script from the backend directory")
    sys.exit(1)


def get_all_models():
    """Get all SQLAlchemy models from the models module."""
    models = []
    
    # Get the db.Model class for comparison
    from app.models import db as models_db
    
    # Iterate through all attributes in the models module
    for name in dir(models_module):
        # Skip private attributes and imports
        if name.startswith('_'):
            continue
            
        try:
            obj = getattr(models_module, name)
            
            # Check if it's a class that inherits from db.Model
            if (isinstance(obj, type) and 
                issubclass(obj, models_db.Model) and 
                obj != models_db.Model and
                hasattr(obj, '__tablename__')):
                models.append(obj)
        except (TypeError, AttributeError):
            # Skip if object is not a class or doesn't have __bases__
            continue
    
    return models


def check_tables():
    """Check if all tables from models exist in the database."""
    app = create_app()
    
    with app.app_context():
        try:
            # Get database inspector
            inspector = inspect(db.engine)
            existing_tables = set(inspector.get_table_names())
            
            # Get all models
            models = get_all_models()
            
            # Get expected table names from models
            expected_tables = {}
            for model in models:
                if hasattr(model, '__tablename__'):
                    table_name = model.__tablename__
                    model_name = model.__name__
                    expected_tables[table_name] = model_name
            
            # Check which tables exist and which don't
            missing_tables = []
            existing_model_tables = []
            extra_tables = []
            
            # Check expected tables
            for table_name, model_name in expected_tables.items():
                if table_name in existing_tables:
                    existing_model_tables.append((table_name, model_name))
                else:
                    missing_tables.append((table_name, model_name))
            
            # Find tables in database that aren't in models
            model_table_names = set(expected_tables.keys())
            for table_name in existing_tables:
                if table_name not in model_table_names:
                    extra_tables.append(table_name)
            
            # Print results
            print("=" * 80)
            print("DATABASE TABLE VERIFICATION REPORT")
            print("=" * 80)
            print(f"\nTotal models found: {len(expected_tables)}")
            print(f"Total tables in database: {len(existing_tables)}")
            print(f"Expected tables found: {len(existing_model_tables)}")
            print(f"Missing tables: {len(missing_tables)}")
            print(f"Extra tables (not in models): {len(extra_tables)}")
            print("=" * 80)
            
            # Print existing tables
            if existing_model_tables:
                print("\n✓ EXISTING TABLES ({}):".format(len(existing_model_tables)))
                print("-" * 80)
                for table_name, model_name in sorted(existing_model_tables):
                    print(f"  ✓ {table_name:50s} ({model_name})")
            
            # Print missing tables
            if missing_tables:
                print("\n✗ MISSING TABLES ({}):".format(len(missing_tables)))
                print("-" * 80)
                for table_name, model_name in sorted(missing_tables):
                    print(f"  ✗ {table_name:50s} ({model_name})")
            
            # Print extra tables
            if extra_tables:
                print("\n⚠ EXTRA TABLES (not in models) ({}):".format(len(extra_tables)))
                print("-" * 80)
                for table_name in sorted(extra_tables):
                    print(f"  ⚠ {table_name}")
            
            # Summary
            print("\n" + "=" * 80)
            if missing_tables:
                print(f"❌ FAILED: {len(missing_tables)} table(s) are missing from the database")
                print("\nTo create missing tables, run:")
                print("  python -c 'from app import create_app, db; from app.models import *; app = create_app(); app.app_context().push(); db.create_all()'")
                return 1
            else:
                print("✅ SUCCESS: All expected tables exist in the database!")
                if extra_tables:
                    print(f"   Note: {len(extra_tables)} extra table(s) found in database (not in models)")
                return 0
                
        except OperationalError as e:
            print(f"❌ ERROR: Could not connect to database: {e}")
            print("\nPlease check:")
            print("  1. Database server is running")
            print("  2. Database credentials in .env file are correct")
            print("  3. Database exists")
            return 1
        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    exit_code = check_tables()
    sys.exit(exit_code)

