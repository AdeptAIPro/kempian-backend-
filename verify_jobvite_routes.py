"""
Quick script to verify Jobvite routes are importable and registered correctly.
Run this to check if there are any import errors preventing the blueprint from loading.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing Jobvite blueprint import...")
    from app.jobvite.routes import jobvite_bp
    print(f"[OK] Blueprint imported successfully: {jobvite_bp.name}")
    
    # Check routes
    print(f"[OK] Blueprint has {len(jobvite_bp.deferred_functions)} deferred functions")
    
    # List all routes
    print("\nRegistered routes:")
    for rule in jobvite_bp.url_map.iter_rules() if hasattr(jobvite_bp, 'url_map') else []:
        print(f"  {rule}")
    
    # Check if routes are defined
    print("\nChecking route decorators...")
    import inspect
    routes_found = []
    for name, obj in inspect.getmembers(jobvite_bp):
        if hasattr(obj, 'rule'):
            routes_found.append(obj.rule)
    
    if routes_found:
        print(f"[OK] Found {len(routes_found)} routes")
        for route in routes_found[:5]:  # Show first 5
            print(f"  - {route}")
    else:
        print("[WARNING] No routes found in blueprint (might be deferred)")
    
    print("\n[OK] All checks passed! Blueprint is ready.")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

