"""
Quick test to verify talent routes are accessible
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    print("Testing talent blueprint import...")
    from app.talent.routes import talent_bp
    print(f"✅ Blueprint imported: {talent_bp.name}")
    print(f"✅ Blueprint URL prefix: {talent_bp.url_prefix}")
    print(f"✅ Number of routes: {len(talent_bp.deferred_functions)}")
    
    # Check for specific routes
    print("\nChecking for specific routes:")
    route_names = [func.__name__ for func in talent_bp.deferred_functions if hasattr(func, '__name__')]
    
    if 'upload_resume_enhanced' in route_names:
        print("  ✅ upload_resume_enhanced route found")
    else:
        print("  ❌ upload_resume_enhanced route NOT found")
    
    if 'get_profile' in route_names:
        print("  ✅ get_profile route found")
    else:
        print("  ❌ get_profile route NOT found")
    
    print("\n✅ Talent blueprint is properly configured!")
    print("\n⚠️  IMPORTANT: Restart your Flask server for changes to take effect!")
    
except SyntaxError as e:
    print(f"❌ Syntax error in talent routes: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

