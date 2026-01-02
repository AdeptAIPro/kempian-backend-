"""
Script to check if talent blueprint routes are registered
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app

def check_talent_routes():
    """Check if talent routes are registered"""
    app = create_app()
    
    print("=" * 80)
    print("CHECKING TALENT BLUEPRINT ROUTES")
    print("=" * 80)
    print()
    
    # Check if talent blueprint is registered
    talent_routes = []
    all_routes = []
    
    for rule in app.url_map.iter_rules():
        all_routes.append(rule.rule)
        if '/talent' in rule.rule or 'talent' in rule.endpoint:
            talent_routes.append({
                'rule': rule.rule,
                'endpoint': rule.endpoint,
                'methods': list(rule.methods)
            })
    
    print(f"Total routes registered: {len(all_routes)}")
    print(f"Talent-related routes found: {len(talent_routes)}")
    print()
    
    if talent_routes:
        print("Talent routes:")
        for route in talent_routes:
            print(f"  {route['rule']} -> {route['endpoint']} [{', '.join(route['methods'])}]")
    else:
        print("⚠️  No talent routes found!")
        print()
        print("Possible issues:")
        print("  1. ENABLE_SERVICE_TALENT might be disabled")
        print("  2. Blueprint import/registration failed")
        print("  3. Check backend logs for import errors")
    
    print()
    print("Checking for upload-resume-enhanced route specifically:")
    upload_routes = [r for r in talent_routes if 'upload-resume' in r['rule']]
    if upload_routes:
        print("  ✅ Found upload-resume routes:")
        for route in upload_routes:
            print(f"     {route['rule']}")
    else:
        print("  ❌ upload-resume-enhanced route NOT found!")
        print()
        print("  Checking if blueprint is imported:")
        try:
            from app.talent.routes import talent_bp
            print(f"  ✅ Blueprint imported: {talent_bp.name}")
            print(f"  ✅ Blueprint URL prefix: {talent_bp.url_prefix}")
            
            # Check if route is defined in blueprint
            routes_in_blueprint = [rule for rule in talent_bp.deferred_functions if hasattr(rule, '__name__')]
            print(f"  Routes in blueprint: {len(routes_in_blueprint)}")
            
            # Try to get routes from blueprint
            with app.app_context():
                blueprint_routes = []
                for rule in app.url_map.iter_rules():
                    if rule.endpoint.startswith('talent.'):
                        blueprint_routes.append(rule.rule)
                
                if blueprint_routes:
                    print(f"  ✅ Found {len(blueprint_routes)} routes registered from talent blueprint")
                else:
                    print("  ❌ No routes registered from talent blueprint")
                    print("  This means the blueprint was not registered with the app")
                    
        except Exception as e:
            print(f"  ❌ Error importing blueprint: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 80)

if __name__ == '__main__':
    check_talent_routes()

