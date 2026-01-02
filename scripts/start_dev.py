#!/usr/bin/env python3
"""
Development startup script for Kempian Backend
Handles missing services gracefully and provides helpful error messages
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def check_services():
    """Check if required services are available"""
    print("🔍 Checking required services...")
    
    services_status = {
        'mysql': False,
        'redis': False,
        'sqlite': True  # SQLite is always available
    }
    
    # Check MySQL
    try:
        import pymysql
        connection = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='',
            connect_timeout=5
        )
        connection.close()
        services_status['mysql'] = True
        print("✅ MySQL is running")
    except Exception as e:
        print(f"❌ MySQL not available: {e}")
        print("   Using SQLite fallback")
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        services_status['redis'] = True
        print("✅ Redis is running")
    except Exception as e:
        print(f"❌ Redis not available: {e}")
        print("   Using in-memory cache fallback")
    
    return services_status

def start_backend():
    """Start the backend server"""
    print("\n🚀 Starting Kempian Backend...")
    
    try:
        from app import create_app
        app = create_app()
        
        print("✅ Backend application created successfully")
        print("🌐 Starting server on http://localhost:5000")
        print("📊 Admin dashboard: http://localhost:5000/admin")
        print("🔍 Health check: http://localhost:5000/health")
        print("\nPress Ctrl+C to stop the server")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure you're in the backend directory")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check if port 5000 is available")
        print("4. For MySQL: Start MySQL service or use SQLite")
        print("5. For Redis: Start Redis service or use in-memory cache")
        return False
    
    return True

def main():
    """Main function"""
    print("=" * 60)
    print("🎯 KEMPIAN BACKEND - DEVELOPMENT MODE")
    print("=" * 60)
    
    # Check services
    services = check_services()
    
    # Show configuration
    print(f"\n📋 Configuration:")
    print(f"   Database: {'MySQL' if services['mysql'] else 'SQLite (fallback)'}")
    print(f"   Cache: {'Redis' if services['redis'] else 'In-memory (fallback)'}")
    print(f"   Mode: Development")
    
    # Start backend
    if not start_backend():
        sys.exit(1)

if __name__ == "__main__":
    main()
