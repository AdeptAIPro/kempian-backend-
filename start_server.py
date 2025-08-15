#!/usr/bin/env python3
"""
Quick Start Server Script
Starts the backend server with proper configuration
"""

import os
import sys
from app import create_app

def main():
    """Start the Flask server"""
    print("🚀 Starting AdeptAI-Master Backend Server")
    print("=" * 50)
    
    # Create Flask app
    app = create_app()
    
    # Configure for production
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"🌐 Server will start on: http://{host}:{port}")
    print("📊 API Endpoints available:")
    print("   - GET  /health")
    print("   - POST /api/search")
    print("   - POST /api/semantic-match")
    print("=" * 50)
    
    try:
        # Start the server
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 