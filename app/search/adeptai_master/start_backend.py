#!/usr/bin/env python3
"""
Simple backend starter script
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adeptai.app import create_app

if __name__ == '__main__':
    print("🚀 Starting AdeptAI Backend...")
    
    # Create app
    app = create_app()
    
    # Run
    print("🌐 Server running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
