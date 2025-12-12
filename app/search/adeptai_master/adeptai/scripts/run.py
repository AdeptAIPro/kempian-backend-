#!/usr/bin/env python3
"""
AdeptAI Run Script
=================

Simple run script for the AdeptAI recruitment system.
This is the recommended way to start the application.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from start import main
    main()
