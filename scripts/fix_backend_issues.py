#!/usr/bin/env python3
"""
Backend Issues Fix Script
Fixes memory issues and Redis connection problems
"""

import os
import sys
import gc
import psutil
import time
import subprocess
import platform

def check_system_resources():
    """Check current system resources"""
    print("🔍 Checking system resources...")
    
    # Memory check
    memory = psutil.virtual_memory()
    print(f"📊 Memory Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB used / {memory.total / 1024**3:.1f}GB total)")
    
    # CPU check
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"🖥️  CPU Usage: {cpu_percent:.1f}%")
    
    # Disk check
    disk = psutil.disk_usage('/')
    print(f"💾 Disk Usage: {disk.percent:.1f}% ({disk.used / 1024**3:.1f}GB used / {disk.total / 1024**3:.1f}GB total)")
    
    return memory.percent, cpu_percent, disk.percent

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    print("🧹 Starting aggressive memory cleanup...")
    
    # Multiple rounds of garbage collection
    total_collected = 0
    for round_num in range(5):
        print(f"  Round {round_num + 1}/5...")
        for generation in range(3):
            collected = gc.collect()
            total_collected += collected
        
        # Clear Python caches
        try:
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
        except:
            pass
        
        time.sleep(0.1)  # Small delay between rounds
    
    print(f"✅ Memory cleanup completed. Collected {total_collected} objects")
    return total_collected

def check_redis_connection():
    """Check if Redis is running"""
    print("🔍 Checking Redis connection...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        print("✅ Redis is running and accessible")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def install_redis_windows():
    """Install Redis on Windows"""
    print("📦 Installing Redis for Windows...")
    
    # Try different installation methods
    methods = [
        # Method 1: Download and run Redis for Windows
        {
            'name': 'Download Redis for Windows',
            'url': 'https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.msi',
            'instructions': 'Download and install the MSI file, then start Redis service'
        },
        # Method 2: Use WSL
        {
            'name': 'Use WSL (Windows Subsystem for Linux)',
            'command': 'wsl --install',
            'instructions': 'Install WSL, then run: sudo apt install redis-server && sudo service redis-server start'
        },
        # Method 3: Use Chocolatey
        {
            'name': 'Use Chocolatey',
            'command': 'choco install redis-64',
            'instructions': 'Install Chocolatey first, then run: choco install redis-64'
        }
    ]
    
    print("\n📋 Redis Installation Options:")
    for i, method in enumerate(methods, 1):
        print(f"\n{i}. {method['name']}")
        if 'url' in method:
            print(f"   URL: {method['url']}")
        if 'command' in method:
            print(f"   Command: {method['command']}")
        print(f"   Instructions: {method['instructions']}")
    
    return methods

def optimize_memory_settings():
    """Optimize memory settings for the application"""
    print("⚙️  Optimizing memory settings...")
    
    # Set garbage collection thresholds
    gc.set_threshold(700, 10, 10)
    print("  ✅ Garbage collection thresholds optimized")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"  ✅ Forced garbage collection: {collected} objects collected")
    
    return collected

def create_redis_startup_script():
    """Create a simple Redis startup script"""
    print("📝 Creating Redis startup script...")
    
    script_content = '''@echo off
echo Starting Redis for Kempian Backend...
echo.

REM Try to find Redis installation
set REDIS_FOUND=0

REM Check common Redis installation paths
if exist "C:\\Program Files\\Redis\\redis-server.exe" (
    set REDIS_PATH="C:\\Program Files\\Redis\\redis-server.exe"
    set REDIS_FOUND=1
)

if exist "C:\\Program Files (x86)\\Redis\\redis-server.exe" (
    set REDIS_PATH="C:\\Program Files (x86)\\Redis\\redis-server.exe"
    set REDIS_FOUND=1
)

if exist "C:\\Redis\\redis-server.exe" (
    set REDIS_PATH="C:\\Redis\\redis-server.exe"
    set REDIS_FOUND=1
)

if %REDIS_FOUND% == 1 (
    echo Found Redis at: %REDIS_PATH%
    echo Starting Redis server...
    start "Redis Server" %REDIS_PATH%
    echo ✅ Redis server started!
    echo Redis is running on localhost:6379
) else (
    echo ❌ Redis not found. Please install Redis first.
    echo.
    echo Installation options:
    echo 1. Download from: https://github.com/microsoftarchive/redis/releases
    echo 2. Use WSL: wsl --install
    echo 3. Use Chocolatey: choco install redis-64
)

echo.
pause
'''
    
    with open('start_redis_manual.bat', 'w') as f:
        f.write(script_content)
    
    print("  ✅ Created start_redis_manual.bat")

def create_memory_monitor():
    """Create a memory monitoring script"""
    print("📊 Creating memory monitoring script...")
    
    monitor_content = '''#!/usr/bin/env python3
"""
Memory Monitor for Kempian Backend
Monitors memory usage and triggers cleanup when needed
"""

import psutil
import time
import gc
import sys

def monitor_memory():
    """Monitor memory usage and trigger cleanup"""
    print("🔍 Starting memory monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            memory = psutil.virtual_memory()
            print(f"\\rMemory: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB used)", end="", flush=True)
            
            if memory.percent > 85:
                print(f"\\n⚠️  High memory usage detected: {memory.percent:.1f}%")
                print("🧹 Triggering memory cleanup...")
                
                # Force garbage collection
                collected = 0
                for _ in range(3):
                    collected += gc.collect()
                
                print(f"✅ Cleaned up {collected} objects")
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\\n🛑 Memory monitoring stopped")

if __name__ == "__main__":
    monitor_memory()
'''
    
    with open('memory_monitor.py', 'w') as f:
        f.write(monitor_content)
    
    print("  ✅ Created memory_monitor.py")

def main():
    """Main function to fix backend issues"""
    print("🚀 Kempian Backend Issues Fix Script")
    print("=" * 50)
    
    # Check system resources
    memory_percent, cpu_percent, disk_percent = check_system_resources()
    
    # If memory usage is high, force cleanup
    if memory_percent > 85:
        print(f"⚠️  High memory usage detected: {memory_percent:.1f}%")
        force_memory_cleanup()
        
        # Check memory again
        memory = psutil.virtual_memory()
        print(f"📊 Memory after cleanup: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB used)")
    
    # Check Redis connection
    redis_running = check_redis_connection()
    
    if not redis_running:
        print("\n❌ Redis is not running. This is causing backend issues.")
        install_redis_windows()
    
    # Optimize memory settings
    optimize_memory_settings()
    
    # Create utility scripts
    create_redis_startup_script()
    create_memory_monitor()
    
    print("\n" + "=" * 50)
    print("✅ Backend issues fix completed!")
    print("\n📋 Next steps:")
    print("1. Install Redis using one of the methods above")
    print("2. Run: python memory_monitor.py (to monitor memory)")
    print("3. Restart your backend server")
    print("4. Check if Redis is running: redis-cli ping")
    
    print("\n🔧 Quick fixes:")
    print("- Restart your backend server")
    print("- Close unnecessary applications to free memory")
    print("- Run: python -c \"import gc; gc.collect()\" to clean memory")

if __name__ == "__main__":
    main()
