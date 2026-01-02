@echo off
echo Starting Redis for Kempian Backend...
echo.

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% == 0 (
    echo Using Docker to start Redis...
    docker run -d -p 6379:6379 --name kempian-redis redis:latest
    if %errorlevel% == 0 (
        echo ✅ Redis started successfully with Docker!
        echo Redis is running on localhost:6379
        echo.
        echo To stop Redis: docker stop kempian-redis
        echo To start Redis again: docker start kempian-redis
        echo To remove Redis: docker rm kempian-redis
    ) else (
        echo ❌ Failed to start Redis with Docker
        echo Please install Docker Desktop or use alternative method
    )
) else (
    echo Docker not found. Please install Redis manually:
    echo.
    echo Option 1: Install Docker Desktop
    echo Option 2: Use Chocolatey: choco install redis-64
    echo Option 3: Download from: https://github.com/microsoftarchive/redis/releases
    echo Option 4: Use WSL: wsl --install
    echo.
    echo See REDIS_SETUP.md for detailed instructions
)

echo.
pause
