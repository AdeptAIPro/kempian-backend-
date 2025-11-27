@echo off
echo ========================================
echo Starting Local Redis with Docker
echo ========================================
echo.

REM Check if Docker is installed and running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not running!
    echo.
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    echo.
    pause
    exit /b 1
)

echo [OK] Docker is running
echo.

REM Check if Redis container already exists
docker ps -a | findstr redis-local >nul
if not errorlevel 1 (
    echo Redis container already exists. Starting it...
    docker start redis-local
) else (
    echo Creating and starting Redis container...
    docker run -d -p 6379:6379 --name redis-local redis:latest
)

echo.
echo [SUCCESS] Redis is running on localhost:6379
echo.
echo You can test it with:
echo   redis-cli ping
echo   (or install redis-cli for Windows)
echo.
echo Stopping this window will NOT stop Redis.
echo To stop Redis, run: docker stop redis-local
echo.

pause

