@echo off
echo ========================================
echo Stopping Local Redis
echo ========================================
echo.

docker stop redis-local

if errorlevel 1 (
    echo [ERROR] Failed to stop Redis container
    pause
    exit /b 1
)

echo [SUCCESS] Redis has been stopped
echo.

pause

