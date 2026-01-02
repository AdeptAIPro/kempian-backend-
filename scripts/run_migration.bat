@echo off
REM Script to run database migration for communication tables (Windows)
REM Usage: backend\run_migration.bat

echo 🚀 Starting Communication Tables Migration...
echo.

REM Check if we're in the right directory
if not exist "backend\app\__init__.py" (
    echo ❌ Error: Please run this script from the project root directory
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Run Python migration script
echo 📊 Creating communication tables...
python backend\migrations\create_communication_tables.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ Migration completed successfully!
    echo.
    echo 📝 Next step: Run the template creation script:
    echo    python backend\create_default_templates.py
) else (
    echo.
    echo ❌ Migration failed. Please check the error messages above.
    exit /b 1
)

