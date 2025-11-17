#!/bin/bash
# Script to run database migration for communication tables
# Usage: bash backend/run_migration.sh

echo "🚀 Starting Communication Tables Migration..."
echo ""

# Check if we're in the right directory
if [ ! -f "backend/app/__init__.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Run Python migration script
echo "📊 Creating communication tables..."
python backend/migrations/create_communication_tables.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Migration completed successfully!"
    echo ""
    echo "📝 Next step: Run the template creation script:"
    echo "   python backend/create_default_templates.py"
else
    echo ""
    echo "❌ Migration failed. Please check the error messages above."
    exit 1
fi

