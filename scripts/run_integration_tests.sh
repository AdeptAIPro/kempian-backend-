#!/bin/bash
# Integration test runner for Jobvite integration
# Requires: JOBVITE_STAGE_API_KEY, JOBVITE_STAGE_API_SECRET, JOBVITE_STAGE_COMPANY_ID

echo "=========================================="
echo "Jobvite Integration Tests"
echo "=========================================="
echo ""

# Check environment variables
if [ -z "$JOBVITE_STAGE_API_KEY" ] || [ -z "$JOBVITE_STAGE_API_SECRET" ] || [ -z "$JOBVITE_STAGE_COMPANY_ID" ]; then
    echo "❌ Missing required environment variables:"
    echo "   - JOBVITE_STAGE_API_KEY"
    echo "   - JOBVITE_STAGE_API_SECRET"
    echo "   - JOBVITE_STAGE_COMPANY_ID"
    echo ""
    echo "Set these variables and run again."
    exit 1
fi

echo "✅ Environment variables set"
echo ""

# Run integration tests
cd backend
python -m pytest tests/integration/test_jobvite_integration.py -v

echo ""
echo "=========================================="
echo "Integration tests complete"
echo "=========================================="

