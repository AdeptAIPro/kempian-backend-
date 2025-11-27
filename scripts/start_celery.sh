#!/bin/bash
# Start Celery workers and beat for Jobvite integration

echo "=========================================="
echo "Starting Celery for Jobvite Integration"
echo "=========================================="
echo ""

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running!"
    echo "   Start Redis first: redis-server"
    exit 1
fi

echo "✅ Redis is running"
echo ""

# Check environment variables
if [ -z "$CELERY_BROKER_URL" ]; then
    echo "⚠️  CELERY_BROKER_URL not set, using default: redis://localhost:6379/0"
    export CELERY_BROKER_URL="redis://localhost:6379/0"
fi

if [ -z "$CELERY_RESULT_BACKEND" ]; then
    echo "⚠️  CELERY_RESULT_BACKEND not set, using default: redis://localhost:6379/0"
    export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
fi

cd backend

echo "Starting Celery worker..."
celery -A app.jobvite.celery_config worker --loglevel=info --concurrency=4 &

WORKER_PID=$!
echo "Worker PID: $WORKER_PID"

echo ""
echo "Starting Celery beat (scheduled tasks)..."
celery -A app.jobvite.celery_config beat --loglevel=info &

BEAT_PID=$!
echo "Beat PID: $BEAT_PID"

echo ""
echo "✅ Celery started"
echo ""
echo "To stop:"
echo "  kill $WORKER_PID"
echo "  kill $BEAT_PID"
echo ""
echo "Or use: pkill -f 'celery.*jobvite'"

